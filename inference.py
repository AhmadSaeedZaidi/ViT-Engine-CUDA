import argparse
import math
import urllib.request
import json
import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision import transforms
from PIL import Image
import timm

# Import your compiled C++ extension
import vit_cuda

class ViTBlockCUDA(nn.Module):
    def __init__(self, state, idx):
        super().__init__()
        prefix = f'blocks.{idx}.'
        
        self.register_buffer('norm1_gamma', state[prefix + 'norm1.weight'])
        self.register_buffer('norm1_beta', state[prefix + 'norm1.bias'])
        
        self.register_buffer('qkv_weight', state[prefix + 'attn.qkv.weight'])
        self.register_buffer('qkv_bias', state[prefix + 'attn.qkv.bias'])
        self.register_buffer('proj_weight', state[prefix + 'attn.proj.weight'])
        self.register_buffer('proj_bias', state[prefix + 'attn.proj.bias'])
        
        self.register_buffer('norm2_gamma', state[prefix + 'norm2.weight'])
        self.register_buffer('norm2_beta', state[prefix + 'norm2.bias'])
        
        self.register_buffer('fc1_weight', state[prefix + 'mlp.fc1.weight'])
        self.register_buffer('fc1_bias', state[prefix + 'mlp.fc1.bias'])
        self.register_buffer('fc2_weight', state[prefix + 'mlp.fc2.weight'])
        self.register_buffer('fc2_bias', state[prefix + 'mlp.fc2.bias'])

    def forward(self, x, scale, eps):
        residual = x
        x = vit_cuda.layernorm_forward(x, self.norm1_gamma, self.norm1_beta, eps)

        q, k, v = vit_cuda.qkv_proj(x, self.qkv_weight, self.qkv_bias)

        attn_out = vit_cuda.flash_attn_2(q, k, v, scale)
        attn_out = vit_cuda.gemm_bias(attn_out, self.proj_weight, self.proj_bias)
        
        x = residual + attn_out
        residual = x
        
        x = vit_cuda.layernorm_forward(x, self.norm2_gamma, self.norm2_beta, eps)

        mlp_out_list = vit_cuda.mlp_forward(
            x, 
            self.fc1_weight, self.fc1_bias, 
            self.fc2_weight, self.fc2_bias
        )
        mlp_out = mlp_out_list[0]
        
        return residual + mlp_out

class ViTCUDA(nn.Module):
    def __init__(self, num_classes=1000, state_dict=None, pretrained=True):
        super().__init__()
        
        # Load pre-trained weights from standard timm model (or use caller-supplied state)
        if state_dict is None:
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            state = model.state_dict()
        else:
            state = state_dict
        
        self.num_classes = num_classes
        self.scale = 1.0 / math.sqrt(64)
        self.eps = 1e-6
        
        # patch projection conv weight -> reshape to [embed_dim, patch_volume] at forward time
        self.register_buffer('patch_weight', state['patch_embed.proj.weight'])
        self.register_buffer('patch_bias', state['patch_embed.proj.bias'])
        # cls_token is [1,1,E] in timm; keep as [1, E] and expand per-batch in forward
        self.register_buffer('cls_token', state['cls_token'].squeeze(0))
        # pos_embed: [1, seq_len+1, E] -> [seq_len+1, E]
        self.register_buffer('pos_embed', state['pos_embed'].squeeze(0))
        
        self.blocks = nn.ModuleList([ViTBlockCUDA(state, i) for i in range(12)])
        
        self.register_buffer('norm_gamma', state['norm.weight'])
        self.register_buffer('norm_beta', state['norm.bias'])
        
        self.register_buffer('head_weight', state['head.weight'])
        self.register_buffer('head_bias', state['head.bias'])

    def forward(self, x):
        # CUDA graph replay path (fixed input shape/address)
        if getattr(self, '_graph', None) is not None:
            self._static_input.copy_(x)
            self._graph.replay()
            return self._static_output
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # reshape patch conv weights -> [embed_dim, patch_volume]
        pw = self.patch_weight
        if pw.dim() == 4:
            pw2 = pw.reshape(pw.size(0), -1)
        else:
            pw2 = pw

        x = vit_cuda.patch_embed(x, pw2, self.patch_bias)

        # expand cls_token to batch size expected by pos_encoding wrapper
        B = x.size(0)
        cls = self.cls_token
        if cls.dim() == 1:
            cls = cls.unsqueeze(0)
        cls_b = cls.expand(B, -1).contiguous()

        x = vit_cuda.pos_encoding(x, cls_b, self.pos_embed)

        for block in self.blocks:
            x = block(x, self.scale, self.eps)

        x = vit_cuda.layernorm_forward(x, self.norm_gamma, self.norm_beta, self.eps)
        out = vit_cuda.classifier_forward(x, self.head_weight, self.head_bias)

        return out

    def capture_graph(self, static_input, num_warmup=3):
        """Capture the full forward pass as a CUDA graph.

        Must be called on an already-trained, eval-mode model. `static_input`
        is a CUDA float32 tensor of the exact shape the graph will replay; its
        storage is used as the graph's input buffer, so keep a reference.

        After capture, `model(x)` copies `x` into the static input and replays
        the graph. Shapes and device of `x` must match `static_input`.
        """
        if not static_input.is_cuda:
            raise RuntimeError("capture_graph: static_input must be a CUDA tensor")
        if static_input.dtype != torch.float32:
            raise RuntimeError("capture_graph: static_input must be float32")
        if getattr(self, '_graph', None) is not None:
            raise RuntimeError("capture_graph: a graph is already captured")

        static_input = static_input.contiguous()
        self.eval()

        # Warm up on a side stream so kernels are JIT'd and the caching
        # allocator's graph memory pool is populated before capture begins.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(num_warmup):
                    self._forward_impl(static_input)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # Capture on the current stream.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            with torch.no_grad():
                self._static_output = self._forward_impl(static_input)
        self._static_input = static_input
        torch.cuda.synchronize()
        return self._graph

def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def main():
    parser = argparse.ArgumentParser(description="Run Custom CUDA ViT Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    # Standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()

    model = ViTCUDA().cuda()
    model.eval()

    labels = get_imagenet_labels()

    # Warmup
    with torch.no_grad():
        _ = model(input_tensor)

    # Actual inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)

    print(f"\nPredictions for {args.image}:")
    for i in range(top5_prob.size(0)):
        class_id = top5_catid[i].item()
        score = top5_prob[i].item()
        print(f"{i+1}: {labels[class_id]} ({score * 100:.2f}%)")

if __name__ == "__main__":
    main()
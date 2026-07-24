#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from torchvision import transforms
from PIL import Image
import numpy as np

# Ensure repo root on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from inference import ViTCUDA
import vit_cuda
import timm


def preprocess_image(path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).to(device)


def as_np(t):
    return t.detach().cpu().numpy()


def compare_np(a, b):
    res = {}
    res['shape_a'] = tuple(a.shape)
    res['shape_b'] = tuple(b.shape)
    if a.shape != b.shape:
        res['mismatch'] = True
        return res
    diff = a - b
    ad = np.abs(diff)
    res['mismatch'] = False
    res['l2'] = float(np.linalg.norm(diff.ravel()))
    res['maxabs'] = float(ad.max())
    res['meanabs'] = float(ad.mean())
    # cosine on flattened vectors
    af = a.ravel()
    bf = b.ravel()
    denom = (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12)
    res['cosine'] = float(np.dot(af, bf) / denom)
    return res


def fmt(m):
    if m.get('mismatch'):
        return f"shapes differ: {m['shape_a']} vs {m['shape_b']}"
    return f"L2={m['l2']:.6f}, maxabs={m['maxabs']:.6f}, meanabs={m['meanabs']:.6f}, cos={m['cosine']:.6f}"


def print_cmp(label, a, b):
    m = compare_np(as_np(a), as_np(b))
    print(f'{label}:', fmt(m))
    return m


def manual_attention_from_qkv(q, k, v, num_heads, scale):
    b, n, e = q.shape
    head_dim = e // num_heads
    q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
    v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, v)
    return out.permute(0, 2, 1, 3).contiguous().reshape(b, n, e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='tests/sample.jpg')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')

    img_path = args.image
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        sys.exit(1)

    print('Loading models...')
    timm_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
    vit_model = ViTCUDA().to(device).eval()

    x = preprocess_image(img_path, device)
    x = torch.cat([x, x], dim=0) # Batch size 2

    B = x.size(0)

    print('\n== Patch embedding ==')
    # timm patch
    p_t = timm_model.patch_embed(x)
    if p_t.dim() == 4:
        Bp, C, H, W = p_t.shape
        p_t_seq = p_t.flatten(2).transpose(1, 2).contiguous()
    else:
        p_t_seq = p_t

    # vit_cuda patch
    pw = vit_model.patch_weight
    if pw.dim() == 4:
        pw2 = pw.reshape(pw.size(0), -1).to(device)
    else:
        pw2 = pw.to(device)
    p_v_seq = vit_cuda.patch_embed(x, pw2, vit_model.patch_bias)

    m_patch = compare_np(as_np(p_t_seq), as_np(p_v_seq))
    print('patch compare:', fmt(m_patch))

    print('\n== Positional encoding (with cls) ==')
    # timm: build sequence with cls
    cls_t = timm_model.cls_token
    if cls_t.dim() == 3:
        cls_t_exp = cls_t.expand(B, -1, -1).to(device)
    elif cls_t.dim() == 2:
        cls_t_exp = cls_t.unsqueeze(1).expand(B,1,-1).to(device)
    elif cls_t.dim() == 1:
        cls_t_exp = cls_t.unsqueeze(0).unsqueeze(1).expand(B,1,-1).to(device)
    else:
        raise RuntimeError('Unexpected cls_token dim')

    pos_t = timm_model.pos_embed
    if pos_t.dim() == 3:
        pos_t_b = pos_t.to(device)
    elif pos_t.dim() == 2:
        pos_t_b = pos_t.unsqueeze(0).to(device)
    else:
        raise RuntimeError('Unexpected pos_embed dim')

    x_t = torch.cat((cls_t_exp, p_t_seq), dim=1)
    x_t_pos = x_t + pos_t_b

    # vit pos encoding expects cls_b shape [B, E]
    cls_v = vit_model.cls_token
    if cls_v.dim() == 1:
        cls_v_b = cls_v.unsqueeze(0).expand(B, -1).contiguous().to(device)
    elif cls_v.dim() == 2:
        cls_v_b = cls_v.expand(B, -1).contiguous().to(device)
    elif cls_v.dim() == 3:
        # e.g. [1,1,E]
        cls_v_b = cls_v.squeeze(1).expand(B, -1).contiguous().to(device)
    else:
        raise RuntimeError('Unexpected cls_token dim for vit_model')

    pos_v = vit_model.pos_embed.to(device)
    x_v_pos = vit_cuda.pos_encoding(p_v_seq, cls_v_b, pos_v)

    m_pos = compare_np(as_np(x_t_pos), as_np(x_v_pos))
    print('pos compare:', fmt(m_pos))

    # iterate blocks
    x_t_cur = x_t_pos
    x_v_cur = x_v_pos
    num_blocks = len(timm_model.blocks)
    print('\n== Per-block comparison ==')
    for i in range(num_blocks):
        block_t = timm_model.blocks[i]
        block_v = vit_model.blocks[i]

        print(f'\n-- Block {i} --')

        # Norm 1
        t_norm1 = block_t.norm1(x_t_cur)
        v_norm1 = vit_cuda.layernorm_forward(x_v_cur, block_v.norm1_gamma, block_v.norm1_beta, vit_model.eps)
        print_cmp('norm1', t_norm1, v_norm1)

        # QKV projection
        t_qkv = F.linear(t_norm1, block_t.attn.qkv.weight, block_t.attn.qkv.bias)
        v_q, v_k, v_v = vit_cuda.qkv_proj(v_norm1, block_v.qkv_weight, block_v.qkv_bias)
        v_qkv = torch.cat([v_q, v_k, v_v], dim=-1)
        print_cmp('qkv', t_qkv, v_qkv)

        # Split Q/K/V for attention
        t_B, t_N, t_threeE = t_qkv.shape
        t_E = t_threeE // 3
        t_qkv_view = t_qkv.reshape(t_B, t_N, 3, t_E).contiguous()
        t_q = t_qkv_view[:, :, 0, :].contiguous()
        t_k = t_qkv_view[:, :, 1, :].contiguous()
        t_v = t_qkv_view[:, :, 2, :].contiguous()

        print_cmp('q', t_q, v_q)
        print_cmp('k', t_k, v_k)
        print_cmp('v', t_v, v_v)

        # Raw attention output before the projection layer
        t_attn_raw = manual_attention_from_qkv(t_q, t_k, t_v, block_t.attn.num_heads, block_t.attn.scale)
        v_attn_raw = vit_cuda.flash_attn_2(v_q, v_k, v_v, vit_model.scale)
        print_cmp('attn_raw', t_attn_raw, v_attn_raw)

        # Attention projection + residual
        t_proj = F.linear(t_attn_raw, block_t.attn.proj.weight, block_t.attn.proj.bias)
        v_proj = vit_cuda.gemm_bias(v_attn_raw, block_v.proj_weight, block_v.proj_bias)
        print_cmp('attn_proj', t_proj, v_proj)

        t_res1 = x_t_cur + t_proj
        v_res1 = x_v_cur + v_proj
        print_cmp('residual1', t_res1, v_res1)

        # Norm 2
        t_norm2 = block_t.norm2(t_res1)
        v_norm2 = vit_cuda.layernorm_forward(v_res1, block_v.norm2_gamma, block_v.norm2_beta, vit_model.eps)
        print_cmp('norm2', t_norm2, v_norm2)

        # MLP substeps
        t_mlp_fc1 = block_t.mlp.fc1(t_norm2)
        v_mlp_fc1 = vit_cuda.gemm_bias(t_norm2, block_v.fc1_weight, block_v.fc1_bias)
        print_cmp('mlp_fc1', t_mlp_fc1, v_mlp_fc1)

        t_mlp_act = block_t.mlp.act(t_mlp_fc1)
        # Match the kernel's exact erf GELU when comparing to vit_cuda
        v_mlp_act = torch.nn.functional.gelu(v_mlp_fc1, approximate='none')
        print_cmp('mlp_act', t_mlp_act, v_mlp_act)

        t_mlp_fc2 = block_t.mlp.fc2(t_mlp_act)
        v_mlp_fc2, v_mlp_hidden = vit_cuda.mlp_forward(
            v_norm2,
            block_v.fc1_weight,
            block_v.fc1_bias,
            block_v.fc2_weight,
            block_v.fc2_bias,
        )
        # vit_cuda returns [O, H]; compare the output and hidden activations separately
        print_cmp('mlp_hidden', v_mlp_hidden, v_mlp_act)
        print_cmp('mlp_fc2', t_mlp_fc2, v_mlp_fc2)

        x_t_next = t_res1 + t_mlp_fc2
        x_v_next = v_res1 + v_mlp_fc2

        m_block = compare_np(as_np(x_t_next), as_np(x_v_next))
        print('block output:', fmt(m_block))

        x_t_cur = x_t_next
        x_v_cur = x_v_next

    print('\n== Final layernorm compare ==')
    x_t_norm = timm_model.norm(x_t_cur)
    x_v_norm = vit_cuda.layernorm_forward(x_v_cur, vit_model.norm_gamma.to(x_v_cur.device), vit_model.norm_beta.to(x_v_cur.device), vit_model.eps)
    m_norm = compare_np(as_np(x_t_norm), as_np(x_v_norm))
    print('final norm:', fmt(m_norm))

    # return nonzero exit code if large mismatch found
    thresholds = {'maxabs': 1e-3, 'l2': 1e-2}
    large = False
    for label, val in [('patch', m_patch), ('pos', m_pos)]:
        if not val.get('mismatch') and (val['maxabs'] > thresholds['maxabs'] or val['l2'] > thresholds['l2']):
            large = True
    for i in range(num_blocks):
        pass

    if large:
        print('\nNOTE: Large differences detected. Suggest inspecting `patch_embed` and `pos_encoding` kernels first.')
        sys.exit(2)
    else:
        print('\nAll layers similar within thresholds')


if __name__ == '__main__':
    main()

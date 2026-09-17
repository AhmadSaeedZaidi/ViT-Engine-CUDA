from PIL import Image
import torch
from torchvision import transforms
import gradio as gr

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from vit_cuda.model import ViTCUDA, get_imagenet_labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_labels():
    try:
        return get_imagenet_labels()
    except Exception:
        return [f"class_{i}" for i in range(1000)]


model = None
labels = None


def init_model():
    global model
    if model is None:
        model = ViTCUDA().cuda().eval()


def predict_image(img):
    init_model()
    if img is None:
        return {}

    img_t = preprocess(img).unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.nn.functional.softmax(logits[0], dim=0)

    top5_prob, top5_idx = torch.topk(probs, 5)
    results = []
    for p, idx in zip(top5_prob.tolist(), top5_idx.tolist()):
        lbl = labels[idx] if idx < len(labels) else str(idx)
        results.append((lbl, float(p)))

    return {r[0]: r[1] for r in results}


def main():
    global labels
    labels = load_labels()

    try:
        input_component = gr.Image(type="pil", label="Upload")
    except TypeError:
        input_component = gr.inputs.Image(type="pil", label="Upload")

    iface = gr.Interface(
        fn=predict_image,
        inputs=[input_component],
        outputs=gr.Label(num_top_classes=5),
        title="ViTCUDA ImageNet Demo",
        description="Upload an image. Runs on GPU with compiled vit_cuda backend.",
        flagging_mode="never",
    )
    iface.launch(server_name="0.0.0.0", share=False)


if __name__ == "__main__":
    main()
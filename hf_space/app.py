import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from huggingface_hub import hf_hub_download

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_SIZE = 224
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_DESC = {
    'akiec': 'Actinic Keratoses / Bowen\'s Disease',
    'bcc':   'Basal Cell Carcinoma',
    'bkl':   'Benign Keratosis-like Lesions',
    'df':    'Dermatofibroma',
    'mel':   'Melanoma',
    'nv':    'Melanocytic Nevi',
    'vasc':  'Vascular Lesions',
}
MALIGNANT = {'mel', 'bcc', 'akiec'}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model definition (must match training) ─────────────────────────────────────
class HybridModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet_features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.effnet = timm.create_model('tf_efficientnetv2_s.in1k', pretrained=False, num_classes=0)

        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            f1 = self.resnet_features(dummy).view(1, -1).shape[1]
            f2 = self.effnet(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(f1 + f2, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        f1 = self.resnet_features(x).view(x.size(0), -1)
        f2 = self.effnet(x).view(x.size(0), -1)
        return self.classifier(torch.cat((f1, f2), dim=1))


# ── Grad-CAM helper ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        # Hook onto ResNet's last conv block (layer4), not avgpool
        model.resnet_features[-2].register_forward_hook(self._save_act)
        model.resnet_features[-2].register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output

    def _save_grad(self, _, __, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, tensor, class_idx):
        self.model.zero_grad()
        out = self.model(tensor)
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((self.activations * weights).sum(dim=1).squeeze())
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()


# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    model = HybridModel(num_classes=7).to(device)
    # Loads weights from the same HF Space repo (upload best_model.pth to the repo)
    weights_path = "best_model.pth"
    if not os.path.exists(weights_path):
        # Fallback: download from HF Hub if you've uploaded it as a model repo
        # Replace YOUR_HF_USERNAME/YOUR_MODEL_REPO with your actual repo
        weights_path = hf_hub_download(
            repo_id=os.environ.get("MODEL_REPO", "rajasri77/skin-cancer-model"),
            filename="best_model.pth",
        )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


model = load_model()
gradcam = GradCAM(model)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(image: Image.Image):
    if image is None:
        return None, None, "Please upload an image."

    image = image.convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    is_malignant = pred_class in MALIGNANT
    risk_label = "MALIGNANT" if is_malignant else "BENIGN"
    risk_color = "#e74c3c" if is_malignant else "#2ecc71"

    # ── Grad-CAM overlay ──────────────────────────────────────────────────────
    tensor_grad = TRANSFORM(image).unsqueeze(0).to(device).requires_grad_(True)
    # re-enable grad for CAM
    with torch.enable_grad():
        cam = gradcam(tensor_grad, pred_idx)

    cam_resized = np.array(Image.fromarray(cam).resize(image.size, Image.BILINEAR))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM (ResNet branch)", fontsize=13)
    axes[1].axis("off")
    plt.suptitle(
        f"Predicted: {pred_class.upper()}  —  {CLASS_DESC[pred_class]}  [{risk_label}]",
        fontsize=12, color=risk_color, fontweight="bold",
    )
    plt.tight_layout()
    cam_fig_path = "/tmp/gradcam.png"
    plt.savefig(cam_fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    # ── Probability bar chart ─────────────────────────────────────────────────
    sorted_idx = np.argsort(probs)[::-1]
    colors = ["#e74c3c" if CLASS_NAMES[i] in MALIGNANT else "#3498db" for i in sorted_idx]
    fig2, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(
        [CLASS_NAMES[i] for i in sorted_idx],
        [probs[i] * 100 for i in sorted_idx],
        color=colors, edgecolor="white", height=0.6,
    )
    ax.set_xlabel("Confidence (%)", fontsize=11)
    ax.set_title("Class Probabilities", fontsize=13)
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, [probs[i] * 100 for i in sorted_idx]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Malignant"),
        Patch(color="#3498db", label="Benign"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    bar_fig_path = "/tmp/probs.png"
    plt.savefig(bar_fig_path, dpi=120, bbox_inches="tight")
    plt.close()

    # ── Result text ───────────────────────────────────────────────────────────
    result_text = (
        f"## {pred_class.upper()} — {CLASS_DESC[pred_class]}\n\n"
        f"**Risk:** `{risk_label}`\n\n"
        f"**Confidence:** {confidence:.1f}%\n\n"
        f"### Top-3 Predictions\n"
    )
    top3 = np.argsort(probs)[::-1][:3]
    for rank, i in enumerate(top3, 1):
        flag = " (malignant)" if CLASS_NAMES[i] in MALIGNANT else ""
        result_text += f"{rank}. **{CLASS_NAMES[i]}** — {probs[i]*100:.1f}%{flag}\n"

    result_text += (
        "\n---\n"
        "> **Disclaimer:** This tool is for educational purposes only. "
        "It is not a substitute for professional medical diagnosis. "
        "Always consult a qualified dermatologist."
    )

    return Image.open(cam_fig_path), Image.open(bar_fig_path), result_text


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft(), title="Skin Cancer Detector") as demo:
    gr.Markdown(
        """
        # Skin Lesion Classification
        ### ResNet50 + EfficientNetV2-S Ensemble · HAM10000 Dataset · 7 Classes
        Upload a dermoscopic image of a skin lesion to get an AI-powered classification.
        Red bars = malignant classes · Blue bars = benign classes
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Skin Lesion Image")
            run_btn = gr.Button("Analyse", variant="primary", size="lg")
            gr.Examples(
                examples=[],   # add example image paths here after uploading
                inputs=img_input,
            )
        with gr.Column(scale=2):
            result_md = gr.Markdown(label="Result")
            with gr.Row():
                cam_out  = gr.Image(label="Grad-CAM Heatmap", show_label=True)
                prob_out = gr.Image(label="Class Probabilities", show_label=True)

    run_btn.click(
        fn=predict,
        inputs=img_input,
        outputs=[cam_out, prob_out, result_md],
    )
    img_input.change(
        fn=predict,
        inputs=img_input,
        outputs=[cam_out, prob_out, result_md],
    )

    gr.Markdown(
        """
        ---
        **Classes:** `akiec` Actinic Keratoses · `bcc` Basal Cell Carcinoma · `bkl` Benign Keratosis ·
        `df` Dermatofibroma · `mel` Melanoma · `nv` Melanocytic Nevi · `vasc` Vascular Lesions
        """
    )

if __name__ == "__main__":
    demo.launch()

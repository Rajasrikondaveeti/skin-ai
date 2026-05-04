---
title: Skin Cancer Detector
emoji: 🔬
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
license: mit
short_description: Skin cancer detection ensemble, 88.7% val accuracy
python_version: "3.11"
---

# Skin Cancer Detection — Ensemble Deep Learning Model

> **Try the live app:** [rajasri77/skin-cancer-detector](https://huggingface.co/spaces/rajasri77/skin-cancer-detector)

Classifies dermoscopic skin lesion images into **7 categories** using a hybrid ensemble of ResNet50 and EfficientNetV2-S, trained on the HAM10000 dataset. Upload a skin lesion image to get a classification, confidence scores, and a Grad-CAM heatmap showing which regions influenced the prediction.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | **88.67%** (Epoch 18) |
| Best Validation Loss | **0.398** (Epoch 16) |
| Final Training Loss | **0.055** (Epoch 20) |
| Epochs Trained | 20 |

### Training Curve Summary

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|----------|-------------|
| 1 | 1.237 | 0.951 | 57.6% |
| 5 | 0.453 | 0.544 | 76.8% |
| 10 | 0.175 | 0.446 | 83.5% |
| 15 | 0.080 | 0.448 | 87.1% |
| 18 | 0.063 | 0.412 | **88.7%** |
| 20 | 0.055 | 0.412 | 88.5% |

---

## Architecture

Two deep learning backbones fine-tuned on HAM10000 run in parallel — their feature vectors are concatenated and passed through a shared MLP classifier:

```
ResNet50 (2048-d) ──┐
                    ├─► concat ─► Linear(3328→512) ─► BN+ReLU+Dropout(0.3)
EfficientNetV2-S ───┘              ─► Linear(512→256) ─► BN+ReLU+Dropout(0.2)
                                   ─► Linear(256→7)
```

---

## Dataset — HAM10000

10,015 dermoscopic images across 7 classes:

| Class | Description | Type |
|-------|-------------|------|
| `nv` | Melanocytic Nevi | Benign |
| `mel` | Melanoma | **Malignant** |
| `bkl` | Benign Keratosis-like Lesions | Benign |
| `bcc` | Basal Cell Carcinoma | **Malignant** |
| `akiec` | Actinic Keratoses / Bowen's Disease | **Malignant** |
| `vasc` | Vascular Lesions | Benign |
| `df` | Dermatofibroma | Benign |

---

## Training Details

- **Optimiser:** AdamW (lr=1e-4)
- **LR Schedule:** CosineAnnealingLR
- **Loss:** CrossEntropyLoss with balanced class weights
- **Augmentation:** Random crops, flips, rotation, colour jitter
- **Mixed Precision:** AMP (float16) on GPU
- **Best checkpoint:** Epoch 18, saved to `best_model.pth`

---

## Interpretability — Grad-CAM

The app generates a **Grad-CAM heatmap** by hooking into ResNet50's final convolutional block (`layer4`, spatial resolution 7×7). This highlights which regions of the lesion most influenced the classification decision.

---

## Tech Stack

| Component | Detail |
|-----------|--------|
| Framework | PyTorch 2.x |
| Backbones | ResNet50 + EfficientNetV2-S (timm) |
| UI | Gradio 6.14.0 |
| Python | 3.11 |
| Model weights | Hosted on [rajasri77/skin-cancer-model](https://huggingface.co/rajasri77/skin-cancer-model) (556 MB) |

---

> **Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified dermatologist.

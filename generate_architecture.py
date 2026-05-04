import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(-0.4, 10)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# ── colour palette ──────────────────────────────────────────────
C_INPUT   = "#4A90D9"
C_RESNET  = "#E67E22"
C_EFFNET  = "#27AE60"
C_CONCAT  = "#8E44AD"
C_FC      = "#2980B9"
C_OUTPUT  = "#C0392B"
C_TEXT    = "#FFFFFF"
C_SUB     = "#BDC3C7"
C_ARROW   = "#95A5A6"

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.25):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          linewidth=1.5, edgecolor=color,
                          facecolor=color + "33")
    ax.add_patch(rect)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C_TEXT)
    if sublabel:
        ax.text(x, y - 0.28, sublabel,
                ha="center", va="center", fontsize=7.5, color=C_SUB)

def arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14))

# ── Title ────────────────────────────────────────────────────────
ax.text(7, 9.5, "Skin Cancer Detection — Ensemble Architecture",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_TEXT)

# ── Input ────────────────────────────────────────────────────────
box(ax, 7, 8.6, 3.2, 0.7, C_INPUT, "Input Dermoscopic Image", "224 × 224 RGB")

# arrows input → both branches
arrow(ax, 5.5, 8.25, 3.8, 7.45)
arrow(ax, 8.5, 8.25, 10.2, 7.45)

# ── Branch labels ────────────────────────────────────────────────
ax.text(3.0, 7.85, "Branch 1", ha="center", fontsize=8,
        color=C_RESNET, fontstyle="italic")
ax.text(11.0, 7.85, "Branch 2", ha="center", fontsize=8,
        color=C_EFFNET, fontstyle="italic")

# ── ResNet50 branch ──────────────────────────────────────────────
box(ax, 3.0, 7.1, 3.6, 0.7, C_RESNET, "ResNet50", "ImageNet pre-trained · fine-tuned")
arrow(ax, 3.0, 6.75, 3.0, 6.15)
box(ax, 3.0, 5.8, 3.0, 0.6, C_RESNET, "Global Avg Pool", "2048-d feature vector")

# ── EfficientNetV2-S branch ──────────────────────────────────────
box(ax, 11.0, 7.1, 3.6, 0.7, C_EFFNET, "EfficientNetV2-S", "ImageNet pre-trained · fine-tuned")
arrow(ax, 11.0, 6.75, 11.0, 6.15)
box(ax, 11.0, 5.8, 3.0, 0.6, C_EFFNET, "Global Avg Pool", "1280-d feature vector")

# ── arrows → concat ──────────────────────────────────────────────
arrow(ax, 3.0, 5.5, 5.8, 4.85)
arrow(ax, 11.0, 5.5, 8.2, 4.85)

# ── Concat ───────────────────────────────────────────────────────
box(ax, 7, 4.55, 3.6, 0.6, C_CONCAT, "Feature Concatenation", "3328-d  (2048 + 1280)")

# ── MLP Classifier ───────────────────────────────────────────────
arrow(ax, 7, 4.25, 7, 3.65)
box(ax, 7, 3.35, 4.2, 0.6, C_FC, "FC Layer 1", "Linear(3328→512)  ·  BN  ·  ReLU  ·  Dropout(0.3)")

arrow(ax, 7, 3.05, 7, 2.45)
box(ax, 7, 2.15, 4.2, 0.6, C_FC, "FC Layer 2", "Linear(512→256)  ·  BN  ·  ReLU  ·  Dropout(0.2)")

arrow(ax, 7, 1.85, 7, 1.25)
box(ax, 7, 0.95, 3.2, 0.6, C_OUTPUT, "Output Layer", "Linear(256→7)  ·  Softmax")

# ── Class labels ─────────────────────────────────────────────────
classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
colors_cls = [C_EFFNET, C_OUTPUT, C_EFFNET, C_OUTPUT, C_OUTPUT, C_INPUT, C_INPUT]
n = len(classes)
xs = [7 + (i - n//2) * 1.72 for i in range(n)]

arrow(ax, 7, 0.65, 7, 0.38)
for i, (cls, xc, cc) in enumerate(zip(classes, xs, colors_cls)):
    rect = FancyBboxPatch((xc - 0.7, 0.1), 1.4, 0.42,
                          boxstyle="round,pad=0.04,rounding_size=0.1",
                          linewidth=1.2, edgecolor=cc, facecolor=cc + "44")
    ax.add_patch(rect)
    ax.text(xc, 0.31, cls, ha="center", va="center",
            fontsize=8, fontweight="bold", color=C_TEXT)

# legend for class colours
ax.text(0.3, 0.38, "Malignant", fontsize=7.5, color=C_OUTPUT, va="center")
ax.text(0.3, 0.14, "Benign", fontsize=7.5, color=C_EFFNET, va="center")

# ── Grad-CAM callout ─────────────────────────────────────────────
# Grad-CAM box sits directly to the LEFT of ResNet50 at the same height
# ResNet50 left edge = 3.0 - 3.6/2 = 1.2, so keep right edge < 1.2
gc_cx, gc_cy, gc_w, gc_h = 0.55, 7.1, 1.0, 0.65
gc_box = FancyBboxPatch((gc_cx - gc_w/2, gc_cy - gc_h/2), gc_w, gc_h,
                        boxstyle="round,pad=0.05,rounding_size=0.1",
                        linewidth=1.2, edgecolor="#F39C12",
                        facecolor="#F39C1222", linestyle="dashed")
ax.add_patch(gc_box)
ax.text(gc_cx, gc_cy + 0.12, "Grad-CAM", ha="center", va="center",
        fontsize=8, fontweight="bold", color="#F39C12")
ax.text(gc_cx, gc_cy - 0.1, "hooked on layer4", ha="center", va="center",
        fontsize=6.5, color="#F39C12")
# Horizontal arrow pointing right into ResNet50 left edge
arrow(ax, gc_cx + gc_w/2, gc_cy, 1.2, gc_cy, color="#F39C12")

plt.tight_layout()
plt.savefig("architecture.png", dpi=180, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved: architecture.png")

# classifier_analytics_app_modern.py
import os
import time
import torch
import requests
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.models import ResNet18_Weights # <-- IMPORT ADDED

# ---------------- SETTINGS ----------------
BRAND_NAME = "Hadi Wasim — Computer Vision & AI Researcher"
ACCENT_A = "#7C3AED"  # purple
ACCENT_B = "#06B6D4"  # cyan
BG = "#0b0f1a"        # dark
CARD_BG = "#0f1724"
TEXT = "#E6EEF3"
TMP_DIR = "tmp_outputs"
os.makedirs(TMP_DIR, exist_ok=True)

# ---------------- MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Updated model loading to use 'weights' argument
model = torch.hub.load("pytorch/vision:v0.6.0", "resnet18", weights=ResNet18_Weights.DEFAULT)
model.eval().to(device)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.eval().to(device)

# Labels (ImageNet human-readable)
labels = requests.get("https://git.io/JJkYN").text.strip().split("\n")

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------- UTILITY ----------------
def compute_entropy(probs):
    eps = 1e-12
    p = probs.clamp(min=eps)
    return float(-(p * p.log()).sum().item())

def make_topk_bar(top_labels, top_probs, title="Top Predictions"):
    fig, ax = plt.subplots(figsize=(6, 3), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.barh(top_labels, top_probs, color=ACCENT_B)
    ax.set_xlabel("Confidence", color=TEXT)
    ax.set_title(title, color=TEXT)
    ax.set_xlim(0, 1)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig

def feature_histogram(feature_vec):
    fig, ax = plt.subplots(figsize=(6,3), facecolor=BG)
    fig.patch.set_facecolor(BG)
    ax.hist(feature_vec, bins=40, color=ACCENT_A)
    ax.set_title("Feature Vector Distribution (penultimate layer)", color=TEXT)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()
    return fig

# ---------------- INFERENCE ----------------
def analyze_image(image):
    if image is None:
        return {}, None, None, "No image uploaded.", None, None

    pil_img = Image.fromarray(image).convert("RGB")

    # Image stats
    arr = np.array(pil_img).astype(np.float32) / 255.0
    stats = {
        'Filename': f"uploaded_{int(time.time())}.png",
        'Mode': pil_img.mode,
        'Size': f"{pil_img.width} x {pil_img.height}",
        'R_mean': float(np.mean(arr[:,:,0])),
        'G_mean': float(np.mean(arr[:,:,1])),
        'B_mean': float(np.mean(arr[:,:,2])),
        'R_std': float(np.std(arr[:,:,0])),
        'G_std': float(np.std(arr[:,:,1])),
        'B_std': float(np.std(arr[:,:,2])),
    }

    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
        feat = feature_extractor(img_tensor).reshape(-1).cpu().numpy()

    num_classes = probs.size(0)
    num_labels = min(len(labels), num_classes)
    probs_cpu = probs.cpu().numpy()
    df = pd.DataFrame({"label": labels[:num_labels], "probability": probs_cpu[:num_labels]})
    df_sorted = df.sort_values("probability", ascending=False).reset_index(drop=True)

    # Top-5 bar
    topk = df_sorted.iloc[:5]
    top_labels = topk['label'].tolist()[::-1]
    top_probs = topk['probability'].tolist()[::-1]
    bar_fig = make_topk_bar(top_labels, top_probs)

    # Feature histogram
    feat_fig = feature_histogram(feat)

    # CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(TMP_DIR, f"predictions_{ts}.csv")
    df_sorted.to_csv(csv_path, index=False)

    # Analytics
    entropy = compute_entropy(probs)
    max_conf = float(probs.max().item())
    mean_top5 = float(topk['probability'][:5].mean())
    analytics_md = f"""
**{BRAND_NAME}** **Model:** ResNet-18 · **Device:** {device}
---
**Image:** {stats['Size']} · Mode: {stats['Mode']}
**RGB Means:** R={stats['R_mean']:.3f}, G={stats['G_mean']:.3f}, B={stats['B_mean']:.3f}  
**RGB Stddevs:** R={stats['R_std']:.3f}, G={stats['G_std']:.3f}, B={stats['B_std']:.3f}

**Model Confidence Metrics**
- Entropy: `{entropy:.4f}` (lower = more certain)
- Max Confidence: `{max_conf:.4f}`
- Top-5 Mean Confidence: `{mean_top5:.4f}`

**Top Prediction:** `{topk.iloc[0]['label']}` — Confidence `{topk.iloc[0]['probability']:.4f}`
    """

    top3 = {row['label']: float(row['probability']) for _, row in df_sorted.iloc[:3].iterrows()}

    return top3, bar_fig, df_sorted, analytics_md, feat_fig, csv_path

# ---------------- GRADIO UI ----------------
css = f"""
body {{ background: {BG}; color: {TEXT}; }}
.gradio-container {{ background: {BG}; color: {TEXT}; }}
.header {{
  padding: 18px;
  border-radius: 8px;
  background: linear-gradient(90deg, rgba(124,58,237,0.12), rgba(6,182,212,0.06));
  display:flex;
  justify-content:space-between;
  align-items:center;
}}
.brand {{
  font-weight:700;
  font-size:20px;
  color: {TEXT};
}}
.tagline {{
  font-size:12px;
  color: #9fb7c7;
}}
.card {{
  background: {CARD_BG};
  border-radius: 10px;
  padding: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.6);
  color: {TEXT};
}}
.btn-primary {{ background: linear-gradient(90deg, {ACCENT_A}, {ACCENT_B}); color: #fff; }}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    # Header
    with gr.Row(elem_classes="header"):
        with gr.Column(scale=8):
            gr.HTML(f"<div class='brand'>{BRAND_NAME}</div><div class='tagline'>Dark Neon Vision Analytics</div>")
        with gr.Column(scale=2):
            gr.HTML(f"<div style='text-align:right'><small style='color:#9fb7c7'>ResNet-18 · Neon Theme</small></div>")

    # Main content
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Upload Image")
                image_in = gr.Image(label="Upload (or drag & drop)", type="numpy")
                analyze_btn = gr.Button("Analyze Image", elem_id="analyze_btn")
        with gr.Column(scale=8):
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Results & Insights")
                with gr.Row():
                    col1 = gr.Column(scale=4)
                    col2 = gr.Column(scale=8)
                    with col1:
                        label_out = gr.Label(num_top_classes=3, label="Top-3 Predictions")
                        download_file = gr.File(label="Download CSV of full probabilities")
                    with col2:
                        chart_out = gr.Plot(label="Top-5 Confidence Chart")
                # Table + Analytics + Feature plot
                with gr.Row():
                    # --- THIS IS THE CORRECTED BLOCK ---
                    df_out = gr.Dataframe(
                        headers=["label", "probability"],
                        datatype=["str", "number"],  # <-- Corrected parameter
                        type="pandas",             # <-- Added parameter
                        label="All Class Probabilities (sortable)",
                        interactive=True,
                        row_count=(1, "dynamic"),
                        col_count=(2, "fixed")
                    )
                    # -----------------------------------
                with gr.Row():
                    md_out = gr.Markdown()
                with gr.Row():
                    feat_out = gr.Plot(label="Feature Vector Distribution")

    analyze_btn.click(
        fn=analyze_image,
        inputs=image_in,
        outputs=[label_out, chart_out, df_out, md_out, feat_out, download_file]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
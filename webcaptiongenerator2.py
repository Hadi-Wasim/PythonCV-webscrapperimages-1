import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
from urllib.parse import urljoin
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
import gradio as gr
import time
import pandas as pd  # For CSV export

warnings.filterwarnings("ignore", message=".*TorchScript is not supported.*")

# ====================== MODEL SETUP ======================
def setup_model():
    print("Loading BLIP model... (first time takes ~20-30s)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return processor, model, device

processor, model, device = setup_model()
if model is None:
    exit()

# ====================== CAPTION FUNCTION (FIXED) ======================
def generate_caption(pil_image: Image.Image) -> str:
    try:
        # Critical fix: re-open image after verify() consumes it
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Resize only if too large (avoid OOM)
        if pil_image.size[0] > 1024 or pil_image.size[1] > 1024:
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_length=60, num_beams=5)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.capitalize()
    except Exception as e:
        return f"[Error: {str(e)[:40]}]"

# ====================== MAIN SCRAPER (ROBUST) ======================
def scrape_and_caption(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    gallery = []        # List of (PIL.Image, str)
    failed = 0

    yield "Fetching page...", gallery, failed, gr.update(visible=False)

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        yield f"Failed to fetch page: {e}", gallery, failed, gr.update(visible=False)
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    img_tags = soup.find_all("img")

    candidates = []
    for tag in img_tags:
        src = tag.get("src") or tag.get("data-src")
        if not src:
            continue
        full_url = urljoin(url, src)
        if not full_url.startswith("http"):
            continue
        if full_url.lower().endswith((".svg", ".gif")) or "logo" in full_url.lower():
            continue
        candidates.append(full_url)

    candidates = list(dict.fromkeys(candidates))[:20]  # Dedupe + limit

    yield f"Found {len(candidates)} images. Processing...", gallery, failed, gr.update(visible=False)

    for i, img_url in enumerate(candidates):
        status = f"Processing {i+1}/{len(candidates)}..."
        yield status, gallery, failed, gr.update(visible=False)

        for attempt in range(3):
            try:
                r = requests.get(img_url, headers=headers, timeout=15)
                if len(r.content) < 2000:  # Skip tiny images
                    raise ValueError("Too small")
                if not r.headers.get("Content-Type", "").startswith("image/"):
                    raise ValueError("Not an image")

                bytes_io = io.BytesIO(r.content)
                img = Image.open(bytes_io)
                img.verify()  # Check integrity

                # Critical: Re-open after verify() consumes the stream
                bytes_io.seek(0)
                img = Image.open(bytes_io)

                caption = generate_caption(img)
                gallery.append((img, caption))
                break  # Success

            except Exception as e:
                if attempt == 2:
                    failed += 1
                    print(f"Failed {img_url}: {e}")
                time.sleep(0.5)
            else:
                break

        yield status, gallery, failed, gr.update(visible=False)

    final_status = f"Done! {len(gallery)} captioned, {failed} failed."
    yield final_status, gallery, failed, gr.update(visible=True)

# ====================== GRADIO UI ======================
css = """
body, .gradio-container { background: #0b0f1a; color: #E6EEF3; }
.card { background: #0f1724; border-radius: 12px; padding: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.6); margin: 10px 0; }
.gr-button { background: linear-gradient(90deg, #7C3AED, #06B6D4); border: none; font-weight: bold; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Webpage Image Captioner with BLIP")
    gr.Markdown("Enter any URL → automatically finds & captions all images")

    with gr.Column(elem_classes="card"):
        url_input = gr.Textbox(placeholder="e.g. wikipedia.org/wiki/Donald_Trump", label="URL")
        btn = gr.Button("Start Captioning", elem_classes="gr-button")

    with gr.Column(elem_classes="card"):
        status = gr.Markdown("Ready")

    with gr.Column(elem_classes="card"):
        gallery = gr.Gallery(label="Results", columns=3, height="auto", object_fit="contain")

    with gr.Column(elem_classes="card"):
        download_btn = gr.Button("Download CSV", visible=False)
        csv_file = gr.File(label="CSV will appear here")

    def create_csv(results):
        if not results:
            return None
        df = pd.DataFrame([{"Caption": cap} for _, cap in results])
        path = "captions.csv"
        df.to_csv(path, index=False)
        return path

    download_btn.click(create_csv, inputs=gallery, outputs=csv_file)

    btn.click(
        fn=scrape_and_caption,
        inputs=url_input,
        outputs=[status, gallery, gr.Number(visible=False), download_btn]
    )

print("Launching at http://127.0.0.1:7860")
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
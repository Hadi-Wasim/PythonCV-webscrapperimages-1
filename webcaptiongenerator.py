import requests
from bs4 import BeautifulSoup
from PIL import Image
import io
from urllib.parse import urljoin
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
import gradio as gr  # <-- Import Gradio

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", message=".*TorchScript is not supported with functionality.*")

# --- 1. Setup Image Captioning Model ---
# (This section is unchanged)

def setup_model():
    """Loads the BLIP image captioning model and processor."""
    print("Loading BLIP model and processor... (This may take a moment)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print(f"Model loaded successfully and running on: {device}")
        return processor, model, device
    
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have 'transformers', 'torch', and 'Pillow' installed.")
        return None, None, None

def generate_caption(image: Image.Image, processor, model, device):
    """Generates a caption for a single PIL image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    inputs = processor(images=image, text=None, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_length=75)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

# --- 2. NEW GRADIO-COMPATIBLE Scraping and Processing ---

def gradio_scrape_and_caption(base_url, processor, model, device):
    """
    A generator function that scrapes images, captions them, and yields
    live updates for the Gradio interface.
    """
    
    # --- 2a. Initial Setup & URL Validation ---
    if not (base_url.startswith('http://') or base_url.startswith('https://')):
        base_url = 'https://' + base_url
    
    status_message = f"Starting scrape for: {base_url}"
    results_gallery = []
    yield status_message, results_gallery  # Yield initial status
    
    try:
        # Download the page
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
    except requests.exceptions.RequestException as e:
        yield f"Error: Could not retrieve URL. {e}", []
        return

    # --- 2b. Parse HTML with BeautifulSoup ---
    status_message = "Page downloaded. Parsing HTML to find images..."
    yield status_message, results_gallery
    
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')
    
    image_urls = set()
    for img in img_tags:
        src = img.get('src')
        if not src:
            continue
            
        absolute_url = urljoin(base_url, src)
        
        # Filter out tiny data URLs or non-http URLs
        if absolute_url.startswith('data:'):
            continue
        if not absolute_url.startswith(('http://', 'https://')):
            continue
            
        image_urls.add(absolute_url)

    if not image_urls:
        yield "No valid image URLs found on this page.", []
        return

    total_images = len(image_urls)
    status_message = f"Found {total_images} unique images. Starting captioning..."
    yield status_message, results_gallery
    
    # --- 2c. Download and Caption Images (Synchronously) ---
    for i, img_url in enumerate(image_urls):
        current_image_filename = img_url.split('/')[-1]
        status_message = f"Processing image {i+1}/{total_images}: {current_image_filename}"
        yield status_message, results_gallery
        
        try:
            # Download the image data with a timeout
            img_response = requests.get(img_url, headers=headers, timeout=10)
            img_response.raise_for_status()
            
            # Open the image from the downloaded bytes
            img_data = io.BytesIO(img_response.content)
            image = Image.open(img_data)
            
            # Generate the caption
            caption = generate_caption(image, processor, model, device)
            
            print(f"  > CAPTION for {img_url}: {caption}")
            
            # Add the (image, caption) tuple to our gallery list
            # We change this from a tuple to a list, as Dataframe expects a list of rows
            results_gallery.append([image, caption])
            
            # Yield the status and the UPDATED gallery
            yield status_message, results_gallery
            
        except requests.exceptions.RequestException as e:
            print(f"  > Failed to download {img_url}: {e}")
        except Image.UnidentifiedImageError:
            print(f"  > Failed: {img_url} was not a valid image format.")
        except Exception as e:
            print(f"  > An unexpected error occurred for {img_url}: {e}")

    # --- 2d. Final Update ---
    status_message = f"All {len(results_gallery)} images captioned successfully!"
    yield status_message, results_gallery

# --- 3. Main execution: Setup Model and Launch Gradio App ---
if __name__ == "__main__":
    
    # 1. Set up the model *before* launching the UI
    # This way, the model is loaded only once.
    processor, model, device = setup_model()
    
    if model is None:
        print("Could not run captioning process because model failed to load.")
        exit() # Exit if model fails to load

    # 2. Define the Gradio UI
    css = """
    body, .gradio-container { background: #0b0f1a; color: #E6EEF3; }
    .card { background: #0f1724; border-radius: 10px; padding: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.6); }
    .label-text { font-size: 18px; font-weight: bold; }
    .gr-button { background: linear-gradient(90deg, #7C3AED, #06B6D4); color: #fff; }
    .gr-gallery { min-height: 500px; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🤖 Automated Webpage Image Captioner")
        gr.Markdown("Enter a URL and this tool will scrape all images, run them through the BLIP Vision-Language Model, and generate captions for each one.")
        
        # --- ADDING THESE BLOCKS BACK IN ---
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 1. Input URL")
            with gr.Row():
                url_input = gr.Textbox(
                    label="Enter URL to scrape",
                    placeholder="e.g., en.wikipedia.org/wiki/IBM",
                    scale=4
                )
                submit_btn = gr.Button("Scrape and Caption", elem_classes="gr-button", scale=1)
        
        with gr.Column(elem_classes="card"):
            gr.Markdown("### 2. Live Status")
            status_output = gr.Markdown("Waiting for URL...")
        # --- END OF ADDED BLOCKS ---

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 3. Results Table")
            table_output = gr.Dataframe(
                headers=["Image", "Generated Caption"],
                datatype=["image", "str"],
                label="Captioned Images Results",
                show_label=False,
                elem_id="results_table",
                interactive=False # User shouldn't edit the results
            )

        # 3. Connect the UI elements
        # We need to bind the 'gradio_scrape_and_caption' function
        # to our model variables.
        # We create a new function that calls it with the model args.
        def start_process(url):
            # This generator will yield updates
            for status, gallery in gradio_scrape_and_caption(url, processor, model, device):
                yield status, gallery
        
        submit_btn.click(
            fn=start_process,
            inputs=url_input,
            outputs=[status_output, table_output]
        )

    # 4. Launch the app
    print("Launching Gradio app... Access it at http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port= 7860)
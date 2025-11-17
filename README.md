# Webpage Image Captioner  
### Automatically extract & caption every image on any website — powered by BLIP + Gradio

![](https://raw.githubusercontent.com/gradio-app/gradio/main/assets/gradio-logo.svg?sanitize=true)  
> **No manual uploads. No APIs. Just paste a URL → get AI captions for all images instantly.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Hugging Face](https://img.shields.io/badge/Model-BLIP%20(base)-orange?logo=huggingface)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![Gradio](https://img.shields.io/badge/Interface-Gradio%204-7C3AED?logo=gradio)](https://gradio.app)

---

## Features

- Zero-click image extraction from any public webpage  
- State-of-the-art captioning using Salesforce BLIP (`blip-image-captioning-base`)  
- Real-time progress updates with live gallery preview  
- Smart filtering – skips SVGs, icons, logos, tiny thumbnails  
- Robust error handling – never crashes on broken images  
- One-click CSV export of all captions  
- Beautiful dark neon UI with custom CSS  
- CPU & GPU ready (auto-detects CUDA)  
- Fully offline-capable after first model download  

---

## Quick Start (30 seconds)

```bash
git clone https://github.com/yourusername/webpage-image-captioner.git
cd webpage-image-captioner

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers gradio beautifulsoup4 requests pillow pandas

python webcaptiongenerator.py
```

Open → http://127.0.0.1:7860

---

## Usage Example

1. Paste any URL:  
   `https://en.wikipedia.org/wiki/Donald_Trump`  
   `https://www.nasa.gov`  
   `https://httpbin.org/image/jpeg`

2. Click **"Start Captioning"**

3. Watch images appear with AI captions in real time

4. When done → **"Download Captions as CSV"**

Done!

---


## Tech Stack

| Component               | Technology                                 |
|-------------------------|--------------------------------------------|
| Vision-Language Model   | Salesforce BLIP (`base`)                   |
| Web Interface           | Gradio 4+                                  |
| HTML Parsing            | BeautifulSoup4                             |
| Image Processing        | PIL / Pillow                               |
| Deep Learning           | PyTorch + Hugging Face Transformers        |
| Export                  | Pandas → CSV                               |

---

## Project Structure

```
webpage-image-captioner/
├── webcaptiongenerator.py     ← Main app (just run this)
├── requirements.txt           ← Optional: pip freeze > requirements.txt
├── captions.csv               ← Generated on export
├── assets/                    ← Put your screenshots here
└── README.md                  ← You're reading it!
```

---

## Roadmap

- [ ] Add BLIP-Large / GIT-Large model toggle
- [ ] Progress bar + estimated time
- [ ] Export as HTML gallery
- [ ] Public Gradio Space deployment
- [ ] Batch URL input (multiple pages)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/webpage-image-captioner&type=Date)](https://star-history.com/#yourusername/webpage-image-captioner)

---

---

**Made with passion by an AI researcher who was tired of uploading images one by one**

⭐ **Love this tool? Star this repo – it helps others find it!** ⭐

---
```

```

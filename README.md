---
title: Florence-2 + SAM 2 Demo
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Florence-2 + SAM 2 Image Recognition Demo

A combined pipeline that uses **Florence-2** for image captioning and object detection, then feeds detected objects into **SAM 2** for high-quality segmentation masks. Includes a **Gradio** web UI for interactive use.

## Architecture

```
Input Image
    │
    ▼
┌──────────────┐
│  Florence-2   │  → Caption + Object Detection (bounding boxes)
│  (Microsoft)  │
└──────┬───────┘
       │ bounding boxes
       ▼
┌──────────────┐
│    SAM 2      │  → Segmentation masks for each detected object
│    (Meta)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Visualization │  → Overlay bboxes, masks, and captions on image
└──────┬───────┘
       │
       ▼
  Gradio Web UI
```

## Features

- **Image Captioning** — Generate natural language descriptions of images
- **Object Detection** — Detect objects with bounding boxes using Florence-2
- **Instance Segmentation** — Pixel-level masks via SAM 2 for each detected object
- **Interactive Web UI** — Upload images and see results instantly via Gradio

## Setup

### Prerequisites

- Python 3.10+
- ~4GB disk space for model weights (downloaded on first run)

### Installation

```bash
git clone https://github.com/caminogr/florence2-sam2-demo.git
cd florence2-sam2-demo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Demo

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## Usage

1. Open the Gradio web UI
2. Upload an image (or select an example)
3. View results:
   - **Caption** — natural language description
   - **Detection** — image with bounding boxes and labels
   - **Segmentation** — image with colored masks overlaid

## Configuration

The pipeline runs on **CPU by default**. To use GPU, ensure CUDA is available — the code auto-detects and uses it.

## Project Structure

```
florence2-sam2-demo/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── florence2_model.py    # Florence-2 inference wrapper
│   ├── sam2_model.py         # SAM 2 inference wrapper
│   ├── pipeline.py           # Combined pipeline
│   └── visualization.py      # Drawing bboxes, masks, captions
├── app.py                    # Gradio demo
├── examples/                 # Sample images
├── .gitignore
└── LICENSE
```

## Screenshots

<!-- Add screenshots here -->
_Screenshots will be added after first successful run._

## Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | Microsoft | Captioning + Object Detection |
| [SAM 2](https://github.com/facebookresearch/segment-anything-2) | Meta | Segmentation Masks |

## License

MIT — see [LICENSE](LICENSE).

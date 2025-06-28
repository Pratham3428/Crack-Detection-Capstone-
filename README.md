---
title: YOLOv8 Demo
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: gradio  # streamlit
sdk_version: "4.44.1"  # Gradio/Streamlit 
app_file: app.py  # Gradio/Streamlit
pinned: false
---

# YOLOv8 Road-Crack & General Object Detection Demo

This repository wraps the **Ultralytics YOLOv8** detector inside a user-friendly Gradio web interface and command line utilities.

* 🛣 **Road-Cracks Detection** – Identify 4 kinds of pavement damage (alligator, lateral, longitudinal cracks and potholes) using a custom model trained on the `roadCracksDataset`.
* 📦 **General Object Detection** – Fall-back to the official pre-trained YOLOv8 models for everyday objects.
* 🖼️/🎥/📷 Works with images, videos or a live webcam feed.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Inference](#inference)
6. [How It Works](#how-it-works)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

---

## Quick Start

```bash
# 1. Clone & enter the repo
git clone https://github.com/<your-username>/yolov12-road-cracks.git
cd yolov12-road-cracks

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
# Ultralytics is not pinned in the requirements – install the latest stable:
pip install ultralytics

# 4. Launch the web demo
python app.py           # Opens http://127.0.0.1:7860 in your browser
```

> You can also try the hosted demo on **🤗 Hugging Face Spaces** (when published) – the YAML block at the top of this file is read by Spaces to configure the build.

---

## Project Structure

```text
├── app.py                 # Gradio UI with two tabs (Road Cracks & General Detection)
├── test_road_cracks.py    # Stand-alone CLI tester for the road-crack model
├── roadCracksDataset/     # Training/validation/test data (images & YOLO labels)
│   ├── train/{images,labels}
│   ├── valid/{images,labels}
│   └── test/{images,labels}
├── requirements.txt       # Third-party Python dependencies
└── README.md              # ← you are here
```

---

## Dataset

`roadCracksDataset` is organised exactly the way the **Ultralytics** trainer expects:

* **Splits** – `train`, `valid`, `test`
* Within each split:
  * `images/` – JPEG/PNG frames of asphalt
  * `labels/` – YOLO-format text files: `class_id x_center y_center width height`

The four class IDs map to:

| ID | Class Name          |
|----|---------------------|
| 0  | alligator_crack     |
| 1  | lateral_crack       |
| 2  | longitudinal_crack  |
| 3  | pothole             |

Sample counts:
* Train: **81** images
* Val  : **80** images
* Test : **42** images

Trained on 3000+ dataset 

You may point to your own dataset by editing the `data.yaml` file or passing the path on the command line.

---

## Training

## The road-crack model shipped here was trained with:

```bash
yolo detect train \
  data=roadCracksDataset/data.yaml \
  model=yolov12n.pt \
  imgsz=640 \
  epochs=100 \
  batch=16 \
  project=road_cracks_results
```

Tweak the hyper-parameters as needed (larger backbone, more epochs, etc.). The final weights are saved to:
`road_cracks_results/road_cracks_model/weights/best.pt`

---

## Inference

### 1. Command Line

```bash
python test_road_cracks.py \
  --weights road_cracks_results/road_cracks_model/weights/best.pt \
  --source path/to/image_or_video/or/folder \
  --conf 0.25 --iou 0.45 --device 0
```

`test_road_cracks.py` will automatically create a `runs/test/…` folder with the annotated results.

### 2. Web Interface

Launch the UI and select one of the two tabs:

* **Road Cracks Detection** – Provide a custom weights path (defaults to the model above) and optionally tune the confidence slider. The panel on the right lists each detection (`class: score`).
* **General Detection** – Choose one of the light-weight YOLOv8 checkpoints for everyday objects.

You can switch between *Image*, *Video* and *Webcam* inputs. The annotated media or live feed is returned instantly in the browser.

---

## How It Works

1. **Model Loading** – `app.py` uses the Ultralytics `YOLO` class. If the provided weights file is missing it gracefully falls back to a small `yolov12n.pt` model or, in the general tab, the tiny `yolov8n.pt`.
2. **Inference** – The helper functions `yolov12_inference()` and `road_cracks_inference()` unify the logic for images, recorded videos and webcam streams.
3. **Annotation** – After each forward pass, `results[0].plot()` overlays bounding boxes. For road-cracks, per-box info (class name + confidence) is collected and printed in the UI.
4. **Gradio UI** – Two top-level functions build separate Blocks layouts, while a small wrapper puts them in tabs so you can toggle modes without reloading the page.
5. **CLI** – `test_road_cracks.py` uses the same Ultralytics API but adds batch processing over an input directory and stores outputs under `runs/test` for easy inspection.

---

## Troubleshooting

* **CUDA not available** – Either install the CUDA-enabled PyTorch wheel or add `--device cpu` to run on the CPU.
* **Model load error** – Check the path you entered in the textbox or re-download the weights.
* **Low FPS on webcam** – Reduce `image_size` (e.g. 320) or use the lighter `n/s` model variants.

---

## Contributing

Pull requests are welcome! If you have:
* 🎯 Better training hyper-parameters
* 🐛 Bug fixes or UI improvements
* 📄 New datasets or tasks

please open an issue first to discuss the change.

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.




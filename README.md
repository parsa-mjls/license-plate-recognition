# License Plate Detection & OCR (Farsi/Persian)

An end-to-end computer vision pipeline that detects vehicles/motorcycles and their license plates in images and video, then reads the plate text using a custom-trained OCR model for Persian characters.

This repository contains an **inference-only demo** of a larger production system. It showcases the detection + OCR pipeline design, not the training code or proprietary datasets used to build the underlying models.

<!-- 📸 PLACEHOLDER: add a hero image/GIF here, e.g. a detected car with its plate box + OCR label -->
<!-- ![demo](assets/demo_hero.gif) -->

---

## Overview

The pipeline runs in two stages per frame:

1. **Detection** — a YOLO (Ultralytics) model locates vehicles/motorcycles and crops the plate region.
2. **Recognition (OCR)** — the cropped plate is passed to a custom PaddleOCR recognition model trained on Persian license plates, which returns the plate text and a confidence score.

The result is rendered back onto the original image/video frame: a bounding box around the plate plus an arrow pointing to a readable text label (shaped and rendered correctly for right-to-left Persian script).

<!-- 📸 PLACEHOLDER: side-by-side "before / after" image -->
<!-- ![before-after](assets/before_after.jpg) -->

### Why a separate OCR process?

PaddleOCR/PaddlePaddle and PyTorch can conflict when imported in the same Python process on some environments. To avoid this, OCR runs in a fully isolated subprocess (`src/ocr_worker.py`), communicating with the main process over a simple line-based stdin/stdout protocol. This keeps the detector (torch) and the recognizer (paddle) completely decoupled, at the cost of a small IPC overhead per plate crop.

---

## Project structure

```
license-plate-recognition/
├── assets/
│   └── fonts/              # Persian-capable .ttf font (e.g. Vazir.ttf) — not included, see Setup
├── examples/
│   ├── images/             # sample input images go here
│   └── videos/             # sample input videos go here
├── models/
│   ├── detector/           # YOLO weights (best.pt) — not included, see Setup
│   └── ocr/                # PaddleOCR recognition model + dictionary — not included, see Setup
├── outputs/                 # demo results are written here
├── src/
│   ├── config.py           # all paths & tunable settings in one place
│   ├── utils.py             # small shared helpers (font resolution, etc.)
│   ├── detector.py          # YOLO loading + inference
│   ├── ocr_client.py        # subprocess client that talks to ocr_worker.py
│   ├── ocr_worker.py        # isolated PaddleOCR process (never imports torch)
│   ├── visualizer.py        # all drawing/annotation code (boxes, arrows, Persian text)
│   └── pipeline.py          # shared detect → crop → OCR → annotate pipeline
├── demo.py                  # CLI entry point (auto-detects image vs. video input)
├── requirements.txt
└── README.md
```

Detection, OCR, and drawing logic each live in their own module, and a single `pipeline.py` is shared by both image and video inputs — there is no duplicated "image version" and "video version" of the same logic.

---

## Requirements

- Python 3.9+
- A CUDA-capable GPU is strongly recommended (CPU inference works but is much slower for video).
- The exact package versions this project was developed and tested with are pinned in [`requirements.txt`](./requirements.txt).

Install:

```bash
pip install -r requirements.txt
```

> The `torch`/`torchvision` lines are pinned to a CUDA 11.8 build. If you're on CPU-only or a different CUDA version, install the matching `torch`/`torchvision` build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the rest of `requirements.txt`.

---

## Setup

The trained model weights are **not included** in this repository (they belong to the original production system). To run the demo yourself, provide your own:

1. **Detector weights** — place a YOLO `.pt` file at `models/detector/best.pt` (or update `DETECTOR_MODEL_PATH` in `src/config.py`). Any YOLO model that outputs a "vehicle"/"plate" class will work; adjust `ALLOWED_CLASSES` in `config.py` to match your model's class ids.
2. **OCR model** — place your PaddleOCR recognition model directory at `models/ocr/rec_model/` and its character dictionary at `models/ocr/dict.txt` (or update `OCR_MODEL_DIR` / `OCR_DICT_PATH` in `src/config.py`).
3. **Font** — download a Persian-capable TrueType font (e.g. [Vazir](https://github.com/rastikerdar/vazir-font)) and place it at `assets/fonts/Vazir.ttf`, or point `FONT_PATH` in `config.py` to any `.ttf` on your system. A few common Windows font fallbacks are already configured in `FALLBACK_FONTS`.

All of the above are configured in **one place**: [`src/config.py`](./src/config.py).

---

## Usage

Run on an image:

```bash
python demo.py --source examples/images/car.jpg
```

Run on a video:

```bash
python demo.py --source examples/videos/demo.mp4 --output outputs/result.mp4
```

`demo.py` automatically decides whether the input is an image or a video based on its file extension — no separate scripts or flags needed. If `--output` is omitted, the result is written to `outputs/<source_name>_result.<ext>`.

---

## Sample results

<!-- 📸 PLACEHOLDER: add 2-3 example input/output pairs below -->
<!--
| Input | Output |
|---|---|
| ![input1](examples/images/car1.jpg) | ![output1](outputs/car1_result.jpg) |
-->

<!-- 🎞️ PLACEHOLDER: add a short GIF or a link to a sample output video -->
<!-- ![video-demo](assets/video_demo.gif) -->

<!-- 🧾 PLACEHOLDER: add a short console log snippet showing detection + OCR output, e.g.:
[Detector] Loading model from: models/detector/best.pt
[OCR] Starting PaddleOCR worker process (this may take a few seconds)...
[OCR] Worker ready.
[Demo] Mode: image | Source: examples/images/car1.jpg | Output: outputs/car1_result.jpg
[Demo] Done. Result saved to: outputs/car1_result.jpg
-->

---

## Design notes

- **Single shared pipeline**: `pipeline.py` implements one `process_frame()` method used by both `process_image()` and `process_video()`, eliminating the ~80% code duplication that existed between separate image/video scripts.
- **Process isolation for OCR**: keeps PaddlePaddle and PyTorch from ever coexisting in the same interpreter, sidestepping a real-world library conflict rather than papering over it.
- **Centralized configuration**: every tunable value (paths, confidence threshold, allowed classes, font) lives in `config.py`, so running the demo on a new machine/model only requires editing one file.
- **No training code or proprietary data**: this repo intentionally contains only the inference/demo path. The detector and OCR models referenced here were trained separately as part of a private production system.

---

## Limitations

- Plate reading accuracy depends entirely on the quality of the underlying detector/OCR models, which are not distributed with this repo.
- The subprocess-based OCR client processes one crop at a time; for high-throughput/batch scenarios a persistent batched OCR service would be a better fit.
- Video processing is currently single-threaded (read → detect → OCR → write); frame-level parallelism could improve real-time throughput on capable hardware.

---

## License

<!-- ✏️ PLACEHOLDER: add your preferred license, e.g. MIT, and a LICENSE file -->

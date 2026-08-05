# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for the license plate detection + OCR pipeline.
All paths are relative to the project root, so the project can be cloned
and run on any machine without editing hardcoded absolute paths.
"""

from pathlib import Path

# Project root = the folder that contains this "src" directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# YOLO plate/vehicle detector weights.
DETECTOR_MODEL_PATH = BASE_DIR / "models" / "detector" / "best.pt"

# PaddleOCR recognition model + character dictionary.
OCR_MODEL_DIR = BASE_DIR / "models" / "ocr" / "rec_model"
OCR_DICT_PATH = BASE_DIR / "models" / "ocr" / "dict.txt"

# Worker script used to run PaddleOCR in an isolated subprocess
# (kept separate from torch/ultralytics to avoid library conflicts).
OCR_WORKER_SCRIPT = BASE_DIR / "src" / "ocr_worker.py"

# ---------------------------------------------------------------------------
# Font used to draw Persian/Arabic plate text on frames
# ---------------------------------------------------------------------------
# Adding a custom font is OPTIONAL. By default FONT_PATH points to a file
# that doesn't exist unless you add one yourself, so resolve_font_path()
# will automatically fall through to FALLBACK_FONTS -- the standard Windows
# system fonts, which already render Persian/Arabic correctly. Only set
# FONT_PATH if you specifically want a different font (e.g. Vazir.ttf).
FONT_PATH = BASE_DIR / "assets" / "fonts" / "Vazir.ttf"
FALLBACK_FONTS = [
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_SIZE = 22

# ---------------------------------------------------------------------------
# Detector settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
ALLOWED_CLASSES = [0, 1]  # 0: car, 1: motorcycle (depends on your trained model)

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cuda"  # "cuda" or "cpu" -- ultralytics will fall back to CPU automatically if no GPU is found

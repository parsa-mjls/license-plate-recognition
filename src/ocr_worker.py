# -*- coding: utf-8 -*-
"""
ocr_worker.py
=============
Fully isolated worker process that ONLY imports paddle / paddleocr.
torch and ultralytics must NEVER be imported in this file/process.

Line-based protocol over stdin/stdout:
- Each input line: path to an image file (a cropped plate).
- Each output line: "recognized_text<TAB>confidence"
- The line "EXIT" shuts the process down.
- Once the model is loaded, the line "READY" is printed before anything else.

This script is not meant to be run manually -- it is started as a
subprocess by ocr_client.py and is shared by both the image and the
video demos.
"""

import logging
import sys
from pathlib import Path

# Force UTF-8 on this process's own stdin/stdout, regardless of the OS
# default codepage (e.g. cp1252 on Windows). This is a safety net in
# addition to ocr_client.py setting PYTHONIOENCODING=utf-8 when it launches
# this script -- without one of the two, writing Persian/Arabic plate text
# here can crash with a UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

# Make sure "config" (in the same folder) can be imported regardless of
# the current working directory this process is started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# PaddleOCR/PaddlePaddle log warnings and internal errors via the standard
# `logging` module, which by default goes to stderr. That's fine on its own
# (stderr is no longer merged into stdout -- see ocr_client.py), but we also
# raise the log level here so routine noise doesn't clutter the console.
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)


def _clean(text):
    """Make sure a value is a single clean line safe for the stdout protocol."""
    return str(text).replace("\n", " ").replace("\r", " ").replace("\t", " ")


def main():
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        use_angle_cls=False,
        rec_model_dir=str(config.OCR_MODEL_DIR),
        rec_char_dict_path=str(config.OCR_DICT_PATH),
        det=False,
        show_log=False,
    )

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        if line == "EXIT":
            break

        image_path = line

        # Everything below is wrapped in one try/except so that, no matter
        # what goes wrong (bad image, model error, unexpected result shape),
        # exactly ONE line is written back to stdout. This keeps the
        # request/response protocol in lock-step even on failure.
        try:
            text, conf = "", 0.0
            result = ocr.ocr(image_path, det=False, rec=True, cls=False)
            if result and len(result) > 0 and result[0]:
                entry = result[0][0]
                text = _clean(entry[0])
                conf = float(entry[1])
        except Exception as e:
            text = "ERROR:" + _clean(e)
            conf = 0.0

        sys.stdout.write(f"{text}\t{conf}\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

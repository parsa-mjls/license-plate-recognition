# -*- coding: utf-8 -*-
"""
demo.py
=======
Single entry point for the license plate detection + OCR demo.
Automatically detects whether --source is an image or a video and
routes it to the right pipeline method.

Examples
--------
python demo.py --source examples/images/car.jpg
python demo.py --source examples/videos/demo.mp4 --output outputs/result.mp4
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Allow running "python demo.py" from the project root while pipeline
# code lives under src/.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import PlateRecognitionPipeline  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description="License Plate Detection + OCR demo")
    parser.add_argument("--source", required=True, help="Path to an input image or video file")
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the result (default: outputs/<source_name>_result.<ext>)",
    )
    return parser.parse_args()


def resolve_mode(source_path: Path) -> str:
    ext = source_path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        f"Supported images: {sorted(IMAGE_EXTS)}, videos: {sorted(VIDEO_EXTS)}"
    )


def resolve_output_path(source_path: Path, mode: str, output_arg: Optional[str]) -> Path:
    if output_arg:
        return Path(output_arg)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    suffix = source_path.suffix if mode == "image" else ".mp4"
    return out_dir / f"{source_path.stem}_result{suffix}"


def main():
    args = parse_args()
    source_path = Path(args.source)

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    mode = resolve_mode(source_path)
    output_path = resolve_output_path(source_path, mode, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Demo] Mode: {mode} | Source: {source_path} | Output: {output_path}")

    pipeline = PlateRecognitionPipeline()
    try:
        if mode == "image":
            pipeline.process_image(source_path, output_path)
        else:
            pipeline.process_video(source_path, output_path)
    finally:
        pipeline.close()

    print(f"[Demo] Done. Result saved to: {output_path}")


if __name__ == "__main__":
    main()

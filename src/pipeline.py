# -*- coding: utf-8 -*-
"""
pipeline.py
===========
The single shared pipeline used by both the image and video demo paths:
detect plates -> crop -> OCR -> annotate frame.

This replaces what used to be two ~80%-duplicated scripts
(LP_Det_and_OCR_on_Image.py / LP_Det_and_OCR_on_Video.py).
"""

import cv2
from PIL import ImageFont

import config
from utils import resolve_font_path
from detector import load_detector, get_plate_boxes
from ocr_client import PaddleOCRClient
from visualizer import annotate_plate_side_label, annotate_plate_top_label


class PlateRecognitionPipeline:
    def __init__(self):
        font_path = resolve_font_path(config.FONT_PATH, config.FALLBACK_FONTS)
        self.font = ImageFont.truetype(font_path, config.FONT_SIZE)
        print(f"[Pipeline] Using font: {font_path}")

        self.detector = load_detector(config.DETECTOR_MODEL_PATH)

        self.ocr_client = PaddleOCRClient(config.OCR_WORKER_SCRIPT)
        self.ocr_client.start()

    def close(self):
        self.ocr_client.close()

    def _read_plate(self, plate_crop):
        try:
            text, conf = self.ocr_client.ocr(plate_crop)
        except Exception as e:
            print("[OCR] Error:", e)
            return "UNKNOWN", 0.0

        if text.startswith("ERROR:"):
            print("[OCR] Error:", text)
            return "UNKNOWN", 0.0

        return text, conf

    def process_frame(self, frame, label_style="top"):
        """
        Run detection + OCR + annotation on a single frame (in place, and
        also returned).
        label_style: "top" (image demo) or "side" (video demo).
        """
        frame_height, frame_width = frame.shape[:2]
        boxes = get_plate_boxes(frame, self.detector, config.CONFIDENCE_THRESHOLD, config.ALLOWED_CLASSES)

        for x1, y1, x2, y2, score, cls_id in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            plate_text, plate_conf = self._read_plate(plate_crop)

            if label_style == "side":
                frame = annotate_plate_side_label(frame, x1, y1, x2, y2, plate_text, self.font, frame_width)
            else:
                frame = annotate_plate_top_label(frame, x1, y1, x2, y2, plate_text, self.font, frame_width)

        return frame

    def process_image(self, image_path, output_path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not open image: {image_path}")

        frame = self.process_frame(frame, label_style="top")
        cv2.imwrite(str(output_path), frame)
        return output_path

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame, label_style="side")
                out.write(frame)
        finally:
            cap.release()
            out.release()

        return output_path

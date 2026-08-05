# -*- coding: utf-8 -*-
"""
detector.py
===========
Thin wrapper around the YOLO (Ultralytics) plate/vehicle detector.
"""

from ultralytics import YOLO


def load_detector(model_path):
    """Load the YOLO detector from the given weights path."""
    print(f"[Detector] Loading model from: {model_path}")
    return YOLO(str(model_path))


def get_plate_boxes(frame, detector_model, threshold, allowed_classes):
    """
    Run the detector on a single frame and return a list of boxes:
    [x1, y1, x2, y2, confidence, class_id]
    filtered by confidence threshold and allowed class ids.
    """
    detected_boxes = []

    results = detector_model(frame, verbose=False)[0]
    for r in results.boxes:
        conf = float(r.conf[0])
        cls_id = int(r.cls[0])

        if conf >= threshold and cls_id in allowed_classes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            detected_boxes.append([x1, y1, x2, y2, conf, cls_id])

    return detected_boxes

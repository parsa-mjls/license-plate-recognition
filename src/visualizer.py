# -*- coding: utf-8 -*-
"""
visualizer.py
=============
All drawing/annotation code (boxes, arrows, Persian text rendering) lives
here, kept separate from the detection/OCR logic in pipeline.py.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display


def _shape_persian_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def draw_persian_text(img_bgr, text, position, font, text_color=(0, 0, 0)):
    """Draw shaped Persian/Arabic text at a fixed top-left position."""
    bidi_text = _shape_persian_text(text)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, bidi_text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_persian_text_centered(img_bgr, text, box_coords, font, text_color=(0, 0, 0)):
    """Draw shaped Persian/Arabic text centered inside box_coords = (x1, y1, x2, y2)."""
    bidi_text = _shape_persian_text(text)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    x1, y1, x2, y2 = box_coords
    box_w, box_h = x2 - x1, y2 - y1

    try:
        bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        offset_y = bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions without textbbox
        text_w, text_h = draw.textsize(bidi_text, font=font)
        offset_y = 0

    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h - text_h) // 2 - offset_y

    draw.text((text_x, text_y), bidi_text, font=font, fill=text_color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def annotate_plate_side_label(frame, x1, y1, x2, y2, text, font, frame_width):
    """
    Video-style annotation: red box around the plate + an arrow pointing to
    a fixed label panel on the right edge of the frame.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    center_y = (y1 + y2) // 2
    center_x = x2

    text_box_x1 = frame_width - 280
    text_box_x2 = frame_width - 20
    text_box_y1 = center_y - 25
    text_box_y2 = center_y + 25

    cv2.arrowedLine(frame, (center_x, center_y), (text_box_x1, center_y), (0, 255, 255), 2, tipLength=0.1)
    cv2.rectangle(frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (255, 255, 255), -1)
    cv2.rectangle(frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (0, 0, 0), 2)

    text_position = (text_box_x1 + 15, text_box_y1 + 5)
    return draw_persian_text(frame, text, text_position, font)


def annotate_plate_top_label(frame, x1, y1, x2, y2, text, font, frame_width):
    """
    Image-style annotation: red box around the plate + an arrow pointing to
    a label panel centered directly above (or below, if there's no room)
    the plate.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    box_width, box_height, margin = 260, 50, 35
    plate_center_x = (x1 + x2) // 2

    text_box_x1 = plate_center_x - (box_width // 2)
    text_box_x2 = text_box_x1 + box_width
    text_box_y2 = y1 - margin
    text_box_y1 = text_box_y2 - box_height

    if text_box_y1 < 0:
        text_box_y1 = y2 + margin
        text_box_y2 = text_box_y1 + box_height

    if text_box_x1 < 0:
        text_box_x1 = 10
        text_box_x2 = text_box_x1 + box_width
    elif text_box_x2 > frame_width:
        text_box_x2 = frame_width - 10
        text_box_x1 = text_box_x2 - box_width

    if text_box_y2 < y1:
        cv2.arrowedLine(frame, (plate_center_x, y1), (plate_center_x, text_box_y2), (0, 0, 255), 4, tipLength=0.2)
    else:
        cv2.arrowedLine(frame, (plate_center_x, y2), (plate_center_x, text_box_y1), (0, 0, 255), 3, tipLength=0.2)

    cv2.rectangle(frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (255, 255, 255), -1)
    cv2.rectangle(frame, (text_box_x1, text_box_y1), (text_box_x2, text_box_y2), (0, 0, 0), 2)

    box_coords = (text_box_x1, text_box_y1, text_box_x2, text_box_y2)
    return draw_persian_text_centered(frame, text, box_coords, font)

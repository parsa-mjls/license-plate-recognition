# -*- coding: utf-8 -*-
"""
utils.py
========
Small shared helper functions used across the pipeline.
"""

import os


def resolve_font_path(font_path, fallback_fonts):
    """
    Return the first existing font path from [font_path] + fallback_fonts.
    Raises a clear error if none of the candidates exist, instead of failing
    later with a cryptic PIL OSError.
    """
    candidates = [str(font_path)] + [str(f) for f in fallback_fonts]
    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No valid font file was found. Please set FONT_PATH in config.py to "
        "a real .ttf file that supports Persian/Arabic script (e.g. Vazir.ttf), "
        "or add a valid system font to FALLBACK_FONTS."
    )

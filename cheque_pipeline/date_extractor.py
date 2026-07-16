"""
date_extractor.py — finds and reads the handwritten date field.

Geometry (finding WHERE the date is) is unchanged from
test_date_extraction.py: fuzzy-match a TARIKH/DATE label among PaddleOCR's
already-detected lines, then crop a search band relative to that anchor
(never a fixed page coordinate, so it moves with wherever the label
actually landed).

What's different from test_date_extraction.py: the crop is read with
TrOCR, not re-run through PaddleOCR's recognizer — per your own testing,
PaddleOCR detects the date box fine but cannot read handwritten digits,
while TrOCR reads that same crop correctly.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import cv2
import numpy as np

from . import config, trocr_engine

logger = logging.getLogger(__name__)

DATE_PATTERN = re.compile(r"\d{1,2}[/\-]?\d{1,2}[/\-]?\d{2,4}")


def _find_anchor(lines: list[dict], aliases: list[str]) -> dict | None:
    """Return the highest-confidence OCR line matching a date-label alias."""
    best = None
    for line in lines:
        text_upper = line["text"].upper().replace(" ", "")
        for alias in aliases:
            if alias.upper() in text_upper:
                if best is None or line["confidence"] > best["confidence"]:
                    best = line
    return best


def _search_band(
    anchor_box: list[list[float]],
    img_shape: tuple[int, int],
    direction: str,
    width: int,
    height_pad: int,
) -> tuple[int, int, int, int]:
    """Crop rectangle (x0,y0,x1,y1) relative to the anchor box, clamped to
    the image bounds — moves with the anchor, not a fixed page offset."""
    h, w = img_shape[:2]
    xs = [p[0] for p in anchor_box]
    ys = [p[1] for p in anchor_box]

    if direction == "right":
        x0 = max(xs)
        x1 = x0 + width
        y0 = min(ys) - height_pad
        y1 = max(ys) + height_pad
    elif direction == "below":
        x0 = min(xs)
        x1 = max(xs)
        y0 = max(ys)
        y1 = y0 + width
    else:
        raise ValueError(f"Unknown direction: {direction}")

    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return x0, y0, x1, y1


def extract_date(img_bgr: np.ndarray, lines: list[dict[str, Any]]) -> dict[str, Any]:
    """Full pipeline: find TARIKH/DATE anchor among PaddleOCR's line
    results -> crop+upscale the band to its right -> read it with TrOCR.

    `lines` must come from paddle_ocr._run_ocr(img_bgr, ocr) run on the
    SAME img_bgr passed here, so box coordinates line up.

    Returns:
        {
          "status": "ok" | "needs_review" | "anchor_not_found" | "empty_crop",
          "date_text": str | None,       # digits/separators only
          "raw_text": str | None,        # TrOCR's unfiltered output
          "confidence": float,           # TrOCR's own confidence
          "anchor_text": str | None,
          "crop": np.ndarray | None,     # BGR, for debug/QA output
        }
    """
    anchor = _find_anchor(lines, config.DATE_LABEL_ALIASES)
    if anchor is None:
        return {"status": "anchor_not_found", "date_text": None, "raw_text": None,
                "confidence": 0.0, "anchor_text": None, "crop": None}

    x0, y0, x1, y1 = _search_band(
        anchor["box"], img_bgr.shape, config.DATE_SEARCH_DIRECTION,
        config.DATE_SEARCH_WIDTH, config.DATE_HEIGHT_PAD,
    )
    if x1 <= x0 or y1 <= y0:
        return {"status": "invalid_band", "date_text": None, "raw_text": None,
                "confidence": 0.0, "anchor_text": anchor["text"], "crop": None}

    crop = img_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return {"status": "empty_crop", "date_text": None, "raw_text": None,
                "confidence": 0.0, "anchor_text": anchor["text"], "crop": None}

    crop_upscaled = cv2.resize(
        crop, None,
        fx=config.DATE_CROP_UPSCALE_FACTOR, fy=config.DATE_CROP_UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    raw_text, confidence = trocr_engine.recognize(crop_upscaled, is_bgr=True)
    digits_only = re.sub(r"[^0-9/\-]", "", raw_text)
    status = "ok" if DATE_PATTERN.fullmatch(digits_only) else "needs_review"

    return {
        "status": status,
        "date_text": digits_only,
        "raw_text": raw_text,
        "confidence": confidence,
        "anchor_text": anchor["text"],
        "crop": crop_upscaled,
    }

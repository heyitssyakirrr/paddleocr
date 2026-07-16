"""
signature_extractor.py — detects whether a signature is present, and crops it.

PRIMARY method: direct ink-blob analysis inside config.SIGNATURE_REGION.
Does not depend on PP-Structure at all. Steps:
    1. Otsu-threshold the region to get a dark-ink mask (adapts per-image,
       robust to scan/photo exposure differences).
    2. Subtract long straight printed lines (box borders, underlines) via
       directional morphological opening — a signature's strokes are curvy
       and short relative to a printed line, so this separates them cleanly.
    3. Connected-component analysis on what's left; drop components too
       short to plausibly be a signature stroke (this is what filters out
       printed caption text like "no signature below this line" without
       needing OCR to read and recognize it as text).
    4. What survives is treated as signature ink. Its total area vs the
       region's area is the existence signal; its union bounding box is the
       crop.

SECONDARY signal: PP-Structure layout detection (layout_engine.py). Used
only to refine the crop box when a figure/seal box happens to agree with
where the ink-blob method already found something — never to override an
ink-blob "not found" into "found" (see config.py's comment on why: layout
models have no real signature class and are out-of-domain on cheques).

Every result reports `detection_method` so you can audit which signal
decided the case (`ink_blob`, `ink_blob+layout`, or `none`).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config, layout_engine

logger = logging.getLogger(__name__)


def _region_to_pixels(img_shape: tuple[int, int], region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    x0, y0, x1, y1 = region
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def _box_center_inside(box: tuple[float, float, float, float], region_px: tuple[int, int, int, int]) -> bool:
    bx0, by0, bx1, by1 = box
    rx0, ry0, rx1, ry1 = region_px
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _find_ink_blobs(region_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float]:
    """Return (surviving component bboxes as (x,y,w,h), ink_ratio) after
    removing printed straight lines and short/small (likely-text) blobs.
    All coordinates are relative to region_bgr, not the full image.
    """
    if region_bgr.size == 0:
        return [], 0.0

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape

    # Printed borders/underlines are long and straight; a signature's
    # strokes are not. Detecting each orientation separately (rather than
    # one all-directions kernel) keeps this precise regardless of the
    # signature's own slant.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 15), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 15)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel),
    )
    ink_only = cv2.bitwise_and(mask, cv2.bitwise_not(lines))

    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(ink_only, connectivity=8)
    min_area = config.SIGNATURE_MIN_COMPONENT_AREA_FRAC * (h * w)
    min_height = config.SIGNATURE_MIN_COMPONENT_HEIGHT_FRAC * h

    kept = []
    total_area = 0
    for i in range(1, n):  # label 0 is background
        x, y, cw, ch, area = stats[i]
        if area >= min_area and ch >= min_height:
            kept.append((int(x), int(y), int(cw), int(ch)))
            total_area += int(area)

    ink_ratio = total_area / (h * w) if h * w else 0.0
    return kept, ink_ratio


def _union_bbox(boxes: list[tuple[int, int, int, int]], pad: int, bounds: tuple[int, int]) -> tuple[int, int, int, int]:
    """Bounding box covering all given (x,y,w,h) boxes, padded and clamped
    to (width, height) bounds."""
    w, h = bounds
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(w, max(b[0] + b[2] for b in boxes) + pad)
    y1 = min(h, max(b[1] + b[3] for b in boxes) + pad)
    return x0, y0, x1, y1


def detect_signature(img_bgr: np.ndarray) -> dict[str, Any]:
    """Returns:
        {
          "signature_exists": bool,
          "detection_method": "ink_blob" | "ink_blob+layout" | "none",
          "score": float | None,       # layout confidence, only if layout corroborated
          "ink_ratio": float | None,   # surviving-ink-area / region-area
          "box": (x0,y0,x1,y1) | None, # absolute pixel coords of the crop actually used
          "crop": np.ndarray | None,   # BGR crop, for saving to disk
        }
    """
    region_px = _region_to_pixels(img_bgr.shape, config.SIGNATURE_REGION)
    rx0, ry0, rx1, ry1 = region_px
    region_crop = img_bgr[ry0:ry1, rx0:rx1]

    blobs, ink_ratio = _find_ink_blobs(region_crop)
    exists = bool(blobs) and ink_ratio >= config.SIGNATURE_INK_RATIO_THRESHOLD

    if not exists:
        return {
            "signature_exists": False,
            "detection_method": "none",
            "score": None,
            "ink_ratio": ink_ratio,
            "box": None,
            "crop": None,
        }

    rh, rw = region_crop.shape[:2]
    lx0, ly0, lx1, ly1 = _union_bbox(blobs, config.SIGNATURE_CROP_PADDING_PX, (rw, rh))
    abs_box = (rx0 + lx0, ry0 + ly0, rx0 + lx1, ry0 + ly1)
    method = "ink_blob"
    score = None

    # Secondary corroboration only — never used to flip a "not found" result,
    # only to prefer a layout-model box if it independently agrees.
    layout_regions = layout_engine.detect_regions(img_bgr)
    layout_hits = [
        r for r in layout_regions
        if r["label"].lower() in config.SIGNATURE_LABELS
        and r["score"] >= config.SIGNATURE_LAYOUT_SCORE_THRESHOLD
        and _box_center_inside(r["box"], region_px)
        and _box_center_inside(r["box"], abs_box)  # must also agree with where the ink actually is
    ]
    if layout_hits:
        method = "ink_blob+layout"
        score = layout_hits[0]["score"]  # score-sorted by layout_engine.detect_regions()

    crop = img_bgr[abs_box[1]:abs_box[3], abs_box[0]:abs_box[2]]
    return {
        "signature_exists": True,
        "detection_method": method,
        "score": score,
        "ink_ratio": ink_ratio,
        "box": abs_box,
        "crop": crop if crop.size else None,
    }

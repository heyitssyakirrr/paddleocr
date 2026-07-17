"""
signature_extractor.py — detects whether a signature is present, and crops it.

REGION: anchored to the MICR line (the printed account/cheque-number line
at the bottom of every cheque) rather than a fixed fraction of the page —
see config.py's comment block above SIGNATURE_BAND_HEIGHT_FRAC for why a
fixed fraction doesn't generalize across real scans.

PRIMARY detection: direct ink-blob analysis inside that band. Does not
depend on PP-Structure at all.
    1. Otsu-threshold the band to get a dark-ink mask (adapts per-image).
    2. Subtract long straight printed lines (box borders, underlines) via
       directional morphological opening.
    3. Connected-component analysis on what's left; drop components too
       short to plausibly be a signature stroke (filters printed caption
       text like "no signature below this line" without needing OCR to
       read it).
    4. What survives is signature ink. Its area vs. the band's area is the
       existence signal; its union bounding box is the crop.

SECONDARY signal: PP-Structure layout detection (layout_engine.py) — only
refines the crop box when a figure/seal box agrees with where ink-blob
already found something. Never overrides an ink-blob "not found".

IMPORTANT: the raw band crop (`region_crop`) is always returned, whether
or not a signature was found — batch_process.py saves it for every cheque
so a false negative can actually be inspected instead of debugged blind.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config, layout_engine

logger = logging.getLogger(__name__)


def _find_micr_anchor_y(lines: list[dict], img_h: int) -> int | None:
    """Return the top-y of the lowest detected OCR line (the MICR line,
    on virtually every real cheque) — or None if no lines were detected
    at all, in which case the caller falls back to a fixed region."""
    if not lines:
        return None
    bottom_line = max(lines, key=lambda ln: max(p[1] for p in ln["box"]))
    return int(min(p[1] for p in bottom_line["box"]))


def _compute_region(img_shape: tuple[int, int], lines: list[dict]) -> tuple[int, int, int, int]:
    h, w = img_shape[:2]
    micr_top_y = _find_micr_anchor_y(lines, h)

    if micr_top_y is None:
        logger.warning("No OCR lines detected at all — falling back to fixed signature region")
        x0f, y0f, x1f, y1f = config.SIGNATURE_REGION_FALLBACK
        return int(x0f * w), int(y0f * h), int(x1f * w), int(y1f * h)

    y1 = max(0, micr_top_y - int(config.SIGNATURE_BAND_BOTTOM_MARGIN_FRAC * h))
    y0 = max(0, y1 - int(config.SIGNATURE_BAND_HEIGHT_FRAC * h))
    left_margin, right_margin = config.SIGNATURE_BAND_X_MARGIN_FRAC
    x0 = int(left_margin * w)
    x1 = w - int(right_margin * w)
    return x0, y0, x1, y1


def _box_center_inside(box: tuple[float, float, float, float], region_px: tuple[int, int, int, int]) -> bool:
    bx0, by0, bx1, by1 = box
    rx0, ry0, rx1, ry1 = region_px
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    return rx0 <= cx <= rx1 and ry0 <= cy <= ry1


def _find_ink_blobs(region_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], float, dict[str, np.ndarray]]:
    """Return (surviving component bboxes as (x,y,w,h), ink_ratio, debug_masks)
    after removing printed straight lines and short/small (likely-text) blobs.
    All coordinates are relative to region_bgr, not the full image.

    debug_masks holds the intermediate steps, always populated (whether or
    not anything survives), so a false negative can actually be inspected:
        "otsu_mask"           — ink vs background, before any filtering
        "after_line_removal"  — after printed straight lines subtracted
        "kept_mask"           — after area/height filtering (final ink used for ink_ratio)
    """
    if region_bgr.size == 0:
        return [], 0.0, {}

    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = mask.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 15), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 15)))
    lines_mask = cv2.bitwise_or(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_kernel),
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_kernel),
    )
    ink_only = cv2.bitwise_and(mask, cv2.bitwise_not(lines_mask))

    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(ink_only, connectivity=8)
    min_area = config.SIGNATURE_MIN_COMPONENT_AREA_FRAC * (h * w)
    min_height = config.SIGNATURE_MIN_COMPONENT_HEIGHT_FRAC * h

    kept_mask = np.zeros_like(ink_only)
    kept = []
    total_area = 0
    for i in range(1, n):  # label 0 is background
        x, y, cw, ch, area = stats[i]
        if area >= min_area and ch >= min_height:
            kept.append((int(x), int(y), int(cw), int(ch)))
            total_area += int(area)
            kept_mask[labels == i] = 255

    ink_ratio = total_area / (h * w) if h * w else 0.0
    debug_masks = {
        "otsu_mask": mask,
        "after_line_removal": ink_only,
        "kept_mask": kept_mask,
    }
    return kept, ink_ratio, debug_masks


def _union_bbox(boxes: list[tuple[int, int, int, int]], pad: int, bounds: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = bounds
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(w, max(b[0] + b[2] for b in boxes) + pad)
    y1 = min(h, max(b[1] + b[3] for b in boxes) + pad)
    return x0, y0, x1, y1


def detect_signature(img_bgr: np.ndarray, lines: list[dict[str, Any]]) -> dict[str, Any]:
    """`lines` must come from paddle_ocr._run_ocr(img, ocr) on this SAME
    img_bgr (same convention date_extractor.py already uses), so the MICR
    anchor's coordinates line up.

    Returns:
        {
          "signature_exists": bool,
          "detection_method": "ink_blob" | "ink_blob+layout" | "none",
          "score": float | None,
          "ink_ratio": float,
          "box": (x0,y0,x1,y1) | None,   # signature-only crop box, if found
          "crop": np.ndarray | None,      # tight signature crop, if found
          "region_box": (x0,y0,x1,y1),    # the full searched band — ALWAYS present
          "region_crop": np.ndarray,       # the full searched band image — ALWAYS present, for debugging
        }
    """
    region_px = _compute_region(img_bgr.shape, lines)
    rx0, ry0, rx1, ry1 = region_px
    region_crop = img_bgr[ry0:ry1, rx0:rx1]

    blobs, ink_ratio, debug_masks = _find_ink_blobs(region_crop)
    exists = bool(blobs) and ink_ratio >= config.SIGNATURE_INK_RATIO_THRESHOLD

    base_result = {
        "ink_ratio": ink_ratio,
        "region_box": region_px,
        "region_crop": region_crop if region_crop.size else None,
        "debug_masks": debug_masks,
    }

    if not exists:
        return {
            "signature_exists": False,
            "detection_method": "none",
            "score": None,
            "box": None,
            "crop": None,
            **base_result,
        }

    rh, rw = region_crop.shape[:2]
    lx0, ly0, lx1, ly1 = _union_bbox(blobs, config.SIGNATURE_CROP_PADDING_PX, (rw, rh))
    abs_box = (rx0 + lx0, ry0 + ly0, rx0 + lx1, ry0 + ly1)
    method = "ink_blob"
    score = None

    layout_regions = layout_engine.detect_regions(img_bgr)
    layout_hits = [
        r for r in layout_regions
        if r["label"].lower() in config.SIGNATURE_LABELS
        and r["score"] >= config.SIGNATURE_LAYOUT_SCORE_THRESHOLD
        and _box_center_inside(r["box"], region_px)
        and _box_center_inside(r["box"], abs_box)
    ]
    if layout_hits:
        method = "ink_blob+layout"
        score = layout_hits[0]["score"]

    crop = img_bgr[abs_box[1]:abs_box[3], abs_box[0]:abs_box[2]]
    return {
        "signature_exists": True,
        "detection_method": method,
        "score": score,
        "box": abs_box,
        "crop": crop if crop.size else None,
        **base_result,
    }

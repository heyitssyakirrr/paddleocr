"""
pipeline.py — orchestrates one cheque image through:
    1. paddle_ocr's existing loader + preprocessing (unchanged, reused as-is)
    2. paddle_ocr's existing OCR pass, to get line boxes for date anchoring
    3. date_extractor (PaddleOCR anchor -> TrOCR read)
    4. signature_extractor (PP-Structure layout -> ink-density backstop)

This is the only file that imports from paddle_ocr.py — every other module
in this package is engine-specific and stays decoupled from it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from paddle_ocr import _get_ocr, _load_image, _run_ocr, preprocess_image

from . import date_extractor, signature_extractor

logger = logging.getLogger(__name__)


def process_cheque(image_path: "str | Path") -> dict[str, Any]:
    """Run the full pipeline on one cheque image (PNG/JPG — for PDFs, render
    each page to an image first and call this per page).

    Returns a flat dict ready to become one CSV row, plus the extra
    diagnostic/crop fields batch_process.py uses for debug output:
        {
          "filename": str,
          "date": str,
          "signature_exists": bool,
          # diagnostics:
          "date_status": str, "date_raw_text": str, "date_confidence": float,
          "date_crop": np.ndarray | None,
          "signature_detection_method": str, "signature_score": float | None,
          "signature_ink_ratio": float | None, "signature_crop": np.ndarray | None,
        }
    """
    path = Path(image_path)
    img = _load_image(str(path))          # RGB, per paddle_ocr's own convention
    img = preprocess_image(img)            # denoise/contrast/deskew/upscale

    # img from paddle_ocr is RGB; this package's cv2-based helpers
    # (date_extractor's box math is orientation-agnostic, but
    # signature_extractor's ink threshold and any cv2.imwrite() downstream
    # expect BGR) — convert once, here, so every downstream module has one
    # consistent convention to rely on.
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ocr = _get_ocr()
    lines = _run_ocr(img, ocr)  # paddle_ocr's recognizer expects the RGB array it was built around

    date_result = date_extractor.extract_date(img_bgr, lines)
    signature_result = signature_extractor.detect_signature(img_bgr, lines)

    return {
        "filename": path.name,
        "date": date_result["date_text"] or "",
        "signature_exists": signature_result["signature_exists"],

        "date_status": date_result["status"],
        "date_raw_text": date_result["raw_text"],
        "date_confidence": date_result["confidence"],
        "date_anchor_text": date_result["anchor_text"],
        "date_crop": date_result["crop"],

        "signature_detection_method": signature_result["detection_method"],
        "signature_score": signature_result["score"],
        "signature_ink_ratio": signature_result["ink_ratio"],
        "signature_crop": signature_result["crop"],
        "signature_region_crop": signature_result["region_crop"],
        "signature_region_box": signature_result["region_box"],
        "signature_debug_masks": signature_result["debug_masks"],
    }

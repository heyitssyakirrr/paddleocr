"""
layout_engine.py — PP-Structure's layout-detection module only (not the
full PP-StructureV3 pipeline, which also bundles a second OCR pass, table
recognition, and doc-unwarping — none of which this addon needs, and all
of which would cost extra CPU time for nothing). Using just LayoutDetection
keeps this fast: it's a single small object-detection model.

Read config.py's comment block above SIGNATURE_LABELS before trusting this
module's output on cheques — it was not trained on cheque imagery and has
no dedicated "signature" class.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from . import config

logger = logging.getLogger(__name__)

_layout_instance = None
_layout_lock = threading.Lock()  # same non-thread-safety assumption as paddle_ocr._ocr_lock


def _get_layout_model():
    global _layout_instance
    if _layout_instance is not None:
        return _layout_instance

    with _layout_lock:
        if _layout_instance is not None:
            return _layout_instance

        try:
            from paddleocr import LayoutDetection
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR's LayoutDetection is unavailable — check your "
                "paddleocr/paddlepaddle install."
            ) from exc

        # Same air-gapped-first pattern as paddle_ocr._get_ocr(): use a local
        # copy if fetch_cheque_models.py has already cached one, else fall
        # back to an online download.
        model_dir = config.MODELS_DIR / config.LAYOUT_MODEL_NAME
        kwargs: dict[str, Any] = {"model_name": config.LAYOUT_MODEL_NAME}
        if model_dir.exists() and any(model_dir.iterdir()):
            logger.info("Using local layout model at %s", model_dir)
            kwargs["model_dir"] = str(model_dir)
        else:
            logger.warning(
                "Local layout model dir missing at %s — PaddleOCR will try "
                "to download %s from the internet. Run fetch_cheque_models.py "
                "once online first for air-gapped use.",
                model_dir, config.LAYOUT_MODEL_NAME,
            )

        logger.info("Loading layout detection model %s ...", config.LAYOUT_MODEL_NAME)
        _layout_instance = LayoutDetection(**kwargs)
        logger.info("Layout model loaded.")

    return _layout_instance


def detect_regions(image_bgr: np.ndarray) -> list[dict[str, Any]]:
    """Run layout detection on one preprocessed cheque image (BGR numpy
    array, as produced by paddle_ocr.preprocess_image).

    Returns a list of: {"label": str, "score": float, "box": (x0,y0,x1,y1)}
    with box in absolute pixel coordinates, sorted by score descending.
    """
    model = _get_layout_model()

    with _layout_lock:  # predict() shares the same not-documented-thread-safe caveat
        try:
            results = model.predict(image_bgr, batch_size=1)
        except Exception:
            logger.exception("Layout detection failed")
            return []

    regions: list[dict[str, Any]] = []
    for res in results:
        boxes = res.get("boxes") if isinstance(res, dict) else getattr(res, "boxes", None)
        if not boxes:
            continue
        for box in boxes:
            x0, y0, x1, y1 = box["coordinate"]
            regions.append({
                "label": str(box["label"]),
                "score": float(box["score"]),
                "box": (float(x0), float(y0), float(x1), float(y1)),
            })

    regions.sort(key=lambda r: r["score"], reverse=True)
    return regions

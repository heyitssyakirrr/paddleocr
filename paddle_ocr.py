from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import time
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import cv2

logger = logging.getLogger(__name__)

_ocr_instance = None


# ---------------------------------------------------------------------------
# Model version config
#
# Switching v5 <-> v6 is a one-line change here — no reinstall, no changes
# anywhere else in the codebase, since paddleocr>=3.7.0 ships both.
# Override via env var so it can be flipped per-deployment without a code
# change (e.g. OCR_MODEL_VERSION=v6 in your service's env config).
# ---------------------------------------------------------------------------
OCR_VERSION = os.environ.get("OCR_MODEL_VERSION", "v5")  # "v5" or "v6"

_MODEL_NAMES = {
    "v5": {"det": "PP-OCRv5_server_det", "rec": "PP-OCRv5_server_rec"},
    "v6": {"det": "PP-OCRv6_medium_det", "rec": "PP-OCRv6_medium_rec"},
}

# PaddleOCR pipeline objects are not documented as thread-safe, and app.py
# dispatches OCR calls to a thread-pool executor. Serialize predict() calls
# through one shared instance rather than risk concurrent-call corruption.
_ocr_lock = threading.Lock()


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError("PaddleOCR not installed.")

    if OCR_VERSION not in _MODEL_NAMES:
        raise RuntimeError(
            f"Unknown OCR_MODEL_VERSION={OCR_VERSION!r}; expected one of {list(_MODEL_NAMES)}"
        )
    names = _MODEL_NAMES[OCR_VERSION]

    # For air-gapped environments: use pre-downloaded models from local
    # directory. Passing *_model_dir bypasses PaddleOCR's hosting-platform
    # lookup entirely — it will not attempt any network call.
    model_base = Path(__file__).resolve().parent / "models"
    det_dir = model_base / names["det"]
    rec_dir = model_base / names["rec"]
    textline_dir = model_base / "PP-LCNet_x1_0_textline_ori"

    local_kwargs = {}
    if det_dir.exists() and rec_dir.exists():
        logger.info("Using local models from %s", model_base)
        local_kwargs = {
            "text_detection_model_dir": str(det_dir),
            "text_recognition_model_dir": str(rec_dir),
        }
        if textline_dir.exists():
            local_kwargs["textline_orientation_model_dir"] = str(textline_dir)
    else:
        logger.warning(
            "Local model dir missing for %s (%s / %s) — PaddleOCR will try "
            "to download from the internet, which will fail in an "
            "air-gapped deployment. Run fetch_models.py on an online "
            "machine first and copy the result into %s.",
            OCR_VERSION, names["det"], names["rec"], model_base,
        )

    logger.info("Loading PaddleOCR %s (%s / %s, CPU)...", OCR_VERSION, names["det"], names["rec"])
    _ocr_instance = PaddleOCR(
        text_detection_model_name=names["det"],
        text_recognition_model_name=names["rec"],
        use_doc_orientation_classify=False,   # we deskew ourselves in preprocess_image()
        use_doc_unwarping=False,              # ditto — avoid doing it twice
        use_textline_orientation=True,        # per-line rotation (stamps/seals can rotate independently)
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.8,
        text_rec_score_thresh=0.3,
        enable_mkldnn=False,
        **local_kwargs,
    )
    logger.info("Model loaded.")
    return _ocr_instance


# ---------------------------------------------------------------------------
# Preprocessing
#
# Deliberately format-agnostic: no cropping, no assumptions about where any
# field lives on the page. Every step here is something that helps *any*
# scanned cheque regardless of issuing bank — suppressing background
# patterns/watermarks, correcting whole-page skew, and normalizing
# resolution. Bank-specific layout logic does not belong in this file.
# ---------------------------------------------------------------------------

def _estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate whole-page rotation from the orientation of ink pixels.

    Works across formats because it only looks at where the ink is, not at
    any known layout. Deliberately conservative: only correct small skew
    (photographed/scanned docs), and bail out to 0 if the estimate looks
    unreliable rather than risk rotating a page that wasn't skewed.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return 0.0  # not enough ink to trust an angle estimate

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) > 15:
        return 0.0  # implausible for a photographed/scanned cheque; don't trust it
    return angle


def _deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    angle = _estimate_skew_angle(gray)
    if abs(angle) < 0.3:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed by %.2f degrees", angle)
    return rotated


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on the luminance channel only (LAB space), to lift faint
    print/ink against guilloché backgrounds and watermarks without
    distorting the color channels some banks use for their own security
    features (e.g. color-shifting ink, colored background panels)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_chan = clahe.apply(l_chan)
    lab = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _denoise(img: np.ndarray) -> np.ndarray:
    """Light denoise before contrast enhancement — CLAHE will otherwise
    amplify JPEG/sensor noise along with the ink."""
    return cv2.fastNlMeansDenoisingColored(
        img, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21,
    )


def _upscale_if_small(img: np.ndarray, min_width: int = 1600) -> np.ndarray:
    """Bank photos/scans vary wildly in resolution. Small text (MICR line,
    amount box) needs enough pixels for the recognizer to have a chance —
    normalize a floor width rather than assuming a fixed DPI."""
    h, w = img.shape[:2]
    if w >= min_width:
        return img
    scale = min_width / w
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    logger.debug("Upscaled %dx%d -> %dx%d (x%.2f)", w, h, resized.shape[1], resized.shape[0], scale)
    return resized


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Format-agnostic preprocessing applied identically to every cheque,
    regardless of issuing bank.

    Order matters: upscale first so later filters operate on more pixels,
    denoise before contrast so CLAHE doesn't amplify noise, deskew last so
    rotation is computed on the cleaned-up image.
    """
    img = _upscale_if_small(img)
    img = _denoise(img)
    img = _enhance_contrast(img)
    img = _deskew(img)
    return img


# ---------------------------------------------------------------------------
# Image / PDF loading
# ---------------------------------------------------------------------------

def _load_image(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, np.ndarray]]:
    """Render every page of a PDF to a numpy array using pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError("pypdfium2 not installed. Run: pip install pypdfium2")

    doc = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72  # pypdfium2 default is 72 DPI
    images = []

    for page_num, page in enumerate(doc, start=1):
        bitmap = page.render(scale=scale, rotation=0)
        img_array = bitmap.to_numpy()

        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        images.append((page_num, img_array))
        logger.debug("Page %d rendered at %d DPI (%dx%d px)",
                     page_num, dpi, img_array.shape[1], img_array.shape[0])

    doc.close()
    return images


# ---------------------------------------------------------------------------
# OCR execution
# ---------------------------------------------------------------------------

def _run_ocr(img: np.ndarray, ocr) -> list[dict[str, Any]]:
    """Run OCR on a single preprocessed image and return structured,
    line-level results — same shape as before: [{"text","confidence","box"}].

    Deliberately generic — no field semantics here, same contract as the
    2.x version. debug_inspect.py and downstream field-extraction code do
    not need to change.
    """
    try:
        with _ocr_lock:
            results = ocr.predict(img)
    except Exception as exc:
        logger.warning("OCR error: %s", exc)
        return []

    lines: list[dict[str, Any]] = []
    if not results:
        return lines

    for res in results:
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        polys = res["rec_polys"]  # 4-point polygon per line, same format as before

        for text, score, poly in zip(texts, scores, polys):
            text = str(text).strip()
            if not text:
                continue
            lines.append({
                "text": text,
                "confidence": float(score),
                "box": np.asarray(poly).tolist(),  # keep as list of [x,y] pairs
            })
    return lines


def process_image(image: "str | np.ndarray", apply_preprocess: bool = True) -> list[dict[str, Any]]:
    """OCR a single image (file path or already-loaded RGB numpy array).

    Returns a list of line dicts: {"text", "confidence", "box"}.
    This is the primitive a downstream field-extraction module should
    build on: it carries position and confidence so that module can locate
    fields and decide what's missing or too low-confidence to trust,
    without this module needing to know anything about cheque semantics.
    """
    img = _load_image(image) if isinstance(image, (str, Path)) else image

    if apply_preprocess:
        img = preprocess_image(img)

    ocr = _get_ocr()
    t0 = time.time()
    lines = _run_ocr(img, ocr)
    logger.info("OCR done in %.2fs — %d lines", time.time() - t0, len(lines))
    return lines


def process_pdf(pdf_path: str, dpi: int = 300, apply_preprocess: bool = True) -> list[dict[str, Any]]:
    """OCR every page of a PDF.

    Returns a list of page dicts:
        [{"page": int, "lines": [{"text","confidence","box"}, ...]}, ...]
    """
    pdf_path = str(pdf_path)
    logger.info("Processing: %s at %d DPI", pdf_path, dpi)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    ocr = _get_ocr()

    pages: list[dict[str, Any]] = []
    t0 = time.time()

    for page_num, img_array in images:
        img = preprocess_image(img_array) if apply_preprocess else img_array
        lines = _run_ocr(img, ocr)
        pages.append({"page": page_num, "lines": lines})
        logger.debug("Page %d: %d lines", page_num, len(lines))

    elapsed = time.time() - t0
    total_lines = sum(len(p["lines"]) for p in pages)
    logger.info("Done in %.1fs — %d lines across %d page(s)", elapsed, total_lines, len(pages))
    return pages


def flatten_text(pages_or_lines: list) -> str:
    """Convenience helper: collapse structured results back into plain
    text — e.g. for logging/debugging, or to preserve the old .txt-file
    behavior in app.py without that route needing to understand the
    structured format. Accepts either process_pdf's page list or
    process_image's line list.
    """
    if pages_or_lines and "lines" in pages_or_lines[0]:
        lines = [ln["text"] for page in pages_or_lines for ln in page["lines"]]
    else:
        lines = [ln["text"] for ln in pages_or_lines]
    return "\n".join(lines)
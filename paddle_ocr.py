from __future__ import annotations

# Must be set BEFORE any paddle import — C++ runtime reads it at DLL init time.
import os
os.environ["FLAGS_use_mkldnn"] = "0"

import io
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton OCR instance
# ---------------------------------------------------------------------------

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError(
            "PaddleOCR not installed. Run: pip install paddlepaddle==3.2.2 paddleocr>=3.3.2"
        )

    logger.info("Loading PP-OCRv5 model (CPU)...")
    _ocr_instance = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.8,
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
        # ── oneDNN / Windows fix ──────────────────────────────────────────
        # Forces PaddleX to use run_mode="paddle" (standard CPU kernels)
        # instead of the default run_mode="mkldnn", bypassing the broken
        # PIR/MKLDNN code path in PaddlePaddle 3.3.0+.
        enable_mkldnn=False,
        # ─────────────────────────────────────────────────────────────────
    )
    logger.info("Model loaded.")
    return _ocr_instance


# ---------------------------------------------------------------------------
# PDF → PIL images
# ---------------------------------------------------------------------------

def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    """Render every page of a PDF to a PIL Image at the given DPI."""
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow not installed. Run: pip install Pillow")

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images: list[tuple[int, object]] = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = __import__("PIL.Image", fromlist=["Image"]).open(
            io.BytesIO(pix.tobytes("png"))
        )
        images.append((page_num, img))
        logger.debug("Page %d rendered at %d DPI (%dx%d px)", page_num, dpi, *img.size)

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, dpi: int = 300) -> str:
    """
    Run OCR on a PDF file and return extracted text as a single string.

    Parameters
    ----------
    pdf_path : str | Path
        Absolute or relative path to the PDF file.
    dpi : int
        Render resolution. Use 300 for standard docs, 400 for degraded scans.

    Returns
    -------
    str
        Extracted text, one OCR'd line per newline. Empty string if no text found.

    Raises
    ------
    RuntimeError
        If a required dependency is missing or the model fails to load.
    Exception
        Propagates any unexpected error from PaddleOCR or PyMuPDF.
    """
    import numpy as np

    pdf_path = str(pdf_path)
    logger.info("Processing: %s at %d DPI", pdf_path, dpi)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    ocr    = _get_ocr()

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        logger.debug("OCR on page %d...", page_num)
        img_array = np.array(img.convert("RGB"))

        try:
            results = ocr.predict(img_array)
        except Exception as exc:
            logger.warning("Page %d OCR error: %s", page_num, exc)
            continue

        if not results:
            logger.debug("Page %d: no text detected.", page_num)
            continue

        for page_result in results:
            if page_result is None:
                continue
            rec_texts  = page_result.get("rec_texts",  []) or []
            rec_scores = page_result.get("rec_scores", []) or []
            if len(rec_scores) < len(rec_texts):
                rec_scores = list(rec_scores) + [0.0] * (len(rec_texts) - len(rec_scores))
            for text, _score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if text:
                    all_lines.append(text)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs — %d lines extracted", elapsed, len(all_lines))
    return "\n".join(all_lines)
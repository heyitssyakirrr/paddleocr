from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError("PaddleOCR not installed.")

    # For air-gapped environments: use pre-downloaded models from local directory.
    # __file__ = <project_root>/app/services/paddle_ocr.py
    # .parent       = app/services/
    # .parent.parent = app/
    # .parent.parent.parent = <project_root>/
    model_base = Path(__file__).resolve().parent / "models"
    det_dir = model_base / "en_PP-OCRv3_det_infer"
    rec_dir = model_base / "en_PP-OCRv3_rec_infer"
    cls_dir = model_base / "ch_ppocr_mobile_v2.0_cls_infer"

    local_kwargs = {}
    if det_dir.exists() and rec_dir.exists():
        logger.info("Using local models from %s", model_base)
        local_kwargs = {
            "det_model_dir": str(det_dir),
            "rec_model_dir": str(rec_dir),
            "cls_model_dir": str(cls_dir) if cls_dir.exists() else None,
        }
        local_kwargs = {k: v for k, v in local_kwargs.items() if v is not None}

    logger.info("Loading PaddleOCR model (CPU)...")
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
        enable_mkldnn=False,
        **local_kwargs,
    )
    logger.info("Model loaded.")
    return _ocr_instance


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    """Render every page of a PDF to a numpy array using pypdfium2."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise RuntimeError("pypdfium2 not installed. Run: pip install pypdfium2")

    import numpy as np

    doc = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72  # pypdfium2 default is 72 DPI
    images = []

    for page_num, page in enumerate(doc, start=1):
        bitmap = page.render(scale=scale, rotation=0)
        img_array = bitmap.to_numpy()

        # Drop alpha channel if present (RGBA -> RGB)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        images.append((page_num, img_array))
        logger.debug("Page %d rendered at %d DPI (%dx%d px)",
                     page_num, dpi, img_array.shape[1], img_array.shape[0])

    doc.close()
    return images


def process_pdf(pdf_path: str, dpi: int = 300) -> str:
    pdf_path = str(pdf_path)
    logger.info("Processing: %s at %d DPI", pdf_path, dpi)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    ocr = _get_ocr()

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img_array in images:
        logger.debug("OCR on page %d...", page_num)

        try:
            results = ocr.ocr(img_array, cls=True)
        except Exception as exc:
            logger.warning("Page %d OCR error: %s", page_num, exc)
            continue

        if not results:
            logger.debug("Page %d: no text detected.", page_num)
            continue

        for page_result in results:
            if page_result is None:
                continue
            for line in page_result:
                if line is None:
                    continue
                text_info = line[1]  # (text, confidence)
                text = str(text_info[0]).strip()
                if text:
                    all_lines.append(text)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs — %d lines extracted", elapsed, len(all_lines))
    return "\n".join(all_lines)
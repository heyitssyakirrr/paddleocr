from __future__ import annotations

import os
os.environ["FLAGS_use_mkldnn"] = "0"

import io
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

    logger.info("Loading PP-OCRv5 model (CPU)...")
    _ocr_instance = PaddleOCR(
        lang="en",
        text_detection_model_name="PP-OCRv5_mobile_det",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_thresh=0.3,
        text_det_box_thresh=0.5,
        text_det_unclip_ratio=1.8,
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
        enable_mkldnn=False,
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
        # to_numpy() returns H x W x channels (RGB or RGBA)
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
    import numpy as np

    pdf_path = str(pdf_path)
    logger.info("Processing: %s at %d DPI", pdf_path, dpi)

    images = _pdf_to_images(pdf_path, dpi=dpi)
    ocr = _get_ocr()

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img_array in images:
        logger.debug("OCR on page %d...", page_num)

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
            dt_polys   = page_result.get("dt_polys",   []) or []

            if len(rec_scores) < len(rec_texts):
                rec_scores = list(rec_scores) + [0.0] * (len(rec_texts) - len(rec_scores))

            # Pair each text with its bounding box x-start and y-center
            boxes = []
            for i, (text, _score) in enumerate(zip(rec_texts, rec_scores)):
                text = str(text).strip()
                if not text:
                    continue
                if i < len(dt_polys) and dt_polys[i] is not None:
                    poly = dt_polys[i]
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    x_min = min(xs)
                    x_max = max(xs)
                    y_center = (min(ys) + max(ys)) / 2
                    width = x_max - x_min
                else:
                    x_min, y_center, width = 0, 0, 0
                boxes.append((y_center, x_min, x_max, width, text))

            # Sort by y first (top to bottom), then x (left to right)
            boxes.sort(key=lambda b: (round(b[0] / 15), b[1]))

            # Group into lines by y proximity, then insert spaces by x gaps
            line_groups = []
            current_group = []
            prev_y = None
            for item in boxes:
                y_center = item[0]
                if prev_y is None or abs(y_center - prev_y) < 20:
                    current_group.append(item)
                else:
                    line_groups.append(current_group)
                    current_group = [item]
                prev_y = y_center
            if current_group:
                line_groups.append(current_group)

            for group in line_groups:
                group.sort(key=lambda b: b[1])  # sort by x_min
                line_text = group[0][4]
                for j in range(1, len(group)):
                    prev_x_max = group[j-1][2]
                    curr_x_min = group[j][1]
                    gap = curr_x_min - prev_x_max
                    # If gap is significant relative to avg char width, insert space
                    avg_char_width = group[j-1][3] / max(len(group[j-1][4]), 1)
                    if gap > avg_char_width * 0.3:
                        line_text += " " + group[j][4]
                    else:
                        line_text += group[j][4]
                all_lines.append(line_text)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs — %d lines extracted", elapsed, len(all_lines))
    return "\n".join(all_lines)
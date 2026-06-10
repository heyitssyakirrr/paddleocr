"""
paddle_ocr.py
=============
Simple PaddleOCR text extractor — Linux, CPU only.
Outputs extracted text only, no metrics, no confidence scores.

Usage:
    python paddle_ocr.py --folder input_files
    python paddle_ocr.py --folder input_files --dpi 400
    python paddle_ocr.py --folder input_files --output-dir results
"""

from __future__ import annotations
import argparse
import io
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 300) -> list[tuple[int, object]]:
    try:
        import fitz
    except ImportError:
        sys.exit("[ERROR] PyMuPDF not installed.  pip install PyMuPDF")
    try:
        from PIL import Image
    except ImportError:
        sys.exit("[ERROR] Pillow not installed.  pip install Pillow")

    doc  = fitz.open(pdf_path)
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    imgs = []

    for page_num, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        imgs.append((page_num, img))
        print("  [PDF] Page %d rendered at %d DPI: %dx%d px" % (
            page_num, dpi, img.size[0], img.size[1]
        ))

    doc.close()
    return imgs


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

_ocr_instance = None


def _get_ocr():
    global _ocr_instance
    if _ocr_instance is not None:
        return _ocr_instance

    from paddleocr import PaddleOCR
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
    )
    return _ocr_instance


def run_paddle(images: list) -> str:
    try:
        import numpy as np
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] PaddleOCR not installed.  pip install paddlepaddle>=3.1.1 paddleocr>=3.6.0")

    import numpy as np

    print("  [PaddleOCR] Loading PP-OCRv5 model (CPU)...")
    try:
        ocr = _get_ocr()
    except Exception as exc:
        sys.exit("[ERROR] PaddleOCR model load failed: %s" % exc)

    all_lines: list[str] = []
    t0 = time.time()

    for page_num, img in images:
        print("  [PaddleOCR] Processing page %d..." % page_num)
        img_array = np.array(img.convert("RGB"))

        try:
            results = ocr.predict(img_array)
        except Exception as exc:
            print("    [PaddleOCR] Page %d error: %s" % (page_num, exc))
            continue

        if not results:
            print("    [PaddleOCR] Page %d: no text detected." % page_num)
            continue

        for page_result in results:
            if page_result is None:
                continue
            rec_texts  = page_result.get("rec_texts",  []) or []
            rec_scores = page_result.get("rec_scores", []) or []
            if len(rec_scores) < len(rec_texts):
                rec_scores = list(rec_scores) + [0.0] * (len(rec_texts) - len(rec_scores))
            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if text:
                    all_lines.append(text)

    elapsed = time.time() - t0
    print("  [PaddleOCR] Done in %.1fs" % elapsed)
    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def list_pdfs(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        sys.exit("[ERROR] Folder not found: %s" % folder)
    pdfs = sorted(folder_path.glob("*.pdf"))
    if not pdfs:
        sys.exit("[ERROR] No PDF files found in: %s" % folder)
    return pdfs


def pick_files(pdfs: list[Path]) -> list[Path]:
    print("\nPDF files found:")
    for i, p in enumerate(pdfs, start=1):
        print("  [%d] %s" % (i, p.name))
    print("  [A] All files\n")
    choice = input("Select file(s) — number(s) comma-separated, or A for all: ").strip()

    if choice.upper() == "A":
        return pdfs

    selected: list[Path] = []
    for part in choice.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(pdfs):
                selected.append(pdfs[idx])
            else:
                print("  [WARN] Invalid number: %s (skipped)" % part)
        else:
            print("  [WARN] Invalid input: %s (skipped)" % part)

    if not selected:
        sys.exit("[ERROR] No valid files selected.")
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR text extractor — Linux CPU only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python paddle_ocr.py\n"
            "  python paddle_ocr.py --folder input_files\n"
            "  python paddle_ocr.py --folder input_files --dpi 400\n"
            "  python paddle_ocr.py --folder input_files --output-dir results\n"
        ),
    )
    parser.add_argument("--folder",     default="input_files",
                        help="Folder containing PDF files (default: input_files)")
    parser.add_argument("--dpi",        type=int, default=300,
                        help="PDF render DPI (default: 300; try 400 for degraded scans)")
    parser.add_argument("--output-dir", default="ocr_results",
                        help="Output folder for result files (default: ocr_results)")
    args = parser.parse_args()

    pdfs          = list_pdfs(args.folder)
    selected_pdfs = pick_files(pdfs)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSelected : %s" % ", ".join(p.name for p in selected_pdfs))
    print("DPI      : %d" % args.dpi)
    print("Output   : %s" % output_dir)
    print()

    for pdf_path in selected_pdfs:
        print("\n" + "=" * 60)
        print("  FILE: %s" % pdf_path.name)
        print("=" * 60)

        try:
            images  = pdf_to_images(str(pdf_path), dpi=args.dpi)
            text    = run_paddle(images)
            n_lines = len([l for l in text.split("\n") if l.strip()])

            # Write output — plain text only
            stem     = pdf_path.stem.lower().replace(" ", "_")
            out_file = output_dir / ("paddle_%s.txt" % stem)
            out_file.write_text(text, encoding="utf-8")

            print("  Lines extracted : %d" % n_lines)
            print("  Saved           : %s" % out_file)

        except Exception as exc:
            print("  [ERROR] %s: %s" % (pdf_path.name, exc))

    print("\n" + "=" * 60)
    print("  DONE — results saved to: %s/" % output_dir)
    print("=" * 60)
    for f in sorted(output_dir.glob("paddle_*.txt")):
        print("  - %s" % f.name)
    print()


if __name__ == "__main__":
    main()

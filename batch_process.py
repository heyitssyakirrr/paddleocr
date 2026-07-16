"""
batch_process.py — batch-run the cheque pipeline (date + signature
extraction) over a file or folder, writing:

    output/results.csv                          — cheque | date | signature_exists
    output/signatures/<safe_stem>_signature.png  — cropped signature, when found
    output/debug/<safe_stem>/date_crop.png       — what TrOCR actually read (QA)
    output/debug/<safe_stem>/result.json         — full diagnostics per cheque

USAGE:
    python batch_process.py uploads/bank_cheque_1.png
    python batch_process.py uploads/                    # whole folder
    python batch_process.py uploads/ --out my_results.csv

Only .png/.jpg/.jpeg are processed directly. For PDFs, render pages to
images first (see paddle_ocr._pdf_to_images) — kept out of scope here so
this script has one job.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from pathlib import PurePosixPath

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cheque_pipeline import config
from cheque_pipeline.pipeline import process_cheque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}  # PDFs excluded here by design — see module docstring


def _safe_stem(filename: str) -> str:
    """Filesystem-safe base name — strips directory components and any
    character that isn't alnum/._- , preventing path traversal or writing
    outside the intended output directories via a crafted filename."""
    name = PurePosixPath(filename).name
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return name or "file"


def _csv_safe(value) -> str:
    """Neutralize formula-injection characters (=, +, -, @) when a cell is
    opened in Excel/Sheets — prefix with a straight quote, standard OWASP
    mitigation for CSV output that may be viewed in a spreadsheet app."""
    text = str(value)
    if text and text[0] in ("=", "+", "-", "@"):
        return "'" + text
    return text


def process_one(path: Path) -> dict:
    result = process_cheque(path)
    safe_stem = _safe_stem(path.stem)

    if result["signature_exists"] and result["signature_crop"] is not None:
        config.SIGNATURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.SIGNATURES_DIR / f"{safe_stem}_signature.png"
        cv2.imwrite(str(out_path), result["signature_crop"])
        logger.info("Saved signature crop: %s", out_path)

    debug_dir = config.DEBUG_DIR / safe_stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    if result["date_crop"] is not None:
        cv2.imwrite(str(debug_dir / "date_crop.png"), result["date_crop"])

    diagnostics = {k: v for k, v in result.items() if not isinstance(v, (type(None),)) and "crop" not in k}
    with open(debug_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="Cheque image file or folder of images")
    parser.add_argument("--out", default=str(config.OUTPUT_DIR / "results.csv"), help="CSV output path")
    args = parser.parse_args()

    target = Path(args.target)
    if target.is_dir():
        paths = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    elif target.suffix.lower() in IMAGE_EXTENSIONS:
        paths = [target]
    else:
        print(f"Unsupported input: {target} (expected {sorted(IMAGE_EXTENSIONS)} file or a folder of them)")
        sys.exit(1)

    if not paths:
        print(f"No supported images found at {target}")
        sys.exit(1)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in paths:
        logger.info("Processing %s ...", path.name)
        try:
            result = process_one(path)
            rows.append({
                "cheque": result["filename"],
                "date": result["date"],
                "signature_exists": result["signature_exists"],
            })
        except Exception:
            logger.exception("Failed on %s", path.name)
            rows.append({"cheque": path.name, "date": "ERROR", "signature_exists": False})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cheque", "date", "signature_exists"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_safe(v) for k, v in row.items()})

    print("\n" + "=" * 70)
    print(f"{'Cheque':<28} {'Date':<14} {'Signature'}")
    print("=" * 70)
    for row in rows:
        print(f"{row['cheque']:<28} {str(row['date']):<14} {row['signature_exists']}")
    print("=" * 70)
    print(f"\nCSV written to: {out_path.resolve()}")
    print(f"Signature crops: {config.SIGNATURES_DIR.resolve()}")
    print(f"Per-cheque diagnostics: {config.DEBUG_DIR.resolve()}")


if __name__ == "__main__":
    main()

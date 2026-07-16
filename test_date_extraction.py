"""
test_date_extraction.py

Batch-tests anchor-based date field extraction against your existing cheque
images, using the SAME _get_ocr / _load_image / preprocess_image / _run_ocr
functions already in paddle_ocr.py. Nothing in paddle_ocr.py or
debug_inspect.py needs to change.

WHAT IT DOES, per image:
  1. Runs your normal full-image OCR pass (as debug_inspect.py already does).
  2. Fuzzy-matches OCR lines against known date-label variants
     (TARIKH / DATE / 日期 / common garbled forms seen in your samples).
  3. Builds a search band relative to that anchor's box (NOT a fixed
     page coordinate — it moves with wherever the label actually landed).
  4. Crops that band from the preprocessed image, upscales it, and re-runs
     OCR on just that crop.
  5. Regex-validates the result against a date-like pattern.
  6. Saves the crop (before/after upscale) + prints a results table, so you
     can see exactly what happened per cheque without doing anything by hand.

USAGE:
  Drop this file in the same folder as paddle_ocr.py, then:

    python test_date_extraction.py uploads/bank_cheque_1.png
    python test_date_extraction.py uploads/            # run on a whole folder

  Output goes to: date_extraction_debug/<stem>/
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuses your existing pipeline exactly as-is — no changes needed there.
from paddle_ocr import _get_ocr, _load_image, preprocess_image, _run_ocr

# ---------------------------------------------------------------------------
# Config — this is the ONLY part you'll likely need to extend as you add
# banks. No coordinates here, only text labels and search geometry.
# ---------------------------------------------------------------------------

DATE_LABEL_ALIASES = [
    "TARIKH", "DATE", "日期",
    # garbled forms actually seen in your OCR output — add more as you find them
    "B期DATE", "UVAVE", "DDATE", "口期",
]

# How far to search relative to the anchor, and in which direction.
# "right" covers the layouts you've shown so far (label then digit boxes to
# its right). If a bank places digits BELOW the label instead, add that
# bank's case by trying "below" — still no fixed coordinates, just a
# direction relative to wherever the anchor was actually found.
SEARCH_DIRECTION = "right"
SEARCH_WIDTH = 450     # generous on purpose — better to over-capture than miss
HEIGHT_PAD = 25
UPSCALE_FACTOR = 2.5

DATE_PATTERN = re.compile(r"\d{1,2}[/\-]?\d{1,2}[/\-]?\d{2,4}")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_anchor(lines: list[dict], aliases: list[str]) -> dict | None:
    """Return the OCR line that best matches a known date-label alias."""
    best = None
    for line in lines:
        text_upper = line["text"].upper().replace(" ", "")
        for alias in aliases:
            if alias.upper() in text_upper:
                # Prefer the highest-confidence match if multiple hit
                if best is None or line["confidence"] > best["confidence"]:
                    best = line
    return best


def get_search_band(
    anchor_box: list[list[float]],
    img_shape: tuple[int, int],
    direction: str = "right",
    width: int = SEARCH_WIDTH,
    height_pad: int = HEIGHT_PAD,
) -> tuple[int, int, int, int]:
    """Build a crop rectangle (x0, y0, x1, y1) relative to the anchor box,
    clamped to the image bounds. This is the part that keeps the approach
    bank-agnostic — it moves with the anchor, never a fixed offset from
    the page origin."""
    h, w = img_shape[:2]
    xs = [p[0] for p in anchor_box]
    ys = [p[1] for p in anchor_box]

    if direction == "right":
        x0 = max(xs)
        x1 = x0 + width
        y0 = min(ys) - height_pad
        y1 = max(ys) + height_pad
    elif direction == "below":
        x0 = min(xs)
        x1 = max(xs)
        y0 = max(ys)
        y1 = y0 + width
    else:
        raise ValueError(f"Unknown direction: {direction}")

    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    return x0, y0, x1, y1


def extract_date_field(img: np.ndarray, lines: list[dict], ocr) -> dict:
    """Full pipeline: find anchor -> crop+upscale -> re-OCR -> validate."""
    anchor = find_anchor(lines, DATE_LABEL_ALIASES)
    if anchor is None:
        return {"status": "anchor_not_found", "value": None, "crop": None}

    x0, y0, x1, y1 = get_search_band(anchor["box"], img.shape, SEARCH_DIRECTION)
    if x1 <= x0 or y1 <= y0:
        return {"status": "invalid_band", "value": None, "crop": None}

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return {"status": "empty_crop", "value": None, "crop": None}

    crop_upscaled = cv2.resize(
        crop, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR,
        interpolation=cv2.INTER_CUBIC,
    )

    reread = _run_ocr(crop_upscaled, ocr)
    combined = "".join(l["text"] for l in reread)
    digits_only = re.sub(r"[^0-9/\-]", "", combined)

    match = DATE_PATTERN.fullmatch(digits_only)
    status = "ok" if match else "needs_review"

    return {
        "status": status,
        "value": digits_only,
        "raw": combined,
        "anchor_text": anchor["text"],
        "anchor_confidence": anchor["confidence"],
        "crop": crop,
        "crop_upscaled": crop_upscaled,
        "reread_lines": reread,
    }


# ---------------------------------------------------------------------------
# Batch runner + debug output (mirrors debug_inspect.py's style)
# ---------------------------------------------------------------------------

def process_one(path: Path, ocr, out_root: Path) -> dict:
    img = _load_image(str(path))
    img = preprocess_image(img)
    lines = _run_ocr(img, ocr)

    result = extract_date_field(img, lines, ocr)

    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if result.get("crop") is not None:
        cv2.imwrite(str(out_dir / "01_date_band_crop.png"), result["crop"])
    if result.get("crop_upscaled") is not None:
        cv2.imwrite(str(out_dir / "02_date_band_upscaled.png"), result["crop_upscaled"])

    with open(out_dir / "result.txt", "w", encoding="utf-8") as f:
        f.write(f"file: {path.name}\n")
        f.write(f"status: {result['status']}\n")
        f.write(f"value: {result.get('value')}\n")
        f.write(f"raw_reread_text: {result.get('raw')}\n")
        f.write(f"anchor_text: {result.get('anchor_text')}\n")
        f.write(f"anchor_confidence: {result.get('anchor_confidence')}\n")

    return {"file": path.name, **{k: v for k, v in result.items()
                                    if k not in ("crop", "crop_upscaled", "reread_lines")}}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        image_paths = sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")
        )
    else:
        image_paths = [target]

    if not image_paths:
        print(f"No images found at {target}")
        sys.exit(1)

    ocr = _get_ocr()
    out_root = Path("date_extraction_debug")
    out_root.mkdir(exist_ok=True)

    results = []
    for path in image_paths:
        print(f"Processing {path.name}...")
        try:
            results.append(process_one(path, ocr, out_root))
        except Exception as exc:
            print(f"  ERROR on {path.name}: {exc}")
            results.append({"file": path.name, "status": "error", "value": str(exc)})

    print("\n" + "=" * 70)
    print(f"{'File':<20} {'Status':<16} {'Extracted value'}")
    print("=" * 70)
    for r in results:
        print(f"{r['file']:<20} {r['status']:<16} {r.get('value')}")
    print("=" * 70)
    print(f"\nFull crops + per-file details written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
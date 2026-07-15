"""
debug_inspect.py
----------------
Diagnostic tool — NOT part of the production pipeline.

Runs the same preprocessing + OCR as paddle_ocr.py on a single image, then:
  1. Saves the preprocessed image (so you can see exactly what the model sees).
  2. Draws every detected box + its recognized text + confidence onto a copy
     of the image, so you can see what got detected vs silently dropped.
  3. Saves each detected box as its own small crop file, named with its
     confidence score, so you can inspect exactly what the recognizer was
     looking at for any specific field.

Usage:
    python debug_inspect.py path/to/cheque.png [--no-preprocess]

Output goes to ./debug_out/<image_stem>/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from paddle_ocr import _get_ocr, _load_image, preprocess_image, _run_ocr  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--no-preprocess", action="store_true",
                         help="Skip preprocess_image() to compare raw vs preprocessed detection")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    out_dir = Path(__file__).parent / "debug_out" / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img = _load_image(str(img_path))
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")

    if not args.no_preprocess:
        img = preprocess_image(img)
        print(f"After preprocessing: {img.shape[1]}x{img.shape[0]}")

    cv2.imwrite(str(out_dir / "00_preprocessed.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    ocr = _get_ocr()
    lines = _run_ocr(img, ocr)

    print(f"\n{len(lines)} lines detected:\n")
    annotated = img.copy()

    for i, line in enumerate(sorted(lines, key=lambda l: l["confidence"])):
        text = line["text"]
        conf = line["confidence"]
        box = np.array(line["box"], dtype=np.int32)

        print(f"[{i:02d}] conf={conf:.3f}  text={text!r}  box={line['box']}")

        # Draw on the full annotated image
        color = (0, 200, 0) if conf > 0.5 else (0, 0, 255)  # red = low confidence
        cv2.polylines(annotated, [box], isClosed=True, color=color, thickness=2)
        cv2.putText(annotated, f"{conf:.2f}", tuple(box[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Save an individual crop for this box (padded a bit) so you can zoom in
        x_min, y_min = box[:, 0].min(), box[:, 1].min()
        x_max, y_max = box[:, 0].max(), box[:, 1].max()
        pad = 10
        h, w = img.shape[:2]
        x0, y0 = max(0, x_min - pad), max(0, y_min - pad)
        x1, y1 = min(w, x_max + pad), min(h, y_max + pad)
        crop = img[y0:y1, x0:x1]
        if crop.size:
            safe_text = "".join(c if c.isalnum() else "_" for c in text)[:20] or "empty"
            crop_name = f"{i:02d}_conf{conf:.2f}_{safe_text}.png"
            cv2.imwrite(str(out_dir / crop_name), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    cv2.imwrite(str(out_dir / "01_annotated_full.png"), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print(f"\nOutput written to: {out_dir.resolve()}")
    print("  00_preprocessed.png   - exactly what the model saw")
    print("  01_annotated_full.png - every detected box, green=confident red=low-confidence")
    print("  NN_confX.XX_text.png  - individual crop per detected box")


if __name__ == "__main__":
    main()
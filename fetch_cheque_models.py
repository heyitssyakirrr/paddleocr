# fetch_cheque_models.py — run ONCE, online, to cache the two NEW models
# this addon needs locally (does not touch PaddleOCR's own PP-OCR models —
# fetch_models.py already handles those).
#
# Usage:
#   python fetch_cheque_models.py
#
# After running, copy/keep the resulting folders under models/:
#   models/trocr-base-handwritten/
#   models/PP-DocLayout-S/

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 1. TrOCR (handwriting recognition on small field crops)
print("Downloading microsoft/trocr-base-handwritten ...")
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

trocr_dir = MODELS_DIR / "trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor.save_pretrained(trocr_dir)
model.save_pretrained(trocr_dir, safe_serialization=True)  # write safetensors, not pickle .bin
print(f"  saved to {trocr_dir}")

# 2. PP-Structure layout detection (signature/figure proxy)
print("Downloading PP-DocLayout-S layout detection model ...")
from paddleocr import LayoutDetection

LayoutDetection(model_name="PP-DocLayout-S")

# PaddleX (which paddleocr uses under the hood) caches this at a fixed
# location regardless of cwd — NOT under this project's models/ folder.
# For air-gapped/production use, copy it into models/ yourself so
# layout_engine.py's local-dir check picks it up, same pattern paddle_ocr.py
# already uses for the PP-OCR models:
default_cache = Path.home() / ".paddlex" / "official_models" / "PP-DocLayout-S"
target = MODELS_DIR / "PP-DocLayout-S"
print(f"  downloaded to: {default_cache}")
print(f"  for air-gapped use, copy it to: {target}")
print(f'  e.g. (Windows PowerShell): Copy-Item -Recurse "{default_cache}" "{target}"')

print("\nDone. Both models are ready for offline use where applicable.")

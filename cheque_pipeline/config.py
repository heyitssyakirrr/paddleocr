"""
config.py — single source of truth for the cheque pipeline addon.

Everything tunable lives here: model selection, paths, and the ROIs/
thresholds used by date_extractor.py and signature_extractor.py. Nothing
here duplicates paddle_ocr.py's own config (OCR_VERSION, model dirs for
PP-OCR itself) — this only adds the two NEW models (TrOCR + layout
detection) this addon introduces.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
SIGNATURES_DIR = OUTPUT_DIR / "signatures"
DEBUG_DIR = OUTPUT_DIR / "debug"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}

# ---------------------------------------------------------------------------
# TrOCR (handwriting recognition on small crops only — never the whole page)
# ---------------------------------------------------------------------------
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
TROCR_LOCAL_DIR = MODELS_DIR / "trocr-base-handwritten"
TROCR_DEVICE = "cpu"
TROCR_NUM_BEAMS = 4
TROCR_MAX_NEW_TOKENS = 16  # a date is a handful of characters; keep this small and fast

# ---------------------------------------------------------------------------
# PP-Structure layout detection (secondary corroboration only)
# ---------------------------------------------------------------------------
# IMPORTANT, read before tuning: PaddleOCR's layout-detection models do not
# have a "signature" class. They're trained on papers/magazines/contracts/
# reports, not cheques, and their closest classes to a handwritten signature
# are "figure" (a catch-all for non-text ink/graphics) and "seal" (stamps).
# Because that's not reliable on cheques, it is NOT the primary decision
# here (see signature_extractor.py's contour-based ink-blob detection,
# configured below) — it's only used to refine the crop box when it happens
# to agree, and never to overrule "no ink found" into "signature present".
LAYOUT_MODEL_NAME = "PP-DocLayout-S"  # 23-class model, includes "figure" and "seal"; small + fast on CPU
SIGNATURE_LABELS = {"figure", "seal", "image"}  # "image" covers the 20-cls model naming too, if swapped in
SIGNATURE_LAYOUT_SCORE_THRESHOLD = 0.3

# Signature band geometry — ANCHORED to the MICR line (the printed
# account/cheque-number line at the very bottom of every cheque), not a
# fixed fraction of the page. A fixed fraction broke across real scans
# with different crop margins/resolutions per cheque; anchoring to the
# MICR line — reliably the lowest detected text on any cheque, since it's
# machine-printed — makes the band move with each cheque's actual layout
# instead of assuming every image is cropped identically.
#
# Band = from (MICR line's top y - BAND_HEIGHT_FRAC * image_height) up to
# (MICR line's top y - BAND_BOTTOM_MARGIN_FRAC * image_height), spanning
# most of the width.
SIGNATURE_BAND_HEIGHT_FRAC = 0.32
SIGNATURE_BAND_BOTTOM_MARGIN_FRAC = 0.02
SIGNATURE_BAND_X_MARGIN_FRAC = (0.03, 0.02)  # (left, right) margins, as fractions of width

# Fallback ONLY if no OCR lines were detected at all (so no MICR anchor
# exists to anchor off of) — same fixed box as before, better than nothing.
SIGNATURE_REGION_FALLBACK: tuple[float, float, float, float] = (0.30, 0.55, 1.0, 0.90)

# ---------------------------------------------------------------------------
# Primary signature detection: contour/ink-blob analysis inside
# SIGNATURE_REGION. This does NOT depend on PP-Structure at all — it looks
# directly at the ink, which is what actually distinguishes "a signature is
# here" from "it isn't," and is tunable specifically for a cheque layout
# rather than inherited from a document-layout model that's never seen one.
#
# Approach: Otsu-threshold the region -> subtract long straight printed
# lines (box borders) via directional morphological opening -> connected
# components on what's left -> drop components too short to be a
# signature stroke (catches printed caption text like "authorised
# signature") -> what survives is treated as signature ink.
# ---------------------------------------------------------------------------
SIGNATURE_MIN_COMPONENT_AREA_FRAC = 0.0008   # vs region area — drops speckle/JPEG noise
SIGNATURE_MIN_COMPONENT_HEIGHT_FRAC = 0.08   # vs region height — drops printed caption text
SIGNATURE_INK_RATIO_THRESHOLD = 0.006        # surviving-ink-area / region-area, to call it "present"
SIGNATURE_CROP_PADDING_PX = 6

# ---------------------------------------------------------------------------
# Date-field anchor search (reuses PaddleOCR's own line detection — no
# separate model). Same aliases/geometry as test_date_extraction.py.
# ---------------------------------------------------------------------------
DATE_LABEL_ALIASES = [
    "TARIKH", "DATE", "日期",
    "B期DATE", "UVAVE", "DDATE", "口期",
]
DATE_SEARCH_DIRECTION = "right"
DATE_SEARCH_WIDTH = 450
DATE_HEIGHT_PAD = 25
DATE_CROP_UPSCALE_FACTOR = 2.5

# Env override so a deployment can point at a different model without a code change
LAYOUT_MODEL_NAME = os.environ.get("CHEQUE_LAYOUT_MODEL", LAYOUT_MODEL_NAME)

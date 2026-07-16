# Cheque pipeline addon — date + signature extraction

Combines this repo's existing PaddleOCR engine with two new engines
(TrOCR for handwriting, PP-Structure layout detection for signatures) into
one pipeline: cheque image in, `{date, signature_exists}` out, batched to CSV.

## What was added, and why

Your own testing found:
- PaddleOCR detects text boxes reliably but can't *read* handwritten dates.
- TrOCR reads a handwritten date crop well, but can't read a whole cheque page.
- PP-Structure's layout detector has no dedicated "signature" class — the
  closest proxies are its "figure"/"seal" classes (see the warning in
  `cheque_pipeline/config.py`, right above `SIGNATURE_LABELS`).

So the pipeline is: **PaddleOCR finds the date anchor and its box → crop →
TrOCR reads the crop**, and **signature detection looks directly at the ink
in the expected signature area** — Otsu threshold, subtract printed straight
lines (box borders), drop components too short to be a signature stroke
(filters out printed captions like "no signature below this line"), and
treat what survives as the signature. PP-Structure's figure/seal detection
only runs as a secondary corroboration on the crop box — it's never what
decides "present" vs "absent", since it has no real signature class.

## New folder structure

```
paddleocr-repo/
├── paddle_ocr.py               # UNCHANGED — your existing PaddleOCR engine
├── app.py, templates/, ...      # UNCHANGED — existing web app
├── batch_process.py             # NEW — CLI entry point (this is what you run)
├── fetch_cheque_models.py       # NEW — run once, online, to cache the 2 new models
├── requirements.txt              # UPDATED — added torch/transformers/sentencepiece
└── cheque_pipeline/              # NEW package — one engine/concern per file
    ├── config.py                 # all paths/models/ROIs/thresholds, one place
    ├── trocr_engine.py           # TrOCR model + recognize() — the "TrOCR file"
    ├── layout_engine.py          # PP-Structure LayoutDetection wrapper
    ├── date_extractor.py         # anchor (PaddleOCR) -> crop -> TrOCR read
    ├── signature_extractor.py    # layout figure/seal detect + ink fallback
    └── pipeline.py                # orchestrates the above into one result/cheque
```

`paddle_ocr.py` itself was **not modified** — `pipeline.py` only imports its
existing `_get_ocr`, `_load_image`, `preprocess_image`, `_run_ocr`.

## What to `pip install`

Your existing `requirements.txt` already had PaddleOCR's deps. Three lines
were added at the bottom for the new engines:

```
torch>=2.2.0
transformers>=4.41.0
sentencepiece>=0.2.0
```

Run (in your existing venv):
```powershell
pip install -r requirements.txt
```
CPU-only `torch` is fine — TrOCR only ever runs on small already-cropped
images here, never a whole page, so it's cheap.

## What models to download

Two **new** models are needed beyond what you already have (`fetch_models.py`
still only handles PP-OCR itself — untouched):

```powershell
python fetch_cheque_models.py
```

This downloads and caches:
1. **`microsoft/trocr-base-handwritten`** (TrOCR) → saved directly to
   `models/trocr-base-handwritten/` (safetensors, not pickle).
2. **`PP-DocLayout-S`** (PP-Structure layout detection, 23 classes incl.
   `figure`/`seal`) → PaddleX caches this at
   `~/.paddlex/official_models/PP-DocLayout-S` (its own fixed location, not
   your project folder). For air-gapped/production use, copy that folder to
   `models/PP-DocLayout-S/` — the script prints the exact copy command for
   your machine when it finishes.

Both `trocr_engine.py` and `layout_engine.py` check `models/` first and only
fall back to an online download if nothing local is found — same pattern
`paddle_ocr.py` already uses for the PP-OCR weights.

## Usage

```powershell
python batch_process.py uploads\bank_cheque_1.png
python batch_process.py uploads\                       # whole folder
python batch_process.py uploads\ --out my_results.csv
```

Output:
```
output/
├── results.csv                              # cheque | date | signature_exists
├── signatures/<name>_signature.png          # cropped signature, when found
└── debug/<name>/
    ├── date_crop.png                        # exactly what TrOCR read — check this first
    └── result.json                          # full diagnostics (method used, scores, etc.)
```

`results.csv` has exactly the 3 columns you asked for. `debug/<name>/result.json`
carries everything else (which detection method decided the signature
verdict, TrOCR's confidence, the matched date-label text) so you can audit
*why* a row came out the way it did without re-running anything.

## Tuning, if results are off

- **Date not found**: check `debug/<name>/result.json`'s `date_status` field.
  `anchor_not_found` means none of `TARIKH`/`DATE`/`日期` (or the garbled
  variants already in `config.DATE_LABEL_ALIASES`) matched — add whatever
  garbled form your OCR actually produced. `needs_review` means a crop was
  read but didn't match a date-like pattern — check `date_crop.png`.
- **Signature wrong**: primary detection is ink-blob analysis, not PP-Structure —
  check `config.SIGNATURE_REGION` first (fractional box, as fractions of image
  width/height); narrow or move it if it's catching the bank logo or missing
  the actual signature panel on your cheque layout. If the region is right but
  the verdict is still wrong, check `output/debug/<name>/result.json`'s
  `ink_ratio` against `config.SIGNATURE_INK_RATIO_THRESHOLD` — and
  `config.SIGNATURE_MIN_COMPONENT_HEIGHT_FRAC` if printed caption text
  (e.g. "no signature below this line") is getting counted as a signature.

## Honest limitation

I couldn't run this against real PaddleOCR/PaddlePaddle/TrOCR/PaddleX model
downloads in the environment I wrote this in (no access to Hugging Face or
Paddle's model-hosting domains there). Every module was smoke-tested with
this exact cheque image using lightweight stand-ins for `torch`,
`transformers`, and `paddleocr` to verify the control flow, cropping math,
and file I/O are all correct end-to-end — but the real accuracy of PP-
DocLayout-S's figure/seal detection on your actual cheque stock, and of
TrOCR on your actual handwriting, can only be judged by running it on your
machine. Start with a handful of samples and check `debug_out/` before
trusting it on a full batch.

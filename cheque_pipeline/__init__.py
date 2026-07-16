"""
cheque_pipeline — combines this repo's PaddleOCR engine (text detection +
box-level anchoring) with a TrOCR engine (accurate handwriting recognition
on small crops) and a PP-Structure layout-detection engine (figure/seal
regions, used as a signature proxy) into one cheque -> {date, signature}
extraction pipeline.

Module map (one concern per file, as requested):
    config.py                — all paths/models/thresholds/ROIs in one place
    trocr_engine.py          — TrOCR model loading + recognize() (singleton)
    layout_engine.py         — PP-Structure LayoutDetection wrapper (singleton)
    date_extractor.py        — anchor-based date crop (via PaddleOCR boxes) + TrOCR read
    signature_extractor.py   — layout-based figure/seal detection + ink-density backstop
    pipeline.py              — orchestrates the above into one result per cheque

Nothing in paddle_ocr.py needed to change — this package only imports the
existing public/underscored helpers it already exposes.
"""

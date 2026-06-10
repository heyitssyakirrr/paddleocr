# PaddleOCR Web App

PDF text extraction via PaddleOCR PP-OCRv5, served through a FastAPI UI.

## Requirements

- Linux (Ubuntu 20.04+)
- Python 3.10
- No GPU required — CPU-only inference

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python app.py                  # default: http://0.0.0.0:8000
python app.py --port 8080      # custom port
python app.py --host 127.0.0.1 # localhost only
```

Open `http://localhost:8000` in your browser.

## Usage

1. Drag-and-drop or click to select one or more PDF files.
2. Choose DPI (300 for standard docs, 400 for degraded scans).
3. Click **Run OCR**.
4. Download individual `.txt` files or use **Download All (.zip)**.

## Project structure

```
ocr_app/
├── app.py            # FastAPI server — routes, upload handling, ZIP packaging
├── paddle_ocr.py     # OCR engine — process_pdf(path, dpi) -> str
├── requirements.txt
├── templates/
│   └── index.html    # Single-page UI
├── uploads/          # Temporary upload staging (auto-cleaned after OCR)
└── results/          # Output .txt files (cleared via UI or manually)
```

## FastAPI notes

- File uploads use `UploadFile` + `python-multipart` (required dependency).
- `process_pdf()` is CPU-bound. It runs via `loop.run_in_executor(None, ...)`,
  dispatched to a thread-pool worker so the async event loop is never blocked.
- The ZIP download uses `StreamingResponse` with an in-memory `BytesIO` buffer —
  no temp file written to disk.
- Server: Uvicorn (ASGI). Auto-reload is off in production mode.

## oneDNN / PIR fix

`enable_mkldnn=False` in `paddle_ocr.py` and `FLAGS_use_mkldnn=0` env-var guard
against a PaddlePaddle 3.3.0+ PIR/MKLDNN crash on CPU.
See: https://github.com/PaddlePaddle/PaddleOCR/issues/17539
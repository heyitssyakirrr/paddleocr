from __future__ import annotations

import asyncio
import io
import logging
import uuid
import zipfile
from functools import partial
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
TEMPLATES  = BASE_DIR / "templates"

ALLOWED_EXT     = {".pdf"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per file

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PaddleOCR Service", version="1.0.0")
templates = Jinja2Templates(directory=str(TEMPLATES))


def _safe_stem(filename: str) -> str:
    """Return a filesystem-safe base name, stripping path components."""
    from pathlib import PurePosixPath
    name = PurePosixPath(filename).name          # strip any directory parts
    name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
    return name or "file"


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    dpi:   int              = Form(300),
):
    """
    Accept one or more PDF files, run OCR on each in a thread-pool worker,
    and return a JSON list of results.

    Response shape:
        { "results": [ { "original": str, "output": str | null,
                          "status": "done"|"error", "lines": int,
                          "error": str | null } ] }
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files received.")

    # Import here so the module (and PaddlePaddle) loads only once needed.
    from paddle_ocr import process_pdf

    loop    = asyncio.get_event_loop()
    results = []

    for upload_file in files:
        original_name = upload_file.filename or "unnamed.pdf"

        if not _allowed(original_name):
            results.append({
                "original": original_name,
                "output":   None,
                "status":   "error",
                "lines":    0,
                "error":    "Not a PDF file.",
            })
            continue

        safe_name   = _safe_stem(original_name)
        unique_stem = f"{uuid.uuid4().hex}_{safe_name}"
        upload_path = UPLOAD_DIR / unique_stem

        # Stream upload to disk
        try:
            contents = await upload_file.read()
            if len(contents) > MAX_UPLOAD_BYTES:
                results.append({
                    "original": original_name,
                    "output":   None,
                    "status":   "error",
                    "lines":    0,
                    "error":    f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
                })
                continue
            upload_path.write_bytes(contents)
            logger.info("Saved upload: %s", upload_path)
        except Exception as exc:
            logger.exception("Failed to save upload: %s", original_name)
            results.append({
                "original": original_name,
                "output":   None,
                "status":   "error",
                "lines":    0,
                "error":    f"Upload failed: {exc}",
            })
            continue
        finally:
            await upload_file.close()

        # Run OCR in thread-pool (blocking CPU work — must not run in event loop)
        try:
            text = await loop.run_in_executor(
                None,
                partial(process_pdf, str(upload_path), dpi),
            )
            n_lines = len([ln for ln in text.split("\n") if ln.strip()])

            stem        = Path(safe_name).stem.lower().replace(" ", "_")
            result_name = f"paddle_{stem}.txt"
            result_path = RESULT_DIR / result_name
            result_path.write_text(text, encoding="utf-8")

            results.append({
                "original": original_name,
                "output":   result_name,
                "status":   "done",
                "lines":    n_lines,
                "error":    None,
            })
            logger.info("Result saved: %s (%d lines)", result_path, n_lines)

        except Exception as exc:
            logger.exception("OCR failed for %s", safe_name)
            results.append({
                "original": original_name,
                "output":   None,
                "status":   "error",
                "lines":    0,
                "error":    str(exc),
            })
        finally:
            upload_path.unlink(missing_ok=True)

    return JSONResponse({"results": results})


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a single result .txt file."""
    safe      = _safe_stem(filename)
    file_path = RESULT_DIR / safe
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=str(file_path),
        media_type="text/plain",
        filename=safe,
    )


@app.get("/download-all")
async def download_all():
    """Bundle all result .txt files into an in-memory ZIP and stream it."""
    txt_files = sorted(RESULT_DIR.glob("*.txt"))
    if not txt_files:
        raise HTTPException(status_code=404, detail="No results to download.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in txt_files:
            zf.write(f, arcname=f.name)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="ocr_results.zip"'},
    )


@app.delete("/clear")
async def clear():
    """Remove all .txt result files from the results directory."""
    deleted = 0
    for f in RESULT_DIR.glob("*.txt"):
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    return JSONResponse({"deleted": deleted})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    logger.info("Starting OCR server on %s:%d", args.host, args.port)
    uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
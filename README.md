# PaddleOCR Text Extractor — Linux

Simple PDF text extractor using PaddleOCR PP-OCRv5. Outputs plain text only.

## Setup

```bash
# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```bash
# Put your PDF files in input_files/
mkdir input_files
cp your_file.pdf input_files/

# Run
python paddle_ocr.py

# With options
python paddle_ocr.py --folder input_files
python paddle_ocr.py --folder input_files --dpi 400
python paddle_ocr.py --folder input_files --output-dir results
```

## Output

Plain text files saved to `ocr_results/` (or your `--output-dir`):
- One `.txt` file per PDF
- Extracted text only, one line per detected text block
- No confidence scores, no metrics

## Notes

- No flags or env vars needed on Linux — works out of the box
- Default DPI is 300; use 400 for degraded or low-quality scans
- First run downloads PP-OCRv5 models (~100MB) to `~/.paddlex/`

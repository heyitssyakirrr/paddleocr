# fetch_models.py — run once online, not part of the app
from paddleocr import PaddleOCR

for det, rec in [
    ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),  # grab this now too, saves a trip later
]:
    print(f"Downloading {det} / {rec}...")
    PaddleOCR(text_detection_model_name=det, text_recognition_model_name=rec,
              use_doc_orientation_classify=False, use_doc_unwarping=False,
              use_textline_orientation=True)
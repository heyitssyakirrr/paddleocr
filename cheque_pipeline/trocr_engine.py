"""
trocr_engine.py — TrOCR model loading + inference. Kept entirely separate
from paddle_ocr.py: this is the "TrOCR in one file" half of the addon.

TrOCR is only ever called on small, already-cropped, single-line images
(see date_extractor.py) — never a whole cheque page. TrOCR (IAM-trained)
reads a short handwritten line well; it reads a whole multi-field document
badly. That split of responsibility is deliberate, not incidental.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import config

logger = logging.getLogger(__name__)

_model_instance: VisionEncoderDecoderModel | None = None
_processor_instance: TrOCRProcessor | None = None
_load_lock = threading.Lock()  # same rationale as paddle_ocr._ocr_lock: don't
                                # assume concurrent generate() calls are safe


def _resolve_model_source() -> str:
    """Prefer a local pre-downloaded copy (air-gapped/production use); fall
    back to the HF hub name only when internet is available. Run
    fetch_cheque_models.py once, online, to populate the local copy."""
    if config.TROCR_LOCAL_DIR.exists() and any(config.TROCR_LOCAL_DIR.iterdir()):
        logger.info("Using local TrOCR model at %s", config.TROCR_LOCAL_DIR)
        return str(config.TROCR_LOCAL_DIR)

    logger.warning(
        "No local TrOCR model at %s — will attempt to download from "
        "Hugging Face (fails with no internet). Run fetch_cheque_models.py "
        "once on a machine with internet, then copy models/ here.",
        config.TROCR_LOCAL_DIR,
    )
    return config.TROCR_MODEL_NAME


def get_engine() -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    """Return the (model, processor) pair, loading once and caching —
    mirrors paddle_ocr._get_ocr()'s singleton pattern exactly."""
    global _model_instance, _processor_instance

    if _model_instance is not None and _processor_instance is not None:
        return _model_instance, _processor_instance

    with _load_lock:
        if _model_instance is not None and _processor_instance is not None:
            return _model_instance, _processor_instance

        source = _resolve_model_source()
        logger.info("Loading TrOCR processor + model from %s ...", source)

        _processor_instance = TrOCRProcessor.from_pretrained(source)
        _model_instance = VisionEncoderDecoderModel.from_pretrained(
            source,
            use_safetensors=True,  # refuse to silently load a pickle .bin file
        )
        _model_instance.to(config.TROCR_DEVICE)
        _model_instance.eval()

        logger.info("TrOCR model loaded on device=%s", config.TROCR_DEVICE)

    return _model_instance, _processor_instance


def recognize(image_bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> tuple[str, float]:
    """Run TrOCR on a single small crop (numpy array, HxWx3).

    Set is_bgr=True (default) when passing an OpenCV-loaded array; the
    caller in this package always deals in BGR (same convention as
    paddle_ocr's cv2-based loaders), so this defaults to converting.

    Returns (text, confidence). Confidence is derived from the model's own
    generation scores — a real uncertainty signal, not a fixed placeholder.
    """
    import cv2

    if image_bgr_or_rgb.size == 0:
        return "", 0.0

    img_rgb = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2RGB) if is_bgr else image_bgr_or_rgb
    pil_image = Image.fromarray(img_rgb).convert("RGB")

    model, processor = get_engine()
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(config.TROCR_DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            num_beams=config.TROCR_NUM_BEAMS,
            max_new_tokens=config.TROCR_MAX_NEW_TOKENS,
            output_scores=True,
            return_dict_in_generate=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    if outputs.sequences_scores is not None:
        confidence = float(torch.exp(outputs.sequences_scores[0]))
    else:
        confidence = 1.0

    return text.strip(), confidence

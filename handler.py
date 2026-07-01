"""RunPod serverless handler for NVIDIA LocateAnything-3B.

Contract:
  Input:  {"image_b64": "<base64 jpeg>",
            "queries": ["clothes","dishes","cups","bottles","cans","boxes","bags","trash","shoes","papers","plates","blankets"]}
            (or "text": "<single string>" — split on commas)

  Output: {"detections": [{"label": str, "box": [x1,y1,x2,y2], "conf": float}],
            "count": int}
          boxes are normalized 0..1 (x1,y1 = top-left; x2,y2 = bottom-right).

Model loads once at container start; stays warm across invocations.
"""
import base64
import io
import os
import re
import logging

import runpod
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("LA_MODEL_PATH", "/runpod-volume/models/locate-anything")

# Box pattern: <box><x1><y1><x2><y2></box> — coords are ints in [0, 1000]
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

# Module-level globals — loaded once, reused across invocations
_model_loaded = False
_generate = None
_load = None
_load_pil = None


def _ensure_loaded():
    global _model_loaded, _generate, _load, _load_pil

    if _model_loaded:
        return

    # Set env vars before importing batch_utils — they configure attn backend on import
    os.environ.setdefault("LA_FLASH_MODEL", MODEL_PATH)
    os.environ.setdefault("LA_FLASH_ATTN", "sdpa")         # safe default; no custom kernels needed
    os.environ.setdefault("LA_FLASH_VISION_ATTN", "auto")

    log.info("Loading LocateAnything-3B from %s ...", MODEL_PATH)
    import sys
    sys.path.insert(0, MODEL_PATH)
    from batch_utils import load, generate_batch_hybrid
    from batch_utils.hybrid_runtime import load_pil as _lp  # noqa (used for type hint only)

    _load = load
    _generate = generate_batch_hybrid
    _load_pil = _lp

    load()
    _model_loaded = True
    log.info("Model loaded.")


def _parse_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Extract normalized [0,1] boxes from model output text."""
    boxes = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
        # Skip degenerate boxes
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def handler(job: dict) -> dict:
    """RunPod serverless entry point."""
    _ensure_loaded()

    inp = job.get("input", {})

    # Decode image
    b64 = inp.get("image_b64", "")
    if not b64:
        return {"error": "image_b64 is required"}
    try:
        img_bytes = base64.b64decode(b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"image decode failed: {e}"}

    # Resolve query list
    queries = inp.get("queries") or []
    if not queries:
        text_in = inp.get("text", "")
        if text_in:
            queries = [q.strip() for q in text_in.split(",") if q.strip()]
    if not queries:
        return {"error": "provide 'queries' list or 'text' string"}

    log.info("Locating %d categories in %dx%d image", len(queries), pil_image.width, pil_image.height)

    # Build (image, query) pairs — one per category for clean label association
    pairs = []
    for term in queries:
        prompt = f"Locate all the instances that matches the following description: {term}."
        pairs.append((pil_image, prompt))

    # Single batched inference call — faster than N sequential calls
    try:
        texts = _generate(
            pairs,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=2048,
            scheduler="eager",
        )
    except Exception as e:
        log.exception("Inference failed")
        return {"error": f"inference error: {e}"}

    detections = []
    for term, raw_text in zip(queries, texts):
        boxes = _parse_boxes(raw_text)
        for (x1, y1, x2, y2) in boxes:
            detections.append({
                "label": term,
                "box": [x1, y1, x2, y2],
                "conf": 1.0,    # LocateAnything does not output calibrated confidence scores
            })

    log.info("Found %d detections across %d categories", len(detections), len(queries))
    return {"output": {"detections": detections, "count": len(detections)}}


runpod.serverless.start({"handler": handler})

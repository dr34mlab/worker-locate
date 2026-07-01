"""RunPod serverless handler for NVIDIA LocateAnything-3B.

Contract:
  Input:  {"image_b64": "<base64 jpeg>",
            "queries": ["clothes","dishes","cups","bottles","cans","boxes","bags","trash","shoes","papers","plates","blankets"]}
            (or "text": "<single string>" — split on commas)

  Output: {"detections": [{"label": str, "box": [x1,y1,x2,y2], "conf": float}],
            "count": int}
          boxes are normalized 0..1 (x1,y1 = top-left; x2,y2 = bottom-right).

Model loads once at container start; stays warm across invocations.

Attention: LA_FLASH_ATTN=la_flash (MagiAttention sparse) via env — drops peak
VRAM from ~35 GB (dense SDPA) to ~11.7 GB, required for 24 GB GPUs.
Queries run sequentially (batch_size=1) with cache flush between each to keep
peak VRAM flat across N queries.
Input is downscaled to MAX_SIDE px on the long edge before inference.
"""
import base64
import gc
import io
import os
import re
import logging

import runpod
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = os.environ.get("LA_MODEL_PATH", "/runpod-volume/models/locate-anything")

# Downscale input to this max dimension — jimi sends 1920x1080 which OOMs with dense attn
MAX_SIDE = 1280

# Box pattern: <box><x1><y1><x2><y2></box> — coords are ints in [0, 1000]
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

# Module-level globals — loaded once, reused across invocations
_model_loaded = False
_generate = None


def _ensure_loaded():
    global _model_loaded, _generate

    if _model_loaded:
        return

    # Set env vars before importing batch_utils — they configure the attn backend on import.
    # LA_FLASH_ATTN=la_flash (MagiAttention sparse) is set via the RunPod template env;
    # fall back to flash_attention_2 (standard flash-attn), then sdpa.
    os.environ.setdefault("LA_FLASH_MODEL", MODEL_PATH)
    os.environ.setdefault("LA_FLASH_ATTN", "flash_attention_2")
    os.environ.setdefault("LA_FLASH_VISION_ATTN", "flash_attention_2")

    log.info("Loading LocateAnything-3B from %s (attn=%s) ...",
             MODEL_PATH, os.environ.get("LA_FLASH_ATTN"))
    import sys
    sys.path.insert(0, MODEL_PATH)
    from batch_utils import load, generate_batch_hybrid

    _generate = generate_batch_hybrid
    load()
    _model_loaded = True
    log.info("Model loaded.")


def _downscale(img: Image.Image) -> Image.Image:
    """Downscale to MAX_SIDE on the long edge if needed, preserving aspect ratio."""
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_SIDE:
        return img
    scale = MAX_SIDE / longest
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _parse_boxes(text: str) -> list[tuple[float, float, float, float]]:
    """Extract normalized [0,1] boxes from model output text."""
    boxes = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(g) / 1000.0 for g in m.groups())
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def handler(job: dict) -> dict:
    """RunPod serverless entry point."""
    _ensure_loaded()

    inp = job.get("input", {})

    # Decode and downscale image
    b64 = inp.get("image_b64", "")
    if not b64:
        return {"error": "image_b64 is required"}
    try:
        img_bytes = base64.b64decode(b64)
        pil_image = _downscale(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
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

    log.info("Locating %d queries in %dx%d image", len(queries), pil_image.width, pil_image.height)

    # Run queries sequentially (batch_size=1) to keep peak VRAM flat.
    # generate_batch_hybrid with N pairs allocates N attention matrices simultaneously;
    # one-at-a-time with cache flush keeps it at single-query peak throughout.
    import torch
    detections = []
    for term in queries:
        prompt = f"Locate all the instances that matches the following description: {term}."
        try:
            texts = _generate(
                [(pil_image, prompt)],
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                max_new_tokens=2048,
                scheduler="eager",
            )
        except torch.cuda.OutOfMemoryError as e:
            log.error("OOM on query %r: %s", term, e)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            log.error("Inference error on query %r: %s", term, e)
            continue

        for x1, y1, x2, y2 in _parse_boxes(texts[0]):
            detections.append({
                "label": term,
                "box": [x1, y1, x2, y2],
                "conf": 1.0,    # model does not output calibrated probabilities
            })

        # Flush CUDA cache between queries so each starts with the same VRAM headroom
        torch.cuda.empty_cache()

    log.info("Found %d detections across %d queries", len(detections), len(queries))
    return {"output": {"detections": detections, "count": len(detections)}}


runpod.serverless.start({"handler": handler})

# worker-locate — RunPod serverless worker for NVIDIA LocateAnything-3B
#
# Build:  docker build -t ghcr.io/dr34mlab/worker-locate:latest .
# Push:   docker push ghcr.io/dr34mlab/worker-locate:latest
#
# Model is expected on RunPod network volume at /runpod-volume/models/locate-anything
# (downloaded once via deploy/download_model.sh).

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System deps for opencv-python-headless (libGL alternative)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — pin exact versions from the model card
RUN pip install --no-cache-dir \
    runpod==1.7.9 \
    transformers==4.57.1 \
    "numpy<2" \
    Pillow==11.1.0 \
    peft \
    torchvision \
    "decord==0.6.0" \
    "lmdb==1.7.5" \
    opencv-python-headless \
    huggingface_hub[hf_transfer]

# flash-attn build (for flash_attention_2 vision attn — optional, sdpa used by default in handler)
# Include it so the model card's recommended attn is available; build takes ~5 min
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn install failed — will use sdpa fallback (performance only, correctness ok)"

COPY handler.py /app/handler.py

ENV LA_MODEL_PATH=/runpod-volume/models/locate-anything
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "/app/handler.py"]

# GPU production image.
#
#   docker build -t human-action-monitor .
#   docker run --gpus all -p 8000:8000 -e ACCESS_TOKEN=change-me human-action-monitor
#
# The CUDA runtime ships inside the PyTorch wheels, so a plain Python base is
# enough; the host needs only the NVIDIA driver + NVIDIA Container Toolkit.
# Without a GPU the same image runs on the CPU.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/models/hf \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DEVICE=auto \
    PROBE_CAMERAS=0

# OpenCV runtime libraries; ffmpeg for browser-playable batch output.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements change.
COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements-gpu.txt \
 || (pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
     && pip install -r requirements.txt)

# CLIP, VideoMAE (GPU activity model) and SmolVLM2 (GPU subtitles) are read offline;
# bake them into the image.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)" \
 && python -c "from transformers import VideoMAEForVideoClassification as M, VideoMAEImageProcessor as P; \
c='MCG-NJU/videomae-base-finetuned-kinetics'; M.from_pretrained(c); P.from_pretrained(c)" \
 && python -c "from transformers import AutoModelForImageTextToText as M, AutoProcessor as P; \
c='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'; M.from_pretrained(c); P.from_pretrained(c)"

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY ui/dist ./ui/dist
COPY data/samples ./data/samples

RUN useradd --create-home --uid 1000 app \
 && mkdir -p logs \
 && chown -R app:app /app
USER app

EXPOSE 8000
# /api/health is open (no token) precisely so probes like this work.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "scripts/run_pipeline.py"]

# Production image, for a server with an NVIDIA GPU (default) or any CPU
# machine / edge board, x86_64 or arm64 (PROFILE=cpu). docker-compose.yml
# has both; docs/DEPLOYMENT.md explains them.
#
#   docker build -t human-action-monitor .                          # NVIDIA GPU
#   docker build --build-arg PROFILE=cpu -t human-action-monitor:cpu .   # CPU
#   docker run --gpus all -p 8000:8000 -e ACCESS_TOKEN=change-me human-action-monitor
#
# GPU: the CUDA runtime ships inside the PyTorch wheels, so a plain Python
# base is enough; the host needs only the NVIDIA driver + NVIDIA Container
# Toolkit. CPU: CPU-only PyTorch (PyPI's default would add ~3 GB of CUDA
# libraries) and the CPU runtimes; the models are compiled for the machine
# on the container's first start (scripts/optimise_for_device.py, a few
# minutes) and kept on a volume. A Jetson runs natively: ./setup.sh.
FROM python:3.11-slim-bookworm
ARG PROFILE=gpu

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
COPY requirements.txt requirements-gpu.txt requirements-edge.txt ./
RUN if [ "$PROFILE" = cpu ]; then \
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      && pip install -r requirements.txt -r requirements-edge.txt; \
    else \
      pip install -r requirements-gpu.txt \
      || (pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
          && pip install -r requirements.txt); \
    fi

COPY config ./config
COPY src ./src
COPY scripts ./scripts
COPY models ./models

# The Hugging Face models are read offline at run time: bake them in. The GPU
# image takes all of them (the build machine has no GPU to ask); the CPU
# image only CLIP, the one the CPU profile uses.
RUN if [ "$PROFILE" = cpu ]; then python scripts/download_models.py; \
    else python scripts/download_models.py --all; fi
COPY ui/dist ./ui/dist
COPY data/samples ./data/samples

RUN useradd --create-home --uid 1000 app \
 && mkdir -p logs models/compiled \
 && chown -R app:app /app
USER app

EXPOSE 8000
# /api/health is open (no token) precisely so probes like this work.
# start-period covers the CPU image's first start, which compiles the models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# On a CPU, compile the models for this machine once (kept on the
# models/compiled volume); on a GPU the script returns at once.
CMD ["sh", "-c", "[ -f models/compiled/manifest.json ] || python scripts/optimise_for_device.py; exec python scripts/run_pipeline.py"]

FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu

WORKDIR /app
USER root

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# antiword reads legacy .doc files. Optional: the build continues without it.
RUN (apt-get update && apt-get install -y --no-install-recommends antiword && rm -rf /var/lib/apt/lists/*) \
    || echo "WARNING: antiword not installed; legacy .doc files will be rejected"

# Keep PaddleOCR's CUDA/NCCL packages intact. The Paddle base image already
# carries the CUDA 12.6 user-space libraries it requires. Installing Torch with
# normal dependency resolution would downgrade NCCL (Paddle wants 2.25.1,
# torch 2.6 metadata asks for 2.21.5). We do not use torch.distributed/NCCL in
# this single-GPU extraction worker, so install Torch itself without replacing
# the base image's NVIDIA packages, and add its non-CUDA Python dependencies.
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir \
      "sympy==1.13.1" "mpmath==1.3.0" \
      --index-url https://pypi.org/simple && \
    python -m pip install --no-cache-dir \
      "triton==3.2.0" \
      --index-url "${TORCH_INDEX}" && \
    python -m pip install --no-cache-dir --no-deps \
      "torch==${TORCH_VERSION}" \
      --index-url "${TORCH_INDEX}"

# This extraction worker intentionally does NOT bake or load any fine-tuned LoRA.
# All extracted facts must be grounded in the current uploaded document.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Optional: bake Qwen3-8B (~16 GB) into the image so cold starts don't download it.
# Build with: --build-arg BAKE_QWEN=1
ARG BAKE_QWEN=0
RUN if [ "$BAKE_QWEN" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='/models/Qwen3-8B')"; \
    fi

COPY config.py handler.py ./
COPY pipeline/ ./pipeline/

# Docker builders normally do NOT expose an NVIDIA driver, so importing the
# GPU build of Paddle here would fail with: libcuda.so.1: cannot open ...
# Validate Paddle/PaddleOCR installation by package metadata at build time.
# The real import is lazy and happens only on a GPU worker when OCR is needed.
RUN python -c "import torch, transformers, accelerate, docx, openpyxl, xlrd, pymupdf, json_repair, runpod; import handler; print('torch', torch.__version__, '| transformers', transformers.__version__)" && \
    python -c "from importlib.metadata import version; print('paddlepaddle-gpu', version('paddlepaddle-gpu'), '| paddleocr', version('paddleocr'))"

CMD ["python", "-u", "handler.py"]

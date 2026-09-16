FROM python:3.10-slim-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_ENGINE=tesseract \
    OCR_LANG=ara+eng

# Small system layer: legacy .doc support + Arabic/English OCR for scanned PDFs.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      antiword \
      libgomp1 \
      tesseract-ocr \
      tesseract-ocr-ara \
      tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# CUDA-enabled PyTorch. CUDA user-space libraries are pulled as wheel dependencies,
# so the image works on RunPod GPU workers without the huge PaddleOCR-VL base image.
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url "${TORCH_INDEX}"

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Do NOT bake Qwen into the container: that would make image pulls huge again.
# config.py can use RunPod cached-model mounts when configured on the endpoint.
COPY config.py handler.py ./
COPY pipeline/ ./pipeline/

# Build-time checks must not require an NVIDIA driver.
RUN python -c "import torch, transformers, accelerate, docx, openpyxl, xlrd, pymupdf, json_repair, runpod; import handler; print('torch', torch.__version__, '| transformers', transformers.__version__)" && \
    python -c "import shutil, subprocess; assert shutil.which('tesseract'); print(subprocess.check_output(['tesseract','--version'], text=True).splitlines()[0])"

CMD ["python", "-u", "handler.py"]

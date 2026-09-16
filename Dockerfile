# PyTorch already includes torch + CUDA 12.6 + cuDNN.
# This avoids downloading several GB of CUDA wheel dependencies during GitHub Actions.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_ENGINE=tesseract \
    OCR_LANG=ara+eng

# Document/OCR system dependencies only.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates \
      antiword \
      libgomp1 \
      tesseract-ocr \
      tesseract-ocr-ara \
      tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Do NOT bake Qwen weights into the image.
COPY config.py handler.py ./
COPY pipeline/ ./pipeline/

# Build-time checks do not require an NVIDIA driver.
RUN python -c "import torch, transformers, accelerate, docx, openpyxl, xlrd, pymupdf, json_repair, runpod; import handler; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)" && \
    python -c "import shutil, subprocess; assert shutil.which('tesseract'); print(subprocess.check_output(['tesseract','--version'], text=True).splitlines()[0])"

CMD ["python", "-u", "handler.py"]

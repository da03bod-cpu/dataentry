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

# Torch must match the CUDA libs of the Paddle base image. If Paddle breaks after
# this step (cudnn/cublas conflicts), rebuild with a matching pair, e.g.
#   --build-arg TORCH_VERSION=2.7.1 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.6.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124
RUN python -m pip install --no-cache-dir "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# This extraction worker intentionally does NOT bake or load any fine-tuned LoRA.
# All extracted facts must be grounded in the current uploaded document.

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Optional: bake Qwen3-8B (~16 GB) into the image so cold starts don't download it.
# Build with: --build-arg BAKE_QWEN=1   (config.py picks /models/Qwen3-8B up automatically)
ARG BAKE_QWEN=0
RUN if [ "$BAKE_QWEN" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B', local_dir='/models/Qwen3-8B')"; \
    fi

COPY config.py handler.py ./
COPY pipeline/ ./pipeline/

# Fail the BUILD (not the first request) if the Python stack is broken.
RUN python -c "import torch, transformers, peft, accelerate, docx, openpyxl, xlrd, pymupdf, json_repair, runpod; \
import handler; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)" && \
    (python -c "import paddle, paddleocr; print('paddle', paddle.__version__)" \
     || echo "WARNING: paddle/paddleocr import failed; scanned PDFs will not work")

CMD ["python", "-u", "handler.py"]

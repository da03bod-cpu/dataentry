"""Runtime configuration. Import this module FIRST (before transformers/paddle)
so the cache environment variables take effect."""
import logging
import os


def _bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- caches
# Only use the network volume when it is actually mounted; otherwise caches
# stay inside the container instead of pointing at a missing path.
_VOLUME = "/runpod-volume"
if os.path.isdir(_VOLUME):
    os.environ.setdefault("HF_HOME", f"{_VOLUME}/huggingface")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", f"{_VOLUME}/paddlex")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("TRANSFORMERS_CACHE", None)  # deprecated; HF_HOME is enough

# ---------------------------------------------------------------- models
_BAKED_QWEN = "/models/Qwen3-8B"  # filled when the image is built with BAKE_QWEN=1
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME") or (
    _BAKED_QWEN if os.path.isfile(os.path.join(_BAKED_QWEN, "config.json")) else "Qwen/Qwen3-8B"
)
LORA_PATH = os.getenv("LORA_PATH", "/app/lora-release")
LOAD_IN_4BIT = _bool("LOAD_IN_4BIT", True)
PRELOAD_MODELS = _bool("PRELOAD_MODELS", True)

# ---------------------------------------------------------------- generation
MAX_INPUT_CHARS = _int("MAX_INPUT_CHARS", 120000)
QWEN_CHUNK_CHARS = _int("QWEN_CHUNK_CHARS", 35000)
MAX_NEW_TOKENS = _int("MAX_NEW_TOKENS", 8192)
TEMPERATURE = _float("TEMPERATURE", 0.0)
DO_SAMPLE = _bool("DO_SAMPLE", False)

# program_autofill fills ONE form and the backend waits synchronously, so it
# gets a single Qwen pass and a smaller token budget.
AUTOFILL_MAX_INPUT_CHARS = _int("AUTOFILL_MAX_INPUT_CHARS", 35000)
AUTOFILL_MAX_NEW_TOKENS = _int("AUTOFILL_MAX_NEW_TOKENS", 2048)

# ---------------------------------------------------------------- input files
DEFAULT_RUN_TYPE = os.getenv("DEFAULT_RUN_TYPE", "program_autofill")
MAX_FILES = _int("MAX_FILES", 5)
MAX_FILE_BYTES = _int("MAX_FILE_BYTES", 25 * 1024 * 1024)
DOWNLOAD_TIMEOUT = _int("DOWNLOAD_TIMEOUT", 60)
MAX_SHEET_ROWS = _int("MAX_SHEET_ROWS", 3000)

# ---------------------------------------------------------------- PDF / OCR
ENABLE_OCR = _bool("ENABLE_OCR", True)
PDF_FORCE_OCR = _bool("PDF_FORCE_OCR", False)
PDF_MIN_CHARS_PER_PAGE = _int("PDF_MIN_CHARS_PER_PAGE", 40)
PDF_OCR_DPI = _int("PDF_OCR_DPI", 200)
MAX_PDF_PAGES = _int("MAX_PDF_PAGES", 200)

# ---------------------------------------------------------------- logging
LOG_TEXT_PREVIEW_CHARS = _int("LOG_TEXT_PREVIEW_CHARS", 1500)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

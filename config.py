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
_VOLUME = "/runpod-volume"


def _writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


# A real network volume is used for caches only if it is writable. RunPod's
# "Cached model" feature also mounts under /runpod-volume, but read-only.
if os.path.isdir(_VOLUME) and _writable(f"{_VOLUME}/huggingface"):
    os.environ.setdefault("HF_HOME", f"{_VOLUME}/huggingface")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", f"{_VOLUME}/paddlex")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.pop("TRANSFORMERS_CACHE", None)  # deprecated; HF_HOME is enough


# ---------------------------------------------------------------- models
def _runpod_cached_snapshot(repo_id):
    """Local snapshot dir of a model added in the endpoint's "Cached model" field."""
    base = os.path.join(_VOLUME, "huggingface-cache", "hub", "models--" + repo_id.replace("/", "--"))
    ref = os.path.join(base, "refs", "main")
    candidates = []
    if os.path.isfile(ref):
        with open(ref) as f:
            candidates.append(os.path.join(base, "snapshots", f.read().strip()))
    snapshots = os.path.join(base, "snapshots")
    if os.path.isdir(snapshots):
        candidates += sorted(
            (os.path.join(snapshots, d) for d in os.listdir(snapshots)),
            key=os.path.getmtime, reverse=True,
        )
    return next((c for c in candidates if os.path.isfile(os.path.join(c, "config.json"))), None)


def _resolve_model(repo_id):
    if os.path.isdir(repo_id):
        return repo_id
    baked = "/models/" + repo_id.split("/")[-1]  # filled when built with BAKE_QWEN=1
    if os.path.isfile(os.path.join(baked, "config.json")):
        return baked
    return _runpod_cached_snapshot(repo_id) or repo_id


QWEN_REPO_ID = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen3-8B")
QWEN_MODEL_NAME = _resolve_model(QWEN_REPO_ID)
LOAD_IN_4BIT = _bool("LOAD_IN_4BIT", True)
PRELOAD_MODELS = _bool("PRELOAD_MODELS", True)
STRICT_GROUNDING = _bool("STRICT_GROUNDING", True)
APP_VERSION = os.getenv("APP_VERSION", "strict-grounded-v6")

# ---------------------------------------------------------------- generation
MAX_INPUT_CHARS = _int("MAX_INPUT_CHARS", 0)  # 0 = do not truncate program_extract
QWEN_CHUNK_CHARS = _int("QWEN_CHUNK_CHARS", 35000)
MAX_NEW_TOKENS = _int("MAX_NEW_TOKENS", 8192)
TEMPERATURE = _float("TEMPERATURE", 0.0)
DO_SAMPLE = _bool("DO_SAMPLE", False)

# program_autofill fills ONE form and the backend waits synchronously, so it
# gets a single Qwen pass and a smaller token budget.
AUTOFILL_MAX_INPUT_CHARS = _int("AUTOFILL_MAX_INPUT_CHARS", 35000)
AUTOFILL_MAX_NEW_TOKENS = _int("AUTOFILL_MAX_NEW_TOKENS", 2048)

# ---------------------------------------------------------------- input files
DEFAULT_RUN_TYPE = os.getenv("DEFAULT_RUN_TYPE", "program_extract")
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

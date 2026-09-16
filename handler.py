import config  # noqa: F401  (must be first: sets cache env vars)

import logging
import traceback

import runpod

from config import PRELOAD_MODELS, APP_VERSION
from pipeline.inputs import InputError
from pipeline.pipeline import preload, process_request

log = logging.getLogger("handler")
log.info("Starting document extractor version=%s", APP_VERSION)


def handler(job):
    try:
        return process_request(job.get("input"))
    except InputError as exc:
        log.warning("Bad input: %s", exc)
        # A dict with "error" marks the RunPod job FAILED, so n8n can return non-2xx.
        return {"error": str(exc), "error_type": "InputError"}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}


if __name__ == "__main__":
    if PRELOAD_MODELS:
        try:
            preload()
        except Exception:
            # Keep the worker alive so each job reports the real error to n8n.
            log.exception("Model preload failed")
    runpod.serverless.start({"handler": handler})

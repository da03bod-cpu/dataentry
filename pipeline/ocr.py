"""Lean OCR wrapper for scanned PDF pages.

V8 deliberately uses the system Tesseract binary instead of PaddleOCR-VL so the
container stays much smaller and RunPod can pull/start it reliably. Digital PDFs
still use PyMuPDF text extraction first; OCR runs only on pages that need it.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("ocr")


def get_ocr_pipeline():
    """Compatibility hook used by preload(); validates the tiny OCR runtime."""
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("Tesseract OCR is not installed in this worker.")
    return exe


def ocr_image(image_path):
    """OCR one rendered page image and return plain text."""
    exe = get_ocr_pipeline()
    lang = os.getenv("OCR_LANG", "ara+eng").strip() or "ara+eng"
    psm = os.getenv("OCR_PSM", "6").strip() or "6"
    timeout = int(os.getenv("OCR_TIMEOUT", "120"))

    cmd = [
        exe,
        str(Path(image_path)),
        "stdout",
        "-l",
        lang,
        "--psm",
        psm,
        "--dpi",
        os.getenv("PDF_OCR_DPI", "200"),
    ]
    log.info("Running Tesseract OCR lang=%s psm=%s", lang, psm)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Tesseract failed ({proc.returncode}): {proc.stderr[-800:]}")
    if proc.stderr.strip():
        log.debug("Tesseract stderr: %s", proc.stderr[-800:])
    return proc.stdout.strip()

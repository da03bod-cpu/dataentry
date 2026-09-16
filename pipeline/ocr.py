"""PaddleOCR-VL wrapper. paddleocr is imported lazily so DOCX/XLSX/TXT requests
(and digital PDFs) keep working even if the OCR stack has a problem."""
import html
import json
import logging
import re
import threading

log = logging.getLogger("ocr")
_pipeline = None
_lock = threading.Lock()


def get_ocr_pipeline():
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from paddleocr import PaddleOCRVL

                log.info("Loading PaddleOCR-VL ...")
                _pipeline = PaddleOCRVL()
                log.info("PaddleOCR-VL loaded")
    return _pipeline


def _html_table_to_text(text):
    if "<t" not in text:
        return text
    text = re.sub(r"</t[dh]\s*>", " | ", text, flags=re.I)
    text = re.sub(r"</tr\s*>|<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def _as_dict(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _markdown_text(result):
    md = getattr(result, "markdown", None)
    if isinstance(md, dict):
        md = md.get("markdown_texts") or md.get("markdown")
    return md if isinstance(md, str) and md.strip() else ""


def _block_text(result):
    raw = _as_dict(getattr(result, "json", result))
    if raw is None:
        return ""
    res = raw.get("res", raw)
    if not isinstance(res, dict):
        return ""
    for key in ("parsing_res_list", "blocks", "ocr_res", "results"):
        blocks = res.get(key)
        if isinstance(blocks, list):
            texts = []
            for block in blocks:
                if isinstance(block, dict):
                    for field in ("block_content", "text", "content"):
                        value = block.get(field)
                        if isinstance(value, str) and value.strip():
                            texts.append(value.strip())
                            break
            return "\n".join(texts)
    rec_texts = res.get("rec_texts")
    if isinstance(rec_texts, list):
        return "\n".join(t for t in rec_texts if isinstance(t, str))
    return ""


def ocr_image(image_path):
    """OCR one page image and return readable text (never a raw JSON dump)."""
    parts = []
    for result in get_ocr_pipeline().predict(str(image_path)):
        text = _block_text(result) or _markdown_text(result)
        if text:
            parts.append(_html_table_to_text(text))
    return "\n".join(parts)

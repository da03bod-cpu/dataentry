"""Request orchestration. Branches on `type`, like the n8n contract:

- program_autofill : ONE object shaped exactly like StoreProgramRequest (the contract response)
- program_extract  : [ {program}, {program}, ..., {project}, ... ]   (default)
- extract_text     : text extracted from the files, no model call (debugging)
"""
import logging
import tempfile
from pathlib import Path

from config import (
    AUTOFILL_MAX_INPUT_CHARS,
    AUTOFILL_MAX_NEW_TOKENS,
    ENABLE_OCR,
    LOG_TEXT_PREVIEW_CHARS,
    MAX_INPUT_CHARS,
    MAX_NEW_TOKENS,
    QWEN_CHUNK_CHARS,
)
from pipeline.extractors import extract_file
from pipeline.inputs import InputError, normalize_request, materialize
from pipeline.validator import FIELDS, clean_programs, filled_fields, merge_programs

log = logging.getLogger("pipeline")

RUN_TYPES = {"program_autofill", "program_extract", "extract_text"}


# ------------------------------------------------------------------ sources
def collect_sources(req):
    """Return (document_text, sources_info, warnings)."""
    sections, sources, warnings = [], [], []

    if req.text:
        sections.append(("USER TEXT", req.text))
        sources.append({"name": None, "kind": "text", "method": "input.text", "chars": len(req.text)})

    with tempfile.TemporaryDirectory() as tmp:
        for spec in req.files:
            path = materialize(spec, tmp)  # InputError here = bad upload, fail loudly
            try:
                result = extract_file(path, spec.name, spec.mime_type)
            except Exception as exc:
                log.exception("Could not read %s", spec.label)
                warnings.append(f"تعذر قراءة الملف {spec.label}: {exc}")
                sources.append({"name": spec.name, "kind": "error", "method": None, "chars": 0})
                continue
            warnings.extend(f"{spec.label}: {w}" for w in result.warnings)
            if not result.text.strip():
                warnings.append(f"لم يُستخرج أي نص من الملف {spec.label}.")
            else:
                sections.append((f"FILE {spec.index}: {spec.name or Path(path).name}", result.text))
            sources.append({"name": spec.name, "kind": result.kind, "method": result.method, "chars": len(result.text)})

    if not sections:
        if not req.files:
            raise InputError("Nothing to process: send 'text' and/or at least one file.")
        raise InputError("No usable text could be extracted from the uploaded file(s). " + " | ".join(warnings))

    if len(sections) == 1 and not req.text:
        document = sections[0][1]  # single file: same layout the LoRA was trained on
    else:
        document = "\n\n".join(f"===== {title} =====\n{body}" for title, body in sections)

    log.info("run_id=%s type=%s sources=%s total_chars=%d", req.run_id, req.type, sources, len(document))
    log.info("DOCUMENT PREVIEW:\n%s", document[:LOG_TEXT_PREVIEW_CHARS])
    return document, sources, warnings


def _truncate(document, limit, warnings):
    if len(document) <= limit:
        return document
    log.warning("document has %d chars; truncating to %d", len(document), limit)
    warnings.append(f"المحتوى طويل ({len(document)} حرف)؛ تمت قراءة أول {limit} حرف فقط.")
    return document[:limit]


def _run_model(document, max_new_tokens, chunk_chars):
    from pipeline.qwen import generate_program_items  # lazy: torch only when needed

    items, meta = generate_program_items(document, max_new_tokens=max_new_tokens, chunk_chars=chunk_chars)
    return merge_programs(clean_programs(items)), meta


# ------------------------------------------------------------------ run types
def _join_notes(*parts):
    text = "\n".join(p.strip() for p in parts if p and p.strip())
    return text or None


def run_program_autofill(req):
    document, sources, warnings = collect_sources(req)
    document = _truncate(document, AUTOFILL_MAX_INPUT_CHARS, warnings)
    programs, meta = _run_model(
        document,
        max_new_tokens=AUTOFILL_MAX_NEW_TOKENS,
        chunk_chars=max(QWEN_CHUNK_CHARS, AUTOFILL_MAX_INPUT_CHARS),  # always one pass
    )

    extra_notes = list(warnings)
    if meta.get("truncated"):
        extra_notes.append("استجابة النموذج كانت طويلة وتم قطعها؛ راجع الحقول قبل الحفظ.")

    if programs:
        # prefer the most complete record; ties keep document order
        best_index = max(range(len(programs)), key=lambda i: (filled_fields(programs[i]), -i))
        best = dict(programs[best_index])
        if len(programs) > 1:
            others = "، ".join(p["name"] for i, p in enumerate(programs) if i != best_index)[:500]
            extra_notes.insert(0, f"تم العثور على {len(programs)} برامج/مشاريع؛ تمت تعبئة النموذج بـ «{best['name']}». عناصر أخرى: {others}")
    else:
        fallback = next((Path(s["name"]).stem for s in sources if s.get("name")), None)
        if not fallback and req.text:
            fallback = req.text.splitlines()[0]
        best = {k: None for k in FIELDS}
        best.update(type="program", name=(fallback or "برنامج جديد")[:255])
        extra_notes.insert(0, "لم يتمكن النموذج من استخراج بيانات البرنامج بثقة؛ الاسم تخميني ويرجى مراجعة الحقول وتعبئتها يدويًا.")

    best["notes"] = _join_notes(best.get("notes"), *extra_notes)
    return {k: best.get(k) for k in FIELDS}  # exact contract keys, nothing else


def run_program_extract(req):
    """Return a JSON array: every program, then every project, each object with
    exactly the 10 contract fields in contract order."""
    document, sources, warnings = collect_sources(req)
    document = _truncate(document, MAX_INPUT_CHARS, warnings)
    programs, meta = _run_model(document, max_new_tokens=MAX_NEW_TOKENS, chunk_chars=QWEN_CHUNK_CHARS)
    warnings.extend(meta.get("chunk_errors", []))
    if meta.get("truncated"):
        warnings.append("model output hit max_new_tokens; the last item(s) may be missing")

    ordered = [p for p in programs if p["type"] == "program"] + [p for p in programs if p["type"] == "project"]
    log.info("run_id=%s extracted %d item(s) (%d programs, %d projects) from %s; warnings=%s",
             req.run_id, len(ordered), sum(p["type"] == "program" for p in ordered),
             sum(p["type"] == "project" for p in ordered), sources, warnings)
    return [{k: p.get(k) for k in FIELDS} for p in ordered]


def run_extract_text(req):
    document, sources, warnings = collect_sources(req)
    return {"text": document, "chars": len(document), "sources": sources, "warnings": warnings}


def process_request(job_input):
    req = normalize_request(job_input)
    if req.type not in RUN_TYPES:
        raise InputError(f"Unsupported type '{req.type}'. Supported: {', '.join(sorted(RUN_TYPES))}.")
    if req.type == "program_autofill":
        return run_program_autofill(req)
    if req.type == "program_extract":
        return run_program_extract(req)
    return run_extract_text(req)


def preload():
    from pipeline.qwen import load_model

    load_model()
    if ENABLE_OCR:
        try:
            from pipeline.ocr import get_ocr_pipeline

            get_ocr_pipeline()
        except Exception:
            log.exception("PaddleOCR-VL preload failed; scanned PDFs will fail until this is fixed")

"""Strict post-generation grounding.

The LLM is only a candidate extractor. Nothing is returned unless the candidate
can be tied back to text that was actually extracted from the uploaded file.
"""
from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger("grounding")

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")
_GENERIC_PREFIX = re.compile(r"^(?:ال)?(?:برنامج|مشروع|المشروع|برنامج|مبادرة|المبادرة|نشاط|النشاط)\s*[:\-–—]?\s*", re.I)
_NUM_RE = re.compile(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MULTIPLIERS = (
    (re.compile(r"مليار|billion|\bbn\b", re.I), 1e9),
    (re.compile(r"مليون|million|\bmn\b|\bm\b", re.I), 1e6),
    (re.compile(r"ألف|الف|آلاف|الاف|thousand|\bk\b", re.I), 1e3),
)


def _norm(value) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).translate(_DIGITS)
    s = _DIACRITICS.sub("", s)
    # Conservative Arabic normalization: enough for OCR variants, without
    # turning unrelated words into matches.
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    s = _PUNCT.sub(" ", s.lower())
    return " ".join(s.split())


def _name_candidates(name: str):
    name = " ".join(str(name or "").split())
    if not name:
        return []
    candidates = [name]
    stripped = _GENERIC_PREFIX.sub("", name).strip()
    if stripped and stripped != name:
        candidates.append(stripped)
    # longest first to avoid matching a tiny accidental fragment
    return sorted(dict.fromkeys(candidates), key=len, reverse=True)


def _contains_token_phrase(haystack_norm: str, needle_norm: str) -> bool:
    hay = haystack_norm.split()
    needle = needle_norm.split()
    if not needle or len(needle) > len(hay):
        return False
    n = len(needle)
    return any(hay[i:i+n] == needle for i in range(len(hay) - n + 1))


def _find_evidence_lines(document: str, name: str, radius: int = 5):
    raw_lines = document.splitlines()
    norm_lines = [_norm(x) for x in raw_lines]
    candidates = [(c, _norm(c)) for c in _name_candidates(name)]

    for candidate_raw, candidate_norm in candidates:
        if not candidate_norm:
            continue
        for idx, line in enumerate(norm_lines):
            if _contains_token_phrase(line, candidate_norm):
                lo, hi = max(0, idx - radius), min(len(raw_lines), idx + radius + 1)
                return candidate_raw, raw_lines[idx], "\n".join(raw_lines[lo:hi])

    # OCR can insert/remove punctuation and split words. For multi-token names,
    # allow a high token-coverage match on ONE line, but never for one-word names.
    for candidate_raw, candidate_norm in candidates:
        tokens = [t for t in candidate_norm.split() if len(t) > 1]
        if len(tokens) < 2:
            continue
        needed = max(2, int(len(tokens) * 0.85 + 0.999))
        for idx, line in enumerate(norm_lines):
            line_tokens = set(line.split())
            hits = sum(1 for t in tokens if t in line_tokens)
            if hits >= needed:
                lo, hi = max(0, idx - radius), min(len(raw_lines), idx + radius + 1)
                return candidate_raw, raw_lines[idx], "\n".join(raw_lines[lo:hi])
    return None, None, None


def _text_is_grounded(value, evidence: str) -> bool:
    if value is None:
        return True
    value_n = _norm(value)
    if not value_n:
        return True
    evidence_n = _norm(evidence)
    return _contains_token_phrase(evidence_n, value_n)


def _numbers(text: str):
    s = unicodedata.normalize("NFKC", text or "").translate(_DIGITS)
    s = s.replace("٬", ",").replace("٫", ".")
    out = []
    for m in _NUM_RE.finditer(s):
        raw = re.sub(r"[, ]", "", m.group(0))
        try:
            number = float(raw)
        except ValueError:
            continue
        rest = s[m.end():m.end() + 20]
        for pattern, factor in _MULTIPLIERS:
            if pattern.search(rest):
                number *= factor
                break
        out.append(number)
    return out


def _number_is_grounded(value, evidence: str) -> bool:
    if value is None:
        return True
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    for n in _numbers(evidence):
        tolerance = max(0.01, abs(target) * 1e-9)
        if abs(n - target) <= tolerance:
            return True
    return False


def _ground_type(current_type, evidence: str):
    e_tokens = set(_norm(evidence).split())
    # Explicit labels and common table headers win over the model's guess.
    has_program = bool(e_tokens.intersection({"برنامج", "البرنامج", "برامج"}))
    has_project = bool(e_tokens.intersection({"مشروع", "المشروع", "مشاريع", "المشاريع", "مبادرة", "المبادرة", "نشاط", "النشاط"}))
    if has_program and not has_project:
        return "program"
    if has_project and not has_program:
        return "project"
    # If both occur in a wide evidence window, only keep the model type when its
    # own Arabic/English label is actually represented nearby.
    if current_type == "program" and has_program:
        return "program"
    if current_type == "project" and has_project:
        return "project"
    return None


def ground_programs(programs, document: str):
    """Return only records supported by the extracted document text.

    - Missing/unsupported name => drop the whole record.
    - Unsupported non-key field => replace with null, never invent.
    - Unsupported type => drop the record because the contract requires a type.
    """
    grounded = []
    dropped = 0
    nulled = 0

    for p in programs or []:
        matched_name, record_line, context = _find_evidence_lines(document, p.get("name"))
        if not record_line:
            dropped += 1
            log.warning("Dropped ungrounded item name=%r", p.get("name"))
            continue

        item = dict(p)
        # If the model added a generic prefix not present in the source, use the
        # grounded candidate instead of returning invented wording.
        item["name"] = matched_name

        grounded_type = _ground_type(item.get("type"), context)
        if not grounded_type:
            dropped += 1
            log.warning("Dropped item with ungrounded type name=%r type=%r", item.get("name"), item.get("type"))
            continue
        item["type"] = grounded_type

        for key in ("year", "beneficiaries_count", "budget"):
            if item.get(key) is not None and not _number_is_grounded(item[key], record_line):
                log.warning("Nulling ungrounded numeric field %s=%r for %r", key, item[key], item.get("name"))
                item[key] = None
                nulled += 1

        for key in ("description", "beneficiary_value", "target_audience", "delivery_method", "notes"):
            if item.get(key) is not None and not _text_is_grounded(item[key], record_line):
                log.warning("Nulling ungrounded text field %s=%r for %r", key, item[key], item.get("name"))
                item[key] = None
                nulled += 1

        grounded.append(item)

    log.info("grounding: kept=%d dropped=%d nulled_fields=%d", len(grounded), dropped, nulled)
    return grounded, {"grounded_kept": len(grounded), "grounded_dropped": dropped, "grounded_nulled_fields": nulled}

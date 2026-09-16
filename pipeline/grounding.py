"""Strict post-generation grounding.

The LLM is only a candidate extractor. Nothing is returned unless the candidate
can be tied back to text that was actually extracted from the uploaded file.

v5 changes:
- prefer the *detailed* occurrence of an item over a table-of-contents occurrence
- ground fields against the bounded item section, not only the title line
- validate numeric fields against their semantic labels when possible
- allow a document-level year only when that exact year is present in the file
"""
from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger("grounding")

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")
_GENERIC_PREFIX = re.compile(r"^(?:ال)?(?:برنامج|مشروع|المشروع|مبادرة|المبادرة|نشاط|النشاط)\s*[:\-–—]?\s*", re.I)
_NUM_RE = re.compile(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_MULTIPLIERS = (
    (re.compile(r"(?<!\w)(?:مليار|billion|bn)(?!\w)", re.I), 1e9),
    (re.compile(r"(?<!\w)(?:مليون|million|mn|m)(?!\w)", re.I), 1e6),
    (re.compile(r"(?<!\w)(?:ألف|الف|آلاف|الاف|thousand|k)(?!\w)", re.I), 1e3),
)

_FIELD_HINTS = (
    "عدد المستفيدين", "عدد المستفيدات", "المستفيدين", "المستفيدات",
    "الفئة المستهدفة", "مكان التنفيذ", "آلية التنفيذ", "طريقة التنفيذ",
    "الميزانية", "الموازنة", "التكلفة", "عدد المتطوعين", "ساعات التطوع",
    "عدد الهدايا", "عدد الوجبات", "عدد الحملات",
)

_BENEFICIARY_LABELS = (
    "عدد المستفيدين", "عدد المستفيدات", "إجمالي المستفيدين", "اجمالي المستفيدين",
    "المستفيدين", "المستفيدات",
)
_BUDGET_LABELS = (
    "الميزانية", "الموازنة", "التكلفة", "قيمة المشروع", "تكلفة المشروع",
    "إجمالي التكلفة", "اجمالي التكلفة", "budget", "cost",
)


def _norm(value) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).translate(_DIGITS)
    s = _DIACRITICS.sub("", s)
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
    return sorted(dict.fromkeys(candidates), key=len, reverse=True)


def _contains_token_phrase(haystack_norm: str, needle_norm: str) -> bool:
    hay = haystack_norm.split()
    needle = needle_norm.split()
    if not needle or len(needle) > len(hay):
        return False
    n = len(needle)
    return any(hay[i:i+n] == needle for i in range(len(hay) - n + 1))


def _line_matches_name(line_norm: str, name: str) -> bool:
    for candidate in _name_candidates(name):
        c = _norm(candidate)
        if c and _contains_token_phrase(line_norm, c):
            return True
    return False


def _occurrence_score(raw_lines, norm_lines, idx: int) -> int:
    # Detailed sections normally have field labels shortly after the heading.
    after = "\n".join(raw_lines[idx:min(len(raw_lines), idx + 20)])
    after_n = _norm(after)
    score = 0
    score += 6 * sum(1 for h in _FIELD_HINTS if _norm(h) in after_n)

    raw = raw_lines[idx].strip()
    # TOC/list entries commonly begin with page/sequence numbers.
    if re.match(r"^\d+\s+(?:\d+[\.\-]?\s*)?(?:مبادرة|مشروع|برنامج|مسابقة|دورة|ملتقى|الدليل)", raw):
        score -= 12
    if re.match(r"^\d+\s+\d+[\.]", raw):
        score -= 8
    # Prefer later occurrence when scores tie (summary first, detail later is common).
    score += min(idx, 10000) // 1000
    return score


def _find_best_occurrence(document: str, name: str):
    raw_lines = document.splitlines()
    norm_lines = [_norm(x) for x in raw_lines]
    hits = []

    for idx, line in enumerate(norm_lines):
        if _line_matches_name(line, name):
            hits.append(idx)

    # OCR fuzzy fallback for multi-token names only.
    if not hits:
        for candidate in _name_candidates(name):
            tokens = [t for t in _norm(candidate).split() if len(t) > 1]
            if len(tokens) < 2:
                continue
            needed = max(2, int(len(tokens) * 0.85 + 0.999))
            for idx, line in enumerate(norm_lines):
                line_tokens = set(line.split())
                if sum(1 for t in tokens if t in line_tokens) >= needed:
                    hits.append(idx)

    if not hits:
        return None, None, None, raw_lines, norm_lines

    best_idx = max(dict.fromkeys(hits), key=lambda i: (_occurrence_score(raw_lines, norm_lines, i), i))
    # Return the candidate wording actually represented by the matched line.
    matched_name = name
    for candidate in _name_candidates(name):
        if _contains_token_phrase(norm_lines[best_idx], _norm(candidate)):
            matched_name = candidate
            break
    return matched_name, raw_lines[best_idx], best_idx, raw_lines, norm_lines


def _build_record_context(raw_lines, norm_lines, start_idx: int, all_names, max_lines: int = 35):
    """Bound one item section without leaking facts from the next item.

    We start at the best detailed title occurrence and stop at the next known
    item title. A small amount of preceding context is included for table headers.
    """
    lo = max(0, start_idx - 2)
    hi = min(len(raw_lines), start_idx + max_lines)

    for j in range(start_idx + 1, hi):
        line_n = norm_lines[j]
        if not line_n:
            continue
        if any(_line_matches_name(line_n, other) for other in all_names if _norm(other) != _norm(all_names[0] if all_names else "")):
            # This broad check is refined below by verifying the line doesn't also
            # match the current title.  It protects short adjacent project sections.
            current_line = norm_lines[start_idx]
            if line_n != current_line:
                hi = j
                break

    return "\n".join(raw_lines[lo:hi])


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


def _number_equals(value, evidence: str) -> bool:
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    for n in _numbers(evidence):
        tolerance = max(0.01, abs(target) * 1e-9)
        if abs(n - target) <= tolerance:
            return True
    return False




def _semantic_numbers_after_labels(evidence: str, labels, max_follow_lines: int = 3):
    """Extract numbers semantically attached to a field label.

    Handles common OCR layouts such as:
      عدد المستفيدين\n415900
      الفئة المستهدفة عدد المستفيدين\nضيوف الرحمن 480
      عدد المستفيدين\n3 4حملة مستفيدة   -> 34 (OCR split)
    and simple pipe-delimited tables by using the matching column.
    """
    lines = evidence.splitlines()
    norm_lines = [_norm(x) for x in lines]
    label_norms = [_norm(x) for x in labels]
    values = []

    def add(nums):
        for n in nums:
            if n not in values:
                values.append(n)

    for i, line_n in enumerate(norm_lines):
        matching = [lbl for lbl in label_norms if lbl and lbl in line_n]
        if not matching:
            continue

        # Pipe table: map header label to the corresponding cell in the next row.
        if "|" in lines[i]:
            header = [c.strip() for c in lines[i].split("|")]
            header_n = [_norm(c) for c in header]
            col = next((k for k, cell in enumerate(header_n) if any(lbl in cell for lbl in label_norms)), None)
            if col is not None:
                for j in range(i + 1, min(len(lines), i + max_follow_lines + 2)):
                    if "|" not in lines[j]:
                        continue
                    cells = [c.strip() for c in lines[j].split("|")]
                    if col < len(cells):
                        add(_numbers(cells[col]))
                        break
                continue

        # Number on same line after the label.
        raw = unicodedata.normalize("NFKC", lines[i]).translate(_DIGITS)
        pos = min((raw.find(lbl_raw) for lbl_raw in labels if raw.find(lbl_raw) >= 0), default=-1)
        if pos >= 0:
            tail = raw[pos:]
            add(_numbers(tail))

        # Follow a few lines, stopping at the next obvious field label.
        for j in range(i + 1, min(len(lines), i + max_follow_lines + 1)):
            next_n = norm_lines[j]
            if j > i + 1 and any(_norm(h) in next_n for h in _FIELD_HINTS):
                break
            raw_next = unicodedata.normalize("NFKC", lines[j]).translate(_DIGITS).strip()

            # OCR sometimes splits a two-digit value immediately before Arabic text: "3 4حملة".
            m = re.match(r"^(\d)\s+(\d)(?=\s*[\u0600-\u06FF])", raw_next)
            if m:
                add([float(m.group(1) + m.group(2))])
                continue
            add(_numbers(raw_next))

    return values


def _unique_document_year(document: str):
    years = []
    for n in _numbers(document):
        if float(n).is_integer() and 1900 <= int(n) <= 2100:
            y = int(n)
            if y not in years:
                years.append(y)
    return years[0] if len(years) == 1 else None

def _number_near_labels(value, evidence: str, labels, lookahead_lines: int = 5) -> bool:
    if value is None:
        return True
    values = _semantic_numbers_after_labels(evidence, labels, max_follow_lines=lookahead_lines)
    try:
        target = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(float(v) - target) <= max(0.01, abs(target) * 1e-9) for v in values)


def _ground_type(current_type, evidence: str):
    e_tokens = set(_norm(evidence).split())
    has_program = bool(e_tokens.intersection({"برنامج", "البرنامج", "برامج"}))
    has_project = bool(e_tokens.intersection({"مشروع", "المشروع", "مشاريع", "المشاريع", "مبادرة", "المبادرة", "نشاط", "النشاط", "مسابقة", "دورة", "ملتقى", "دليل", "الدليل"}))
    if has_program and not has_project:
        return "program"
    if has_project and not has_program:
        return "project"
    if current_type == "program" and has_program:
        return "program"
    if current_type == "project" and has_project:
        return "project"
    return None


def ground_programs(programs, document: str):
    """Return only records supported by the current extracted document."""
    grounded = []
    dropped = 0
    nulled = 0
    names = [p.get("name") for p in (programs or []) if p.get("name")]

    for p in programs or []:
        matched_name, record_line, start_idx, raw_lines, norm_lines = _find_best_occurrence(document, p.get("name"))
        if start_idx is None:
            dropped += 1
            log.warning("Dropped ungrounded item name=%r", p.get("name"))
            continue

        # Build context, stopping at the next *different* known item title.
        lo = max(0, start_idx - 2)
        hi = min(len(raw_lines), start_idx + 35)
        current_norm = _norm(p.get("name"))
        for j in range(start_idx + 1, hi):
            line_n = norm_lines[j]
            for other in names:
                if _norm(other) == current_norm:
                    continue
                if _line_matches_name(line_n, other):
                    hi = j
                    break
            else:
                continue
            break
        context = "\n".join(raw_lines[lo:hi])

        item = dict(p)
        item["name"] = matched_name

        grounded_type = _ground_type(item.get("type"), context)
        if not grounded_type:
            dropped += 1
            log.warning("Dropped item with ungrounded type name=%r type=%r", item.get("name"), item.get("type"))
            continue
        item["type"] = grounded_type

        # Year is allowed as a document-level fact. If the model omitted it and the
        # file contains one unique Gregorian report year, fill it deterministically.
        doc_year = _unique_document_year(document)
        if item.get("year") is not None:
            if not _number_equals(item["year"], document):
                item["year"] = None
                nulled += 1
        elif doc_year is not None:
            item["year"] = doc_year

        # Numeric fields are repaired only from values semantically attached to
        # their labels inside this exact item section.
        ben_values = _semantic_numbers_after_labels(context, _BENEFICIARY_LABELS)
        if len(ben_values) == 1:
            grounded_ben = int(ben_values[0]) if float(ben_values[0]).is_integer() else ben_values[0]
            if item.get("beneficiaries_count") != grounded_ben:
                if item.get("beneficiaries_count") is not None:
                    log.warning("Repairing beneficiaries_count %r -> %r for %r", item.get("beneficiaries_count"), grounded_ben, item.get("name"))
                item["beneficiaries_count"] = grounded_ben
        elif item.get("beneficiaries_count") is not None and not _number_near_labels(
            item["beneficiaries_count"], context, _BENEFICIARY_LABELS
        ):
            log.warning("Nulling ungrounded beneficiaries_count=%r for %r", item["beneficiaries_count"], item.get("name"))
            item["beneficiaries_count"] = None
            nulled += 1

        budget_values = _semantic_numbers_after_labels(context, _BUDGET_LABELS)
        if len(budget_values) == 1:
            grounded_budget = float(budget_values[0])
            if item.get("budget") != grounded_budget:
                item["budget"] = grounded_budget
        elif item.get("budget") is not None and not _number_near_labels(item["budget"], context, _BUDGET_LABELS):
            log.warning("Nulling ungrounded budget=%r for %r", item["budget"], item.get("name"))
            item["budget"] = None
            nulled += 1

        for key in ("description", "beneficiary_value", "target_audience", "delivery_method", "notes"):
            if item.get(key) is not None and not _text_is_grounded(item[key], context):
                log.warning("Nulling ungrounded text field %s=%r for %r", key, item[key], item.get("name"))
                item[key] = None
                nulled += 1

        grounded.append(item)

    log.info("grounding: kept=%d dropped=%d nulled_fields=%d", len(grounded), dropped, nulled)
    return grounded, {"grounded_kept": len(grounded), "grounded_dropped": dropped, "grounded_nulled_fields": nulled}

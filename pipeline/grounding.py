"""Strict post-generation grounding and structural recovery.

The LLM is a candidate extractor, not the source of truth.  Every returned
record must be supported by text extracted from the *current* upload.

v6 adds a deterministic structure pass for report-style documents:
- recover top-level Arabic section headings (programs)
- recover repeated initiative/project headings from TOC + detail pages
- preserve canonical heading names, including prefixes such as "مبادرة"
- force type from the heading itself instead of words later in the section
- prefer multi-line detailed headings over table-of-contents occurrences
- fill exact descriptions from the bounded item section when the model omits them
- repair labelled beneficiary/budget values, including simple OCR digit splits

For unstructured spreadsheets/text files the old candidate-grounding path is
kept, so this does not require a TOC/report layout.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import OrderedDict

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
_TARGET_LABELS = ("الفئة المستهدفة", "الفئات المستهدفة", "الفئة المستفيدة")
_DELIVERY_LABELS = ("مكان التنفيذ", "موقع التنفيذ", "آلية التنفيذ", "طريقة التنفيذ")

# Normalized Arabic ordinals commonly used as report section headings.
_ORDINALS = {
    "اولا", "اولا", "اولاً", "اول", "ثانيا", "ثانيًا", "ثالثا", "ثالثًا",
    "رابعا", "رابعًا", "خامسا", "خامسًا", "سادسا", "سادسًا", "سابعا",
    "سابعًا", "ثامنا", "ثامنًا", "تاسعا", "تاسعًا", "عاشرا", "عاشرًا",
}
_ORDINALS_NORM = set()
_ORDINAL_RANK = {
    "اولا": 1, "اول": 1, "ثانيا": 2, "ثالثا": 3, "رابعا": 4,
    "خامسا": 5, "سادسا": 6, "سابعا": 7, "ثامنا": 8, "تاسعا": 9, "عاشرا": 10,
}
_PROJECT_PREFIXES = ("مبادرة", "مشروع", "المشروع", "مسابقة", "دورة", "ملتقى", "الدليل")
_GENERIC_PROGRAM_HEADINGS = {
    "البرامج والمبادرات", "برامج ومبادرات", "برامج ومشاريع", "البرامج والمشاريع",
}


def _norm(value) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).translate(_DIGITS)
    s = _DIACRITICS.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    s = _PUNCT.sub(" ", s.lower())
    return " ".join(s.split())


_ORDINALS_NORM.update(_norm(x) for x in _ORDINALS)


def _name_candidates(name: str):
    name = " ".join(str(name or "").split())
    if not name:
        return []
    candidates = [name]
    stripped = _GENERIC_PREFIX.sub("", name).strip()
    if stripped and stripped != name:
        candidates.append(stripped)
    return sorted(dict.fromkeys(candidates), key=len, reverse=True)


def _name_key(name: str) -> str:
    """Loose key used only to match a model candidate to a document heading."""
    n = _norm(name)
    toks = n.split()
    if toks and toks[0] in {_norm(x) for x in _PROJECT_PREFIXES}:
        toks = toks[1:]
    return " ".join(toks)


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


def _window_norm(norm_lines, idx: int, span: int = 3) -> str:
    return " ".join(x for x in norm_lines[idx:min(len(norm_lines), idx + span)] if x)


def _window_matches_name(norm_lines, idx: int, name: str, span: int = 3) -> bool:
    return _line_matches_name(_window_norm(norm_lines, idx, span), name)


def _strip_toc_numbering(raw: str) -> str:
    s = raw.strip().lstrip("*•-–— ")
    # Page number + item number, e.g. "16 1.مبادرة ..." or "22 .1 مبادرة ..."
    s = re.sub(r"^\d+\s+(?:[.\-]?\s*)?", "", s)
    s = re.sub(r"^\.?\s*\d+\s*[.\-:]?\s*", "", s)
    return s.strip()


def _canonical_project_title(raw_lines, idx: int):
    raw = _strip_toc_numbering(raw_lines[idx])
    n = _norm(raw)
    if not n:
        return None
    tokens = n.split()
    first = tokens[0]
    prefix_norms = {_norm(x) for x in _PROJECT_PREFIXES}
    if first not in prefix_norms:
        return None

    # Two-column TOCs can concatenate two independent item headings on one line.
    # Never treat that whole combined line as one project title.
    if sum(tok in prefix_norms for tok in tokens) > 1:
        return None

    # Reject obvious prose that happens to start with a marker.
    if n.startswith("ملتقي يهدف ") or n.startswith("دورة تدريبية ") or n.startswith("دوره تدريبيه "):
        return None

    title = " ".join(raw.split())
    # A heading may be split across two lines, especially parenthetical subtitles.
    if idx + 1 < len(raw_lines):
        nxt = " ".join(raw_lines[idx + 1].strip().split())
        if nxt.startswith("(") and ")" in nxt and len(nxt) <= 100 and "(" not in title:
            title = f"{title} {nxt}"
    if len(title) > 180:
        return None
    return title


def _canonical_program_title(raw: str):
    stripped = raw.strip().lstrip("*•-–— ")
    n = _norm(stripped)
    if not n:
        return None
    toks = n.split()
    if not toks or toks[0] not in _ORDINALS_NORM:
        return None

    # OCR of two-column TOCs can concatenate two section headings onto one line.
    if sum(tok in _ORDINALS_NORM for tok in toks) > 1:
        return None

    if ":" in stripped:
        title = stripped.split(":", 1)[1].strip()
    elif "：" in stripped:
        title = stripped.split("：", 1)[1].strip()
    else:
        # Remove the first raw token and punctuation immediately following it.
        parts = stripped.split(maxsplit=1)
        title = parts[1] if len(parts) == 2 else ""
        title = title.lstrip(":：.-–— ")

    title = " ".join(title.split())
    if not title or len(title) > 160:
        return None
    if _norm(title) in {_norm(x) for x in _GENERIC_PROGRAM_HEADINGS}:
        return None
    return title


def _occurrence_score(raw_lines, norm_lines, idx: int) -> int:
    after = "\n".join(raw_lines[idx:min(len(raw_lines), idx + 20)])
    after_n = _norm(after)
    score = 0
    score += 6 * sum(1 for h in _FIELD_HINTS if _norm(h) in after_n)
    raw = raw_lines[idx].strip()
    if re.match(r"^\d+\s+(?:\d+[.\-]?\s*)?(?:مبادرة|مشروع|برنامج|مسابقة|دورة|ملتقى|الدليل)", raw):
        score -= 12
    if re.match(r"^\d+\s+\d+[.]", raw):
        score -= 8
    score += min(idx, 10000) // 1000
    return score


def _find_best_occurrence(document: str, name: str):
    raw_lines = document.splitlines()
    norm_lines = [_norm(x) for x in raw_lines]
    hits = []

    name_tokens = [t for t in _norm(name).split() if len(t) > 1]
    for idx in range(len(norm_lines)):
        if _line_matches_name(norm_lines[idx], name):
            hits.append(idx)
            continue
        if _window_matches_name(norm_lines, idx, name, span=3):
            # Multi-line headings are allowed only when the first line already
            # contains meaningful title tokens. This prevents a table header on
            # the previous line from stealing the occurrence.
            line_tokens = set(norm_lines[idx].split())
            needed = min(2, len(name_tokens))
            if needed and sum(1 for t in name_tokens if t in line_tokens) >= needed:
                hits.append(idx)

    # OCR fuzzy fallback for multi-token names only.
    if not hits:
        for candidate in _name_candidates(name):
            tokens = [t for t in _norm(candidate).split() if len(t) > 1]
            if len(tokens) < 2:
                continue
            needed = max(2, int(len(tokens) * 0.75 + 0.999))
            for idx in range(len(norm_lines)):
                line_tokens = set(_window_norm(norm_lines, idx, 3).split())
                if sum(1 for t in tokens if t in line_tokens) >= needed:
                    hits.append(idx)

    if not hits:
        return None, None, None, raw_lines, norm_lines

    best_idx = max(dict.fromkeys(hits), key=lambda i: (_occurrence_score(raw_lines, norm_lines, i), i))
    return name, raw_lines[best_idx], best_idx, raw_lines, norm_lines


def _discover_structure(document: str):
    """Discover report structure without trusting the model.

    Structured mode is intentionally conservative: it activates only when there
    are at least two ordinal section headings and at least three project headings
    repeated in the document (typically TOC + detail page).  That prevents prose
    in arbitrary text/spreadsheets from being misread as a hierarchy.
    """
    raw_lines = document.splitlines()
    norm_lines = [_norm(x) for x in raw_lines]

    programs = OrderedDict()
    program_meta = {}
    for i, raw in enumerate(raw_lines):
        title = _canonical_program_title(raw)
        if title:
            key = _norm(title)
            first_tok = _norm(raw.strip().lstrip("*•-–— ")).split()[:1]
            rank = _ORDINAL_RANK.get(first_tok[0], 999) if first_tok else 999
            if key not in program_meta or (rank, i) < (program_meta[key][0], program_meta[key][1]):
                program_meta[key] = (rank, i, title)
    for key, (_rank, _idx, title) in sorted(program_meta.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        programs[key] = title

    project_occ = OrderedDict()
    for i in range(len(raw_lines)):
        title = _canonical_project_title(raw_lines, i)
        if not title:
            continue
        key = _name_key(title)
        bucket = project_occ.setdefault(key, {"title": title, "indexes": []})
        bucket["indexes"].append(i)
        # Prefer the more complete title variant (useful for parenthetical subtitles).
        if len(_norm(title)) > len(_norm(bucket["title"])):
            bucket["title"] = title

    repeated = []
    for bucket in project_occ.values():
        # Exact repeated heading or a high-confidence detailed heading with fields.
        uniq = list(dict.fromkeys(bucket["indexes"]))
        detail_score = max((_occurrence_score(raw_lines, norm_lines, i) for i in uniq), default=0)
        if len(uniq) >= 2 or detail_score >= 18:
            repeated.append(bucket["title"])

    structured = len(programs) >= 2 and len(repeated) >= 3
    if not structured:
        return [], [], False
    return list(programs.values()), repeated, True


def _blank_record(name: str, typ: str):
    return {
        "type": typ, "name": name, "year": None, "description": None,
        "beneficiaries_count": None, "budget": None, "beneficiary_value": None,
        "target_audience": None, "delivery_method": None, "notes": None,
    }


def _best_candidate(candidates, title):
    key = _name_key(title)
    matches = [p for p in candidates if _name_key(p.get("name")) == key]
    if not matches:
        return None
    return max(matches, key=lambda p: sum(v is not None for k, v in p.items() if k not in {"type", "name"}))


def _augment_from_structure(programs, document: str):
    section_titles, project_titles, structured = _discover_structure(document)
    if not structured:
        return list(programs or []), False

    out = []
    for title in section_titles:
        base = dict(_best_candidate(programs or [], title) or _blank_record(title, "program"))
        base["name"] = title
        base["type"] = "program"
        out.append(base)
    for title in project_titles:
        base = dict(_best_candidate(programs or [], title) or _blank_record(title, "project"))
        base["name"] = title
        base["type"] = "project"
        out.append(base)

    log.info("structural recovery: programs=%d projects=%d (model_candidates=%d)",
             len(section_titles), len(project_titles), len(programs or []))
    return out, True


def _text_is_grounded(value, evidence: str) -> bool:
    if value is None:
        return True
    value_n = _norm(value)
    if not value_n:
        return True
    return _contains_token_phrase(_norm(evidence), value_n)


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
    return any(abs(n - target) <= max(0.01, abs(target) * 1e-9) for n in _numbers(evidence))


def _semantic_numbers_after_labels(evidence: str, labels, max_follow_lines: int = 3):
    lines = evidence.splitlines()
    norm_lines = [_norm(x) for x in lines]
    label_norms = [_norm(x) for x in labels]
    values = []

    def add(nums):
        for n in nums:
            if n not in values:
                values.append(n)

    for i, line_n in enumerate(norm_lines):
        if not any(lbl and lbl in line_n for lbl in label_norms):
            continue

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

        raw = unicodedata.normalize("NFKC", lines[i]).translate(_DIGITS)
        # Same-line values are accepted only after the label text.
        positions = []
        for original in labels:
            pos = raw.find(original)
            if pos >= 0:
                positions.append(pos + len(original))
        if positions:
            add(_numbers(raw[min(positions):]))

        for j in range(i + 1, min(len(lines), i + max_follow_lines + 1)):
            next_n = norm_lines[j]
            if j > i + 1 and any(_norm(h) in next_n for h in _FIELD_HINTS):
                break
            raw_next = unicodedata.normalize("NFKC", lines[j]).translate(_DIGITS).strip()
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


def _canonical_name_from_occurrence(raw_lines, idx: int, fallback: str):
    proj = _canonical_project_title(raw_lines, idx)
    if proj and _name_key(proj) == _name_key(fallback):
        return proj
    prog = _canonical_program_title(raw_lines[idx])
    if prog and _name_key(prog) == _name_key(fallback):
        return prog
    return fallback


def _ground_type(current_type, raw_lines, idx: int, canonical_name: str):
    proj = _canonical_project_title(raw_lines, idx)
    if proj and _name_key(proj) == _name_key(canonical_name):
        return "project"
    prog = _canonical_program_title(raw_lines[idx])
    if prog and _name_key(prog) == _name_key(canonical_name):
        return "program"
    # Unstructured fallback: only trust explicit words on/around the title line.
    title_n = _window_norm([_norm(x) for x in raw_lines], idx, 2)
    first = _norm(canonical_name).split()[:1]
    if first and first[0] in {_norm(x) for x in _PROJECT_PREFIXES}:
        return "project"
    if _norm(canonical_name).startswith("برنامج ") or _norm(canonical_name).startswith("برامج "):
        return "program"
    if current_type in {"program", "project"} and _line_matches_name(title_n, canonical_name):
        return current_type
    return None


def _starts_name_window(norm_lines, idx: int, name: str, span: int = 3) -> bool:
    if _line_matches_name(norm_lines[idx], name):
        return True
    if not _window_matches_name(norm_lines, idx, name, span=span):
        return False
    tokens = [t for t in _norm(name).split() if len(t) > 1]
    line_tokens = set(norm_lines[idx].split())
    needed = min(2, len(tokens))
    return bool(needed and sum(1 for t in tokens if t in line_tokens) >= needed)


def _context_bounds(raw_lines, norm_lines, start_idx: int, all_names, max_lines: int = 45):
    # Include a couple of preceding lines so a table row can inherit its column
    # headers, while description inference still starts exactly at start_idx.
    lo = max(0, start_idx - 2)
    hi = min(len(raw_lines), start_idx + max_lines)
    current_key = _name_key(all_names[0]) if all_names else ""
    for j in range(start_idx + 1, hi):
        # Report page footer is a reliable hard boundary in extracted annual reports.
        if norm_lines[j] == "التقرير" and j > start_idx + 1:
            hi = j
            return lo, hi
        for other in all_names[1:]:
            if _name_key(other) == current_key:
                continue
            if _starts_name_window(norm_lines, j, other, span=3):
                hi = j
                return lo, hi
    return lo, hi


def _infer_description(raw_lines, start_idx: int, hi: int, canonical_name: str):
    out = []
    j = start_idx + 1
    # Parenthetical subtitle may be the second line of the heading.
    if j < hi:
        nxt = " ".join(raw_lines[j].strip().split())
        if nxt.startswith("(") and ")" in nxt and _norm(nxt) in _norm(canonical_name):
            j += 1

    for k in range(j, hi):
        raw = " ".join(raw_lines[k].strip().split())
        if not raw:
            continue
        n = _norm(raw)
        if n == "التقرير" or re.fullmatch(r"ف\s*\d+", n):
            continue
        if any(_norm(h) in n for h in _FIELD_HINTS):
            break
        # Defensive stop if a new structural heading slips past the context bound.
        if k > j and (_canonical_project_title(raw_lines, k) or _canonical_program_title(raw_lines[k])):
            break
        out.append(raw)

    text = " ".join(out).strip()
    return text[:3000] if text else None


def _infer_simple_label_text(context: str, labels):
    """Conservative extraction for simple one-column label/value layouts."""
    lines = context.splitlines()
    norm_lines = [_norm(x) for x in lines]
    label_norms = [_norm(x) for x in labels]
    all_hint_norms = [_norm(h) for h in _FIELD_HINTS]
    for i, line_n in enumerate(norm_lines):
        matching = [lbl for lbl in label_norms if lbl and lbl in line_n]
        if not matching:
            continue
        # Skip multi-column header lines; the LLM can still provide a grounded value.
        other_hints = [h for h in all_hint_norms if h in line_n and h not in matching]
        if other_hints:
            continue
        values = []
        for j in range(i + 1, min(len(lines), i + 4)):
            n = norm_lines[j]
            if any(h in n for h in all_hint_norms):
                break
            raw = " ".join(lines[j].strip().split())
            if not raw or _numbers(raw) and len(_norm(raw).split()) <= 2:
                continue
            values.append(raw)
        if values:
            return " ".join(values)[:1000]
    return None


def ground_programs(programs, document: str):
    """Return only records supported by the current extracted document."""
    candidates, structured = _augment_from_structure(programs, document)
    grounded = []
    dropped = 0
    nulled = 0
    names = [p.get("name") for p in candidates if p.get("name")]
    doc_year = _unique_document_year(document)

    for p in candidates:
        _matched_name, _record_line, start_idx, raw_lines, norm_lines = _find_best_occurrence(document, p.get("name"))
        if start_idx is None:
            dropped += 1
            log.warning("Dropped ungrounded item name=%r", p.get("name"))
            continue

        canonical_name = _canonical_name_from_occurrence(raw_lines, start_idx, p.get("name"))
        # Put current item first for context-bound helper, then all other headings.
        other_names = [canonical_name] + [n for n in names if _name_key(n) != _name_key(canonical_name)]
        lo, hi = _context_bounds(raw_lines, norm_lines, start_idx, other_names)
        context = "\n".join(raw_lines[lo:hi])

        item = dict(p)
        item["name"] = canonical_name

        grounded_type = _ground_type(item.get("type"), raw_lines, start_idx, canonical_name)
        if not grounded_type:
            dropped += 1
            log.warning("Dropped item with ungrounded type name=%r type=%r", item.get("name"), item.get("type"))
            continue
        item["type"] = grounded_type

        if item.get("year") is not None:
            if not _number_equals(item["year"], document):
                item["year"] = None
                nulled += 1
        elif doc_year is not None:
            item["year"] = doc_year

        ben_values = _semantic_numbers_after_labels(context, _BENEFICIARY_LABELS)
        if len(ben_values) == 1:
            grounded_ben = int(ben_values[0]) if float(ben_values[0]).is_integer() else ben_values[0]
            item["beneficiaries_count"] = grounded_ben
        elif item.get("beneficiaries_count") is not None and not _number_near_labels(
            item["beneficiaries_count"], context, _BENEFICIARY_LABELS
        ):
            item["beneficiaries_count"] = None
            nulled += 1

        budget_values = _semantic_numbers_after_labels(context, _BUDGET_LABELS)
        if len(budget_values) == 1:
            item["budget"] = float(budget_values[0])
        elif item.get("budget") is not None and not _number_near_labels(item["budget"], context, _BUDGET_LABELS):
            item["budget"] = None
            nulled += 1

        inferred_description = _infer_description(raw_lines, start_idx, hi, canonical_name)
        if item.get("description") is not None and _text_is_grounded(item["description"], context):
            pass
        elif inferred_description:
            item["description"] = inferred_description
        elif item.get("description") is not None:
            item["description"] = None
            nulled += 1

        # Keep model text fields only when they occur in this exact item section.
        for key in ("beneficiary_value", "target_audience", "delivery_method", "notes"):
            if item.get(key) is not None and not _text_is_grounded(item[key], context):
                item[key] = None
                nulled += 1

        # Conservative deterministic fill for simple label/value layouts.
        if item.get("target_audience") is None:
            item["target_audience"] = _infer_simple_label_text(context, _TARGET_LABELS)
        if item.get("delivery_method") is None:
            item["delivery_method"] = _infer_simple_label_text(context, _DELIVERY_LABELS)

        grounded.append(item)

    log.info("grounding v6: structured=%s kept=%d dropped=%d nulled_fields=%d",
             structured, len(grounded), dropped, nulled)
    return grounded, {
        "grounded_kept": len(grounded),
        "grounded_dropped": dropped,
        "grounded_nulled_fields": nulled,
        "structured_recovery": structured,
    }

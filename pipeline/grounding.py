"""Strict post-generation grounding and structural recovery.

The LLM is a candidate extractor, not the source of truth.  Every returned
record must be supported by text extracted from the *current* upload.

v8.4 generalizes structural recovery across heterogeneous reports and keeps conservative field cleanup:
- reconstruct target_audience + delivery_method from wrapped RTL multi-column rows
- prevent beneficiary_value from duplicating target_audience
- keep all repairs bounded to the current item section

The deterministic structure pass supports report-style documents:
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
    "عدد الهدايا", "عدد الوجبات", "عدد الحملات", "ملاحظات", "ملاحظة",
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
_NOTES_LABELS = ("ملاحظات", "ملاحظة", "ملاحظات المشروع", "ملاحظات البرنامج", "notes")

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
    "البرامج التعليمية", "البرامج", "أبرز البرامج", "ابرز البرامج",
    "قائمة البرامج", "برامجنا", "key programs", "featured programs", "programs",
}

# Generic list/container labels.  These are structural labels, not entity names.
_PROJECT_LIST_TEXT = (
    "أبرز المشاريع", "ابرز المشاريع", "المشاريع", "قائمة المشاريع",
    "المبادرات", "أبرز المبادرات", "ابرز المبادرات", "projects", "initiatives",
)
_PROGRAM_LIST_TEXT = (
    "أبرز البرامج", "ابرز البرامج", "قائمة البرامج", "برامجنا",
    "featured programs", "key programs", "program list",
)
_ACTIVITY_LIST_TEXT = (
    "أبرز الأنشطة", "ابرز الانشطة", "الأنشطة", "الفعاليات", "activities", "events",
)

# Strong standalone entity nouns.  Type can still be overridden by an explicit
# source list context (e.g. "مسابقة رتل" under "أبرز البرامج" is a program).
_PROGRAM_PREFIXES = ("برنامج", "البرنامج", "برامج")
_PROJECT_STRONG_PREFIXES = ("مشروع", "المشروع", "مبادرة", "المبادرة")
_FLEX_ENTITY_PREFIXES = ("مسابقة", "دورة", "ملتقى", "دليل", "الدليل", "حملة", "خدمة", "منصة", "مسار")

# Common prose starts that look like entity nouns but are actually descriptions.
_DESCRIPTION_STARTS = (
    "برنامج يهدف", "برنامج يعنى", "برنامج يسعى", "برنامج لتعليم", "برنامج مخصص",
    "مشروع مخصص", "مشروع يهدف", "مشروع مفتتح", "مبادرة تهدف", "دورة تدريبية مقدمة", "ملتقى يهدف",
    "حلق تعنى", "حلقة مخصصة", "حلقات تعتني", "منصة تعليمية",
    "program aims", "program designed", "project aims", "initiative aims",
)


def _norm(value) -> str:
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).translate(_DIGITS)
    s = s.replace("ـ", "")
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
    raw = str(name or "")
    # Remove a trailing/unfinished parenthetical subtitle before punctuation is
    # normalized away. This collapses OCR variants like `Name`, `Name (sub...`,
    # and `Name (subtitle)` while the displayed title keeps the fullest source form.
    raw = re.sub(r"\s*\(.*$", "", raw).strip()
    n = _norm(raw)
    toks = n.split()
    identity_prefixes = {_norm(x) for x in (_PROJECT_PREFIXES + _PROGRAM_PREFIXES)}
    if toks and toks[0] in identity_prefixes:
        toks = toks[1:]
    # Compact only for identity matching. This tolerates OCR splits such as
    # "إ كرام" vs "إكرام" without changing the displayed source title.
    return re.sub(r"\s+", "", " ".join(toks))


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
    # Conservative TOC numbering only. Do not strip arbitrary metrics such as
    # "570 مشروع ..." or "34 حملة مستفيدة".
    m = re.match(r"^(\d{1,3})\s+(\d{1,2})\s*[.\-:]\s*", s)
    if m:
        return s[m.end():].strip()
    m = re.match(r"^\.?\s*(\d{1,2})\s*[.\-:]\s*", s)
    if m:
        return s[m.end():].strip()
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
    # In RTL extraction, `description :Title` can start with words such as
    # "مشروع مخصص...". If the line has a plausible short title after the colon,
    # the left side is prose and must not become a second entity.
    if any(c in raw for c in (":", "：")):
        after = re.split(r"[:：]", raw)[-1].strip()
        if after and _title_score(after) >= 1 and _description_like(re.split(r"[:：]", raw)[0]):
            return None
    if _description_like(raw):
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
    if sum(tok in _ORDINALS_NORM for tok in toks) > 1 or toks.count(_norm("برامج")) > 1:
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


def _occurrence_score(raw_lines, norm_lines, idx: int, name: str | None = None) -> int:
    score = 0
    raw = raw_lines[idx].strip()
    n = norm_lines[idx]

    if name:
        name_n = _norm(name)
        stripped_n = _norm(_strip_toc_numbering(raw))
        canon_proj = _canonical_project_title(raw_lines, idx)
        canon_prog = _canonical_program_title(raw)
        if (canon_proj and _name_key(canon_proj) == _name_key(name)) or (canon_prog and _name_key(canon_prog) == _name_key(name)):
            score += 24
        elif stripped_n == name_n:
            score += 24
        elif _contains_token_phrase(n, name_n):
            # Mention inside a longer sentence is weaker than an exact heading.
            extra = max(0, len(n.split()) - len(name_n.split()))
            score += max(0, 8 - min(extra, 8))

    # Read local evidence only until a page/section boundary so fields on the next
    # page cannot make a passing mention look like a detailed entity section.
    evidence = []
    for j in range(idx, min(len(raw_lines), idx + 20)):
        if j > idx and (_is_page_artifact(raw_lines[j]) or _looks_like_major_section_heading(raw_lines[j])):
            break
        evidence.append(raw_lines[j])
    after_n = _norm("\n".join(evidence))
    score += 5 * sum(1 for h in _FIELD_HINTS if _norm(h) in after_n)

    if re.match(r"^\d{1,3}\s+\d{1,2}[.\-:]\s*(?:مبادرة|مشروع|برنامج|مسابقة|دورة|ملتقى|الدليل)", raw):
        score -= 12
    if _nearest_list_type(raw_lines, idx):
        score += 8
    # Nearby program-list heading below a parent container is good evidence.
    for j in range(idx + 1, min(len(raw_lines), idx + 25)):
        if _is_page_artifact(raw_lines[j]):
            continue
        if _list_context_from_heading(raw_lines[j]):
            score += 8
            break
        if _looks_like_major_section_heading(raw_lines[j]) and j > idx + 1:
            break
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

    best_idx = max(dict.fromkeys(hits), key=lambda i: (_occurrence_score(raw_lines, norm_lines, i, name), i))
    return name, raw_lines[best_idx], best_idx, raw_lines, norm_lines


def _is_page_artifact(raw: str) -> bool:
    """Recognize page labels/footers without knowing the organization name."""
    text = " ".join(str(raw or "").split())
    n = _norm(text)
    if not n:
        return False
    if re.search(r"صفح(?:ة|ه)\s*pdf\s*رقم\s*\d+", n):
        return True
    if re.search(r"page\s*\d+(?:\s*of\s*\d+)?$", n, re.I):
        return True
    # Annual-report footers commonly combine a year, organization and "annual report".
    if "التقرير السنوي" in n and any(ch.isdigit() for ch in n):
        return True
    if "annual report" in n and any(ch.isdigit() for ch in n):
        return True
    return False


def _is_short_heading(raw: str) -> bool:
    text = " ".join(str(raw or "").strip().lstrip("*•-–— ").split())
    if not text or len(text) > 140:
        return False
    n = _norm(text)
    if not n or _is_page_artifact(text):
        return False
    words = n.split()
    if len(words) > 12:
        return False
    # A heading generally does not look like a full prose sentence.
    if text.count(".") >= 2 or text.count("؛") >= 2:
        return False
    return True


def _list_context_from_heading(raw: str):
    """Return source-declared item type for list headings, else None."""
    n = _norm(raw)
    if not n or not _is_short_heading(raw):
        return None
    if any(n == _norm(x) or n.startswith(_norm(x) + " ") for x in _PROGRAM_LIST_TEXT):
        return "program"
    if any(n == _norm(x) or n.startswith(_norm(x) + " ") for x in _PROJECT_LIST_TEXT):
        return "project"
    # Activities/events have no separate schema type; treat independently named
    # activities as projects unless the document itself places them under programs.
    if any(n == _norm(x) or n.startswith(_norm(x) + " ") for x in _ACTIVITY_LIST_TEXT):
        return "project"
    return None


def _looks_like_major_section_heading(raw: str) -> bool:
    """Generic section boundary used to stop a prior list context."""
    text = " ".join(str(raw or "").strip().split())
    if not _is_short_heading(text):
        return False
    n = _norm(text)
    if _list_context_from_heading(text):
        return False
    # Colon at either visual edge is common after RTL extraction.
    if text.startswith((":", "：")) or text.endswith((":", "：")):
        return True
    # Broad report-section nouns; do not require a particular organization/report.
    section_starts = (
        "منجزات", "إنجازات", "انجازات", "نتائج",
        "الموارد البشرية", "الحوكمة", "التقرير المالي", "الشراكات", "الاستدامة المالية",
        "مجلس الادارة", "مؤشرات الاداء", "الهيكل التنظيمي", "فروع الجمعية", "الاتصال",
        "تقنية المعلومات", "من نحن", "رؤية ورسالة", "الادارات النسائية",
        "human resources", "governance", "financial report", "partnerships", "contents",
    )
    return any(n.startswith(_norm(x)) for x in section_starts)


def _description_like(text: str) -> bool:
    n = _norm(text)
    if not n:
        return False
    if any(n.startswith(_norm(x)) for x in _DESCRIPTION_STARTS) or len(n.split()) >= 13:
        return True
    if any(c in str(text) for c in ("،", ";", "؛")) and len(n.split()) >= 6:
        return True
    # Generic descriptive verbs near the start are a strong signal that a line is
    # prose rather than an entity heading (e.g. "دليل إرشادي شامل يضم ...").
    toks = n.split()
    early = set(toks[:7])
    verbs = {_norm(x) for x in (
        "يهدف", "تهدف", "يعنى", "تعنى", "يسعى", "تسعى", "يقدم", "تقدم",
        "يضم", "تضم", "يحتوي", "تحتوي", "مخصص", "مخصصة", "مقدمة", "مقدم",
        "لتعليم", "لتحسين", "لتأهيل", "لتقديم", "لاعداد", "لإعداد",
    )}
    if early & verbs:
        return True
    if n.startswith(_norm("دورة تدريبية ")) and len(toks) >= 3:
        return True
    return False


def _title_score(text: str) -> int:
    """Heuristic score for a short entity title fragment around a colon."""
    t = " ".join(str(text or "").strip().lstrip("*•-–— ").split())
    n = _norm(t)
    if not t or not n or _is_page_artifact(t):
        return -99
    words = n.split()
    score = 0
    if len(words) <= 4:
        score += 5
    elif len(words) <= 8:
        score += 3
    elif len(words) <= 12:
        score += 1
    else:
        score -= 6
    if len(t) <= 90:
        score += 2
    if any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _PROGRAM_PREFIXES + _PROJECT_STRONG_PREFIXES + _FLEX_ENTITY_PREFIXES):
        score += 3
    if _description_like(t):
        score -= 7
    if any(_norm(h) == n for h in _FIELD_HINTS):
        score -= 8
    if n in {_norm(x) for x in _GENERIC_PROGRAM_HEADINGS}:
        score -= 8
    return score


def _inline_title_description(raw: str, forced_type=None):
    """Parse either `title: description` or RTL-extracted `description :title`."""
    text = " ".join(str(raw or "").strip().split())
    if not text or not any(c in text for c in (":", "：")):
        return None
    # Evaluate every separator; OCR can leave other colons inside prose.
    best = None
    for m in re.finditer(r"[:：]", text):
        left, right = text[:m.start()].strip(), text[m.end():].strip()
        for title, desc in ((left, right), (right, left)):
            if not title:
                continue
            score = _title_score(title)
            if forced_type and score >= 0:
                score += 3
            # Prefer an opposite side that actually looks descriptive, while still
            # allowing a bare `name:` heading with no same-line description.
            if desc and (_description_like(desc) or len(_norm(desc).split()) >= 5):
                score += 2
            if best is None or score > best[0]:
                best = (score, title, desc or None)
    if not best or best[0] < (1 if forced_type else 4):
        return None
    title = _strip_toc_numbering(best[1]).strip(" :：.-–—")
    if not title or len(title) > 180:
        return None
    return title, best[2]


def _standalone_entity(raw: str, forced_type=None):
    """Recognize a short standalone entity heading without interpreting prose."""
    text = _strip_toc_numbering(str(raw or "")).strip(" :：.-–—")
    if not _is_short_heading(text):
        return None
    n = _norm(text)
    if not n or _description_like(text) or re.fullmatch(r"\d+(?:[.,]\d+)?", n):
        return None
    prefix_norms = {_norm(x) for x in _PROGRAM_PREFIXES + _PROJECT_STRONG_PREFIXES + _FLEX_ENTITY_PREFIXES}
    if sum(tok in prefix_norms for tok in n.split()) > 1:
        return None
    # Two-column TOCs often concatenate another numbered entity later on the line.
    if re.search(r"\s\d{1,2}\s*[.]\s*(?:مبادرة|مشروع|برنامج|مسابقة|دورة|ملتقى|الدليل)", text):
        return None
    if n in {_norm(x) for x in _GENERIC_PROGRAM_HEADINGS}:
        return None
    if any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _PROGRAM_PREFIXES):
        if forced_type is None:
            # Deterministic recovery is conservative outside a declared list; the
            # LLM can still contribute longer unusual names and grounding verifies them.
            if len(n.split()) > 5 or re.search(r"\d", n):
                return None
        return text, "program"
    if any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _PROJECT_STRONG_PREFIXES):
        return text, forced_type or "project"
    if forced_type and any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _FLEX_ENTITY_PREFIXES):
        return text, forced_type
    if forced_type:
        # Inside an explicit source list, short names need no lexical prefix.
        return text, forced_type
    return None


def _nearest_list_type(raw_lines, idx: int, lookback: int = 80):
    """Find the nearest explicit list heading without crossing a major section."""
    for j in range(idx - 1, max(-1, idx - lookback) - 1, -1):
        raw = raw_lines[j]
        if _is_page_artifact(raw) or not raw.strip():
            continue
        typ = _list_context_from_heading(raw)
        if typ:
            return typ
        if _looks_like_major_section_heading(raw):
            return None
    return None


def _program_group_title(raw: str, raw_lines=None, idx=None):
    """Recover parent program/group titles only with structural evidence."""
    text = _strip_toc_numbering(str(raw or "")).strip(" :：.-–—")
    if not _is_short_heading(text):
        return None
    n = _norm(text)
    if not n or n in {_norm(x) for x in _GENERIC_PROGRAM_HEADINGS} or _description_like(text):
        return None
    if len(n.split()) > 7 or any(c in text for c in ("،", ";", "؛")):
        return None

    starts_programs = n.startswith(_norm("برامج") + " ")
    starts_program = n.startswith(_norm("برنامج") + " ")
    starts_container = any(n.startswith(_norm(x) + " ") for x in ("منصة", "مسار", "حزمة"))
    if not (starts_programs or starts_program or starts_container):
        return None

    if raw_lines is None or idx is None:
        return text if starts_programs or starts_program else None

    # Strong source-structure evidence: a nested program list appears soon after.
    for k in range(idx + 1, min(len(raw_lines), idx + 35)):
        if _list_context_from_heading(raw_lines[k]) == "program":
            return text
        if _looks_like_major_section_heading(raw_lines[k]) and k > idx + 1:
            break

    # A single named program can also be a heading when it is nested beneath a
    # program-family heading immediately above (e.g. Programs > Program X).
    if starts_program:
        for k in range(idx - 1, max(-1, idx - 10), -1):
            if _is_page_artifact(raw_lines[k]) or not raw_lines[k].strip():
                continue
            prev_n = _norm(raw_lines[k])
            if prev_n.startswith(_norm("برامج") + " ") or _list_context_from_heading(raw_lines[k]) == "program":
                return text
            if _looks_like_major_section_heading(raw_lines[k]):
                break

    return None


def _generic_item_at_line(raw_lines, idx: int, list_type=None):
    raw = raw_lines[idx]
    inline = _inline_title_description(raw, forced_type=list_type)
    if inline:
        title, _desc = inline
        ent = _standalone_entity(title, forced_type=list_type)
        if ent:
            return ent[0], ent[1]

    # Explicit lexical headings can stand alone anywhere.
    explicit = _standalone_entity(raw, forced_type=None)
    if explicit:
        return explicit

    if list_type:
        # Prefixless names are common inside `أبرز البرامج`. Do not accept every
        # short continuation line: require evidence that the next meaningful line
        # behaves like a description, field, or inline next structure.
        tentative = _standalone_entity(raw, forced_type=list_type)
        if not tentative or _description_like(raw):
            return None
        n = _norm(tentative[0])
        if n.startswith((_norm("وذلك"), _norm("حيث"), _norm("ضمن"), _norm("حتى"), _norm("مع "))):
            return None
        next_raw = None
        for j in range(idx + 1, min(len(raw_lines), idx + 4)):
            if not raw_lines[j].strip() or _is_page_artifact(raw_lines[j]):
                continue
            next_raw = raw_lines[j]
            break
        if next_raw is None:
            return None
        if _inline_title_description(next_raw, forced_type=list_type):
            # A bare title immediately followed by another item is suspicious;
            # usually the current line is a wrapped continuation.
            return None
        if _description_like(next_raw) or len(_norm(next_raw).split()) >= 6 or any(_norm(h) in _norm(next_raw) for h in _FIELD_HINTS):
            return tentative
    return None


def _discover_structure(document: str):
    """Discover high-confidence entities from heterogeneous report layouts.

    Recovery uses source-declared structure (ordinal sections, program/project
    list headings, inline `name: description` pairs, short standalone headings,
    and explicit entity nouns). It is deliberately a supplement to the model,
    not a replacement: model candidates are unioned later and still must pass
    strict grounding against the current document.
    """
    raw_lines = document.splitlines()
    norm_lines = [_norm(x) for x in raw_lines]
    found = OrderedDict()

    def add(title, typ, idx, confidence=1, ordinal_rank=None):
        title = " ".join(str(title or "").split()).strip()
        if not title or typ not in {"program", "project"}:
            return
        key = _name_key(title) or _norm(title)
        if not key:
            return
        prev = found.get(key)
        item = {"title": title, "type": typ, "idx": idx, "confidence": confidence, "ordinal_rank": ordinal_rank}
        if prev is None or (confidence, len(_norm(title)), -idx) > (prev["confidence"], len(_norm(prev["title"])), -prev["idx"]):
            found[key] = item
        elif prev is not None and prev.get("ordinal_rank") is None and ordinal_rank is not None:
            prev["ordinal_rank"] = ordinal_rank

    # Preserve the proven ordinal-section recovery from earlier versions.
    for i, raw in enumerate(raw_lines):
        title = _canonical_program_title(raw)
        if title:
            first_tok = _norm(raw.strip().lstrip("*•-–— ")).split()[:1]
            rank = _ORDINAL_RANK.get(first_tok[0]) if first_tok else None
            add(title, "program", i, 5, ordinal_rank=rank)
        group = _program_group_title(raw, raw_lines, i)
        if group:
            add(group, "program", i, 4)

    # Explicit project/initiative headings remain valid outside list contexts,
    # but flexible nouns such as competition/course/forum need detail evidence so
    # references inside an achievements section are not promoted to entities.
    project_occ = OrderedDict()
    strong_norms = {_norm(x) for x in _PROJECT_STRONG_PREFIXES}
    for i in range(len(raw_lines)):
        title = _canonical_project_title(raw_lines, i)
        if not title:
            continue
        key = _name_key(title)
        project_occ.setdefault(key, {"title": title, "indexes": []})["indexes"].append(i)
    for bucket in project_occ.values():
        title, indexes = bucket["title"], list(dict.fromkeys(bucket["indexes"]))
        first_tok = _norm(title).split()[:1]
        is_strong = bool(first_tok and first_tok[0] in strong_norms)
        best_idx = max(indexes, key=lambda i: (_occurrence_score(raw_lines, norm_lines, i, title), i))
        list_type = _nearest_list_type(raw_lines, best_idx)
        detail_score = _occurrence_score(raw_lines, norm_lines, best_idx, None)
        strong_supported = is_strong and (len(indexes) >= 2 or detail_score >= 5)
        flexible_supported = detail_score >= 10 or (len(indexes) >= 2 and detail_score >= 5)
        if list_type or strong_supported or flexible_supported:
            add(title, list_type or "project", best_idx, 5 if list_type else 4)

    # Generic list-aware pass. Context is reset by real section headings, but page
    # artifacts do not reset it because lists frequently continue across pages.
    active_type = None
    for i, raw in enumerate(raw_lines):
        if _is_page_artifact(raw) or not raw.strip():
            continue
        ctx = _list_context_from_heading(raw)
        if ctx:
            active_type = ctx
            continue
        if _looks_like_major_section_heading(raw):
            active_type = None
            continue

        # The generic pass is list-aware by design. Outside an explicit list,
        # conservative prefix recovery is handled above and future unusual layouts
        # are supplied by the model + strict grounding. This avoids promoting
        # financial-note mentions such as "مشروع كفالة..." into execution records.
        if active_type is None:
            continue
        ent = _generic_item_at_line(raw_lines, i, active_type)
        if not ent:
            continue
        title, typ = ent
        add(title, typ, i, 5)

    vals = list(found.values())
    programs = [x for x in vals if x["type"] == "program"]
    projects = [x for x in vals if x["type"] == "project"]
    if sum(x.get("ordinal_rank") is not None for x in programs) >= 2:
        programs.sort(key=lambda x: (x.get("ordinal_rank") is None, x.get("ordinal_rank") or 999, x["idx"]))
    else:
        programs.sort(key=lambda x: x["idx"])
    projects.sort(key=lambda x: x["idx"])
    items = programs + projects
    # Even one explicit program heading is useful; no brittle "2 sections + 3
    # repeated projects" gate is required. Grounding later removes false positives.
    return items, bool(items)


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
    discovered, structured = _discover_structure(document)
    if not structured:
        return list(programs or []), False

    out, seen = [], set()
    for spec in discovered:
        title, typ = spec["title"], spec["type"]
        base = dict(_best_candidate(programs or [], title) or _blank_record(title, typ))
        base["name"] = title
        base["type"] = typ
        out.append(base)
        seen.add(_name_key(title))

    # Generalization rule: deterministic recovery supplements model extraction.
    # Never throw away a model candidate merely because a future document uses a
    # structure we did not anticipate; strict grounding will still verify it.
    for p in programs or []:
        key = _name_key(p.get("name"))
        if key and key not in seen:
            out.append(dict(p))
            seen.add(key)

    log.info("structural recovery: discovered=%d (programs=%d projects=%d) model_candidates=%d union=%d",
             len(discovered), sum(x["type"] == "program" for x in discovered),
             sum(x["type"] == "project" for x in discovered), len(programs or []), len(out))
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
            if _is_page_artifact(lines[j]) or next_n == _norm("التقرير") or re.fullmatch(r"ف\s*\d+", next_n):
                break
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
    """Infer a report-wide Gregorian year conservatively.

    Prefer an explicit annual/report year. Otherwise use a strongly dominant year
    rather than requiring the document to contain no historical comparison years.
    """
    text = unicodedata.normalize("NFKC", document or "").translate(_DIGITS)
    explicit = []
    patterns = (
        r"(?:التقرير\s+(?:السنوي|النصف\s+سنوي)|تقرير\s+الأعمال[^\n]{0,40}?لعام)\D{0,20}(20\d{2})",
        r"(?:annual\s+report|report\s+for)\D{0,20}(20\d{2})",
    )
    for pat in patterns:
        explicit.extend(int(x) for x in re.findall(pat, text, flags=re.I))
    if explicit:
        from collections import Counter
        c = Counter(explicit)
        year, count = c.most_common(1)[0]
        if count >= 1:
            return year

    from collections import Counter
    years = [int(n) for n in _numbers(text) if float(n).is_integer() and 1900 <= int(n) <= 2100]
    if not years:
        return None
    c = Counter(years)
    ranked = c.most_common(2)
    if len(ranked) == 1:
        return ranked[0][0]
    (y1, c1), (_y2, c2) = ranked
    if c1 >= 3 and c1 >= c2 * 2:
        return y1
    return None


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
    # Prefer structural title parsers that can join wrapped parenthetical subtitles
    # before the generic short-title parser.
    proj = _canonical_project_title(raw_lines, idx)
    if proj and _name_key(proj) == _name_key(fallback):
        return proj
    prog = _canonical_program_title(raw_lines[idx])
    if prog and _name_key(prog) == _name_key(fallback):
        return prog
    group = _program_group_title(raw_lines[idx], raw_lines, idx)
    if group and _name_key(group) == _name_key(fallback):
        return group
    list_type = _nearest_list_type(raw_lines, idx)
    generic = _generic_item_at_line(raw_lines, idx, list_type)
    if generic and _name_key(generic[0]) == _name_key(fallback):
        return generic[0]
    return fallback


def _ground_type(current_type, raw_lines, idx: int, canonical_name: str):
    # Source-declared list semantics outrank lexical prefixes. Example: a report
    # can list "مسابقة رتل" under "أبرز البرامج"; it must remain a program.
    list_type = _nearest_list_type(raw_lines, idx)
    if list_type:
        return list_type

    prog = _canonical_program_title(raw_lines[idx])
    if prog and _name_key(prog) == _name_key(canonical_name):
        return "program"
    group = _program_group_title(raw_lines[idx], raw_lines, idx)
    if group and _name_key(group) == _name_key(canonical_name):
        return "program"

    n = _norm(canonical_name)
    if any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _PROGRAM_PREFIXES):
        return "program"
    if any(n.startswith(_norm(x) + " ") or n == _norm(x) for x in _PROJECT_STRONG_PREFIXES):
        return "project"

    proj = _canonical_project_title(raw_lines, idx)
    if proj and _name_key(proj) == _name_key(canonical_name):
        return "project"

    # Flexible nouns (competition/course/forum/etc.) are ambiguous without a list
    # context, so keep the model/deterministic type only when the title is grounded.
    title_n = _window_norm([_norm(x) for x in raw_lines], idx, 2)
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


def _context_bounds(raw_lines, norm_lines, start_idx: int, all_names, max_lines: int = 60):
    # Include preceding lines only when they look like actual table/field headers.
    # Blindly taking two prior lines can leak the previous item's values into the
    # current item in sequential report layouts.
    lo = start_idx
    for j in range(max(0, start_idx - 2), start_idx):
        n = norm_lines[j]
        hint_count = sum(1 for h in _FIELD_HINTS if _norm(h) in n)
        if "|" in raw_lines[j] or hint_count >= 2:
            lo = j
            break
    hi = min(len(raw_lines), start_idx + max_lines)
    current_key = _name_key(all_names[0]) if all_names else ""
    current_list = _nearest_list_type(raw_lines, start_idx)
    for j in range(start_idx + 1, hi):
        if _is_page_artifact(raw_lines[j]):
            # Page artifacts are excluded from evidence but not always a hard
            # boundary; a list may continue on the next page.
            continue
        # A new explicit section ends this entity unless it is merely the same
        # list heading repeated on a continuation page.
        if _looks_like_major_section_heading(raw_lines[j]):
            hi = j
            return lo, hi
        ctx = _list_context_from_heading(raw_lines[j])
        if ctx and ctx != current_list and j > start_idx + 1:
            hi = j
            return lo, hi
        for other in all_names[1:]:
            if _name_key(other) == current_key:
                continue
            if _starts_name_window(norm_lines, j, other, span=3):
                hi = j
                return lo, hi
        # Generic inline/standalone next item catches names the model did not emit.
        ent = _generic_item_at_line(raw_lines, j, current_list)
        if ent and _name_key(ent[0]) != current_key:
            hi = j
            return lo, hi
    return lo, hi


def _inline_description_for_name(raw: str, canonical_name: str):
    parsed = _inline_title_description(raw, forced_type=None)
    if not parsed:
        # In a list, otherwise-prefixless titles still need parsing.
        parsed = _inline_title_description(raw, forced_type="program") or _inline_title_description(raw, forced_type="project")
    if not parsed:
        return None
    title, desc = parsed
    if _name_key(title) != _name_key(canonical_name):
        return None
    cleaned = " ".join((desc or "").split()) or None
    if cleaned and _norm(cleaned) in _ORDINALS_NORM:
        return None
    return cleaned


def _looks_like_detail_list_start(raw: str) -> bool:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        return False
    # Numbered achievements/people after a short program description.
    if re.match(r"^(?:[.\-]?\s*\d{1,2}\s*[.)\-:]|\d{1,2}\s+)", text):
        return True
    if re.search(r"[.\s]\d{1,2}$", text):
        return True
    n = _norm(text)
    return n.startswith(_norm("المركز الاول")) or n.startswith(_norm("المركز الثاني")) or n.startswith(_norm("المركز الثالث"))


def _infer_description(raw_lines, start_idx: int, hi: int, canonical_name: str):
    out = []
    inline = _inline_description_for_name(raw_lines[start_idx], canonical_name)
    if inline:
        out.append(inline)

    j = start_idx + 1
    # Parenthetical subtitle may be the second line of the heading.
    if j < hi:
        nxt = " ".join(raw_lines[j].strip().split())
        if nxt.startswith("(") and ")" in nxt and _norm(nxt) in _norm(canonical_name):
            j += 1

    current_list = _nearest_list_type(raw_lines, start_idx)
    for k in range(j, hi):
        raw = " ".join(raw_lines[k].strip().split())
        if not raw or _is_page_artifact(raw):
            continue
        n = _norm(raw)
        if n == _norm("التقرير") or re.fullmatch(r"ف\s*\d+", n):
            break
        if re.fullmatch(r"\d{1,3}", n):
            continue
        if any(_norm(h) in n for h in _FIELD_HINTS):
            break
        if n in {_norm("الإجمالي"), _norm("الإجاملي"), _norm("اجمالي"), _norm("اجاملي"), _norm("المجموع"), _norm("total")} or n.startswith(_norm("عدد ")):
            break
        if _looks_like_major_section_heading(raw) or _list_context_from_heading(raw):
            break
        ent = _generic_item_at_line(raw_lines, k, current_list)
        if ent and _name_key(ent[0]) != _name_key(canonical_name):
            break
        # Avoid swallowing achievement/person lists after a completed standalone
        # description (common in annual reports).
        if out and _looks_like_detail_list_start(raw):
            break
        # If the current line itself is the title occurrence, do not echo it.
        if _line_matches_name(_norm(raw), canonical_name) and len(_norm(raw).split()) <= len(_norm(canonical_name).split()) + 1:
            continue
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



# Generic lexical anchors used only to split wrapped RTL rows whose header contains
# both "مكان التنفيذ" and "الفئة المستهدفة".  These are not entity names and are
# intentionally broad so the logic stays document-agnostic.
_PLACE_MARKERS = tuple(_norm(x) for x in (
    "مقر", "موقع", "مكان", "فندق", "الفنادق", "مستشفى", "مسجد", "المسجد",
    "مصلى", "مصليات", "المصليات", "ساحات", "الساحات", "المنطقة", "غرفة", "قاعة", "مركز", "المركز",
    "المطار", "المطارات", "المحطة", "المحطات", "عن بعد", "الحملات", "التخصصي",
))
_AUDIENCE_MARKERS = tuple(_norm(x) for x in (
    "ضيوف", "ضيفات", "الحاجات", "الحجاج", "المعتمرات", "الزائرات", "الرجال",
    "النساء", "أطفال", "اطفال", "فتيات", "مشرفات", "المريضات", "المر يضات", "مر يضات",
    "العاملون", "المهتمون", "المتطوعات", "المتطوعين", "حملات حجاج",
    "حجاج الداخل", "الخارج", "الخار ج", "العرب",
))


def _strip_leading_metrics(raw: str) -> str:
    """Remove volunteer/hour numeric cells that precede wrapped text values."""
    s = unicodedata.normalize("NFKC", raw or "").translate(_DIGITS).strip()
    # Two/three leading numeric cells are common in rows such as "12 116 الفنادق ...".
    s = re.sub(r"^(?:(?:\d+(?:[.,]\d+)?)\s+){1,3}(?=[\u0600-\u06FF])", "", s)
    return " ".join(s.split())


def _word_marker_index(words, markers):
    nwords = [_norm(w) for w in words]

    def eq_token(word, marker):
        if word == marker:
            return True
        # Arabic conjunction is often attached to the next cell fragment after OCR,
        # e.g. "والزائرات" should match the marker "الزائرات".
        return len(word) > 2 and word.startswith("و") and word[1:] == marker

    best = None
    for marker in markers:
        mtoks = marker.split()
        if not mtoks:
            continue
        for i in range(len(nwords) - len(mtoks) + 1):
            if all(eq_token(nwords[i + k], mtoks[k]) for k in range(len(mtoks))):
                cand = (i, len(mtoks))
                if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
                break
    return best


def _dynamic_place_index(words, current_delivery: str | None):
    if not current_delivery:
        return None
    d = _norm(current_delivery)
    nw = [_norm(w) for w in words]
    # "حملات الحج" is a location phrase, while "حملات حجاج ..." is an audience.
    if _norm("حملات") in d and _norm("الحج") in nw:
        return nw.index(_norm("الحج"))
    return None


def _dynamic_audience_index(words, current_target: str | None):
    """Recognize a few common continuation tokens only when the target phrase needs them."""
    if not current_target:
        return None
    t = _norm(current_target)
    nw = [_norm(w) for w in words]
    candidates = []
    if any(x in t for x in (_norm("ضيوف"), _norm("ضيفات"))) and _norm("الرحمن") in nw:
        candidates.append(nw.index(_norm("الرحمن")))
    if _norm("البيت") in t and _norm("الحرام") in nw:
        candidates.append(nw.index(_norm("الحرام")))
    if any(x in t for x in (_norm("العاملون"), _norm("المهتمون"))):
        for token in (_norm("بالقطاع"), _norm("القطاع"), _norm("الربحي")):
            if token in nw:
                candidates.append(nw.index(token))
    if _norm("حجاج") in t:
        for token in (_norm("الخارج"), _norm("الخار"), _norm("العرب")):
            if token in nw:
                candidates.append(nw.index(token))
    # In RTL tables the target can wrap from "مشرفات اللجان الثقافية" to
    # "في حملات الحج" while delivery is simply "عن بعد".
    if _norm("مشرفات") in t and _norm("حملات") in nw and _norm("الحج") in nw:
        candidates.append(nw.index(_norm("حملات")))
    return min(candidates) if candidates else None


def _append_part(parts, text):
    text = " ".join(str(text or "").split()).strip(" ،,;؛")
    if text:
        parts.append(text)


def _join_parts(parts):
    if not parts:
        return None
    # Preserve source wording while removing exact repeated chunks introduced by OCR wrapping.
    out = []
    seen = set()
    for part in parts:
        n = _norm(part)
        if n and n not in seen:
            out.append(part)
            seen.add(n)
    return " ".join(out)[:1000] if out else None


def _infer_multicolumn_target_delivery(context: str):
    """Recover target/delivery from wrapped RTL multi-column rows.

    The routine activates only when the same header line contains both field labels,
    then consumes the few lines immediately below that header.  It never searches
    outside the current item context.
    """
    lines = context.splitlines()
    norm_lines = [_norm(x) for x in lines]
    target_labels = [_norm(x) for x in _TARGET_LABELS]
    delivery_labels = [_norm(x) for x in _DELIVERY_LABELS]

    header_idx = None
    for i, n in enumerate(norm_lines):
        if any(x and x in n for x in target_labels) and any(x and x in n for x in delivery_labels):
            header_idx = i
            break
    if header_idx is None:
        return None, None

    target_parts, delivery_parts = [], []
    unknown = []
    all_hint_norms = [_norm(h) for h in _FIELD_HINTS]

    for j in range(header_idx + 1, min(len(lines), header_idx + 7)):
        raw = _strip_leading_metrics(lines[j])
        n = _norm(raw)
        if not n:
            continue
        if n == "التقرير" or re.fullmatch(r"ف\s*\d+", n):
            break
        if any(h in n for h in all_hint_norms):
            break

        words = raw.split()
        place_hit = _word_marker_index(words, _PLACE_MARKERS)
        audience_hit = _word_marker_index(words, _AUDIENCE_MARKERS)
        dyn_place = _dynamic_place_index(words, _join_parts(delivery_parts))
        if dyn_place is not None and (place_hit is None or dyn_place < place_hit[0]):
            place_hit = (dyn_place, 1)
        dyn_idx = _dynamic_audience_index(words, _join_parts(target_parts))
        if dyn_idx is not None and (audience_hit is None or dyn_idx < audience_hit[0]):
            audience_hit = (dyn_idx, 1)

        if place_hit is not None and audience_hit is not None:
            pi, _ = place_hit
            ai, _ = audience_hit
            if pi < ai:
                _append_part(delivery_parts, " ".join(words[:ai]))
                _append_part(target_parts, " ".join(words[ai:]))
            elif ai < pi:
                _append_part(target_parts, " ".join(words[:pi]))
                _append_part(delivery_parts, " ".join(words[pi:]))
            else:
                # Same start is rare; prefer the more specific audience marker only if
                # it is a known audience phrase such as "حملات حجاج".
                if _norm(" ".join(words[ai:ai + audience_hit[1]])) in _AUDIENCE_MARKERS:
                    _append_part(target_parts, raw)
                else:
                    _append_part(delivery_parts, raw)
            continue

        if audience_hit is not None:
            # If an audience marker appears anywhere on an otherwise unsplit line,
            # the whole phrase usually belongs to the target column.
            _append_part(target_parts, raw)
            continue
        if place_hit is not None:
            _append_part(delivery_parts, raw)
            continue
        unknown.append(raw)

    # Conservative continuation handling for proper names / wrapped endings.
    for raw in unknown:
        n = _norm(raw)
        current_target = _join_parts(target_parts) or ""
        current_delivery = _join_parts(delivery_parts) or ""
        if (_norm("الرحمن") in n and any(x in _norm(current_target) for x in (_norm("ضيوف"), _norm("ضيفات")))) \
                or (_norm("الحرام") in n and _norm("البيت") in _norm(current_target)) \
                or (_norm("القطاع") in n and any(x in _norm(current_target) for x in (_norm("العاملون"), _norm("المهتمون")))) \
                or (_norm("مشرفات") in _norm(current_target) and _norm("حملات") in n and _norm("الحج") in n):
            _append_part(target_parts, raw)
        elif any(x in _norm(current_delivery) for x in (_norm("فندق"), _norm("مسجد"), _norm("مستشفى"), _norm("غرفة"), _norm("عن بعد"))):
            _append_part(delivery_parts, raw)

    return _join_parts(target_parts), _join_parts(delivery_parts)


# Conservative display-only OCR repairs. These are common Arabic OCR token
# splits, not semantic rewrites. Grounding still happens against the raw source
# before this cleanup is applied.
_OCR_DISPLAY_FIXES = (
    # Do not require word boundaries here: Arabic clitics such as و/ل/ب are
    # commonly attached to the broken token (e.g. "وتوز يع", "لتعز يز").
    # Each pattern still contains an impossible internal whitespace sequence,
    # so the repair is presentation-only and does not invent content.
    (re.compile(r"توز\s+يع"), "توزيع"),
    (re.compile(r"ز\s+يارة"), "زيارة"),
    (re.compile(r"تعز\s+يز"), "تعزيز"),
    (re.compile(r"تدر\s+يب"), "تدريب"),
    (re.compile(r"المعتمر\s+ين"), "المعتمرين"),
    (re.compile(r"الخار\s+ج"), "الخارج"),
    (re.compile(r"المركز\s+ية"), "المركزية"),
    (re.compile(r"الإنجليز\s+ية"), "الإنجليزية"),
    (re.compile(r"التذكار\s+ية"), "التذكارية"),
    (re.compile(r"الكر\s+يم"), "الكريم"),
    (re.compile(r"المر\s+يضات"), "المريضات"),
    (re.compile(r"مر\s+يضات"), "مريضات"),
    (re.compile(r"نم\s+اذج"), "نماذج"),
    (re.compile(r"فر\s+ق"), "فرق"),
    (re.compile(r"ب\s+فاعلية"), "بفاعلية"),
    (re.compile(r"\bالتلقني\b"), "التلقين"),
    (re.compile(r"\bالصغري\b"), "الصغير"),
    (re.compile(r"\bالقرآين\b"), "القرآني"),
    (re.compile(r"\bتحسني\b"), "تحسين"),
    (re.compile(r"\bغري\b"), "غير"),
    (re.compile(r"اءّالقر"), "القراء"),
)


def _clean_display_text(value):
    """Clean only obvious OCR/spacing artefacts after grounding."""
    if value is None:
        return None
    s = unicodedata.normalize("NFKC", str(value)).replace("ـ", "")
    # Remove a stray Arabic combining mark that OCR leaves after whitespace,
    # e.g. "مبادرة ُسلوان". Do not strip valid marks inside words.
    s = re.sub(r"(^|\s)[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+", r"\1", s)
    for pattern, repl in _OCR_DISPLAY_FIXES:
        s = pattern.sub(repl, s)
    s = s.strip().lstrip(".،,؛;: ")
    s = re.sub(r"\s*[،,]\s*", "، ", s)
    s = re.sub(r"\s+([؛;:.!?؟])", r"\1", s)
    s = re.sub(r"([\(\[«])\s+", r"\1", s)
    s = re.sub(r"\s+([\)\]»])", r"\1", s)
    # Common OCR punctuation inversions around parenthesized lists.
    s = re.sub(r"\b(تتضمن|شملت)\s*\(\s*:\s*", r"\1: (", s)
    # Missing space after sentence punctuation can join two valid words.
    s = re.sub(r"([.!؟])(?=[\u0600-\u06FF])", r"\1 ", s)
    # Two high-confidence OCR diacritic-order errors present in Arabic reports.
    s = s.replace("أرًزا", "أرزًا").replace("دعًما", "دعمًا")
    s = " ".join(s.split()).strip()
    return s or None


def _clean_delivery_display(value):
    s = _clean_display_text(value)
    if not s:
        return s
    # Re-introduce obvious separators lost by multi-column OCR. This is only
    # formatting; it does not add a new place that was not already present.
    s = re.sub(r"^(مقر الجمعية)\s+(?=(?:الفنادق|الحملات|حملات|مسجد|مصلى|مصليات|مستشفى|غرفة|ساحات|المنطقة))", r"\1، ", s)
    s = re.sub(r"\s+،\s*", "، ", s)
    return s


def _explicit_notes(context: str):
    """Return source notes only when a notes label is actually present."""
    context_n = _norm(context)
    if not any(_norm(lbl) in context_n for lbl in _NOTES_LABELS):
        return None
    return _infer_simple_label_text(context, _NOTES_LABELS)


def _beneficiary_value_is_audience(value, target_audience) -> bool:
    """Reject audience text accidentally copied into beneficiary_value."""
    if not value:
        return False
    v = _norm(value)
    t = _norm(target_audience)
    if t and (v == t or _contains_token_phrase(t, v) or _contains_token_phrase(v, t)):
        return True
    words = v.split()
    if not words:
        return False
    marker_tokens = set()
    for marker in _AUDIENCE_MARKERS:
        marker_tokens.update(marker.split())
    # Strong audience terms at the start are enough to reject this semantic mix-up.
    return words[0] in marker_tokens

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
        # Notes are stricter: arbitrary nearby text is not a note unless the
        # source itself has an explicit notes label.
        for key in ("beneficiary_value", "target_audience", "delivery_method"):
            if item.get(key) is not None and not _text_is_grounded(item[key], context):
                item[key] = None
                nulled += 1

        source_notes = _explicit_notes(context)
        if source_notes:
            item["notes"] = source_notes
        elif item.get("notes") is not None:
            item["notes"] = None
            nulled += 1

        # Deterministic fill for labelled layouts. First try the wrapped RTL
        # multi-column parser, then fall back to one-column label/value extraction.
        multi_target, multi_delivery = _infer_multicolumn_target_delivery(context)
        if multi_target:
            # Multi-column reconstruction is more specific than a partial model value.
            item["target_audience"] = multi_target
        elif item.get("target_audience") is None:
            item["target_audience"] = _infer_simple_label_text(context, _TARGET_LABELS)

        if multi_delivery:
            item["delivery_method"] = multi_delivery
        elif item.get("delivery_method") is None:
            item["delivery_method"] = _infer_simple_label_text(context, _DELIVERY_LABELS)

        # beneficiary_value is the delivered benefit/service, never the audience.
        # If the model copied the target group into this field, null it rather than guess.
        if _beneficiary_value_is_audience(item.get("beneficiary_value"), item.get("target_audience")):
            item["beneficiary_value"] = None
            nulled += 1

        # Final presentation cleanup happens only after strict grounding.
        for key in ("name", "description", "beneficiary_value", "target_audience", "notes"):
            item[key] = _clean_display_text(item.get(key))
        item["delivery_method"] = _clean_delivery_display(item.get("delivery_method"))

        grounded.append(item)

    log.info("grounding v8.4: structured=%s kept=%d dropped=%d nulled_fields=%d",
             structured, len(grounded), dropped, nulled)
    return grounded, {
        "grounded_kept": len(grounded),
        "grounded_dropped": dropped,
        "grounded_nulled_fields": nulled,
        "structured_recovery": structured,
    }

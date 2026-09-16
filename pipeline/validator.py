"""Coerce model output to the StoreProgramRequest rules instead of crashing.

Model output like "2025م", "1,200 مستفيد", "٤٥٬٠٠٠ ريال" or "مشروع" is converted,
not rejected. Only items without a usable name are dropped."""
import logging
import re

log = logging.getLogger("validator")

FIELDS = (
    "type", "name", "year", "description", "beneficiaries_count", "budget",
    "beneficiary_value", "target_audience", "delivery_method", "notes",
)
TEXT_FIELDS = ("description", "beneficiary_value", "target_audience", "delivery_method", "notes")

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_NULL_WORDS = {"", "null", "none", "n/a", "na", "-", "—", "غير متوفر", "غير محدد", "لا يوجد", "لا توجد"}
_MULTIPLIERS = (
    (re.compile(r"(?<!\w)(?:مليار|billion|bn)(?!\w)", re.I), 1e9),
    (re.compile(r"(?<!\w)(?:مليون|million|mn|m)(?!\w)", re.I), 1e6),
    (re.compile(r"(?<!\w)(?:ألف|الف|آلاف|الاف|thousand|k)(?!\w)", re.I), 1e3),
)


def _parse_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.translate(_DIGITS).replace("٬", ",").replace("٫", ".").strip()
    if s.lower() in _NULL_WORDS:
        return None
    m = re.search(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?", s)
    if not m:
        return None
    raw = re.sub(r"[, ]", "", m.group(0))
    try:
        number = float(raw)
    except ValueError:
        return None
    rest = s[m.end():]
    for pattern, factor in _MULTIPLIERS:
        if pattern.search(rest[:12]):
            number *= factor
            break
    return number


def _to_int(value, minimum=0, maximum=None):
    number = _parse_number(value)
    if number is None or number < minimum or (maximum is not None and number > maximum):
        return None
    return int(round(number))


def _to_budget(value):
    number = _parse_number(value)
    if number is None or number < 0:
        return None
    return round(float(number), 2)


def _to_year(value):
    number = _parse_number(value)
    if number is None:
        return None
    year = int(number)
    return year if 1900 <= year <= 2100 else None


def _to_text(value, max_len=None):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (list, tuple)):
        value = "، ".join(str(v) for v in value if v not in (None, ""))
    elif isinstance(value, dict):
        return None
    text = " ".join(str(value).split())
    if text.lower() in _NULL_WORDS:
        return None
    return text[:max_len] if max_len else text


def _to_type(value):
    s = str(value or "").strip().lower()
    if s in {"program", "project"}:
        return s
    if "برنامج" in s or "program" in s:
        return "program"
    return "project"  # historical default of this pipeline


def clean_program(item):
    if not isinstance(item, dict):
        return None
    name = _to_text(item.get("name"), max_len=255)
    if not name:
        return None
    cleaned = {
        "type": _to_type(item.get("type")),
        "name": name,
        "year": _to_year(item.get("year")),
        "description": None,
        "beneficiaries_count": _to_int(item.get("beneficiaries_count")),
        "budget": _to_budget(item.get("budget")),
        "beneficiary_value": None,
        "target_audience": None,
        "delivery_method": None,
        "notes": None,
    }
    for key in TEXT_FIELDS:
        cleaned[key] = _to_text(item.get(key))
    return cleaned


def clean_programs(items):
    programs, dropped = [], 0
    for item in items or []:
        cleaned = clean_program(item)
        if cleaned is None:
            dropped += 1
        else:
            programs.append(cleaned)
    if dropped:
        log.warning("Dropped %d item(s) without a usable name.", dropped)
    return programs


def _key(p):
    name = re.sub(r"\s+", " ", p["name"]).strip().lower()
    return (name, p["year"], p["beneficiaries_count"], p["budget"])


def merge_programs(programs):
    merged, seen = [], set()
    for p in programs:
        key = _key(p)
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)
    return merged


def filled_fields(p):
    return sum(1 for k in FIELDS if k not in {"type", "name"} and p.get(k) is not None)

"""Robust parsing of the model's JSON output.

Handles: ```json fences, <think> blocks, text around the JSON, trailing commas,
and output that was cut off at max_new_tokens (keeps only complete items)."""
import json
import logging
import re

log = logging.getLogger("json_utils")
_decoder = json.JSONDecoder()


def _strip_wrappers(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"```(?:json)?", "", text, flags=re.I)
    return text.strip()


def _salvage_items(text):
    """Return every COMPLETE object inside the programs array."""
    m = re.search(r'"programs"\s*:\s*\[', text)
    if m:
        pos = m.end()
    else:
        pos = text.find("[")
        if pos < 0:
            return []
        pos += 1
    items = []
    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            break
        try:
            obj, pos = _decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            items.append(obj)
    return items


def _items_of(data):
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        programs = data.get("programs")
        if isinstance(programs, list):
            return [x for x in programs if isinstance(x, dict)]
        if "name" in data:
            return [data]
        for value in data.values():  # e.g. {"result": {"programs": [...]}}
            if isinstance(value, (dict, list)):
                nested = _items_of(value)
                if nested:
                    return nested
    return []


def parse_program_items(raw_text, truncated=False):
    """Return a list of program dicts from raw model output. Raises ValueError."""
    text = _strip_wrappers(raw_text)

    # Only the OUTERMOST JSON value counts; trying later "{" positions would
    # silently return a single inner item from a truncated/invalid array.
    m = re.search(r"[\{\[]", text)
    if m:
        try:
            data, pos = _decoder.raw_decode(text, m.start())
            items = _items_of(data)
            # The model sometimes emits several top-level objects instead of one
            # array: {...}\n{...}\n{...}. Keep reading until the text runs out.
            while True:
                nxt = re.match(r"[\s,]*(?=[\{\[])", text[pos:])
                if not nxt:
                    break
                try:
                    more, pos = _decoder.raw_decode(text, pos + nxt.end())
                except json.JSONDecodeError:
                    break
                items.extend(_items_of(more))
            return items
        except json.JSONDecodeError:
            pass

    salvaged = _salvage_items(text)
    repaired = []
    try:
        from json_repair import repair_json

        repaired = _items_of(repair_json(text, return_objects=True))
        if truncated and repaired:
            repaired = repaired[:-1]  # the last item was cut mid-way
    except ImportError:
        pass
    except Exception:
        log.warning("json_repair could not repair the output", exc_info=True)

    best = repaired if len(repaired) > len(salvaged) else salvaged
    if best:
        log.warning("Model JSON was invalid%s; recovered %d item(s).",
                    " (truncated)" if truncated else "", len(best))
        return best
    if not text:
        raise ValueError("Model returned an empty response.")
    raise ValueError("Model output did not contain valid JSON.")

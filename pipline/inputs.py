"""Normalize whatever n8n sends into one Request object and write files to disk.

Accepted shapes (all inside RunPod's {"input": ...}):

1. Recommended
   {"run_id": "42", "type": "program_autofill", "text": "...",
    "files": [{"name": "brief.pdf", "data": "<base64>"}]}

2. Contract fields forwarded as-is from the Laravel webhook
   {"run_id": "42", "type": "program_autofill",
    "input": "{\"text\":\"...\",\"files\":[\"brief.pdf\"]}",
    "attachments": ["<base64>", ...]}            # same order as input.files

3. Files by URL:  "files": [{"name": "a.xlsx", "url": "https://..."}]
4. Legacy:        {"file_url": "https://...", "file_name": "report.pdf"}
"""
from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

from config import DEFAULT_RUN_TYPE, DOWNLOAD_TIMEOUT, MAX_FILE_BYTES, MAX_FILES


class InputError(ValueError):
    """The caller sent a bad request (as opposed to a server/model failure)."""


@dataclass
class FileSpec:
    index: int
    name: str | None = None
    data: str | None = None
    url: str | None = None
    mime_type: str | None = None

    @property
    def label(self):
        return self.name or f"file #{self.index}"


@dataclass
class Request:
    type: str
    run_id: str | None = None
    text: str | None = None
    files: list[FileSpec] = field(default_factory=list)


_BASE64_RE = re.compile(r"^[A-Za-z0-9+/=_\-\s]+$")


def _loads(value, where):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = value.strip()
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise InputError(f"'{where}' is not valid JSON: {exc}") from exc


def _first(d, *keys):
    for key in keys:
        value = d.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _looks_like_base64(s):
    return s.startswith("data:") or (len(s) >= 64 and bool(_BASE64_RE.match(s[:4096])))


def _apply_part(spec, part, is_attachment):
    if part is None:
        return
    if isinstance(part, str):
        s = part.strip()
        if s.startswith(("http://", "https://")):
            spec.url = spec.url or s
        elif is_attachment or _looks_like_base64(s):
            spec.data = spec.data or s
        else:
            spec.name = spec.name or s
        return
    if isinstance(part, dict):
        spec.name = spec.name or _first(part, "name", "filename", "file_name", "fileName")
        spec.data = spec.data or _first(part, "data", "base64", "content", "file_base64")
        spec.url = spec.url or _first(part, "url", "file_url", "download_url")
        spec.mime_type = spec.mime_type or _first(part, "mime_type", "mimeType", "content_type", "contentType")
        return
    raise InputError(f"File #{spec.index} must be an object or a string, got {type(part).__name__}.")


def normalize_request(raw) -> Request:
    if raw is None:
        raise InputError("Missing 'input'.")
    if isinstance(raw, (str, bytes)):
        raw = _loads(raw, "input")
    if not isinstance(raw, dict):
        raise InputError("'input' must be a JSON object.")

    outer = raw
    inner = raw.get("input")
    if isinstance(inner, (str, bytes)):
        inner = _loads(inner, "input.input")

    if isinstance(inner, dict):
        payload = {k: v for k, v in outer.items() if k != "input"}
        payload.update(inner)
        # Contract case: inner "files" are only names, outer "files" carry content.
        if "files" in inner and "files" in outer and not payload.get("attachments"):
            payload["attachments"] = outer["files"]
    else:
        payload = dict(outer)

    entries = _as_list(payload.get("files"))
    attachments = _as_list(
        payload.get("attachments") or payload.get("files_base64") or payload.get("file_data")
    )

    specs = []
    for i in range(max(len(entries), len(attachments))):
        spec = FileSpec(index=i + 1)
        _apply_part(spec, entries[i] if i < len(entries) else None, is_attachment=False)
        _apply_part(spec, attachments[i] if i < len(attachments) else None, is_attachment=True)
        if not spec.data and not spec.url:
            raise InputError(
                f"{spec.label} has no content. Send the file bytes as base64 in 'data' "
                f"(or a downloadable 'url'); a filename alone is not enough."
            )
        specs.append(spec)

    legacy_url = _first(payload, "file_url", "url", "document_url")
    if legacy_url:
        specs.append(FileSpec(
            index=len(specs) + 1,
            name=_first(payload, "file_name", "filename", "name"),
            url=legacy_url,
        ))

    if len(specs) > MAX_FILES:
        raise InputError(f"Too many files: {len(specs)} (max {MAX_FILES}).")

    text = payload.get("text")
    text = text.strip() if isinstance(text, str) else None

    run_type = str(payload.get("type") or DEFAULT_RUN_TYPE).strip().lower()
    run_id = payload.get("run_id")

    return Request(
        type=run_type,
        run_id=str(run_id) if run_id is not None else None,
        text=text or None,
        files=specs,
    )


# ------------------------------------------------------------------ materialize
def _safe_suffix(name):
    suffix = Path(name or "").suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,6}", suffix or "") else ""


def _decode_base64(data, spec):
    data = data.strip()
    if data.startswith("data:"):
        header, _, data = data.partition(",")
        if not spec.mime_type:
            spec.mime_type = header[5:].split(";")[0] or None
    data = re.sub(r"\s+", "", data)
    data += "=" * (-len(data) % 4)
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        try:
            return base64.urlsafe_b64decode(data)
        except (binascii.Error, ValueError) as exc:
            raise InputError(f"{spec.label}: 'data' is not valid base64.") from exc


def _download(spec, path):
    try:
        with requests.get(spec.url, timeout=DOWNLOAD_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            if not spec.mime_type:
                spec.mime_type = (r.headers.get("content-type") or "").split(";")[0].strip() or None
            size = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise InputError(f"{spec.label} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
                    f.write(chunk)
        return size
    except requests.RequestException as exc:
        raise InputError(f"{spec.label}: download failed: {exc}") from exc


def materialize(spec: FileSpec, directory) -> Path:
    path = Path(directory) / f"file_{spec.index}{_safe_suffix(spec.name)}"
    if spec.data:
        content = _decode_base64(spec.data, spec)
        if len(content) > MAX_FILE_BYTES:
            raise InputError(f"{spec.label} is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        path.write_bytes(content)
    else:
        _download(spec, path)
    if path.stat().st_size == 0:
        raise InputError(f"{spec.label} is empty.")
    return path

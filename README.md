# Nonprofit Document AI Pipeline

RunPod serverless worker that turns PDF / Word / Excel / CSV / TXT / JSON into
program JSON using **Qwen3-8B + LoRA** (PaddleOCR-VL only for scanned PDF pages).

```
Laravel  ──multipart──▶  n8n webhook  ──JSON (base64 files)──▶  RunPod /runsync  ──▶ this worker
   ▲                                                                                   │
   └──────────────── contract object {type, name, year, ...} ◀──────────────────────────┘
```

| file | reader |
|---|---|
| PDF (digital) | PyMuPDF text layer — fast, no OCR |
| PDF (scanned / empty / reversed Arabic pages) | PaddleOCR-VL, page by page |
| DOCX | python-docx (paragraphs + tables + text boxes) |
| DOC | antiword (fallback LibreOffice if installed) |
| XLSX / XLS | openpyxl / xlrd, every sheet |
| CSV / TSV / TXT / MD / JSON | text (UTF-8, UTF-16, cp1256) |

File type is detected from the **content**, not the name (xlsx and docx are both ZIP files).

## RunPod input

```json
{
  "input": {
    "run_id": "42",
    "type": "program_autofill",
    "text": "انظر الملف المرفق",
    "files": [
      { "name": "brief.pdf", "mime_type": "application/pdf", "data": "<base64>" }
    ]
  }
}
```

Also accepted: the contract fields forwarded as-is
(`"input": "{\"text\":\"...\",\"files\":[\"brief.pdf\"]}"` + `"attachments": ["<base64>"]` in the same order),
files by `url`, and the legacy `{"file_url": "..."}`.

`type`:

| type | output |
|---|---|
| `program_autofill` (default) | **one object with exactly the contract fields** — return it to Laravel as-is |
| `program_extract` | `{"programs": [...], "program_count": n, "sources": [...], "warnings": [...]}` |
| `extract_text` | the extracted text only, no model call — use it to debug a file |

## RunPod output (`program_autofill`)

```json
{
  "type": "program",
  "name": "Clean Water Initiative",
  "year": 2025,
  "description": "...",
  "beneficiaries_count": 1200,
  "budget": 45000,
  "beneficiary_value": "...",
  "target_audience": "...",
  "delivery_method": "...",
  "notes": "..."
}
```

Model output is **coerced**, not rejected: `"٢٠٢٥م"` → `2025`, `"1,200 مستفيد"` → `1200`,
`"1.5 مليون"` → `1500000`, `"مشروع"` → `"project"`, empty strings → `null`, years outside
1900–2100 (e.g. Hijri) → `null`. If several programs are found, the most complete one fills
the form and the others are listed in `notes`. Unreadable files and truncation are also
reported in `notes` instead of failing the run.

Errors (no text + no files, filename without content, unsupported `type`, every file
unreadable, model failure) return `{"error": "..."}`, which RunPod marks as **FAILED**.

## n8n workflow

1. **Webhook** — POST, Binary data on, Respond: *When Last Node Finishes*.
2. **Code** — build the RunPod body:
   ```js
   const item = $input.first();
   const body = item.json.body ?? item.json;
   const parsed = typeof body.input === 'string' && body.input ? JSON.parse(body.input) : (body.input ?? {});
   const names = parsed.files ?? [];
   const num = k => parseInt((k.match(/\d+$/) || ['0'])[0], 10);
   const keys = Object.keys(item.binary ?? {}).sort((a, b) => num(a) - num(b));
   const files = [];
   for (let i = 0; i < keys.length; i++) {
     const buf = await this.helpers.getBinaryDataBuffer(0, keys[i]);
     files.push({ name: names[i] ?? item.binary[keys[i]].fileName,
                  mime_type: item.binary[keys[i]].mimeType,
                  data: buf.toString('base64') });
   }
   return [{ json: { input: { run_id: body.run_id, type: body.type, text: parsed.text, files } } }];
   ```
   Check the binary key names in one test execution: files must stay in the same order as `input.files`.
3. **HTTP Request** — POST `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`,
   header `Authorization: Bearer <RUNPOD_API_KEY>`, JSON body `{{ $json }}`.
4. **IF** `{{ $json.status }}` equals `COMPLETED`
   - true → **Code** `return [{ json: $json.output }];` (last node → 200 + the contract object)
   - false → **Stop and Error** with `{{ $json.error ?? $json.status }}` (n8n answers non-2xx → run `failed`)

RunPod limits the `/runsync` payload to ~20 MB, and base64 adds ~33%.

## Latency (Laravel waits only 60 s)

A cold worker downloads Qwen3-8B (~16 GB) and loads two models — far more than 60 s.
To stay under the limit:

- Build with `--build-arg BAKE_QWEN=1`, or attach a network volume (caches go to `/runpod-volume` automatically).
- Keep **at least 1 active worker** on the endpoint (models are preloaded at start, `PRELOAD_MODELS=true`).
- On a GPU with ≥ 24 GB set `LOAD_IN_4BIT=false`: bitsandbytes 4-bit generation is noticeably slower.
- Autofill runs one pass with `AUTOFILL_MAX_NEW_TOKENS=2048` and `AUTOFILL_MAX_INPUT_CHARS=35000`.

If it is still too slow, the backend timeout must be raised or the flow made asynchronous.

## Environment variables

| var | default |
|---|---|
| `QWEN_MODEL_NAME` | `Qwen/Qwen3-8B` (or `/models/Qwen3-8B` if baked) |
| `LORA_PATH` | `/app/lora-release` |
| `LOAD_IN_4BIT` | `true` |
| `PRELOAD_MODELS` | `true` |
| `AUTOFILL_MAX_INPUT_CHARS` / `AUTOFILL_MAX_NEW_TOKENS` | `35000` / `2048` |
| `MAX_INPUT_CHARS` / `QWEN_CHUNK_CHARS` / `MAX_NEW_TOKENS` | `120000` / `35000` / `8192` |
| `ENABLE_OCR` / `PDF_FORCE_OCR` | `true` / `false` |
| `PDF_MIN_CHARS_PER_PAGE` / `PDF_OCR_DPI` / `MAX_PDF_PAGES` | `40` / `200` / `200` |
| `MAX_FILES` / `MAX_FILE_BYTES` | `5` / `26214400` |
| `LOG_TEXT_PREVIEW_CHARS` | `1500` |

## Tests (no GPU needed)

```bash
pip install python-docx openpyxl xlrd pymupdf json-repair requests xlwt pytest
python -m pytest -q tests
```

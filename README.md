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
    "type": "program_extract",
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
| `program_extract` (default) | **JSON array** — all programs first, then all projects, each object with exactly the 10 fields below |
| `program_autofill` | **one object** with the same 10 fields (the Laravel autofill contract needs an object, not an array) |
| `extract_text` | the extracted text only, no model call — use it to debug a file |

## Output (`program_extract`)

```json
[
  {
    "type": "program",
    "name": "Clean Water Initiative",
    "year": 2025,
    "description": "Provides potable water access to rural communities through well construction and maintenance training.",
    "beneficiaries_count": 1200,
    "budget": 45000.0,
    "beneficiary_value": "Each household gains reliable year-round access to clean drinking water.",
    "target_audience": "Rural households without piped water access",
    "delivery_method": "Community-led well construction with quarterly maintenance visits",
    "notes": null
  },
  {
    "type": "project",
    "name": "مشروع كفالة الأيتام",
    "year": 2025,
    "description": null,
    "beneficiaries_count": 300,
    "budget": 150000.0,
    "beneficiary_value": null,
    "target_audience": "الأيتام",
    "delivery_method": null,
    "notes": null
  }
]
```

No programs found → `[]`. File/model warnings are written to the worker logs.

Model output is **coerced**, not rejected: `"٢٠٢٥م"` → `2025`, `"1,200 مستفيد"` → `1200`,
`"1.5 مليون"` → `1500000.0`, `"مشروع"` → `"project"`, empty strings → `null`, years outside
1900–2100 (e.g. Hijri) → `null`. Items without a name are dropped.

`program_autofill` returns one object in the same shape: the most complete item fills the
form, and other items, unreadable files and truncation are listed in its `notes`.

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
   return [{ json: { input: { run_id: body.run_id, type: body.type || 'program_extract', text: parsed.text, files } } }];
   ```
   Check the binary key names in one test execution: files must stay in the same order as `input.files`.
3. **HTTP Request** — POST `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`,
   header `Authorization: Bearer <RUNPOD_API_KEY>`, JSON body `{{ $json }}`.
4. **IF** `{{ $json.status }}` equals `COMPLETED`
   - true, `program_extract` (array) → **Code** `return $json.output.map(p => ({ json: p }));`
     and set the Webhook node's **Response Data = All Entries**. With the default
     "First Entry JSON", n8n answers with the FIRST program only.
   - true, `program_autofill` (object) → **Code** `return [{ json: $json.output }];`, Response Data = First Entry JSON
   - false → **Stop and Error** with `{{ $json.error ?? $json.status }}` (n8n answers non-2xx → run `failed`)

RunPod limits the `/runsync` payload to ~20 MB, and base64 adds ~33%.

## Latency (Laravel waits only 60 s)

A cold worker downloads Qwen3-8B (~16 GB) and loads two models — far more than 60 s.
To stay under the limit:

- In the endpoint set **Cached model** = `https://huggingface.co/Qwen/Qwen3-8B` (picked up automatically from `/runpod-volume/huggingface-cache`), or build with `--build-arg BAKE_QWEN=1`.
- Keep **at least 1 active worker** on the endpoint (models are preloaded at start, `PRELOAD_MODELS=true`).
- On a GPU with ≥ 24 GB set `LOAD_IN_4BIT=false`: bitsandbytes 4-bit generation is noticeably slower.
- Autofill runs one pass with `AUTOFILL_MAX_NEW_TOKENS=2048` and `AUTOFILL_MAX_INPUT_CHARS=35000`.

If it is still too slow, the backend timeout must be raised or the flow made asynchronous.

## Environment variables

| var | default |
|---|---|
| `QWEN_MODEL_NAME` | `Qwen/Qwen3-8B` — resolved to the baked copy or the RunPod cached snapshot when present |
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

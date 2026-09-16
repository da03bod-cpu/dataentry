import logging
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    DO_SAMPLE,
    LOAD_IN_4BIT,
    LOG_TEXT_PREVIEW_CHARS,
    MAX_NEW_TOKENS,
    QWEN_CHUNK_CHARS,
    QWEN_MODEL_NAME,
    TEMPERATURE,
)
from pipeline.json_utils import parse_program_items
from pipeline.prompt import build_messages

log = logging.getLogger("qwen")

_tokenizer = None
_model = None
_lock = threading.Lock()


def load_model():
    """Load the BASE Qwen model only.

    This extraction endpoint intentionally does not load the nonprofit LoRA.
    A memorized fine-tuning adapter can leak training examples into extraction,
    which violates the requirement that every returned fact comes from the file.
    """
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model
    with _lock:
        if _model is not None:
            return _tokenizer, _model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available in this worker.")

        log.info("Loading BASE model %s (4bit=%s), no LoRA adapter", QWEN_MODEL_NAME, LOAD_IN_4BIT)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
        if LOAD_IN_4BIT:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_NAME, **kwargs)
        model.eval()
        _tokenizer, _model = tokenizer, model
        log.info("Base Qwen loaded")
    return _tokenizer, _model


# ------------------------------------------------------------------ chunking
_SPLIT_BEFORE = re.compile(r"(?=^===== (?:TABLE START|PAGE \d+|SHEET: .*|FILE \d+.*|USER TEXT) =====$)", re.M)


def split_document(text, chunk_chars=QWEN_CHUNK_CHARS):
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]

    chunks, current = [], ""
    for block in (b.strip() for b in _SPLIT_BEFORE.split(text)):
        if not block:
            continue
        candidate = f"{current}\n{block}" if current else block
        if current and len(candidate) > chunk_chars:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)

    final = []
    for chunk in chunks:
        if len(chunk) <= chunk_chars:
            final.append(chunk)
            continue
        buf = ""
        for line in chunk.splitlines():
            while len(line) > chunk_chars:
                if buf:
                    final.append(buf)
                    buf = ""
                final.append(line[:chunk_chars])
                line = line[chunk_chars:]
            candidate = f"{buf}\n{line}" if buf else line
            if buf and len(candidate) > chunk_chars:
                final.append(buf)
                buf = line
            else:
                buf = candidate
        if buf:
            final.append(buf)
    return final


# ------------------------------------------------------------------ generation
def _generate_one(document_text, chunk_index, total_chunks, max_new_tokens):
    tokenizer, model = load_model()
    messages = build_messages(document_text, chunk_index, total_chunks)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    log.info("chunk %d/%d: %d chars, %d input tokens, max_new_tokens=%d",
             chunk_index, total_chunks, len(document_text), input_tokens, max_new_tokens)

    gen = {
        "max_new_tokens": max_new_tokens,
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if DO_SAMPLE:
        gen["temperature"] = TEMPERATURE
    else:
        gen.update(temperature=None, top_p=None, top_k=None)

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen)

    generated = outputs[0][input_tokens:]
    output_tokens = int(generated.shape[-1])
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    truncated = output_tokens >= max_new_tokens

    log.info("chunk %d/%d: %d output tokens%s", chunk_index, total_chunks, output_tokens,
             " (HIT max_new_tokens)" if truncated else "")
    log.info("RAW OUTPUT (first %d chars):\n%s", LOG_TEXT_PREVIEW_CHARS, decoded[:LOG_TEXT_PREVIEW_CHARS])
    return parse_program_items(decoded, truncated=truncated), truncated


def generate_program_items(document_text, max_new_tokens=None, chunk_chars=None):
    max_new_tokens = max_new_tokens or MAX_NEW_TOKENS
    chunks = split_document(document_text, chunk_chars or QWEN_CHUNK_CHARS)
    log.info("document: %d chars -> %d chunk(s)", len(document_text), len(chunks))

    items, errors, truncated_any = [], [], False
    for index, chunk in enumerate(chunks, 1):
        try:
            chunk_items, truncated = _generate_one(chunk, index, len(chunks), max_new_tokens)
            items.extend(chunk_items)
            truncated_any = truncated_any or truncated
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
        except Exception as exc:
            log.exception("chunk %d/%d failed", index, len(chunks))
            errors.append(f"chunk {index}: {exc}")

    if errors and len(errors) == len(chunks):
        raise RuntimeError("Model extraction failed: " + "; ".join(errors))
    return items, {"chunks": len(chunks), "chunk_errors": errors, "truncated": truncated_any}

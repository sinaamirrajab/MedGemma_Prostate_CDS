#!/usr/bin/env python3
"""Minimal HTTP API that serves MedGemma treatment recommendations."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _select_writable_dir(preferred: str, fallback: str) -> str:
    for candidate in (preferred, fallback):
        try:
            os.makedirs(candidate, exist_ok=True)
            probe = os.path.join(candidate, ".write_probe")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"No writable directory among: {preferred}, {fallback}")


def _read_token_from_file(path: str) -> str:
    try:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if not raw:
            return ""
        if raw.startswith("{"):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return (
                    parsed.get("token")
                    or parsed.get("hf_token")
                    or parsed.get("access_token")
                    or ""
                )
        return raw.splitlines()[0].strip()
    except Exception:
        return ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


LOGGER = logging.getLogger("medgemma_api")

MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-1.5-4b-it")
HOST = os.environ.get("MEDGEMMA_API_HOST", "127.0.0.1")
PORT = int(os.environ.get("MEDGEMMA_API_PORT", "8000"))
PDQ_PATH = os.environ.get("PDQ_PDF_PATH", "PDQ.pdf")
PROMPT_LOG_PATH = os.environ.get(
    "MEDGEMMA_PROMPT_LOG_PATH", "backend/medgemma_prompts_debug.txt"
)
OUTPUT_LOG_PATH = os.environ.get(
    "MEDGEMMA_OUTPUT_LOG_PATH", "backend/medgemma_outputs_log.jsonl"
)
RECOMMEND_MAX_NEW_TOKENS = max(256, _env_int("MEDGEMMA_RECOMMEND_MAX_NEW_TOKENS", 1200))
RECOMMEND_RETRY_MAX_NEW_TOKENS = max(
    RECOMMEND_MAX_NEW_TOKENS,
    _env_int("MEDGEMMA_RECOMMEND_RETRY_MAX_NEW_TOKENS", 1600),
)
RECOMMEND_MIN_NEW_TOKENS = max(64, _env_int("MEDGEMMA_RECOMMEND_MIN_NEW_TOKENS", 128))
CHAT_MAX_NEW_TOKENS = max(256, _env_int("MEDGEMMA_CHAT_MAX_NEW_TOKENS", 700))
CHAT_RETRY_MAX_NEW_TOKENS = max(
    CHAT_MAX_NEW_TOKENS,
    _env_int("MEDGEMMA_CHAT_RETRY_MAX_NEW_TOKENS", 900),
)
CHAT_MIN_NEW_TOKENS = max(24, _env_int("MEDGEMMA_CHAT_MIN_NEW_TOKENS", 48))

DEFAULT_HF_HOME = _select_writable_dir(
    os.environ.get("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")),
    "/tmp/medgemma_hf_cache",
)
os.environ["HF_HOME"] = DEFAULT_HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = _select_writable_dir(
    os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub")),
    os.path.join(DEFAULT_HF_HOME, "hub"),
)
os.environ["TRANSFORMERS_CACHE"] = _select_writable_dir(
    os.environ.get("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers")),
    os.path.join(DEFAULT_HF_HOME, "transformers"),
)
os.makedirs(os.environ["HUGGINGFACE_HUB_CACHE"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
LEGACY_HF_HUB = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


@dataclass
class RecommenderState:
    tokenizer: Any | None = None
    model: Any | None = None
    load_error: str | None = None
    rag_chunks: list[dict[str, Any]] | None = None
    active_gpu: int | None = None
    failed_gpus: set[int] = field(default_factory=set)
    hf_token: str | None = None
    hf_token_checked: bool = False


STATE = RecommenderState()
MODEL_LOCK = threading.RLock()


class RecommendationParseError(RuntimeError):
    def __init__(self, message: str, raw_output: str) -> None:
        super().__init__(message)
        self.raw_output = raw_output


class ApiError(RuntimeError):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _strip_pdq_page_citations(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"\[\s*PDQ\s*p\.\s*\d+\s*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bPDQ\s*p\.\s*\d+\b", "PDQ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _append_prompt_log(kind: str, prompt: str) -> None:
    try:
        log_dir = os.path.dirname(PROMPT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(f"\n\n===== {kind} | {timestamp} =====\n")
            handle.write(prompt)
            handle.write("\n")
    except Exception as exc:  # pragma: no cover - logging should not break flow
        LOGGER.warning("Could not write prompt log: %s", exc)


def _append_output_log(
    kind: str, output: str, metadata: dict[str, Any] | None = None
) -> None:
    try:
        log_dir = os.path.dirname(OUTPUT_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "output": output if isinstance(output, str) else str(output),
        }
        if metadata:
            record["metadata"] = metadata
        with open(OUTPUT_LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    except Exception as exc:  # pragma: no cover - logging should not break flow
        LOGGER.warning("Could not write output log: %s", exc)


def _is_cuda_assert_error(exc: Exception) -> bool:
    text = str(exc).lower()
    indicators = (
        "cuda error",
        "device-side assert",
        "cublas_status",
        "an illegal memory access",
    )
    return any(token in text for token in indicators)


def _parse_gpu_ids(raw: str, max_count: int) -> list[int]:
    if not raw.strip():
        return list(range(max_count))
    parsed: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            continue
        idx = int(token)
        if 0 <= idx < max_count and idx not in parsed:
            parsed.append(idx)
    return parsed


def _candidate_gpus() -> list[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        total = torch.cuda.device_count()
    except Exception:
        return []

    requested = _parse_gpu_ids(os.environ.get("MEDGEMMA_GPU_IDS", ""), total)
    all_ids = list(range(total))
    candidates = [gpu for gpu in requested if gpu not in STATE.failed_gpus]
    if not candidates:
        # If requested GPUs are exhausted, try any other visible GPU before giving up.
        backup = [gpu for gpu in all_ids if gpu not in STATE.failed_gpus]
        if backup:
            candidates = backup
        elif STATE.failed_gpus:
            STATE.failed_gpus.clear()
            candidates = requested if requested else all_ids
    if STATE.active_gpu is not None and STATE.active_gpu in candidates:
        candidates = [STATE.active_gpu] + [gpu for gpu in candidates if gpu != STATE.active_gpu]
    return candidates


def _get_hf_token() -> str:
    if STATE.hf_token_checked:
        return STATE.hf_token or ""

    env_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    if env_token:
        STATE.hf_token = env_token
        STATE.hf_token_checked = True
        return env_token

    home = os.path.expanduser("~")
    candidate_files = [
        os.path.join(home, ".cache", "huggingface", "token"),
        os.path.join(home, ".huggingface", "token"),
        os.path.join(os.environ.get("HF_HOME", ""), "token"),
    ]
    for token_file in candidate_files:
        token = _read_token_from_file(token_file)
        if token:
            STATE.hf_token = token
            STATE.hf_token_checked = True
            os.environ["HF_TOKEN"] = token
            return token

    STATE.hf_token = ""
    STATE.hf_token_checked = True
    return ""


def _resolve_local_model_path(model_id: str) -> str:
    explicit = os.environ.get("MEDGEMMA_MODEL_PATH", "").strip()
    if explicit and os.path.isdir(explicit):
        return explicit

    if os.path.isdir(model_id):
        return model_id

    repo_folder = f"models--{model_id.replace('/', '--')}"
    snapshots_dir = os.path.join(LEGACY_HF_HUB, repo_folder, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshot_candidates = sorted(
            [
                os.path.join(snapshots_dir, entry)
                for entry in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, entry))
            ],
            reverse=True,
        )
        for candidate in snapshot_candidates:
            config_path = os.path.join(candidate, "config.json")
            if os.path.isfile(config_path):
                return candidate
    return ""


def _mark_gpu_failed(reason: str) -> None:
    previous = STATE.active_gpu
    if previous is not None:
        STATE.failed_gpus.add(previous)
    STATE.model = None
    STATE.load_error = None
    STATE.active_gpu = None
    LOGGER.error("GPU failure on cuda:%s: %s", previous, reason)
    _append_prompt_log("gpu_failure", f"cuda:{previous} | {reason}")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def _max_context_tokens() -> int:
    if STATE.model is None:
        return 4096
    config = getattr(STATE.model, "config", None)
    for key in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return value
    return 4096


def _model_device() -> Any:
    if STATE.model is None:
        raise RuntimeError("Model not initialized.")
    try:
        return next(STATE.model.parameters()).device
    except Exception:
        return STATE.model.device


def _tokenize_prompt_for_generation(
    prompt: str, reserve_new_tokens: int, prefer_recent_tokens: bool = False
) -> Any:
    if STATE.tokenizer is None or STATE.model is None:
        _load_model()
    if STATE.tokenizer is None or STATE.model is None:
        raise RuntimeError(STATE.load_error or "Tokenizer/model not initialized.")

    import torch

    model_ctx = _max_context_tokens()
    safe_reserve = max(64, reserve_new_tokens)
    max_input_tokens = max(256, model_ctx - safe_reserve)

    tokenizer = STATE.tokenizer
    # Match notebook behavior: use chat template for instruction-tuned MedGemma.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            chat_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if chat_ids.shape[-1] > max_input_tokens:
                chat_ids = (
                    chat_ids[:, -max_input_tokens:]
                    if prefer_recent_tokens
                    else chat_ids[:, :max_input_tokens]
                )
            inputs = {
                "input_ids": chat_ids,
                "attention_mask": torch.ones_like(chat_ids),
            }
            return {key: value.to(_model_device()) for key, value in inputs.items()}
        except Exception:
            pass

    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left" if prefer_recent_tokens else "right"
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
    finally:
        tokenizer.truncation_side = original_truncation_side

    return {key: value.to(_model_device()) for key, value in inputs.items()}


def _chunk_text(text: str, page: int, chunk_size: int = 900, overlap: int = 160) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            token_counts = Counter(_tokenize(chunk))
            if sum(token_counts.values()) < 40:
                if end >= length:
                    break
                start = max(0, end - overlap)
                continue
            chunks.append(
                {
                    "page": page,
                    "text": chunk,
                    "token_counts": token_counts,
                }
            )
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def _sanitize_pdf_text(text: str) -> str:
    if not text:
        return ""
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "https://www.cancer.gov/types/prostate/hp/prostate-treatment-pdq" in line:
            continue
        if re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", line) and "PDQ" in line:
            continue
        if re.search(r"\b\d+/\d+\b", line) and "PDQ" in line:
            continue
        if re.fullmatch(r"[•\-\s]+", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _load_rag_chunks() -> list[dict[str, Any]]:
    if STATE.rag_chunks is not None:
        return STATE.rag_chunks

    chunks: list[dict[str, Any]] = []
    try:
        from pypdf import PdfReader

        if not os.path.exists(PDQ_PATH):
            LOGGER.warning("PDQ file not found at %s, continuing without RAG", PDQ_PATH)
            STATE.rag_chunks = []
            return STATE.rag_chunks

        reader = PdfReader(PDQ_PATH)
        for idx, page in enumerate(reader.pages):
            page_text = _sanitize_pdf_text(page.extract_text() or "")
            chunks.extend(_chunk_text(page_text, idx + 1))
        LOGGER.info("Loaded %s RAG chunks from %s", len(chunks), PDQ_PATH)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.exception("Failed to load PDQ RAG context: %s", exc)
        chunks = []

    STATE.rag_chunks = chunks
    return STATE.rag_chunks


def _retrieve_rag_context(patient: dict[str, Any], model_prediction: dict[str, Any], top_k: int = 4) -> str:
    chunks = _load_rag_chunks()
    if not chunks:
        return "No PDQ context available."

    query = (
        f"prostate cancer localized treatment options radical prostatectomy radiation therapy adt "
        f"active surveillance risk group csPCa {model_prediction.get('csPcaPrediction')} "
        f"PSA {patient.get('psa')} c{patient.get('tStage')} c{patient.get('nStage')} c{patient.get('mStage')} "
        f"ECOG {patient.get('performance')} comorbidities {' '.join(patient.get('comorbidities') or [])} "
        f"preferences {patient.get('preferencesTreatmentIntensity')} {patient.get('preferencesSexualFunction')}"
    )
    query_tokens = Counter(_tokenize(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_tokens = chunk["token_counts"]
        overlap = sum(min(query_tokens[tok], chunk_tokens.get(tok, 0)) for tok in query_tokens)
        if overlap <= 0:
            continue
        score = float(overlap) / max(20.0, len(chunk_tokens))
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [item[1] for item in scored[:top_k]]
    if not top_chunks:
        return "No relevant PDQ context retrieved."

    context_lines = []
    for chunk in top_chunks:
        cleaned = re.sub(r"\s+", " ", chunk["text"]).strip()
        context_lines.append(f"[PDQ] {cleaned}")
    return "\n".join(context_lines)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    payload = (text or "").strip()
    if not payload:
        return None

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", payload, flags=re.IGNORECASE)
    candidate_blocks = []
    if fenced:
        candidate_blocks.append(fenced.group(1))
    candidate_blocks.append(payload)

    def _remove_trailing_commas(value: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", value)

    for block in candidate_blocks:
        brace_match = re.search(r"\{[\s\S]*\}", block)
        if not brace_match:
            continue
        snippet = brace_match.group(0)
        for candidate in (snippet, _remove_trailing_commas(snippet)):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def _normalize_recommendation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("MedGemma output is not a JSON object.")

    if not isinstance(payload.get("summary"), str) or not payload["summary"].strip():
        raise ValueError("MedGemma JSON missing non-empty 'summary'.")

    options = payload.get("options")
    if not isinstance(options, list) or len(options) == 0:
        raise ValueError("MedGemma JSON missing 'options' list.")

    normalized_options: list[dict[str, str]] = []
    for option in options[:3]:
        if not isinstance(option, dict):
            raise ValueError("Each option must be an object with title/reasoning/description.")
        title = str(option.get("title", "")).strip()
        rationale = _strip_pdq_page_citations(
            str(option.get("reasoning") or option.get("rationale") or "").strip()
        )
        details = _strip_pdq_page_citations(
            str(option.get("description") or option.get("details") or "").strip()
        )
        if not title:
            raise ValueError("Option is missing title.")
        normalized_options.append(
            {
                "title": title,
                "rationale": rationale,
                "details": details,
                "reasoning": rationale,
                "description": details,
            }
        )

    payload["summary"] = _strip_pdq_page_citations(str(payload["summary"]).strip())
    payload["fullDetails"] = _strip_pdq_page_citations(
        str(payload.get("fullDetails", payload["summary"])).strip()
    )
    payload["options"] = normalized_options
    payload["structuredDetails"] = {}
    payload["note"] = _strip_pdq_page_citations(str(payload.get("note", "")).strip())
    payload["from_medgemma"] = True
    payload["rag_used"] = True
    return payload


def _parse_recommendation_output(generated: str) -> dict[str, Any]:
    json_obj = _extract_json_object(generated)
    if json_obj is None:
        raise ValueError("MedGemma did not return valid JSON output.")
    return _normalize_recommendation_payload(json_obj)


def _build_prompt(
    patient: dict[str, Any], model_prediction: dict[str, Any], rag_context: str
) -> str:
    patient_block = json.dumps(patient, ensure_ascii=True, indent=2)
    prediction_block = json.dumps(model_prediction, ensure_ascii=True, indent=2)
    return f"""
You are an oncology assistant for prostate cancer.
Use the full case data and AI prediction below plus PDQ context to recommend treatment.
Return STRICT JSON only. Do not use markdown, code fences, or extra keys.

Clinical data:
{patient_block}

AI prediction:
{prediction_block}

Required JSON schema:
{{
  "summary": "brief overall recommendation summary",
  "options": [
    {{
      "title": "treatment option name",
      "reasoning": "why this option fits this patient, personalized to inputs and PDQ evidence",
      "description": "what the treatment involves, expected benefits/risks, and practical considerations"
    }},
    {{
      "title": "second option",
      "reasoning": "patient-specific reasoning grounded in PDQ evidence",
      "description": "treatment description"
    }},
    {{
      "title": "third option",
      "reasoning": "patient-specific reasoning grounded in PDQ evidence",
      "description": "treatment description"
    }}
  ],
  "note": "clinician safety caveat"
}}
Rules:
- Provide exactly 3 ranked treatment options.
- Keep output valid JSON.
- Keep each option's reasoning and description concise (about 2-4 sentences each).
- End only after a complete JSON object (all quotes/braces closed).
- If evidence is uncertain, state uncertainty in reasoning.
- Do not include page-number citations such as PDQ p.74 or [PDQ p.74].

PDQ RAG context:
{rag_context}
""".strip()


def _build_repair_prompt(base_prompt: str, previous_output: str, parse_error: str) -> str:
    previous = (previous_output or "").strip()
    if len(previous) > 3000:
        previous = previous[-3000:]
    return f"""
{base_prompt}

Your previous response was invalid or truncated JSON.
Parsing error: {parse_error}
Return a NEW complete JSON object now.
Do not include markdown or code fences.
Ensure all quotes, brackets, and braces are closed.

Previous invalid output (tail):
{previous}
""".strip()


def _looks_like_internal_reasoning(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if "<unused" in lowered:
        return True
    if "<thought>" in lowered or "<response>" in lowered:
        return True
    if re.match(r"^\s*(thought|analysis|reasoning|plan)\b", lowered):
        return True
    if "the user is asking" in lowered and re.search(r"\b1\.\s", candidate):
        return True
    if "the user wants me to" in lowered:
        return True
    if "identify the core concept" in lowered or "recall/search" in lowered:
        return True
    # Heuristic: numbered planning steps at the start = internal reasoning
    if re.match(r"^\s*1\.\s+\*{0,2}(identify|define|recall|locate|extract|determine)", lowered):
        return True
    return False


def _sanitize_chat_response(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    # The model uses <unusedN> as structural tokens:
    #   <unused94> = "begin thinking"  |  <unused95> = "begin response"
    # Take everything after the LAST <unusedN> tag; that is the actual reply.
    last_unused_match = None
    for m in re.finditer(r"<unused\d+>", cleaned, flags=re.IGNORECASE):
        last_unused_match = m
    if last_unused_match:
        after = cleaned[last_unused_match.end():].strip()
        if after:  # non-empty reply follows the last token
            cleaned = after
        else:
            # Nothing after the last token; strip ALL <unusedN> tags and fall through
            cleaned = re.sub(r"<unused\d+>", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip <thought>...</thought> blocks entirely (keep nothing inside)
    cleaned = re.sub(r"(?is)<thought>.*?</thought>", "", cleaned).strip()

    # Unwrap <response>...</response>, keeping only inner content
    response_block = re.search(r"(?is)<response>(.*?)</response>", cleaned)
    if response_block:
        cleaned = response_block.group(1).strip()
    else:
        # No closing tag: take everything after an opening <response>
        response_open = re.search(r"(?is)<response>", cleaned)
        if response_open:
            cleaned = cleaned[response_open.end():].strip()

    # Remove any residual <...> tags (safety net)
    cleaned = re.sub(r"<[^>]{1,40}>", "", cleaned).strip()

    # Strip leading role labels that models sometimes emit
    cleaned = re.sub(r"^\s*assistant\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*thought\b\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^\s*internal reasoning\b\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Collapse excessive blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return _strip_pdq_page_citations(cleaned)


def _build_chat_cleanup_prompt(user_message: str, raw_response: str) -> str:
    previous = (raw_response or "").strip()
    if len(previous) > 2500:
        previous = previous[-2500:]
    return f"""
Rewrite the previous draft as the final user-facing answer.
Do not include internal reasoning, planning, or hidden tokens.
Do not include tags like <unused...> or labels like "thought".
Use readable markdown with short paragraphs and bullet points when useful.

User question:
{user_message}

Previous draft to clean:
{previous}
""".strip()


def _build_chat_prompt(
    patient: dict[str, Any],
    model_prediction: dict[str, Any],
    recommendation: dict[str, Any],
    rag_context: str,
    messages: list[dict[str, str]],
    user_message: str,
) -> str:
    history_lines = []
    for msg in messages[-8:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = (msg.get("content") or "").strip()
        if content:
            history_lines.append(f"{role}: {content}")
    conversation_history = "\n".join(history_lines) or "No prior chat turns."
    compact_recommendation = {
        "summary": recommendation.get("summary"),
        "topOptions": [
            {
                "title": item.get("title"),
                "reasoning": item.get("reasoning") or item.get("rationale"),
                "description": item.get("description") or item.get("details"),
            }
            for item in (recommendation.get("options") or [])[:3]
            if isinstance(item, dict)
        ],
        "note": recommendation.get("note"),
    }
    return f"""
You are MedGemma acting as a prostate cancer clinical assistant.
The information below is context for this turn only.
Use it to ground your response to the user's latest question.
This is context, not an output template.

Clinical data:
{json.dumps(patient, ensure_ascii=True, indent=2)}

AI prediction data:
{json.dumps(model_prediction, ensure_ascii=True, indent=2)}

Current treatment recommendation object:
{json.dumps(compact_recommendation, ensure_ascii=True, indent=2)}

Conversation history:
{conversation_history}

PDQ RAG context:
{rag_context}

Use PDQ evidence when relevant, but do not include page-number citations.
Answer naturally and clinically.
Use readable markdown formatting (short sections and bullet points when helpful).
Do not include internal reasoning text, planning traces, or tags like <unused...> or "thought".

Latest user message:
{user_message}
""".strip()


def _load_model() -> None:
    if STATE.model is not None:
        return
    STATE.load_error = None

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        local_model_path = _resolve_local_model_path(MODEL_ID)
        model_source = local_model_path or MODEL_ID
        local_only = bool(local_model_path)
        hf_token = _get_hf_token() if not local_only else ""
        if not local_only and not hf_token:
            raise RuntimeError(
                "No Hugging Face token found for gated model access. "
                "Set HF_TOKEN or run huggingface-cli login."
            )

        if STATE.tokenizer is None:
            tokenizer_kwargs: dict[str, Any] = {
                "cache_dir": os.environ["TRANSFORMERS_CACHE"],
            }
            if local_only:
                tokenizer_kwargs["local_files_only"] = True
            else:
                tokenizer_kwargs["token"] = hf_token
            STATE.tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)

        gpu_candidates = _candidate_gpus()
        if not gpu_candidates:
            raise RuntimeError(
                "No CUDA GPUs available for MedGemma in GPU-only mode. "
                "Check CUDA visibility and MEDGEMMA_GPU_IDS."
            )

        load_errors: list[str] = []
        for gpu_id in gpu_candidates:
            try:
                torch.cuda.set_device(gpu_id)
                model_kwargs: dict[str, Any] = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": {"": f"cuda:{gpu_id}"},
                    "cache_dir": os.environ["TRANSFORMERS_CACHE"],
                }
                if local_only:
                    model_kwargs["local_files_only"] = True
                else:
                    model_kwargs["token"] = hf_token
                STATE.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    **model_kwargs,
                )
                STATE.model.eval()
                STATE.active_gpu = gpu_id
                LOGGER.info(
                    "Loaded model %s on cuda:%s (%s source)",
                    MODEL_ID,
                    gpu_id,
                    "local-cache" if local_only else "remote",
                )
                return
            except Exception as gpu_exc:  # pragma: no cover - environment dependent
                load_errors.append(f"cuda:{gpu_id} -> {gpu_exc}")
                STATE.model = None
                STATE.active_gpu = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        raise RuntimeError(" | ".join(load_errors))
    except Exception as exc:  # pragma: no cover - environment dependent
        STATE.load_error = str(exc)
        LOGGER.exception("Failed to load %s", MODEL_ID)


def _generate_text(
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
    prefer_recent_tokens: bool = False,
    output_kind: str = "model_output",
    output_metadata: dict[str, Any] | None = None,
) -> str:
    with MODEL_LOCK:
        _load_model()
        if STATE.model is None or STATE.tokenizer is None:
            raise RuntimeError(STATE.load_error or "MedGemma is unavailable.")

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": STATE.tokenizer.eos_token_id,
            "pad_token_id": STATE.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature if temperature is not None else 0.35
            generation_kwargs["top_p"] = top_p if top_p is not None else 0.9

        for attempt in range(2):
            try:
                if STATE.model is None or STATE.tokenizer is None:
                    _load_model()
                    if STATE.model is None or STATE.tokenizer is None:
                        raise RuntimeError(STATE.load_error or "MedGemma is unavailable.")

                inputs = _tokenize_prompt_for_generation(
                    prompt,
                    reserve_new_tokens=max_new_tokens + 16,
                    prefer_recent_tokens=prefer_recent_tokens,
                )
                import torch

                with torch.inference_mode():
                    output_ids = STATE.model.generate(**inputs, **generation_kwargs)
                generated_text = STATE.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
                ).strip()
                _append_output_log(output_kind, generated_text, output_metadata)
                return generated_text
            except RuntimeError as exc:
                if _is_cuda_assert_error(exc) and attempt == 0:
                    _mark_gpu_failed(str(exc))
                    _load_model()
                    if STATE.model is None or STATE.tokenizer is None:
                        raise RuntimeError(STATE.load_error or str(exc)) from exc
                    continue
                raise

    return ""


def _generate_with_medgemma(
    patient: dict[str, Any], model_prediction: dict[str, Any], rag_context: str
) -> dict[str, Any]:
    base_prompt = _build_prompt(patient, model_prediction, rag_context)
    _append_prompt_log("recommendation_prompt", base_prompt)

    attempt_specs = (
        {
            "max_new_tokens": RECOMMEND_MAX_NEW_TOKENS,
            "min_new_tokens": RECOMMEND_MIN_NEW_TOKENS,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
        {
            "max_new_tokens": RECOMMEND_RETRY_MAX_NEW_TOKENS,
            "min_new_tokens": RECOMMEND_MIN_NEW_TOKENS,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
        },
    )

    last_generated = ""
    last_error: Exception | None = None

    for attempt_index, spec in enumerate(attempt_specs, start=1):
        prompt = (
            base_prompt
            if attempt_index == 1
            else _build_repair_prompt(base_prompt, last_generated, str(last_error or "parse error"))
        )
        if attempt_index > 1:
            _append_prompt_log(f"recommendation_prompt_retry_{attempt_index}", prompt)

        generated = _generate_text(
            prompt=prompt,
            max_new_tokens=spec["max_new_tokens"],
            min_new_tokens=spec["min_new_tokens"],
            do_sample=spec["do_sample"],
            temperature=spec["temperature"],
            top_p=spec["top_p"],
            prefer_recent_tokens=False,
            output_kind=f"recommendation_raw_output_attempt_{attempt_index}",
            output_metadata={
                "attempt": attempt_index,
                "max_new_tokens": spec["max_new_tokens"],
            },
        )
        _append_prompt_log(f"recommendation_raw_output_attempt_{attempt_index}", generated)
        if attempt_index == 1:
            # Preserve legacy key in prompt debug log.
            _append_prompt_log("recommendation_raw_output", generated)

        last_generated = generated
        try:
            parsed = _parse_recommendation_output(generated)
            parsed.setdefault(
                "note",
                "Generated by MedGemma. Clinician review is required before acting on recommendations.",
            )
            _append_output_log(
                "recommendation_parsed_output",
                json.dumps(parsed, ensure_ascii=False),
                {"attempt": attempt_index},
            )
            return parsed
        except Exception as exc:
            last_error = exc
            _append_output_log(
                "recommendation_parse_error",
                str(exc),
                {"attempt": attempt_index},
            )

    raise RecommendationParseError(
        str(last_error or "MedGemma did not return valid JSON output."),
        last_generated,
    )


def _generate_chat_response(
    patient: dict[str, Any],
    model_prediction: dict[str, Any],
    recommendation: dict[str, Any],
    rag_context: str,
    messages: list[dict[str, str]],
    user_message: str,
) -> str:
    prompt = _build_chat_prompt(
        patient, model_prediction, recommendation, rag_context, messages, user_message
    )
    _append_prompt_log("chat_prompt", prompt)

    generation_attempts = (
        {"max_new_tokens": CHAT_MAX_NEW_TOKENS, "min_new_tokens": CHAT_MIN_NEW_TOKENS, "do_sample": False},
        {
            "max_new_tokens": CHAT_RETRY_MAX_NEW_TOKENS,
            "min_new_tokens": CHAT_MIN_NEW_TOKENS,
            "do_sample": True,
            "temperature": 0.35,
            "top_p": 0.9,
        },
    )

    last_raw_response = ""
    for attempt_index, attempt in enumerate(generation_attempts, start=1):
        raw_response = _generate_text(
            prompt=prompt,
            max_new_tokens=attempt["max_new_tokens"],
            min_new_tokens=attempt["min_new_tokens"],
            do_sample=attempt["do_sample"],
            temperature=attempt.get("temperature"),
            top_p=attempt.get("top_p"),
            prefer_recent_tokens=True,
            output_kind=f"chat_raw_output_attempt_{attempt_index}",
            output_metadata={"attempt": attempt_index, "stage": "primary"},
        )
        last_raw_response = raw_response
        _append_prompt_log(f"chat_raw_output_attempt_{attempt_index}", raw_response)
        if attempt_index == 1:
            # Keep legacy log key for continuity with existing debugging workflow.
            _append_prompt_log("chat_raw_output", raw_response)

        response = _sanitize_chat_response(raw_response)
        if response and not _looks_like_internal_reasoning(response):
            _append_output_log(
                "chat_selected_output",
                response,
                {"attempt": attempt_index, "stage": "primary", "sanitized": True},
            )
            return response

    shorter_prompt = _build_chat_prompt(
        patient,
        model_prediction,
        recommendation,
        _retrieve_rag_context(patient, model_prediction, top_k=2),
        messages[-4:],
        user_message,
    )
    _append_prompt_log("chat_prompt_retry_short", shorter_prompt)
    raw_retry_response = _generate_text(
        prompt=shorter_prompt,
        max_new_tokens=CHAT_RETRY_MAX_NEW_TOKENS,
        min_new_tokens=CHAT_MIN_NEW_TOKENS,
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        prefer_recent_tokens=True,
        output_kind="chat_raw_output_retry_short",
        output_metadata={"stage": "retry_short"},
    )
    _append_prompt_log("chat_raw_output_retry_short", raw_retry_response)
    last_raw_response = raw_retry_response
    retry_response = _sanitize_chat_response(raw_retry_response)
    if retry_response and not _looks_like_internal_reasoning(retry_response):
        _append_output_log(
            "chat_selected_output",
            retry_response,
            {"stage": "retry_short", "sanitized": True},
        )
        return retry_response

    cleanup_prompt = _build_chat_cleanup_prompt(user_message, last_raw_response)
    _append_prompt_log("chat_prompt_cleanup", cleanup_prompt)
    cleanup_raw_response = _generate_text(
        prompt=cleanup_prompt,
        max_new_tokens=CHAT_MAX_NEW_TOKENS,
        min_new_tokens=CHAT_MIN_NEW_TOKENS,
        do_sample=False,
        prefer_recent_tokens=True,
        output_kind="chat_raw_output_cleanup",
        output_metadata={"stage": "cleanup"},
    )
    _append_prompt_log("chat_raw_output_cleanup", cleanup_raw_response)
    cleanup_response = _sanitize_chat_response(cleanup_raw_response)
    if cleanup_response:
        _append_output_log(
            "chat_selected_output",
            cleanup_response,
            {"stage": "cleanup", "sanitized": True},
        )
        return cleanup_response

    return _sanitize_chat_response(last_raw_response)


def _fallback_chat_response(
    patient: dict[str, Any], recommendation: dict[str, Any], user_message: str
) -> str:
    text = user_message.lower()
    if "medical history" in text or "history" in text:
        comorbidities = ", ".join(patient.get("comorbidities") or []) or "none reported"
        return (
            "Medical history directly affects treatment ranking because it changes safety, "
            "tolerability, and quality-of-life tradeoffs. In this case, comorbidities "
            f"({comorbidities}), ECOG ({patient.get('performance')}), urinary symptoms "
            f"({patient.get('lowerUrinarySymptoms')}), and patient priorities are used to "
            "balance surgery versus radiation versus surveillance. For example, higher "
            "cardiopulmonary/surgical risk can shift preference away from radical prostatectomy, "
            "while baseline urinary or sexual-function concerns can change how radiation and "
            "surgery are weighted. The recommendation object is therefore conditioned on both "
            "oncologic risk and medical-history context."
        )
    return (
        "I could not generate a model response, so this is a fallback answer. "
        "The treatment recommendation is based on staging, PSA profile, predicted csPCa risk, "
        "medical history/comorbidities, and patient preferences, with PDQ RAG context used "
        "for evidence-grounded rationale."
    )


def _validate_json_object(body: Any, field_name: str = "body") -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ApiError(
            400,
            "INVALID_JSON_BODY",
            f"Request {field_name} must be a JSON object.",
        )
    return body


def _validate_dict_field(body: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = body.get(field_name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(
            422,
            "VALIDATION_ERROR",
            f"Field '{field_name}' must be an object.",
            {"field": field_name},
        )
    return value


def _validate_messages(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ApiError(422, "VALIDATION_ERROR", "Field 'messages' must be an array.", {"field": "messages"})
    cleaned: list[dict[str, str]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ApiError(
                422,
                "VALIDATION_ERROR",
                "Each message must be an object with 'role' and 'content'.",
                {"field": "messages", "index": idx},
            )
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            raise ApiError(
                422,
                "VALIDATION_ERROR",
                "Each message must include valid 'role' and non-empty 'content'.",
                {"field": "messages", "index": idx},
            )
        cleaned.append({"role": role, "content": content})
    return cleaned


def _validate_common_payload(body: Any) -> dict[str, Any]:
    payload = _validate_json_object(body)
    return {
        "patient": _validate_dict_field(payload, "patient"),
        "modelPrediction": _validate_dict_field(payload, "modelPrediction"),
        "recommendation": _validate_dict_field(payload, "recommendation"),
        "messages": _validate_messages(payload.get("messages")),
        "message": str(payload.get("message", "")).strip(),
    }


class MedGemmaHandler(BaseHTTPRequestHandler):
    def _write_json(self, code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._write_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_json(200, {"ok": True, "model_id": MODEL_ID})
            return
        self._write_json(404, {"error_code": "UNKNOWN_PATH", "error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/api/medgemma/recommend", "/api/medgemma/chat"):
            self._write_json(404, {"error_code": "UNKNOWN_PATH", "error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length)
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError as exc:
                raise ApiError(400, "INVALID_JSON_BODY", "Request body is not valid JSON.") from exc

            payload = _validate_common_payload(body)
            patient = payload["patient"]
            model_prediction = payload["modelPrediction"]
            rag_context = _retrieve_rag_context(patient, model_prediction)

            if self.path == "/api/medgemma/recommend":
                try:
                    recommendation = _generate_with_medgemma(
                        patient, model_prediction, rag_context
                    )
                except RecommendationParseError as exc:
                    self._write_json(
                        422,
                        {
                            "error_code": "MODEL_OUTPUT_PARSE_ERROR",
                            "error": str(exc),
                            "parse_error": True,
                            "rawOutput": exc.raw_output,
                            "from_medgemma": True,
                            "rag_used": bool(
                                rag_context and rag_context != "No PDQ context available."
                            ),
                        },
                    )
                    return
                except Exception as exc:  # pragma: no cover - environment dependent
                    LOGGER.exception("Recommendation generation failed: %s", exc)
                    self._write_json(
                        502,
                        {
                            "error_code": "MODEL_GENERATION_ERROR",
                            "error": str(exc),
                            "from_medgemma": False,
                            "rag_used": bool(
                                rag_context and rag_context != "No PDQ context available."
                            ),
                        },
                    )
                    return
                self._write_json(200, recommendation)
                return

            messages = payload["messages"]
            user_message = payload["message"]
            recommendation = payload["recommendation"]
            if not user_message:
                raise ApiError(422, "VALIDATION_ERROR", "Field 'message' is required.", {"field": "message"})
            try:
                response = _generate_chat_response(
                    patient,
                    model_prediction,
                    recommendation,
                    rag_context,
                    messages,
                    user_message,
                )
                if not response.strip():
                    response = _fallback_chat_response(
                        patient, recommendation, user_message
                    )
            except Exception as exc:  # pragma: no cover - environment dependent
                LOGGER.exception("Chat generation failed: %s", exc)
                response = _fallback_chat_response(patient, recommendation, user_message)
            self._write_json(200, {"reply": response})
        except ApiError as exc:
            self._write_json(
                exc.status,
                {"error_code": exc.code, "error": exc.message, "details": exc.details},
            )
        except Exception as exc:
            self._write_json(500, {"error_code": "INTERNAL_ERROR", "error": "Internal server error."})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((HOST, PORT), MedGemmaHandler)
    LOGGER.info("MedGemma API listening on http://%s:%s", HOST, PORT)
    server.serve_forever()


if __name__ == "__main__":
    main()

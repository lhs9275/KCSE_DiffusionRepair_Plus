"""Recomputable provenance for the exact tokenizer input."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


MODEL_INPUT_SCHEMA = "tokenizer_input_v1"
RENDER_MODES = frozenset({"chat_template", "legacy_plain_fallback"})


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_token_bytes(token_ids: Sequence[int]) -> bytes:
    return json.dumps(
        [int(token) for token in token_ids], separators=(",", ":")
    ).encode("ascii")


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(_canonical_token_bytes(token_ids)).hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    identity = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": str(getattr(tokenizer, "name_or_path", "") or ""),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "special_tokens_map": _json_safe(
            getattr(tokenizer, "special_tokens_map", {}) or {}
        ),
        "chat_template_sha256": _sha256_text(template) if isinstance(template, str) else None,
    }
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["identity_sha256"] = _sha256_text(payload)
    return identity


def _validate_tokenizer_identity(identity: Mapping[str, Any]) -> None:
    if not isinstance(identity, Mapping):
        raise ValueError("model input provenance: tokenizer identity missing")
    payload = dict(identity)
    recorded = payload.pop("identity_sha256", None)
    canonical = json.dumps(
        _json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if recorded != _sha256_text(canonical):
        raise ValueError("model input provenance: tokenizer identity hash mismatch")


def build_model_input_provenance(
    *,
    user_prompt: str,
    rendered_input: str,
    token_ids: Sequence[int],
    tokenizer: Any,
    render_mode: str,
    system_prompt: str,
    max_input_tokens: int,
) -> dict[str, Any]:
    if render_mode not in RENDER_MODES:
        raise ValueError(f"model input: unknown render mode {render_mode!r}")
    ids = [int(token) for token in token_ids]
    record = {
        "schema": MODEL_INPUT_SCHEMA,
        "render_mode": render_mode,
        "system_message_included": bool(system_prompt) and render_mode == "chat_template",
        "system_prompt_sha256": _sha256_text(system_prompt) if system_prompt else None,
        "user_prompt_sha256": _sha256_text(user_prompt),
        "final_input_text": rendered_input,
        "final_input_sha256": _sha256_text(rendered_input),
        "token_ids": ids,
        "token_ids_sha256": token_ids_sha256(ids),
        "input_tokens": len(ids),
        "max_input_tokens": int(max_input_tokens),
        "truncation_requested": False,
        "truncated": False,
        "tokenizer": tokenizer_identity(tokenizer),
    }
    validate_model_input_provenance(record, expected_user_prompt=user_prompt)
    return record


def validate_model_input_provenance(
    record: Mapping[str, Any], *, expected_user_prompt: str | None = None
) -> None:
    if not isinstance(record, Mapping) or record.get("schema") != MODEL_INPUT_SCHEMA:
        raise ValueError("model input provenance: schema mismatch")
    if record.get("render_mode") not in RENDER_MODES:
        raise ValueError("model input provenance: render mode mismatch")
    text = record.get("final_input_text")
    ids = record.get("token_ids")
    if not isinstance(text, str) or not isinstance(ids, list) or any(
        type(token) is not int for token in ids
    ):
        raise ValueError("model input provenance: malformed text or token IDs")
    if record.get("final_input_sha256") != _sha256_text(text):
        raise ValueError("model input provenance: final input hash mismatch")
    if record.get("token_ids_sha256") != token_ids_sha256(ids):
        raise ValueError("model input provenance: token hash mismatch")
    if record.get("input_tokens") != len(ids):
        raise ValueError("model input provenance: token count mismatch")
    limit = record.get("max_input_tokens")
    if type(limit) is not int or limit < 1 or len(ids) > limit:
        raise ValueError("model input provenance: token limit mismatch")
    if record.get("truncation_requested") is not False or record.get("truncated") is not False:
        raise ValueError("model input provenance: truncation is not permitted")
    if expected_user_prompt is not None and record.get("user_prompt_sha256") != _sha256_text(expected_user_prompt):
        raise ValueError("model input provenance: user prompt mismatch")
    tokenizer = record.get("tokenizer")
    _validate_tokenizer_identity(tokenizer)
    has_system = bool(record.get("system_prompt_sha256"))
    expected_system_included = has_system and record.get("render_mode") == "chat_template"
    if record.get("system_message_included") is not expected_system_included:
        raise ValueError("model input provenance: system-message boundary mismatch")

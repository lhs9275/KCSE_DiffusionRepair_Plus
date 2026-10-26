"""Canonical manifests and isolated corrected-replay output roots."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from src.replay.ranking import candidate_multiset_digest


OUTPUT_DIRECTORIES = (
    "manifests",
    "prompts",
    "results",
    "evaluations",
    "metrics",
    "logs",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def build_manifest(**fields: Any) -> dict[str, Any]:
    """Return a JSON-serializable manifest without adding mutable timestamps."""
    return {key: _json_value(value) for key, value in fields.items()}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def candidate_sequence_digest(candidates: list[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        _json_value(candidates),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def add_candidate_integrity(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy carrying hashes recomputable from serialized candidates."""
    result = copy.deepcopy(dict(record))
    candidates = result.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("result candidates must be a list")
    result["candidate_count"] = len(candidates)
    result["candidate_sequence_sha256"] = candidate_sequence_digest(candidates)
    result["candidate_multiset_sha256"] = candidate_multiset_digest(candidates)
    return result


def result_record_is_valid(
    record: Mapping[str, Any],
    *,
    expected_input_provenance: Mapping[str, Any],
    expected_config: str,
    expected_candidates: int,
) -> bool:
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > expected_candidates:
        return False
    if record.get("candidate_count") != len(candidates):
        return False
    if record.get("input_provenance") != expected_input_provenance:
        return False
    if record.get("config") != expected_config:
        return False
    # generation_index addresses the shared raw pool, so a view selects a
    # distinct subset of it rather than a 0..n-1 prefix.
    try:
        indices = [int(item["generation_index"]) for item in candidates]
    except (KeyError, TypeError, ValueError):
        return False
    if len(set(indices)) != len(indices) or any(index < 0 for index in indices):
        return False
    return (
        record.get("candidate_sequence_sha256")
        == candidate_sequence_digest(candidates)
        and record.get("candidate_multiset_sha256")
        == candidate_multiset_digest(candidates)
    )


def prepare_output_root(
    root: Path, manifest: Mapping[str, Any], *, resume: bool
) -> Path:
    """Create an isolated root or verify an identical manifest for resume."""
    root = Path(root)
    manifest_path = root / "manifests" / "run_manifest.json"
    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"resume manifest missing: {manifest_path}")
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        proposed = _json_value(manifest)
        if stored != proposed:
            raise ValueError("resume manifest does not match proposed manifest")
        return root

    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")

    for name in OUTPUT_DIRECTORIES:
        (root / name).mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(_canonical_json(manifest), encoding="utf-8")
    temporary.replace(manifest_path)
    return root

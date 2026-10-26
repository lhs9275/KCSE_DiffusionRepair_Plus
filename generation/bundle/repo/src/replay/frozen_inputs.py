"""Load immutable upstream records for corrected patch-generation replay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from src.replay.input_contract import build_analysis_input, build_generation_input
from src.utils.source_coordinates import normalized_bug_line
from src.utils.ast_hints import AST_FREE_POLICY, without_ast_diagnostics


@dataclass(frozen=True)
class FrozenReplayInput:
    bug_id: str
    seed: int
    source_config: str
    source_path: Path
    source_sha256: str
    prompt_input: Dict[str, Any]
    plans: List[Dict[str, Any]]


def _buggy_line_context(metadata: Dict[str, Any]) -> str:
    function = metadata.get("function") or {}
    source = function.get("function_before") or ""
    relative_line = normalized_bug_line(metadata)
    if not source or relative_line is None:
        return ""
    lines = source.splitlines()
    if not lines:
        return ""
    start = max(0, relative_line - 4)
    end = min(len(lines), relative_line + 3)
    return "\n".join(
        f"{index + 1:4d}:{'-> ' if index == relative_line - 1 else '   '}{lines[index]}"
        for index in range(start, end)
    )


def load_frozen_inputs(
    metadata_path: Path,
    source_dir: Path,
    expected_config: str,
    expected_seed: int,
    expected_retrieval_count: int,
    *,
    use_ast: bool = True,
    use_retrieval: bool = True,
    use_diffusion: bool = True,
) -> List[FrozenReplayInput]:
    """Reconstruct prompt inputs while rejecting incomplete or mixed records."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_paths = {
        path.stem: path
        for path in source_dir.glob("*.json")
        if path.name not in {"eval_results.json", "metrics.json"}
    }
    if set(source_paths) != {str(key) for key in metadata}:
        raise ValueError("source bug IDs do not match metadata")

    loaded: List[FrozenReplayInput] = []
    for bug_id in sorted(source_paths, key=int):
        path = source_paths[bug_id]
        record = json.loads(path.read_text(encoding="utf-8"))
        if str(record.get("bug_id")) != bug_id:
            raise ValueError(f"bug ID mismatch for {path}")
        try:
            actual_seed = int(record.get("seed"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"seed mismatch for bug {bug_id}") from exc
        if actual_seed != expected_seed:
            raise ValueError(f"seed mismatch for bug {bug_id}")
        if record.get("config") != expected_config:
            raise ValueError(f"config mismatch for bug {bug_id}")

        plans = record.get("diffusion_plan") or []
        static_nodes = record.get("static_analysis_nodes") or []
        retrieved = record.get("retrieved_examples") or []
        if not use_ast and (use_retrieval or use_diffusion):
            if record.get('ast_conditioning') != AST_FREE_POLICY or static_nodes:
                raise ValueError(f'AST hint: AST-free upstream must be regenerated for bug {bug_id}')
        if (use_diffusion and len(plans) != 12) or (use_ast and not static_nodes):
            raise ValueError(f"incomplete frozen upstream data for bug {bug_id}")
        if len(retrieved) != expected_retrieval_count:
            raise ValueError(f"retrieval count mismatch for bug {bug_id}")

        prompt_input = build_analysis_input(metadata[bug_id])
        if not use_ast:
            prompt_input = without_ast_diagnostics(prompt_input)
        prompt_input["buggy_line_context"] = _buggy_line_context(prompt_input)
        prompt_input["suspicious_nodes_topk_reranked"] = copy.deepcopy(static_nodes) if use_ast else []
        prompt_input["bm25"] = {"top": copy.deepcopy(retrieved)} if use_retrieval else {'top': []}
        prompt_input = build_generation_input(
            prompt_input,
            use_ast=use_ast,
            use_retrieval=use_retrieval,
        )
        loaded.append(
            FrozenReplayInput(
                bug_id=bug_id,
                seed=expected_seed,
                source_config=expected_config,
                source_path=path,
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                prompt_input=prompt_input,
                plans=copy.deepcopy(plans) if use_diffusion else [],
            )
        )
    return loaded

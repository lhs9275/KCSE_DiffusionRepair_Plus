"""결과 JSON 저장 스키마 및 유틸리티.

한 버그 × 시드 × 설정 조합의 결과를 표준 구조로 직렬화한다.
메트릭은 저장하지 않는다 — 사후 compute_metrics.py에서 계산.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def require_complete_versioned_generation(
    *, versioned: bool, saved_count: int, expected_count: int,
    failed_bug_ids: List[str],
) -> None:
    """Make scheduler success depend on complete, nonempty case coverage."""
    if versioned and (failed_bug_ids or expected_count <= 0 or saved_count != expected_count):
        raise RuntimeError(
            "versioned primary generation incomplete: "
            f"saved={saved_count}/{expected_count}, failed={','.join(failed_bug_ids)}"
        )


def make_candidate_record(
    patch_text: str,
    ast_parse_ok: bool,
    modified_lines: Optional[List[int]] = None,
    modified_ast_nodes: Optional[List[int]] = None,
    diff_plan_used: Optional[List[Dict[str, Any]]] = None,
    test_passed: Optional[bool] = None,
    rank_score: Optional[float] = None,
    tokens_input: Optional[int] = None,
    tokens_output: Optional[int] = None,
    inference_time_sec: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "patch_text": patch_text,
        "ast_parse_ok": ast_parse_ok,
        "modified_lines": modified_lines or [],
        "modified_ast_nodes": modified_ast_nodes or [],
        "diff_plan_used": diff_plan_used or [],
        "test_passed": test_passed,
        "rank_score": rank_score,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "inference_time_sec": inference_time_sec,
    }


def make_bug_result(
    bug_id: str,
    seed: int,
    config: str,
    candidates: List[Dict[str, Any]],
    diffusion_plan: Optional[List[Dict[str, Any]]] = None,
    static_analysis_nodes: Optional[List[Dict[str, Any]]] = None,
    retrieved_examples: Optional[List[Dict[str, Any]]] = None,
    ground_truth_modified_lines: Optional[List[int]] = None,
    ground_truth_modified_nodes: Optional[List[int]] = None,
    vram_peak_mb: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "bug_id": bug_id,
        "seed": seed,
        "config": config,
        "candidates": candidates,
        "diffusion_plan": diffusion_plan or [],
        "static_analysis_nodes": static_analysis_nodes or [],
        "retrieved_examples": retrieved_examples or [],
        "ground_truth_modified_lines": ground_truth_modified_lines or [],
        "ground_truth_modified_nodes": ground_truth_modified_nodes or [],
        "vram_peak_mb": vram_peak_mb,
    }


def result_dir(results_root: str, config: str, seed: int) -> Path:
    d = Path(results_root) / f"{config}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_bug_result(record: Dict[str, Any], results_root: str, compress: bool = False,
                    refuse_existing: bool = False) -> Path:
    """버그 결과를 results/<config>_seed<n>/<bug_id>.json[.gz]에 저장."""
    config = record["config"]
    seed = record["seed"]
    bug_id = record["bug_id"]
    d = result_dir(results_root, config, seed)
    ext = ".json.gz" if compress else ".json"
    out_path = d / f"{bug_id}{ext}"
    content = json.dumps(record, ensure_ascii=False, indent=2)
    if compress:
        with gzip.open(out_path, "xt" if refuse_existing else "wt", encoding="utf-8") as f:
            f.write(content)
    else:
        if refuse_existing:
            with out_path.open("x", encoding="utf-8") as f:
                f.write(content)
        else:
            out_path.write_text(content, encoding="utf-8")
    return out_path


def load_bug_result(path) -> Dict[str, Any]:
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_results(results_dir) -> Dict[str, Any]:
    """한 실험 폴더 내 모든 버그 결과를 bug_id → record 딕셔너리로 로드."""
    results_dir = Path(results_dir)
    _skip = {"metrics.json", "eval_results.json"}
    records: Dict[str, Any] = {}
    candidates = sorted(results_dir.glob("*.json")) + sorted(results_dir.glob("*.json.gz"))
    for p in candidates:
        if p.name in _skip or p.name == "run.log":
            continue
        try:
            rec = load_bug_result(p)
            bug_id = str(rec.get("bug_id", p.stem))
            records[bug_id] = rec
        except Exception:
            pass
    return records

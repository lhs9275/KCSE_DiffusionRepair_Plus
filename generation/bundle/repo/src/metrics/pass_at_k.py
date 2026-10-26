"""Pass@k 계산.

표준 조합 공식: pass@k = 1 - C(n-c, k) / C(n, k)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from math import comb


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """n개 샘플 중 c개 성공일 때 pass@k 확률."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - float(comb(n - c, k)) / float(comb(n, k))


def evaluate_pass_at_k(
    eval_results: Dict[str, Dict],
    k_list: Sequence[int],
    total_bug_count: int,
) -> Dict[str, str]:
    """
    eval_results: bug_id → {"nucleus_sampling_flags": ["Pass"/"Fail"/...]}
    k_list: 계산할 k 값 목록
    total_bug_count: 전체 벤치마크 버그 수 (분모)
    """
    scores_per_bug: List[Dict] = []
    for bug_id, entry in eval_results.items():
        flags = (entry or {}).get("nucleus_sampling_flags") or []
        n = len(flags)
        c = sum(1 for f in flags if f == "Pass")
        if n > 0:
            scores_per_bug.append({"n": n, "c": c})

    solved = sum(1 for item in scores_per_bug if item["c"] > 0)
    result: Dict[str, str] = {}

    for k in sorted(k_list):
        total_score = sum(compute_pass_at_k(item["n"], item["c"], k) for item in scores_per_bug)
        avg = total_score / max(1, total_bug_count)
        result[f"pass@{k}"] = f"{avg:.2%} ({solved}/{total_bug_count})"

    return result


def evaluate_pass_at_k_numeric(
    eval_results: Dict[str, Dict],
    k_list: Sequence[int],
    total_bug_count: int,
) -> Dict[str, float]:
    """숫자 값만 반환하는 버전 (표 생성용)."""
    scores_per_bug: List[Dict] = []
    for entry in eval_results.values():
        flags = (entry or {}).get("nucleus_sampling_flags") or []
        n, c = len(flags), sum(1 for f in flags if f == "Pass")
        if n > 0:
            scores_per_bug.append({"n": n, "c": c})

    result: Dict[str, float] = {}
    for k in k_list:
        total_score = sum(compute_pass_at_k(item["n"], item["c"], k) for item in scores_per_bug)
        result[f"pass@{k}"] = total_score / max(1, total_bug_count)
    return result

"""Over-editing 지표 계산.

- SOE (Soft Over-Edit ratio): 여분 편집 마커 / 전체 후보 편집 마커
- Strict-OER: 최선 후보의 SOE > 0인 버그 비율
- Success-conditioned SOE: 성공한 후보 중 최소 SOE 평균
"""
from __future__ import annotations

import re
import textwrap
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 코드 정규화 / diff 마커
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def _normalize_code(text: Optional[str]) -> str:
    if not text:
        return ""
    m = _CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = textwrap.dedent(text).strip("\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _code_lines(text: Optional[str]) -> List[str]:
    return _normalize_code(text).splitlines() or []


def _changed_markers(before: List[str], after: List[str]) -> List[Tuple]:
    markers: List[Tuple] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=before, b=after).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            for i in range(i1, i2):
                markers.append(("line", i))
        if tag in ("replace", "insert"):
            for offset, _ in enumerate(range(j1, j2)):
                markers.append(("ins", i1, offset))
    return markers


def over_edit_ratio(
    buggy_code: str,
    candidate_code: str,
    fixed_code: str,
) -> float:
    """SOE: 후보에서 GT를 초과하는 마커 수 / 후보 전체 마커 수."""
    b = _code_lines(buggy_code)
    c = _code_lines(candidate_code)
    f = _code_lines(fixed_code)
    cand_markers = Counter(_changed_markers(b, c))
    if not cand_markers:
        return 0.0
    gt_markers = Counter(_changed_markers(b, f))
    over = cand_markers - gt_markers
    return sum(over.values()) / sum(cand_markers.values())


# ---------------------------------------------------------------------------
# 버그 단위 집계
# ---------------------------------------------------------------------------

def compute_soe_for_bug(
    candidates: List[Dict[str, Any]],
    buggy_code: str,
    fixed_code: str,
    flags: List[str],
    k: int,
) -> Dict[str, Any]:
    """k 개 후보에 대해 SOE 관련 값을 계산한다."""
    ratios: List[float] = []
    pass_ratios: List[float] = []

    b = _code_lines(buggy_code)
    f = _code_lines(fixed_code)

    if not b or not f:
        return {"min_soe": 1.0, "strict_over_edit": True, "success_soe": None}

    for idx, cand in enumerate(candidates[:k]):
        code = cand.get("code") or cand.get("patch_text") or ""
        if not code:
            continue
        c = _code_lines(code)
        if not c:
            continue
        cand_markers = Counter(_changed_markers(b, c))
        if not cand_markers:
            ratios.append(0.0)
            if idx < len(flags) and flags[idx] == "Pass":
                pass_ratios.append(0.0)
            continue
        gt_markers = Counter(_changed_markers(b, f))
        over = cand_markers - gt_markers
        ratio = sum(over.values()) / sum(cand_markers.values())
        ratios.append(ratio)
        if idx < len(flags) and flags[idx] == "Pass":
            pass_ratios.append(ratio)

    min_soe = min(ratios) if ratios else 1.0
    success_soe = min(pass_ratios) if pass_ratios else None
    return {
        "min_soe": min_soe,
        "strict_over_edit": min_soe > 0.0,
        "success_soe": success_soe,
    }


# ---------------------------------------------------------------------------
# 전체 데이터셋 집계
# ---------------------------------------------------------------------------

def summarize_over_editing(
    inference_data: Dict[str, Any],
    eval_data: Dict[str, Any],
    bug_meta_data: Dict[str, Any],
    k_list: Sequence[int],
) -> Dict[str, Any]:
    """k 값별로 SOE/Strict-OER/Success-conditioned SOE를 집계한다."""
    bug_ids = sorted(str(bid) for bid in bug_meta_data.keys())
    total = len(bug_ids)
    metrics: Dict[str, Any] = {}

    for k in k_list:
        soe_sum = 0.0
        strict_count = 0
        sc_soe_values: List[float] = []

        for bug_id in bug_ids:
            entry = inference_data.get(bug_id) or inference_data.get(int(bug_id) if bug_id.isdigit() else bug_id) or {}
            cands = (entry.get("candidates") or [])
            eval_entry = eval_data.get(bug_id) or {}
            flags = (eval_entry.get("nucleus_sampling_flags") or [])
            meta = bug_meta_data.get(bug_id) or bug_meta_data.get(int(bug_id) if bug_id.isdigit() else bug_id) or {}
            buggy = (meta.get("function") or {}).get("function_before") or entry.get("code") or ""
            fixed = (meta.get("function") or {}).get("function_after") or ""

            result = compute_soe_for_bug(cands, buggy, fixed, flags, k)
            soe_sum += result["min_soe"]
            if result["strict_over_edit"]:
                strict_count += 1
            if result["success_soe"] is not None:
                sc_soe_values.append(result["success_soe"])

        metrics[f"soe@{k}"] = {
            "value": soe_sum / total if total else 0.0,
            "total": total,
        }
        metrics[f"strict_oer@{k}"] = {
            "value": strict_count / total if total else 0.0,
            "failures": strict_count,
            "total": total,
        }
        n_sc = len(sc_soe_values)
        metrics[f"success_conditioned_soe@{k}"] = {
            "value": sum(sc_soe_values) / n_sc if n_sc else None,
            "count": n_sc,
        }

    return metrics

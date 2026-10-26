"""계획 품질 지표.

KCSE 지표:
  - LocHit@k : top-k 계획 중 하나라도 실제 buggy line ±window 이내 편집을 포함하는 버그 비율
  - OpMatch@k : top-k 계획 중 하나라도 실제 수정 연산(Replace/Insert/Delete)과 일치하는 버그 비율
  - Alignment@k : LocHit AND OpMatch 모두 만족하는 버그 비율

IJCAI 지표:
  - PF (Plan Fidelity)@k : 후보 코드에서 계획 힌트가 실제로 적용된 비율의 best@k
"""
from __future__ import annotations

import ast
import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from src.utils.repair_plans import MAX_PROMPT_EDITS, PLAN_BINDING_POLICY, source_span_touch_fraction

LEGACY_PF_POLICY = "legacy_diagnostic_snippet_v0"
SAME_PLAN_ALIGNMENT_POLICY = "same_plan_v1"
OPERATION_SHAPE_POLICY = "operation_shape_proxy_v1"

_GUARD_CONSTRUCTS = (ast.If, ast.IfExp, ast.Try, ast.Assert)

_REPAIR_PLAN_HEADER_RE = re.compile(r"^### Context 0: Recommended Repair Plan", re.IGNORECASE)
_REPAIR_PLAN_LINE_RE = re.compile(r"\*\*(?P<op>[A-Za-z]+)\*\* at Line \d+: `(?P<code>[^`]+)`")


# ---------------------------------------------------------------------------
# AST 노드 → 라인 매핑
# ---------------------------------------------------------------------------

def node_id_to_line_map(function_code: str) -> Dict[int, int]:
    try:
        tree = ast.parse(function_code)
    except Exception:
        return {}
    return {
        i: node.lineno
        for i, node in enumerate(ast.walk(tree))
        if hasattr(node, "lineno")
    }


def changed_source_lines(before: str, after: str) -> Set[int]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    total_before = len(before_lines)
    changed: Set[int] = set()
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, before_lines, after_lines).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            changed.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            anchor = max(1, min(i1 + 1, total_before))
            changed.add(anchor)
    return changed


# ---------------------------------------------------------------------------
# LocHit
# ---------------------------------------------------------------------------

def _plan_edit_lines(
    diffusion_candidates: List[Dict[str, Any]],
    plan_idx: int,
    node_map: Dict[int, int],
) -> List[int]:
    """plan_idx번째 계획의 편집이 대상으로 하는 함수 내 줄 번호 목록."""
    if plan_idx >= len(diffusion_candidates):
        return []
    edits = diffusion_candidates[plan_idx].get("edits") or []
    lines: List[int] = []
    for e in edits:
        try:
            nid = int(e.get("node", -1))
        except Exception:
            continue
        line = node_map.get(nid)
        if line is not None:
            lines.append(line)
    return lines


def _find_buggy_line_in_function(function_code: str, buggy_line_content: str) -> Optional[int]:
    if not function_code or not buggy_line_content:
        return None
    target = buggy_line_content.strip()
    for i, line in enumerate(function_code.splitlines(), 1):
        if line.strip() == target:
            return i
    for i, line in enumerate(function_code.splitlines(), 1):
        if target in line:
            return i
    return None


def loc_hit_for_bug(
    diffusion_candidates: List[Dict[str, Any]],
    function_code: str,
    buggy_line_content: str,
    buggy_line_location: Optional[int],
    func_start_line: int,
    k: int,
    window: int = 3,
    bug_line_in_function: Optional[int] = None,
) -> bool:
    """top-k 계획 중 하나라도 LocHit이면 True."""
    # 함수 내 상대 줄 번호 계산
    buggy_rel = bug_line_in_function
    if buggy_rel is None:
        buggy_rel = _find_buggy_line_in_function(function_code, buggy_line_content or "")
    if buggy_rel is None and buggy_line_location is not None:
        try:
            rel = int(buggy_line_location) - int(func_start_line) + 1
            total = len(function_code.splitlines())
            buggy_rel = min(max(rel, 1), total) if total > 0 else None
        except Exception:
            pass
    if buggy_rel is None:
        return False

    node_map = node_id_to_line_map(function_code)
    for idx in range(min(k, len(diffusion_candidates))):
        edit_lines = _plan_edit_lines(diffusion_candidates, idx, node_map)
        if edit_lines and min(abs(l - buggy_rel) for l in edit_lines) <= window:
            return True
    return False


# ---------------------------------------------------------------------------
# OpMatch
# ---------------------------------------------------------------------------

_REPLACE_OPS = {"replace", "delete"}
_REPAIR_OPS = _REPLACE_OPS
_INSERT_OPS = {"insertbefore", "insertafter", "guardadd"}


def _infer_actual_ops(before: str, after: str) -> Set[str]:
    """실제 수정에서 Replace/Insert/Delete 중 어떤 연산이 쓰였는지 추론한다."""
    ops: Set[str] = set()
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    for tag, _, _, _, _ in SequenceMatcher(None, before_lines, after_lines).get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace", "delete"):
            ops.add("replace")
        if tag in ("replace", "insert"):
            ops.add("insert")
        if tag == "delete":
            ops.add("delete")
    return ops


def op_match_for_bug(
    diffusion_candidates: List[Dict[str, Any]],
    actual_ops: Set[str],
    k: int,
) -> bool:
    """top-k 계획 중 하나라도 실제 연산과 교집합이 있으면 True."""
    for idx in range(min(k, len(diffusion_candidates))):
        edits = diffusion_candidates[idx].get("edits") or []
        plan_ops: Set[str] = set()
        for e in edits:
            op = (e.get("op") or "").lower()
            if op in _REPLACE_OPS:
                plan_ops.add("replace")
            elif op in _INSERT_OPS:
                plan_ops.add("insert")
            elif op == "delete":
                plan_ops.add("delete")
        if plan_ops & actual_ops:
            return True
    return False


# ---------------------------------------------------------------------------
# Plan Fidelity (PF) — IJCAI 지표
# ---------------------------------------------------------------------------

def _extract_plan_hints(text: Optional[str]) -> List[Tuple[str, str]]:
    hints: List[Tuple[str, str]] = []
    in_plan = False
    for line in (text or "").splitlines():
        if _REPAIR_PLAN_HEADER_RE.search(line):
            in_plan = True
            continue
        if in_plan and line.startswith("### "):
            break
        if in_plan:
            m = _REPAIR_PLAN_LINE_RE.search(line)
            if m:
                hints.append((m.group("op").strip(), m.group("code").strip()))
    return hints


def _hints_from_diffusion(
    diffusion_candidates: List[Dict[str, Any]],
    static_nodes: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """저장된 diffusion_plan과 static_analysis_nodes로부터 PF용 hints를 재구성.

    프롬프트 텍스트가 결과 JSON에 저장되지 않으므로 prompt_generator.format_diffusion_plan과
    동일한 방식(node_id → code 매핑, 57자 truncation)으로 (op, snippet) 쌍을 만든다.
    """
    if not diffusion_candidates:
        return []
    edits = diffusion_candidates[0].get("edits") or []
    if not edits:
        return []
    node_map: Dict[int, Dict[str, Any]] = {}
    for n in (static_nodes or []):
        nid = n.get("node_id")
        if nid is None:
            continue
        try:
            node_map[int(nid)] = n
        except Exception:
            pass
    hints: List[Tuple[str, str]] = []
    seen: set = set()
    for e in edits:
        op = (e.get("op") or "").strip()
        try:
            nid = int(e.get("node", -1))
        except Exception:
            continue
        info = node_map.get(nid)
        if not info:
            continue
        code = (info.get("code") or "").strip()
        if not code:
            continue
        snippet = code[:57]
        if len(code) > 60:
            snippet += "..."
        key = (op, snippet)
        if key in seen:
            continue
        seen.add(key)
        hints.append((op, snippet))
    return hints


def plan_fidelity_ratio(base_code: str, cand_code: str, hints: List[Tuple[str, str]]) -> float:
    if not base_code or not cand_code or not hints:
        return 0.0
    base_lines = base_code.splitlines()
    cand_lines = cand_code.splitlines()
    changed: set = set()
    inserted_at: set = set()
    for tag, i1, i2, _j1, _j2 in SequenceMatcher(None, base_lines, cand_lines).get_opcodes():
        if tag in ("replace", "delete"):
            changed.update(range(i1, i2))
        elif tag == "insert":
            inserted_at.add(i1)
    satisfied = 0
    for _op, snippet in hints:
        snippet = (snippet or "").strip()
        if snippet.endswith("..."):
            snippet = snippet[:-3].rstrip()
        if not snippet:
            continue
        for i, line in enumerate(base_lines):
            if snippet in line:
                if i in changed or i in inserted_at or (i + 1) in inserted_at:
                    satisfied += 1
                break
    return satisfied / max(1, len(hints))


# ---------------------------------------------------------------------------
# Operation-semantic plan fidelity (versioned; legacy PF above is unchanged)
# ---------------------------------------------------------------------------

def _line_operations(before: str, after: str) -> Tuple[Set[int], Set[int], Set[int]]:
    """1-based before-line sets for replace and delete, plus insertion boundaries."""
    replaced: Set[int] = set()
    deleted: Set[int] = set()
    inserted: Set[int] = set()
    matcher = SequenceMatcher(None, before.splitlines(), after.splitlines(), autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag == "replace":
            replaced.update(range(i1 + 1, i2 + 1))
        elif tag == "delete":
            deleted.update(range(i1 + 1, i2 + 1))
        elif tag == "insert":
            inserted.add(i1)  # boundary after i1 original lines
    return replaced, deleted, inserted


def _guard_construct_count(code: str) -> Optional[int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    return sum(1 for node in ast.walk(tree) if isinstance(node, _GUARD_CONSTRUCTS))


def operation_shape_breakdown(
    before: str, after: str, edits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify each sent edit as satisfied / unsatisfied / undecidable.

    A *shape* proxy over a line diff, not semantic fidelity: it observes whether
    the change at the edit's target span has the shape the operation names. It
    does not prove the edit is correct, nor that nothing outside the plan moved.

    GuardAdd is the deliberate gap. A line diff cannot bind an added conditional
    to a particular target — a guard introduced elsewhere in the function while
    the target is merely rewritten produces exactly the same opcodes as a guard
    placed around the target. So GuardAdd is never counted as satisfied. It is
    reported as *decidably unsatisfied* only in the two cases where no guard can
    have reached the target at all: the target span is untouched, or the patched
    function contains no guard construct whatsoever. Note that a *net* guard
    count is not evidence either way — a patch may add a guard at the target
    while removing one elsewhere — so the count is never used to rule a guard
    out. Everything else is *undecidable*: excluded from the denominator and
    reported separately, never scored zero.
    """
    from src.utils.repair_plans import validate_repair_edits

    nodes = validate_repair_edits(before, edits)
    counts = {"satisfied": 0, "unsatisfied": 0, "undecidable": 0,
              "policy": OPERATION_SHAPE_POLICY, "value": None, "sent_edits": len(edits)}
    if not edits:
        return counts
    replaced, deleted, inserted = _line_operations(before, after)
    guards_after = _guard_construct_count(after)
    # None means the patch does not parse; absence of any guard is the only
    # count-based fact strong enough to rule GuardAdd out.
    patch_holds_no_guard = guards_after == 0

    for edit in edits:
        node = nodes[edit["node"]]
        start, end = node.lineno, node.end_lineno
        span = set(range(start, end + 1))
        touched = bool(span & (replaced | deleted)) or any(
            start - 1 <= boundary <= end for boundary in inserted)
        operation = edit["op"]
        if operation == "GuardAdd":
            # Decidably unsatisfied, or unattributable — never satisfied.
            if not touched or patch_holds_no_guard:
                counts["unsatisfied"] += 1
            else:
                counts["undecidable"] += 1
            continue
        if not touched:
            counts["unsatisfied"] += 1
            continue
        if operation == "Replace":
            hit = bool(span & replaced)
        elif operation == "Delete":
            hit = bool(span & deleted)
        elif operation == "InsertBefore":
            hit = (start - 1) in inserted
        elif operation == "InsertAfter":
            hit = end in inserted
        else:  # validate_repair_edits already restricts the vocabulary
            hit = False
        counts["satisfied" if hit else "unsatisfied"] += 1

    decidable = counts["satisfied"] + counts["unsatisfied"]
    counts["decidable"] = decidable
    counts["value"] = counts["satisfied"] / decidable if decidable else None
    return counts


def operation_shape_fidelity(
    before: str, after: str, edits: List[Dict[str, Any]],
) -> Optional[float]:
    """Satisfied fraction over the *decidable* sent edits, else None.

    See `operation_shape_breakdown` for the classification and its limits.
    """
    return operation_shape_breakdown(before, after, edits)["value"]


def _plan_is_loc_hit(
    plan: Dict[str, Any], node_map: Dict[int, int], buggy_rel: Optional[int], window: int,
) -> bool:
    if buggy_rel is None:
        return False
    lines = [node_map[int(e["node"])] for e in (plan.get("edits") or [])
             if isinstance(e.get("node"), int) and int(e["node"]) in node_map]
    return bool(lines) and min(abs(line - buggy_rel) for line in lines) <= window


def _plan_ops(plan: Dict[str, Any]) -> Set[str]:
    ops: Set[str] = set()
    for edit in (plan.get("edits") or []):
        name = (edit.get("op") or "").lower()
        if name in _REPAIR_OPS:
            ops.add("replace")
        elif name in _INSERT_OPS:
            ops.add("insert")
        if name == "delete":
            ops.add("delete")
    return ops


def same_plan_alignment_for_bug(
    diffusion_candidates: List[Dict[str, Any]], function_code: str, actual_ops: Set[str],
    buggy_rel: Optional[int], k: int, window: int,
) -> bool:
    """One single plan must satisfy LocHit and OpMatch, not two different plans."""
    node_map = node_id_to_line_map(function_code)
    for index in range(min(k, len(diffusion_candidates))):
        plan = diffusion_candidates[index]
        if _plan_is_loc_hit(plan, node_map, buggy_rel, window) and (_plan_ops(plan) & actual_ops):
            return True
    return False


# ---------------------------------------------------------------------------
# 전체 데이터셋 집계
# ---------------------------------------------------------------------------

def summarize_plan_quality(
    diffusion_data: Dict[str, Any],
    inference_data: Dict[str, Any],
    bug_meta_data: Dict[str, Any],
    k_list: Sequence[int],
    window: int = 3,
) -> Dict[str, Any]:
    """LocHit@k, OpMatch@k, Alignment@k, PF@k를 집계한다."""
    bug_ids = sorted(str(bid) for bid in bug_meta_data.keys())
    total = len(bug_ids)
    metrics: Dict[str, Any] = {}

    for k in k_list:
        loc_hit_count = 0
        op_match_count = 0
        alignment_count = 0
        same_plan_alignment_count = 0
        pf_sum = 0.0

        for bug_id in bug_ids:
            meta = bug_meta_data.get(bug_id) or bug_meta_data.get(int(bug_id) if bug_id.isdigit() else bug_id) or {}
            func = meta.get("function") or {}
            function_code = (func.get("function_before") or "").strip()
            buggy_line_content = meta.get("buggy_line_content") or ""
            buggy_line_location = meta.get("buggy_line_location")
            func_start_line = func.get("function_before_start_line") or 1
            fixed_code = func.get("function_after") or ""

            diff_row = diffusion_data.get(bug_id) or diffusion_data.get(int(bug_id) if bug_id.isdigit() else bug_id) or {}
            diff_cands = diff_row.get("candidates") or []

            # LocHit
            from src.utils.source_coordinates import normalized_bug_line

            lh = loc_hit_for_bug(
                diff_cands, function_code, buggy_line_content,
                buggy_line_location, func_start_line, k, window,
                bug_line_in_function=normalized_bug_line(meta),
            ) if function_code else False

            # OpMatch
            actual_ops = _infer_actual_ops(function_code, fixed_code) if function_code and fixed_code else set()
            om = op_match_for_bug(diff_cands, actual_ops, k) if diff_cands else False

            if lh:
                loc_hit_count += 1
            if om:
                op_match_count += 1
            if lh and om:
                alignment_count += 1
            # Versioned replacement: the same single plan must satisfy both.
            if diff_cands and function_code and same_plan_alignment_for_bug(
                    diff_cands, function_code, actual_ops,
                    normalized_bug_line(meta), k, window):
                same_plan_alignment_count += 1

            # PF — 저장된 diffusion_plan + static_analysis_nodes로부터 hints 재구성
            entry = inference_data.get(bug_id) or {}
            patch_cands = entry.get("candidates") or []
            static_nodes = entry.get("static_analysis_nodes") or []
            hints = _hints_from_diffusion(diff_cands, static_nodes)
            # 후방 호환: 과거 결과에 text 필드가 남아 있으면 거기서도 시도
            if not hints:
                for cand in patch_cands[:k]:
                    h = _extract_plan_hints(cand.get("text"))
                    if h:
                        hints = h
                        break
            best_pf = 0.0
            if hints and function_code:
                for cand in patch_cands[:k]:
                    code = cand.get("code") or cand.get("patch_text") or ""
                    if code:
                        pf = plan_fidelity_ratio(function_code, code, hints)
                        best_pf = max(best_pf, pf)
            pf_sum += best_pf

        metrics[f"loc_hit@{k}"] = {"value": loc_hit_count / total if total else 0.0, "hit": loc_hit_count, "total": total}
        metrics[f"op_match@{k}"] = {"value": op_match_count / total if total else 0.0, "hit": op_match_count, "total": total}
        metrics[f"alignment@{k}"] = {"value": alignment_count / total if total else 0.0, "hit": alignment_count, "total": total}
        metrics[f"alignment_same_plan@{k}"] = {
            "value": same_plan_alignment_count / total if total else 0.0,
            "hit": same_plan_alignment_count, "total": total,
            "policy": SAME_PLAN_ALIGNMENT_POLICY,
        }
        metrics[f"pf@{k}"] = {"value": pf_sum / total if total else 0.0, "total": total,
                              "policy": LEGACY_PF_POLICY}

    return metrics


def summarize_addressed_plan_fidelity(inference_data, bug_meta_data, k_list):
    """Versioned source-span touch proxy over sent edits; legacy PF stays intact.

    Undefined/missing-provenance cases are counted explicitly, never scored zero.
    The mean is conditional on defined bugs, not an unconditional benchmark rate.
    """
    metrics = {'policy': 'source_span_touch_v1', 'shape_policy': OPERATION_SHAPE_POLICY,
               'aggregation': 'mean_best_k_over_defined_bugs'}
    for k in k_list:
        if type(k) is not int or k < 1:
            raise ValueError('addressed PF: k must be positive')
        values = []
        shape_values = []
        undecidable_only = 0
        counts = {'missing_provenance_bugs': 0, 'no_plan_bugs': 0, 'no_candidate_bugs': 0}
        for bid, metadata in bug_meta_data.items():
            record = inference_data.get(str(bid)) or {}
            if record.get('plan_binding_policy') != PLAN_BINDING_POLICY or not record.get('source_code_sha256'):
                counts['missing_provenance_bugs'] += 1
                continue
            source = (metadata.get('function') or {}).get('function_before') or ''
            if record.get('source_code_sha256') != hashlib.sha256(source.encode()).hexdigest():
                raise ValueError(f'addressed PF: source hash mismatch for bug {bid}')
            candidates = (record.get('candidates') or [])[:k]
            if not candidates:
                counts['no_candidate_bugs'] += 1
                continue
            if any('diff_plan_used' not in candidate for candidate in candidates):
                counts['missing_provenance_bugs'] += 1
                continue
            scores = []
            shape_scores = []
            for candidate in candidates:
                edits = candidate['diff_plan_used']
                if not isinstance(edits, list) or len(edits) > MAX_PROMPT_EDITS:
                    raise ValueError('addressed PF: invalid sent edit list')
                patch = candidate.get('patch_text') or candidate.get('code')
                if not isinstance(patch, str) or not patch.strip():
                    raise ValueError('addressed PF: candidate code missing')
                score = source_span_touch_fraction(source, patch, edits)
                if score is not None:
                    scores.append(score)
                shape = operation_shape_fidelity(source, patch, edits)
                if shape is not None:
                    shape_scores.append(shape)
            if scores:
                values.append(max(scores))
            else:
                counts['no_plan_bugs'] += 1
            if shape_scores:
                shape_values.append(max(shape_scores))
            else:
                undecidable_only += 1
        metrics[f'pf_source_span@{k}'] = {'value': sum(values) / len(values) if values else None,
            'defined_bugs': len(values), 'total_bugs': len(bug_meta_data), **counts}
        metrics[f'pf_operation_shape@{k}'] = {
            'value': sum(shape_values) / len(shape_values) if shape_values else None,
            'defined_bugs': len(shape_values), 'total_bugs': len(bug_meta_data),
            'undecidable_only_bugs': undecidable_only,
            'policy': OPERATION_SHAPE_POLICY, **counts}
    return metrics

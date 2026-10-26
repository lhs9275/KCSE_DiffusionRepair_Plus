"""Explicit corrected-replay repair, locality, and paired metrics."""

from __future__ import annotations

import math
import random
import re
import textwrap
from difflib import SequenceMatcher
from typing import Any, Mapping, Optional, Sequence

from src.metrics.over_editing import over_edit_ratio


_CODE_FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)


def _normalize_code(value: Any) -> str:
    text = str(value or "")
    match = _CODE_FENCE.search(text)
    if match:
        text = match.group(1)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = textwrap.dedent(text).strip("\n")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def top_k_plausible_rate(
    evaluation: Mapping[str, Mapping[str, Any]],
    *,
    k: int,
    benchmark_total: int,
    benchmark_bug_ids: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Fraction of successfully evaluated bugs solved within ordered ranks 1..k."""
    if k < 1:
        raise ValueError("k must be positive")
    normalized = {str(key): value for key, value in evaluation.items()}
    if benchmark_bug_ids is not None:
        expected_ids = {str(value) for value in benchmark_bug_ids}
        if set(normalized) != expected_ids:
            raise ValueError("evaluation coverage does not match benchmark bug IDs")
        entries = (normalized[bug_id] for bug_id in expected_ids)
    else:
        if len(normalized) != benchmark_total:
            raise ValueError("evaluation coverage does not match benchmark total")
        entries = iter(normalized.values())
    evaluable = 0
    solved = 0
    failed = 0
    unknown = 0
    non_evaluable = 0
    for entry in entries:
        if (entry or {}).get("status") != "Success":
            non_evaluable += 1
            continue
        evaluable += 1
        outcome = prefix_repair_outcome(
            (entry or {}).get("nucleus_sampling_flags") or [], k=k
        )
        solved += int(outcome == "success")
        failed += int(outcome == "failure")
        unknown += int(outcome == "unknown")
    return {
        "solved": solved,
        "failed": failed,
        "unknown": unknown,
        "non_evaluable": non_evaluable,
        "evaluable": evaluable,
        "benchmark_total": benchmark_total,
        "value": solved / evaluable if evaluable else None,
        "observed_lower_bound": solved / benchmark_total if benchmark_total else None,
        "possible_upper_bound": (
            (solved + unknown + non_evaluable) / benchmark_total
            if benchmark_total else None
        ),
    }


def prefix_repair_outcome(flags: Sequence[Any], *, k: int) -> str:
    """Classify one ordered prefix without converting unresolved execution to Fail."""
    if k < 1:
        raise ValueError("k must be positive")
    prefix = list(flags[:k])
    if any(flag == "Pass" for flag in prefix):
        return "success"
    if any(flag not in {"Pass", "Fail"} for flag in prefix):
        return "unknown"
    return "failure"


def validate_metric_coverage(
    inference: Mapping[str, Any],
    evaluation: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    expected_candidates: int,
) -> dict[str, Any]:
    """Report exact ID and bounded candidate coverage before metric calculation."""
    expected = {str(key) for key in metadata}
    result_values = {str(key): value for key, value in inference.items()}
    evaluation_values = {str(key): value for key, value in evaluation.items()}
    result_ids = set(result_values)
    evaluation_ids = set(evaluation_values)
    sort_key = lambda value: int(value) if value.isdigit() else value
    counts = {
        bug_id: len(_candidate_list(value))
        for bug_id, value in result_values.items()
    }
    exact_records = sum(count == expected_candidates for count in counts.values())
    within_target_records = sum(
        count <= expected_candidates for count in counts.values()
    )
    zero_records = sum(count == 0 for count in counts.values())
    over_target_ids = sorted(
        (bug_id for bug_id, count in counts.items() if count > expected_candidates),
        key=sort_key,
    )
    flag_count_mismatches: list[str] = []
    unresolved_outcome_ids: list[str] = []
    for bug_id in sorted(expected & result_ids & evaluation_ids, key=sort_key):
        entry = evaluation_values[bug_id] or {}
        if entry.get("status") != "Success":
            continue
        if len(entry.get("nucleus_sampling_flags") or []) != counts[bug_id]:
            flag_count_mismatches.append(bug_id)
        if any(flag not in {"Pass", "Fail"} for flag in entry.get("nucleus_sampling_flags") or []):
            unresolved_outcome_ids.append(bug_id)
    result_missing = sorted(expected - result_ids, key=sort_key)
    result_extra = sorted(result_ids - expected, key=sort_key)
    evaluation_missing = sorted(expected - evaluation_ids, key=sort_key)
    evaluation_extra = sorted(evaluation_ids - expected, key=sort_key)
    valid = not any(
        (
            result_missing,
            result_extra,
            evaluation_missing,
            evaluation_extra,
            flag_count_mismatches,
            over_target_ids,
        )
    )
    return {
        "valid": valid,
        "benchmark_ids": len(expected),
        "result_ids": len(result_ids),
        "evaluation_ids": len(evaluation_ids),
        "result_missing_ids": result_missing,
        "result_extra_ids": result_extra,
        "evaluation_missing_ids": evaluation_missing,
        "evaluation_extra_ids": evaluation_extra,
        "candidate_target_maximum": expected_candidates,
        "candidate_count_exact": expected_candidates,
        "candidate_count_exact_records": exact_records,
        "candidate_count_within_target_records": within_target_records,
        "zero_candidate_records": zero_records,
        "candidate_count_over_target_ids": over_target_ids,
        "candidate_count_by_bug": counts,
        "evaluation_flag_count_mismatch_ids": flag_count_mismatches,
        "unresolved_candidate_outcome_ids": unresolved_outcome_ids,
    }


def changed_marker_locations(before: str, after: str) -> list[int]:
    """Return one-based source locations for edits, including insertion anchors."""
    before_lines = _normalize_code(before).splitlines()
    after_lines = _normalize_code(after).splitlines()
    locations: list[int] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(
        a=before_lines, b=after_lines
    ).get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "delete"}:
            locations.extend(range(i1 + 1, i2 + 1))
        if tag == "insert":
            locations.extend([i1 + 1] * (j2 - j1))
        elif tag == "replace" and (j2 - j1) > (i2 - i1):
            locations.extend([i2 + 1] * ((j2 - j1) - (i2 - i1)))
    return locations


def _candidate_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        candidates = value.get("candidates") or []
        return candidates if isinstance(candidates, list) else []
    return []


def _metadata_value(metadata: Mapping[str, Any], bug_id: str) -> Mapping[str, Any]:
    if bug_id in metadata:
        return metadata[bug_id]
    if bug_id.isdigit() and int(bug_id) in metadata:
        return metadata[int(bug_id)]
    return {}


def _before_and_bug_line(metadata: Mapping[str, Any]) -> tuple[str, Optional[int]]:
    from src.utils.source_coordinates import normalized_bug_line

    function = metadata.get("function") or {}
    before = metadata.get("function_before") or function.get("function_before") or ""
    return before, normalized_bug_line(metadata)


def summarize_window_soe(
    inference: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    k: int,
    evaluation: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    """Summarize minimum per-bug edit fraction outside the ±3-line window."""
    ratios_by_bug: dict[str, float] = {}
    success_ratios_by_bug: dict[str, float] = {}
    for raw_bug_id in metadata:
        bug_id = str(raw_bug_id)
        meta = _metadata_value(metadata, bug_id)
        before, bug_line = _before_and_bug_line(meta)
        if not _normalize_code(before) or bug_line is None:
            continue
        entry = inference.get(bug_id)
        if entry is None and bug_id.isdigit():
            entry = inference.get(int(bug_id))
        candidates = _candidate_list(entry)
        candidate_ratios: list[float] = []
        successful_ratios: list[float] = []
        flags = ((evaluation or {}).get(bug_id) or {}).get(
            "nucleus_sampling_flags"
        ) or []
        lower, upper = bug_line - 3, bug_line + 3
        for index, candidate in enumerate(candidates[:k]):
            code = candidate.get("patch_text") or candidate.get("code") or ""
            if not _normalize_code(code):
                continue
            markers = changed_marker_locations(before, code)
            ratio = (
                sum(not (lower <= marker <= upper) for marker in markers)
                / len(markers)
                if markers
                else 0.0
            )
            candidate_ratios.append(ratio)
            if index < len(flags) and flags[index] == "Pass":
                successful_ratios.append(ratio)
        if candidate_ratios:
            ratios_by_bug[bug_id] = min(candidate_ratios)
        if successful_ratios:
            success_ratios_by_bug[bug_id] = min(successful_ratios)

    ratios = list(ratios_by_bug.values())
    success_ratios = list(success_ratios_by_bug.values())
    strict = sum(value > 0 for value in ratios)
    return {
        "window": {"radius": 3, "lower_inclusive": True, "upper_inclusive": True},
        "defined_bugs": len(ratios),
        "benchmark_bugs": len(metadata),
        "mean_min_ratio": sum(ratios) / len(ratios) if ratios else None,
        "strict_over_edit_bugs": strict,
        "strict_over_edit_rate": strict / len(ratios) if ratios else None,
        "success_conditioned_defined_bugs": len(success_ratios),
        "success_conditioned_mean_min_ratio": (
            sum(success_ratios) / len(success_ratios) if success_ratios else None
        ),
        "per_bug_min_ratio": ratios_by_bug,
    }


def summarize_reference_excess(
    inference: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    """Summarize the legacy reference-excess edit ratio under an explicit name."""
    values: dict[str, float] = {}
    for raw_bug_id in metadata:
        bug_id = str(raw_bug_id)
        meta = _metadata_value(metadata, bug_id)
        function = meta.get("function") or {}
        before = meta.get("function_before") or function.get("function_before") or ""
        after = meta.get("function_after") or function.get("function_after") or ""
        entry = inference.get(bug_id)
        if entry is None and bug_id.isdigit():
            entry = inference.get(int(bug_id))
        ratios = []
        for candidate in _candidate_list(entry)[:k]:
            code = candidate.get("patch_text") or candidate.get("code") or ""
            if _normalize_code(before) and _normalize_code(after) and _normalize_code(code):
                ratios.append(over_edit_ratio(before, code, after))
        if ratios:
            values[bug_id] = min(ratios)
    return {
        "defined_bugs": len(values),
        "benchmark_bugs": len(metadata),
        "mean_min_ratio": sum(values.values()) / len(values) if values else None,
        "per_bug_min_ratio": values,
    }


def _mcnemar_exact_two_sided(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_binary_comparison(
    left: Mapping[str, bool], right: Mapping[str, bool]
) -> dict[str, Any]:
    """Return paired outcome counts and the exact two-sided McNemar p-value."""
    left_ids = {str(key) for key in left}
    right_ids = {str(key) for key in right}
    if left_ids != right_ids:
        raise ValueError("paired comparisons require identical bug IDs")
    left_values = {str(key): bool(value) for key, value in left.items()}
    right_values = {str(key): bool(value) for key, value in right.items()}
    both = left_only = right_only = neither = 0
    for bug_id in sorted(left_ids, key=lambda value: int(value) if value.isdigit() else value):
        pair = (left_values[bug_id], right_values[bug_id])
        if pair == (True, True):
            both += 1
        elif pair == (True, False):
            left_only += 1
        elif pair == (False, True):
            right_only += 1
        else:
            neither += 1
    total = len(left_ids)
    differences = [
        int(left_values[bug_id]) - int(right_values[bug_id])
        for bug_id in sorted(
            left_ids,
            key=lambda value: int(value) if value.isdigit() else value,
        )
    ]
    if differences:
        rng = random.Random(0)
        bootstrap = sorted(
            sum(rng.choice(differences) for _ in differences) / len(differences)
            for _ in range(10000)
        )
        lower = bootstrap[int(0.025 * len(bootstrap))]
        upper = bootstrap[int(0.975 * len(bootstrap)) - 1]
    else:
        lower = upper = None
    return {
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "paired_bugs": total,
        "effect_left_minus_right": (
            (left_only - right_only) / total if total else None
        ),
        "mcnemar_exact_two_sided_p": _mcnemar_exact_two_sided(
            left_only, right_only
        ),
        "effect_ci_95": {
            "method": "paired_bootstrap_percentile",
            "iterations": 10000,
            "seed": 0,
            "lower": lower,
            "upper": upper,
        },
    }


def paired_top_k_comparison(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    benchmark_bug_ids: Sequence[str],
    *,
    k: int,
) -> dict[str, Any]:
    """Compare top-k repair outcomes only over status-matched evaluable bugs."""
    expected = {str(value) for value in benchmark_bug_ids}
    left_values = {str(key): value for key, value in left.items()}
    right_values = {str(key): value for key, value in right.items()}
    if set(left_values) != expected or set(right_values) != expected:
        raise ValueError("paired evaluation coverage does not match benchmark IDs")
    status_drift = sorted(
        bug_id
        for bug_id in expected
        if (left_values[bug_id] or {}).get("status")
        != (right_values[bug_id] or {}).get("status")
    )
    if status_drift:
        raise ValueError(f"paired evaluator status drift for bugs: {status_drift}")
    evaluable = sorted(
        (
            bug_id
            for bug_id in expected
            if (left_values[bug_id] or {}).get("status") == "Success"
        ),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    outcomes = {
        bug_id: (
            prefix_repair_outcome((left_values[bug_id] or {}).get("nucleus_sampling_flags") or [], k=k),
            prefix_repair_outcome((right_values[bug_id] or {}).get("nucleus_sampling_flags") or [], k=k),
        )
        for bug_id in evaluable
    }
    paired_unknown = sorted(
        (bug_id for bug_id, pair in outcomes.items() if "unknown" in pair),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    paired_known = [bug_id for bug_id in evaluable if bug_id not in set(paired_unknown)]
    left_solved = {bug_id: outcomes[bug_id][0] == "success" for bug_id in paired_known}
    right_solved = {bug_id: outcomes[bug_id][1] == "success" for bug_id in paired_known}
    result = paired_binary_comparison(left_solved, right_solved)
    result.update(
        {
            "k": k,
            "benchmark_total": len(expected),
            "paired_evaluable": len(paired_known),
            "paired_unknown": len(paired_unknown),
            "paired_unknown_ids": paired_unknown,
            "excluded_non_evaluable": len(expected) - len(evaluable),
        }
    )
    return result

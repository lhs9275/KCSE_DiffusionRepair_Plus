"""Answer-free target views and condition-specific generation contracts."""

from __future__ import annotations

import ast
import copy
import hashlib
import re
from typing import Any, Mapping, MutableMapping, Sequence


TARGET_FUNCTION_FIELDS = (
    "function_name",
    "function_before",
    "function_before_start_line",
    "function_before_end_line",
    "function_before_token_count",
    "function_parent",
)
TARGET_FILE_ANALYSIS_FIELDS = ("file_name", "file_path", "file_before")
TARGET_TOP_LEVEL_FIELDS = (
    "id",
    "bugsinpy_id",
    "project_name",
    "project_url",
    "buggy_line_content",
    "buggy_line_location",
    "bug_line_in_function",
    "buggy_line_context",
    "test",
)
TARGET_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "commit",
        "ground_truth_modified_lines",
        "ground_truth_modified_nodes",
        "evaluation",
        "eval_results",
        "nucleus_sampling_flags",
        "test_passed",
    }
)
TARGET_FORBIDDEN_FUNCTION_FIELDS = frozenset(
    {
        "function_after",
        "function_after_start_line",
        "function_after_end_line",
        "function_after_token_count",
    }
)
TARGET_FORBIDDEN_FILE_FIELDS = frozenset({"file_after"})
BASE_FORBIDDEN_MARKERS = (
    "### Context 0: Recommended Repair Plan",
    "### Context 1: Static Analysis Hints",
    "### Context 2: Similar Fix Examples",
    "### Context 3: Related Commit Info",
    "Required edit hints:",
    "AI-Generated Repair Plan",
    "Recommended Repair Plan",
    "repair plan",
)


def _copy_selected(source: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {
        field: copy.deepcopy(source[field])
        for field in fields
        if field in source
    }


def build_analysis_input(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the target fields needed before retrieval/planning, never its answer."""
    if not isinstance(metadata, Mapping):
        raise ValueError("generation input: target metadata must be an object")
    result = _copy_selected(metadata, TARGET_TOP_LEVEL_FIELDS)
    function = metadata.get("function") or {}
    file_info = metadata.get("file") or {}
    if not isinstance(function, Mapping) or not isinstance(file_info, Mapping):
        raise ValueError("generation input: target function/file must be objects")
    result["function"] = _copy_selected(function, TARGET_FUNCTION_FIELDS)
    result["file"] = _copy_selected(file_info, TARGET_FILE_ANALYSIS_FIELDS)
    if not (result["function"].get("function_before") or "").strip():
        raise ValueError("generation input: target function_before is empty")
    assert_target_input_safe(result, allow_analysis_source=True)
    return result


def _retrieval_example_view(example: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields needed to display an explicitly disclosed prior fix."""
    function = example.get("function") or {}
    if not isinstance(function, Mapping):
        raise ValueError("generation input: retrieval function must be an object")
    allowed = (
        "id",
        "project_name",
        "file_name",
        "file_path",
        "buggy_line_content",
        "score",
        "score_bm25_raw",
        "score_bm25_normalized",
        "score_dense_cosine",
        "score_dense_normalized",
        "score_hybrid",
        "score_reranked",
    )
    result = _copy_selected(example, allowed)
    result["function"] = _copy_selected(
        function,
        ("function_name", "function_before", "function_after"),
    )
    return result


def build_generation_input(
    analyzed: Mapping[str, Any],
    *,
    use_ast: bool,
    use_retrieval: bool,
) -> dict[str, Any]:
    """Return the only object that prompt construction and patch generation may see."""
    result = _copy_selected(analyzed, TARGET_TOP_LEVEL_FIELDS)
    function = analyzed.get("function") or {}
    file_info = analyzed.get("file") or {}
    if not isinstance(function, Mapping) or not isinstance(file_info, Mapping):
        raise ValueError("generation input: analyzed function/file must be objects")
    result["function"] = _copy_selected(function, TARGET_FUNCTION_FIELDS)
    result["file"] = _copy_selected(file_info, ("file_name", "file_path"))
    result["file_path"] = copy.deepcopy(
        analyzed.get("file_path") or result["file"].get("file_path") or "N/A"
    )
    result["suspicious_nodes_topk"] = copy.deepcopy(
        (analyzed.get("suspicious_nodes_topk") or []) if use_ast else []
    )
    result["suspicious_nodes_topk_reranked"] = copy.deepcopy(
        (analyzed.get("suspicious_nodes_topk_reranked") or []) if use_ast else []
    )
    raw_retrieved = ((analyzed.get("bm25") or {}).get("top") or []) if use_retrieval else []
    if not isinstance(raw_retrieved, list) or any(
        not isinstance(item, Mapping) for item in raw_retrieved
    ):
        raise ValueError("generation input: retrieval examples must be objects")
    result["bm25"] = {
        "top": [_retrieval_example_view(item) for item in raw_retrieved]
    }
    assert_target_input_safe(result, allow_analysis_source=False)
    return result


def assert_target_input_safe(
    value: Mapping[str, Any], *, allow_analysis_source: bool = False
) -> None:
    """Fail if target-answer or evaluation fields cross the generation boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("generation input: expected an object")
    forbidden = sorted(TARGET_FORBIDDEN_TOP_LEVEL.intersection(value))
    if forbidden:
        raise ValueError(f"generation input: forbidden target fields {forbidden}")
    function = value.get("function") or {}
    file_info = value.get("file") or {}
    if not isinstance(function, Mapping) or not isinstance(file_info, Mapping):
        raise ValueError("generation input: function/file must be objects")
    forbidden_function = sorted(TARGET_FORBIDDEN_FUNCTION_FIELDS.intersection(function))
    forbidden_file = sorted(TARGET_FORBIDDEN_FILE_FIELDS.intersection(file_info))
    if forbidden_function or forbidden_file:
        raise ValueError(
            "generation input: forbidden target answer fields "
            f"{forbidden_function + forbidden_file}"
        )
    if not allow_analysis_source and "file_before" in file_info:
        raise ValueError("generation input: full target file crossed the patch boundary")


def source_fingerprint(value: Any) -> str | None:
    """Canonical Python/source digest used only to reject duplicate retrieval rows."""
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return None
    try:
        normalized = ast.dump(ast.parse(text), include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_base_contract(
    *,
    prompt: str,
    edits: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    generation_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify BASE at the final-string boundary, not at an earlier saved prompt."""
    assert_target_input_safe(generation_input, allow_analysis_source=False)
    violations: list[str] = []
    if edits:
        violations.append("sent_edits")
    if plans:
        violations.append("plans")
    for marker in BASE_FORBIDDEN_MARKERS:
        if marker in prompt:
            violations.append(marker)
    if generation_input.get("suspicious_nodes_topk"):
        violations.append("suspicious_nodes_topk")
    if generation_input.get("suspicious_nodes_topk_reranked"):
        violations.append("suspicious_nodes_topk_reranked")
    if ((generation_input.get("bm25") or {}).get("top") or []):
        violations.append("retrieval_examples")
    if violations:
        raise ValueError("BASE contract: forbidden plan/context input: " + ", ".join(violations))
    return {
        "policy": "base_prompt_no_ast_retrieval_plan_v1",
        "sent_edits": 0,
        "plans": 0,
        "plan_filter_enabled": False,
        "plan_bonus_enabled": False,
        "forbidden_context_markers": 0,
    }

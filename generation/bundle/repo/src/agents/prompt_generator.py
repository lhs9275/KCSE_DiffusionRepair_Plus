"""버그 수정 프롬프트 생성기.

IJCAI 버전(6.GenerateBugfixPrompt.py)을 모듈화.
메인 API: create_prompt()
"""
from __future__ import annotations

import ast
import difflib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


TASK_GUARDRAILS = (
    "### Task & Guardrails\n"
    "- Apply the smallest correct fix near the buggy line; avoid broad refactors.\n"
    "- Follow the 'Recommended Repair Plan' when provided; ignore it only if it conflicts with the code/tests.\n"
    "- Preserve function/class signature, external behavior, and data ordering/mutability.\n"
    "- Keep existing decorators and wrapper structure; do not drop or reorder decorators when patching.\n"
    "- Do not modify tests or unrelated files; keep imports unchanged unless required to fix the bug.\n"
    "- Do not add imports/logging/prints/dependencies; no TODO/ellipsis/placeholder or NotImplementedError.\n"
    "- Handle edge cases/None safely without changing APIs; keep the fix compact and contained.\n"
)

OUTPUT_PROTOCOL = (
    "### Output Protocol\n"
    "- Block 1: Brief analysis/plan (root cause, minimal patch idea, quick checklist). No code.\n"
    "- Block 2: After the literal marker `##correct`, output ONLY the fixed function/class in plain text. "
    "No markdown fences or extra text; the first line after `##correct` must start with `def` or `class`.\n"
)

PATCH_CHECKLIST = (
    "### Patch Checklist\n"
    "- Identify the concrete root cause using the buggy line/context.\n"
    "- Keep changes inside the target function/file; avoid touching helpers unless necessary.\n"
    "- Preserve inputs/outputs, None-handling, ordering, and side effects.\n"
    "- No new dependencies/logging; minimal diff that compiles and passes listed tests.\n"
)

_MAX_DIFF_LINES = 120
_MAX_PLAN_LINES = 6
_MAX_PLAN_DIST = 2


def _as_str(x: Any, fallback: str = "") -> str:
    return x if isinstance(x, str) else fallback


def _lines(txt: Optional[str]) -> List[str]:
    return (txt or "").replace("\r\n", "\n").splitlines()


def _trim(text: Optional[str], max_lines: int) -> str:
    if not text:
        return ""
    ls = text.replace("\r\n", "\n").splitlines()
    if len(ls) > max_lines:
        return "\n".join(ls[:max_lines]) + "\n... (truncated)"
    return "\n".join(ls)


def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    cur = data
    for key in path:
        try:
            cur = cur[key]
        except Exception:
            return default
    return cur


def _safe_unparse(node: ast.AST) -> str:
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(node).strip()
    except Exception:
        pass
    return "<code>"


def get_buggy_function_code(bug_data: Dict[str, Any]) -> str:
    for code in (
        safe_get(bug_data, ["function", "function_before"]),
        safe_get(bug_data, ["bm25", "top", 0, "function", "function_before"]),
    ):
        if isinstance(code, str) and code.strip():
            return code
    return ""


def _infer_buggy_line_in_function(bug_data: Dict[str, Any]) -> Optional[int]:
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()
    if not buggy_line_content:
        return None
    func_code = get_buggy_function_code(bug_data)
    if not func_code:
        return None
    for idx, line in enumerate(func_code.splitlines(), 1):
        if line.strip() == buggy_line_content:
            return idx
    for idx, line in enumerate(func_code.splitlines(), 1):
        if buggy_line_content in line:
            return idx
    return None


def _unified_diff(a: str, b: str) -> str:
    diff = list(difflib.unified_diff(
        _lines(a), _lines(b), fromfile="before", tofile="after", lineterm="", n=3
    ))
    return _trim("\n".join(diff), _MAX_DIFF_LINES)


def _score_of_item(item: Dict[str, Any]) -> float:
    for k in ("score_reranked", "score_bm25_norm", "score"):
        v = item.get(k)
        try:
            return float(v)
        except Exception:
            continue
    return 0.0


def get_similar_fix_examples(bug_data: Dict[str, Any], count: int) -> List[Dict[str, str]]:
    if count <= 0:
        return []
    top = safe_get(bug_data, ["bm25", "top"], []) or []
    try:
        top = sorted(top, key=_score_of_item, reverse=True)
    except Exception:
        pass
    out: List[Dict[str, str]] = []
    seen: set = set()
    for item in top:
        func = item.get("function", {}) or {}
        before, after = func.get("function_before"), func.get("function_after")
        if not (before and after):
            continue
        key = (str(item.get("project_name")), str(func.get("function_name")))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "project_name": str(item.get("project_name", "unknown")),
            "function_name": str(func.get("function_name", "")),
            "before_code": before,
            "after_code": after,
            "score": f"{_score_of_item(item):.4f}",
        })
        if len(out) >= count:
            break
    return out


def summarize_suspicious_nodes(bug_data: Dict[str, Any], max_lines: int = 30) -> str:
    nodes = safe_get(bug_data, ["suspicious_nodes_topk"], []) or []
    if not nodes:
        return ""
    parts = []
    for i, node in enumerate(nodes[:5], 1):
        code = _trim(_as_str(node.get("code"), ""), 5).strip()
        parts.append(
            f"{i}. **Line {node.get('line', 'N/A')}** (type: `{node.get('type', 'N/A')}`)\n"
            f"   - Code: `{code}`\n"
            f"   - Reason: {_as_str(node.get('reason'), '')}"
        )
    return _trim("\n".join(parts), max_lines)


def _mark_buggy_line(full_code: str, buggy_line_content: str) -> str:
    if not (full_code and buggy_line_content):
        return full_code
    target = buggy_line_content.strip()
    lines = full_code.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == target:
            lines[i] = f"{line}  # <--- BUGGY LINE"
            return "\n".join(lines)
    for i, line in enumerate(lines):
        if target and target in line:
            lines[i] = f"{line}  # <--- BUGGY LINE"
            return "\n".join(lines)
    return full_code


def format_diffusion_plan(
    bug_id: str,
    bug_data: Dict[str, Any],
    diffusion_candidates: List[Dict[str, Any]],
) -> str:
    """디퓨전 계획을 사람이 읽을 수 있는 텍스트로 변환한다."""
    if not diffusion_candidates:
        return ""
    best = diffusion_candidates[0]
    edits = best.get("edits", [])
    if not edits:
        return ""

    # node_id → line/code 매핑
    node_map: Dict[int, Dict[str, Any]] = {}
    for key in ("suspicious_nodes_topk_reranked", "suspicious_nodes_topk"):
        for n in (bug_data.get(key) or []):
            nid = n.get("node_id")
            if nid is not None:
                try:
                    node_map.setdefault(int(nid), n)
                except Exception:
                    pass

    buggy_line_in_func = _infer_buggy_line_in_function(bug_data)
    plan_lines = []
    seen: set = set()
    for ix, e in enumerate(edits):
        op = e.get("op", "Unknown")
        try:
            nid = int(e.get("node", -1))
        except Exception:
            continue
        info = node_map.get(nid)
        if not info:
            continue
        line = info.get("line")
        if line is None:
            continue
        key = (op, line)
        if key in seen:
            continue
        seen.add(key)
        snippet = (info.get("code") or "...")[:57]
        if len(info.get("code") or "") > 60:
            snippet += "..."
        dist = abs(line - buggy_line_in_func) if buggy_line_in_func is not None else None
        plan_lines.append({"line": line, "dist": dist, "text": f"   {ix+1}. **{op}** at Line {line}: `{snippet}`"})

    if not plan_lines:
        return ""

    if buggy_line_in_func is not None:
        near = [p for p in plan_lines if p["dist"] is not None and p["dist"] <= _MAX_PLAN_DIST]
        if near:
            plan_lines = near
        plan_lines = sorted(plan_lines, key=lambda x: (x["dist"] or 9999, x["line"]))[:_MAX_PLAN_LINES]
    else:
        plan_lines = plan_lines[:_MAX_PLAN_LINES]

    return "**AI-Generated Repair Plan (Heuristic):**\n" + "\n".join(p["text"] for p in plan_lines)


def create_prompt(
    bug_id: str,
    bug_data: Dict[str, Any],
    diffusion_candidates: Optional[List[Dict[str, Any]]] = None,
    max_similar: int = 2,
    max_code_lines: int = 200,
    max_list_lines: int = 30,
    max_full_code_lines: int = 1000,
    use_ast: bool = True,
    use_retrieval: bool = True,
    use_diffusion: bool = True,
) -> str:
    """단일 버그에 대한 LLM 수정 프롬프트를 생성한다."""
    project_name = str(bug_data.get("project_name", "N/A"))
    file_path = str(bug_data.get("file_path", "N/A"))
    function_name = str(safe_get(bug_data, ["function", "function_name"], "N/A"))
    buggy_line_content = (_as_str(bug_data.get("buggy_line_content"), "") or "").strip()
    buggy_line_location = bug_data.get("buggy_line_location")
    tests_text = _format_tests(bug_data)
    local_context = _format_buggy_context(bug_data)
    full_buggy_code = get_buggy_function_code(bug_data)

    if full_buggy_code and buggy_line_content:
        full_buggy_code = _mark_buggy_line(full_buggy_code, buggy_line_content)

    pieces: List[str] = []

    if use_diffusion and diffusion_candidates:
        plan = format_diffusion_plan(bug_id, bug_data, diffusion_candidates)
        if plan:
            pieces.append("### Context 0: Recommended Repair Plan (Important)\n" + plan)

    if use_ast:
        susp = summarize_suspicious_nodes(bug_data, max_list_lines)
        if susp:
            pieces.append("### Context 1: Static Analysis Hints\n" + susp)

    if use_retrieval:
        examples = get_similar_fix_examples(bug_data, max_similar)
        if examples:
            blocks = []
            for idx, ex in enumerate(examples, 1):
                diff = _unified_diff(ex["before_code"], ex["after_code"])
                blocks.append(
                    f"#### Example {idx} (score={ex['score']})\n"
                    f"- Project: `{ex['project_name']}`\n"
                    f"- Diff:\n```diff\n{diff}\n```"
                )
            pieces.append("### Context 2: Similar Fix Examples\n" + "\n\n".join(blocks))

    commit_summary = _summarize_commit_info(bug_data, max_code_lines)
    if commit_summary:
        pieces.append("### Context 3: Related Commit Info\n" + commit_summary)

    context_string = (
        "\n\n---\n\n".join(pieces)
        if pieces else "No additional context is available."
    )

    summary_lines = [
        f"- Bug ID: {bug_id}",
        f"- Project: {project_name}",
        f"- File: {file_path}",
        f"- Function: {function_name}",
        f"- Buggy line content: {buggy_line_content}",
    ]
    if buggy_line_location is not None:
        summary_lines.append(f"- Buggy line number: {buggy_line_location}")

    parts = [
        "Task: Fix the Python function. Change only what is necessary to make tests pass.\n\n",
        "Quick facts:\n",
        "\n".join(summary_lines) + "\n\n",
    ]
    if tests_text:
        parts.append(f"Tests to pass (subset):\n{tests_text}\n\n")
    if local_context:
        parts.extend([
            "Local code context around the buggy line:\n",
            "```python\n", f"{local_context}\n", "```\n\n",
        ])
    parts.extend([
        "Buggy function (problem line marked `# <--- BUGGY LINE`):\n",
        "```python\n",
        f"{_trim(full_buggy_code, max_full_code_lines)}\n",
        "```\n\n",
        "Context (use if helpful; prioritize any 'Recommended Repair Plan'):\n",
        f"{context_string}\n\n",
        TASK_GUARDRAILS,
        PATCH_CHECKLIST,
        OUTPUT_PROTOCOL,
        "##correct\n",
    ])
    return "".join(parts)


def _format_tests(bug_data: Dict[str, Any], max_items: int = 6) -> str:
    tests = bug_data.get("test") or []
    if isinstance(tests, str):
        tests = [tests]
    tests = [str(t).strip() for t in tests if str(t).strip()]
    if not tests:
        return ""
    displayed = tests[:max_items]
    more = f"\n- ... (+{len(tests) - max_items} more)" if len(tests) > max_items else ""
    return "- " + "\n- ".join(displayed) + more


def _format_buggy_context(bug_data: Dict[str, Any], max_lines: int = 80) -> str:
    context = bug_data.get("buggy_line_context")
    if isinstance(context, str) and context.strip():
        return _trim(context, max_lines)
    code_line = safe_get(bug_data, ["bm25", "code_line"])
    if isinstance(code_line, str) and code_line.strip():
        return _trim(code_line, max_lines)
    return ""


def _summarize_commit_info(bug_data: Dict[str, Any], max_code_lines: int) -> str:
    msg = (_as_str(safe_get(bug_data, ["commit", "commit_message"]), "") or "").strip()
    diff = _trim(safe_get(bug_data, ["commit", "commit_file_diff"], ""), max_code_lines)
    parts = []
    if msg:
        parts.append(f"- Message: {msg}")
    if diff:
        parts.append(f"- Diff:\n```diff\n{diff}\n```")
    return "\n".join(parts)

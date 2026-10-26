"""Fail-closed checks for answer-bearing prompt and retrieval context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class LeakageAudit:
    passed: bool
    hard_matches: Tuple[str, ...]
    diagnostic_added_line_matches: Tuple[str, ...]


def _normalize(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _record_match(matches: list[str], label: str, needle: Any, haystack: str) -> None:
    normalized = _normalize(needle)
    if normalized and normalized in haystack:
        matches.append(label)


def _normalized_fix_hunks(diff: Any) -> tuple[str, ...]:
    """Return individual unified-diff hunk bodies with whitespace normalized."""
    hunks: list[str] = []
    current: list[str] | None = None
    for line in _normalize(diff).splitlines():
        if line.startswith("@@"):
            if current and any(
                item.startswith("+") and not item.startswith("+++")
                for item in current
            ):
                hunks.append("\n".join(item.strip() for item in current))
            current = []
            continue
        if current is not None and not line.startswith("\\ No newline"):
            current.append(line)
    if current and any(
        item.startswith("+") and not item.startswith("+++") for item in current
    ):
        hunks.append("\n".join(item.strip() for item in current))
    return tuple(hunk for hunk in hunks if hunk)


def audit_prompt(
    prompt: str,
    metadata: Mapping[str, Any],
    retrieved: Sequence[Mapping[str, Any]],
) -> LeakageAudit:
    """Audit a prompt and its retrieval inputs, raising on any hard match."""
    normalized_prompt = _normalize(prompt)
    hard_matches: list[str] = []

    _record_match(
        hard_matches,
        "commit_heading",
        "Related Commit Info",
        normalized_prompt,
    )
    compact_prompt = "\n".join(line.strip() for line in normalized_prompt.splitlines())
    for index, hunk in enumerate(
        _normalized_fix_hunks(_nested(metadata, "commit", "commit_file_diff"))
    ):
        _record_match(
            hard_matches,
            f"commit_fix_hunk:{index}",
            hunk,
            compact_prompt,
        )
    _record_match(
        hard_matches,
        "commit_message",
        _nested(metadata, "commit", "commit_message"),
        normalized_prompt,
    )
    _record_match(
        hard_matches,
        "commit_diff",
        _nested(metadata, "commit", "commit_file_diff"),
        normalized_prompt,
    )
    _record_match(
        hard_matches,
        "fixed_function",
        _nested(metadata, "function", "function_after"),
        normalized_prompt,
    )

    current_id = metadata.get("id")
    current_before = _normalize(_nested(metadata, "function", "function_before"))
    current_after = _normalize(_nested(metadata, "function", "function_after"))
    for index, example in enumerate(retrieved):
        example_id = example.get("id")
        if current_id is not None and example_id is not None:
            if str(example_id) == str(current_id):
                hard_matches.append(f"self_retrieval_id:{index}")
        example_before = _normalize(_nested(example, "function", "function_before"))
        example_after = _normalize(_nested(example, "function", "function_after"))
        if current_before and example_before == current_before:
            hard_matches.append(f"retrieval_before_exact:{index}")
        if current_after and example_after == current_after:
            hard_matches.append(f"retrieval_after_exact:{index}")

    diagnostics: list[str] = []
    diff = _normalize(_nested(metadata, "commit", "commit_file_diff"))
    for line in diff.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        added = line[1:].strip()
        if added and added in normalized_prompt and added not in diagnostics:
            diagnostics.append(added)

    if hard_matches:
        raise ValueError("prompt leakage detected: " + ", ".join(hard_matches))
    return LeakageAudit(
        passed=True,
        hard_matches=(),
        diagnostic_added_line_matches=tuple(diagnostics),
    )

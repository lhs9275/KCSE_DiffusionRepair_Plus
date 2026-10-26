"""Map supplied localization onto normalized before-code, never fixing code."""
from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any, Mapping, Optional


@lru_cache(maxsize=128)
def _map_source_line(raw_code: str, normalized_code: str, target: int) -> int:
    normalized = ast.parse(normalized_code).body[0]
    shape = ast.dump(normalized, include_attributes=False)
    matches = [node for node in ast.walk(ast.parse(raw_code))
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               and node.lineno <= target <= node.end_lineno
               and ast.dump(node, include_attributes=False) == shape]
    if len(matches) != 1:
        raise ValueError("source function is not an unambiguous structural match")
    pairs = list(zip(ast.walk(matches[0]), ast.walk(normalized)))
    selected = [(raw, norm) for raw, norm in pairs if getattr(raw, "lineno", None) == target]
    if selected:
        lines = {norm.lineno for _, norm in selected}
    else:
        covering = [(raw, norm) for raw, norm in pairs
                    if hasattr(raw, "lineno") and raw.lineno <= target <= raw.end_lineno]
        if not covering:
            raise ValueError("bug location is not covered by the source AST")
        width = min(raw.end_lineno - raw.lineno for raw, _ in covering)
        lines = {line for raw, norm in covering if raw.end_lineno - raw.lineno == width
                 for line in range(norm.lineno, norm.end_lineno + 1)}
    if len(lines) != 1:
        raise ValueError(f"normalized bug line is ambiguous: {sorted(lines)}")
    return next(iter(lines))


def normalized_bug_line(metadata: Mapping[str, Any]) -> Optional[int]:
    function = metadata.get("function") or {}
    code = metadata.get("function_before") or function.get("function_before") or ""
    if not code:
        return None
    explicit = metadata.get("bug_line_in_function")
    raw = (metadata.get("file") or {}).get("file_before")
    target = metadata.get("buggy_line_location")
    if raw and target is not None:
        mapped = _map_source_line(raw, code, int(target))
        if explicit is not None and int(explicit) != mapped:
            raise ValueError("explicit bug line disagrees with the source mapping")
        return mapped
    if explicit is not None:
        relative = int(explicit)
    else:
        text = str(metadata.get("buggy_line_content") or "").strip()
        matches = [i for i, line in enumerate(code.splitlines(), 1) if text and line.strip() == text]
        if len(matches) == 1:
            return matches[0]
        if text:
            return None  # Never pick the first duplicate or substring.
        if target is None:
            return None
        # Legacy inputs without raw source/text have only their supplied offset.
        relative = int(target) - int(function.get("function_before_start_line") or 1) + 1
    return relative if 1 <= relative <= len(code.splitlines()) else None

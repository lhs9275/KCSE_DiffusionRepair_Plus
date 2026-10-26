"""JSON I/O 유틸리티."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Union


def load_json(path: Union[str, Path]) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def normalize_dict(data: Any) -> Dict[str, Any]:
    """JSON 최상위가 list이면 dict로 변환."""
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    if isinstance(data, list):
        return {str(i): v for i, v in enumerate(data)}
    raise TypeError(f"지원하지 않는 JSON 최상위 타입: {type(data)}")

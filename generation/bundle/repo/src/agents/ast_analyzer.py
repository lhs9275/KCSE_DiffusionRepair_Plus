"""AST + LLM 기반 의심 노드 분석기.

IJCAI 버전(3.LlmASTReasonAnalyzer.py)을 모듈화.
메인 진입점: analyze_one_bug(), load_analyzer_model()
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import astunparse
except Exception:
    astunparse = None


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


def safe_unparse(node: ast.AST, source: Optional[str] = None) -> str:
    try:
        if hasattr(ast, "unparse"):
            return ast.unparse(node).strip()
    except Exception:
        pass
    if astunparse is not None:
        try:
            return astunparse.unparse(node).strip()
        except Exception:
            pass
    if source is not None:
        try:
            seg = ast.get_source_segment(source, node)
            if seg:
                return seg.strip()
        except Exception:
            pass
    return type(node).__name__


def ast_to_dict_limited(node: ast.AST, max_depth: int = 2, _depth: int = 0) -> Any:
    if node is None:
        return None
    if _depth >= max_depth:
        return {"_type": type(node).__name__}
    result: Dict[str, Any] = {"_type": type(node).__name__}
    for field in getattr(node, "_fields", ()):
        value = getattr(node, field, None)
        if isinstance(value, list):
            result[field] = [
                ast_to_dict_limited(v, max_depth, _depth + 1) if isinstance(v, ast.AST) else v
                for v in value[:10]
            ]
        elif isinstance(value, ast.AST):
            result[field] = ast_to_dict_limited(value, max_depth, _depth + 1)
        else:
            result[field] = str(value)[:120] if isinstance(value, (str, bytes)) else value
    return result


_ASCII_PRINTABLE_RE = re.compile(r"[^\x09\x0a\x0d\x20-\x7E]")


def enforce_english_ascii(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _ASCII_PRINTABLE_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def get_code_context(source: str, lineno: Optional[int], window: int = 3) -> str:
    lines = source.splitlines()
    if lineno is None or lineno < 1 or lineno > len(lines):
        return ""
    start = max(0, lineno - 1 - window)
    end = min(len(lines), lineno + window)
    out = []
    for i in range(start, end):
        prefix = "-> " if i == lineno - 1 else "   "
        out.append(f"{i + 1:4d}:{prefix}{lines[i]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# AST 의심 노드 추출
# ---------------------------------------------------------------------------

def extract_suspicious_nodes_with_ast(
    function_code: str,
    max_depth: int = 2,
) -> List[Dict[str, Any]]:
    class NodeVisitor(ast.NodeVisitor):
        def __init__(self, src: str, walk_index: Dict[int, int]):
            self.src = src
            self.walk_index = walk_index
            self.nodes: List[Dict[str, Any]] = []

        def add(self, node: ast.AST, nodetype: str) -> None:
            local_id = len(self.nodes)
            ast_id = self.walk_index.get(id(node))
            self.nodes.append({
                "type": nodetype,
                "line": getattr(node, "lineno", None),
                "code": safe_unparse(node, self.src),
                "ast_subtree_json": ast_to_dict_limited(node, max_depth=max_depth),
                "node_id": ast_id if ast_id is not None else local_id,
                "node_id_local": local_id,
            })

        def visit_For(self, node: ast.AST) -> None: self.add(node, "For"); self.generic_visit(node)
        def visit_If(self, node: ast.AST) -> None: self.add(node, "If"); self.generic_visit(node)
        def visit_While(self, node: ast.AST) -> None: self.add(node, "While"); self.generic_visit(node)
        def visit_Call(self, node: ast.AST) -> None: self.add(node, "Call"); self.generic_visit(node)
        def visit_Assign(self, node: ast.AST) -> None: self.add(node, "Assign"); self.generic_visit(node)
        def visit_Return(self, node: ast.AST) -> None: self.add(node, "Return"); self.generic_visit(node)
        def visit_With(self, node: ast.AST) -> None: self.add(node, "With"); self.generic_visit(node)
        def visit_Try(self, node: ast.AST) -> None: self.add(node, "Try"); self.generic_visit(node)
        def visit_ExceptHandler(self, node: ast.AST) -> None: self.add(node, "ExceptHandler"); self.generic_visit(node)

    try:
        tree = ast.parse(function_code)
    except Exception as e:
        return [{"error": f"AST parsing failed: {e}"}]
    walk_index = {id(n): i for i, n in enumerate(ast.walk(tree))}
    visitor = NodeVisitor(function_code, walk_index)
    visitor.visit(tree)
    return visitor.nodes


# ---------------------------------------------------------------------------
# 노드 랭킹
# ---------------------------------------------------------------------------

HEURISTIC_BONUS: List[Tuple[str, int]] = [
    ("index", -1), (".pop", 0), ("range(", -1), ("None", 1),
    ("try:", 1), ("except", 1), (".get(", 0), ("deque", 2), ("reversed", 2),
]
TYPE_WEIGHT: Dict[str, int] = {
    "Call": 0, "Assign": 1, "If": 1, "Return": 2,
    "For": 2, "While": 2, "Try": 1, "ExceptHandler": 1, "With": 2,
}


def rank_nodes(
    nodes: List[Dict[str, Any]],
    buggy_line: Optional[int],
    func_start_line: int = 1,
    line_tolerance: int = 2,
) -> List[Dict[str, Any]]:
    scored = []
    for n in nodes:
        rel_line = n.get("line")
        if rel_line is not None and buggy_line is not None:
            abs_line = func_start_line + rel_line - 1
            raw_dist = abs(abs_line - buggy_line)
            dist = 0 if raw_dist <= line_tolerance else raw_dist - line_tolerance
        else:
            dist = None
            abs_line = None

        weight = TYPE_WEIGHT.get(n.get("type"), 3)
        code = (n.get("code") or "").strip()
        bonus = sum(val for sig, val in HEURISTIC_BONUS if sig in code)
        score = (dist if dist is not None else 9999) * 10 + weight + bonus

        m = dict(n)
        m["_score"] = score
        m["_dist"] = dist
        m["_bonus"] = bonus
        if abs_line is not None:
            m["line_absolute"] = abs_line
        scored.append(m)
    scored.sort(key=lambda x: x["_score"])
    return scored


# ---------------------------------------------------------------------------
# LLM 리랭킹
# ---------------------------------------------------------------------------

_REASON_KEYWORDS: Dict[str, float] = {
    "indexerror": 3.0, "keyerror": 2.2, "typeerror": 2.0, "valueerror": 1.6,
    "zerodivision": 2.4, "none": 1.8, "null": 1.2, "empty": 1.4, "off-by-one": 1.6,
    "boundary": 1.2, "out of range": 1.8, "logic": 3.0, "order": 3.0,
    "sequence": 2.5, "reverse": 2.5, "intended": 2.0, "mismatch": 2.0,
    "incorrect": 1.5, "behavior": 1.5, "deque": 2.5, "direction": 2.0, "missing": 1.4,
}
_CODE_CUES: Dict[str, float] = {
    "range(": 0.6, "len(": 0.5, "[": 0.5, ".get(": 0.4,
    ".append": 0.3, ".pop": 0.3, "try:": 0.4, "except": 0.4,
    "== None": 0.6, "is None": 0.8, "!= None": 0.4,
}


def _kw_score(text: str, table: Dict[str, float]) -> float:
    t = (text or "").lower()
    return sum(w for k, w in table.items() if k in t)


def rerank_by_reason_and_code(
    nodes_topk: List[Dict[str, Any]],
    suspicious_results: List[Dict[str, Any]],
    alpha_reason: float = 8.0,
    beta_code: float = 3.0,
    gamma_dist: float = 0.0,
) -> List[Dict[str, Any]]:
    ranked = []
    for n, r in zip(nodes_topk, suspicious_results):
        base = float(n.get("_score", 1e4))
        reason = str(r.get("reason", ""))
        code = str(n.get("code", "") or r.get("code", ""))
        dist = n.get("_dist", None)
        final = base - alpha_reason * _kw_score(reason, _REASON_KEYWORDS) - beta_code * _kw_score(code, _CODE_CUES)
        if dist is not None and gamma_dist > 0:
            final += gamma_dist * max(0, dist)
        rr = dict(r)
        rr["reranker_score"] = round(final, 3)
        ranked.append(rr)
    ranked.sort(key=lambda x: x["reranker_score"])
    return ranked


# ---------------------------------------------------------------------------
# LLM 캐시
# ---------------------------------------------------------------------------

class LLMCache:
    def __init__(self, cache_path: Optional[str] = None, max_items: int = 50000):
        self.cache_path = cache_path
        self.max_items = max_items
        self.mem: Dict[str, Dict[str, Any]] = {}
        if cache_path and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.mem = json.load(f)
            except Exception:
                self.mem = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.mem.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if len(self.mem) >= self.max_items:
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = value

    def flush(self) -> None:
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.mem, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM 배치 생성
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"(\[\s*\{(?:.|\n)*?\}\s*\])", re.MULTILINE)


def _extract_last_json_array(text: str) -> Any:
    for raw in reversed(_JSON_BLOCK_RE.findall(text or "")):
        try:
            return json.loads(raw)
        except Exception:
            continue
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return []


def _build_messages(code_context: str, node: Dict[str, Any]) -> List[Dict[str, str]]:
    user_prompt = (
        "Analyze the Python code snippet for potential bugs.\n"
        "Focus on TWO types of bugs:\n"
        "1. Runtime Errors: (e.g., IndexError, KeyError, TypeError, NoneType)\n"
        "2. Logical Errors: (e.g., Wrong execution order, side effects, misuse of data structures)\n\n"
        f"Code Context:\n{code_context}\n\n"
        f"Candidate Suspicious Node:\n{json.dumps([node], indent=2, ensure_ascii=False)}\n\n"
        "Explain in ONE short English sentence why this node is suspicious.\n"
        'Return EXACTLY ONE JSON array with a single object. Keys: "line" (int), "reason" (English, <= 20 words).\n'
        'Example: [{"line": 15, "reason": "Appending to deque without checking order requirements"}]'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python Static Analyzer. "
                "You are sensitive to both runtime crashes AND logical flaws. "
                "You MUST output ONLY a valid JSON array. "
                "Do NOT output markdown blocks or extra text."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


@torch.no_grad()
def query_llm_batch(
    tokenizer: Any,
    model: Any,
    messages_list: List[List[Dict[str, str]]],
    sampling_params: Dict[str, Any],
    max_new_tokens: int = 192,
    timeout_s: int = 30,
    retry: int = 1,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [{} for _ in range(len(messages_list))]
    bs = max(1, int(batch_size))

    def _msgs_to_text(msgs: List[Dict[str, str]]) -> str:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Fallback for base models (e.g. StarCoder2) without a chat template
        parts = []
        for m in msgs:
            role, content = m.get("role", ""), m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _single_pass(msgs_batch: List, idx_offset: int) -> None:
        texts = [_msgs_to_text(m) for m in msgs_batch]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id, **sampling_params)
        start = time.time()
        outputs = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            **gen_kwargs,
        )
        if time.time() - start > timeout_s:
            raise TimeoutError(f"LLM generation exceeded {timeout_s}s")
        for bi in range(outputs.size(0)):
            gen_ids = outputs[bi][len(model_inputs.input_ids[bi]):]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            parsed = _extract_last_json_array(text)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and \
               "line" in parsed[0] and "reason" in parsed[0]:
                obj = dict(parsed[0])
                obj["reason"] = enforce_english_ascii(obj.get("reason"))
                results[idx_offset + bi] = obj
            else:
                results[idx_offset + bi] = {
                    "line": None,
                    "reason": f"No valid JSON. Raw: {enforce_english_ascii(text[:200])}..."
                }

    for attempt in range(retry + 1):
        try:
            for i in range(0, len(messages_list), bs):
                _single_pass(messages_list[i:i + bs], i)
            break
        except (torch.cuda.OutOfMemoryError, TimeoutError) as e:
            torch.cuda.empty_cache()
            if attempt >= retry:
                for j in range(len(results)):
                    if not results[j]:
                        results[j] = {"line": None, "reason": f"LLM error: {type(e).__name__}"}
            else:
                bs = max(2, bs // 2)
    return results


# ---------------------------------------------------------------------------
# 모델 로딩
# ---------------------------------------------------------------------------

def load_analyzer_model(
    model_path: str,
    max_memory_gb: int = 8,
) -> Tuple[Any, Any]:
    """AST 분석용 LLM 로드 (4-bit 양자화)."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )
    max_memory_config = {0: f"{max_memory_gb}GiB", "cpu": "90GiB"}
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    kwargs: Dict[str, Any] = dict(
        quantization_config=quantization_config,
        device_map="auto",
        max_memory=max_memory_config,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, attn_implementation="flash_attention_2", **kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# 단일 버그 분석 (메인 API)
# ---------------------------------------------------------------------------

def analyze_one_bug(
    bug_id: str,
    item: Dict[str, Any],
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
    topk: int = 10,
    max_ast_depth: int = 2,
    cache: Optional[LLMCache] = None,
    max_new_tokens: int = 192,
    gen_timeout: int = 30,
    batch_size: int = 8,
    enable_rerank: bool = True,
    rerank_alpha_reason: float = 8.0,
    rerank_beta_code: float = 3.0,
    rerank_gamma_dist: float = 0.0,
) -> Dict[str, Any]:
    """단일 버그에 대해 AST 추출 + LLM 이유 분석 + 리랭크를 수행한다."""
    function_info = item.get("function", {}) or {}
    function_before = function_info.get("function_before", "") or ""
    func_start_line = int(function_info.get("function_before_start_line") or 1)
    buggy_line_location = item.get("buggy_line_location")
    if buggy_line_location is not None:
        try:
            buggy_line_location = int(buggy_line_location)
        except Exception:
            buggy_line_location = None

    if not function_before:
        return {"error": f"bug_id={bug_id} has empty function_before", **item}

    nodes = extract_suspicious_nodes_with_ast(function_before, max_depth=max_ast_depth)
    if nodes and "error" in nodes[0]:
        return {"error": nodes[0]["error"], **item}

    nodes_ranked = rank_nodes(nodes, buggy_line_location, func_start_line=func_start_line)
    nodes_topk = nodes_ranked[:topk]

    msg_list: List[List[Dict[str, str]]] = []
    cache_keys: List[str] = []
    suspicious_results: List[Dict[str, Any]] = []

    for node in nodes_topk:
        line = node.get("line")
        code_context = get_code_context(function_before, line, window=3)
        node_payload = {
            "line": line,
            "line_absolute": node.get("line_absolute"),
            "node_id": node.get("node_id"),
            "node_id_local": node.get("node_id_local"),
            "type": node.get("type"),
            "code": node.get("code"),
            "distance_to_buggy": node.get("_dist"),
            "context": code_context,
            "ast_subtree_json": node.get("ast_subtree_json"),
        }
        key = sha1(json.dumps({"prompt": code_context, "node": node}, ensure_ascii=False))
        cache_keys.append(key)
        cached = cache.get(key) if cache else None
        if cached:
            suspicious_results.append({**node_payload, "reason": enforce_english_ascii(cached.get("reason"))})
        else:
            suspicious_results.append({**node_payload, "reason": "LLM pending"})
            msg_list.append(_build_messages(code_context, node))

    if msg_list and model and tokenizer and sampling_params:
        outs = query_llm_batch(
            tokenizer, model, msg_list, sampling_params,
            max_new_tokens=max_new_tokens, timeout_s=gen_timeout,
            retry=1, batch_size=batch_size,
        )
        oi = 0
        for i in range(len(suspicious_results)):
            if suspicious_results[i]["reason"] == "LLM pending":
                res = outs[oi] if oi < len(outs) else {"line": None, "reason": "no_output"}
                suspicious_results[i]["reason"] = enforce_english_ascii(res.get("reason"))
                if cache:
                    cache.put(cache_keys[i], {"reason": suspicious_results[i]["reason"]})
                oi += 1
    elif msg_list:
        for i in range(len(suspicious_results)):
            if suspicious_results[i]["reason"] == "LLM pending":
                suspicious_results[i]["reason"] = "LLM not executed"

    if enable_rerank:
        suspicious_results_reranked = rerank_by_reason_and_code(
            nodes_topk, suspicious_results,
            alpha_reason=rerank_alpha_reason,
            beta_code=rerank_beta_code,
            gamma_dist=rerank_gamma_dist,
        )
    else:
        suspicious_results_reranked = suspicious_results

    result = item.copy()
    rel_buggy_line = None
    if buggy_line_location is not None and function_before:
        rel = buggy_line_location - func_start_line + 1
        total_lines = len(function_before.splitlines())
        rel_buggy_line = min(max(rel, 1), total_lines) if total_lines > 0 else None
    result["buggy_line_context"] = get_code_context(function_before, rel_buggy_line, window=3)
    result["suspicious_nodes_topk"] = suspicious_results
    result["suspicious_nodes_topk_reranked"] = suspicious_results_reranked
    return result

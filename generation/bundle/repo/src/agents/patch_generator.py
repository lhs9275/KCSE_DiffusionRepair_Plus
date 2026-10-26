"""LLM 기반 패치 생성 에이전트.

IJCAI 버전(7.TokenPatchGeneratorAgent.py)을 모듈화.
메인 API: PatchGeneratorClient, generate_candidates_for_bug()
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import logging
import os
import re
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
DEFCLASS_HEAD_RE = re.compile(r"^\s*(def|class)\s", re.MULTILINE)
REPAIR_PLAN_HEADER_RE = re.compile(r"^### Context 0: Recommended Repair Plan", re.IGNORECASE)
REPAIR_PLAN_LINE_RE = re.compile(r"\*\*(?P<op>[A-Za-z]+)\*\* at Line \d+: `(?P<code>[^`]+)`")
BANNED_PATTERNS = [r"\.\.\.", r"\braise\s+NotImplementedError\b", r"\bprint\s*\("]
BANNED_TOKENS = [re.compile(p) for p in BANNED_PATTERNS]


# ---------------------------------------------------------------------------
# 코드 파싱 유틸
# ---------------------------------------------------------------------------

def safe_get(data: Any, path: List[Any], default: Any = None) -> Any:
    cur = data
    for k in path:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def canon(code: str) -> str:
    s = re.sub(r"#.*", "", code)
    return re.sub(r"\s+", " ", s).strip()


def ast_ok(py_text: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(py_text)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def extract_correct_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"^[ \t]*##correct[ \t]*$", text, flags=re.IGNORECASE | re.MULTILINE)
    sub = text[m.end():].strip() if m else text.strip()
    m2 = CODE_FENCE_RE.search(sub)
    candidate = (m2.group(1) if m2 else sub).strip()
    candidate = re.split(r"^###\s", candidate, flags=re.MULTILINE)[0].strip()
    m3 = DEFCLASS_HEAD_RE.search(candidate)
    if m3:
        candidate = candidate[m3.start():]
    candidate = candidate.replace("\x00", "").replace("```", "")
    try:
        candidate = textwrap.dedent(candidate).strip()
    except IndentationError:
        pass
    return candidate


def _first_toplevel_def_or_class(py_text: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(py_text)
    except Exception:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node
    return None


def validate_candidate_code(code: str, base_code: str, expected_name: str) -> Tuple[bool, str]:
    if not code or not code.strip():
        return False, "empty"
    if canon(code) == canon(base_code):
        return False, "no_change"
    node = _first_toplevel_def_or_class(code)
    if node is None:
        return False, "no_def"
    if expected_name and getattr(node, "name", None) != expected_name:
        return False, "name_mismatch"
    return True, ""


def change_fraction(base_code: str, code: str) -> float:
    try:
        base_lines = base_code.splitlines()
        new_lines = code.splitlines()
        if not base_lines:
            return 1.0
        changed = sum(
            (ahi - alo)
            for tag, alo, ahi, blo, bhi in difflib.SequenceMatcher(None, base_lines, new_lines).get_opcodes()
            if tag != "equal"
        )
        return changed / max(1, len(base_lines))
    except Exception:
        return 0.0


def length_ratio(base_code: str, code: str) -> float:
    try:
        return len(code.splitlines()) / max(1, len(base_code.splitlines()))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# 계획 준수 유틸
# ---------------------------------------------------------------------------

def _extract_repair_plan_hints(prompt: str) -> List[Tuple[str, str]]:
    hints: List[Tuple[str, str]] = []
    in_plan = False
    for line in (prompt or "").splitlines():
        if REPAIR_PLAN_HEADER_RE.search(line):
            in_plan = True
            continue
        if in_plan and line.startswith("### "):
            break
        if in_plan:
            m = REPAIR_PLAN_LINE_RE.search(line)
            if m:
                hints.append((m.group("op").strip(), m.group("code").strip()))
    return hints


def _repair_hint_bonus(prompt: str, base_code: str, code: str) -> int:
    hints = _extract_repair_plan_hints(prompt)
    if not hints:
        return 0
    bonus = 0.0
    for op, snippet in hints:
        if not snippet or snippet == "..." or snippet not in base_code:
            continue
        in_cand = snippet in code
        op_lower = op.lower()
        if op_lower in ("replace", "delete"):
            bonus += 1.0 if not in_cand else -0.25
        elif op_lower in ("insertbefore", "insertafter", "guardadd"):
            if in_cand:
                bonus += 0.5
    return int(round(max(0.0, min(bonus, 4.0)) * 10))


def _plan_compliance(prompt: str, base_code: str, code: str, hints: Optional[List[Tuple[str, str]]] = None) -> bool:
    if hints is None:
        hints = _extract_repair_plan_hints(prompt)
    if not hints:
        return True
    base_lines = base_code.splitlines()
    cand_lines = code.splitlines()
    hint_lines: set = set()
    for _op, snippet in hints:
        snippet = (snippet or "").strip()
        if snippet.endswith("..."):
            snippet = snippet[:-3].rstrip()
        if not snippet:
            continue
        for i, line in enumerate(base_lines):
            if snippet in line:
                hint_lines.add(i)
    if not hint_lines:
        return True
    sm = difflib.SequenceMatcher(None, base_lines, cand_lines)
    changed: set = set()
    inserted_at: set = set()
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            changed.update(range(i1, i2))
        elif tag == "insert":
            inserted_at.add(i1)
    for idx in hint_lines:
        if idx in changed or idx in inserted_at or (idx + 1) in inserted_at:
            return True
    return False


# ---------------------------------------------------------------------------
# 후보 스코어링
# ---------------------------------------------------------------------------

def score_candidate(
    context_row: Dict[str, Any],
    prom_row: Dict[str, Any],
    base_code: str,
    code: str,
    ast_pass: bool,
) -> int:
    if not code.strip() or not ast_pass:
        return -1000 if not code.strip() else -500
    score = 100
    buggy_line = (safe_get(prom_row, ["buggy_line_content"]) or "").strip()
    if buggy_line:
        if buggy_line in code and buggy_line in base_code:
            score -= 15
        elif buggy_line in base_code and buggy_line not in code:
            score += 15
    fn_name = (safe_get(context_row, ["function", "function_name"]) or context_row.get("function_name") or "").strip()
    if fn_name and re.search(rf"^\s*(def|class)\s+{re.escape(fn_name)}\b", code, re.M):
        score += 35
    else:
        score -= 120
    if base_code:
        score += int(difflib.SequenceMatcher(None, canon(base_code), canon(code)).ratio() * 45)
        diff_frac = change_fraction(base_code, code)
        if diff_frac < 0.005:
            score -= 80
        elif diff_frac <= 0.03:
            score += 40
        elif diff_frac <= 0.12:
            score += 30
        elif diff_frac <= 0.25:
            score += 20
        elif diff_frac <= 0.5:
            score += 5
        else:
            score -= 10
        lr = length_ratio(base_code, code)
        if 0.8 <= lr <= 1.2:
            score += 20
        elif 0.6 <= lr <= 1.5:
            score += 8
        elif lr < 0.25 or lr > 2.5:
            score -= 30
    prompt_text = prom_row.get("prompt") if isinstance(prom_row, dict) else ""
    score += _repair_hint_bonus(prompt_text, base_code, code)
    for pat in BANNED_TOKENS:
        if pat.search(code):
            score -= 80
    if re.search(r"^\s*(import|from)\s+", code, flags=re.M):
        score -= 200
    n_lines = len(code.strip().splitlines())
    if n_lines <= 2:
        score -= 30
    elif n_lines > 200:
        score -= 25
    return score


# ---------------------------------------------------------------------------
# LLM 클라이언트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python bug fixer. "
    "Follow the user's guardrails exactly. "
    "After the marker `##correct`, output ONLY the fixed function/class code; "
    "no analysis, no markdown fences, no extra text."
)


class PatchGeneratorClient:
    """StarCoder2/DeepSeek 계열 LLM 래퍼 (4-bit 양자화, torch.compile)."""

    def __init__(self, model_name: str, system_prompt: str = SYSTEM_PROMPT):
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.system_prompt = system_prompt.strip()

        use_fa2 = False
        try:
            import flash_attn  # noqa
            use_fa2 = True
        except Exception:
            pass
        attn_impl = "flash_attention_2" if use_fa2 else "sdpa"

        try:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                quantization_config=qconfig, dtype=dtype,
                attn_implementation=attn_impl, device_map="auto",
            )
        except Exception as e:
            logging.warning(f"4-bit 로드 실패 ({e}), 일반 로드 시도")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                attn_implementation=attn_impl, device_map="auto",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            self.model = torch.compile(self.model)
            logging.info("torch.compile 활성화됨.")
        except Exception as e:
            logging.warning(f"torch.compile 스킵: {e}")

        self.metrics: Dict[str, float] = {
            "input_tokens": 0.0, "output_tokens": 0.0,
            "batches": 0.0, "max_gpu_mb": 0.0,
        }

    def _gpu_mem_mb(self) -> float:
        try:
            if torch.cuda.is_available():
                return float(torch.cuda.memory_allocated() / (1024 * 1024))
        except Exception:
            pass
        return 0.0

    @torch.no_grad()
    def generate(self, prompts: List[str], gen_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        chats = []
        for p in prompts:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": p})
            try:
                chats.append(self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))
            except Exception:
                chats.append(p)

        enc = self.tokenizer(
            chats, return_tensors="pt", padding=True, truncation=True,
            max_length=gen_args.get("max_input_tokens", 4096),
        ).to(self.model.device)
        input_lens = enc.attention_mask.sum(dim=1).tolist()

        bsz = enc.input_ids.shape[0]
        outputs = []
        batch_size = gen_args.get("batch_size", 4)
        start = 0
        while start < bsz:
            end = min(start + batch_size, bsz)
            cur = {k: v[start:end] for k, v in enc.items()}
            gen_out = self.model.generate(
                **cur,
                do_sample=gen_args.get("do_sample", True),
                temperature=float(gen_args.get("temperature", 0.8)),
                top_p=float(gen_args.get("top_p", 0.9)),
                repetition_penalty=float(gen_args.get("repetition_penalty", 1.05)),
                max_new_tokens=int(gen_args.get("max_new_tokens", 480)),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            for i in range(gen_out.size(0)):
                txt = self.tokenizer.decode(gen_out[i], skip_special_tokens=True)
                out_tokens = max(0, gen_out[i].size(0) - int(input_lens[start + i]))
                outputs.append({"text": txt, "tokens_in": int(input_lens[start + i]), "tokens_out": out_tokens})
            gpu_mb = self._gpu_mem_mb()
            self.metrics["max_gpu_mb"] = max(self.metrics["max_gpu_mb"], gpu_mb)
            self.metrics["batches"] += 1
            self.metrics["input_tokens"] += sum(input_lens[start:end])
            self.metrics["output_tokens"] += sum(max(0, gen_out[j].size(0) - int(input_lens[start + j])) for j in range(gen_out.size(0)))
            start = end
        return outputs


# ---------------------------------------------------------------------------
# 가이드 프롬프트 구성
# ---------------------------------------------------------------------------

def _build_guided_prompt(
    base_code: str,
    edits: List[Dict[str, Any]],
    full_prompt: Optional[str],
) -> str:
    edits_txt = "\n".join(
        f"# edit{ix+1}: {e.get('op', '')} node={e.get('node', '')}"
        for ix, e in enumerate(edits[:6])
    )

    if full_prompt and full_prompt.strip():
        strict_note = (
            "IMPORTANT:\n"
            "- If a 'Recommended Repair Plan' is present above, you MUST follow it.\n"
            "- Modify ONLY lines covered by the repair plan.\n"
            "- After `##correct`, output ONLY the fixed code. No analysis, no markdown.\n"
        )
        # ##correct 직전에 삽입
        m = None
        for mm in re.finditer(r"^[ \t]*##correct[ \t]*$", full_prompt, re.IGNORECASE | re.MULTILINE):
            m = mm
        if m:
            before = full_prompt[:m.start()].rstrip()
            prompt = before + "\n\n" + strict_note
            if edits_txt:
                prompt += f"\nRequired edit hints:\n{edits_txt}\n"
            prompt += "\n\n##correct\n"
        else:
            prompt = full_prompt.rstrip() + "\n\n" + strict_note + "\n\n##correct\n"
        return prompt

    guide = (
        "Fix the following Python function. The patch must be executable with no syntax errors.\n"
        "- Keep the function signature identical and avoid unrelated edits\n"
        "- Do NOT add extra import/print/ellipsis/NotImplementedError\n"
        "- Start the patched code immediately, beginning with `def ...`\n"
    )
    edits_block = f"Required edit hints:\n{edits_txt}\n\n" if edits_txt else ""
    prompt = f"{guide}\n{edits_block}[Original]\n{textwrap.dedent(base_code).strip()}\n\n##correct\n"
    return prompt


# ---------------------------------------------------------------------------
# 후보 생성 (메인 API)
# ---------------------------------------------------------------------------

def generate_candidates_for_bug(
    client: PatchGeneratorClient,
    bug_id: str,
    base_code: str,
    edits: List[Dict[str, Any]],
    prom_row: Dict[str, Any],
    target: int = 20,
    batch_size: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
    max_new_tokens: int = 640,
    no_rerank: bool = False,
) -> List[Dict[str, Any]]:
    """단일 버그에 대해 최대 target개의 수정 후보를 생성한다."""
    full_prompt = prom_row.get("prompt") if isinstance(prom_row, dict) else None
    prompt = _build_guided_prompt(base_code, edits, full_prompt)
    plan_hints = _extract_repair_plan_hints(prompt)
    plan_required = bool(plan_hints)

    base_node = _first_toplevel_def_or_class(base_code)
    expected_name = (
        safe_get(prom_row, ["function", "function_name"])
        or getattr(base_node, "name", "")
        or ""
    ).strip()

    sampling_grid = [
        (temperature, top_p),
        (max(0.35, temperature * 0.65), min(0.99, top_p + 0.08)),
        (min(1.15, temperature * 1.15), max(0.7, top_p * 0.9)),
    ]

    seen_hashes: set = set()
    candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    max_attempts = max(target * 10, target * max(2, len(sampling_grid)))

    def _try_add(text: str, temp: float, tp: float,
                 tokens_in: Optional[int] = None,
                 tokens_out: Optional[int] = None,
                 elapsed_sec: Optional[float] = None) -> None:
        code = extract_correct_block(text)
        if not code.strip():
            return
        ok, err = ast_ok(code)
        if not ok:
            return
        is_valid, _ = validate_candidate_code(code, base_code, expected_name)
        if not is_valid:
            return
        h = hashlib.sha1(canon(code).encode("utf-8")).hexdigest()
        if h in seen_hashes:
            return
        seen_hashes.add(h)
        sc = score_candidate(
            {"function_name": expected_name},
            prom_row, base_code, code, ok,
        )
        plan_ok = _plan_compliance(prompt, base_code, code, plan_hints)
        entry = {
            "code": code, "text": text,
            "ast": {"ok": ok, "err": err},
            "score": sc,
            "temperature": float(temp), "top_p": float(tp),
            "plan_ok": plan_ok,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "elapsed_sec": elapsed_sec,
        }
        if plan_required and not plan_ok:
            fallback_candidates.append(entry)
        else:
            candidates.append(entry)

    # Greedy 패스 1회
    _t0 = time.time()
    out = client.generate([prompt], {
        "batch_size": 1, "max_new_tokens": max_new_tokens,
        "do_sample": False, "temperature": 0.0, "top_p": 1.0,
        "repetition_penalty": repetition_penalty, "max_input_tokens": 4096,
    })[0]
    _greedy_elapsed = time.time() - _t0
    _try_add(out.get("text") or "", 0.0, 1.0,
             tokens_in=out.get("tokens_in"), tokens_out=out.get("tokens_out"),
             elapsed_sec=_greedy_elapsed)

    # 샘플링 패스
    attempts = 0
    while attempts < max_attempts and len(candidates) < target:
        base_temp, base_tp = sampling_grid[attempts % len(sampling_grid)]
        import random as _random
        jitter = 0.08 * (_random.random() - 0.5)
        temp = max(0.2, min(1.2, base_temp + base_temp * jitter))
        tp = max(0.6, min(0.98, base_tp + jitter))
        cur_batch = min(batch_size, max_attempts - attempts)
        _t0 = time.time()
        outs = client.generate([prompt] * cur_batch, {
            "batch_size": cur_batch, "max_new_tokens": max_new_tokens,
            "do_sample": True, "temperature": temp, "top_p": tp,
            "repetition_penalty": repetition_penalty, "max_input_tokens": 4096,
        })
        _batch_elapsed = (time.time() - _t0) / max(1, len(outs))
        for o in outs:
            _try_add(o.get("text") or "", temp, tp,
                     tokens_in=o.get("tokens_in"), tokens_out=o.get("tokens_out"),
                     elapsed_sec=_batch_elapsed)
        attempts += cur_batch

    # plan_compliance 통과 후보가 없으면 fallback 사용
    if not candidates and fallback_candidates:
        candidates = fallback_candidates

    if not no_rerank:
        candidates.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return candidates[:target]

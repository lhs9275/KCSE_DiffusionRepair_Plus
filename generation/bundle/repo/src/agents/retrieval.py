"""BM25 + Dense (E5) 하이브리드 검색 에이전트.

IJCAI 버전(4.Bm25.py)을 모듈화.
메인 API: RetrievalEngine — build(), query()
"""
from __future__ import annotations

import json
import os
import re
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise SystemExit("rank_bm25 미설치: pip install rank-bm25") from e

try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


# ---------------------------------------------------------------------------
# 토큰화 유틸
# ---------------------------------------------------------------------------

_token_re = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    return _token_re.findall((text or "").lower())


def _repeat_tokens(tokens: List[str], weight: float) -> List[str]:
    n = int(weight)
    frac = weight - n
    out = tokens * max(n, 0)
    if frac > 0:
        out += tokens[:max(1, int(len(tokens) * frac))]
    return out


def tokens_from_bug_entry(
    entry: Dict[str, Any],
    w_buggy: float = 3.0,
    w_func: float = 0.6,
    w_susp: float = 1.2,
    w_file: float = 0.4,
    w_reason: float = 0.8,
) -> List[str]:
    toks: List[str] = []
    toks += _repeat_tokens(tokenize(entry.get("buggy_line_content") or ""), w_buggy)
    func = entry.get("function") or {}
    func_code = (func.get("function_before") or func.get("code") or "") if isinstance(func, dict) else ""
    if func_code:
        toks += _repeat_tokens(tokenize(func_code)[:256], w_func)
    for node in (entry.get("suspicious_nodes_topk") or [])[:5]:
        if not isinstance(node, dict):
            continue
        toks += _repeat_tokens(tokenize(node.get("code") or "")[:64], w_susp)
        toks += _repeat_tokens(tokenize(node.get("reason") or "")[:32], w_reason)
    file_info = entry.get("file") or {}
    if isinstance(file_info, dict):
        toks += _repeat_tokens(tokenize(file_info.get("file_name") or ""), w_file)
        toks += _repeat_tokens(tokenize(file_info.get("file_path") or ""), w_file)
    toks += _repeat_tokens(tokenize(entry.get("project_name") or ""), 0.3)
    return [t for t in toks if t]


# ---------------------------------------------------------------------------
# 쿼리 확장
# ---------------------------------------------------------------------------

_REASON_EXPAND = [
    ("indexerror", ["index", "out", "range", "len", "bounds"]),
    ("out of range", ["index", "range", "bounds", "len"]),
    ("off-by-one", ["index", "range", "len"]),
    ("keyerror", ["dict", "get", "in", "keys"]),
    ("typeerror", ["isinstance", "type", "cast"]),
    ("none", ["is", "==", "null", "empty"]),
    ("empty", ["len", "==", "0"]),
]
_CODE_CUES_EXPAND = [
    ("range(", ["index", "len"]),
    ("len(", ["index", "empty"]),
    (".get(", ["dict", "key"]),
    ("is None", ["none"]),
    ("== None", ["none"]),
]


def expand_query_tokens(tokens: List[str], func_obj: Dict[str, Any]) -> List[str]:
    out = list(tokens)
    reasons = " ".join(
        node.get("reason", "").lower()
        for node in (func_obj.get("suspicious_nodes_topk") or [])[:5]
        if isinstance(node, dict)
    )
    for k, adds in _REASON_EXPAND:
        if k in reasons:
            out += adds
    bl = func_obj.get("buggy_line_content") or ""
    base_text = " ".join(tokens)
    for cue, adds in _CODE_CUES_EXPAND:
        if cue in bl or cue in base_text:
            out += adds
    seen: set = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


# ---------------------------------------------------------------------------
# Dense 인코더
# ---------------------------------------------------------------------------

def _l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


def _mean_pool(last_hidden: "torch.Tensor", attn_mask: "torch.Tensor") -> "torch.Tensor":
    mask = attn_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class DenseEncoder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 64):
        if not _HAS_TORCH:
            raise RuntimeError("torch가 필요합니다.")
        self.model_name = model_name
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.batch_size = max(1, int(batch_size))
        self._use_st = False
        self.model = None
        self.tokenizer = None

        if _HAS_ST:
            try:
                self.model = SentenceTransformer(model_name, device=self.device)
                self._use_st = True
                return
            except Exception:
                warnings.warn("SentenceTransformer 로드 실패, HF AutoModel로 폴백합니다.")

        from transformers import AutoModel, AutoTokenizer as _Tok
        self.tokenizer = _Tok.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + (t or "") for t in texts]
        if self._use_st:
            v = self.model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )
            return v.astype(np.float32, copy=False)
        embs = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            toks = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**toks)
            pooled = _mean_pool(out.last_hidden_state, toks["attention_mask"])
            embs.append(pooled.cpu().detach().numpy())
        x = np.concatenate(embs, axis=0)
        return _l2_normalize(x.astype(np.float32, copy=False))


# ---------------------------------------------------------------------------
# 검색 엔진
# ---------------------------------------------------------------------------

def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx, mn = max(scores), min(scores)
    if mx == mn:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _file_ext(name: Optional[str]) -> str:
    if not name:
        return ""
    _, ext = os.path.splitext(os.path.basename(name))
    return ext.lower()


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class RetrievalEngine:
    """BM25 + Dense(E5) 하이브리드 검색 엔진."""

    def __init__(
        self,
        all_bugs: Dict[str, Dict[str, Any]],
        bm25_k1: float = 1.6,
        bm25_b: float = 0.7,
        corpus_cache: Optional[str] = None,
        dense_model_path: Optional[str] = None,
        dense_device: str = "auto",
        dense_batch: int = 64,
        dense_cache: Optional[str] = None,
        dense_weight: float = 0.6,
        stage_topm: int = 200,
        **token_weights: float,
    ):
        self.all_bugs = all_bugs
        self.dense_weight = dense_weight
        self.stage_topm = stage_topm

        # BM25 인덱스 구축
        bug_id_list: List[str] = []
        tokenized_corpus: List[List[str]] = []
        fp = f"{len(all_bugs)}"
        loaded_from_cache = False

        if corpus_cache and os.path.exists(corpus_cache):
            try:
                cache_data = json.loads(open(corpus_cache, encoding="utf-8").read())
                if (cache_data.get("meta", {}).get("fingerprint") == fp
                        and cache_data.get("meta", {}).get("k1") == bm25_k1
                        and cache_data.get("meta", {}).get("b") == bm25_b):
                    tokenized_corpus = cache_data["tokenized_corpus"]
                    bug_id_list = cache_data["bug_id_list"]
                    loaded_from_cache = True
            except Exception:
                pass

        if not loaded_from_cache:
            for k, v in all_bugs.items():
                doc_id = str(v.get("bug_id", v.get("id", k)))
                toks = tokens_from_bug_entry(v, **{
                    kk: vv for kk, vv in token_weights.items()
                    if kk in ("w_buggy", "w_func", "w_susp", "w_file", "w_reason")
                })
                bug_id_list.append(doc_id)
                tokenized_corpus.append(toks)
            if corpus_cache:
                os.makedirs(os.path.dirname(corpus_cache) or ".", exist_ok=True)
                with open(corpus_cache, "w", encoding="utf-8") as f:
                    json.dump({
                        "meta": {"fingerprint": fp, "k1": bm25_k1, "b": bm25_b},
                        "bug_id_list": bug_id_list,
                        "tokenized_corpus": tokenized_corpus,
                    }, f, ensure_ascii=False)

        self.bm25 = BM25Okapi(tokenized_corpus, k1=bm25_k1, b=bm25_b)
        self.bug_id_list = bug_id_list

        # Dense 인코더 (선택적)
        self.encoder: Optional[DenseEncoder] = None
        self.dense_corpus: Optional[np.ndarray] = None
        self.id2row: Optional[Dict[str, int]] = None
        if dense_model_path and os.path.exists(dense_model_path):
            try:
                self.encoder = DenseEncoder(dense_model_path, device=dense_device, batch_size=dense_batch)
                self._build_dense_corpus(dense_cache)
            except Exception as e:
                warnings.warn(f"Dense 인코더 초기화 실패: {e}")

    def _build_dense_corpus(self, cache_path: Optional[str]) -> None:
        assert self.encoder is not None
        if cache_path and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                ids = data["ids"].tolist()
                if ids == self.bug_id_list:
                    self.dense_corpus = data["embeddings"].astype(np.float32)
                    self.id2row = {str(bid): idx for idx, bid in enumerate(ids)}
                    return
            except Exception:
                pass

        docs = [self._doc_text(self.all_bugs.get(k, {})) for k in self.bug_id_list]
        self.dense_corpus = self.encoder.encode(docs, is_query=False).astype(np.float32)
        self.id2row = {str(bid): idx for idx, bid in enumerate(self.bug_id_list)}
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            np.savez_compressed(
                cache_path,
                embeddings=self.dense_corpus,
                ids=np.array(self.bug_id_list, dtype=object),
                meta=np.array(json.dumps({"model": self.encoder.model_name}), dtype=object),
            )

    @staticmethod
    def _doc_text(entry: Dict[str, Any]) -> str:
        parts = [entry.get("buggy_line_content") or ""]
        file_info = entry.get("file") or {}
        if isinstance(file_info, dict):
            parts.append(str(file_info.get("file_name") or ""))
        susp = entry.get("suspicious_nodes_topk") or []
        reasons = [
            node.get("reason", "")
            for node in susp[:3] if isinstance(node, dict) and node.get("reason")
        ]
        if reasons:
            parts.append(" | ".join(reasons))
        return " \n ".join(p for p in parts if p)

    @staticmethod
    def _query_text(func_obj: Dict[str, Any]) -> str:
        parts = [func_obj.get("buggy_line_content") or ""]
        susp = func_obj.get("suspicious_nodes_topk") or []
        reasons = [
            node.get("reason", "")
            for node in susp[:5] if isinstance(node, dict) and node.get("reason")
        ]
        if reasons:
            parts.append(" | ".join(reasons))
        return " \n ".join(p for p in parts if p)

    def query(
        self,
        func_obj: Dict[str, Any],
        current_bug_id: str,
        topk: int = 5,
        expand_query: bool = True,
        bonus_same_project: float = 0.10,
        bonus_same_ext: float = 0.05,
        bonus_token_jaccard: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """단일 버그에 대해 유사 예제 top-k를 반환한다."""
        # 쿼리 토큰 구성
        query_text = func_obj.get("buggy_line_content") or ""
        for node in (func_obj.get("suspicious_nodes_topk") or [])[:1]:
            if isinstance(node, dict) and node.get("code"):
                query_text = query_text or node["code"]
        base_tokens = tokenize(query_text)
        if expand_query:
            base_tokens = expand_query_tokens(base_tokens, func_obj)

        if not base_tokens:
            return []

        raw_scores = self.bm25.get_scores(base_tokens).tolist()
        M = max(topk, self.stage_topm)
        idxs = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:M]
        raw_sel = [raw_scores[i] for i in idxs]
        raw_norm = _normalize_scores(raw_sel)

        base_list: List[Tuple[str, Dict[str, Any], float]] = [
            (self.bug_id_list[idx], self.all_bugs.get(self.bug_id_list[idx], {}), raw_norm[ri])
            for ri, idx in enumerate(idxs)
        ]

        # Dense 결합
        if self.encoder is not None and self.dense_corpus is not None and self.id2row is not None:
            q_emb = self.encoder.encode([self._query_text(func_obj)], is_query=True)
            sims = []
            for idx in idxs:
                bid = self.bug_id_list[idx]
                row = self.id2row.get(bid)
                if row is not None:
                    sim = float(np.dot(q_emb[0:1], self.dense_corpus[row:row+1].T)[0][0])
                else:
                    cand_text = self._doc_text(self.all_bugs.get(bid, {}))
                    v = self.encoder.encode([cand_text], is_query=False)[0]
                    sim = float(np.dot(q_emb[0], v))
                sims.append(sim)
            sim_norm = _normalize_scores(sims)
            base_list = [
                (bid, info, (1.0 - self.dense_weight) * bm + self.dense_weight * sim_norm[ri])
                for ri, (bid, info, bm) in enumerate(base_list)
            ]
            base_list.sort(key=lambda x: x[2], reverse=True)

        # 규칙 기반 리랭크
        current_ids = {str(current_bug_id)}
        q_proj = (func_obj.get("project_name") or "").lower()
        q_file = (func_obj.get("file") or {}).get("file_name") or ""
        q_ext = _file_ext(q_file)
        q_line = func_obj.get("buggy_line_content") or ""

        seen_ids: set = set()
        results: List[Dict[str, Any]] = []
        for bid, info, score in base_list:
            if bid in current_ids or bid in seen_ids:
                continue
            seen_ids.add(bid)
            s = score
            if q_proj and (info.get("project_name") or "").lower() == q_proj:
                s += bonus_same_project
            fi = info.get("file") or {}
            cand_file = (fi.get("file_name") or "") if isinstance(fi, dict) else ""
            if q_ext and _file_ext(cand_file) == q_ext:
                s += bonus_same_ext
            s += bonus_token_jaccard * _token_jaccard(q_line, info.get("buggy_line_content") or "")
            results.append({
                "id": bid,
                "project_name": info.get("project_name"),
                "buggy_line_content": info.get("buggy_line_content"),
                "function": info.get("function"),
                "file_name": (fi.get("file_name") if isinstance(fi, dict) else None),
                "file_path": (fi.get("file_path") if isinstance(fi, dict) else None),
                "score_reranked": float(s),
            })
            if len(results) >= topk:
                break

        results.sort(key=lambda x: x["score_reranked"], reverse=True)
        return results[:topk]

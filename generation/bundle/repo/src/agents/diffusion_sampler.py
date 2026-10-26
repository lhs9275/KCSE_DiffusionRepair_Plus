"""디퓨전 모델 기반 편집 계획 생성.

IJCAI 버전(5.Diffusion.py)을 모듈화.
메인 API: load_diffusion_model(), DiffusionSampler, sample_plans_for_bug()
"""
from __future__ import annotations

import ast
import contextlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


OPS = ["KEEP", "Replace", "InsertBefore", "InsertAfter", "Delete", "GuardAdd"]
OP2ID: Dict[str, int] = {op: i for i, op in enumerate(OPS)}
ID2OP: Dict[int, str] = {i: op for op, i in OP2ID.items()}
DEFAULT_NODE_TYPE_COUNT = 56


# ---------------------------------------------------------------------------
# 모델 아키텍처
# ---------------------------------------------------------------------------

class EnhancedTransformerUNet(nn.Module):
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        hidden: int = 1024,
        nhead: int = 16,
        num_layers: int = 24,
    ):
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.emb_op = nn.Embedding(vocab_sizes["op"], hidden)
        self.emb_node_id = nn.Embedding(vocab_sizes["node_idx"], hidden)
        self.emb_node_type = nn.Embedding(vocab_sizes["node_type"], hidden)
        self.time_emb = nn.Embedding(64, hidden)
        self.input_proj = nn.Linear(hidden * 3, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=4096,
            activation="gelu", batch_first=True, dropout=0.1, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_proj = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.head_op = nn.Linear(hidden, vocab_sizes["op"])
        self.head_node_type = nn.Linear(hidden, vocab_sizes["node_type"])

    def forward(
        self,
        op_ids: torch.Tensor,
        node_ids: torch.Tensor,
        type_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(torch.cat([
            self.emb_op(op_ids), self.emb_node_id(node_ids), self.emb_node_type(type_ids)
        ], dim=-1))
        h = h + self.time_proj(self.time_emb(t).unsqueeze(1))
        h = self.ln(h)
        h = self.transformer_encoder(h)
        return self.head_op(h), self.head_node_type(h)


# ---------------------------------------------------------------------------
# 샘플러
# ---------------------------------------------------------------------------

class DiffusionSampler:
    def __init__(
        self,
        model: EnhancedTransformerUNet,
        max_nodes: int,
        T: int = 16,
        device: str = "cpu",
        node_guidance: float = 3.0,
        op_guidance: float = 1.5,
    ):
        self.model = model.to(device)
        self.max_nodes = max_nodes
        self.T = T
        self.device = device
        self.node_guidance = node_guidance
        self.op_guidance = op_guidance
        base = [0.70, 0.10, 0.04, 0.04, 0.01, 0.11]
        self.op_prior = torch.tensor(base, dtype=torch.float32, device=self.device)
        self.model.eval()

    def sample(
        self,
        length: int,
        node_types: List[int],
        cond_prior: Optional[List[float]] = None,
    ) -> Tuple[List[int], List[int]]:
        op = torch.randint(low=0, high=len(OPS), size=(1, length), device=self.device)

        if cond_prior is None:
            cond_prior_vec = torch.full(
                (self.max_nodes,), 1.0 / float(self.max_nodes),
                dtype=torch.float32, device=self.device,
            )
        else:
            cond_prior_vec = torch.tensor(cond_prior, dtype=torch.float32, device=self.device)
            if cond_prior_vec.numel() < self.max_nodes:
                cond_prior_vec = torch.cat([
                    cond_prior_vec,
                    torch.zeros(self.max_nodes - cond_prior_vec.numel(), device=self.device),
                ])
            elif cond_prior_vec.numel() > self.max_nodes:
                cond_prior_vec = cond_prior_vec[:self.max_nodes]
            cond_prior_vec = (cond_prior_vec + 1e-8) / (cond_prior_vec + 1e-8).sum()

        node_target = torch.multinomial(cond_prior_vec, length, replacement=True).unsqueeze(0)
        node_types_tensor = torch.tensor(node_types, dtype=torch.long, device=self.device)
        if node_types_tensor.numel() < self.max_nodes:
            node_types_tensor = torch.cat([
                node_types_tensor,
                torch.zeros(self.max_nodes - node_types_tensor.numel(), dtype=torch.long, device=self.device),
            ])
        else:
            node_types_tensor = node_types_tensor[:self.max_nodes]
        type_tensor = node_types_tensor[node_target]
        op_prior_log = torch.log(self.op_prior + 1e-8)

        use_bf16 = self.device.startswith("cuda") and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_ctx = torch.cuda.amp.autocast(dtype=dtype) if self.device.startswith("cuda") else contextlib.nullcontext()

        with amp_ctx:
            for step in range(self.T, 0, -1):
                t = torch.full((1,), step % 64, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits_op, _ = self.model(op, node_target, type_tensor, t)
                phase = 1.0 + 0.5 * (1.0 - step / self.T)
                logits_op = logits_op + (self.op_guidance * phase) * op_prior_log
                node_weights = cond_prior_vec[node_target]
                logits_op[:, :, 1:] = logits_op[:, :, 1:] + self.node_guidance * node_weights.unsqueeze(-1)
                op = torch.distributions.Categorical(logits=logits_op).sample()

        op_list = op.squeeze(0).cpu().tolist()
        node_list = node_target.squeeze(0).cpu().tolist()
        final_ops: List[int] = []
        final_nodes: List[int] = []
        for raw_op_idx, n_idx in zip(op_list, node_list):
            if raw_op_idx != OP2ID["KEEP"]:
                final_ops.append(raw_op_idx)
                final_nodes.append(int(n_idx))

        if not final_ops:
            final_ops = [OP2ID["GuardAdd"]]
            final_nodes = [int(torch.argmax(cond_prior_vec).item())]

        return final_ops, final_nodes


# ---------------------------------------------------------------------------
# 헬퍼 함수들
# ---------------------------------------------------------------------------

def _normalize(v: List[float]) -> List[float]:
    s = sum(v) + 1e-8
    return [x / s for x in v]


def build_node_prior_from_item(item: Dict[str, Any], max_nodes: int) -> List[float]:
    nodes = (
        item.get("suspicious_nodes_topk_reranked")
        or item.get("suspicious_nodes_topk")
        or []
    )
    pri = [0.0] * max_nodes
    for i, nd in enumerate(nodes):
        node_id = nd.get("node_id")
        if node_id is not None and 0 <= node_id < max_nodes:
            pri[node_id] += (len(nodes) - i) * 2.0
    if sum(pri) == 0:
        pri[0] = 1.0
    return _normalize(pri)


def get_node_types_from_code(
    code: str,
    max_nodes: int,
    type_map: Optional[Dict[str, int]] = None,
    node_type_vocab_size: int = DEFAULT_NODE_TYPE_COUNT,
) -> List[int]:
    types = [0] * max_nodes
    vocab_size = max(2, node_type_vocab_size)
    try:
        tree = ast.parse(code)
        for i, node in enumerate(ast.walk(tree)):
            if i >= max_nodes:
                break
            t_name = type(node).__name__
            if type_map:
                idx = int(type_map.get(t_name, 0))
            else:
                idx = hash(t_name) % (vocab_size - 1) + 1
            types[i] = idx % vocab_size
    except Exception:
        pass
    return types


def canonicalize_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set = set()
    uniq: List[Dict[str, Any]] = []
    for e in edits:
        op, node = e.get("op"), e.get("node")
        if op is None or node is None:
            continue
        key = (op, int(node))
        if key not in seen:
            seen.add(key)
            uniq.append({"op": op, "node": int(node)})
    return uniq


def score_edits(edits: List[Dict[str, Any]], node_prior: List[float]) -> float:
    if not edits:
        return -1e9
    op_w = {"Replace": 1.3, "GuardAdd": 1.2, "InsertBefore": 1.05, "InsertAfter": 1.05, "Delete": 0.9}
    score = sum(
        (node_prior[e.get("node", 0)] if 0 <= e.get("node", 0) < len(node_prior) else 0.0)
        * 10.0 * op_w.get(e.get("op", ""), 1.0)
        for e in edits
    )
    score += min(6, len(edits)) * 0.5
    if len(edits) > 8:
        score -= 2.0
    return score


# ---------------------------------------------------------------------------
# 모델 로딩 및 계획 생성 API
# ---------------------------------------------------------------------------

def load_diffusion_model(
    model_path: str,
    type_map_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[DiffusionSampler, Dict[str, int], int, int]:
    """체크포인트를 로드하고 (sampler, type_map, max_nodes, node_type_vocab_size)를 반환한다."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # type_map 로드
    type_map: Dict[str, int] = {}
    if type_map_path and Path(type_map_path).exists():
        try:
            raw = json.loads(Path(type_map_path).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                type_map = {str(k): int(v) for k, v in raw.items()}
                type_map.setdefault("Unknown", 0)
        except Exception:
            pass

    state = torch.load(model_path, map_location=device)

    def _shape(key: str) -> Optional[int]:
        return state[key].shape[0] if key in state else None

    node_type_vocab = (
        _shape("emb_node_type.weight")
        or (max(int(v) for v in type_map.values()) + 1 if type_map else None)
        or DEFAULT_NODE_TYPE_COUNT
    )
    max_nodes = _shape("emb_node_id.weight") or 1024

    vocab_sizes = {"op": len(OPS), "node_idx": max_nodes, "node_type": node_type_vocab}
    model = EnhancedTransformerUNet(vocab_sizes, hidden=1024, nhead=16, num_layers=24)
    model.load_state_dict(state, strict=True)

    sampler = DiffusionSampler(model, max_nodes, device=device)
    return sampler, type_map, max_nodes, node_type_vocab


def sample_plans_for_bug(
    sampler: DiffusionSampler,
    item: Dict[str, Any],
    type_map: Dict[str, int],
    max_nodes: int,
    node_type_vocab_size: int,
    k: int = 12,
    sample_multiplier: int = 3,
) -> List[Dict[str, Any]]:
    """단일 버그에 대해 편집 계획 top-k를 샘플링한다."""
    func = item.get("function", {}) or {}
    code = (func.get("function_before") or item.get("code") or item.get("buggy_code") or "").strip()
    if not code:
        return []

    node_types = get_node_types_from_code(code, max_nodes, type_map, node_type_vocab_size)
    node_prior = build_node_prior_from_item(item, max_nodes)

    num_samples = max(k * max(1, sample_multiplier), k + 2)
    cands: List[Dict[str, Any]] = []
    seen_edits: set = set()

    for _ in range(num_samples):
        op_ids, node_ids = sampler.sample(length=16, node_types=node_types, cond_prior=node_prior)
        raw_edits = [
            {"op": ID2OP.get(op_i, "KEEP"), "node": int(n_i)}
            for op_i, n_i in zip(op_ids, node_ids)
            if ID2OP.get(op_i, "KEEP") != "KEEP"
        ]
        edits = canonicalize_edits(raw_edits)
        sig = str(edits)
        if not edits or sig in seen_edits:
            continue
        seen_edits.add(sig)
        cands.append({
            "analysis": "Hybrid Diffusion Model",
            "edits": edits,
            "score": score_edits(edits, node_prior),
        })

    cands.sort(key=lambda x: x.get("score", -1e9), reverse=True)
    return cands[:k]

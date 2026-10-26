from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import runpy
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

OPS = ["KEEP", "Replace", "InsertBefore", "InsertAfter", "Delete", "GuardAdd"]
OP2ID = {name: index for index, name in enumerate(OPS)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_original_training(path: Path) -> dict[str, Any]:
    """Load definitions without executing the archive's guarded main()."""
    return runpy.run_path(str(path), run_name="planner_revision_archive")


def encode_record(record: dict[str, Any], type_map: dict[str, int], seq_len: int = 32,
                  max_nodes: int = 1024) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    ops = [0] * seq_len
    nodes = [0] * seq_len
    types = [type_map["Unknown"]] * seq_len
    for index, edit in enumerate(record.get("edits", [])[:seq_len]):
        name = edit.get("op")
        if name in OP2ID:
            ops[index] = OP2ID[name]
            node = edit.get("node")
            if isinstance(node, int) and not isinstance(node, bool):
                nodes[index] = min(max(0, node), max_nodes - 1)
            types[index] = type_map.get(edit.get("type", "Unknown"), 0)
    return tuple(ops), tuple(nodes), tuple(types)


def _ast_key(record: dict[str, Any]) -> str | None:
    code = record.get("buggy_code")
    if not isinstance(code, str):
        return None
    try:
        canonical = ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False)
    except (SyntaxError, ValueError, TypeError):
        canonical = "invalid:" + code
    return hashlib.sha256(canonical.encode()).hexdigest()


def _tensor_key(record: dict[str, Any], type_map: dict[str, int], seq_len: int,
                max_nodes: int) -> str:
    encoded = encode_record(record, type_map, seq_len, max_nodes)
    return hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest()


def build_group_split(records: list[dict[str, Any]], type_map: dict[str, int], *, seed: int = 42,
                      val_ratio: float = 0.1, seq_len: int = 32, max_nodes: int = 1024) -> dict[str, Any]:
    if len(records) < 2 or not 0 < val_ratio < 1:
        raise ValueError("group split needs at least two records and 0 < val_ratio < 1")
    parent = list(range(len(records)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    ast_keys, tensor_keys = [], []
    seen_ast: dict[str, int] = {}
    seen_tensor: dict[str, int] = {}
    for index, record in enumerate(records):
        ast_key = _ast_key(record)
        tensor_key = _tensor_key(record, type_map, seq_len, max_nodes)
        ast_keys.append(ast_key); tensor_keys.append(tensor_key)
        if ast_key is not None:
            if ast_key in seen_ast: union(index, seen_ast[ast_key])
            else: seen_ast[ast_key] = index
        if tensor_key in seen_tensor: union(index, seen_tensor[tensor_key])
        else: seen_tensor[tensor_key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)): groups[find(index)].append(index)
    components = list(groups.values())
    if len(components) < 2:
        raise ValueError(f"group constraints form one component containing all {len(records)} records")
    random.Random(seed).shuffle(components)
    target = len(records) * val_ratio
    val_groups: list[list[int]] = []
    remaining = len(records)
    for group in components:
        if remaining - len(group) == 0:
            continue
        current = sum(map(len, val_groups))
        if not val_groups or abs(current + len(group) - target) < abs(current - target):
            val_groups.append(group); remaining -= len(group)
    val = sorted(index for group in val_groups for index in group)
    train = sorted(set(range(len(records))) - set(val))
    if not train or not val:
        raise ValueError("group constraints cannot produce non-empty train and validation sets")
    train_set = set(train)

    def overlap(keys: list[str | None]) -> int:
        left = {key for i, key in enumerate(keys) if i in train_set and key is not None}
        right = {key for i, key in enumerate(keys) if i not in train_set and key is not None}
        return len(left & right)

    return {
        "algorithm": "union(ast-equivalent,identical-padded-op-node-type)", "seed": seed,
        "target_validation_ratio": val_ratio, "record_count": len(records),
        "group_count": len(components), "largest_group_records": max(map(len, components)),
        "train_indices": train, "validation_indices": val,
        "actual_train_records": len(train), "actual_validation_records": len(val),
        "actual_validation_ratio": len(val) / len(records),
        "cross_split_ast_keys": overlap(ast_keys), "cross_split_tensor_keys": overlap(tensor_keys),
    }


def rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoints(inference_path: Path, resume_path: Path, model: torch.nn.Module,
                     optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
                     training_state: dict[str, Any], provenance: dict[str, Any]) -> None:
    for path in (inference_path, resume_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    torch.save(model.state_dict(), inference_path)
    payload = {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
               "rng": rng_state(), "training_state": training_state, "provenance": provenance}
    torch.save(payload, resume_path)


def save_rolling_checkpoints(inference_path: Path, resume_path: Path, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any,
                             training_state: dict[str, Any], provenance: dict[str, Any],
                             *, save_best: bool) -> None:
    """Atomically retain one best inference file and one epoch-boundary resume file."""
    inference_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": 1, "resume_boundary": "completed_epoch", "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict() if scaler else None, "rng": rng_state(),
               "training_state": training_state, "provenance": provenance}
    temporary = resume_path.with_suffix(resume_path.suffix + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, resume_path)
    if save_best:
        temporary = inference_path.with_suffix(inference_path.suffix + ".tmp")
        torch.save(model.state_dict(), temporary); os.replace(temporary, inference_path)
        metadata = inference_path.with_name("best_model_metadata.json")
        temporary_metadata = metadata.with_suffix(".json.tmp")
        temporary_metadata.write_text(json.dumps({"best_epoch": training_state["best_epoch"],
                                                   "best_val_loss": training_state["best_val_loss"]}) + "\n")
        os.replace(temporary_metadata, metadata)


def load_resume_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                           scheduler: Any, scaler: Any, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None: payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload["scaler"] is not None: scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng"])
    return dict(payload["training_state"])


def scaler_step_succeeded(scale_before: float, scale_after: float) -> bool:
    """GradScaler lowers its scale exactly when it skipped the optimizer step."""
    return scale_after >= scale_before


class RevisionTrainer:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scaler: Any,
                 device: torch.device):
        self.model, self.optimizer, self.scaler, self.device = model, optimizer, scaler, device

    def backward_and_step(self, loss: torch.Tensor) -> None:
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler is None:
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
        else:
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer); self.scaler.update()

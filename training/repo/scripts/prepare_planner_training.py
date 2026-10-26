#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.planner_revision import build_group_split, load_original_training, sha256_file

ARCHIVE = Path("/workspace/users/2022810061/shared/kcse_replay_artifacts/KCSE2026_DiffusionRepair_8aed78a")
EXPECTED = {"trainer": "29d9c65f16ca9696c28779ddb5cf01007438082c8c3e96504f8133bf3e595af8",
            "data": "2caf9218100070d65f8bc6edf2f1081f45d6092d8dc09bf3d1488444c27ed0f3",
            "type_map": "a1ff78928e60a5d69bf1f0c7fc31b10959e49412d6008363c33f03d6426f3d77"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--val-ratio", type=float, default=.1)
    args = parser.parse_args()
    paths = {"trainer": args.archive / "1.train.py", "data": args.archive / "Results/train_data_llm_synthetic.json",
             "type_map": args.archive / "Results/type_map.json"}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if args.archive == ARCHIVE and hashes != EXPECTED: raise SystemExit(f"archive hash mismatch: {hashes}")
    records = json.loads(paths["data"].read_text()); type_map = json.loads(paths["type_map"].read_text())
    if type_map.get("Unknown") != 0 or sorted(type_map.values()) != list(range(len(type_map))):
        raise SystemExit("type map must be contiguous with Unknown=0")
    observed = {"Unknown": 0}
    for record in records:
        for edit in record.get("edits", []):
            name = edit.get("type", "Unknown")
            if name not in observed: observed[name] = len(observed)
    if observed != type_map: raise SystemExit("archived type map differs from data-derived insertion order")
    original = load_original_training(paths["trainer"])
    if (original["MAX_NODES"], original["BATCH_SIZE"], original["EPOCHS"], original["LR"]) != (1024, 32, 80, 2e-4):
        raise SystemExit("original settings differ from preserved configuration")
    manifest = build_group_split(records, type_map, seed=args.seed, val_ratio=args.val_ratio)
    torch.manual_seed(args.seed)
    train_size = len(records) - max(1, int(len(records) * args.val_ratio))
    old_train, old_val = torch.utils.data.random_split(range(len(records)), [train_size, len(records)-train_size])
    manifest.update({"schema_version": 1, "input_sha256": hashes, "type_map": type_map,
                     "original_random_split_reference": {"seed": args.seed, "train_indices": list(old_train.indices),
                                                         "validation_indices": list(old_val.indices),
                                                         "claim": "reproduced from archived code, not authenticated historical split"},
                     "preserved_config": {"hidden": 1024, "num_layers": 24, "nhead": 16, "batch_size": 32,
                                          "lr": 2e-4, "epochs": 80, "patience": 8, "seed": 42,
                                          "seq_len": 32, "max_nodes": 1024}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists(): raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("record_count", "group_count", "largest_group_records", "actual_train_records", "actual_validation_records", "actual_validation_ratio", "cross_split_ast_keys", "cross_split_tensor_keys")}, indent=2))


if __name__ == "__main__": main()

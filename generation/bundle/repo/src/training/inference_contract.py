"""Declared, machine-checked contract between planner training and inference.

Every value here was measured from live code by
`scripts/audit_train_inference_contract.py`, not copied from a write-up.

The contract exists so that:

* the settings a pilot or full run uses are fixed and quotable;
* the known train/inference mismatches are *declared* rather than implied to be
  absent — `strict state_dict` loading only proves the shapes line up;
* nobody can silently close a mismatch (which would be a method change needing
  its own comparison) or silently widen one.

`status` values:
  matched              — training and inference agree.
  mismatch_declared    — they disagree; kept as-is, must be stated in the paper.
  corrected            — changed in this revision; `correction` says how.
"""
from __future__ import annotations

from typing import Any, Dict, List

CONTRACT_VERSION = "planner_train_inference_v1"

# Architecture and optimiser settings that a pilot or full run must preserve.
PRESERVED_TRAINING_CONFIG: Dict[str, Any] = {
    "hidden": 1024, "nhead": 16, "num_layers": 24, "dim_feedforward": 4096,
    "norm_first": True, "dropout": 0.1, "activation": "gelu",
    "time_embedding_size": 64, "max_nodes": 1024, "seq_len": 32,
    "batch_size": 32, "lr": 2e-4, "epochs": 80, "patience": 8, "seed": 42,
    "optimizer": "AdamW", "weight_decay": 0.01, "eps": 1e-8,
    "scheduler": "CosineAnnealingWarmRestarts", "T_0": 10, "T_mult": 2,
    "keep_class_weight": 0.1, "edit_class_weight": 3.0, "edit_term_multiplier": 5.0,
    "label_smoothing": 0.1, "gradient_clip": 1.0, "amp": "GradScaler + autocast",
}

PRESERVED_INFERENCE_CONFIG: Dict[str, Any] = {
    "T": 16, "sample_length": 16, "max_nodes": 1024,
    "op_prior": [0.70, 0.10, 0.04, 0.04, 0.01, 0.11],
    "op_guidance": 1.5, "node_guidance": 3.0,
    "phase": "1 + 0.5*(1 - step/T)",
    "samples_per_bug": 36, "plans_kept": 12, "edits_shown_in_prompt": 6,
    "plan_ranking": "score_edits heuristic",
}

ITEMS: List[Dict[str, Any]] = [
    {
        "item": "op vocabulary",
        "training": "KEEP, Replace, InsertBefore, InsertAfter, Delete, GuardAdd (ids 0-5)",
        "inference": "identical list and ids",
        "status": "matched",
    },
    {
        "item": "node type vocabulary",
        "training": "type_map built from the synthetic edit labels, 56 entries, Unknown=0",
        "inference": "same type_map applied to type(node).__name__",
        "status": "corrected",
        "correction": "an absent or empty type map used hash(name), which is "
                      "PYTHONHASHSEED-dependent and silently non-reproducible; the "
                      "sampler now refuses to run instead of guessing ids.",
    },
    {
        "item": "sequence length",
        "training": 32,
        "inference": 16,
        "status": "mismatch_declared",
        "note": "the encoder has no positional embedding, so this is a distribution "
                "shift in how many KEEP slots the model sees, not a shape error.",
    },
    {
        "item": "time index",
        "training": "one uniform draw per example from randint(0, 64)",
        "inference": "deterministic 16,15,...,1 (step % 64)",
        "status": "mismatch_declared",
        "note": "inference visits only 16 of the 64 trained time values.",
    },
    {
        "item": "corruption level vs. declared time",
        "training": "min(max(int(L*(t/64)**2), 2), int(L*0.8)); at L=32 this is "
                    "2 slots for t<=16, 8 at t=32, 18 at t=48, 25 at t=63",
        "inference": "the first step declares t=16 but every one of the 16 op slots "
                     "is uniformly random",
        "status": "mismatch_declared",
        "note": "t=16 means ~6% corrupted slots in training and 100% at inference; "
                "no schedule maps the 16 inference iterations onto the training time "
                "axis. Closing this is a method change, not an error fix.",
    },
    {
        "item": "op input",
        "training": "ground truth with a subset of slots resampled uniformly",
        "inference": "uniform random at step T, then Categorical(model logits + guidance)",
        "status": "mismatch_declared",
    },
    {
        "item": "node input",
        "training": "ground-truth edit target ids, never corrupted; padding slots are 0",
        "inference": "multinomial draw from the AST prior with replacement, then frozen "
                     "for all T steps",
        "status": "mismatch_declared",
        "note": "node positions are chosen by the prior, not predicted by the model.",
    },
    {
        "item": "type input",
        "training": "corrupted on the same slots as op",
        "inference": "real AST types of the drawn nodes, never corrupted",
        "status": "mismatch_declared",
    },
    {
        "item": "padding / valid target mask",
        "training": "KEEP op, node 0, Unknown type after the last edit",
        "inference": "prior zeroed outside source-addressable node ids "
                     "(source_node_map), smoothing applied only to valid targets",
        "status": "mismatch_declared",
        "note": "training never saw a masked-target distribution.",
    },
    {
        "item": "guidance",
        "training": "none",
        "inference": "op prior log added with weight 1.5*phase and node prior with "
                     "weight 3.0 at every step",
        "status": "mismatch_declared",
        "note": "an always-KEEP mock model still emits a GuardAdd plan, so the "
                "presence of a plan is not evidence of a learned edit decision.",
    },
    {
        "item": "type head",
        "training": "supervised with cross-entropy",
        "inference": "output discarded",
        "status": "mismatch_declared",
    },
    {
        "item": "all-KEEP fallback",
        "training": "no equivalent",
        "inference": "forces GuardAdd at argmax(prior)",
        "status": "mismatch_declared",
    },
    {
        "item": "plan selection",
        "training": "no equivalent",
        "inference": "36 samples, dedupe, heuristic score_edits, keep 12; "
                     "the prompt shows plan 0's first 6 edits",
        "status": "matched",
        "note": "score_edits is a heuristic over the AST prior, not a model likelihood.",
    },
]


def declared(item: str) -> Dict[str, Any]:
    for entry in ITEMS:
        if entry["item"] == item:
            return entry
    raise KeyError(item)


def mismatches() -> List[str]:
    return [entry["item"] for entry in ITEMS if entry["status"] == "mismatch_declared"]


def verify_audit_report(report: Dict[str, Any]) -> List[str]:
    """Compare a fresh audit report against this declaration; return deviations."""
    problems: List[str] = []
    training, inference = report["training"], report["inference"]

    if training["sequence_length"] != declared("sequence length")["training"]:
        problems.append("training sequence length drifted")
    if inference["sequence_length"] != declared("sequence length")["inference"]:
        problems.append("inference sequence length drifted")
    if inference["time_index"]["T"] != PRESERVED_INFERENCE_CONFIG["T"]:
        problems.append("inference T drifted")
    if inference["time_index"]["observed_sequence_first_sample"] != list(range(16, 0, -1)):
        problems.append("inference time sequence drifted")
    if not training["time_index"]["per_example"]:
        problems.append("training no longer draws a time value per example")
    if training["op_input"]["fully_random"]:
        problems.append("training op input became fully random")
    if not inference["op_input"]["fully_random"]:
        problems.append("inference op input is no longer fully random at the first step; "
                        "this closes a declared mismatch and is a method change")
    if not training["node_input"]["identical_to_labels_every_call"]:
        problems.append("training node input is no longer the uncorrupted ground truth")
    if not inference["node_input"]["all_values_source_addressable"]:
        problems.append("inference sampled a node with no source span")
    if not inference["node_input"]["constant_within_each_sample"]:
        problems.append("inference node targets are no longer frozen across steps")
    if inference["type_input"]["type_head_output_consumed"]:
        problems.append("inference started consuming the type head")
    if not training["corruption"]["observed_changed_slots_never_exceed_formula"]:
        problems.append("training corrupted more slots than the schedule allows")
    for key, expected in (("op_guidance", 1.5), ("node_guidance", 3.0)):
        if inference["guidance"][key] != expected:
            problems.append(f"inference {key} drifted")
    observed_prior = [round(value, 4) for value in inference["guidance"]["op_prior"]]
    if observed_prior != PRESERVED_INFERENCE_CONFIG["op_prior"]:
        problems.append("inference op prior drifted")
    if inference["plan_selection"]["num_samples"] != PRESERVED_INFERENCE_CONFIG["samples_per_bug"]:
        problems.append("plan sample count drifted")
    if inference["plan_selection"]["kept"] != PRESERVED_INFERENCE_CONFIG["plans_kept"]:
        problems.append("kept plan count drifted")
    if inference["prompt_transfer"]["edits_shown"] != PRESERVED_INFERENCE_CONFIG["edits_shown_in_prompt"]:
        problems.append("prompt edit budget drifted")
    return problems

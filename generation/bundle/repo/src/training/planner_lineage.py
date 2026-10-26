"""Fork an immutable parent planner checkpoint into a separate child run.

Design: docs/revision/2026-09-08-planner-child-run-design.md

The parent is read and never written. Only the execution code hashes and the
execution environment/limits may differ between parent and child; anything that
constitutes the training semantics — architecture, data split, optimiser, loss,
noise and scheduler settings — must match, or the fork is refused.

There is deliberately no blanket override: no `ignore_provenance`, no
`strict=False` path, and no way to skip hashing the parent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

LINEAGE_SCHEMA = "planner_child_lineage_v1"

# This module is part of the child's execution code: it decides what is restored
# from the parent and how the child is stored, so its hash belongs in the child
# identity and in the pre-submit freeze check.
LINEAGE_MODULE = "src/training/planner_lineage.py"

# Identity sections that must agree for a fork to be a continuation of the same
# training run rather than a different experiment wearing its weights.
SEMANTIC_IDENTITY_FIELDS = ("config", "manifest_sha256")

# Sections that may legitimately differ, recorded in the lineage either way.
EXECUTION_IDENTITY_FIELDS = ("trainer_sha256", "utility_sha256", "lineage_sha256", "execution")


class ForkRejected(ValueError):
    """The parent cannot be continued into this child run."""


@dataclass(frozen=True)
class ParentCheckpoint:
    path: Path
    sha256: str
    identity: dict
    training_state: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parent_checkpoint(path: Path) -> ParentCheckpoint:
    """Read a parent checkpoint. Opens it read-only and writes nothing."""
    import torch

    path = Path(path)
    if not path.is_file():
        raise ForkRejected(f"parent checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # corrupt or unreadable: refuse, never guess
        raise ForkRejected(f"parent checkpoint could not be read: {path}") from error
    if not isinstance(payload, dict):
        raise ForkRejected("parent checkpoint is not a checkpoint payload")
    for key in ("model", "provenance", "training_state"):
        if key not in payload:
            raise ForkRejected(f"parent checkpoint is missing {key!r}")
    identity = payload["provenance"]
    state = payload["training_state"]
    if not isinstance(identity, dict) or not isinstance(state, dict):
        raise ForkRejected("parent checkpoint provenance or training state is malformed")
    for key in ("epoch", "global_step"):
        if not isinstance(state.get(key), int):
            raise ForkRejected(f"parent training state is missing an integer {key!r}")
    return ParentCheckpoint(path=path.resolve(), sha256=sha256_file(path),
                            identity=identity, training_state=state)


def compare_parent_identity(parent_identity: Mapping[str, Any],
                            child_identity: Mapping[str, Any]) -> None:
    """Raise ForkRejected unless the child continues the parent's training semantics."""
    for field in SEMANTIC_IDENTITY_FIELDS:
        if field not in parent_identity:
            raise ForkRejected(f"parent identity is missing {field!r}")
        if field not in child_identity:
            raise ForkRejected(f"child identity is missing {field!r}")
        if parent_identity[field] != child_identity[field]:
            raise ForkRejected(
                f"{field} differs between parent and child; a fork continues the same "
                "training setup, it does not migrate weights into a different one"
            )


def build_child_identity(child_identity: Mapping[str, Any],
                         parent: ParentCheckpoint) -> dict:
    """Child identity plus the parent lineage it is derived from.

    The free-text lineage note is deliberately NOT part of this identity: the run
    directory is derived from the identity hash, so letting a note change it would
    mean a reworded note silently forks a second run instead of refusing to mix
    with the existing one. The note is recorded in the run's provenance file.
    """
    compare_parent_identity(parent.identity, child_identity)
    state = parent.training_state
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "parent_checkpoint": str(parent.path),
        "parent_checkpoint_sha256": parent.sha256,
        "parent_identity": json.loads(json.dumps(parent.identity, sort_keys=True)),
        "parent_completed_epoch": state["epoch"],
        "parent_global_step": state["global_step"],
        "parent_attempted_batches": state.get("attempted_batches"),
        "parent_overflow_skips": state.get("overflow_skips"),
        "parent_best_epoch": state.get("best_epoch"),
        "parent_best_val_loss": state.get("best_val_loss"),
    }
    return dict(child_identity) | {"lineage": lineage}


def child_best_metadata(parent_best: Mapping[str, Any], *, child_best_epoch: Optional[int],
                        child_best_val_loss: float) -> dict:
    """Attribute the best weights to whichever run actually produced them.

    A child that has not beaten the parent has no weight file of its own; saying
    so is not an error state, and the parent's metadata is never rewritten.
    """
    parent_epoch = parent_best.get("best_epoch")
    parent_loss = parent_best.get("best_val_loss", float("inf"))
    improved = child_best_epoch is not None and child_best_val_loss < parent_loss
    return {
        "best_owner": "child" if improved else "parent",
        "best_epoch": child_best_epoch if improved else parent_epoch,
        "best_val_loss": child_best_val_loss if improved else parent_loss,
        "parent_best_epoch": parent_epoch,
        "parent_best_val_loss": parent_loss,
    }

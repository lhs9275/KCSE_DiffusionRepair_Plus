"""Identity contract for the revision_rerun_v2 primary planner."""
from __future__ import annotations

from typing import Any, Mapping

FRESH_LINEAGE_SCHEMA = "planner_fresh_lineage_v2"
FRESH_LINEAGE_MODULE = "src/training/planner_fresh_lineage.py"
INITIAL_COUNTERS = {"epoch": 0, "global_step": 0, "attempted_batches": 0,
                    "overflow_skips": 0}


def build_fresh_identity(identity: Mapping[str, Any]) -> dict:
    if "lineage" in identity:
        raise ValueError("fresh identity must not inherit a lineage")
    return dict(identity) | {"lineage": {
        "schema": FRESH_LINEAGE_SCHEMA,
        "kind": "fresh",
        "parent_checkpoint": None,
        "initial_state": dict(INITIAL_COUNTERS),
    }}


def validate_fresh_start(identity: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    lineage = identity.get("lineage")
    if not isinstance(lineage, dict) or lineage != build_fresh_identity(
            {key: value for key, value in identity.items() if key != "lineage"})["lineage"]:
        raise ValueError("v2 primary requires explicit parent-null fresh lineage")
    if any(state.get(key) != value for key, value in INITIAL_COUNTERS.items()):
        raise ValueError("v2 primary must start with zero training counters")
    if state.get("parent_baseline") is not None:
        raise ValueError("v2 primary cannot inherit parent counters")

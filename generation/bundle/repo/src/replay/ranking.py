"""Deterministic views over one raw candidate generation pool.

Both replay conditions are cut from the *same* raw pool at the *same* evaluation
budget:

  ranked   — `rank_score_topk_v1`:        top-`budget` by score, ties by generation order
  unranked — `generation_order_firstk_v1`: first `budget` in generation order

The two views are therefore deliberately *not* the same multiset whenever the
pool is larger than the budget: that difference is the ranker's selection
contribution, which is what the `no_ranking` ablation is meant to isolate. What
must be identical, and is asserted here, is the pool and the budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

POOL_POLICY = "shared_raw_pool_v1"
RANKED_SELECTION = "rank_score_topk_v1"
UNRANKED_SELECTION = "generation_order_firstk_v1"


@dataclass(frozen=True)
class CandidateViews:
    ranked: List[Dict[str, Any]]
    unranked: List[Dict[str, Any]]
    raw_pool_digest: str = ""
    raw_pool_size: int = 0
    budget: Optional[int] = None
    pool_policy: str = POOL_POLICY


def candidate_multiset_digest(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Hash candidate values with multiplicity while ignoring list order."""
    serialized = sorted(
        json.dumps(candidate, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for candidate in candidates
    )
    payload = "\n".join(serialized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def annotate_pool(raw_candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Address every raw candidate by its generation index; views cite these."""
    return [dict(candidate, generation_index=index)
            for index, candidate in enumerate(raw_candidates)]


def select_view(
    annotated_pool: Sequence[Mapping[str, Any]],
    policy: str,
    budget: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Cut one view out of the shared pool under a named selection policy."""
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("candidate budget must be a positive integer")
    limit = len(annotated_pool) if budget is None else min(budget, len(annotated_pool))
    if policy == UNRANKED_SELECTION:
        ordered: List[Mapping[str, Any]] = list(annotated_pool)
    elif policy == RANKED_SELECTION:
        ordered = sorted(
            annotated_pool,
            key=lambda item: (-float(item.get("score") or 0), int(item["generation_index"])),
        )
    else:
        raise ValueError(f"unknown candidate selection policy: {policy}")
    return copy.deepcopy(ordered[:limit])


def build_candidate_views(
    raw_candidates: Sequence[Mapping[str, Any]],
    budget: Optional[int] = None,
) -> CandidateViews:
    annotated = annotate_pool(raw_candidates)
    pool_digest = candidate_multiset_digest(annotated)
    limit = len(annotated) if budget is None else min(budget, len(annotated))

    unranked = select_view(annotated, UNRANKED_SELECTION, budget)
    ranked = select_view(annotated, RANKED_SELECTION, budget)

    for name, view in (("ranked", ranked), ("unranked", unranked)):
        indices = [int(item["generation_index"]) for item in view]
        if len(set(indices)) != len(indices):
            raise AssertionError(f"{name} view repeats a raw pool candidate")
        if any(not 0 <= index < len(annotated) for index in indices):
            raise AssertionError(f"{name} view escaped the raw pool")
        if len(view) != limit:
            raise AssertionError(f"{name} view did not spend the shared budget")
    return CandidateViews(
        ranked=ranked, unranked=unranked, raw_pool_digest=pool_digest,
        raw_pool_size=len(annotated), budget=budget,
    )

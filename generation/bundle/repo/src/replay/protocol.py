"""Immutable contract for the corrected replay version one protocol."""

from __future__ import annotations

from typing import Any, Mapping


PROTOCOL_NAME = "corrected_replay_v1"
METADATA_SHA256 = "3028d78586def91bebca1d6dbd376db2384b1b70bf1f465e911ae80186e1a4a7"
MODEL = {
    "id": "bigcode/starcoder2-7b",
    "revision": "bb9afde76d7945da5745592525db122d4d729eb1",
    "path": (
        "/workspace/models/huggingface/bigcode/starcoder2-7b/"
        "bb9afde76d7945da5745592525db122d4d729eb1"
    ),
}
GENERATION = {
    "target": 10,
    "max_attempts": 100,
    "batch_size": 8,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "max_new_tokens": 640,
}
CONDITIONS = {
    "starcoder2_full_nocommit": {
        "source_config": "starcoder2_full",
        "expected_retrieval_count": 5,
        "derive_no_ranking": True,
    },
    "starcoder2_no_retrieval_nocommit": {
        "source_config": "starcoder2_no_retrieval",
        "expected_retrieval_count": 0,
        "derive_no_ranking": False,
    },
}
SEEDS = {
    "starcoder2_full_nocommit": (42, 43, 44, 45, 46, 47, 123, 124, 125),
    "starcoder2_no_retrieval_nocommit": (42, 43, 44),
}


# The ablation extension answers a question version one never asked: does the
# proposed machinery contribute at all? It reuses version one's metadata, model
# and generation settings verbatim so its numbers are comparable, and differs
# only in which prompt context blocks are composed. It is a separate protocol
# name because version one's condition set is frozen by contract.
ABLATION_PROTOCOL_NAME = "corrected_replay_v1_ablation"
ABLATION_CONDITIONS = {
    "starcoder2_bare_nocommit": {
        "source_config": "starcoder2_full",
        "expected_retrieval_count": 5,
        "derive_no_ranking": False,
        "prompt_flags": {
            "use_ast": False,
            "use_retrieval": False,
            "use_diffusion": False,
        },
    },
    "starcoder2_no_diffusion_nocommit": {
        "source_config": "starcoder2_full",
        "expected_retrieval_count": 5,
        "derive_no_ranking": False,
        "prompt_flags": {
            "use_ast": True,
            "use_retrieval": True,
            "use_diffusion": False,
        },
    },
}
ABLATION_SEEDS = {
    "starcoder2_bare_nocommit": (42, 43, 44),
    "starcoder2_no_diffusion_nocommit": (42, 43, 44),
}


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"corrected_replay_v1 protocol drift in {label}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_v1_protocol(
    config: Mapping[str, Any], condition: str, seed: int
) -> None:
    """Reject any setting that would be mislabeled as protocol version one."""
    _require_equal("protocol", config.get("protocol"), PROTOCOL_NAME)
    _require_equal("metadata SHA-256", config.get("metadata_sha256"), METADATA_SHA256)
    _require_equal("model", config.get("model"), MODEL)
    _require_equal("generation", config.get("generation"), GENERATION)
    _require_equal("conditions", config.get("conditions"), CONDITIONS)
    if condition not in SEEDS:
        raise ValueError(f"unknown corrected_replay_v1 condition: {condition}")
    if int(seed) not in SEEDS[condition]:
        raise ValueError(
            f"seed {seed} is not allowed for corrected_replay_v1 condition {condition}"
        )


def validate_ablation_protocol(
    config: Mapping[str, Any], condition: str, seed: int
) -> None:
    """Reject drift in the ablation extension, including its prompt flags."""
    _require_equal("protocol", config.get("protocol"), ABLATION_PROTOCOL_NAME)
    _require_equal("metadata SHA-256", config.get("metadata_sha256"), METADATA_SHA256)
    _require_equal("model", config.get("model"), MODEL)
    _require_equal("generation", config.get("generation"), GENERATION)
    _require_equal("conditions", config.get("conditions"), ABLATION_CONDITIONS)
    if condition not in ABLATION_SEEDS:
        raise ValueError(f"unknown ablation condition: {condition}")
    if int(seed) not in ABLATION_SEEDS[condition]:
        raise ValueError(
            f"seed {seed} is not allowed for ablation condition {condition}"
        )


def validate_protocol(
    config: Mapping[str, Any], condition: str, seed: int
) -> None:
    """Dispatch to the validator for the protocol the config declares."""
    name = config.get("protocol")
    if name == PROTOCOL_NAME:
        return validate_v1_protocol(config, condition, seed)
    if name == ABLATION_PROTOCOL_NAME:
        return validate_ablation_protocol(config, condition, seed)
    raise ValueError(f"unknown protocol: {name!r}")

"""Validate first, then replay patch generation from frozen intermediates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from src.agents.prompt_generator import build_generation_prompt, create_prompt, validate_generation_prompt
from src.replay.frozen_inputs import FrozenReplayInput, load_frozen_inputs
from src.replay.input_contract import validate_base_contract
from src.replay.leakage import audit_prompt
from src.replay.model_input import validate_model_input_provenance
from src.replay.protocol import validate_protocol, validate_v1_protocol
from src.replay.provenance import (
    OUTPUT_DIRECTORIES,
    add_candidate_integrity,
    build_manifest,
    prepare_output_root,
    result_record_is_valid,
)
from src.replay.ranking import (POOL_POLICY, annotate_pool, build_candidate_views,
                                candidate_multiset_digest, select_view)
from src.replay.runtime import validate_gpu_runtime
from src.utils.ast_hints import validate_ast_hints
from src.utils.repair_plans import MAX_PROMPT_EDITS, PLAN_BINDING_POLICY, validate_repair_edits


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidationReport:
    total_bugs: int
    valid_bugs: int
    hard_leakage_matches: int
    diagnostic_added_line_matches: int
    run_name: str
    prompt_dir: Path
    manifest_path: Path


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _code_files_sha256() -> dict[str, str]:
    paths = [*REPO_ROOT.joinpath("src").rglob("*.py"), REPO_ROOT / "scripts/run_frozen_replay.py"]
    return {str(path.relative_to(REPO_ROOT)): _sha256_file(path) for path in sorted(paths)}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _valid_git_object_id(value: str) -> Optional[str]:
    value = value.strip().lower()
    if len(value) == 40 and all(char in "0123456789abcdef" for char in value):
        return value
    return None


def _git_directory(path: Path) -> Optional[Path]:
    marker = Path(path) / ".git"
    if marker.is_dir():
        return marker.resolve()
    if not marker.is_file():
        return None
    try:
        line = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir: "
    if not line.startswith(prefix):
        return None
    git_dir = Path(line[len(prefix) :])
    if not git_dir.is_absolute():
        git_dir = marker.parent / git_dir
    return git_dir.resolve()


def _common_git_directory(git_dir: Path) -> Path:
    marker = Path(git_dir) / "commondir"
    if not marker.is_file():
        return Path(git_dir)
    try:
        common = Path(marker.read_text(encoding="utf-8").strip())
    except OSError:
        return Path(git_dir)
    if not common.is_absolute():
        common = Path(git_dir) / common
    return common.resolve()


def _read_git_ref(git_dir: Path, common_dir: Path, ref: str) -> Optional[str]:
    if not ref.startswith("refs/") or ".." in Path(ref).parts:
        return None
    for root in (git_dir, common_dir):
        path = root / ref
        if path.is_file():
            try:
                value = _valid_git_object_id(path.read_text(encoding="utf-8"))
            except OSError:
                value = None
            if value is not None:
                return value
    packed = common_dir / "packed-refs"
    if packed.is_file():
        try:
            lines = packed.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if not line or line.startswith(("#", "^")):
                continue
            fields = line.split(" ", 1)
            if len(fields) == 2 and fields[1] == ref:
                return _valid_git_object_id(fields[0])
    return None


def _git_commit_from_metadata(path: Path) -> Optional[str]:
    git_dir = _git_directory(path)
    if git_dir is None:
        return None
    try:
        value = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "ref: "
    if value.startswith(prefix):
        return _read_git_ref(
            git_dir,
            _common_git_directory(git_dir),
            value[len(prefix) :],
        )
    return _valid_git_object_id(value)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, _canonical_json(value))


def _metadata_records(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _git_commit(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        value = _valid_git_object_id(result.stdout)
        if value is not None:
            return value
    return _git_commit_from_metadata(Path(path))


def _backup_emse_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        primary_root = Path(result.stdout.strip()).parent
    else:
        git_dir = _git_directory(REPO_ROOT)
        if git_dir is None:
            return None
        primary_root = _common_git_directory(git_dir).parent
    return _git_commit(primary_root.parent / "backup_EMSE")


def _ensure_output_layout(output_root: Path) -> None:
    for name in OUTPUT_DIRECTORIES:
        (output_root / name).mkdir(parents=True, exist_ok=True)


def _run_name(source_config: str, seed: int) -> str:
    return f"{source_config}_nocommit_seed{seed}"


def validate_condition(
    *,
    metadata_path: Path,
    source_dir: Path,
    source_config: str,
    seed: int,
    expected_retrieval_count: int,
    output_root: Path,
    run_name: Optional[str] = None,
    resume: bool = False,
    manifest_extra: Optional[Mapping[str, Any]] = None,
    prompt_flags: Optional[Mapping[str, bool]] = None,
) -> ValidationReport:
    """Validate all frozen records and prompts before writing any prompt."""
    metadata_path = Path(metadata_path)
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    run_name = run_name or _run_name(source_config, seed)
    metadata = _metadata_records(metadata_path)
    # Audit and persist the final generation input, including raw edit hints.
    flags = {
        "use_ast": True,
        "use_retrieval": expected_retrieval_count > 0,
        "use_diffusion": True,
    }
    flags.update({k: bool(v) for k, v in (prompt_flags or {}).items()})
    frozen = load_frozen_inputs(metadata_path, source_dir, source_config, seed,
                                expected_retrieval_count, **flags)

    validated: list[
        tuple[FrozenReplayInput, str, tuple[str, ...], Mapping[str, Any]]
    ] = []
    for item in frozen:
        # Frozen retrieval was computed using these AST explanations. Hiding
        # their display cannot repair an already contaminated retrieval query.
        if flags["use_ast"]:
            validate_ast_hints(item.prompt_input, item.prompt_input.get("suspicious_nodes_topk_reranked") or [])
        if flags['use_diffusion']:
            for plan in item.plans:
                if not isinstance(plan, dict) or not plan.get('edits'):
                    raise ValueError('repair plan: empty or malformed frozen plan')
                validate_repair_edits((item.prompt_input.get('function') or {}).get('function_before') or '', plan['edits'])
        prompt = create_prompt(
            bug_id=item.bug_id,
            bug_data=item.prompt_input,
            diffusion_candidates=item.plans,
            max_similar=expected_retrieval_count,
            use_ast=flags["use_ast"],
            use_retrieval=flags["use_retrieval"],
            use_diffusion=flags["use_diffusion"],
            include_commit_context=False,
        )
        edits = (item.plans[0].get("edits") or []) if flags["use_diffusion"] else []
        prompt = build_generation_prompt(
            (item.prompt_input.get("function") or {}).get("function_before") or "",
            edits, prompt,
        )
        validate_generation_prompt((item.prompt_input.get('function') or {}).get('function_before') or '', edits, prompt,
                                   source_metadata=item.prompt_input)
        retrieved = (item.prompt_input.get("bm25") or {}).get("top") or []
        audit = audit_prompt(prompt, metadata[item.bug_id], retrieved)
        if not flags["use_ast"] and not flags["use_retrieval"] and not flags["use_diffusion"]:
            condition_contract = validate_base_contract(
                prompt=prompt,
                edits=edits,
                plans=item.plans,
                generation_input=item.prompt_input,
            )
        else:
            condition_contract = {
                "policy": "non_base_condition_v1",
                "use_ast": flags["use_ast"],
                "use_retrieval": flags["use_retrieval"],
                "use_diffusion": flags["use_diffusion"],
            }
        validated.append(
            (item, prompt, audit.diagnostic_added_line_matches, condition_contract)
        )

    prompt_dir = output_root / "prompts" / run_name
    manifest_path = output_root / "manifests" / f"{run_name}.json"
    source_records = [
        {
            "bug_id": item.bug_id,
            "source_path": str(item.source_path.resolve()),
            "source_sha256": item.source_sha256,
            "prompt_sha256": _sha256_text(prompt),
            "diagnostic_added_line_matches": list(diagnostics),
            "condition_contract": dict(condition_contract),
        }
        for item, prompt, diagnostics, condition_contract in validated
    ]
    code_files = _code_files_sha256()
    manifest = build_manifest(
        protocol="corrected_replay_v1",
        run_name=run_name,
        seed=seed,
        source_config=source_config,
        expected_retrieval_count=expected_retrieval_count,
        metadata_path=metadata_path.resolve(),
        metadata_sha256=_sha256_file(metadata_path),
        metadata_git_commit=_git_commit(metadata_path.parent),
        code_git_commit=_git_commit(REPO_ROOT),
        code_files_sha256=code_files,
        implementation_revision="repair_20260908",
        backup_emse_git_commit=_backup_emse_commit(),
        source_records=source_records,
        prompt_policy={"include_commit_context": False, **flags,
                       'ast_off_policy': 'none_v1', 'plan_validation': 'source_addressable_v1',
                       'plan_binding': PLAN_BINDING_POLICY},
        **dict(manifest_extra or {}),
    )
    root_manifest = build_manifest(
        protocol="corrected_replay_v1",
        metadata_path=metadata_path.resolve(),
        metadata_sha256=_sha256_file(metadata_path),
        code_git_commit=_git_commit(REPO_ROOT),
        code_files_sha256=code_files,
        implementation_revision="repair_20260908",
        backup_emse_git_commit=_backup_emse_commit(),
        config_sha256=(manifest_extra or {}).get("config_sha256"),
        model=(manifest_extra or {}).get("model"),
        generation=(manifest_extra or {}).get("generation"),
    )
    root_has_content = output_root.exists() and any(output_root.iterdir())
    prepare_output_root(
        output_root,
        root_manifest,
        resume=root_has_content,
    )

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"validation manifest already exists: {manifest_path}")
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != manifest:
            raise ValueError("resume manifest does not match validated inputs")
    elif resume:
        raise FileNotFoundError(f"resume manifest missing: {manifest_path}")

    if prompt_dir.exists() and any(prompt_dir.iterdir()) and not resume:
        raise FileExistsError(f"prompt output is not empty: {prompt_dir}")
    _ensure_output_layout(output_root)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for item, prompt, _diagnostics, _condition_contract in validated:
        path = prompt_dir / f"{item.bug_id}.txt"
        if resume and path.exists():
            if _sha256_text(path.read_text(encoding="utf-8")) != _sha256_text(prompt):
                raise ValueError(f"resume prompt differs for bug {item.bug_id}")
            continue
        _write_text_atomic(path, prompt)
    if not manifest_path.exists():
        _write_json_atomic(manifest_path, manifest)

    return ValidationReport(
        total_bugs=len(frozen),
        valid_bugs=len(validated),
        hard_leakage_matches=0,
        diagnostic_added_line_matches=sum(len(row[2]) for row in validated),
        run_name=run_name,
        prompt_dir=prompt_dir,
        manifest_path=manifest_path,
    )


POOL_SCHEMA_VERSION = 3

# Fields `_candidate_record` fixes at generation time. Recovery re-derives these
# from the stored raw pool; everything else in a candidate record (test_passed,
# modified_lines, modified_ast_nodes) is filled in later by evaluation and must
# not be treated as generation evidence.
GENERATION_FIXED_CANDIDATE_FIELDS = (
    "patch_text", "ast_parse_ok", "diff_plan_used", "rank_score", "tokens_input",
    "tokens_output", "inference_time_sec", "generation_index",
    "generation_temperature", "generation_top_p", "plan_ok",
)


def _candidate_record(candidate: Mapping[str, Any], edits: list[dict[str, Any]]) -> dict[str, Any]:
    ast_result = candidate.get("ast") or {}
    return {
        "patch_text": candidate.get("code") or "",
        "ast_parse_ok": bool(ast_result.get("ok")),
        "modified_lines": [],
        "modified_ast_nodes": [],
        "diff_plan_used": edits[:MAX_PROMPT_EDITS],
        "test_passed": None,
        "rank_score": candidate.get("score"),
        "tokens_input": candidate.get("tokens_in"),
        "tokens_output": candidate.get("tokens_out"),
        "inference_time_sec": candidate.get("elapsed_sec"),
        "generation_index": candidate["generation_index"],
        "generation_temperature": candidate.get("temperature"),
        "generation_top_p": candidate.get("top_p"),
        "plan_ok": candidate.get("plan_ok"),
    }


def _expected_input_provenance(
    item: FrozenReplayInput,
    prompt: str,
    model: Mapping[str, Any],
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_path": str(item.source_path.resolve()),
        "source_sha256": item.source_sha256,
        "prompt_sha256": _sha256_text(prompt),
        "model_id": model["id"],
        "model_revision": model["revision"],
        "model_path": str(model["path"]),
        "stage_generation_seed": item.seed,
        "generation_settings": dict(generation),
        "implementation_sha256": _sha256_text(_canonical_json(_code_files_sha256())),
    }


def _recover_result_set(
    *,
    output_root: Path,
    run_name: str,
    pool_path: Path,
    paths_by_config: Mapping[str, Path],
    expected_input_provenance: Mapping[str, Any],
    expected_candidates: int,
) -> None:
    """Verify a durable pool, reconstruct missing views, and record file hashes."""
    output_root = Path(output_root)
    try:
        pool = json.loads(Path(pool_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid recovery pool: {pool_path}") from exc
    if pool.get("protocol") != "corrected_replay_v1":
        raise ValueError("recovery pool protocol mismatch")
    if pool.get("input_provenance") != expected_input_provenance:
        raise ValueError("recovery pool input provenance mismatch")
    model_input_provenance = pool.get("model_input_provenance")
    validate_model_input_provenance(model_input_provenance)
    expected_prompt_sha256 = expected_input_provenance.get("prompt_sha256")
    if (
        expected_prompt_sha256 is not None
        and model_input_provenance.get("user_prompt_sha256")
        != expected_prompt_sha256
    ):
        raise ValueError("recovery pool model input does not match its prompt")
    records = pool.get("records")
    if not isinstance(records, dict) or set(records) != set(paths_by_config):
        raise ValueError("recovery pool result views mismatch")

    # Views cut from one shared pool are deliberately different multisets, so the
    # integrity check is the pool, the budget and the candidate contents, not view
    # equality: recompute the stored raw pool and re-derive each view from its
    # declared selection policy, then bind every generation-fixed field back to
    # the raw candidate the view cites.
    if pool.get("pool_schema_version") != POOL_SCHEMA_VERSION:
        raise ValueError(
            f"recovery pool schema {pool.get('pool_schema_version')!r} is not "
            f"{POOL_SCHEMA_VERSION}; pools written before the shared-pool protocol "
            "cannot be verified and are not silently accepted"
        )
    if pool.get("candidate_pool_policy") != POOL_POLICY:
        raise ValueError("recovery pool declares an unexpected candidate pool policy")
    pool_budget = pool.get("candidate_budget")
    if type(pool_budget) is not int or pool_budget < 1:
        raise ValueError("recovery pool candidate budget is not a positive integer")
    if pool_budget != expected_candidates:
        raise ValueError(
            f"recovery pool declared candidate budget {pool_budget} but the protocol "
            f"budget is {expected_candidates}"
        )
    sent_edits = pool.get("sent_edits")
    if not isinstance(sent_edits, list):
        raise ValueError("recovery pool is missing the sent edit list")
    # An empty pool is a legitimate outcome (no candidate survived validation);
    # a missing or non-object pool is not.
    raw_candidates = pool.get("raw_candidates")
    if (not isinstance(raw_candidates, list)
            or any(not isinstance(candidate, dict) for candidate in raw_candidates)):
        raise ValueError("recovery pool raw candidate pool is missing or malformed")
    annotated = annotate_pool(
        [{key: value for key, value in candidate.items() if key != "generation_index"}
         for candidate in raw_candidates]
    )
    if annotated != raw_candidates:
        raise ValueError("recovery pool raw candidates are not generation-indexed in order")
    pool_digest = candidate_multiset_digest(annotated)
    if pool.get("raw_candidate_multiset_sha256") != pool_digest:
        raise ValueError("recovery pool raw pool digest does not match its raw candidates")

    declared_pools = {records[name].get("raw_candidate_multiset_sha256") for name in paths_by_config}
    declared_budgets = {records[name].get("candidate_budget") for name in paths_by_config}
    if len(declared_pools) != 1 or declared_pools != {pool_digest}:
        raise ValueError("recovery records were cut from different raw pools")
    if len(declared_budgets) != 1:
        raise ValueError("recovery records declare different candidate budgets")

    # Validate everything, including already-written views and the manifest, before
    # writing anything: a rejected pool must never leave a partial result set.
    existing_by_config: dict[str, Any] = {}
    for config_name, path in paths_by_config.items():
        record = records[config_name]
        if not isinstance(record, dict) or not result_record_is_valid(
            record,
            expected_input_provenance=expected_input_provenance,
            expected_config=config_name,
            expected_candidates=expected_candidates,
        ):
            raise ValueError(f"invalid recovery record for {config_name}")
        if record.get("candidate_pool_policy") != POOL_POLICY:
            raise ValueError(f"unexpected candidate pool policy for {config_name}")
        if record.get("model_input_provenance") != model_input_provenance:
            raise ValueError(
                f"model input provenance differs for recovery record {config_name}"
            )
        record_budget = record.get("candidate_budget")
        if type(record_budget) is not int or record_budget != expected_candidates:
            raise ValueError(
                f"recovery record for {config_name} declared candidate budget "
                f"{record_budget!r} but the protocol budget is {expected_candidates}"
            )
        if record.get("raw_candidate_pool_size") != len(annotated):
            raise ValueError(
                f"recovery record for {config_name} declares a raw candidate pool size "
                f"of {record.get('raw_candidate_pool_size')!r}, not {len(annotated)}"
            )
        try:
            expected_view = select_view(
                annotated, record.get("candidate_selection_policy"), record_budget,
            )
        except ValueError as exc:
            raise ValueError(f"{exc} for {config_name}") from exc
        actual = [candidate.get("generation_index") for candidate in record["candidates"]]
        if actual != [candidate["generation_index"] for candidate in expected_view]:
            raise ValueError(
                f"recovery record for {config_name} does not match its declared selection"
            )
        # Content binding: the served candidate must be the raw candidate it cites,
        # serialized the way generation serializes it. Re-sealing the record's own
        # hashes cannot substitute for agreeing with the raw pool.
        for served, raw in zip(record["candidates"], expected_view):
            rebuilt = _candidate_record(raw, sent_edits)
            for field in GENERATION_FIXED_CANDIDATE_FIELDS:
                if served.get(field) != rebuilt[field]:
                    raise ValueError(
                        f"recovery record for {config_name} candidate "
                        f"{raw['generation_index']} field {field!r} does not match "
                        "the raw candidate it cites"
                    )
        path = Path(path)
        if path.exists():
            try:
                existing_by_config[config_name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid existing result: {path}") from exc
            if existing_by_config[config_name] != record:
                raise ValueError(f"existing result differs from recovery pool: {path}")

    manifest_path = output_root / "manifests" / f"{run_name}.result-files.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("protocol") != "corrected_replay_v1"
            or manifest.get("run_name") != run_name
            or not isinstance(manifest.get("files"), dict)
        ):
            raise ValueError("result-file manifest is invalid")
    else:
        manifest = {
            "protocol": "corrected_replay_v1",
            "run_name": run_name,
            "files": {},
        }

    for config_name, path in paths_by_config.items():
        if config_name not in existing_by_config:
            _write_json_atomic(Path(path), records[config_name])

    files = [Path(pool_path), *(Path(path) for path in paths_by_config.values())]
    for path in files:
        relative = str(path.relative_to(output_root))
        digest = _sha256_file(path)
        recorded = manifest["files"].get(relative)
        if recorded is not None and recorded != digest:
            raise ValueError(f"recorded result-file hash mismatch: {path}")
        manifest["files"][relative] = digest
    _write_json_atomic(manifest_path, manifest)


def run_generation(
    *,
    config: Mapping[str, Any],
    condition: str,
    seed: int,
    metadata_path: Path,
    source_dir: Path,
    output_root: Path,
    bug_ids: Optional[Sequence[str]] = None,
    pilot: bool = False,
    resume: bool = False,
    config_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Generate corrected records after validation, importing GPU code lazily."""
    validate_protocol(config, condition, seed)
    if bug_ids and not pilot:
        raise ValueError("subset generation requires pilot mode")
    if pilot and not bug_ids:
        raise ValueError("pilot mode requires an explicit bug subset")
    condition_config = config["conditions"][condition]
    source_config = condition_config["source_config"]
    expected_retrieval_count = int(condition_config["expected_retrieval_count"])
    prompt_flags = condition_config.get("prompt_flags") or None
    model_config = dict(config["model"])
    generation_config = dict(config["generation"])
    report = validate_condition(
        metadata_path=metadata_path,
        source_dir=source_dir,
        source_config=source_config,
        seed=seed,
        expected_retrieval_count=expected_retrieval_count,
        output_root=output_root,
        run_name=f"{condition}_seed{seed}",
        resume=resume,
        prompt_flags=prompt_flags,
        manifest_extra={
            "config_sha256": config_sha256,
            "model": model_config,
            "generation": generation_config,
            "derive_no_ranking": bool(condition_config.get("derive_no_ranking")),
            **({"prompt_flags": dict(prompt_flags)} if prompt_flags else {}),
        },
    )
    frozen = load_frozen_inputs(
        metadata_path,
        source_dir,
        source_config,
        seed,
        expected_retrieval_count,
        use_ast=(prompt_flags or {}).get('use_ast', True),
        use_retrieval=(prompt_flags or {}).get('use_retrieval', expected_retrieval_count > 0),
        use_diffusion=(prompt_flags or {}).get('use_diffusion', True),
    )
    selected = {str(value) for value in bug_ids} if bug_ids else None
    if selected is not None:
        unknown = selected - {item.bug_id for item in frozen}
        if unknown:
            raise ValueError(f"unknown bug IDs: {sorted(unknown)}")
        frozen = [item for item in frozen if item.bug_id in selected]

    ranked_dir = Path(output_root) / "results" / f"{condition}_seed{seed}"
    unranked_condition = "starcoder2_no_ranking_nocommit"
    derive_unranked = bool(condition_config.get("derive_no_ranking"))
    unranked_dir = Path(output_root) / "results" / f"{unranked_condition}_seed{seed}"
    pool_dir = Path(output_root) / "results" / "_raw_pools" / report.run_name
    target_candidates = int(generation_config["target"])
    pending: list[tuple[FrozenReplayInput, str, dict[str, Any]]] = []
    for item in frozen:
        prompt = (report.prompt_dir / f"{item.bug_id}.txt").read_text(encoding="utf-8")
        expected = _expected_input_provenance(
            item, prompt, model_config, generation_config
        )
        paths_by_config = {
            condition: ranked_dir / f"{item.bug_id}.json",
        }
        if derive_unranked:
            paths_by_config[unranked_condition] = (
                unranked_dir / f"{item.bug_id}.json"
            )
        pool_path = pool_dir / f"{item.bug_id}.json"
        any_artifact = pool_path.exists() or any(
            path.exists() for path in paths_by_config.values()
        )
        if any_artifact:
            if not resume:
                raise FileExistsError(f"result already exists for bug {item.bug_id}")
            if not pool_path.exists():
                raise ValueError(
                    f"result recovery pool missing for bug {item.bug_id}"
                )
            _recover_result_set(
                output_root=Path(output_root),
                run_name=report.run_name,
                pool_path=pool_path,
                paths_by_config=paths_by_config,
                expected_input_provenance=expected,
                expected_candidates=target_candidates,
            )
            continue
        pending.append((item, prompt, expected))

    if not pending:
        return {"validated": report.valid_bugs, "generated": 0, "skipped": len(frozen)}

    from src.agents.patch_generator import (  # GPU dependencies: lazy by design
        CANDIDATE_POOL_POLICY,
        CANDIDATE_SELECTION_RANKED,
        CANDIDATE_SELECTION_UNRANKED,
        PatchGeneratorClient,
        generate_candidates_for_bug,
    )
    from src.utils.seeding import set_all_seeds

    import torch

    model_path = Path(model_config["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path missing: {model_path}")
    runtime_state = validate_gpu_runtime(torch)
    client = PatchGeneratorClient(str(model_path), require_4bit=True)
    runtime_state.update(client.runtime_state)
    gpu_name = runtime_state["gpu_name"]
    runtime_manifest_path = (
        Path(output_root) / "manifests" / f"{report.run_name}.runtime.json"
    )
    runtime_manifest = {
        "protocol": "corrected_replay_v1",
        "run_name": report.run_name,
        "runtime_state": runtime_state,
    }
    if runtime_manifest_path.exists():
        stored_runtime = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
        if stored_runtime != runtime_manifest:
            raise ValueError("runtime manifest drift")
    else:
        _write_json_atomic(runtime_manifest_path, runtime_manifest)
    generated = 0
    for item, prompt, input_provenance in pending:
        raw_source = json.loads(item.source_path.read_text(encoding="utf-8"))
        prompt_input = dict(item.prompt_input)
        prompt_input["prompt"] = prompt
        edits = (item.plans[0].get("edits") or []) if (prompt_flags or {}).get("use_diffusion", True) else []
        is_base = all(
            not (prompt_flags or {}).get(name, default)
            for name, default in (
                ("use_ast", True),
                ("use_retrieval", expected_retrieval_count > 0),
                ("use_diffusion", True),
            )
        )
        if is_base:
            validate_base_contract(
                prompt=prompt,
                edits=edits,
                plans=item.plans,
                generation_input=prompt_input,
            )
        set_all_seeds(seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        raw_candidates = generate_candidates_for_bug(
            client=client,
            bug_id=item.bug_id,
            base_code=(prompt_input.get("function") or {}).get("function_before") or "",
            edits=edits,
            prom_row=prompt_input,
            target=int(generation_config["target"]),
            batch_size=int(generation_config["batch_size"]),
            temperature=float(generation_config["temperature"]),
            top_p=float(generation_config["top_p"]),
            repetition_penalty=float(generation_config["repetition_penalty"]),
            max_new_tokens=int(generation_config["max_new_tokens"]),
            max_attempts=int(generation_config["max_attempts"]),
            no_rerank=True,
            prepared_prompt=prompt,
            return_raw_pool=True,
        )
        model_input_provenance = client.input_provenance_for(prompt)
        validate_model_input_provenance(
            model_input_provenance, expected_user_prompt=prompt
        )
        # The raw pool may exceed the evaluation budget; each view then spends
        # exactly `target_candidates` of it (CANDIDATE_POOL_POLICY).
        if len(raw_candidates) > int(generation_config["max_attempts"]) + 1:
            raise RuntimeError(
                f"bug {item.bug_id} produced {len(raw_candidates)} candidates, "
                f"more than the sampling budget permits"
            )
        views = build_candidate_views(raw_candidates, budget=target_candidates)
        vram_peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else 0.0
        )

        def build_record(view: Iterable[Mapping[str, Any]], name: str,
                         selection_policy: str) -> dict[str, Any]:
            record = {
                "bug_id": item.bug_id,
                "seed": seed,
                "config": name,
                "ast_conditioning": 'diagnostic_v1' if (prompt_flags or {}).get('use_ast', True) else 'none_v1',
                "plan_binding_policy": PLAN_BINDING_POLICY,
                "candidate_pool_policy": CANDIDATE_POOL_POLICY,
                "candidate_selection_policy": selection_policy,
                "raw_candidate_pool_size": views.raw_pool_size,
                "candidate_budget": target_candidates,
                "source_code_sha256": _sha256_text((item.prompt_input.get('function') or {}).get('function_before') or ''),
                "candidates": [_candidate_record(candidate, edits) for candidate in view],
                "diffusion_plan": item.plans if (prompt_flags or {}).get("use_diffusion", True) else [],
                "static_analysis_nodes": item.prompt_input[
                    "suspicious_nodes_topk_reranked"
                ] if (prompt_flags or {}).get("use_ast", True) else [],
                "retrieved_examples": ((item.prompt_input.get("bm25") or {}).get("top") or []) if (prompt_flags or {}).get("use_retrieval", True) else [],
                "ground_truth_modified_lines": raw_source.get("ground_truth_modified_lines") or [],
                "ground_truth_modified_nodes": raw_source.get("ground_truth_modified_nodes") or [],
                "vram_peak_mb": vram_peak,
                "gpu_name": gpu_name,
                "runtime_state": runtime_state,
                "input_provenance": input_provenance,
                "model_input_provenance": model_input_provenance,
                "leakage_audit": {"passed": True, "hard_matches": []},
            }
            result = add_candidate_integrity(record)
            # The *pool* both conditions were cut from, not this view's own
            # multiset: the views differ by exactly the ranker's selection.
            result["raw_candidate_multiset_sha256"] = views.raw_pool_digest
            return result

        records = {condition: build_record(views.ranked, condition, CANDIDATE_SELECTION_RANKED)}
        if derive_unranked:
            records[unranked_condition] = build_record(
                views.unranked, unranked_condition, CANDIDATE_SELECTION_UNRANKED
            )
        if len(
            {record["raw_candidate_multiset_sha256"] for record in records.values()}
        ) != 1:
            raise AssertionError("conditions were cut from different raw pools")
        if len({record["candidate_budget"] for record in records.values()}) != 1:
            raise AssertionError("conditions were given different candidate budgets")
        pool_path = pool_dir / f"{item.bug_id}.json"
        _write_json_atomic(
            pool_path,
            {
                "protocol": "corrected_replay_v1",
                "bug_id": item.bug_id,
                "input_provenance": input_provenance,
                "model_input_provenance": model_input_provenance,
                # The raw pool itself, so view membership and the selection
                # policy stay re-verifiable after the run.
                "pool_schema_version": POOL_SCHEMA_VERSION,
                "candidate_pool_policy": CANDIDATE_POOL_POLICY,
                "candidate_budget": target_candidates,
                "raw_candidate_multiset_sha256": views.raw_pool_digest,
                "raw_candidates": annotate_pool(raw_candidates),
                "sent_edits": edits[:MAX_PROMPT_EDITS],
                "records": records,
            },
        )
        paths_by_config = {
            condition: ranked_dir / f"{item.bug_id}.json",
        }
        if derive_unranked:
            paths_by_config[unranked_condition] = (
                unranked_dir / f"{item.bug_id}.json"
            )
        _recover_result_set(
            output_root=Path(output_root),
            run_name=report.run_name,
            pool_path=pool_path,
            paths_by_config=paths_by_config,
            expected_input_provenance=input_provenance,
            expected_candidates=target_candidates,
        )
        generated += 1
    return {"validated": report.valid_bugs, "generated": generated, "skipped": len(frozen) - generated}

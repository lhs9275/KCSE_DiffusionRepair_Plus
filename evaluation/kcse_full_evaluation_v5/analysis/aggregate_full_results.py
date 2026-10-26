"""Full experiment accounting over every evaluation root.

Reads the frozen generation results and the evaluation outputs of all three
evaluation roots, and reports the explicit denominator N, the S/F/U split, stage
coverage, and the S/N and (S+U)/N bounds required by full-execution-guideline-v2
section 7.

An attempt may appear in more than one root, because attempts that ended in an
infrastructure error (compile timeout under the withdrawn four-bug batching, or a
candidate file absent from the earlier bundle) were re-measured. The successful
measurement is the one that counts; every earlier attempt is retained and
reported under retries, so nothing is silently dropped or overwritten.

The script reports. It never drops a planned attempt from the denominator, and it
never converts an unresolved execution outcome into a failure.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GEN_ROOT = Path("/cephfs/home/s2022810061/tmp/2026_09_20/kcse_original_method_full_v1")
EVAL_ROOTS = [
    Path("/cephfs/home/s2022810061/tmp/2026_09_21/kcse_full_evaluation_v1"),
    Path("/cephfs/home/s2022810061/tmp/2026_09_21/kcse_full_evaluation_v2"),
    Path("/cephfs/home/s2022810061/tmp/2026_09_21/kcse_full_evaluation_v3"),
    Path("/cephfs/home/s2022810061/tmp/2026_09_21/kcse_full_evaluation_v4"),
    Path("/cephfs/home/s2022810061/tmp/2026_09_21/kcse_full_evaluation_v5"),
]
LATEST_ROOT = EVAL_ROOTS[-1]
METRICS_ROOT = GEN_ROOT / "bundle" / "repo"

sys.path.insert(0, str(METRICS_ROOT))
from src.metrics.corrected_metrics import prefix_repair_outcome  # noqa: E402
from src.metrics.pass_at_k import compute_pass_at_k  # noqa: E402

TASK_DIR = re.compile(r"^task_(\d+)_bug_([^_]+)_(base|full)_seed_(\d+)_job_(\d+)$")
RESULT_DIR = re.compile(
    r"^(?:slot|batch)_\d+_(?:task_\d+_)?(?P<run>\w+_seed\d+)_bugs?_(?P<bugs>[\w-]+)_job_\d+$"
)
INTEGRITY_FILES = ("manifest-check.txt", "host-start.txt", "host-end.txt",
                   "internal-start.txt", "internal-end.txt")
# Pandas IDs whose environment cannot import the package even with the fixed code.
ABI_IDS = {"37", "38", "41", "42", "43", "44", "45", "46"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_generation(tasks: list[dict]) -> dict[int, dict]:
    """Map task_index to its generation state. Missing stays missing."""
    found: dict[int, Path] = {}
    for directory in (GEN_ROOT / "results").iterdir():
        match = TASK_DIR.match(directory.name)
        if match:
            found[int(match.group(1))] = directory

    state: dict[int, dict] = {}
    for task in tasks:
        index = task["task_index"]
        directory = found.get(index)
        if directory is None:
            state[index] = {"generation": "not_run", "directory": None}
            continue
        completed = (directory / "GENERATION_COMPLETED.json").exists()
        result_path = directory / "result.json"
        record = {
            "generation": "completed" if completed else "incomplete",
            "directory": directory.name,
            "candidate_count": None,
            "candidate_sha256": [],
            "result_sha256": None,
        }
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (str(result["bug_id"]), result["arm"], int(result["seed"])) != (
                str(task["bug_id"]), task["arm"], int(task["seed"])
            ):
                raise SystemExit(f"task {index}: generation identity does not match the contract")
            record["candidate_count"] = int(result["candidate_count"])
            record["candidate_sha256"] = [c["patch_sha256"] for c in result["candidates"]]
            record["result_sha256"] = sha256_file(result_path)
        state[index] = record
    return state


def load_evaluation() -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """Map (run, bug) to its measurement, plus the list of superseded attempts.

    A Success entry wins over an error entry; two Success entries for one attempt
    are an error, because the same attempt must not be measured twice.
    """
    state: dict[tuple[str, str], dict] = {}
    superseded: list[dict] = []
    for root in EVAL_ROOTS:
        results = root / "results"
        if not results.exists():
            continue
        for directory in sorted(results.iterdir()):
            match = RESULT_DIR.match(directory.name)
            payload_path = directory / "evaluation.json"
            if not match:
                continue
            run = match.group("run")

            present = 0
            integrity_ok = True
            for name in INTEGRITY_FILES:
                path = directory / name
                # An absent file means the attempt has not reached that checkpoint.
                if not path.exists():
                    continue
                present += 1
                if "FAILED" in path.read_text():
                    integrity_ok = False
            return_code = None
            if (directory / "rc.txt").exists():
                return_code = int((directory / "rc.txt").read_text().strip())
            if not payload_path.exists():
                continue

            for bug_id, entry in json.loads(payload_path.read_text()).items():
                if entry is None:
                    continue
                record = {
                    "root": root.name,
                    "directory": directory.name,
                    "return_code": return_code,
                    "integrity_ok": integrity_ok,
                    "integrity_checkpoints_present": present,
                    "entry": entry,
                    "status": entry.get("status"),
                }
                key = (run, str(bug_id))
                previous = state.get(key)
                if previous is None:
                    state[key] = record
                    continue
                if previous["status"] == "Success" and record["status"] == "Success":
                    raise SystemExit(f"{key}: two successful measurements for one attempt")
                # A Success is the measurement. Between two errors the later one
                # is kept, because it came from the corrected configuration and
                # states why the attempt is still unmeasured.
                if previous["status"] == "Success":
                    superseded.append({"run": run, "bug_id": str(bug_id),
                                       "directory": directory.name,
                                       "status": record["status"]})
                    continue
                superseded.append({"run": run, "bug_id": str(bug_id),
                                   "directory": previous["directory"],
                                   "status": previous["status"]})
                state[key] = record
    return state, superseded


def classify(task: dict, generation: dict, evaluation: dict | None, k: int) -> dict:
    """Assign one attempt to exactly one accounting state."""
    bug = str(task["bug_id"])
    row = {
        "task_index": task["task_index"],
        "bug_id": bug,
        "arm": task["arm"],
        "seed": int(task["seed"]),
        "generation": generation["generation"],
        "candidate_count": generation.get("candidate_count"),
        "evaluation_present": evaluation is not None,
        "evaluation_status": None if evaluation is None else evaluation["status"],
        "integrity_ok": None if evaluation is None else evaluation["integrity_ok"],
        "outcome": None,
        "reason": None,
    }

    if generation["generation"] != "completed":
        row["outcome"] = "not_evaluated"
        row["reason"] = f"generation_{generation['generation']}"
        return row
    if evaluation is None:
        row["outcome"] = "not_evaluated"
        row["reason"] = "evaluation_missing"
        return row
    if evaluation["integrity_checkpoints_present"] < len(INTEGRITY_FILES):
        row["outcome"] = "unknown"
        row["reason"] = "integrity_checkpoints_incomplete"
        return row
    if not evaluation["integrity_ok"]:
        row["outcome"] = "unknown"
        row["reason"] = "integrity_check_failed"
        return row

    entry = evaluation["entry"]
    flags = entry.get("nucleus_sampling_flags") or []
    if entry.get("status") != "Success":
        row["outcome"] = "unknown"
        # The environment cannot import pandas even with the fixed code, so the
        # ground truth control never establishes a baseline to measure against.
        if entry.get("status") in {"Error: ground_truth_test", "Error: abi_rebuild_unresolved"} and bug in ABI_IDS:
            row["reason"] = "environment_numpy_abi_ground_truth_unavailable"
        else:
            row["reason"] = f"evaluation_status_{entry.get('status')}"
        return row
    if generation.get("candidate_count") == 0:
        # A completed generation with zero returned candidates is an observed
        # end-to-end failure under the stated contract, not a missing run.
        row["outcome"] = "failure"
        row["reason"] = "zero_candidate_generation"
        return row
    if not flags:
        row["outcome"] = "unknown"
        row["reason"] = "candidates_present_but_no_flags"
        return row
    if len(flags) != generation["candidate_count"]:
        row["outcome"] = "unknown"
        row["reason"] = (
            f"flag_count_{len(flags)}_vs_candidate_count_{generation['candidate_count']}"
        )
        return row

    row["outcome"] = prefix_repair_outcome(flags, k=k)
    row["reason"] = f"prefix_k{k}"
    return row


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[row["outcome"]] += 1
    solved = counts["success"]
    return {
        "N_planned": total,
        "success": solved,
        "failure": counts["failure"],
        "unknown": counts["unknown"],
        "not_evaluated": counts["not_evaluated"],
        "evaluated": solved + counts["failure"] + counts["unknown"],
        "confirmed_rate_S_over_N": solved / total if total else None,
        "possible_upper_bound_over_N": (
            (solved + counts["unknown"] + counts["not_evaluated"]) / total if total else None
        ),
    }


def pass_at_k_block(rows: list[dict], evaluation: dict[tuple[str, str], dict],
                    k_list: list[int]) -> dict:
    """pass@k over the full planned denominator, unknowns kept visible."""
    samples = []
    unknown_rows = 0
    for row in rows:
        record = evaluation.get((f"{row['arm']}_seed{row['seed']}", row["bug_id"]))
        if record is None or record["status"] != "Success":
            continue
        flags = record["entry"].get("nucleus_sampling_flags") or []
        if not flags:
            continue
        if any(flag not in {"Pass", "Fail"} for flag in flags):
            unknown_rows += 1
        samples.append((len(flags), sum(1 for flag in flags if flag == "Pass")))
    block = {"denominator_N": len(rows), "samples_with_flags": len(samples),
             "samples_containing_unresolved_flag": unknown_rows}
    for k in k_list:
        block[f"pass@{k}"] = (
            sum(compute_pass_at_k(n, c, k) for n, c in samples) / len(rows) if samples else None
        )
    return block


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1, help="prefix rank for the S/F/U split")
    parser.add_argument("--pass-at-k", type=int, nargs="*", default=[1, 5, 10])
    parser.add_argument("--output", default=str(LATEST_ROOT / "analysis" / "ACCOUNTING.json"))
    args = parser.parse_args()

    tasks = json.loads((GEN_ROOT / "bundle" / "contract" / "tasks.json").read_text())
    eligibility = json.loads(
        (GEN_ROOT / "bundle" / "contract" / "control-eligibility-v1.json").read_text()
    )
    manifest_ids = [str(value) for value in eligibility["sets"]["v1_revised_strict"]]

    contract_ids = sorted({str(task["bug_id"]) for task in tasks}, key=int)
    seeds = sorted({int(task["seed"]) for task in tasks})
    arms = sorted({task["arm"] for task in tasks})
    identity = {
        "manifest_id_count": len(manifest_ids),
        "contract_id_count": len(contract_ids),
        "ids_match_manifest": sorted(manifest_ids, key=int) == contract_ids,
        "seeds": seeds,
        "arms": arms,
        "N_expected": len(manifest_ids) * len(arms) * len(seeds),
        "N_in_contract": len(tasks),
    }
    identity["N_consistent"] = identity["N_expected"] == identity["N_in_contract"]

    generation = load_generation(tasks)
    evaluation, superseded = load_evaluation()
    rows = [
        classify(task, generation[task["task_index"]],
                 evaluation.get((f"{task['arm']}_seed{task['seed']}", str(task["bug_id"]))), args.k)
        for task in tasks
    ]

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[f"{row['arm']}_seed{row['seed']}"].append(row)

    reasons = defaultdict(int)
    for row in rows:
        if row["outcome"] in {"unknown", "not_evaluated"}:
            reasons[row["reason"]] += 1

    report = {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "prefix_k": args.k,
        "evaluation_roots": [str(root) for root in EVAL_ROOTS],
        "protocol_notes": {
            "compile_timeout": (
                "Attempts measured in v1 and v2 used the frozen standard 1200s. The v3 "
                "continuation used the frozen extended 3600s, after a single pandas build "
                "was measured at 1054s of 1200s. The limit bounds infrastructure failure "
                "only; a compile that succeeds is unaffected by which limit applied."
            ),
            "abi_rebuild_amendment": (
                "Attempts for pandas IDs 37, 38, 41, 42, 43, 44, 45 were measured with the "
                "harness ABI rebuild enabled, under the written amendment AMENDMENT-20260921.md. "
                "The rebuild changes workspace preparation only; generation, candidate order "
                "and scoring are unchanged. ID 46 is not in the eligibility set and contributes "
                "no attempt."
            ),
            "batching": (
                "Four-bug batching was withdrawn for pandas bugs because concurrent native "
                "builds exceeded the compile limit. pandas bugs are measured one per job."
            ),
        },
        "identity": identity,
        "overall": summarize(rows),
        "per_group": {name: summarize(group) for name, group in sorted(by_group.items())},
        "per_arm": {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in arms},
        "pass_at_k_per_group": {
            name: pass_at_k_block(group, evaluation, args.pass_at_k)
            for name, group in sorted(by_group.items())
        },
        "stage_coverage": {
            "generation_completed": sum(1 for r in rows if r["generation"] == "completed"),
            "generation_incomplete": sum(1 for r in rows if r["generation"] == "incomplete"),
            "generation_not_run": sum(1 for r in rows if r["generation"] == "not_run"),
            "evaluation_present": sum(1 for r in rows if r["evaluation_present"]),
            "integrity_failed": sum(1 for r in rows if r["integrity_ok"] is False),
        },
        "unresolved_reasons": dict(sorted(reasons.items())),
        "superseded_measurements": superseded,
        "distinguished_states": {
            "zero_candidate_generation": sorted(
                r["task_index"] for r in rows if r["reason"] == "zero_candidate_generation"
            ),
            "evaluation_missing": sorted(
                r["task_index"] for r in rows if r["reason"] == "evaluation_missing"
            ),
            "generation_missing_or_incomplete": sorted(
                r["task_index"] for r in rows
                if r["reason"] in {"generation_not_run", "generation_incomplete"}
            ),
            "abi_rebuild_enabled_attempts": sorted(
                r["task_index"] for r in rows if r["bug_id"] in ABI_IDS
            ),
            "environment_abi_unmeasurable": sorted(
                r["task_index"] for r in rows
                if r["reason"] == "environment_numpy_abi_ground_truth_unavailable"
            ),
            "unresolved_after_evaluation": sorted(
                r["task_index"] for r in rows if r["outcome"] == "unknown"
            ),
        },
        "rows": rows,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    overall = report["overall"]
    print(f"N={overall['N_planned']}  evaluated={overall['evaluated']}  "
          f"S={overall['success']} F={overall['failure']} U={overall['unknown']} "
          f"not_evaluated={overall['not_evaluated']}")
    print(f"identity_ok={identity['ids_match_manifest'] and identity['N_consistent']}  "
          f"superseded={len(superseded)}")
    for name, block in report["per_group"].items():
        print(f"  {name:16s} N={block['N_planned']:3d} S={block['success']:3d} "
              f"F={block['failure']:3d} U={block['unknown']:3d} "
              f"pending={block['not_evaluated']:3d}")
    for reason, count in report["unresolved_reasons"].items():
        print(f"  reason {reason}: {count}")
    print(f"written: {output}")


if __name__ == "__main__":
    main()

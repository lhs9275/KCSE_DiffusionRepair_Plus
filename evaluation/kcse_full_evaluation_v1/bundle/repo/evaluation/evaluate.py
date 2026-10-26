"""BugsInPy evaluation harness.

사용법:
    python evaluation/evaluate.py --results_dir results/starcoder2_full_seed42
    python evaluation/evaluate.py --results_dir results/starcoder2_full_seed42 --bugs 2,3,4
    python evaluation/evaluate.py --results_dir results/starcoder2_full_seed42 --num_processes 4
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import multiprocessing
import os
import re
import subprocess
import sys
import textwrap
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_json, save_json
from src.utils.result_schema import load_bug_result

_EVAL_DIR = Path(__file__).parent
_PROJECT_ROOT = _EVAL_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BugsInPy evaluation for DiffusionRepair")
    p.add_argument("--results_dir", required=True, help="결과 디렉토리 (results/<config>_seed<n>/)")
    p.add_argument("--bugs_meta", default="data/bugsinpy_51/bugs_meta_data.json")
    p.add_argument("--bugs", type=str, default="", help="처리할 bug_id (쉼표 구분). 비어있으면 전체.")
    p.add_argument("--controls-only", action="store_true", help="fixed/buggy 대조만 검사. 후보 생성·평가는 하지 않음.")
    p.add_argument("--workspace_root", type=str, default="temp_workspaces")
    p.add_argument("--num_processes", type=int, default=0)
    p.add_argument("--compile-timeout-seconds", type=int, default=1200,
                   help="compile timeout (default 1200; extended diagnostic only)")
    p.add_argument("--diagnostic-abi-rebuild", action="store_true",
                   help="one-shot pandas ABI diagnostic for the frozen eight IDs only")
    p.add_argument("--diagnostic-observations", action="store_true",
                   help="read-only observations for the frozen seven anomalous IDs only")
    p.add_argument("--output", type=str, default="", help="결과 JSON 저장 경로 (기본: results_dir/eval_results.json)")
    return p


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def adjust_indent(code: str, indent: int) -> str:
    code = textwrap.dedent(code)
    # Prefixing continuation lines of a triple-quoted literal changes its value.
    protected = {line for node in ast.walk(ast.parse(code))
                 if isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)))
                 for line in range(node.lineno + 1, node.end_lineno + 1)}
    return "".join((" " * indent if line.strip() and index not in protected else "") + line
                   for index, line in enumerate(code.splitlines(keepends=True), 1))


def _sha256_file(path: str | Path) -> Optional[str]:
    target = Path(path)
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None


def _invalidate_target_bytecode(file_path: str | Path) -> list[str]:
    """Remove only bytecode corresponding to the target source file."""
    target = Path(file_path)
    removed = []
    for bytecode in [target.with_suffix(".pyc"),
                     *(target.parent / "__pycache__").glob(target.stem + ".*.pyc")]:
        if bytecode.exists():
            bytecode.unlink()
            removed.append(str(bytecode))
    return removed


def _run_test_with_target_integrity(adapter, project_path: str, file_path: str) -> tuple[str, dict[str, Any]]:
    """Run the existing adapter test while preserving exceptions and source integrity."""
    details: dict[str, Any] = {
        "target_file_before_test_sha256": _sha256_file(file_path),
        "bytecode_removed_before_test": [],
    }
    adapter.last_test_result = {}
    try:
        details["bytecode_removed_before_test"] = _invalidate_target_bytecode(file_path)
        raw_flag = adapter.test(project_path)
        details.update(dict(getattr(adapter, "last_test_result", {}) or {}))
    except Exception as error:
        details.update(dict(getattr(adapter, "last_test_result", {}) or {}))
        raw_flag = "Error: test_exception"
        details.update({
            "classification": "test_execution_exception",
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "stdout": details.get("stdout", ""),
            "stderr": details.get("stderr", ""),
        })
    details["raw_test_flag"] = raw_flag
    details["target_file_after_test_sha256"] = _sha256_file(file_path)
    details["test_preserved_target_file"] = (
        details["target_file_before_test_sha256"] is not None
        and details["target_file_after_test_sha256"] == details["target_file_before_test_sha256"]
    )
    flag = raw_flag if details["test_preserved_target_file"] else "Error: target_file_mutated_during_test"
    adapter.last_test_result = details
    return flag, details


# ---------------------------------------------------------------------------
# Per-bug worker
# ---------------------------------------------------------------------------

def adjudicate_candidate_error(flag, details, candidate, fixed_code, target_file, fixed_control_passed, patch_line_range=None):
    """Conservative attribution; retain unknown/environment errors and raw evidence."""
    evidence = {"raw_flag": flag, "classification": "unresolved_execution_error"}
    if not flag.startswith("Error") or not fixed_control_passed or not patch_line_range or details.get("timed_out"):
        return flag, evidence
    output = details.get("stdout", "") + "\n" + details.get("stderr", "")
    exceptions = re.findall(r"^(?:E\s+)?([A-Za-z_]\w*(?:Error|Exception)):", output, re.MULTILINE)
    allowed = {"TypeError", "AttributeError", "NameError", "ValueError", "IndexError", "KeyError",
               "ZeroDivisionError", "ImportError", "ModuleNotFoundError", "SyntaxError"}
    if len(exceptions) != 1 or exceptions[0] not in allowed:
        return flag, evidence
    changed = {line.strip() for line in candidate.splitlines()} - {line.strip() for line in fixed_code.splitlines()}
    lines = output.splitlines()
    for index, line in enumerate(lines[:-1]):
        match = re.search(r'File "([^"]+)", line (\d+)', line) or re.match(r"([^\s]+\.py):(\d+):", line)
        if not match:
            continue
        path, number = match.group(1), int(match.group(2))
        source_line = re.sub(r"^E\s+", "", lines[index + 1]).strip()
        if ((path == target_file or path.endswith('/' + target_file))
                and patch_line_range[0] <= number <= patch_line_range[1] and source_line and source_line in changed):
            evidence.update(classification="candidate_failure", exception=exceptions[0],
                            target_file=target_file, source_line=source_line, traceback_line=number,
                            fixed_control_passed=True, patch_line_range=list(patch_line_range))
            return "Fail", evidence
    return flag, evidence


def _execution_test(
    adapter,
    project_path: str,
    bug_meta: Dict[str, Any],
    candidate_code: str,
    fixed_control_passed: bool = False,
) -> str:
    execution_evidence = {
        "candidate_input_sha256": hashlib.sha256(candidate_code.encode("utf-8")).hexdigest(),
        "target_file_before_sha256": None,
        "executed_function_sha256": None,
        "target_file_during_test_sha256": None,
        "target_file_after_restore_sha256": None,
        "restore_integrity": None,
    }
    adapter.last_test_result = execution_evidence
    file_path = os.path.join(project_path, bug_meta["file"]["file_path"])
    backup = file_path + ".backup"
    if not os.path.exists(file_path):
        return "Error: file_not_found"
    execution_evidence["target_file_before_sha256"] = _sha256_file(file_path)
    try:
        compile(textwrap.dedent(candidate_code), "<candidate>", "exec")
    except SyntaxError as error:
        execution_evidence.update(
            target_file_after_restore_sha256=execution_evidence["target_file_before_sha256"],
            restore_integrity=True,
        )
        adapter.last_test_result = {**execution_evidence,
                                    "raw_flag": "Error: candidate_syntax", "classification": "candidate_failure",
                                    "phase": "candidate_compile", "stdout": "", "stderr": str(error),
                                    "flag": "Fail", "returncode": None, "timed_out": False}
        return "Fail"
    try:
        subprocess.run(["cp", file_path, backup], check=True)
        func_start = bug_meta["function"]["function_after_start_line"]
        func_end = bug_meta["function"]["function_after_end_line"]
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        fixed_code = bug_meta["function"].get("function_after")
        if fixed_code:
            expected = ast.parse(fixed_code).body[0]
            shape = ast.dump(expected, include_attributes=False)
            matches = [node for node in ast.walk(ast.parse("".join(lines)))
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                       and ast.dump(node, include_attributes=False) == shape]
            if len(matches) != 1:
                return "Error: function_identity"
            actual = matches[0]
            func_start, func_end = actual.lineno, actual.end_lineno
            candidate_node = ast.parse(textwrap.dedent(candidate_code)).body[0]
            # Keep the original decorator lines outside the function replacement.
            # A supplied copy must agree and must not apply the wrapper twice.
            decorators = getattr(candidate_node, "decorator_list", [])
            if decorators:
                if [ast.dump(n, include_attributes=False) for n in decorators] != [
                    ast.dump(n, include_attributes=False) for n in actual.decorator_list
                ]:
                    return "Error: decorator_identity"
                candidate_code = "\n".join(candidate_code.splitlines()[candidate_node.lineno - 1:])
        indent = len(lines[func_start - 1]) - len(lines[func_start - 1].lstrip(" "))
        patched = adjust_indent(candidate_code, indent)
        new_lines = lines[: func_start - 1] + [patched, "\n"] + lines[func_end:]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("".join(new_lines))
        execution_evidence["executed_function_sha256"] = hashlib.sha256(
            (patched + "\n").encode("utf-8")
        ).hexdigest()
        execution_evidence["target_file_during_test_sha256"] = _sha256_file(file_path)
        flag, details = _run_test_with_target_integrity(adapter, project_path, file_path)
        details.update(execution_evidence)
        adapter.last_test_result = details
        raw_test_flag = details.get("raw_test_flag", flag)
        if (flag == raw_test_flag and raw_test_flag.startswith("Error")
                and details.get("classification") != "test_execution_exception"):
            flag, attribution = adjudicate_candidate_error(raw_test_flag, details, candidate_code, fixed_code or "",
                bug_meta["file"]["file_path"], fixed_control_passed, (func_start, func_start + len(patched.splitlines()) - 1))
            details.update(attribution)
            adapter.last_test_result = details
        return "Pass" if flag == "Plausible" else flag
    except Exception as e:
        return f"Error: {e}"
    finally:
        if os.path.exists(backup):
            details = dict(getattr(adapter, "last_test_result", {}) or {})
            try:
                restored = subprocess.run(["mv", backup, file_path], check=False)
                restore_returncode = restored.returncode
                restore_exception = None
            except Exception as error:
                restore_returncode = None
                restore_exception = {"type": type(error).__name__, "message": str(error)}
            try:
                removed_after_restore = _invalidate_target_bytecode(file_path)
                cache_cleanup_exception = None
            except Exception as error:
                removed_after_restore = []
                cache_cleanup_exception = {"type": type(error).__name__, "message": str(error)}
            after_hash = _sha256_file(file_path)
            details.update(execution_evidence)
            details["target_file_after_restore_sha256"] = after_hash
            details["restore_returncode"] = restore_returncode
            details["restore_exception"] = restore_exception
            details["bytecode_removed_after_restore"] = removed_after_restore
            details["cache_cleanup_exception"] = cache_cleanup_exception
            details["restore_integrity"] = (
                restore_returncode == 0
                and restore_exception is None
                and cache_cleanup_exception is None
                and after_hash == execution_evidence["target_file_before_sha256"]
            )
            adapter.last_test_result = details


def evaluate_bug_worker(args_tuple: tuple) -> Tuple[str, Dict[str, Any]]:
    bug_id, bug_meta, results_dir, workspace_root = args_tuple[:4]
    controls_only = bool(args_tuple[4]) if len(args_tuple) > 4 else False
    compile_timeout_seconds = int(args_tuple[5]) if len(args_tuple) > 5 else 1200
    diagnostic_abi_rebuild = bool(args_tuple[6]) if len(args_tuple) > 6 else False
    diagnostic_observations = bool(args_tuple[7]) if len(args_tuple) > 7 else False
    pid = os.getpid()  # worker 프로세스의 실제 pid (메인 프로세스가 아님)

    from dataset_adapter import BugsInPy
    adapter = BugsInPy()
    if diagnostic_observations and bug_id == "67":
        adapter.observe_timeout_process_tree = True

    project_name = adapter.map_project_name(bug_meta["project_name"])
    bugsinpy_id = str(bug_meta["bugsinpy_id"])

    os.makedirs(workspace_root, exist_ok=True)
    workspace = tempfile.mkdtemp(prefix=f"ws_{bug_id}_{pid}_", dir=workspace_root)
    evidence = []
    evidence_path = Path(workspace_root).parent / "test_evidence" / (Path(workspace).name + ".jsonl")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    def record_test(stage, flag, code=None, candidate_index=None):
        row = {**getattr(adapter, "last_test_result", {}), "stage": stage, "flag": flag,
               "bug_id": bug_id, "candidate_index": candidate_index,
               "patch_sha256": hashlib.sha256(code.encode()).hexdigest() if code is not None else None}
        evidence.append(row)
        # Flush each event outside the disposable workspace, even if the run stops.
        with evidence_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        if adapter.should_skip_bug(bug_id):
            return bug_id, {"status": "skipped", "nucleus_sampling_flags": []}

        identity = adapter.resolve_bug(project_name, bug_meta)
        bugsinpy_id = identity["resolved_bugsinpy_id"]
        if diagnostic_observations and bug_id == "54":
            from evaluation.controls_diagnostic import position_preserving_function_diff
            evidence_path.with_suffix(".observations.json").write_text(
                json.dumps(position_preserving_function_diff(bug_meta), indent=2), encoding="utf-8"
            )
        project_path_base = adapter.build_project_path(workspace, project_name, bugsinpy_id)
        evidence_path.with_suffix(".identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

        # checkout
        if not adapter.checkout(project_name, bugsinpy_id, project_path_base):
            return bug_id, {"status": "Error: checkout", "nucleus_sampling_flags": []}

        compile_path = os.path.join(project_path_base, project_name)
        if not os.path.isdir(compile_path):
            compile_path = project_path_base

        compiled = (adapter.compile(compile_path) if compile_timeout_seconds == 1200
                    else adapter.compile(compile_path, timeout_seconds=compile_timeout_seconds))
        evidence_path.with_suffix(".compile.json").write_text(json.dumps(getattr(adapter, "last_compile_result", {})), encoding="utf-8")
        if not compiled:
            return bug_id, {"status": "Error: compile", "nucleus_sampling_flags": []}

        if diagnostic_observations and bug_id == "61":
            from evaluation.controls_diagnostic import observe_black_import_environment
            evidence_path.with_suffix(".observations.json").write_text(
                json.dumps(observe_black_import_environment(compile_path, bug_meta), indent=2),
                encoding="utf-8"
            )

        target_file = os.path.join(compile_path, bug_meta["file"]["file_path"])
        fixed_flag, fixed_details = _run_test_with_target_integrity(
            adapter, compile_path, target_file
        )
        fixed_details["target_file_sha256"] = _sha256_file(
            target_file
        )
        adapter.last_test_result = fixed_details
        if diagnostic_abi_rebuild and fixed_flag != "Plausible":
            from evaluation.controls_diagnostic import is_numpy_abi_error, one_shot_pandas_abi_rebuild
            record_test("fixed_control_initial", fixed_flag)
            if is_numpy_abi_error(fixed_details):
                rebuild = one_shot_pandas_abi_rebuild(compile_path, bug_meta["file"]["file_path"])
                evidence_path.with_suffix(".abi-rebuild.json").write_text(
                    json.dumps(rebuild, indent=2), encoding="utf-8"
                )
                if rebuild["status"] != "rebuild_completed":
                    return bug_id, {"status": "Error: abi_rebuild_unresolved",
                                    "nucleus_sampling_flags": [], "test_evidence": evidence}
                fixed_flag, fixed_details = _run_test_with_target_integrity(
                    adapter, compile_path, target_file
                )
                fixed_details["target_file_sha256"] = _sha256_file(target_file)
                adapter.last_test_result = fixed_details
                record_test("fixed_control_post_rebuild", fixed_flag)
            else:
                return bug_id, {"status": "Error: abi_trigger_absent",
                                "nucleus_sampling_flags": [], "test_evidence": evidence}
        record_test("fixed_control", fixed_flag)
        if fixed_flag != "Plausible":
            return bug_id, {"status": "Error: ground_truth_test", "nucleus_sampling_flags": [], "test_evidence": evidence}

        buggy_flag = _execution_test(adapter, compile_path, bug_meta, bug_meta["function"]["function_before"], fixed_control_passed=True)
        if diagnostic_observations and bug_id in ("19", "66"):
            mixed_details = dict(getattr(adapter, "last_test_result", {}) or {})
            mixed_details["mixed_failure_error_review_required"] = str(
                mixed_details.get("raw_test_flag", buggy_flag)
            ).startswith("Error")
            mixed_details["adjudication_changed_by_diagnostic"] = False
            adapter.last_test_result = mixed_details
        record_test("buggy_control", buggy_flag, bug_meta["function"]["function_before"])
        buggy_details = dict(getattr(adapter, "last_test_result", {}) or {})
        buggy_restore = bool(buggy_details.get("restore_integrity"))
        buggy_test_preserved = buggy_details.get("test_preserved_target_file") is not False
        if buggy_restore:
            restored_fixed_flag, restored_details = _run_test_with_target_integrity(
                adapter, compile_path, target_file
            )
        else:
            restored_fixed_flag = "Error: restore_integrity"
            restored_details = {"classification": "restore_integrity_failure"}
        restored_details["target_file_sha256"] = _sha256_file(
            target_file
        )
        restored_details["matches_initial_fixed_file"] = (
            restored_details["target_file_sha256"]
            == next((row.get("target_file_sha256") for row in evidence if row["stage"] == "fixed_control"), None)
        )
        adapter.last_test_result = restored_details
        record_test("restored_fixed_control", restored_fixed_flag)
        if (not buggy_restore or not buggy_test_preserved or restored_fixed_flag != "Plausible"
                or not restored_details["matches_initial_fixed_file"]):
            return bug_id, {"status": "Error: restored_fixed_control", "nucleus_sampling_flags": [],
                            "buggy_control_flag": buggy_flag, "test_evidence": evidence}
        if buggy_flag != "Fail":
            return bug_id, {"status": "Error: buggy_control", "nucleus_sampling_flags": [], "buggy_control_flag": buggy_flag, "test_evidence": evidence}

        if controls_only:
            return bug_id, {"status": "ControlsPassed", "nucleus_sampling_flags": [],
                            "test_evidence": evidence, "benchmark_identity": identity}

        # load saved result
        result_file = os.path.join(results_dir, f"{bug_id}.json")
        if not os.path.exists(result_file):
            return bug_id, {"status": "Error: result_file_missing", "nucleus_sampling_flags": []}
        record = load_bug_result(result_file)
        candidates = record.get("candidates") or []

        flags: List[str] = []
        for cand in candidates:
            code = cand.get("patch_text") or ""
            if not code:
                flags.append("Error: empty")
                continue
            flag = _execution_test(adapter, compile_path, bug_meta, code, fixed_control_passed=True)
            record_test("candidate", flag, code, len(flags))
            flags.append(flag)
            if not getattr(adapter, "last_test_result", {}).get("restore_integrity"):
                return bug_id, {"status": "Error: candidate_restore", "nucleus_sampling_flags": flags,
                                "test_evidence": evidence, "benchmark_identity": identity}
            if getattr(adapter, "last_test_result", {}).get("test_preserved_target_file") is False:
                return bug_id, {"status": "Error: candidate_execution_integrity",
                                "nucleus_sampling_flags": flags, "test_evidence": evidence,
                                "benchmark_identity": identity}
            logging.info(f"bug={bug_id} cand={len(flags)} → {flag}")

        return bug_id, {"status": "Success", "nucleus_sampling_flags": flags, "test_evidence": evidence,
                        "benchmark_identity": identity,
                        "result_file_sha256": hashlib.sha256(Path(result_file).read_bytes()).hexdigest()}

    except Exception as e:
        logging.error(f"[{bug_id}] Unhandled exception: {e}\n{traceback.format_exc()}")
        return bug_id, {"status": f"Error: {e}", "nucleus_sampling_flags": []}
    finally:
        shutil.rmtree(workspace)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_parser().parse_args()
    if args.compile_timeout_seconds not in (1200, 3600):
        get_parser().error("compile timeout must be the frozen standard 1200 or extended 3600")
    if args.diagnostic_abi_rebuild:
        from evaluation.controls_diagnostic import ABI_IDS
        selected = {b.strip() for b in args.bugs.split(",") if b.strip()}
        if not args.controls_only or not selected or not selected <= ABI_IDS or args.compile_timeout_seconds != 1200:
            get_parser().error("ABI diagnostic requires controls-only, IDs within the frozen eight, and compile timeout 1200")
    if args.diagnostic_observations:
        from evaluation.controls_diagnostic import OBSERVATION_IDS
        selected = {b.strip() for b in args.bugs.split(",") if b.strip()}
        if (args.diagnostic_abi_rebuild or not args.controls_only or not selected
                or not selected <= OBSERVATION_IDS or args.compile_timeout_seconds != 1200):
            get_parser().error("observation diagnostic requires controls-only, frozen seven IDs, and standard timeout")
    results_dir = Path(args.results_dir)
    bugs_meta_path = Path(args.bugs_meta)
    output_path = Path(args.output) if args.output else results_dir / "eval_results.json"

    bugs_meta: Dict[str, Any] = load_json(str(bugs_meta_path))
    # normalise keys to str
    bugs_meta = {str(k): v for k, v in bugs_meta.items()}

    if args.bugs:
        filter_ids = {b.strip() for b in args.bugs.split(",") if b.strip()}
        bugs_meta = {k: v for k, v in bugs_meta.items() if k in filter_ids}

    workspace_root = args.workspace_root
    if not os.path.isabs(workspace_root):
        workspace_root = str(_PROJECT_ROOT / workspace_root)
    os.makedirs(workspace_root, exist_ok=True)

    tasks = [
        (bug_id, meta, str(results_dir), workspace_root, args.controls_only,
         args.compile_timeout_seconds, args.diagnostic_abi_rebuild,
         args.diagnostic_observations)
        for bug_id, meta in sorted(bugs_meta.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    ]

    num_processes = args.num_processes or max(1, (os.cpu_count() or 2) - 1)
    logging.info(f"Evaluating {len(tasks)} bugs with {num_processes} processes → {results_dir}")

    all_results: Dict[str, Any] = {}
    with multiprocessing.Pool(processes=num_processes) as pool:
        for bug_id, result in pool.map(evaluate_bug_worker, tasks):
            all_results[bug_id] = result
            flags = result.get("nucleus_sampling_flags", [])
            passed = sum(1 for f in flags if f == "Pass")
            logging.info(f"bug={bug_id} status={result['status']} pass={passed}/{len(flags)}")

    save_json(all_results, str(output_path))
    logging.info(f"Saved evaluation results → {output_path}")

    total_bugs = len(all_results)
    solved = sum(
        1 for v in all_results.values()
        if any(f == "Pass" for f in v.get("nucleus_sampling_flags", []))
    )
    logging.info(f"Solved: {solved}/{total_bugs}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()

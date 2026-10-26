"""Performance-blind, one-shot controls diagnostics for the revision evaluator."""
from __future__ import annotations

import hashlib
import difflib
import json
import os
import subprocess
import time
from pathlib import Path

from evaluation.bugsinpy_command import _run_with_timeout

ABI_IDS = frozenset({"37", "38", "41", "42", "43", "44", "45", "46"})
OBSERVATION_IDS = frozenset({"17", "19", "25", "54", "66", "61", "67"})
ABI_REBUILD_SECONDS = 3600
_ABI_EXACT = ("ValueError: numpy.ndarray size changed, may indicate binary incompatibility. "
              "Expected 96 from C header, got 80 from PyObject")


def is_numpy_abi_error(test_details: dict) -> bool:
    output = str(test_details.get("stdout", "")) + "\n" + str(test_details.get("stderr", ""))
    return _ABI_EXACT in output


def position_preserving_function_diff(metadata: dict) -> dict:
    """Observe ordered line edits without changing the evaluator's Error attribution."""
    function = metadata["function"]
    before = function["function_before"].splitlines()
    after = function["function_after"].splitlines()
    first_fixed_line = int(function["function_after_start_line"])
    hunks = []
    for tag, buggy_start, buggy_end, fixed_start, fixed_end in difflib.SequenceMatcher(
            None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        hunks.append({"operation": tag,
                      "buggy_function_lines": list(range(buggy_start + 1, buggy_end + 1)),
                      "fixed_file_lines": list(range(first_fixed_line + fixed_start,
                                                     first_fixed_line + fixed_end)),
                      "fixed_anchor_line": first_fixed_line + fixed_start})
    return {"method": "SequenceMatcher ordered line opcodes, autojunk=False",
            "hunks": hunks, "adjudication_changed": False}


def observe_black_import_environment(project_path: str, metadata: dict) -> dict:
    """Observe import origin and encoding with the project's own Python, no test change."""
    project = Path(project_path)
    python = project / "env/bin/python"
    result = {"module": "black", "cwd": str(project),
              "function_before_sha256": hashlib.sha256(
                  metadata["function"]["function_before"].encode()).hexdigest()}
    if not python.is_file():
        result["status"] = "python_missing"
        return result
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(project), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    code = ("import importlib.util,json,locale,sys; "
            "s=importlib.util.find_spec('black'); "
            "print(json.dumps({'origin':s.origin if s else None,"
            "'filesystem_encoding':sys.getfilesystemencoding(),"
            "'preferred_encoding':locale.getpreferredencoding(False),"
            "'locale_ctype':locale.setlocale(locale.LC_CTYPE,None),"
            "'sys_path':sys.path}))")
    try:
        check = subprocess.run([str(python), "-c", code], cwd=str(project), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
                               check=False)
    except subprocess.TimeoutExpired as error:
        result.update(status="observation_timeout", stdout=(error.stdout or b"").decode(errors="replace"),
                      stderr=(error.stderr or b"").decode(errors="replace"))
        return result
    result.update(returncode=check.returncode,
                  stdout=check.stdout.decode(errors="replace"),
                  stderr=check.stderr.decode(errors="replace"))
    if check.returncode == 0:
        try:
            result.update(json.loads(result["stdout"]))
            result["status"] = "observed"
        except json.JSONDecodeError:
            result["status"] = "unparseable_output"
    else:
        result["status"] = "observation_error"
    return result


def _shared_object_hashes(project_path: Path) -> dict[str, str]:
    package = project_path / "pandas"
    if not package.is_dir():
        return {}
    return {
        str(path.relative_to(project_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.so"))
    }


def _numpy_environment(python: Path) -> dict:
    try:
        check = subprocess.run(
            [str(python), "-c", "import numpy,sys; print(sys.version); print(numpy.__version__); print(numpy.__file__); print(numpy.get_include())"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        return {"timed_out": True, "stdout": (error.stdout or b"").decode(errors="replace"),
                "stderr": (error.stderr or b"").decode(errors="replace")}
    return {"returncode": check.returncode,
            "stdout": check.stdout.decode(errors="replace"),
            "stderr": check.stderr.decode(errors="replace")}


def one_shot_pandas_abi_rebuild(project_path: str, target_file: str) -> dict:
    """Direct rebuild only; never upgrade dependencies or modify the test."""
    project = Path(project_path)
    python = project / "env/bin/python"
    setup = project / "setup.py"
    target = project / target_file
    if not (python.is_file() and setup.is_file() and target.is_file()):
        return {"status": "missing_rebuild_input", "python_exists": python.is_file(),
                "setup_exists": setup.is_file(), "target_exists": target.is_file()}
    before_target = hashlib.sha256(target.read_bytes()).hexdigest()
    result = {"command": [str(python), "setup.py", "build_ext", "--inplace", "--force", "-j", "2"],
              "cwd": str(project), "timeout_seconds": ABI_REBUILD_SECONDS,
              "numpy_before": _numpy_environment(python),
              "shared_objects_before": _shared_object_hashes(project),
              "target_file_before_sha256": before_target}
    started = time.monotonic()
    out, err, returncode, timed_out = _run_with_timeout(
        result["command"], timeout=ABI_REBUILD_SECONDS, cwd=str(project)
    )
    result.update(returncode=returncode, timed_out=timed_out,
                  elapsed_seconds=time.monotonic() - started,
                  stdout=out.decode(errors="replace"), stderr=err.decode(errors="replace"),
                  numpy_after=_numpy_environment(python),
                  shared_objects_after=_shared_object_hashes(project),
                  target_file_after_sha256=hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None)
    result["target_source_preserved"] = result["target_file_after_sha256"] == before_target
    result["status"] = "rebuild_completed" if (returncode == 0 and not timed_out
                        and result["target_source_preserved"]) else "rebuild_unresolved"
    return result

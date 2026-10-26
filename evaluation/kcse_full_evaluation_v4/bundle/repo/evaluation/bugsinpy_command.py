"""BugsInPy checkout / compile / test commands."""
from __future__ import annotations

import os
import hashlib
import shutil
import re
import signal
import subprocess
import time
from pathlib import Path


BUGSINPY_BUGS_FAIL_GROUND_TEST = {
    '1', '12', '13', '14', '15', '40', '55', '56', '57', '58', '59', '62', '63', '64', '65'
}

BUGSINPY_51_IDS = [
    '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
    '26', '27', '28', '29', '30', '31', '32', '33', '35', '36',
    '37', '38', '41', '42', '43', '44', '45', '46', '47', '48',
    '49', '50', '51', '52', '53', '54', '60', '61', '66', '67', '68',
]

SETUPTOOLS_VERSION = "39.0.1"
SETUPTOOLS_WHEEL = (
    "/usr/share/python-wheels/setuptools-39.0.1-py2.py3-none-any.whl"
)
SETUPTOOLS_WHEEL_SHA256 = (
    "0273e0eee59f1e5307cacaeffd9f3880b9c01a849cc6c52193337eb1cf21efba"
)


def map_git_to_bugsinpy_project_name(git_project_name: str) -> str:
    if git_project_name == "cli":
        return "httpie"
    if git_project_name == "spaCy":
        return "spacy"
    return git_project_name


def resolve_benchmark_bug(project_name: str, metadata: dict, *, projects_root=Path("/BugsInPy/projects")) -> dict:
    """Resolve numeric-version drift by the original fixing commit, or stop."""
    commit = str((metadata.get("commit") or {}).get("commit_id") or "").lower()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", project_name) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError("benchmark identity needs a project and fixing commit")
    matches = []
    for path in sorted((Path(projects_root) / project_name / "bugs").glob("*/bug.info")):
        raw = path.read_bytes()
        fields = dict(re.findall(r'''^\s*(\w+)\s*=\s*["']([^"']*)["']''', raw.decode(), re.MULTILINE))
        fixed = fields.get("fixed_commit_id", "").lower()
        if not re.fullmatch(r"[0-9a-f]{7,40}", fixed):
            continue
        if commit.startswith(fixed) or fixed.startswith(commit):
            matches.append((path, fields, raw))
    if len(matches) != 1:
        raise ValueError(f"benchmark identity has {len(matches)} matching IDs for {project_name}@{commit}")
    path, fields, raw = matches[0]
    return {
        "project": project_name,
        "metadata_bugsinpy_id": str(metadata.get("bugsinpy_id")),
        "resolved_bugsinpy_id": path.parent.name,
        "metadata_fix_commit": commit,
        "fixed_commit_id": fields["fixed_commit_id"],
        "buggy_commit_id": fields.get("buggy_commit_id"),
        "test_file": fields.get("test_file"),
        "bug_info_sha256": hashlib.sha256(raw).hexdigest(),
        "run_test_sha256": hashlib.sha256((path.parent / "run_test.sh").read_bytes()).hexdigest(),
    }


def bugsinpy_checkout(project_name: str, bug_id: str, checkout_path: str) -> bool:
    if os.path.isdir(checkout_path):
        shutil.rmtree(checkout_path)
    p = subprocess.Popen(
        ["bugsinpy-checkout", "-p", project_name, "-i", bug_id, "-w", checkout_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    out, err = p.communicate()
    if p.returncode != 0 or not os.path.isdir(checkout_path):
        return False
    return True


def bugsinpy_compile(project_dir: str, *, details=None, timeout_seconds: int = 1200) -> bool:
    started = time.monotonic()
    out, err, returncode, timed_out = _run_with_timeout(
        ["bugsinpy-compile"], timeout=timeout_seconds, cwd=project_dir
    )
    if details is not None:
        details.update(command=["bugsinpy-compile"], cwd=project_dir, stdout=out.decode(errors="replace"),
                       stderr=err.decode(errors="replace"), returncode=returncode, timed_out=timed_out,
                       timeout_seconds=timeout_seconds, elapsed_seconds=time.monotonic() - started)
    if timed_out or returncode != 0:
        return False
    if "This is not a checkout project folder" in out.decode(errors="replace"):
        return False
    return _restore_bundled_setuptools(project_dir)


def _restore_bundled_setuptools(project_dir: str) -> bool:
    """Undo live-index drift using only the wheel embedded in evaluator v6."""
    python = Path(project_dir) / "env/bin/python"
    wheel = Path(SETUPTOOLS_WHEEL)
    if not python.is_file() or not wheel.is_file():
        return False
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != SETUPTOOLS_WHEEL_SHA256:
        return False
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if install.returncode != 0:
        return False
    verify = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.metadata; "
                "print(importlib.metadata.version('setuptools'))"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return (
        verify.returncode == 0
        and verify.stdout.decode(errors="replace").strip() == SETUPTOOLS_VERSION
    )


def bugsinpy_test(project_dir: str, *, details=None, target_file=None,
                 observe_timeout_process_tree: bool = False) -> str:
    # The pytest console script starts from env/bin; a failed editable install
    # can otherwise test the old site-packages copy instead of the patched file.
    import_root = os.path.abspath(project_dir)
    if target_file is not None:
        project = Path(project_dir).resolve()
        target = (project / target_file).resolve()
        target.relative_to(project)  # Refuse paths outside this checkout.
        if not target.is_file():
            raise FileNotFoundError(f"test target missing: {target_file}")
        # Highest enclosing Python package: e.g. lib/ansible -> lib, sanic -> root.
        for parent in target.parents:
            if parent == project:
                break
            if (parent / '__init__.py').is_file():
                import_root = str(parent.parent)
    environment = dict(os.environ)
    paths = [import_root]
    if os.path.abspath(project_dir) != import_root:
        paths.append(os.path.abspath(project_dir))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    timeout_observation = {} if observe_timeout_process_tree else None
    out, err, returncode, timed_out = _run_with_timeout(
        ["bugsinpy-test"], timeout=120, cwd=project_dir, env=environment,
        timeout_observation=timeout_observation
    )
    stdout, stderr = out.decode(errors="replace"), err.decode(errors="replace")
    combined = stdout + "\n" + stderr
    if timed_out:
        flag = "Timeout"
    elif re.search(
        r"\b[1-9]\d* errors?\b|^FAILED \([^\n)]*errors=[1-9]\d*"
        r"|^ERROR collecting\b|^ImportError while loading conftest\b|^INTERNALERROR>"
        r"|^ERROR: (?:usage|file or directory not found)", combined, re.MULTILINE
    ):
        flag = "Error: test_execution"
    elif re.search(r"\b[1-9]\d* failed\b|^FAILED(?: \(|\s*$)", combined, re.MULTILINE):
        flag = "Fail" if returncode in (0, 1) else f"Error: exit_{returncode}"
    elif returncode != 0:
        flag = f"Error: exit_{returncode}"
    elif re.search(r"\b[1-9]\d* passed\b", combined) or (
        re.search(r"Ran [1-9]\d* tests?", combined) and re.search(r"^OK(?:\s|$)", combined, re.MULTILINE)
    ):
        flag = "Plausible"
    else:
        flag = "Error: unrecognized_test_output"
    if details is not None:
        details.update(command=["bugsinpy-test"], cwd=project_dir, stdout=stdout,
                       stderr=stderr, returncode=returncode, timed_out=timed_out, flag=flag,
                       import_root=import_root, pythonpath=environment["PYTHONPATH"])
        if timeout_observation is not None:
            details["timeout_process_observation"] = timeout_observation
    return flag


def _snapshot_process_tree(root_pid: int) -> list[dict]:
    """Best-effort /proc observation of only descendants of our test wrapper."""
    queue = [root_pid]
    visited = set()
    rows = []
    while queue and len(rows) < 256:
        pid = queue.pop(0)
        if pid in visited:
            continue
        visited.add(pid)
        base = Path("/proc") / str(pid)
        try:
            children = (base / "task" / str(pid) / "children").read_text().split()
            status = (base / "status").read_text().splitlines()
            state = next((line for line in status if line.startswith("State:")), None)
            cmdline = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            wchan = (base / "wchan").read_text().strip()
            rows.append({"pid": pid, "state": state, "cmdline": cmdline,
                         "wchan": wchan, "children": list(map(int, children))})
            queue.extend(map(int, children))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            rows.append({"pid": pid, "observation_error": "process_missing_or_denied"})
    return rows


def _run_with_timeout(cmd: list, timeout: int = 300, cwd=None, env=None,
                      timeout_observation: dict | None = None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=env)
    timed_out = False
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        if timeout_observation is not None:
            timeout_observation.update(wrapper_pid=p.pid,
                                       process_tree_before_termination=_snapshot_process_tree(p.pid))
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = p.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, err = p.communicate()
        if timeout_observation is not None:
            timeout_observation["wrapper_returncode_after_termination"] = p.returncode
    return out, err, p.returncode, timed_out

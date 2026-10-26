"""Dataset adapter for BugsInPy evaluation."""
from __future__ import annotations

import os
import shutil
import hashlib
from pathlib import Path

from bugsinpy_command import (
    BUGSINPY_BUGS_FAIL_GROUND_TEST,
    bugsinpy_checkout,
    bugsinpy_compile,
    bugsinpy_test,
    map_git_to_bugsinpy_project_name,
    resolve_benchmark_bug,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BugsInPy:
    dataset_name = "bugsinpy"

    def map_project_name(self, name: str) -> str:
        return map_git_to_bugsinpy_project_name(name)

    def should_skip_bug(self, bug_id: str) -> bool:
        return str(bug_id) in BUGSINPY_BUGS_FAIL_GROUND_TEST

    def resolve_bug(self, project_name: str, metadata: dict) -> dict:
        self.target_file = (metadata.get("file") or {}).get("file_path")
        self.benchmark_identity = resolve_benchmark_bug(project_name, metadata)
        return self.benchmark_identity

    def checkout(self, project_name: str, bug_id: str, project_path: str) -> bool:
        identity = getattr(self, "benchmark_identity", {})
        if identity.get("project") != project_name or identity.get("resolved_bugsinpy_id") != str(bug_id):
            return False
        os.chdir(_PROJECT_ROOT)
        if not bugsinpy_checkout(project_name, bug_id, project_path):
            return False
        info = Path(project_path) / project_name / "bugsinpy_bug.info"
        if not info.is_file():
            info = Path(project_path) / "bugsinpy_bug.info"
        return info.is_file() and hashlib.sha256(info.read_bytes()).hexdigest() == identity["bug_info_sha256"]

    def compile(self, project_path: str, *, timeout_seconds: int = 1200) -> bool:
        self.last_compile_result = {}
        return bugsinpy_compile(project_path, details=self.last_compile_result,
                                timeout_seconds=timeout_seconds)

    def test(self, project_path: str) -> str:
        self.last_test_result = {}
        return bugsinpy_test(project_path, details=self.last_test_result,
                             target_file=getattr(self, 'target_file', None),
                             observe_timeout_process_tree=bool(getattr(self, 'observe_timeout_process_tree', False)))

    def build_project_path(self, workspace_root: str, project_name: str, bug_id: str) -> str:
        return os.path.join(workspace_root, f"{project_name}_{bug_id}")

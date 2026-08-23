"""
Verification — deciding what "done" means for a workspace.

`RepairLoop` and `PlannerLoop` accept `Finish(succeeded=True)` on the model's
word. That is the gap between an agent that runs commands and one that gets
things done: a local model will confidently declare success after writing a
file it never executed.

This module only *chooses and describes* the check. Running it is left to the
loop, via `agentkit.gate.VerifyGate`, so the result arrives through the normal
observation path — journalled, counted by the ledger, and visible to the
guardrail detectors like any other step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["VerifySpec", "detect_verify", "PYTEST", "COMPILEALL", "NPM_TEST"]


@dataclass(frozen=True)
class VerifySpec:
    command: str
    timeout_s: float = 120.0
    label: str = "verification"


PYTEST = VerifySpec("pytest -q", label="pytest")
NPM_TEST = VerifySpec("npm test", label="npm test")
# The weakest useful verifier, and still worth having: it catches the single
# most common local-model failure, which is emitting code that does not parse.
COMPILEALL = VerifySpec("python -m compileall -q .", label="compile check")


def _has_python_tests(workspace: Path) -> bool:
    if (workspace / "pytest.ini").exists() or (workspace / "conftest.py").exists():
        return True
    if (workspace / "tests").is_dir():
        return True
    for pattern in ("test_*.py", "*_test.py"):
        if any(workspace.glob(pattern)):
            return True
    return False


def _npm_test_script(workspace: Path) -> bool:
    manifest = workspace / "package.json"
    if not manifest.is_file():
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    scripts = data.get("scripts")
    return isinstance(scripts, dict) and bool(scripts.get("test"))


def detect_verify(workspace: Path) -> VerifySpec | None:
    """
    Pick a verification command by inspecting the workspace.

    Ordered most-specific first. Returns None when there is nothing to check —
    an empty directory, or one with no code in it — so a documentation-only task
    is not forced through a pointless compile step.
    """
    workspace = Path(workspace)
    if not workspace.is_dir():
        return None
    if _has_python_tests(workspace):
        return PYTEST
    if _npm_test_script(workspace):
        return NPM_TEST
    if any(workspace.rglob("*.py")):
        return COMPILEALL
    return None

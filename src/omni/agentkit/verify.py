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
# A static web project has no test suite, so nothing checked that generated
# JavaScript agreed with the page it runs in. `omni.webcheck` proves the facts
# that can be proved without a browser: referenced files exist, and every id a
# script reaches for is defined somewhere.
WEBCHECK = VerifySpec("python -m omni.webcheck", timeout_s=60.0,
                      label="web contract check")
# The same check plus a real browser: pages are served over HTTP, loaded, and
# every control clicked. Only selected when Playwright and its browser are
# actually present, so the core install stays four pure-Python dependencies.
WEBCHECK_BROWSER = VerifySpec("python -m omni.webcheck --browser",
                              timeout_s=180.0, label="web + browser check")


def browser_available() -> bool:
    """True if a headless browser can actually be launched."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            browser.close()
        return True
    except Exception:                                       # noqa: BLE001
        return False
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


def _has_pages(workspace: Path) -> bool:
    from omni.webcheck import SKIP_DIRS
    return any(not any(part in SKIP_DIRS for part in page.parts)
               for page in workspace.rglob("*.html"))


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
    # Before falling back to a compile check: a project with pages has a
    # contract between them that nothing else here inspects.
    if _has_pages(workspace):
        return WEBCHECK_BROWSER if browser_available() else WEBCHECK
    if any(workspace.rglob("*.py")):
        return COMPILEALL
    return None

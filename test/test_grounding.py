"""
Tests for grounding generated files in the workspace they are written into.

Regression context: a run read `index.html`, then generated a `js/app.js` that
bound to `history-list`, `clear-history` and `history-panel`. None existed — the
real ids were `historyList`, `btnClearHistory` and `sidebar`. Four of five
bindings missed, so the page rendered and silently did nothing.

`generate_file` runs its own completion so a large file need not fit inside a
plan. The cost was that the completion saw only a one-sentence spec: not the
files the loop had just read, not the modules the new file must call. The
information was in the run and was thrown away at that boundary.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from omni import backends as llm_backends
from omni import runtime as ar
from omni.agentkit.dispatch import MultiToolExecutor
from omni.agentkit.tools import build_default_registry, codegen


def run(coro):
    return asyncio.run(coro)


class _Recorder:
    """Captures the prompt the generator was given."""

    def __init__(self, text: str = "// generated\n") -> None:
        self.text = text
        self.users: list[str] = []

    async def complete(self, system, user, *, schema=None, max_tokens=512):
        self.users.append(user)
        return llm_backends.Completion(text=self.text, model="t",
                                       backend=llm_backends.Backend.OLLAMA,
                                       finish_reason="stop")


INDEX_HTML = """<!DOCTYPE html>
<html>
<body>
  <aside id="sidebar">
    <button id="btnClearHistory">Clear</button>
    <ul id="historyList"></ul>
  </aside>
  <div id="display"><div id="expression"></div><div id="result">0</div></div>
  <script src="js/calculator.js"></script>
  <script src="js/history.js"></script>
  <script src="js/app.js"></script>
</body>
</html>
"""

CALCULATOR_JS = "class Calculator {\n  evaluate(expr) { return 0; }\n}\n"
HISTORY_JS = "class SessionHistory {\n  add(entry) {}\n}\n"
STYLES_CSS = ".app-container { display: flex; }\n" * 20


@pytest.fixture
def project(tmp_path) -> Path:
    """The shape of the workspace at the moment app.js was generated."""
    (tmp_path / "js").mkdir()
    (tmp_path / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (tmp_path / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    (tmp_path / "js" / "calculator.js").write_text(CALCULATOR_JS, encoding="utf-8")
    (tmp_path / "js" / "history.js").write_text(HISTORY_JS, encoding="utf-8")
    return tmp_path


def labels(context: str) -> list[str]:
    """Paths of the blocks in a context, without the trailing annotation."""
    out = []
    for line in context.splitlines():
        if not line.startswith("--- "):
            continue
        label = line[4:].rsplit(" ---", 1)[0]
        out.append(label.split(" (")[0].strip())
    return out


# ---------------------------------------------------------------------------
# what gets gathered
# ---------------------------------------------------------------------------

class TestGatherContext:
    def test_the_referencing_html_is_included(self, project):
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert "index.html" in labels(ctx)

    def test_sibling_modules_are_included(self, project):
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert "js/calculator.js" in labels(ctx)
        assert "js/history.js" in labels(ctx)

    def test_every_id_the_failing_run_invented_is_now_visible(self, project):
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        for real_id in ("historyList", "btnClearHistory", "sidebar",
                        "expression", "result", "display"):
            assert real_id in ctx, f"{real_id} not visible to the generator"

    def test_loose_substring_matches_are_excluded(self, project):
        """`styles.css` mentions `.app-container`; that is not a reference."""
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert "styles.css" not in labels(ctx)

    def test_the_target_is_not_listed_as_its_own_reference(self, project):
        (project / "js" / "app.js").write_text("// old\n", encoding="utf-8")
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert labels(ctx).count("js/app.js") == 1

    def test_an_existing_target_is_shown_last_and_labelled(self, project):
        (project / "js" / "app.js").write_text("// PREVIOUS\n", encoding="utf-8")
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert "the file you are rewriting" in ctx
        assert "PREVIOUS" in ctx

    def test_budget_is_respected(self, project):
        ctx = codegen.gather_context(project, project / "js" / "app.js",
                                     budget=500)
        assert len(ctx) < 900          # budget plus per-file headers

    def test_clipped_files_say_so(self, project):
        big = "x = 1;\n" * 2000
        (project / "js" / "big.js").write_text(big, encoding="utf-8")
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        if "js/big.js" in labels(ctx):
            assert "chars elided" in ctx

    def test_empty_workspace_yields_nothing(self, tmp_path):
        assert codegen.gather_context(tmp_path, tmp_path / "new.py") == ""

    def test_binary_neighbours_are_skipped(self, project):
        (project / "js" / "blob.js").write_bytes(b"\x00\x01app.js")
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert "js/blob.js" not in labels(ctx)

    def test_dependency_directories_are_skipped(self, project):
        vendor = project / "node_modules"
        vendor.mkdir()
        (vendor / "dep.js").write_text("require('js/app.js')", encoding="utf-8")
        ctx = codegen.gather_context(project, project / "js" / "app.js")
        assert not any("node_modules" in lbl for lbl in labels(ctx))


# ---------------------------------------------------------------------------
# what reaches the model
# ---------------------------------------------------------------------------

class TestGenerationPrompt:
    def _generate(self, project, client, path="js/app.js", **extra):
        registry = build_default_registry(project, ar.SimulatedShell(), client=client)
        ex = MultiToolExecutor(registry, project)
        args = {"path": path, "spec": "wire the calculator UI", **extra}
        return run(ex.execute(ar.ToolCall(tool="generate_file", args=args), 30))

    def test_real_ids_reach_the_generator(self, project):
        client = _Recorder()
        assert self._generate(project, client).ok
        prompt = client.users[0]
        assert "historyList" in prompt and "btnClearHistory" in prompt

    def test_the_invented_ids_are_absent(self, project):
        client = _Recorder()
        self._generate(project, client)
        assert "history-list" not in client.users[0]

    def test_the_grounding_is_marked_authoritative(self, project):
        client = _Recorder()
        self._generate(project, client)
        assert "authoritative" in client.users[0]
        assert "do not invent alternatives" in client.users[0]

    def test_the_spec_still_reaches_the_model(self, project):
        client = _Recorder()
        self._generate(project, client)
        assert "wire the calculator UI" in client.users[0]

    def test_caller_supplied_context_is_kept_too(self, project):
        client = _Recorder()
        self._generate(project, client, context="prefer const over var")
        assert "prefer const over var" in client.users[0]

    def test_a_blocked_path_never_reaches_the_model(self, project):
        client = _Recorder()
        result = self._generate(project, client, path="../evil.js")
        assert not result.ok and client.users == []

    def test_generation_still_writes_the_file(self, project):
        client = _Recorder("console.log('ok');\n")
        assert self._generate(project, client).ok
        assert (project / "js" / "app.js").read_text() == "console.log('ok');\n"

    def test_a_greenfield_file_has_no_grounding_section(self, tmp_path):
        client = _Recorder("x = 1\n")
        registry = build_default_registry(tmp_path, ar.SimulatedShell(), client=client)
        ex = MultiToolExecutor(registry, tmp_path)
        run(ex.execute(ar.ToolCall(tool="generate_file",
                                   args={"path": "main.py", "spec": "entry point"}), 30))
        assert "EXISTING PROJECT FILES" not in client.users[0]


# ---------------------------------------------------------------------------
# steering away from regenerating what already exists
# ---------------------------------------------------------------------------

class TestEditPreference:
    def test_policy_prefers_edit_for_existing_files(self, tmp_path):
        from omni.agentkit.policy import ToolCallPolicy
        registry = build_default_registry(tmp_path, ar.SimulatedShell(),
                                          client=object())
        policy = ToolCallPolicy(client=None, ledger=ar.Ledger(budget=ar.Budget()),
                                registry=registry)
        assert "use edit_file, not generate_file" in policy.SYSTEM

    def test_tool_description_says_new_file(self, tmp_path):
        registry = build_default_registry(tmp_path, ar.SimulatedShell(),
                                          client=object())
        assert "NEW file" in registry.get("generate_file").description

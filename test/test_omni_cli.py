"""
Tests for the CLI-level pieces: tool-aware plan steps, the repository survey
that feeds the review handler, and plan execution through PlannerLoop.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omni import runtime as ar
from omni import cli as omni_cli
from omni.agentkit.stack import build_agent_stack
from omni.agentkit.survey import collect_digest
from omni.agentkit.tools import build_default_registry


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# plan steps
# ---------------------------------------------------------------------------

class TestPlanStep:
    def test_legacy_command_step_is_unchanged(self):
        step = ar.PlanStep("List", "ls")
        assert step.tool == "shell"
        assert step.to_args() == {"command": "ls"}
        assert step.label == "ls"

    def test_tool_step_carries_arguments(self):
        step = ar.PlanStep("Create", tool="write_file",
                           args={"path": "a.py", "content": "x"})
        assert step.to_args() == {"path": "a.py", "content": "x"}
        assert step.to_call().tool == "write_file"
        assert "write_file(" in step.label


class TestSmartPlannerParsing:
    @pytest.fixture
    def planner(self, tmp_path):
        registry = build_default_registry(tmp_path, ar.SimulatedShell())
        return omni_cli.SmartPlanner(client=None, registry=registry)

    def test_tool_step_is_parsed(self, planner):
        step = planner._to_step({"title": "Create app", "tool": "write_file",
                                 "args": {"path": "app.py", "content": "x"}})
        assert step.tool == "write_file" and step.args["path"] == "app.py"

    def test_legacy_command_step_is_parsed(self, planner):
        step = planner._to_step({"title": "Install", "command": "npm install"})
        assert step.tool == "shell" and step.command == "npm install"

    def test_shell_tool_with_command_in_args_is_normalised(self, planner):
        step = planner._to_step({"title": "Run", "tool": "run_command",
                                 "args": {"command": "pytest -q"}})
        assert step.command == "pytest -q"

    def test_unknown_tool_is_dropped(self, planner):
        assert planner._to_step({"title": "X", "tool": "teleport",
                                 "args": {"a": 1}}) is None

    def test_empty_step_is_dropped(self, planner):
        assert planner._to_step({"title": "X"}) is None

    def test_schema_enumerates_registered_tools(self, planner):
        schema = planner._schema()
        enum = schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"]
        assert "write_file" in enum and "read_file" in enum

    def test_system_prompt_steers_away_from_inlining_files(self, planner):
        """A plan that inlines file contents blows the token cap and is discarded."""
        assert "generate_file" in planner.SYSTEM
        assert "Never put a whole file's source into this plan" in planner.SYSTEM
        assert "write_file" in planner.SYSTEM

    def test_fallback_plan_uses_a_tool_not_a_shell_listing(self, tmp_path):
        """When the model call fails, the fallback should still use real tools."""
        class _Broken:
            async def complete(self, *a, **kw):
                raise RuntimeError("backend down")

        registry = build_default_registry(tmp_path, ar.SimulatedShell())
        planner = omni_cli.SmartPlanner(client=_Broken(), registry=registry)
        plan = run(planner.plan("anything"))
        assert plan[0].tool == "list_dir"

    def test_no_client_uses_the_runtime_default_plan(self, planner):
        assert run(planner.plan("anything")) == ar.default_plan("anything")


# ---------------------------------------------------------------------------
# plan execution
# ---------------------------------------------------------------------------

class TestPlannerLoopWithTools:
    def test_a_plan_can_create_a_file(self, tmp_path):
        """The capability that did not exist: a plan step that writes."""
        ar.set_quiet(True)
        shell = ar.SubprocessShell(workspace=tmp_path)
        stack = build_agent_stack(tmp_path, shell, auto_approve_writes=True,
                                  verify=False)
        loop = ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_p", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            repair_factory=lambda cmd: None)
        plan = [ar.PlanStep("Create module", tool="write_file",
                            args={"path": "mod.py", "content": "VALUE = 42\n"})]
        outcome = run(loop.run(goal="scaffold", plan=plan,
                               ledger=ar.Ledger(budget=ar.Budget())))
        assert outcome.succeeded, outcome.detail
        assert (tmp_path / "mod.py").read_text() == "VALUE = 42\n"

    def test_write_without_approval_suspends(self, tmp_path):
        ar.set_quiet(True)
        shell = ar.SubprocessShell(workspace=tmp_path)
        stack = build_agent_stack(tmp_path, shell, verify=False)   # writes ELEVATED
        loop = ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_p", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            repair_factory=lambda cmd: None)
        plan = [ar.PlanStep("Create module", tool="write_file",
                            args={"path": "mod.py", "content": "x"})]
        outcome = run(loop.run(goal="scaffold", plan=plan,
                               ledger=ar.Ledger(budget=ar.Budget())))
        assert outcome.stop_reason is ar.StopReason.APPROVAL_REQUIRED
        assert "write_file" in (outcome.pending_command or "")
        assert not (tmp_path / "mod.py").exists()

    def test_escaping_plan_step_is_blocked_not_executed(self, tmp_path):
        ar.set_quiet(True)
        shell = ar.SubprocessShell(workspace=tmp_path)
        stack = build_agent_stack(tmp_path, shell, auto_approve_writes=True,
                                  verify=False)

        def give_up(command):
            """A scoped repair loop that declines rather than looping."""
            return ar.RepairLoop(
                executor=stack.executor,
                policy=ar.ScriptedPolicy(
                    [ar.Finish(succeeded=False, summary="cannot repair a jailbreak")]),
                command_policy=ar.CommandPolicy(tmp_path),
                journal=ar.RunJournal("run_r", path=tmp_path / "j.jsonl"),
                workspace=tmp_path, tool_policy=stack.tool_policy)

        loop = ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_p", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            repair_factory=give_up)
        plan = [ar.PlanStep("Escape", tool="write_file",
                            args={"path": "../evil.py", "content": "x"})]
        outcome = run(loop.run(goal="escape", plan=plan,
                               ledger=ar.Ledger(budget=ar.Budget())))
        assert not outcome.succeeded
        assert not (tmp_path.parent / "evil.py").exists()
        blocked = outcome.steps[0]
        assert blocked.result.error_class is ar.ErrorClass.POLICY


# ---------------------------------------------------------------------------
# a failed inspection is a finding, not a blocker
# ---------------------------------------------------------------------------
#
# Regression: the planner guesses at filenames. One plan opened with "inspect
# package.json" in a project that has none; the step failed, the scoped repair
# could not conjure the file, and the whole run was abandoned — along with the
# deliverables an earlier phase had already produced.

class TestFailedInspectionSteps:
    def _loop(self, tmp_path, stack, repair_factory=None):
        ar.set_quiet(True)
        return ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_i", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            is_mutating=stack.is_mutating,
            repair_factory=repair_factory or (lambda cmd: None))

    @pytest.fixture
    def stack(self, tmp_path):
        (tmp_path / "index.html").write_text("<html></html>")
        return build_agent_stack(tmp_path, ar.SubprocessShell(workspace=tmp_path),
                                 auto_approve_writes=True, verify=False)

    def test_missing_inspection_target_does_not_end_the_run(self, tmp_path, stack):
        plan = [
            ar.PlanStep("Inspect root", tool="list_dir", args={"path": "."}),
            ar.PlanStep("Inspect package.json", tool="read_file",
                        args={"path": "package.json"}),
            ar.PlanStep("Write README", tool="write_file",
                        args={"path": "README.md", "content": "# Project\n"}),
        ]
        outcome = run(self._loop(tmp_path, stack).run(
            goal="create a readme", plan=plan, ledger=ar.Ledger(budget=ar.Budget())))
        assert outcome.succeeded, outcome.detail

    def test_the_later_steps_still_run(self, tmp_path, stack):
        plan = [
            ar.PlanStep("Inspect package.json", tool="read_file",
                        args={"path": "package.json"}),
            ar.PlanStep("Write README", tool="write_file",
                        args={"path": "README.md", "content": "# Project\n"}),
        ]
        run(self._loop(tmp_path, stack).run(
            goal="g", plan=plan, ledger=ar.Ledger(budget=ar.Budget())))
        assert (tmp_path / "README.md").exists()

    def test_the_failure_is_still_recorded(self, tmp_path, stack):
        plan = [ar.PlanStep("Inspect package.json", tool="read_file",
                            args={"path": "package.json"})]
        outcome = run(self._loop(tmp_path, stack).run(
            goal="g", plan=plan, ledger=ar.Ledger(budget=ar.Budget())))
        assert outcome.steps[0].result.error_class is ar.ErrorClass.MISSING_PATH

    def test_a_failed_write_still_opens_repair(self, tmp_path, stack):
        """Only inspections are waved through; actions still get repaired."""
        opened: list[str] = []

        def repair_factory(command):
            opened.append(command)
            return ar.RepairLoop(
                executor=stack.executor,
                policy=ar.ScriptedPolicy([ar.Finish(succeeded=False, summary="no")]),
                command_policy=ar.CommandPolicy(tmp_path),
                journal=ar.RunJournal("run_r", path=tmp_path / "j.jsonl"),
                workspace=tmp_path, tool_policy=stack.tool_policy)

        plan = [ar.PlanStep("Escape", tool="write_file",
                            args={"path": "../evil.py", "content": "x"})]
        run(self._loop(tmp_path, stack, repair_factory).run(
            goal="g", plan=plan, ledger=ar.Ledger(budget=ar.Budget())))
        assert opened, "a failed mutating step must still open a repair loop"

    def test_without_the_predicate_behaviour_is_unchanged(self, tmp_path, stack):
        """Callers that do not pass `is_mutating` keep the old semantics."""
        loop = ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_i", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            repair_factory=lambda cmd: ar.RepairLoop(
                executor=stack.executor,
                policy=ar.ScriptedPolicy([ar.Finish(succeeded=False, summary="no")]),
                command_policy=ar.CommandPolicy(tmp_path),
                journal=ar.RunJournal("r2", path=tmp_path / "j.jsonl"),
                workspace=tmp_path))
        plan = [ar.PlanStep("Inspect package.json", tool="read_file",
                            args={"path": "package.json"})]
        outcome = run(loop.run(goal="g", plan=plan,
                               ledger=ar.Ledger(budget=ar.Budget())))
        assert not outcome.succeeded


class TestIsMutating:
    @pytest.fixture
    def stack(self, tmp_path):
        return build_agent_stack(tmp_path, ar.SubprocessShell(workspace=tmp_path),
                                 verify=False)

    @pytest.mark.parametrize("tool", ["read_file", "list_dir", "search_files"])
    def test_reads_are_not_mutating(self, stack, tool):
        assert not stack.is_mutating(ar.ToolCall(tool=tool, args={}))

    @pytest.mark.parametrize("tool", ["write_file", "edit_file", "run_command"])
    def test_writes_and_commands_are_mutating(self, stack, tool):
        assert stack.is_mutating(ar.ToolCall(tool=tool, args={}))

    def test_unknown_tools_are_treated_as_mutating(self, stack):
        """The cautious answer when we cannot tell."""
        assert stack.is_mutating(ar.ToolCall(tool="teleport", args={}))


# ---------------------------------------------------------------------------
# repository survey
# ---------------------------------------------------------------------------

class TestSurvey:
    def _repo(self, tmp_path: Path) -> Path:
        (tmp_path / "README.md").write_text("# Demo project\nDoes a thing.\n")
        (tmp_path / "main.py").write_text("def main():\n    return 1\n")
        (tmp_path / "helpers.py").write_text("X = 1\n" * 50)
        pkg = tmp_path / "venv" / "lib"
        pkg.mkdir(parents=True)
        (pkg / "dependency.py").write_text("SHOULD_NOT_APPEAR = True\n")
        (tmp_path / "logo.png").write_bytes(b"\x00\x01binary")
        return tmp_path

    def test_reads_real_source_not_just_names(self, tmp_path):
        digest = collect_digest(self._repo(tmp_path))
        joined = digest.sources_text()
        assert "def main():" in joined
        assert "# Demo project" in joined

    def test_dependency_directories_are_excluded(self, tmp_path):
        digest = collect_digest(self._repo(tmp_path))
        assert "SHOULD_NOT_APPEAR" not in digest.sources_text()
        assert all("venv" not in path for path, _ in digest.files)

    def test_readme_is_ranked_first(self, tmp_path):
        digest = collect_digest(self._repo(tmp_path))
        assert digest.files[0][0] == "README.md"

    def test_binaries_are_not_read(self, tmp_path):
        digest = collect_digest(self._repo(tmp_path))
        assert all(not path.endswith(".png") for path, _ in digest.files)

    def test_character_budget_is_respected(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "huge.py").write_text("Y = 2\n" * 20_000)
        digest = collect_digest(repo, max_chars=5_000, per_file_chars=2_000)
        assert sum(len(t) for _, t in digest.files) <= 5_000 + 200

    def test_clipped_file_says_so(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "huge.py").write_text("Y = 2\n" * 5_000)
        digest = collect_digest(repo, per_file_chars=500)
        clipped = [t for p, t in digest.files if p == "huge.py"]
        assert clipped and "chars elided" in clipped[0]

    def test_empty_workspace_is_safe(self, tmp_path):
        digest = collect_digest(tmp_path)
        assert digest.files == [] and digest.tree_text() == "(empty)"

    def test_missing_workspace_is_safe(self, tmp_path):
        digest = collect_digest(tmp_path / "nope")
        assert digest.total_files == 0


# ---------------------------------------------------------------------------
# operator approval
# ---------------------------------------------------------------------------

class TestGrantTokens:
    """A granted approval must satisfy the check the loops actually run."""

    def _verdict_for(self, tmp_path, call):
        return build_default_registry(tmp_path, ar.SimulatedShell()).policy_for(call)

    def test_tool_call_approval_is_accepted(self, tmp_path):
        call = ar.ToolCall(tool="write_file", args={"path": "a.py", "content": "x"})
        verdict = self._verdict_for(tmp_path, call)
        pending = ar.describe_call(call)
        assert ar.approval_grants(verdict, omni_cli.grant_tokens(pending))

    def test_tool_name_is_among_the_tokens(self):
        tokens = omni_cli.grant_tokens("write_file(path='a.py', content='x')")
        assert "write_file" in tokens

    def test_shell_command_approval_is_accepted(self, tmp_path):
        verdict = ar.CommandPolicy(tmp_path).classify("rm build")
        assert verdict.risk is ar.Risk.ELEVATED
        assert ar.approval_grants(verdict, omni_cli.grant_tokens("rm build"))

    def test_sudo_command_grants_sudo(self):
        assert "sudo" in omni_cli.grant_tokens("sudo npm publish")

    def test_unbalanced_quotes_do_not_raise(self):
        assert omni_cli.grant_tokens("echo 'unclosed") == ["echo 'unclosed"]

    def test_empty_pending_is_empty(self):
        assert omni_cli.grant_tokens("") == []


class TestStepTable:
    def test_tool_steps_are_rendered_not_blank(self):
        call = ar.ToolCall(tool="write_file", args={"path": "a.py", "content": "x"})
        step = ar.StepRecord(1, "t", call,
                             ar.ToolResult(call_id=call.call_id, ok=True,
                                           exit_code=0, output=""))
        table = omni_cli.format_step_table([step])
        rendered = [c._cells for c in table.columns]
        assert any("write_file" in str(cell) for col in rendered for cell in col)


# ---------------------------------------------------------------------------
# multi-file generation (regression: test-01 produced 2 of 6 files)
# ---------------------------------------------------------------------------

TREE_BLOCK = """test-01/
├── index.html
├── styles.css
└── js/
    └── app.js
"""

MULTI_FILE_RESPONSE = f"""# Scientific Web Calculator

Project Structure

```
{TREE_BLOCK}```

---

index.html

```html
<!DOCTYPE html><html><body>hi</body></html>
```

---

**styles.css**

```css
:root {{ --bg: #101418; }}
```

---

`js/app.js`

```javascript
console.log("go");
```
"""


class TestExtractCodeBlocks:
    def test_every_block_is_kept(self):
        blocks = omni_cli.extract_code_blocks(MULTI_FILE_RESPONSE)
        assert len(blocks) == 3

    def test_filenames_come_from_the_heading_above_the_fence(self):
        names = [n for n, _ in omni_cli.extract_code_blocks(MULTI_FILE_RESPONSE)]
        assert names == ["index.html", "styles.css", "js/app.js"]

    def test_directory_tree_block_is_not_saved_as_a_file(self):
        names = [n for n, _ in omni_cli.extract_code_blocks(MULTI_FILE_RESPONSE)]
        assert not any(n.endswith(".txt") for n in names)
        assert all("index.html" not in code or "<!DOCTYPE" in code
                   for _, code in omni_cli.extract_code_blocks(MULTI_FILE_RESPONSE))

    def test_bold_and_backticked_headings_are_both_read(self):
        blocks = dict(omni_cli.extract_code_blocks(MULTI_FILE_RESPONSE))
        assert "--bg" in blocks["styles.css"]
        assert "console.log" in blocks["js/app.js"]

    def test_filename_comment_is_still_honoured(self):
        text = "```python\n# filename: tool.py\nx = 1\n```"
        assert omni_cli.extract_code_blocks(text)[0][0] == "tool.py"

    def test_generic_name_when_nothing_identifies_the_block(self):
        text = "Here you go:\n\n```python\nx = 1\n```"
        assert omni_cli.extract_code_blocks(text)[0][0] == "script.py"

    def test_unterminated_block_is_not_returned(self):
        """A truncated final block has incomplete content; saving it is worse."""
        text = ("a.py\n\n```python\nx = 1\n```\n\n"
                "b.py\n\n```python\ny = 2\n")
        names = [n for n, _ in omni_cli.extract_code_blocks(text)]
        assert names == ["a.py"]

    def test_truncation_is_detectable(self):
        assert omni_cli.response_was_truncated("```py\nx = 1\n")
        assert not omni_cli.response_was_truncated("```py\nx = 1\n```")


class TestProjectRouting:
    @pytest.mark.parametrize("prompt", [
        "create a java script project for scientific web calculation",
        "build a fastapi app",
        "scaffold a react website",
        "set up a python project with tests",
        "make a dashboard application",
    ])
    def test_multi_file_requests_are_agent_tasks(self, prompt):
        assert omni_cli.is_project_request(prompt)

    @pytest.mark.parametrize("prompt", [
        "create a script to add two numbers",
        "write a python script to parse my project config",
        "make a script that renames files",
        "give me code for a binary search",
        "explain how a project scaffold works",
    ])
    def test_single_file_requests_stay_one_shot(self, prompt):
        assert not omni_cli.is_project_request(prompt)

    def test_the_reported_prompt_routes_to_execute_task(self):
        prompt = ("create a java script project for scientific web calculation . "
                  "which save calculation on sidebar history for each session")
        assert run(omni_cli.classify_intent(prompt, None)) == "EXECUTE_TASK"

    def test_simple_script_still_routes_to_direct_code(self):
        assert run(omni_cli.classify_intent(
            "write a python script to add 2 numbers", None)) == "DIRECT_CODE"

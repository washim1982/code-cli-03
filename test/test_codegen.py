"""
Tests for `generate_file` and truncation detection.

Regression context: a `create a javascript project` run produced a ~10 KB plan
that hit the token cap mid-string. The JSON would not parse, the planner fell
back to an unrelated exploration plan whose `git status` step failed in a
non-repo workspace, and the scoped repair loop burned its whole budget on that
irrelevant failure. The user's request was never attempted.

`generate_file` removes the cause: the plan carries a path and a sentence per
file instead of the files themselves.
"""

from __future__ import annotations

import asyncio

import pytest

from omni import runtime as ar
from omni import backends as llm_backends
from omni import cli as omni_cli
from omni.agentkit.dispatch import MultiToolExecutor
from omni.agentkit.tools import build_default_registry, codegen


def run(coro):
    return asyncio.run(coro)


def call(tool: str, **args) -> ar.ToolCall:
    return ar.ToolCall(tool=tool, args=args)


class _GenClient:
    """Canned backend that records what it was asked for."""

    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text = text
        self.finish_reason = finish_reason
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system, user, *, schema=None, max_tokens=512):
        self.calls.append((system, user))
        return llm_backends.Completion(
            text=self.text, model="t", backend=llm_backends.Backend.OLLAMA,
            finish_reason=self.finish_reason)


class _Finish:
    def __init__(self, reason: str) -> None:
        self.finish_reason = reason


# ---------------------------------------------------------------------------
# truncation detection
# ---------------------------------------------------------------------------

class TestOutputTruncated:
    def test_length_is_truncation(self):
        assert codegen.output_truncated(_Finish("length"))

    def test_max_tokens_is_truncation(self):
        assert codegen.output_truncated(_Finish("max_tokens"))

    def test_stop_is_not(self):
        assert not codegen.output_truncated(_Finish("stop"))

    def test_missing_attribute_is_not(self):
        assert not codegen.output_truncated(object())

    def test_prompt_truncation_flag_is_a_different_signal(self):
        """`truncation_suspected` means the PROMPT was cut, not the output."""
        completion = llm_backends.Completion(
            text="x", model="t", backend=llm_backends.Backend.OLLAMA,
            finish_reason="length", truncation_suspected=False)
        assert codegen.output_truncated(completion)
        assert not completion.truncation_suspected


# ---------------------------------------------------------------------------
# fence stripping
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_plain_content_untouched(self):
        assert codegen.strip_fences("x = 1\n") == "x = 1\n"

    def test_wrapping_fence_removed(self):
        assert codegen.strip_fences("```python\nx = 1\n```").strip() == "x = 1"

    def test_language_free_fence_removed(self):
        assert codegen.strip_fences("```\nhello\n```").strip() == "hello"

    def test_inner_fences_preserved_in_markdown(self):
        text = "# Doc\n\n```js\nlet a = 1;\n```\n"
        assert "```js" in codegen.strip_fences(text)


# ---------------------------------------------------------------------------
# generate_file
# ---------------------------------------------------------------------------

class TestGenerateFile:
    def _registry(self, tmp_path, client):
        return build_default_registry(tmp_path, ar.SimulatedShell(), client=client)

    def _executor(self, tmp_path, client):
        return MultiToolExecutor(self._registry(tmp_path, client), tmp_path)

    def test_registered_only_when_a_client_is_present(self, tmp_path):
        assert "generate_file" not in build_default_registry(
            tmp_path, ar.SimulatedShell())
        assert "generate_file" in self._registry(tmp_path, _GenClient("x"))

    def test_writes_generated_content(self, tmp_path):
        ex = self._executor(tmp_path, _GenClient("console.log('hi');\n"))
        result = run(ex.execute(call("generate_file", path="js/app.js",
                                     spec="log hi"), 30))
        assert result.ok, result.output
        assert (tmp_path / "js" / "app.js").read_text() == "console.log('hi');\n"

    def test_fences_are_stripped_before_writing(self, tmp_path):
        ex = self._executor(tmp_path, _GenClient("```javascript\nlet a = 1;\n```"))
        run(ex.execute(call("generate_file", path="a.js", spec="x"), 30))
        assert not (tmp_path / "a.js").read_text().startswith("```")

    def test_escaping_path_refused_without_spending_a_completion(self, tmp_path):
        client = _GenClient("x")
        ex = MultiToolExecutor(self._registry(tmp_path, client), tmp_path)
        result = run(ex.execute(call("generate_file", path="../evil.js",
                                     spec="x"), 30))
        assert not result.ok and result.error_class is ar.ErrorClass.POLICY
        assert client.calls == [], "must not call the model for a blocked path"
        assert not (tmp_path.parent / "evil.js").exists()

    def test_truncated_generation_is_reported_not_silently_ok(self, tmp_path):
        ex = self._executor(tmp_path, _GenClient("let a = 1;", "length"))
        result = run(ex.execute(call("generate_file", path="a.js", spec="x"), 30))
        assert result.ok and "incomplete" in result.output

    def test_empty_generation_is_a_failure(self, tmp_path):
        ex = self._executor(tmp_path, _GenClient("   "))
        result = run(ex.execute(call("generate_file", path="a.js", spec="x"), 30))
        assert not result.ok
        assert not (tmp_path / "a.js").exists()

    def test_backend_error_becomes_an_observation(self, tmp_path):
        class _Broken:
            async def complete(self, *a, **kw):
                raise RuntimeError("backend down")

        ex = self._executor(tmp_path, _Broken())
        result = run(ex.execute(call("generate_file", path="a.js", spec="x"), 30))
        assert not result.ok and "backend down" in result.output

    def test_is_mutating_so_the_verify_gate_triggers(self, tmp_path):
        reg = self._registry(tmp_path, _GenClient("x = 1"))
        assert reg.get("generate_file").mutating is True

    def test_spec_reaches_the_model(self, tmp_path):
        client = _GenClient("x")
        ex = MultiToolExecutor(self._registry(tmp_path, client), tmp_path)
        run(ex.execute(call("generate_file", path="a.js",
                            spec="a sidebar that persists to sessionStorage"), 30))
        _system, user = client.calls[0]
        assert "sessionStorage" in user and "a.js" in user


# ---------------------------------------------------------------------------
# planner fallback
# ---------------------------------------------------------------------------

class TestPlannerFallback:
    """The fallback must not manufacture a failure the repair loop then chases."""

    def test_fallback_never_runs_git_status(self, tmp_path):
        registry = build_default_registry(tmp_path, ar.SimulatedShell())
        planner = omni_cli.SmartPlanner(client=None, registry=registry)
        steps = planner._fallback_plan("create a javascript project")
        assert all("git" not in (s.command or "") for s in steps)

    def test_fallback_attempts_the_goal_when_generation_is_available(self, tmp_path):
        registry = build_default_registry(tmp_path, ar.SimulatedShell(),
                                          client=_GenClient("x"))
        planner = omni_cli.SmartPlanner(client=_GenClient("x"), registry=registry)
        steps = planner._fallback_plan("create a scientific calculator")
        assert steps[0].tool == "generate_file"
        assert "scientific calculator" in steps[0].args["spec"]

    def test_fallback_without_generation_only_inspects(self, tmp_path):
        registry = build_default_registry(tmp_path, ar.SimulatedShell())
        planner = omni_cli.SmartPlanner(client=None, registry=registry)
        steps = planner._fallback_plan("anything")
        assert steps[0].tool == "list_dir"

    def test_truncated_plan_falls_back_without_raising(self, tmp_path):
        """A plan cut off mid-string must not abort the turn."""
        truncated = '{"steps": [{"title": "a", "tool": "generate_file", "args": {"pa'
        registry = build_default_registry(tmp_path, ar.SimulatedShell(),
                                          client=_GenClient(truncated, "length"))
        planner = omni_cli.SmartPlanner(
            client=_GenClient(truncated, "length"), registry=registry)
        steps = run(planner.plan("create a project"))
        assert steps and steps[0].tool == "generate_file"


# ---------------------------------------------------------------------------
# end to end: a scaffold plan that does not inline file contents
# ---------------------------------------------------------------------------

class TestScaffoldPlan:
    def test_plan_of_generate_steps_creates_every_file(self, tmp_path):
        ar.set_quiet(True)
        from omni.agentkit.stack import build_agent_stack

        client = _GenClient("/* generated */\n")
        shell = ar.SubprocessShell(workspace=tmp_path)
        stack = build_agent_stack(tmp_path, shell, client=client,
                                  auto_approve_writes=True, verify=False)
        loop = ar.PlannerLoop(
            executor=stack.executor, command_policy=ar.CommandPolicy(tmp_path),
            journal=ar.RunJournal("run_s", path=tmp_path / "j.jsonl"),
            workspace=tmp_path, tool_policy=stack.tool_policy,
            repair_factory=lambda cmd: None)
        plan = [
            ar.PlanStep("index", tool="generate_file",
                        args={"path": "index.html", "spec": "calculator shell"}),
            ar.PlanStep("styles", tool="generate_file",
                        args={"path": "styles.css", "spec": "dark theme"}),
            ar.PlanStep("history", tool="generate_file",
                        args={"path": "js/history.js", "spec": "sessionStorage"}),
        ]
        outcome = run(loop.run(goal="scaffold", plan=plan,
                               ledger=ar.Ledger(budget=ar.Budget())))
        assert outcome.succeeded, outcome.detail
        for rel in ("index.html", "styles.css", "js/history.js"):
            assert (tmp_path / rel).exists(), f"{rel} was not created"
        assert len(client.calls) == 3, "one completion per file, not one for the plan"

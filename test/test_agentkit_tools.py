"""
Tests for the tool layer: registry, filesystem tools, dispatch, tool-calling
policy, verification detection, and the verify gate.

The golden-task tests at the bottom drive a real `RepairLoop` over a real
`SubprocessShell` with a scripted policy, which is the only way to check that
the pieces compose the way the loops expect.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from omni import runtime as ar
from omni.agentkit.dispatch import MultiToolExecutor
from omni.agentkit.gate import VerifyGate
from omni.agentkit.policy import ToolCallPolicy
from omni.agentkit.registry import (
    SHELL_TOOL,
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
    validate_args,
)
from omni.agentkit.stack import build_agent_stack
from omni.agentkit.tools import build_default_registry, fs
from omni.agentkit.verify import COMPILEALL, NPM_TEST, PYTEST, VerifySpec, detect_verify


def run(coro):
    return asyncio.run(coro)


def call(tool: str, **args) -> ar.ToolCall:
    return ar.ToolCall(tool=tool, args=args)


@pytest.fixture
def registry(tmp_path) -> ToolRegistry:
    return build_default_registry(tmp_path, ar.SimulatedShell())


@pytest.fixture
def executor(tmp_path, registry) -> MultiToolExecutor:
    return MultiToolExecutor(registry, tmp_path, shell_executor=ar.SimulatedShell())


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_default_registry_has_the_five_fs_tools_and_shell(self, registry):
        assert set(registry.names()) == {
            "read_file", "write_file", "edit_file", "list_dir", "search_files",
            SHELL_TOOL,
        }

    def test_duplicate_registration_is_refused(self, registry):
        spec = registry.get("read_file")
        with pytest.raises(ValueError):
            registry.register(spec)

    def test_unknown_tool_is_forbidden(self, registry):
        verdict = registry.policy_for(call("teleport", x=1))
        assert verdict.risk is ar.Risk.FORBIDDEN
        assert "unknown tool" in verdict.reason

    def test_shell_defers_to_command_policy(self, registry):
        """Returning None is what keeps the whole existing perimeter in force."""
        assert registry.policy_for(call(SHELL_TOOL, command="ls")) is None

    def test_reads_are_safe_and_writes_are_elevated(self, registry):
        assert registry.policy_for(call("read_file", path="a.py")).risk is ar.Risk.SAFE
        assert registry.policy_for(
            call("write_file", path="a.py", content="x")).risk is ar.Risk.ELEVATED

    def test_auto_approve_writes_downgrades_only_risk(self, tmp_path):
        reg = build_default_registry(tmp_path, ar.SimulatedShell(),
                                     auto_approve_writes=True)
        assert reg.policy_for(
            call("write_file", path="a.py", content="x")).risk is ar.Risk.SAFE

    def test_missing_required_argument_is_forbidden(self, registry):
        verdict = registry.policy_for(call("read_file"))
        assert verdict.risk is ar.Risk.FORBIDDEN
        assert "missing required argument" in verdict.reason

    def test_unknown_argument_is_forbidden(self, registry):
        verdict = registry.policy_for(call("read_file", path="a.py", mode="rb"))
        assert verdict.risk is ar.Risk.FORBIDDEN
        assert "unknown argument" in verdict.reason

    def test_wrong_type_is_forbidden(self, registry):
        verdict = registry.policy_for(call("read_file", path="a.py", offset="two"))
        assert verdict.risk is ar.Risk.FORBIDDEN
        assert "must be integer" in verdict.reason

    def test_bool_is_not_an_integer(self, registry):
        spec = registry.get("read_file")
        assert validate_args(spec, {"path": "a.py", "offset": True})

    def test_argv_is_never_treated_as_a_model_argument(self, registry):
        spec = registry.get("read_file")
        assert validate_args(spec, {"path": "a.py", "argv": ["x"]}) == []

    def test_decision_schema_enumerates_tools(self, registry):
        schema = registry.decision_schema()
        act = schema["oneOf"][0]
        assert act["properties"]["kind"]["const"] == "act"
        assert set(act["properties"]["tool"]["enum"]) == set(registry.names())

    def test_prompt_fragment_names_every_tool_and_flags_approval(self, registry):
        rendered = registry.render_for_prompt()
        for name in registry.names():
            assert name in rendered
        assert "requires approval" in rendered

    def test_prompt_fragment_is_byte_stable(self, tmp_path):
        """LLMPolicy relies on a stable system prompt for KV-cache reuse."""
        a = build_default_registry(tmp_path, ar.SimulatedShell())
        b = build_default_registry(tmp_path, ar.SimulatedShell())
        assert a.render_for_prompt() == b.render_for_prompt()

    def test_approval_can_be_granted_by_tool_name(self, registry):
        verdict = registry.policy_for(call("write_file", path="a.py", content="x"))
        assert ar.approval_grants(verdict, ["write_file"])
        assert not ar.approval_grants(verdict, ["read_file"])


# ---------------------------------------------------------------------------
# filesystem tools
# ---------------------------------------------------------------------------

class TestFileTools:
    @pytest.fixture
    def tools(self, tmp_path) -> fs.FileTools:
        return fs.FileTools(tmp_path)

    def test_write_then_read_round_trip(self, tools, tmp_path):
        out = run(tools.write_file("a/b.py", "print(1)\n"))
        assert out.ok and (tmp_path / "a" / "b.py").exists()
        out = run(tools.read_file("a/b.py"))
        assert out.ok and "print(1)" in out.output

    def test_read_is_line_numbered(self, tools):
        run(tools.write_file("x.py", "one\ntwo\nthree\n"))
        out = run(tools.read_file("x.py"))
        assert "1\tone" in out.output and "3\tthree" in out.output

    def test_read_paging(self, tools):
        run(tools.write_file("x.py", "".join(f"line{i}\n" for i in range(1, 11))))
        out = run(tools.read_file("x.py", offset=3, limit=2))
        assert "line3" in out.output and "line4" in out.output
        assert "line5" not in out.output
        assert "offset=5" in out.output          # tells the model how to continue

    def test_write_outside_workspace_is_blocked(self, tools, tmp_path):
        out = run(tools.write_file("../escape.py", "x"))
        assert not out.ok and out.exit_code == 126
        assert not (tmp_path.parent / "escape.py").exists()

    def test_absolute_path_write_is_blocked(self, tools):
        out = run(tools.write_file("C:/Windows/Temp/evil.py", "x"))
        assert not out.ok and out.error_class is ar.ErrorClass.POLICY

    def test_read_missing_file(self, tools):
        out = run(tools.read_file("nope.py"))
        assert not out.ok and out.error_class is ar.ErrorClass.MISSING_PATH

    def test_read_directory_is_redirected_to_list_dir(self, tools, tmp_path):
        (tmp_path / "sub").mkdir()
        out = run(tools.read_file("sub"))
        assert not out.ok and "list_dir" in out.output

    def test_binary_is_refused(self, tools, tmp_path):
        (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02binary")
        out = run(tools.read_file("blob.bin"))
        assert not out.ok and "binary" in out.output

    def test_edit_unique_snippet(self, tools):
        run(tools.write_file("m.py", "a = 1\nb = 2\n"))
        out = run(tools.edit_file("m.py", "b = 2", "b = 3"))
        assert out.ok
        assert "b = 3" in run(tools.read_file("m.py")).output

    def test_edit_refuses_ambiguous_snippet_and_leaves_file_intact(self, tools, tmp_path):
        run(tools.write_file("m.py", "x = 1\nx = 1\n"))
        out = run(tools.edit_file("m.py", "x = 1", "x = 2"))
        assert not out.ok and "2 occurrences" in out.output
        assert (tmp_path / "m.py").read_text() == "x = 1\nx = 1\n"

    def test_edit_replace_all_is_opt_in(self, tools, tmp_path):
        run(tools.write_file("m.py", "x = 1\nx = 1\n"))
        out = run(tools.edit_file("m.py", "x = 1", "x = 2", replace_all=True))
        assert out.ok
        assert (tmp_path / "m.py").read_text() == "x = 2\nx = 2\n"

    def test_edit_missing_snippet_reports_clearly(self, tools):
        run(tools.write_file("m.py", "a = 1\n"))
        out = run(tools.edit_file("m.py", "zzz", "yyy"))
        assert not out.ok and "no occurrence" in out.output

    def test_edit_outside_workspace_is_blocked(self, tools):
        out = run(tools.edit_file("../../x.py", "a", "b"))
        assert not out.ok and out.exit_code == 126

    def test_list_dir_skips_dependency_directories(self, tools, tmp_path):
        (tmp_path / "venv").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "src.py").write_text("x")
        out = run(tools.list_dir("."))
        assert "src.py" in out.output
        assert "venv" not in out.output and "node_modules" not in out.output

    def test_list_dir_glob_filters_files(self, tools, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.txt").write_text("x")
        out = run(tools.list_dir(".", glob="*.py"))
        assert "a.py" in out.output and "b.txt" not in out.output

    def test_search_finds_matches_with_locations(self, tools, tmp_path):
        (tmp_path / "a.py").write_text("import os\ndef go():\n    return 1\n")
        out = run(tools.search_files(r"def \w+"))
        assert out.ok and "a.py:2:" in out.output

    def test_search_respects_glob(self, tools, tmp_path):
        (tmp_path / "a.py").write_text("needle\n")
        (tmp_path / "b.txt").write_text("needle\n")
        out = run(tools.search_files("needle", glob="*.txt"))
        assert "b.txt" in out.output and "a.py" not in out.output

    def test_search_skips_dependency_directories(self, tools, tmp_path):
        vendored = tmp_path / "node_modules"
        vendored.mkdir()
        (vendored / "dep.py").write_text("needle\n")
        out = run(tools.search_files("needle"))
        assert "node_modules" not in out.output

    def test_invalid_regex_is_reported_not_raised(self, tools):
        out = run(tools.search_files("(unclosed"))
        assert not out.ok and out.error_class is ar.ErrorClass.SYNTAX

    def test_write_is_atomic_no_tempfile_left_behind(self, tools, tmp_path):
        run(tools.write_file("a.py", "x"))
        assert not any(p.name.startswith("a.py.tmp") for p in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_unknown_tool_becomes_an_observation(self, executor):
        result = run(executor.execute(call("teleport"), 5))
        assert not result.ok and result.exit_code == 127
        assert "available tools" in result.output

    def test_bad_arguments_become_an_observation(self, executor):
        result = run(executor.execute(call("read_file"), 5))
        assert not result.ok and result.exit_code == 22

    def test_handler_exception_does_not_escape(self, tmp_path):
        async def boom() -> ToolOutcome:
            raise RuntimeError("kaboom")

        reg = ToolRegistry([ToolSpec(name="boom", description="explodes",
                                     params={"type": "object", "properties": {},
                                             "required": []},
                                     handler=boom)])
        ex = MultiToolExecutor(reg, tmp_path)
        result = run(ex.execute(call("boom"), 5))
        assert not result.ok and "kaboom" in result.output

    def test_handler_timeout_is_classified(self, tmp_path):
        async def slow() -> ToolOutcome:
            await asyncio.sleep(5)
            return ToolOutcome(True)

        reg = ToolRegistry([ToolSpec(name="slow", description="naps",
                                     params={"type": "object", "properties": {},
                                             "required": []},
                                     handler=slow, timeout_s=0.05)])
        ex = MultiToolExecutor(reg, tmp_path)
        result = run(ex.execute(call("slow"), 5))
        assert result.error_class is ar.ErrorClass.TIMEOUT

    def test_dirty_is_set_only_by_successful_mutation(self, executor):
        run(executor.execute(call("list_dir", path="."), 5))
        assert executor.dirty is False
        run(executor.execute(call("write_file", path="a.py", content="x"), 5))
        assert executor.dirty is True

    def test_failed_write_does_not_mark_dirty(self, executor):
        run(executor.execute(call("write_file", path="../x.py", content="x"), 5))
        assert executor.dirty is False

    def test_output_is_truncated_and_redacted_like_shell_output(self, tmp_path, registry):
        ex = MultiToolExecutor(registry, tmp_path)
        big = "y" * (ar.MAX_OUTPUT_CHARS + 5_000)
        (tmp_path / "big.txt").write_text(big)
        result = run(ex.execute(call("read_file", path="big.txt"), 10))
        assert result.truncated
        assert len(result.output) < ar.MAX_OUTPUT_CHARS + 500

    def test_shell_tool_is_delegated_to_the_shell_executor(self, tmp_path):
        shell = ar.SimulatedShell()
        reg = build_default_registry(tmp_path, shell)
        ex = MultiToolExecutor(reg, tmp_path, shell_executor=shell)
        run(ex.execute(ar.ToolCall(tool=SHELL_TOOL,
                                   args={"command": "npm install",
                                         "argv": ["npm", "install"]}), 10))
        assert shell.calls == ["npm install"]

    def test_fingerprint_still_distinguishes_tools(self):
        """RepetitionDetector hashes (tool, args); it must see the tool name."""
        a = call("read_file", path="x.py")
        b = call("write_file", path="x.py")
        assert a.fingerprint != b.fingerprint


# ---------------------------------------------------------------------------
# tool-calling policy
# ---------------------------------------------------------------------------

class _Client:
    """Minimal ModelClient returning a canned completion."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.systems: list[str] = []

    async def complete(self, system, user, *, schema=None, max_tokens=512):
        self.systems.append(system)
        text = self.texts.pop(0) if self.texts else "{}"
        return ar_completion(text)


def ar_completion(text: str):
    from omni import backends as llm_backends
    return llm_backends.Completion(text=text, prompt_tokens=1, completion_tokens=1,
                                   backend=llm_backends.Backend.OLLAMA, model="test",
                                   latency_s=0.0)


@pytest.fixture
def policy(tmp_path, registry) -> ToolCallPolicy:
    return ToolCallPolicy(client=_Client(), ledger=ar.Ledger(budget=ar.Budget()),
                          registry=registry)


class TestToolCallPolicy:
    def test_parses_the_tool_form(self, policy):
        decision = policy._parse(json.dumps({
            "kind": "act", "thought": "read it",
            "tool": "read_file", "args": {"path": "a.py"}}))
        assert isinstance(decision, ar.Act)
        assert decision.call.tool == "read_file"
        assert decision.call.args == {"path": "a.py"}

    def test_parses_the_legacy_command_form(self, policy):
        """Old transcripts, plans, and HeuristicRepairPolicy still work."""
        decision = policy._parse(json.dumps({
            "kind": "act", "thought": "run it", "command": "npm install"}))
        assert decision.call.tool == SHELL_TOOL
        assert decision.call.args["command"] == "npm install"

    def test_strips_code_fences(self, policy):
        decision = policy._parse(
            '```json\n{"kind":"finish","succeeded":true,"summary":"ok"}\n```')
        assert isinstance(decision, ar.Finish) and decision.succeeded

    def test_unknown_tool_is_rejected(self, policy):
        with pytest.raises(ValueError, match="unknown tool"):
            policy._parse(json.dumps({"kind": "act", "thought": "t",
                                      "tool": "teleport", "args": {}}))

    def test_missing_required_argument_is_rejected(self, policy):
        with pytest.raises(ValueError, match="requires"):
            policy._parse(json.dumps({"kind": "act", "thought": "t",
                                      "tool": "read_file", "args": {}}))

    def test_non_object_args_rejected(self, policy):
        with pytest.raises(ValueError, match="must be an object"):
            policy._parse(json.dumps({"kind": "act", "thought": "t",
                                      "tool": "read_file", "args": "path=a"}))

    def test_ask_human_round_trips(self, policy):
        decision = policy._parse(json.dumps({
            "kind": "ask_human", "question": "which file?", "options": ["a", "b"]}))
        assert isinstance(decision, ar.AskHuman) and decision.options == ["a", "b"]

    def test_system_prompt_lists_the_tools(self, policy, registry):
        for name in registry.names():
            assert name in policy.SYSTEM

    def test_schema_is_the_registry_schema(self, policy, registry):
        assert policy.schema == registry.decision_schema()


# ---------------------------------------------------------------------------
# verification detection
# ---------------------------------------------------------------------------

class TestVerifyDetection:
    def test_empty_workspace_has_nothing_to_verify(self, tmp_path):
        assert detect_verify(tmp_path) is None

    def test_python_sources_get_a_compile_check(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1\n")
        assert detect_verify(tmp_path) == COMPILEALL

    def test_tests_directory_selects_pytest(self, tmp_path):
        (tmp_path / "tests").mkdir()
        assert detect_verify(tmp_path) == PYTEST

    def test_test_file_selects_pytest(self, tmp_path):
        (tmp_path / "test_x.py").write_text("def test_x(): pass\n")
        assert detect_verify(tmp_path) == PYTEST

    def test_package_json_test_script_selects_npm(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"test": "jest"}}))
        assert detect_verify(tmp_path) == NPM_TEST

    def test_package_json_without_test_script_is_ignored(self, tmp_path):
        (tmp_path / "package.json").write_text(json.dumps({"scripts": {}}))
        assert detect_verify(tmp_path) is None

    def test_malformed_package_json_does_not_raise(self, tmp_path):
        (tmp_path / "package.json").write_text("{not json")
        assert detect_verify(tmp_path) is None


# ---------------------------------------------------------------------------
# verify gate
# ---------------------------------------------------------------------------

class _Fixed:
    def __init__(self, decision) -> None:
        self.decision = decision

    async def propose(self, ctx):
        return self.decision


def context(tmp_path, steps=()) -> ar.PolicyContext:
    return ar.PolicyContext(goal="g", phase="repair", steps=list(steps),
                            warnings=[], resume=None, workspace=tmp_path)


def step(idx: int, tool: str, ok: bool, **args) -> ar.StepRecord:
    c = ar.ToolCall(tool=tool, args=args)
    return ar.StepRecord(idx, "t", c,
                         ar.ToolResult(call_id=c.call_id, ok=ok,
                                       exit_code=0 if ok else 1, output=""))


class _Dirty:
    def __init__(self, dirty: bool) -> None:
        self.dirty = dirty


class TestVerifyGate:
    SPEC = VerifySpec("pytest -q", label="pytest")

    def test_non_finish_decisions_pass_through(self, tmp_path, registry):
        act = ar.Act(thought="t", call=ar.ToolCall(tool="read_file",
                                                   args={"path": "a.py"}))
        gate = VerifyGate(_Fixed(act), self.SPEC, registry)
        assert run(gate.propose(context(tmp_path))) is act

    def test_failed_finish_passes_through(self, tmp_path, registry):
        finish = ar.Finish(succeeded=False, summary="gave up")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry)
        assert run(gate.propose(context(tmp_path))) is finish

    def test_clean_workspace_is_not_verified(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="read only")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(False))
        assert run(gate.propose(context(tmp_path))) is finish

    def test_nothing_to_verify_passes_through(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="done")
        gate = VerifyGate(_Fixed(finish), None, registry, dirty_source=_Dirty(True))
        assert run(gate.propose(context(tmp_path))) is finish

    def test_claimed_success_is_converted_into_a_verification(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="all done")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(True))
        decision = run(gate.propose(context(tmp_path)))
        assert isinstance(decision, ar.Act)
        assert decision.call.args["command"] == "pytest -q"

    def test_success_is_allowed_once_verification_has_passed(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="all done")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(True))
        steps = [step(1, "write_file", True, path="a.py", content="x"),
                 step(2, SHELL_TOOL, True, command="pytest -q")]
        assert run(gate.propose(context(tmp_path, steps))) is finish

    def test_a_later_edit_invalidates_an_earlier_green_run(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="all done")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(True))
        steps = [step(1, SHELL_TOOL, True, command="pytest -q"),
                 step(2, "write_file", True, path="a.py", content="x")]
        assert isinstance(run(gate.propose(context(tmp_path, steps))), ar.Act)

    def test_failed_verification_does_not_count_as_verified(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="all done")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(True))
        steps = [step(1, SHELL_TOOL, False, command="pytest -q")]
        assert isinstance(run(gate.propose(context(tmp_path, steps))), ar.Act)

    def test_unverifiable_claim_becomes_an_honest_failure(self, tmp_path, registry):
        finish = ar.Finish(succeeded=True, summary="all done")
        gate = VerifyGate(_Fixed(finish), self.SPEC, registry,
                          dirty_source=_Dirty(True), max_rounds=2)
        for _ in range(2):
            run(gate.propose(context(tmp_path)))
        final = run(gate.propose(context(tmp_path)))
        assert isinstance(final, ar.Finish) and final.succeeded is False
        assert "did not pass" in final.summary

    def test_spec_is_detected_lazily_after_files_appear(self, tmp_path, registry):
        """The workspace is empty when the stack is built; files arrive later."""
        gate = VerifyGate(_Fixed(ar.Finish(succeeded=True, summary="x")),
                          lambda: detect_verify(tmp_path), registry,
                          dirty_source=_Dirty(True))
        assert gate.spec is None
        (tmp_path / "test_x.py").write_text("def test_x(): pass\n")
        assert gate.spec == PYTEST


# ---------------------------------------------------------------------------
# golden tasks — real loop, real subprocess
# ---------------------------------------------------------------------------

FIZZBUZZ = (
    "def fizzbuzz(n):\n"
    "    if n % 15 == 0:\n        return 'FizzBuzz'\n"
    "    if n % 3 == 0:\n        return 'Fizz'\n"
    "    if n % 5 == 0:\n        return 'Buzz'\n"
    "    return str(n)\n"
)

FIZZBUZZ_TEST = (
    "from fizzbuzz import fizzbuzz\n\n"
    "def test_fizzbuzz():\n"
    "    assert fizzbuzz(3) == 'Fizz'\n"
    "    assert fizzbuzz(5) == 'Buzz'\n"
    "    assert fizzbuzz(15) == 'FizzBuzz'\n"
    "    assert fizzbuzz(7) == '7'\n"
)


def drive(tmp_path, script, fallback):
    """Run a scripted policy through a real RepairLoop over a real subprocess."""
    ar.set_quiet(True)
    shell = ar.SubprocessShell(workspace=tmp_path)
    stack = build_agent_stack(tmp_path, shell, auto_approve_writes=True)
    policy = stack.repair_policy_factory("goal", ar.Ledger(budget=ar.Budget()))
    policy.inner = ar.ScriptedPolicy(script, fallback=fallback)
    loop = ar.RepairLoop(executor=stack.executor, policy=policy,
                         command_policy=ar.CommandPolicy(tmp_path),
                         journal=ar.RunJournal("run_t", path=tmp_path / "j.jsonl"),
                         workspace=tmp_path, tool_policy=stack.tool_policy)
    return run(loop.run(goal="goal",
                        ledger=ar.Ledger(budget=ar.Budget(max_iterations=12))))


def write(path: str, content: str) -> ar.Act:
    return ar.Act(thought=f"write {path}",
                  call=ar.ToolCall(tool="write_file",
                                   args={"path": path, "content": content}))


@pytest.mark.slow
class TestGoldenTasks:
    def test_create_and_verify_succeeds(self, tmp_path):
        outcome = drive(
            tmp_path,
            [write("fizzbuzz.py", FIZZBUZZ),
             write("test_fizzbuzz.py", FIZZBUZZ_TEST),
             ar.Finish(succeeded=True, summary="fizzbuzz implemented")],
            ar.Finish(succeeded=True, summary="done"))
        assert outcome.succeeded, outcome.detail
        assert (tmp_path / "fizzbuzz.py").exists()
        # The gate forced a verification run before accepting success.
        assert any(s.call.tool == SHELL_TOOL and s.result.ok for s in outcome.steps)

    def test_false_success_is_caught(self, tmp_path):
        outcome = drive(
            tmp_path,
            [write("broken.py", "def f(:\n    return 1\n"),
             ar.Finish(succeeded=True, summary="all done!")],
            ar.Finish(succeeded=True, summary="really done!"))
        assert not outcome.succeeded
        assert outcome.stop_reason in (ar.StopReason.GUARDRAIL,
                                       ar.StopReason.MAX_ITERATIONS)

    def test_escape_is_refused_without_ending_the_run(self, tmp_path):
        outcome = drive(
            tmp_path,
            [ar.Act(thought="peek", call=ar.ToolCall(
                tool="read_file", args={"path": "C:/Windows/win.ini"})),
             ar.Finish(succeeded=False, summary="blocked as expected")],
            ar.Finish(succeeded=False, summary="stop"))
        blocked = [s for s in outcome.steps if not s.result.ok]
        assert blocked, "the escape attempt should be recorded as a failed step"
        assert blocked[0].result.error_class is ar.ErrorClass.POLICY

    def test_read_only_session_is_not_verified(self, tmp_path):
        (tmp_path / "note.py").write_text("x = 1\n")
        outcome = drive(
            tmp_path,
            [ar.Act(thought="look", call=ar.ToolCall(tool="list_dir",
                                                     args={"path": "."})),
             ar.Finish(succeeded=True, summary="nothing to change")],
            ar.Finish(succeeded=True, summary="done"))
        assert outcome.succeeded
        assert all(s.call.tool != SHELL_TOOL for s in outcome.steps)


# ---------------------------------------------------------------------------
# output budget — regression: the re-read loop
# ---------------------------------------------------------------------------
#
# A real run spent all 12 iterations re-reading four files and wrote nothing.
# read_file returned "lines 1-98 of 98" while finalize_output quietly removed
# the middle of the body, so the model saw a hole under a header claiming
# completeness, asked for the middle, and got another elided response.

HTML_98_LINES = "".join(
    f'  <div class="row-{i}">content for row number {i}</div>\n'
    for i in range(1, 99)
)


class TestOutputBudget:
    @pytest.fixture
    def big(self, tmp_path):
        (tmp_path / "index.html").write_text(HTML_98_LINES, encoding="utf-8")
        return MultiToolExecutor(
            build_default_registry(tmp_path, ar.SimulatedShell()), tmp_path)

    def test_read_is_not_truncated_by_the_dispatcher(self, big):
        result = run(big.execute(call("read_file", path="index.html"), 30))
        assert result.truncated is False, "the middle would have been elided"

    def test_header_reports_only_what_was_shown(self, big):
        result = run(big.execute(call("read_file", path="index.html"), 30))
        header = result.output.splitlines()[0]
        shown = header.split("lines ")[1].split(" of ")[0]
        first, last = (int(x) for x in shown.split("-"))
        rendered = [ln for ln in result.output.splitlines() if "\t" in ln]
        assert len(rendered) == last - first + 1

    def test_body_is_contiguous_with_no_elision_marker(self, big):
        result = run(big.execute(call("read_file", path="index.html"), 30))
        assert "chars elided" not in result.output

    def test_footer_gives_a_usable_continuation_offset(self, big):
        result = run(big.execute(call("read_file", path="index.html"), 30))
        assert "continue with offset=" in result.output
        offset = int(result.output.split("continue with offset=")[1].split()[0])
        nxt = run(big.execute(call("read_file", path="index.html",
                                   offset=offset), 30))
        assert f"lines {offset}-" in nxt.output.splitlines()[0]

    def test_paging_eventually_reaches_the_end(self, big):
        offset, seen = 1, 0
        for _ in range(20):
            out = run(big.execute(call("read_file", path="index.html",
                                       offset=offset), 30)).output
            seen += len([ln for ln in out.splitlines() if "\t" in ln])
            if "continue with offset=" not in out:
                break
            offset = int(out.split("continue with offset=")[1].split()[0])
        assert seen == 98

    def test_search_output_stays_within_budget(self, tmp_path):
        for n in range(40):
            (tmp_path / f"f{n}.py").write_text("needle = 1\n" * 20)
        ex = MultiToolExecutor(
            build_default_registry(tmp_path, ar.SimulatedShell()), tmp_path)
        result = run(ex.execute(call("search_files", pattern="needle",
                                     max_results=100), 60))
        assert result.truncated is False

    def test_list_dir_output_stays_within_budget(self, tmp_path):
        for n in range(400):
            (tmp_path / f"file_with_a_fairly_long_name_{n}.py").write_text("x")
        ex = MultiToolExecutor(
            build_default_registry(tmp_path, ar.SimulatedShell()), tmp_path)
        result = run(ex.execute(call("list_dir", path="."), 30))
        assert result.truncated is False


# ---------------------------------------------------------------------------
# redundant-call guardrail
# ---------------------------------------------------------------------------

def _read_step(idx: int, path: str) -> ar.StepRecord:
    c = ar.ToolCall(tool="read_file", args={"path": path})
    return ar.StepRecord(idx, "t", c,
                         ar.ToolResult(call_id=c.call_id, ok=True, exit_code=0,
                                       output=f"contents of {path}"))


class TestRedundantCallDetector:
    def test_non_adjacent_repeat_warns(self):
        """A -> B -> C -> A never trips the sliding-window detector."""
        det = ar.RedundantCallDetector()
        steps = [_read_step(1, "a"), _read_step(2, "b"),
                 _read_step(3, "c"), _read_step(4, "a")]
        assert ar.RepetitionDetector(3).observe(steps) is None
        verdict = det.observe(steps)
        assert verdict and verdict.action is ar.GuardAction.WARN

    def test_third_occurrence_suspends(self):
        det = ar.RedundantCallDetector()
        steps = [_read_step(1, "a"), _read_step(2, "b"), _read_step(3, "a"),
                 _read_step(4, "c"), _read_step(5, "a")]
        verdict = det.observe(steps)
        assert verdict and verdict.action is ar.GuardAction.SUSPEND

    def test_distinct_calls_are_fine(self):
        det = ar.RedundantCallDetector()
        steps = [_read_step(i, f"file{i}") for i in range(1, 6)]
        assert det.observe(steps) is None

    def test_different_arguments_are_not_redundant(self):
        det = ar.RedundantCallDetector()
        a = ar.ToolCall(tool="read_file", args={"path": "x", "offset": 1})
        b = ar.ToolCall(tool="read_file", args={"path": "x", "offset": 50})
        steps = [
            ar.StepRecord(1, "t", a, ar.ToolResult(call_id=a.call_id, ok=True,
                                                   exit_code=0, output="1")),
            ar.StepRecord(2, "t", b, ar.ToolResult(call_id=b.call_id, ok=True,
                                                   exit_code=0, output="2")),
        ]
        assert det.observe(steps) is None

    def test_it_is_in_the_default_stack(self):
        codes = {type(d).__name__ for d in ar.GuardrailStack().detectors}
        assert "RedundantCallDetector" in codes

    def test_the_read_loop_is_stopped(self):
        """The exact shape of the failing run: repeated reads, all succeeding."""
        stack = ar.GuardrailStack()
        pattern = ["index.html", "js/calc.js", "index.html", "js/calc.js",
                   "index.html"]
        steps: list[ar.StepRecord] = []
        actions = []
        for i, path in enumerate(pattern, start=1):
            steps.append(_read_step(i, path))
            actions.append(stack.evaluate(steps).action)
        assert ar.GuardAction.SUSPEND in actions, actions


# ---------------------------------------------------------------------------
# guardrail proportionality
# ---------------------------------------------------------------------------
#
# Regression: `GuardrailStack` escalates the second warning carrying the same
# *code* straight to SUSPEND. With one shared code for all redundancy, two
# unrelated repeats — a re-listed directory, a re-read file — read as an agent
# spinning on one call and killed the run. One real run re-listed `js/` after
# `read_file('js/app.js')` failed (reasonable verification) and was suspended on
# step 9 having written nothing.

def _call_step(idx: int, tool: str, ok: bool = True, **args) -> ar.StepRecord:
    c = ar.ToolCall(tool=tool, args=args)
    return ar.StepRecord(
        idx, "t", c,
        ar.ToolResult(call_id=c.call_id, ok=ok, exit_code=0 if ok else 2,
                      # distinct bodies, so NoProgressDetector is not what fires
                      output=f"contents of {args.get('path')}@{args.get('offset', 0)}"))


class TestGuardrailProportionality:
    def _walk(self, sequence) -> list[ar.GuardAction]:
        stack = ar.GuardrailStack()
        steps, actions = [], []
        for i, (tool, args) in enumerate(sequence, start=1):
            steps.append(_call_step(i, tool, **args))
            actions.append(stack.evaluate(steps).action)
        return actions

    def test_two_different_repeats_warn_but_do_not_suspend(self):
        actions = self._walk([
            ("list_dir", {"path": "."}), ("list_dir", {"path": "js"}),
            ("read_file", {"path": "a.py"}),
            ("list_dir", {"path": "."}),          # repeat of step 1
            ("read_file", {"path": "b.py"}),
            ("list_dir", {"path": "js"}),          # repeat of step 2
        ])
        assert actions.count(ar.GuardAction.WARN) == 2
        assert ar.GuardAction.SUSPEND not in actions

    def test_the_real_nine_step_run_is_no_longer_killed(self):
        actions = self._walk([
            ("list_dir", {"path": "."}), ("list_dir", {"path": "js"}),
            ("read_file", {"path": "index.html"}),
            ("list_dir", {"path": "."}),
            ("read_file", {"path": "js/calculator.js"}),
            ("read_file", {"path": "script.txt"}),
            ("read_file", {"path": "index.html", "offset": 56}),
            ("read_file", {"path": "js/app.js"}),
            ("list_dir", {"path": "js"}),
        ])
        assert ar.GuardAction.SUSPEND not in actions

    def test_spinning_on_one_call_still_suspends(self):
        actions = self._walk([
            ("list_dir", {"path": "."}), ("read_file", {"path": "a.py"}),
            ("list_dir", {"path": "."}), ("read_file", {"path": "b.py"}),
            ("list_dir", {"path": "."}),           # third time
        ])
        assert ar.GuardAction.SUSPEND in actions

    def test_warning_codes_are_scoped_to_the_call(self):
        detector = ar.RedundantCallDetector()
        a = [_call_step(1, "list_dir", path="."), _call_step(2, "read_file", path="x"),
             _call_step(3, "list_dir", path=".")]
        b = [_call_step(1, "list_dir", path="js"), _call_step(2, "read_file", path="x"),
             _call_step(3, "list_dir", path="js")]
        assert detector.observe(a).code != detector.observe(b).code

    def test_code_still_identifies_the_detector(self):
        detector = ar.RedundantCallDetector()
        steps = [_call_step(1, "list_dir", path="."), _call_step(2, "read_file", path="x"),
                 _call_step(3, "list_dir", path=".")]
        assert detector.observe(steps).code.startswith("REDUNDANT_REPEATED_CALL")

"""
Tests for the policy's parse-repair round-trip.

Regression context: an agent had read every file it needed and was about to
write `js/app.js`. It put the file's content into `write_file.content`, the
reply hit the 640-token policy cap mid-string, and the JSON failed to parse.
The repair round-trip then told the model its reply was *invalid* — so it
produced the same oversized action again, was cut off in the same place, and
the run suspended asking the operator to supply the next command.

The reply was not invalid. There was simply more of it. Truncation and
malformation need different feedback, because they need different corrections.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from omni import backends as llm_backends
from omni import runtime as ar
from omni.agentkit.policy import ToolCallPolicy
from omni.agentkit.tools import build_default_registry, codegen


def run(coro):
    return asyncio.run(coro)


def completion(text: str, finish_reason: str = "stop") -> llm_backends.Completion:
    return llm_backends.Completion(
        text=text, model="t", backend=llm_backends.Backend.OLLAMA,
        finish_reason=finish_reason)


class _Replies:
    """Returns queued completions and records the prompts it was given."""

    def __init__(self, *replies: llm_backends.Completion) -> None:
        self.replies = list(replies)
        self.users: list[str] = []
        self.max_tokens: list[int] = []

    async def complete(self, system, user, *, schema=None, max_tokens=512):
        self.users.append(user)
        self.max_tokens.append(max_tokens)
        return self.replies.pop(0) if self.replies else completion("{}")


@pytest.fixture
def registry(tmp_path):
    return build_default_registry(tmp_path, ar.SimulatedShell(), client=object())


@pytest.fixture
def context(tmp_path):
    return ar.PolicyContext(goal="g", phase="repair", steps=[], warnings=[],
                            resume=None, workspace=tmp_path)


def policy_for(client, registry) -> ToolCallPolicy:
    return ToolCallPolicy(client=client, ledger=ar.Ledger(budget=ar.Budget()),
                          registry=registry)


# ---------------------------------------------------------------------------
# truncation detection lives in one place
# ---------------------------------------------------------------------------

class TestTruncationSignal:
    def test_runtime_owns_the_check(self):
        assert ar.output_truncated(completion("x", "length"))
        assert not ar.output_truncated(completion("x", "stop"))

    def test_codegen_reexports_the_same_function(self):
        assert codegen.output_truncated is ar.output_truncated

    def test_prompt_truncation_is_a_different_signal(self):
        """`truncation_suspected` means the PROMPT was cut, not the reply."""
        c = completion("x", "length")
        assert ar.output_truncated(c) and not c.truncation_suspected


# ---------------------------------------------------------------------------
# repair feedback
# ---------------------------------------------------------------------------

CUT_OFF = '{"kind":"act","thought":"writing it","tool":"write_file","args":{"path":"js/app.js","content":"const a = 1;'
GOOD = json.dumps({"kind": "act", "thought": "ok", "tool": "list_dir",
                   "args": {"path": "."}})


class TestRepairFeedback:
    def test_truncated_reply_is_not_called_invalid(self, registry, context):
        client = _Replies(completion(CUT_OFF, "length"), completion(GOOD))
        run(policy_for(client, registry).propose(context))
        retry_prompt = client.users[1]
        assert "CUT OFF" in retry_prompt
        assert "was invalid" not in retry_prompt

    def test_truncated_reply_is_steered_to_generate_file(self, registry, context):
        client = _Replies(completion(CUT_OFF, "length"), completion(GOOD))
        run(policy_for(client, registry).propose(context))
        assert "generate_file" in client.users[1]

    def test_malformed_reply_still_reports_invalid(self, registry, context):
        client = _Replies(completion("not json at all", "stop"), completion(GOOD))
        run(policy_for(client, registry).propose(context))
        retry_prompt = client.users[1]
        assert "was invalid" in retry_prompt
        assert "CUT OFF" not in retry_prompt

    def test_repair_recovers_when_the_retry_is_good(self, registry, context):
        client = _Replies(completion(CUT_OFF, "length"), completion(GOOD))
        decision = run(policy_for(client, registry).propose(context))
        assert isinstance(decision, ar.Act)
        assert decision.call.tool == "list_dir"

    def test_feedback_is_appended_never_prepended(self, registry, context):
        """LLMPolicy relies on a stable prefix for KV-cache reuse."""
        client = _Replies(completion(CUT_OFF, "length"), completion(GOOD))
        run(policy_for(client, registry).propose(context))
        assert client.users[1].startswith(client.users[0])

    def test_two_truncated_replies_name_the_real_cause(self, registry, context):
        client = _Replies(completion(CUT_OFF, "length"),
                          completion(CUT_OFF, "length"))
        decision = run(policy_for(client, registry).propose(context))
        assert isinstance(decision, ar.AskHuman)
        assert "cut off" in decision.question.lower()
        assert "generate_file" in decision.question

    def test_two_malformed_replies_ask_generically(self, registry, context):
        client = _Replies(completion("garbage", "stop"), completion("garbage", "stop"))
        decision = run(policy_for(client, registry).propose(context))
        assert isinstance(decision, ar.AskHuman)
        assert "cut off" not in decision.question.lower()

    def test_ledger_is_charged_for_both_attempts(self, registry, context):
        client = _Replies(completion(CUT_OFF, "length"), completion(GOOD))
        ledger = ar.Ledger(budget=ar.Budget())
        policy = ToolCallPolicy(client=client, ledger=ledger, registry=registry)
        run(policy.propose(context))
        assert len(client.users) == 2


# ---------------------------------------------------------------------------
# budget and steering
# ---------------------------------------------------------------------------

class TestPolicyBudget:
    def test_default_budget_fits_a_file_bearing_action(self, registry, context):
        client = _Replies(completion(GOOD))
        run(policy_for(client, registry).propose(context))
        assert client.max_tokens[0] >= 2048, (
            "640 tokens cut off any write_file carrying real content")

    def test_rules_prefer_generate_file_for_real_content(self, registry):
        system = policy_for(_Replies(), registry).SYSTEM
        assert "generate_file" in system
        assert "ONLY for short files" in system

    def test_rules_warn_that_inlining_overruns_the_reply(self, registry):
        assert "overruns the reply limit" in policy_for(_Replies(), registry).SYSTEM

    def test_system_prompt_is_still_byte_stable(self, registry):
        a = policy_for(_Replies(), registry).SYSTEM
        b = policy_for(_Replies(), registry).SYSTEM
        assert a == b


# ---------------------------------------------------------------------------
# action history in the prompt
# ---------------------------------------------------------------------------
#
# Regression: only the last four observations were rendered. An action whose
# observation had scrolled out left no trace at all, so the agent re-issued it.
# A real run read index.html and calculator.js, watched them fall out of the
# window, read both again, and was suspended by the repetition guardrail after
# eleven steps without a single write.

def step(idx: int, tool: str, ok: bool = True, **args) -> ar.StepRecord:
    c = ar.ToolCall(tool=tool, args=args)
    return ar.StepRecord(idx, "t", c,
                         ar.ToolResult(call_id=c.call_id, ok=ok,
                                       exit_code=0 if ok else 1,
                                       output="OBSERVED BODY"))


def rendered(steps, window: int = 6) -> str:
    policy = ar.LLMPolicy(client=None, ledger=ar.Ledger(budget=ar.Budget()),
                          recent_observations=window)
    ctx = ar.PolicyContext(goal="g", phase="repair", steps=steps, warnings=[],
                           resume=None, workspace=None)
    return policy._render(ctx)


class TestActionHistory:
    def test_short_runs_have_no_ledger(self):
        out = rendered([step(1, "list_dir", path=".")])
        assert "ALREADY DONE" not in out

    def test_scrolled_out_calls_are_still_listed(self):
        steps = [step(i, "read_file", path=f"f{i}.py") for i in range(1, 11)]
        out = rendered(steps)
        assert "ALREADY DONE" in out
        assert "read_file(path='f1.py')" in out

    def test_recent_calls_keep_their_full_observation(self):
        steps = [step(i, "read_file", path=f"f{i}.py") for i in range(1, 11)]
        out = rendered(steps)
        assert out.count("OBSERVED BODY") == 6

    def test_every_step_is_accounted_for(self):
        steps = [step(i, "read_file", path=f"f{i}.py") for i in range(1, 11)]
        out = rendered(steps)
        for i in range(1, 11):
            assert f"[{i}]" in out, f"step {i} vanished from the prompt"

    def test_the_repeated_read_is_visible_in_the_ledger(self):
        """The exact call the failing run re-issued must be listed."""
        steps = [
            step(1, "list_dir", path="."), step(2, "list_dir", path="js"),
            step(3, "read_file", path="index.html"),
            step(4, "read_file", path="index.html", offset=56),
            step(5, "read_file", path="js/calculator.js"),
            step(6, "read_file", path="js/calculator.js", offset=74),
            step(7, "read_file", path="js/history.js"),
            step(8, "read_file", path="index.html"),
            step(9, "read_file", path="index.html", offset=56),
            step(10, "read_file", path="js/calculator.js"),
        ]
        out = rendered(steps)
        assert "read_file(path='index.html')" in out
        assert "do not repeat" in out.lower()

    def test_failures_are_marked_in_the_ledger(self):
        steps = [step(1, "write_file", ok=False, path="../x", content="y")]
        steps += [step(i, "read_file", path=f"f{i}.py") for i in range(2, 9)]
        out = rendered(steps)
        assert "exit 1" in out

    def test_window_is_configurable(self):
        steps = [step(i, "read_file", path=f"f{i}.py") for i in range(1, 11)]
        assert rendered(steps, window=2).count("OBSERVED BODY") == 2


class TestDescribeCallRendering:
    def test_integers_are_not_quoted(self):
        """`offset='74'` read as a string argument that was never sent."""
        call = ar.ToolCall(tool="read_file", args={"path": "a.py", "offset": 74})
        assert "offset=74" in ar.describe_call(call)

    def test_strings_are_quoted(self):
        call = ar.ToolCall(tool="read_file", args={"path": "a.py"})
        assert "path='a.py'" in ar.describe_call(call)

    def test_booleans_are_not_quoted(self):
        call = ar.ToolCall(tool="edit_file",
                           args={"path": "a", "old": "b", "new": "c",
                                 "replace_all": True})
        assert "replace_all=True" in ar.describe_call(call)

    def test_long_values_are_clipped(self):
        call = ar.ToolCall(tool="write_file",
                           args={"path": "a.py", "content": "x" * 500})
        assert "..." in ar.describe_call(call)
        assert len(ar.describe_call(call)) < 200

    def test_shell_calls_render_as_the_bare_command(self):
        call = ar.ToolCall(tool="run_command", args={"command": "pytest -q"})
        assert ar.describe_call(call) == "pytest -q"


# ---------------------------------------------------------------------------
# budget pressure
# ---------------------------------------------------------------------------
#
# Regression: the iteration cap was invisible to the agent. It explored until
# the cap cut it off mid-thought, having produced nothing — nine steps of
# reading for "create a readme.md file". A deadline it can see is a deadline it
# can plan against.

def prompt(used: int | None, cap: int | None, steps=()) -> str:
    policy = ar.LLMPolicy(client=None, ledger=ar.Ledger(budget=ar.Budget()))
    ctx = ar.PolicyContext(goal="create a readme", phase="repair", steps=list(steps),
                           warnings=[], resume=None, workspace=None,
                           iterations_used=used, iterations_max=cap)
    return policy._render(ctx)


class TestBudgetPressure:
    def test_absent_budget_says_nothing(self):
        assert "BUDGET" not in prompt(None, None)

    def test_budget_is_stated(self):
        assert "action 3 of 12" in prompt(2, 12)

    def test_early_run_is_not_hurried(self):
        out = prompt(2, 12)
        assert "Stop gathering information" not in out
        assert "LAST ACTION" not in out

    def test_running_low_says_deliver(self):
        out = prompt(9, 12)
        assert "Only 3 actions remain" in out
        assert "Stop gathering information" in out

    def test_final_action_is_explicit(self):
        out = prompt(11, 12)
        assert "LAST ACTION" in out
        assert "succeeded=false" in out

    def test_exhausted_budget_still_renders(self):
        assert "LAST ACTION" in prompt(12, 12)

    def test_iterations_left_is_computed(self):
        ctx = ar.PolicyContext(goal="g", phase="p", steps=[], warnings=[],
                               resume=None, workspace=None,
                               iterations_used=9, iterations_max=12)
        assert ctx.iterations_left == 3

    def test_iterations_left_never_negative(self):
        ctx = ar.PolicyContext(goal="g", phase="p", steps=[], warnings=[],
                               resume=None, workspace=None,
                               iterations_used=20, iterations_max=12)
        assert ctx.iterations_left == 0

    def test_iterations_left_is_none_without_a_budget(self):
        ctx = ar.PolicyContext(goal="g", phase="p", steps=[], warnings=[],
                               resume=None, workspace=None)
        assert ctx.iterations_left is None

    def test_repair_loop_populates_the_budget(self, tmp_path):
        """The loop must actually pass the ledger through, not just support it."""
        seen: list[ar.PolicyContext] = []

        class _Spy:
            async def propose(self, ctx):
                seen.append(ctx)
                return ar.Finish(succeeded=True, summary="done")

        ar.set_quiet(True)
        loop = ar.RepairLoop(executor=ar.SimulatedShell(), policy=_Spy(),
                             command_policy=ar.CommandPolicy(tmp_path),
                             journal=ar.RunJournal("r", path=tmp_path / "j.jsonl"),
                             workspace=tmp_path)
        run(loop.run(goal="g", ledger=ar.Ledger(budget=ar.Budget(max_iterations=7))))
        assert seen and seen[0].iterations_max == 7
        assert seen[0].iterations_used == 0

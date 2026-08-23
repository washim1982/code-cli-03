"""
Tests for the loop trace table.

The step table answers "what did the tools do". This one answers "what did the
loop do around them" — which route was chosen and how confidently, what the plan
was, what verdict each call got from the policy, when a guardrail fired, and what
the run cost. All of it was already in the journal and none of it was visible.
"""

from __future__ import annotations

import pytest

from omni import cli
from omni import runtime as ar


def journal_with(*events: tuple[str, ar.RunState | None, dict]) -> ar.RunJournal:
    j = ar.RunJournal(run_id="run_test123")
    for kind, state, payload in events:
        j.emit(kind, state=state, **payload)
    return j


def cells(table, column: int) -> list[str]:
    return [str(c) for c in table.columns[column]._cells]


DETAIL = 4
PHASE = 2
STATE = 3


class TestEmptyAndShape:
    def test_empty_journal_renders_nothing(self):
        assert cli.format_loop_table(ar.RunJournal(run_id="r")) is None

    def test_title_names_the_run(self):
        table = cli.format_loop_table(
            journal_with(("run.state", ar.RunState.ROUTING, {})))
        assert "run_test123" in table.title

    def test_one_row_per_event(self):
        table = cli.format_loop_table(journal_with(
            ("run.state", ar.RunState.ROUTING, {}),
            ("route.decision", None, {"route": "PLAN", "confidence": 0.9,
                                      "rationale": "build"}),
            ("run.state", ar.RunState.PLANNING, {}),
        ))
        assert len(cells(table, PHASE)) == 3

    def test_states_are_shown(self):
        table = cli.format_loop_table(journal_with(
            ("run.state", ar.RunState.EXECUTING, {})))
        assert "EXECUTING" in cells(table, STATE)[0]

    def test_timestamps_are_relative_to_the_first_event(self):
        table = cli.format_loop_table(journal_with(
            ("run.state", ar.RunState.ROUTING, {}),
            ("run.state", ar.RunState.PLANNING, {}),
        ))
        assert cells(table, 1)[0] == "0"


class TestEventDetails:
    def _detail(self, kind, payload, state=None) -> str:
        table = cli.format_loop_table(journal_with((kind, state, payload)))
        return cells(table, DETAIL)[0]

    def test_route_shows_choice_confidence_and_reason(self):
        d = self._detail("route.decision", {"route": "PLAN", "confidence": 0.92,
                                            "rationale": "build request"})
        assert "PLAN" in d and "0.92" in d and "build request" in d

    def test_plan_shows_step_count_and_titles(self):
        d = self._detail("plan.created", {"steps": ["Create module", "Run tests"]})
        assert "2 step(s)" in d
        assert "Create module" in d and "Run tests" in d

    def test_blocked_policy_verdict_includes_the_reason(self):
        d = self._detail("tool.policy", {"tool": "write_file", "risk": "FORBIDDEN",
                                         "reason": "escapes the workspace root"})
        assert "FORBIDDEN" in d and "escapes the workspace root" in d

    def test_allowed_policy_verdict_stays_terse(self):
        d = self._detail("tool.policy", {"command": "ls", "risk": "SAFE",
                                         "reason": "allowed"})
        assert "SAFE" in d and "allowed" not in d

    def test_tool_result_shows_exit_class_and_duration(self):
        d = self._detail("tool.result", {"exit_code": 127, "error_class": "MISSING_PATH",
                                         "duration_ms": 42})
        assert "exit=127" in d and "MISSING_PATH" in d and "42 ms" in d

    def test_plan_step_result(self):
        d = self._detail("plan.step.result", {"step": "Run tests", "exit_code": 0})
        assert "Run tests" in d and "exit 0" in d

    def test_plan_step_failure_shows_error_class(self):
        d = self._detail("plan.step.failed", {"step": "Run tests",
                                              "error_class": "SYNTAX"})
        assert "Run tests" in d and "SYNTAX" in d

    def test_guardrail_shows_its_code(self):
        d = self._detail("guardrail.suspend", {"code": "REDUNDANT_REPEATED_CALL"})
        assert "REDUNDANT_REPEATED_CALL" in d

    def test_budget_summary_on_completion(self):
        d = self._detail("run.state", {"budget": {"iterations": 4, "tokens": 1200,
                                                  "elapsed_s": 3.5, "usd": 0.0}},
                         state=ar.RunState.SUCCEEDED)
        assert "iterations=4" in d and "tokens=1200" in d and "3.5" in d

    def test_suspension_shows_why(self):
        d = self._detail("run.suspended",
                         {"payload": {"failure_reason": "router abstained"}},
                         state=ar.RunState.SUSPENDED_HITL)
        assert "router abstained" in d

    def test_plain_state_transition_has_no_detail(self):
        assert self._detail("run.state", {}, state=ar.RunState.ROUTING) == ""

    def test_unknown_event_type_still_renders(self):
        d = self._detail("something.new", {"a": 1, "b": "two"})
        assert "a=1" in d and "b=two" in d


class TestAgainstARealRun:
    def test_trace_covers_the_whole_lifecycle(self, tmp_path):
        import asyncio

        ar.set_quiet(True)

        async def planner(goal):
            return [ar.PlanStep("List", "ls")]

        class _Router:
            def route(self, prompt):
                return ar.RouteDecision(ar.Route.PLAN, 0.9, "forced", ())

        orch = ar.Orchestrator(executor=ar.SimulatedShell(), workspace=tmp_path,
                               planner=planner, router=_Router())
        asyncio.run(orch.handle("do a thing"))
        table = cli.format_loop_table(orch.journal)

        phases = cells(table, PHASE)
        states = " ".join(cells(table, STATE))
        assert "route" in phases and "plan" in phases
        for expected in ("ROUTING", "PLANNING", "EXECUTING", "SUCCEEDED"):
            assert expected in states, f"{expected} missing from the trace"

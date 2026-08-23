"""
Tests for running many tasks in one Orchestrator session.

Regression context: the CLI keeps one Orchestrator for the whole session,
because building a new one per turn resets `self.state` to CREATED — the FSM's
SUSPENDED -> RESUMING edge then never fires and every turn opens a separate
run_id. But terminal states are terminal by design, so the *second* task in a
session raised `IllegalTransition: SUCCEEDED -> ROUTING`, which the CLI's
catch-all reduced to `Error occurred: SUCCEEDED -> ROUTING` with the run lost.

A session is many runs. A finished or abandoned run yields a fresh run_id on the
same journal; a suspension that is actually being resumed is left in place.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from omni import runtime as ar


def run(coro):
    return asyncio.run(coro)


class _AlwaysPlan:
    def route(self, prompt):
        return ar.RouteDecision(ar.Route.PLAN, 0.9, "forced", ())


async def _one_step_plan(goal):
    return [ar.PlanStep("List", "ls")]


@pytest.fixture
def orch(tmp_path):
    ar.set_quiet(True)
    return ar.Orchestrator(executor=ar.SimulatedShell(), workspace=tmp_path,
                           planner=_one_step_plan, router=_AlwaysPlan())


class TestConsecutiveRuns:
    def test_three_tasks_in_one_session(self, orch):
        states = [run(orch.handle(f"task {i}")).state for i in range(3)]
        assert states == [ar.RunState.SUCCEEDED] * 3

    def test_each_task_gets_its_own_run_id(self, orch):
        ids = [run(orch.handle(f"task {i}")).run_id for i in range(3)]
        assert len(set(ids)) == 3

    def test_state_is_reset_between_runs(self, orch):
        run(orch.handle("first"))
        assert orch.state is ar.RunState.SUCCEEDED
        orch._begin_run()
        assert orch.state is ar.RunState.CREATED

    def test_journal_path_is_preserved_across_runs(self, orch, tmp_path):
        run(orch.handle("first"))
        before = orch.journal.path
        run(orch.handle("second"))
        assert orch.journal.path == before

    def test_all_runs_land_in_one_journal_file(self, orch, tmp_path):
        run(orch.handle("first"))
        run(orch.handle("second"))
        lines = [json.loads(ln) for ln in
                 (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len({e["run_id"] for e in lines if "run_id" in e}) >= 1
        assert len(lines) > 2


class TestAbandonedSuspension:
    """A new prompt with no resume payload means the operator moved on."""

    @pytest.fixture
    def suspended(self, tmp_path):
        ar.set_quiet(True)
        orch = ar.Orchestrator(executor=ar.SimulatedShell(), workspace=tmp_path,
                               planner=_one_step_plan)
        outcome = run(orch.handle("???"))          # router abstains -> CLARIFY
        assert outcome.state is ar.RunState.SUSPENDED_HITL
        return orch

    def test_new_prompt_without_resume_does_not_raise(self, suspended):
        outcome = run(suspended.handle("a different task", resume=None))
        assert outcome.state is not None           # no IllegalTransition

    def test_new_prompt_starts_a_new_run(self, suspended):
        before = suspended.journal.run_id
        run(suspended.handle("a different task", resume=None))
        assert suspended.journal.run_id != before

    def test_a_resumed_suspension_is_left_in_place(self, suspended):
        """`handle` must still see SUSPENDED so it can walk through RESUMING."""
        state_before = suspended.state
        payload = ar.ResumePayload(
            schema_version=2, run_id=suspended.journal.run_id,
            error_snapshot="", failure_reason="",
            stop_reason=ar.StopReason.POLICY_ASKED_HUMAN)
        suspended._begin_run(payload)
        assert suspended.state is state_before

    def test_terminal_state_is_always_stale(self, orch):
        run(orch.handle("first"))
        orch._begin_run(resume=None)
        assert orch.state is ar.RunState.CREATED


class TestTerminalStatesStayTerminal:
    def test_transition_out_of_succeeded_is_still_illegal(self, orch):
        """The FSM invariant is unchanged; _begin_run starts a new run instead."""
        run(orch.handle("first"))
        with pytest.raises(ar.IllegalTransition):
            orch._transition(ar.RunState.ROUTING)


# ---------------------------------------------------------------------------
# repair that finishes the job
# ---------------------------------------------------------------------------
#
# Regression: the REPAIR route always fell through to the planner. A run wrote
# js/app.js and README.md in the repair loop, then planned seven fresh steps,
# tripped over a package.json that does not exist, and suspended — discarding
# two finished deliverables. `RepairLoop` reports succeeded=True from exactly
# one place, the policy returning Finish(succeeded=True), so a successful repair
# means the goal was reached and the run is over.

class _AlwaysRepair:
    def route(self, prompt):
        return ar.RouteDecision(ar.Route.REPAIR, 0.9, "forced", ())


async def _explodes(goal):
    raise AssertionError("the planner must not run after a successful repair")


class TestRepairCompletesTheRun:
    @pytest.fixture
    def orch(self, tmp_path):
        ar.set_quiet(True)
        return ar.Orchestrator(
            executor=ar.SimulatedShell(), workspace=tmp_path,
            planner=_explodes, router=_AlwaysRepair(),
            repair_policy_factory=lambda cmd, ledger=None: ar.ScriptedPolicy(
                [ar.Finish(succeeded=True, summary="wrote README.md")]))

    def test_successful_repair_ends_the_run(self, orch):
        outcome = run(orch.handle("create a readme and fix any issues"))
        assert outcome.state is ar.RunState.SUCCEEDED

    def test_the_planner_never_runs(self, orch):
        """_explodes would raise; reaching SUCCEEDED proves it was not called."""
        assert run(orch.handle("create a readme")).state is ar.RunState.SUCCEEDED

    def test_the_repair_summary_is_the_outcome(self, orch):
        assert "wrote README.md" in run(orch.handle("create a readme")).detail

    def test_steps_from_the_repair_are_kept(self, tmp_path):
        ar.set_quiet(True)
        call = ar.ToolCall(tool="shell", args={"command": "ls"})
        orch = ar.Orchestrator(
            executor=ar.SimulatedShell(), workspace=tmp_path,
            planner=_explodes, router=_AlwaysRepair(),
            repair_policy_factory=lambda cmd, ledger=None: ar.ScriptedPolicy(
                [ar.Act(thought="look", call=call),
                 ar.Finish(succeeded=True, summary="done")]))
        outcome = run(orch.handle("do it"))
        assert len(outcome.steps) == 1

    def test_a_failed_repair_still_suspends(self, tmp_path):
        ar.set_quiet(True)
        orch = ar.Orchestrator(
            executor=ar.SimulatedShell(), workspace=tmp_path,
            planner=_explodes, router=_AlwaysRepair(),
            repair_policy_factory=lambda cmd, ledger=None: ar.ScriptedPolicy(
                [ar.AskHuman(question="which file?")]))
        assert run(orch.handle("do it")).state is ar.RunState.SUSPENDED_HITL

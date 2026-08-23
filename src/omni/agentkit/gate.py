"""
VerifyGate — success must be earned, not asserted.

A `Policy` decorator. `RepairLoop` ends the moment the policy returns
`Finish(succeeded=True)`, and nothing checks that claim, so a run "succeeds"
whenever the model says so. The gate intercepts that decision and, if the
workspace was modified and no successful verification has happened since,
replaces it with the verification command.

Why a decorator rather than a change to `RepairLoop`:

  * it composes — stack a lint gate on top of a test gate;
  * it is testable against a `ScriptedPolicy` with no model and no subprocess;
  * `RepairLoop`, `PlannerLoop`, and `Orchestrator` need no edits at all.

The forced verification goes back through the loop as an ordinary `Act`, so its
output is journalled, charged to the ledger, and seen by the guardrail stack.
`NoProgressDetector` therefore catches an agent that keeps "fixing" without
changing the observation hash, and `ConsecutiveErrorDetector` suspends it after
three failed rounds — the gate inherits all of that instead of reimplementing it.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from omni import runtime as ar
from omni.agentkit.registry import SHELL_TOOL, ToolRegistry, describe_call
from omni.agentkit.verify import VerifySpec

__all__ = ["VerifyGate"]


class VerifyGate:
    """Wraps a `Policy` and withholds success until verification passes."""

    def __init__(self, inner: "ar.Policy", spec: "VerifySpec | Callable[[], VerifySpec | None] | None",
                 registry: ToolRegistry | None = None,
                 *, dirty_source: Any = None, max_rounds: int = 3) -> None:
        self.inner = inner
        # A callable is resolved lazily and re-tried while it yields None. The
        # workspace is usually empty when the stack is built — the files that
        # decide the verification command are created *by* the run — so
        # detecting once up front picks "nothing to verify" every time.
        self._spec_input = spec
        self._resolved: VerifySpec | None = None
        self.registry = registry
        # Usually the MultiToolExecutor, which flips `dirty` on any successful
        # mutating call. Without one the gate assumes the workspace changed,
        # which is the safe default: verify rather than trust.
        self.dirty_source = dirty_source
        self.max_rounds = max_rounds
        self.rounds = 0

    # -- state inspection --------------------------------------------------- #

    @property
    def spec(self) -> VerifySpec | None:
        """
        The verification to run, resolved on first successful detection.

        Deliberately re-evaluated while it is None: a task that creates the
        project's first test file should be verified by that test file.
        """
        if self._resolved is None:
            self._resolved = (self._spec_input() if callable(self._spec_input)
                              else self._spec_input)
        return self._resolved

    @property
    def _dirty(self) -> bool:
        if self.dirty_source is None:
            return True
        return bool(getattr(self.dirty_source, "dirty", True))

    def _is_verify_step(self, step: "ar.StepRecord") -> bool:
        spec = self.spec
        return (spec is not None
                and describe_call(step.call).strip() == spec.command.strip())

    def _is_mutating(self, step: "ar.StepRecord") -> bool:
        if self.registry is None:
            return step.call.tool != SHELL_TOOL
        spec = self.registry.get(step.call.tool)
        return bool(spec and spec.mutating)

    def _verified(self, steps: Sequence["ar.StepRecord"]) -> bool:
        """
        True if verification has passed and nothing has changed the workspace
        since. Scanning backwards means a later edit correctly invalidates an
        earlier green run.
        """
        for step in reversed(steps):
            if self._is_verify_step(step):
                return bool(step.result.ok)
            if self._is_mutating(step):
                return False
        return False

    # -- Policy ------------------------------------------------------------- #

    async def propose(self, ctx: "ar.PolicyContext") -> "ar.Decision":
        decision = await self.inner.propose(ctx)

        if not isinstance(decision, ar.Finish) or not decision.succeeded:
            return decision
        spec = self.spec
        if spec is None:
            return decision                      # nothing in this workspace to check
        if not self._dirty:
            return decision                      # read-only session
        if self._verified(ctx.steps):
            return decision                      # already proved it

        if self.rounds >= self.max_rounds:
            # Do not let an unverifiable claim through as success. Reporting the
            # run as failed is the honest outcome and keeps the CLI's success
            # panel meaningful.
            return ar.Finish(
                succeeded=False,
                summary=(f"{decision.summary} — but {spec.label} did not pass "
                         f"after {self.rounds} attempt(s)."))

        self.rounds += 1
        return ar.Act(
            thought=(f"Claimed done; running {spec.label} first "
                     f"(attempt {self.rounds}/{self.max_rounds})."),
            call=ar.ToolCall(tool=SHELL_TOOL, args={"command": spec.command}))

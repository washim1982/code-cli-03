"""
One call that assembles the whole capability layer.

The CLI should not have to know the wiring order of registry -> executor ->
policy -> gate. `build_agent_stack` returns the three objects `Orchestrator`
actually takes — an executor, a tool policy, and a repair-policy factory — so
integration is a handful of constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from omni import runtime as ar
from omni.agentkit.dispatch import MultiToolExecutor
from omni.agentkit.gate import VerifyGate
from omni.agentkit.policy import ToolCallPolicy
from omni.agentkit.registry import ToolRegistry
from omni.agentkit.tools import build_default_registry
from omni.agentkit.verify import VerifySpec, detect_verify

__all__ = ["AgentStack", "build_agent_stack"]


@dataclass
class AgentStack:
    registry: ToolRegistry
    executor: MultiToolExecutor
    verify_detector: Callable[[], VerifySpec | None] | None
    repair_policy_factory: Callable[..., Any]

    @property
    def verify_spec(self) -> VerifySpec | None:
        """
        Detected on demand, never cached here.

        The workspace is typically empty when the stack is built — the test file
        that decides the command is created *by* the run — so resolving once up
        front would always answer "nothing to verify".
        """
        return self.verify_detector() if self.verify_detector else None

    @property
    def tool_policy(self) -> Callable[["ar.ToolCall"], "ar.PolicyDecision | None"]:
        return self.registry.policy_for

    def is_mutating(self, call: "ar.ToolCall") -> bool:
        """
        True if this call can change the workspace.

        `PlannerLoop` uses it to tell a failed *inspection* from a failed
        *action*: a read of a file that turns out not to exist is a finding, not
        a blocker, and should not abandon the run. Unknown tools count as
        mutating — the cautious answer when we cannot tell.
        """
        spec = self.registry.get(call.tool)
        return True if spec is None else bool(spec.mutating)

    def orchestrator_kwargs(self) -> dict[str, Any]:
        """Splat straight into `Orchestrator(...)`."""
        return {
            "executor": self.executor,
            "tool_policy": self.tool_policy,
            "is_mutating": self.is_mutating,
            "repair_policy_factory": self.repair_policy_factory,
        }


def build_agent_stack(workspace: Path, shell_executor: "ar.ToolExecutor",
                      *, client: Any | None = None,
                      verify: bool = True,
                      max_verify_rounds: int = 3,
                      tool_timeout_s: float = 60.0,
                      auto_approve_writes: bool = False) -> AgentStack:
    """
    Build the tool layer for one session.

    `client` is an `llm_backends.ModelClient`. Without one the stack still has
    real tools, and the repair policy falls back to the runtime's heuristic —
    which is what simulation mode uses.

    `verify` chooses whether a claimed success has to be proved. It is skipped
    automatically when the workspace has nothing to check.
    """
    registry = build_default_registry(workspace, shell_executor,
                                      tool_timeout_s=tool_timeout_s,
                                      auto_approve_writes=auto_approve_writes,
                                      client=client)
    executor = MultiToolExecutor(registry, workspace, shell_executor=shell_executor)
    detector = (lambda: detect_verify(Path(workspace))) if verify else None

    def repair_policy_factory(command: str, ledger: "ar.Ledger | None" = None) -> Any:
        if client is not None and ledger is not None:
            inner: Any = ToolCallPolicy(client=client, ledger=ledger,
                                        registry=registry)
        else:
            # No model (or no ledger passed by an older factory contract):
            # the heuristic policy still speaks the legacy command shape, which
            # ToolCallPolicy._parse and the loops both accept.
            inner = ar.HeuristicRepairPolicy(command)
        if detector is None:
            return inner
        return VerifyGate(inner, detector, registry, dirty_source=executor,
                          max_rounds=max_verify_rounds)

    return AgentStack(registry=registry, executor=executor,
                      verify_detector=detector,
                      repair_policy_factory=repair_policy_factory)

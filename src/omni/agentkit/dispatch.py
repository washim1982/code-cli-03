"""
Multi-tool executor — satisfies `agent_runtime.ToolExecutor`.

The runtime's loops are written against exactly one interface:

    async def execute(self, call: ToolCall, timeout_s: float) -> ToolResult

so replacing the single-purpose shell executor with a dispatcher over a tool
registry is a constructor argument, not a rewrite. `RepairLoop`, `PlannerLoop`,
and `Orchestrator` are unchanged by this module.

Three properties come free by reusing the runtime's own `ToolResult` and
`finalize_output`:

  * **Bad arguments never raise.** They are returned as an observation with a
    non-zero exit code, the same discipline the loop already applies to policy
    refusals, so the model corrects itself instead of crashing the run.
  * **Redaction-before-truncation applies to file reads.** A `read_file` on a
    `.env` is redacted exactly like shell output, and a secret is never split
    across the truncation boundary.
  * **Guardrails keep working.** `ToolCall.fingerprint` is `digest(tool, args)`,
    so `RepetitionDetector` already catches "read the same file four times"
    with no changes at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from omni import runtime as ar
from omni.agentkit.registry import SHELL_TOOL, ToolOutcome, ToolRegistry, validate_args

__all__ = ["MultiToolExecutor"]


class MultiToolExecutor:
    """Dispatch a `ToolCall` to the handler registered under `call.tool`."""

    def __init__(self, registry: ToolRegistry, workspace: Path,
                 shell_executor: "ar.ToolExecutor | None" = None) -> None:
        self.registry = registry
        self.workspace = Path(workspace)
        self.shell_executor = shell_executor
        # Set by any successful mutating call. `VerifyGate` reads it so a
        # read-only session never pays for a verification run.
        self.dirty = False
        self.calls: list[str] = []

    def reset_dirty(self) -> None:
        self.dirty = False

    async def execute(self, call: "ar.ToolCall", timeout_s: float) -> "ar.ToolResult":
        started = ar.now_ms()
        self.calls.append(call.tool)

        # The shell tool is delegated with its original ToolCall rather than
        # unpacked into kwargs: SubprocessShell sets `truncated` and
        # `redactions` on the ToolResult it builds, and `render_observation`
        # shows those flags to the model. Round-tripping through ToolOutcome
        # would silently drop them.
        if call.tool in (SHELL_TOOL, "shell") and self.shell_executor is not None:
            result = await self.shell_executor.execute(call, timeout_s)
            if result.ok:
                self.dirty = True
            return result

        spec = self.registry.get(call.tool)
        if spec is None:
            known = ", ".join(sorted(self.registry.names())) or "none"
            return self._result(call, ToolOutcome(
                False, 127, f"unknown tool {call.tool!r}; available tools: {known}",
                ar.ErrorClass.POLICY), started)

        args: dict[str, Any] = {k: v for k, v in call.args.items() if k != "argv"}
        problems = validate_args(spec, args)
        if problems:
            return self._result(call, ToolOutcome(
                False, 22,
                f"invalid arguments for {spec.name}: " + "; ".join(problems),
                ar.ErrorClass.SYNTAX), started)

        budget = min(float(timeout_s), float(spec.timeout_s))
        try:
            outcome = await asyncio.wait_for(spec.handler(**args), budget)
        except asyncio.TimeoutError:
            return self._result(call, ToolOutcome(
                False, 124, f"{spec.name} did not return within {budget}s",
                ar.ErrorClass.TIMEOUT), started)
        except TypeError as exc:
            # A handler signature mismatch is a programming error, but it must
            # not kill the run: surface it the same way bad args are surfaced.
            return self._result(call, ToolOutcome(
                False, 22, f"{spec.name} rejected these arguments: {exc}",
                ar.ErrorClass.SYNTAX), started)
        except Exception as exc:                      # noqa: BLE001 - see below
            # Any handler bug becomes an observation. The alternative is an
            # exception escaping into RepairLoop.run, which has no handler and
            # would abort the whole run over one bad tool call.
            return self._result(call, ToolOutcome(
                False, 1, f"{spec.name} raised {type(exc).__name__}: {exc}",
                ar.ErrorClass.UNKNOWN), started)

        if not isinstance(outcome, ToolOutcome):
            return self._result(call, ToolOutcome(
                False, 1,
                f"{spec.name} returned {type(outcome).__name__}, expected ToolOutcome",
                ar.ErrorClass.UNKNOWN), started)

        if spec.mutating and outcome.ok:
            self.dirty = True

        return self._result(call, outcome, started)

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _result(call: "ar.ToolCall", outcome: ToolOutcome,
                started: int) -> "ar.ToolResult":
        output, truncated, redactions = ar.finalize_output(outcome.output or "")
        error_class = outcome.error_class
        if error_class is None:
            error_class = (ar.ErrorClass.NONE if outcome.ok
                           else ar.classify_error(outcome.exit_code, output))
        return ar.ToolResult(
            call_id=call.call_id,
            ok=outcome.ok,
            exit_code=outcome.exit_code,
            output=output,
            error_class=error_class,
            duration_ms=ar.now_ms() - started,
            truncated=truncated,
            redactions=redactions,
        )

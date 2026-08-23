"""
The shell tool — a registry entry for the capability that already existed.

This adds no new power. Its purpose is to put `run_command` in the same tool
list as the filesystem tools, so the model chooses between them explicitly
instead of being forced to express every action as a command string.

Crucially, this tool keeps `defer_to_command_policy=True`, so shell calls are
still classified by `CommandPolicy` — allowlist, shell-metacharacter ban, argv
parsing, and workspace containment all apply exactly as before. The filesystem
tools bypass `CommandPolicy` because they are not shell strings; this one must
not.
"""

from __future__ import annotations

from omni import runtime as ar
from omni.agentkit.registry import SHELL_TOOL, ToolOutcome, ToolSpec

__all__ = ["build_spec"]

DESCRIPTION = (
    "Run one allowlisted command (no pipes, redirects, or chaining). "
    "Use for tests, builds, and git."
)

PARAMS = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}


def build_spec(shell_executor: "ar.ToolExecutor", *,
               timeout_s: float = 60.0) -> ToolSpec:
    """
    Build the shell tool spec.

    `agentkit.dispatch.MultiToolExecutor` normally intercepts this tool and
    hands the original `ToolCall` straight to `shell_executor`, which preserves
    the truncation and redaction flags that `SubprocessShell` sets on its own
    `ToolResult`. The handler below is the fallback for a dispatcher used
    without a shell executor, and it exists so the spec is self-contained.
    """

    async def run_command(command: str, argv: list[str] | None = None,
                          **_ignored: object) -> ToolOutcome:
        call = ar.ToolCall(tool="shell",
                           args={"command": command, "argv": list(argv or [])})
        result = await shell_executor.execute(call, timeout_s)
        return ToolOutcome(ok=result.ok, exit_code=result.exit_code,
                           output=result.output, error_class=result.error_class)

    return ToolSpec(
        name=SHELL_TOOL,
        description=DESCRIPTION,
        params=PARAMS,
        handler=run_command,
        risk=ar.Risk.SAFE,          # real risk is decided by CommandPolicy
        mutating=True,              # a command can change the workspace
        timeout_s=timeout_s,
        defer_to_command_policy=True,
    )

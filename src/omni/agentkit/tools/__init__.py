"""
Built-in tools.

`build_default_registry` is the one call the CLI needs: it returns a registry
holding the five filesystem tools plus the shell tool, bound to one workspace.
"""

from __future__ import annotations

from pathlib import Path

from omni.agentkit.registry import ToolRegistry
from omni.agentkit.tools import codegen, fs, shell

__all__ = ["build_default_registry", "fs", "shell", "codegen"]


def build_default_registry(workspace: Path,
                           shell_executor: object | None = None,
                           *, tool_timeout_s: float = 60.0,
                           auto_approve_writes: bool = False,
                           client: object | None = None) -> ToolRegistry:
    """
    Filesystem tools always; the shell tool only when an executor is supplied.

    Omitting the shell executor yields a read/write-only registry with no
    command execution at all — which is exactly what the repo-review path wants,
    and what makes it safe to run without an approval prompt in front of it.

    `auto_approve_writes` drops `write_file`/`edit_file` from ELEVATED to SAFE,
    skipping the per-write operator prompt. Containment is unaffected.
    """
    from omni import runtime as ar

    write_risk = ar.Risk.SAFE if auto_approve_writes else ar.Risk.ELEVATED

    registry = ToolRegistry(fs.build_specs(workspace, write_risk=write_risk))
    if shell_executor is not None:
        registry.register(shell.build_spec(shell_executor,
                                           timeout_s=tool_timeout_s))
    if client is not None:
        # Only useful with a backend attached: it generates the content it writes.
        registry.register(codegen.build_spec(workspace, client, risk=write_risk))
    return registry

"""
agentkit — additive capability layer for the Omni CLI agent runtime.

Modules here plug into `omni.runtime` through its existing Protocols
(`ToolExecutor`, `Policy`) rather than by modifying it.

  jail      — workspace path containment, shared by CommandPolicy and file writes
  memory    — session-aware context store with embedding-ranked retrieval
  registry  — ToolSpec / ToolRegistry: the type surface for multi-tool dispatch
  tools     — built-in filesystem and shell tools
  dispatch  — MultiToolExecutor, satisfies agent_runtime.ToolExecutor
  policy    — ToolCallPolicy, LLMPolicy that emits {tool, args}
  verify    — picks the workspace's verification command
  gate      — VerifyGate, satisfies agent_runtime.Policy

Every module here imports `agent_runtime`, so they are resolved lazily on first
attribute access rather than at package import:

    from omni.agentkit.registry import ToolRegistry   # or
    agentkit.registry.ToolRegistry

`jail` is now a thin re-export of the top-level `pathguard` module. It used to
hold the implementation, which forced `agent_runtime` to import this package to
reach it — a cycle that only held together because this file kept everything
else lazy. `pathguard` is a standard-library-only leaf, so the graph is now
strictly layered and the laziness below is an optimisation rather than a
load-bearing constraint.
"""

from omni.agentkit import jail
from omni.agentkit.jail import JailBreak, contained, resolve_in

__all__ = [
    "jail",
    "memory",
    "registry",
    "tools",
    "dispatch",
    "policy",
    "verify",
    "gate",
    "JailBreak",
    "contained",
    "resolve_in",
    "ContextStore",
    "MemoryConfig",
    "RetrievedContext",
    "make_session_id",
    "ToolSpec",
    "ToolOutcome",
    "ToolRegistry",
    "MultiToolExecutor",
    "ToolCallPolicy",
    "VerifySpec",
    "detect_verify",
    "VerifyGate",
    "build_default_registry",
    "build_agent_stack",
]

# name -> submodule it lives in. A value of None means the name *is* a submodule.
_LAZY: dict[str, str | None] = {
    "memory": None, "registry": None, "tools": None, "dispatch": None,
    "policy": None, "verify": None, "gate": None,
    "ContextStore": "memory",
    "MemoryConfig": "memory",
    "RetrievedContext": "memory",
    "make_session_id": "memory",
    "ToolSpec": "registry",
    "ToolOutcome": "registry",
    "ToolRegistry": "registry",
    "MultiToolExecutor": "dispatch",
    "ToolCallPolicy": "policy",
    "VerifySpec": "verify",
    "detect_verify": "verify",
    "VerifyGate": "gate",
    "build_default_registry": "tools",
}


def __getattr__(name: str):
    if name == "build_agent_stack":
        from omni.agentkit.stack import build_agent_stack
        return build_agent_stack
    if name in _LAZY:
        import importlib
        where = _LAZY[name] or name
        module = importlib.import_module(f"omni.agentkit.{where}")
        return module if _LAZY[name] is None else getattr(module, name)
    raise AttributeError(f"module 'agentkit' has no attribute {name!r}")

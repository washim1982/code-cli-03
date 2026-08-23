"""
Re-export of `pathguard`, kept so `from agentkit import jail` keeps working.

The implementation moved to a top-level, standard-library-only leaf module.
While it lived here, `agent_runtime` had to import the `agentkit` package to
reach it, and nine `agentkit` modules import `agent_runtime` — a package-level
cycle that held together only because `agentkit/__init__.py` kept everything
except this module behind lazy imports. With `pathguard` as a leaf the graph is
strictly layered and no lazy-import discipline is load-bearing:

    pathguard  <-  agent_runtime  <-  agentkit

New code should import `pathguard` directly.
"""

from omni.pathguard import (  # noqa: F401
    JailBreak,
    contained,
    is_reserved_name,
    resolve_in,
    write_text_in,
)

__all__ = [
    "JailBreak",
    "contained",
    "is_reserved_name",
    "resolve_in",
    "write_text_in",
]

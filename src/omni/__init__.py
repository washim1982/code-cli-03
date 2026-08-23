"""
Omni CLI — a local-first autonomous developer agent.

Layout, in dependency order. Nothing here imports anything above it, so the
graph is strictly layered and acyclic:

    pathguard   workspace path containment; standard library only
    backends    Ollama / LM Studio / llama.cpp clients, retry, circuit breaker
    runtime     the agent loop: FSM, journal, ledger, guardrails, policies
    agentkit    the capability layer: tool registry, filesystem tools,
                dispatch, verification gate, memory, repo survey
    cli         the interactive front end

Containment used to live inside `agentkit`, which forced `runtime` to import
the `agentkit` package while most of `agentkit` imports `runtime` back. That
cycle held together only because `agentkit/__init__.py` kept every other module
behind a lazy import. `pathguard` is a leaf now, so that laziness is an
optimisation rather than a correctness requirement.
"""

__all__ = ["__version__"]

__version__ = "0.2.0"

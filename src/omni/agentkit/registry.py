"""
Tool registry — the type surface for multi-tool dispatch.

`agent_runtime.ToolCall` has always carried a `tool` field, but nothing ever
dispatched on it: `LLMPolicy._parse` hardcoded `tool="shell"` and the executor
read `args["command"]`. There was exactly one tool, so the agent could only act
by spelling a shell command — and `CommandPolicy` bans `>` and `|` wholesale
(correctly), while `echo` alone cannot redirect. The result was an agent that
could not write a file at all.

This module supplies the missing indirection:

    ToolSpec      — name, description, JSON-Schema args, risk, handler
    ToolRegistry  — lookup, prompt rendering, decision schema, policy mapping

Handlers return a plain `ToolOutcome` and stay ignorant of `call_id`, redaction,
and truncation; `agentkit.dispatch` adds those by reusing the runtime's own
`finalize_output`, so a `read_file` on a `.env` is redacted exactly like shell
output. A new tool is roughly fifteen lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable

from omni import runtime as ar

__all__ = [
    "ToolOutcome",
    "ToolSpec",
    "ToolRegistry",
    "describe_call",
    "SHELL_TOOL",
]

# The one tool whose arguments are a shell command string, and which therefore
# must keep going through `CommandPolicy` (allowlist, metacharacter ban, argv
# parsing) rather than through a per-tool risk lookup.
SHELL_TOOL = "run_command"


@dataclass(frozen=True)
class ToolOutcome:
    """
    What a handler returns. Deliberately not a `ToolResult`: handlers should not
    have to know about call ids, timing, redaction, or truncation.
    """
    ok: bool
    exit_code: int = 0
    output: str = ""
    error_class: "ar.ErrorClass | None" = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str                      # one line; rendered into the prompt
    params: dict[str, Any]                # JSON Schema (object) for args
    handler: Callable[..., Awaitable[ToolOutcome]]
    risk: "ar.Risk" = None                # type: ignore[assignment]
    mutating: bool = False                # drives the VerifyGate dirty flag
    timeout_s: float = 30.0
    defer_to_command_policy: bool = False  # true only for the shell tool

    def __post_init__(self) -> None:
        if self.risk is None:
            object.__setattr__(self, "risk", ar.Risk.SAFE)

    @property
    def required(self) -> list[str]:
        return list(self.params.get("required", []))


# Defined in the runtime so `RepairLoop` and `PlannerLoop` can render a call for
# the journal and for `pending_command` without importing agentkit, which sits
# above the runtime in the dependency order.
describe_call = ar.describe_call


# --------------------------------------------------------------------------- #
# Minimal JSON-Schema validation
# --------------------------------------------------------------------------- #
#
# `jsonschema` is not a dependency and is not worth adding for this: the tool
# param schemas here are flat objects of scalars and string arrays. Validation
# failures must never raise — they are returned to the model as an observation
# so it can correct itself, which is the same discipline the loop already
# applies to policy refusals.

_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def _type_ok(value: Any, expected: str) -> bool:
    types = _JSON_TYPES.get(expected)
    if types is None:
        return True
    # bool is a subclass of int; an integer field must not accept True.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, types)


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; empty means valid."""
    problems: list[str] = []
    properties: dict[str, Any] = spec.params.get("properties", {})

    for name in spec.required:
        if name not in args or args[name] is None:
            problems.append(f"missing required argument {name!r}")

    allow_extra = spec.params.get("additionalProperties", False)
    for name, value in args.items():
        if name == "argv":          # injected by the loop, never model-supplied
            continue
        schema = properties.get(name)
        if schema is None:
            if not allow_extra:
                known = ", ".join(sorted(properties)) or "none"
                problems.append(f"unknown argument {name!r} (accepted: {known})")
            continue
        expected = schema.get("type")
        if expected and not _type_ok(value, expected):
            problems.append(
                f"argument {name!r} must be {expected}, got "
                f"{type(value).__name__}")
        if "enum" in schema and value not in schema["enum"]:
            allowed = ", ".join(repr(v) for v in schema["enum"])
            problems.append(f"argument {name!r} must be one of: {allowed}")

    return problems


class ToolRegistry:
    """
    Name -> ToolSpec, plus the three projections the rest of the system needs:
    a prompt fragment, a decision schema for constrained decoding, and a policy
    verdict so the loops can gate tool calls the same way they gate commands.
    """

    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    # -- prompt ------------------------------------------------------------- #

    def render_for_prompt(self) -> str:
        """
        Compact tool list for the system prompt.

        Kept byte-stable for a given registry: `LLMPolicy` relies on the system
        prompt not changing between turns so the inference server can reuse its
        cached KV prefix. Anything that varies per turn belongs in the user
        message, which is append-only.
        """
        lines = []
        for spec in self._specs.values():
            properties: dict[str, Any] = spec.params.get("properties", {})
            required = set(spec.required)
            rendered = []
            for name, schema in properties.items():
                kind = schema.get("type", "string")
                rendered.append(f"{name}:{kind}" if name in required
                                else f"[{name}:{kind}]")
            flag = " (requires approval)" if spec.risk is ar.Risk.ELEVATED else ""
            lines.append(f"- {spec.name}({', '.join(rendered)}) — "
                         f"{spec.description}{flag}")
        return "\n".join(lines)

    # -- constrained decoding ----------------------------------------------- #

    def decision_schema(self) -> dict[str, Any]:
        """
        JSON Schema for one policy decision, mirroring the shape of
        `llm_backends.STRICT_DECISION_SCHEMA` (a `oneOf` over the three kinds)
        so every backend that already handled that one handles this too.

        `args` is intentionally an open object rather than a per-tool `oneOf`:
        nesting a discriminated union inside a discriminated union defeats the
        grammar compilers in several local servers. Shape is enforced here, and
        `validate_args` turns a bad payload into a correctable observation.
        """
        return {
            "oneOf": [
                {"type": "object", "additionalProperties": False,
                 "properties": {
                     "kind": {"const": "act"},
                     "thought": {"type": "string"},
                     "tool": {"type": "string", "enum": self.names()},
                     "args": {"type": "object"}},
                 "required": ["kind", "thought", "tool", "args"]},
                {"type": "object", "additionalProperties": False,
                 "properties": {
                     "kind": {"const": "finish"},
                     "succeeded": {"type": "boolean"},
                     "summary": {"type": "string"}},
                 "required": ["kind", "succeeded", "summary"]},
                {"type": "object", "additionalProperties": False,
                 "properties": {
                     "kind": {"const": "ask_human"},
                     "question": {"type": "string"},
                     "options": {"type": "array", "items": {"type": "string"}}},
                 "required": ["kind", "question"]},
            ]
        }

    # -- policy ------------------------------------------------------------- #

    def policy_for(self, call: "ar.ToolCall") -> "ar.PolicyDecision | None":
        """
        Classify a tool call, or return None to defer to `CommandPolicy`.

        Returning None for the shell tool is what preserves the whole existing
        perimeter — allowlist, metacharacter ban, argv parsing and workspace
        containment still gate every shell command exactly as before. The
        filesystem tools deliberately do *not* go through `CommandPolicy`: they
        are not shell strings, the metacharacter ban is meaningless for them,
        and `omni.pathguard` is the correct control.

        `argv` is set to `[tool, *required-values]` so `approval_grants` can
        authorise an elevated tool by name (`write_file`) or by its exact
        rendering, matching how commands are approved.
        """
        spec = self.get(call.tool)
        if spec is None:
            known = ", ".join(sorted(self._specs)) or "none"
            return ar.PolicyDecision(
                ar.Risk.FORBIDDEN, [call.tool],
                f"unknown tool {call.tool!r}; available tools: {known}")

        if spec.defer_to_command_policy:
            return None

        problems = validate_args(spec, dict(call.args))
        if problems:
            return ar.PolicyDecision(
                ar.Risk.FORBIDDEN, [spec.name],
                "invalid arguments: " + "; ".join(problems))

        reason = ("allowed" if spec.risk is ar.Risk.SAFE
                  else f"requires approval (mutates the workspace)")
        return ar.PolicyDecision(spec.risk, [spec.name, describe_call(call)],
                                 reason)

    # -- convenience -------------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(
            {s.name: {"description": s.description, "params": s.params,
                      "risk": s.risk.value, "mutating": s.mutating}
             for s in self._specs.values()},
            indent=2, sort_keys=True)

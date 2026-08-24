"""
Tool-calling policy — `LLMPolicy` taught to emit `{tool, args}` instead of a
bare command string.

This is a subclass, not an edit. Everything that makes `LLMPolicy` work is
inherited untouched: schema-constrained decoding, the single parse-repair
round-trip with the validation error appended, degradation to `AskHuman` rather
than improvising, ledger charging on every call, and the append-only prompt
layout that keeps the server's cached KV prefix valid.

Two things are overridden:

  * `SYSTEM` — built once from the registry, so it stays byte-stable across
    turns. Anything that varies per turn belongs in the user message.
  * `_parse` — accepts the tool form, and still accepts the legacy
    `{"command": ...}` form so old plans, transcripts, and tests keep working.
"""

from __future__ import annotations

import json
import re
from typing import Any

from omni import runtime as ar
from omni.agentkit.registry import SHELL_TOOL, ToolRegistry

__all__ = ["ToolCallPolicy", "build_system_prompt"]

_PREAMBLE = (
    "You are the acting agent in an autonomous developer system.\n"
    "Reply with a single JSON object and nothing else. One action per turn.\n\n"
    "Shapes:\n"
    '{"kind":"act","thought":"<one sentence>","tool":"<name>","args":{...}}\n'
    '{"kind":"finish","succeeded":true,"summary":"<one sentence>"}\n'
    '{"kind":"ask_human","question":"<one sentence>","options":["..."]}\n\n'
    "Tools:\n"
)

_RULES = (
    "\n\nRules:\n"
    "- Prefer read_file / list_dir / search_files over shell commands for "
    "inspection; they are cheaper and always permitted.\n"
    "- To create a NEW file with real content, use generate_file with a "
    "one-sentence 'spec'. The content is produced in a separate step, so it does "
    "not have to fit inside this reply, and that step is shown the files your "
    "target references — so name them in the spec rather than guessing at their "
    "contents.\n"
    "- To change a file that already EXISTS, use edit_file, not generate_file. "
    "Regenerating a file you have read throws away what you learned from it; "
    "editing forces you to quote the real text.\n"
    "- Use write_file ONLY for short files whose exact bytes you already know. "
    "Putting a long file into write_file's 'content' overruns the reply limit and "
    "the action is discarded. The shell cannot redirect, so it cannot write "
    "files either.\n"
    "- edit_file requires the 'old' text to appear exactly once; read the file "
    "first and quote it precisely.\n"
    "- run_command takes one command with no pipes, redirects, or chaining.\n"
    "- Never repeat a call that already failed unless something changed.\n"
    "- Finish only when the goal is actually done; it will be verified."
)


def build_system_prompt(registry: ToolRegistry) -> str:
    return _PREAMBLE + registry.render_for_prompt() + _RULES


class ToolCallPolicy(ar.LLMPolicy):
    """`LLMPolicy` over a `ToolRegistry`."""

    def __init__(self, *, client: Any, ledger: "ar.Ledger",
                 registry: ToolRegistry, max_tokens: int = 4096) -> None:
        # 640 was sized for a decision carrying a shell command. A decision can
        # now carry file content in `write_file.content`, and 640 tokens cut it
        # off mid-string: the JSON failed to parse, the repair round-trip told
        # the model its reply was "invalid", it produced the same oversized
        # action again, and the run suspended asking the operator for the next
        # command. The rules above steer toward generate_file; this is the
        # headroom for the cases that legitimately need it.
        super().__init__(client=client, ledger=ledger,
                         schema=registry.decision_schema(),
                         max_tokens=max_tokens)
        self.registry = registry
        # Instance attribute shadows the class attribute `LLMPolicy.SYSTEM`,
        # which `propose` reads as `self.SYSTEM`.
        self.SYSTEM = build_system_prompt(registry)

    def _parse(self, text: str) -> "ar.Decision":      # type: ignore[override]
        cleaned = re.sub(r"^```(?:json)?|```$", "", str(text).strip(),
                         flags=re.M).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")

        kind = data.get("kind")

        if kind == "finish":
            return ar.Finish(succeeded=bool(data["succeeded"]),
                             summary=data.get("summary", ""))

        if kind == "ask_human":
            return ar.AskHuman(question=data["question"],
                               options=list(data.get("options", [])))

        if kind != "act":
            raise ValueError(f"unknown decision kind: {kind!r}")

        thought = str(data.get("thought", ""))

        # Legacy shape: a bare command string. Kept so existing transcripts,
        # `HeuristicRepairPolicy`, and the runtime's own tests still parse.
        if "tool" not in data and data.get("command"):
            return ar.Act(thought=thought,
                          call=ar.ToolCall(tool=SHELL_TOOL,
                                           args={"command": str(data["command"])}))

        tool = data.get("tool")
        if not tool:
            raise ValueError("kind='act' requires a non-empty 'tool'")
        if tool not in self.registry:
            known = ", ".join(sorted(self.registry.names())) or "none"
            raise ValueError(f"unknown tool {tool!r}; available tools: {known}")

        args = data.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError("'args' must be an object")

        # Constrained decoding guarantees shape, not sense: the schema permits
        # an empty args object for a tool that requires arguments. Checking here
        # turns that into a clean retry with the reason appended, instead of a
        # failed tool call the model has to infer the cause of.
        spec = self.registry.get(tool)
        missing = [name for name in (spec.required if spec else [])
                   if name not in args or args[name] is None]
        if missing:
            raise ValueError(
                f"tool {tool!r} requires: {', '.join(missing)}")

        return ar.Act(thought=thought,
                      call=ar.ToolCall(tool=str(tool), args=args))

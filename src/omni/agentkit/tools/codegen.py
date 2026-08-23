"""
`generate_file` — write a file whose content the model produces on demand.

A scaffold expressed as `write_file` steps has to carry every file's complete
text inside the plan, so the plan itself becomes the largest artifact in the
run. One `create a javascript project` request produced a ~10 KB plan that hit
the token cap mid-string; the JSON would not parse, and the planner silently
fell back to an unrelated exploration plan. The user's task was never attempted.

Splitting generation per file fixes that structurally:

    plan  ->  [{tool: generate_file, args: {path, spec}}, ...]   small
    step  ->  one completion per file, each with its own budget   bounded

The plan now carries a path and a sentence per file instead of the files
themselves, and no single completion has to hold the whole project.
"""

from __future__ import annotations

import re

from omni import runtime as ar
from omni.agentkit import jail
from omni.agentkit.registry import ToolOutcome, ToolSpec

__all__ = ["CodeGenerator", "build_spec", "strip_fences", "output_truncated"]

_FENCE_BLOCK = re.compile(r"^\s*```[\w+.-]*[ \t]*\r?\n(.*?)```\s*$", re.DOTALL)

# Defined in the runtime, which needs the same check to tell a cut-off policy
# reply apart from a malformed one. Re-exported here so callers in this layer do
# not have to reach past it.
output_truncated = ar.output_truncated


def strip_fences(text: str) -> str:
    """
    Return file content, with a wrapping markdown fence removed if present.

    Models add fences even when told not to. Writing them into a `.js` file
    produces a syntax error on the first line, so this is not cosmetic.
    """
    match = _FENCE_BLOCK.match(text.strip())
    return match.group(1) if match else text


class CodeGenerator:
    def __init__(self, workspace, client, *, max_tokens: int = 4000) -> None:
        self.workspace = workspace
        self.client = client
        self.max_tokens = max_tokens

    SYSTEM = (
        "You write the complete contents of exactly one file.\n"
        "Output ONLY the file's raw content. No markdown fences, no commentary,\n"
        "no explanation before or after. The output is written to disk verbatim."
    )

    async def generate_file(self, path: str, spec: str,
                            context: str = "") -> ToolOutcome:
        if self.client is None:
            return ToolOutcome(False, 1,
                               "no model backend is connected; use write_file with "
                               "explicit content instead",
                               ar.ErrorClass.POLICY)
        # Refuse an escaping path before spending a completion on it.
        try:
            jail.resolve_in(self.workspace, path)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)

        user = f"File to write: {path}\n\nWhat it must contain:\n{spec}"
        if context:
            user += f"\n\nProject context:\n{context}"

        try:
            completion = await self.client.complete(self.SYSTEM, user,
                                                    max_tokens=self.max_tokens)
        except Exception as exc:                       # noqa: BLE001
            return ToolOutcome(False, 1,
                               f"generation failed for {path}: {exc}",
                               ar.ErrorClass.NETWORK)

        content = strip_fences(completion.text or "")
        if not content.strip():
            return ToolOutcome(False, 1, f"model returned nothing for {path}")

        try:
            target = jail.write_text_in(self.workspace, path, content)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)
        except OSError as exc:
            return ToolOutcome(False, 1, f"cannot write {path}: {exc}",
                               ar.ErrorClass.UNKNOWN)

        note = ""
        if output_truncated(completion):
            # Say so rather than reporting a clean success: the file is on disk
            # but incomplete, and a later verification failure would otherwise
            # look inexplicable.
            note = ("  WARNING: generation stopped at the token limit; this file "
                    "is likely incomplete.")
        return ToolOutcome(
            True, 0,
            f"wrote {target.name} ({len(content)} chars, "
            f"{content.count(chr(10)) + 1} lines).{note}")


def build_spec(workspace, client, *, max_tokens: int = 4000,
               risk: "ar.Risk | None" = None) -> ToolSpec:
    generator = CodeGenerator(workspace, client, max_tokens=max_tokens)
    return ToolSpec(
        name="generate_file",
        description=("Create a file from a short description — the content is "
                     "written for you. Prefer this for any substantial file."),
        params={"type": "object", "additionalProperties": False,
                "properties": {
                    "path": {"type": "string"},
                    "spec": {"type": "string"},
                    "context": {"type": "string"}},
                "required": ["path", "spec"]},
        handler=generator.generate_file,
        risk=risk or ar.Risk.ELEVATED,
        mutating=True,
        timeout_s=180.0,
    )

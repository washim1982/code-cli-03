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

import os
import re
from pathlib import Path

from omni import runtime as ar
from omni.agentkit import jail
from omni.agentkit.registry import ToolOutcome, ToolSpec
from omni.agentkit.tools.fs import IGNORED_DIRS, _is_binary

__all__ = ["CodeGenerator", "build_spec", "strip_fences", "output_truncated",
           "gather_context"]

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


#: How much surrounding source to show the generator. Large enough for an HTML
#: shell plus two sibling modules; small enough to leave room for the output.
CONTEXT_BUDGET = 6_000
_PER_FILE = 2_500


def _referencing_files(workspace: Path, target: Path) -> list[Path]:
    """
    Files that mention the target by name — the ones defining its contract.

    `index.html` carrying `<script src="js/app.js">` is what tells the generator
    which element ids `app.js` must bind to. Nothing else in the run does.
    """
    # Filename and relative path only. The bare stem was too loose: generating
    # `js/app.js` matched every file containing the substring "app", which pulled
    # in a stylesheet with an `.app-container` rule and pushed a sibling module
    # out of the budget.
    needles = {target.name}
    try:
        rel = target.relative_to(workspace)
        needles.add(str(rel).replace(os.sep, "/"))
    except ValueError:
        pass

    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            if candidate == target or _is_binary(candidate):
                continue
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(n and n in text for n in needles):
                hits.append(candidate)
    return hits


def _siblings(target: Path) -> list[Path]:
    """Modules alongside the target, which define the API it must call."""
    parent = target.parent
    if not parent.is_dir():
        return []
    return [p for p in sorted(parent.iterdir())
            if p.is_file() and p != target and p.suffix == target.suffix
            and not _is_binary(p)]


def gather_context(workspace: Path, target: Path, *,
                   budget: int = CONTEXT_BUDGET) -> str:
    """
    Collect the source the generated file has to agree with.

    Deterministic and ordered by relevance: whatever already references the
    target, then its siblings, then the target itself if it exists (so a
    regeneration does not silently drop what was there).
    """
    workspace = Path(workspace)
    ordered: list[Path] = []
    for group in (_referencing_files(workspace, target), _siblings(target)):
        for path in group:
            if path not in ordered:
                ordered.append(path)
    if target.is_file() and target not in ordered:
        ordered.append(target)

    blocks: list[str] = []
    used = 0
    for path in ordered:
        if used >= budget:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        clip = min(_PER_FILE, budget - used)
        if len(text) > clip:
            text = text[:clip] + f"\n... [{len(text) - clip} chars elided]"
        try:
            label = str(path.relative_to(workspace)).replace(os.sep, "/")
        except ValueError:
            label = path.name
        note = " (the file you are rewriting)" if path == target else ""
        blocks.append(f"--- {label}{note} ---\n{text}")
        used += len(text)
    return "\n\n".join(blocks)


class CodeGenerator:
    def __init__(self, workspace, client, *, max_tokens: int = 4000,
                 context_budget: int = CONTEXT_BUDGET) -> None:
        self.workspace = workspace
        self.client = client
        self.max_tokens = max_tokens
        self.context_budget = context_budget

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
            target_path = jail.resolve_in(self.workspace, path)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)

        user = f"File to write: {path}\n\nWhat it must contain:\n{spec}"

        # Ground the generation in the workspace it is being written into.
        # Without this the generating completion sees only the one-sentence
        # spec: not the files the loop just read, not the code this file has to
        # interoperate with. A real run read index.html, then generated a
        # js/app.js that bound to `history-list`, `clear-history` and
        # `history-panel` — none of which existed; the actual ids were
        # `historyList`, `btnClearHistory` and `sidebar`. Four of five bindings
        # missed and the page silently did nothing. The information was in the
        # run and was discarded at this boundary.
        grounding = gather_context(self.workspace, target_path,
                                   budget=self.context_budget)
        if grounding:
            user += ("\n\nEXISTING PROJECT FILES — match these exactly. Element "
                     "ids, exported names, function signatures and paths below are "
                     "authoritative; do not invent alternatives:\n" + grounding)
        if context:
            user += f"\n\nAdditional context:\n{context}"

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
        description=("Create a NEW file from a short description; the content is "
                     "written for you, grounded in the existing files it "
                     "references. Use edit_file to change a file that exists."),
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

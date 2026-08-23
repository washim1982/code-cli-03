"""
Filesystem tools — the capability the agent was missing entirely.

Before this module the only way to touch a file was through the shell
allowlist, and `CommandPolicy` bans `>` and `|` wholesale, so `echo` could not
redirect and nothing could be written. Reading was limited to `cat`, which
returns an unbounded blob with no line numbers and no way to page.

Every path here is resolved through `omni.pathguard`, the same containment used
by `CommandPolicy` for shell arguments — one rule, one implementation. These
tools deliberately do *not* go through `CommandPolicy` itself: they are not
shell strings, so the metacharacter ban is meaningless for them.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from pathlib import Path

from omni import runtime as ar
from omni.agentkit import jail
from omni.agentkit.registry import ToolOutcome, ToolSpec

__all__ = ["FileTools", "build_specs", "IGNORED_DIRS"]

# Directories that are never worth walking: they dominate the file count, none
# of it is the user's code, and on a repo with a venv they would consume the
# whole context budget before reaching a single source file.
IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".tox", ".idea", ".vscode", "site-packages",
})

MAX_READ_LINES = 2_000
MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 100
BINARY_SNIFF_BYTES = 8_192

# Tool output larger than this is cut down by `agent_runtime.finalize_output`,
# which keeps a 600-char head and a 600-char tail and elides everything between.
# That is right for a shell transcript and catastrophic for a file listing: the
# reader is handed a header saying "lines 1-98 of 98" attached to a body with
# its middle removed. A real run did exactly that — the model saw the hole, asked
# for the middle, got another middle-elided response, and burned its entire
# iteration budget re-reading four files without ever writing one.
#
# So these tools must fit inside the budget themselves and describe honestly
# what they actually returned.
OUTPUT_CHAR_BUDGET = 2_600


def _is_binary(path: Path) -> bool:
    """A NUL byte in the first 8 KB is the standard heuristic; it is what git uses."""
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace(os.sep, "/")
    except ValueError:
        return str(path)


class FileTools:
    """Handlers bound to one workspace root."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()

    # -- read --------------------------------------------------------------- #

    async def read_file(self, path: str, offset: int = 1,
                        limit: int = MAX_READ_LINES) -> ToolOutcome:
        return await asyncio.to_thread(self._read_file, path, offset, limit)

    def _read_file(self, path: str, offset: int, limit: int) -> ToolOutcome:
        try:
            target = jail.resolve_in(self.workspace, path)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)

        if not target.exists():
            return ToolOutcome(False, 2, f"no such file: {path}",
                               ar.ErrorClass.MISSING_PATH)
        if target.is_dir():
            return ToolOutcome(False, 21, f"{path} is a directory; use list_dir",
                               ar.ErrorClass.UNKNOWN)
        if _is_binary(target):
            size = target.stat().st_size
            return ToolOutcome(False, 1,
                               f"{path} is a binary file ({size} bytes); refusing to read",
                               ar.ErrorClass.UNKNOWN)

        offset = max(1, int(offset))
        limit = max(1, min(int(limit), MAX_READ_LINES))
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return ToolOutcome(False, 1, f"cannot read {path}: {exc}",
                               ar.ErrorClass.UNKNOWN)

        total = len(lines)
        window = lines[offset - 1: offset - 1 + limit]
        if not window:
            return ToolOutcome(True, 0,
                               f"{path}: {total} lines, nothing at offset {offset}")

        # Line numbers let the model quote a location back in an edit, and make
        # a partial read legible rather than mysterious. Stop at the character
        # budget so the dispatcher never has to elide the middle: a body with a
        # hole in it, under a header claiming completeness, is what sends the
        # reader into a re-read loop.
        rendered: list[str] = []
        used = 0
        for i, line in enumerate(window):
            piece = f"{offset + i:>6}\t{line.rstrip(chr(10))}\n"
            if used + len(piece) > OUTPUT_CHAR_BUDGET and rendered:
                break
            rendered.append(piece)
            used += len(piece)

        shown_to = offset + len(rendered) - 1
        header = f"{path} (lines {offset}-{shown_to} of {total})\n"
        footer = ("" if shown_to >= total else
                  f"\n... {total - shown_to} more lines not shown; "
                  f"continue with offset={shown_to + 1}\n")
        return ToolOutcome(True, 0, header + "".join(rendered) + footer)

    # -- list --------------------------------------------------------------- #

    async def list_dir(self, path: str = ".", glob: str | None = None) -> ToolOutcome:
        return await asyncio.to_thread(self._list_dir, path, glob)

    def _list_dir(self, path: str, glob: str | None) -> ToolOutcome:
        try:
            target = jail.resolve_in(self.workspace, path or ".")
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)

        if not target.exists():
            return ToolOutcome(False, 2, f"no such directory: {path}",
                               ar.ErrorClass.MISSING_PATH)
        if not target.is_dir():
            return ToolOutcome(False, 20, f"{path} is a file; use read_file",
                               ar.ErrorClass.UNKNOWN)

        rows: list[str] = []
        truncated = False
        used = 0
        for entry in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if entry.name in IGNORED_DIRS:
                continue
            if glob and entry.is_file() and not fnmatch.fnmatch(entry.name, glob):
                continue
            if len(rows) >= MAX_LIST_ENTRIES or used > OUTPUT_CHAR_BUDGET:
                truncated = True
                break
            used += len(entry.name) + 24
            if entry.is_dir():
                rows.append(f"{entry.name}/")
            else:
                try:
                    rows.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
                except OSError:
                    rows.append(entry.name)

        if not rows:
            return ToolOutcome(True, 0, f"{path}: empty (or everything filtered out)")
        note = (f"\n... more than {MAX_LIST_ENTRIES} entries; narrow with glob"
                if truncated else "")
        return ToolOutcome(True, 0, f"{path}:\n" + "\n".join(rows) + note)

    # -- search ------------------------------------------------------------- #

    async def search_files(self, pattern: str, glob: str = "*",
                           max_results: int = 50) -> ToolOutcome:
        return await asyncio.to_thread(self._search_files, pattern, glob, max_results)

    def _search_files(self, pattern: str, glob: str, max_results: int) -> ToolOutcome:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolOutcome(False, 22, f"invalid regex {pattern!r}: {exc}",
                               ar.ErrorClass.SYNTAX)

        cap = max(1, min(int(max_results), MAX_SEARCH_RESULTS))
        hits: list[str] = []
        scanned = 0
        used = 0
        # Stop on either limit. Staying inside the character budget keeps the
        # dispatcher from eliding the middle of the result set, which would hide
        # matches while appearing to have reported them all.
        done = False

        for root, dirnames, filenames in os.walk(self.workspace):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for name in sorted(filenames):
                if done:
                    break
                if not fnmatch.fnmatch(name, glob or "*"):
                    continue
                candidate = Path(root) / name
                if _is_binary(candidate):
                    continue
                scanned += 1
                try:
                    with open(candidate, "r", encoding="utf-8", errors="replace") as fh:
                        for lineno, line in enumerate(fh, start=1):
                            if not regex.search(line):
                                continue
                            rel = _relative(candidate, self.workspace)
                            hit = f"{rel}:{lineno}: {line.strip()[:200]}"
                            if used + len(hit) > OUTPUT_CHAR_BUDGET:
                                done = True
                                break
                            hits.append(hit)
                            used += len(hit)
                            if len(hits) >= cap:
                                done = True
                                break
                except OSError:
                    continue
            if done:
                break

        if not hits:
            return ToolOutcome(True, 0,
                               f"no match for {pattern!r} in {scanned} files (glob {glob!r})")
        note = ("\n... more matches exist; narrow the pattern or the glob"
                if done else "")
        return ToolOutcome(True, 0,
                           f"{len(hits)} match(es) for {pattern!r}:\n"
                           + "\n".join(hits) + note)

    # -- write -------------------------------------------------------------- #

    async def write_file(self, path: str, content: str) -> ToolOutcome:
        return await asyncio.to_thread(self._write_file, path, content)

    def _write_file(self, path: str, content: str) -> ToolOutcome:
        try:
            existed = False
            try:
                existed = jail.resolve_in(self.workspace, path).exists()
            except jail.JailBreak:
                raise
            target = jail.write_text_in(self.workspace, path, content)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)
        except OSError as exc:
            return ToolOutcome(False, 1, f"cannot write {path}: {exc}",
                               ar.ErrorClass.UNKNOWN)

        verb = "overwrote" if existed else "created"
        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return ToolOutcome(True, 0,
                           f"{verb} {_relative(target, self.workspace)} "
                           f"({len(content)} chars, {lines} lines)")

    # -- edit --------------------------------------------------------------- #

    async def edit_file(self, path: str, old: str, new: str,
                        replace_all: bool = False) -> ToolOutcome:
        return await asyncio.to_thread(self._edit_file, path, old, new, replace_all)

    def _edit_file(self, path: str, old: str, new: str,
                   replace_all: bool) -> ToolOutcome:
        try:
            target = jail.resolve_in(self.workspace, path)
        except jail.JailBreak as exc:
            return ToolOutcome(False, 126, f"BLOCKED: {exc}", ar.ErrorClass.POLICY)

        if not target.exists() or target.is_dir():
            return ToolOutcome(False, 2, f"no such file: {path}",
                               ar.ErrorClass.MISSING_PATH)
        if not old:
            return ToolOutcome(False, 22, "argument 'old' must not be empty",
                               ar.ErrorClass.SYNTAX)

        try:
            original = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolOutcome(False, 1, f"cannot read {path}: {exc}",
                               ar.ErrorClass.UNKNOWN)

        count = original.count(old)
        if count == 0:
            return ToolOutcome(False, 1,
                               f"no occurrence of the given text in {path}; "
                               "read the file and quote it exactly",
                               ar.ErrorClass.UNKNOWN)
        # Silent multi-replace is the most common way an agent corrupts a file.
        # Refusing turns it into a correctable observation instead of damage.
        if count > 1 and not replace_all:
            return ToolOutcome(False, 1,
                               f"{count} occurrences in {path}; the text is ambiguous. "
                               "Include more surrounding context, or pass "
                               "replace_all=true if every occurrence should change.",
                               ar.ErrorClass.UNKNOWN)

        updated = (original.replace(old, new) if replace_all
                   else original.replace(old, new, 1))
        try:
            jail.write_text_in(self.workspace, path, updated)
        except (jail.JailBreak, OSError) as exc:
            return ToolOutcome(False, 1, f"cannot write {path}: {exc}",
                               ar.ErrorClass.UNKNOWN)

        return ToolOutcome(True, 0,
                           f"edited {path}: replaced {count if replace_all else 1} "
                           f"occurrence(s)")


def build_specs(workspace: Path, *,
                write_risk: "ar.Risk | None" = None) -> list[ToolSpec]:
    """
    Construct the five filesystem tools bound to `workspace`.

    `write_risk` defaults to ELEVATED, which routes every create/edit through
    the runtime's existing operator-approval path. Callers running unattended
    can pass `Risk.SAFE` to skip that — containment via `omni.pathguard` still
    applies either way, so this trades a confirmation prompt for convenience,
    not the perimeter.
    """
    tools = FileTools(workspace)
    write_risk = write_risk or ar.Risk.ELEVATED
    return [
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file with line numbers; page with offset/limit.",
            params={"type": "object", "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"}},
                    "required": ["path"]},
            handler=tools.read_file, risk=ar.Risk.SAFE),
        ToolSpec(
            name="list_dir",
            description="List one directory, skipping .git/venv/node_modules.",
            params={"type": "object", "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "glob": {"type": "string"}},
                    "required": []},
            handler=tools.list_dir, risk=ar.Risk.SAFE),
        ToolSpec(
            name="search_files",
            description="Regex search across workspace text files; returns path:line: text.",
            params={"type": "object", "additionalProperties": False,
                    "properties": {
                        "pattern": {"type": "string"},
                        "glob": {"type": "string"},
                        "max_results": {"type": "integer"}},
                    "required": ["pattern"]},
            handler=tools.search_files, risk=ar.Risk.SAFE, timeout_s=60.0),
        ToolSpec(
            name="write_file",
            description="Create or overwrite a file atomically with the given content.",
            params={"type": "object", "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}},
                    "required": ["path", "content"]},
            handler=tools.write_file, risk=write_risk, mutating=True),
        ToolSpec(
            name="edit_file",
            description=("Replace an exact unique snippet in a file; "
                         "fails if the snippet is not unique."),
            params={"type": "object", "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "replace_all": {"type": "boolean"}},
                    "required": ["path", "old", "new"]},
            handler=tools.edit_file, risk=write_risk, mutating=True),
    ]

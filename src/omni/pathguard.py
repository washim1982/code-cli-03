"""
Workspace path containment — the one security primitive shared by both layers.

This lives at the top level, imports nothing but the standard library, and
imports nothing from this project. That is the point: `CommandPolicy` in
`agent_runtime` needs it to gate shell arguments, and the filesystem tools in
`agentkit` need it to gate tool arguments. It previously lived inside `agentkit`,
so `agent_runtime` imported the `agentkit` package while nine `agentkit` modules
imported `agent_runtime` — a package-level cycle that worked only because
`agentkit/__init__.py` kept everything except `jail` behind lazy imports.

Making it a leaf removes the cycle instead of managing it. The dependency graph
is now strictly layered:

    pathguard  <-  agent_runtime  <-  agentkit

`omni.agentkit.jail` remains as a re-export so existing imports keep working.

The two defects this replaced — the check previously lived inline in
`CommandPolicy.classify`:

  1. It only examined tokens that started with "/" or contained a ".." component.
     A Windows drive-absolute path (`C:/Users/...`) is neither, so containment
     never ran and `cat C:/Users/<user>/.ssh/id_rsa` was ALLOWED at Risk.SAFE.

  2. It compared with `str(resolved).startswith(str(root))` — a string prefix
     test, not a path test. `C:\\ws\\python-evil` prefixes `C:\\ws\\python`, so a
     sibling directory read as contained. Windows case-insensitivity broke it in
     the other direction.

Both are fixed here: every candidate is resolved (which also collapses `..` and
follows symlinks, so a link pointing outside the root is caught), then compared
with `Path.is_relative_to` under `os.path.normcase`.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "JailBreak",
    "contained",
    "resolve_in",
    "is_reserved_name",
    "write_text_in",
]


class JailBreak(ValueError):
    """Raised when a path escapes the workspace root."""


# Windows refuses these as filenames regardless of extension. Creating one is a
# reliable way to produce a file that cannot be opened, listed, or deleted
# through normal means, so they are rejected on every platform for consistency.
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


def is_reserved_name(name: str) -> bool:
    """
    True if `name` is a Windows reserved device name, or ends in a dot/space.

    `.` and `..` are explicitly *not* reserved: they are ordinary traversal, and
    `resolve()` collapses them before the containment test, so `sub/../a.py`
    correctly stays inside the root while `../a.py` correctly escapes it.
    """
    if not name or name in (".", ".."):
        return False
    if name != name.rstrip(". "):
        return True
    stem = name.split(".", 1)[0].upper()
    return stem in _WINDOWS_RESERVED


def _norm(path: Path) -> Path:
    """Case-fold on platforms with case-insensitive filesystems."""
    return Path(os.path.normcase(str(path)))


def contained(path: Path, root: Path) -> bool:
    """
    True if `path` is `root` or lies beneath it.

    Both sides are resolved first, so `..` traversal and symlinks are handled.
    Uses `is_relative_to`, never a string prefix comparison.
    """
    try:
        a = _norm(Path(path).resolve())
        b = _norm(Path(root).resolve())
    except (OSError, ValueError):
        return False
    return a == b or a.is_relative_to(b)


def resolve_in(root: Path, token: str) -> Path:
    """
    Resolve `token` relative to `root` and return it, or raise `JailBreak`.

    An absolute `token` is *not* silently rebased onto the root — that is what
    `Path.__truediv__` does (`workspace / "C:/Windows/x"` discards the workspace
    entirely), and it is how the direct-code writer escaped. Absolute inputs are
    resolved as given and then checked, so they escape loudly instead.
    """
    if token is None or not str(token).strip():
        raise JailBreak("empty path")

    token = str(token).strip()
    candidate = Path(token)

    for part in candidate.parts:
        if is_reserved_name(part):
            raise JailBreak(f"reserved or malformed path component: {part!r}")

    target = candidate if candidate.is_absolute() else (Path(root) / candidate)

    try:
        resolved = target.resolve()
    except (OSError, ValueError) as exc:
        raise JailBreak(f"unresolvable path {token!r}: {exc}") from exc

    if not contained(resolved, Path(root)):
        raise JailBreak(f"path {token!r} escapes the workspace root")

    return resolved


def write_text_in(root: Path, token: str, content: str, *,
                  encoding: str = "utf-8") -> Path:
    """
    Write `content` to `token` under `root`, atomically, or raise `JailBreak`.

    Two guarantees the previous direct-write path did not provide:

      * **Containment.** `workspace / filename` silently discards the workspace
        when `filename` is absolute (`Path("ws") / "C:/Windows/x"` is
        `C:/Windows/x`), and the filename came from a regex over model output.
        `resolve_in` refuses instead of rebasing.

      * **Atomicity.** `Path.write_text` truncates in place, so an interrupted
        write leaves a half-file where a working one used to be. This writes a
        sibling temp file and renames over the target.
    """
    target = resolve_in(root, token)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_name(target.name + f".tmp-{os.getpid()}")
    try:
        with open(tmp, "w", encoding=encoding, newline="") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    return target

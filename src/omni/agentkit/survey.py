"""
Repository survey — give the reviewer the actual code.

`handle_project_review` used to send the model a 3,000-character truncation of
`ls -R` plus a few manifest files. It never read a single source file, so the
"Principal Software Architect" review was structurally blind: it could only
comment on filenames. On a repo containing `venv/` or `node_modules/` the
truncation was consumed entirely by dependency paths before reaching the
project's own code.

This module builds a digest from the same primitives the agent uses — the
filesystem tools' ignore rules and jail — so the review sees a real file tree
and real source, ranked and clipped to a stated character budget.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from omni.agentkit.tools.fs import IGNORED_DIRS, _is_binary

__all__ = ["RepoDigest", "collect_digest", "SOURCE_SUFFIXES"]

SOURCE_SUFFIXES = frozenset({
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".sh", ".sql",
})

# Read first, in this order. These answer "what is this project" faster than any
# amount of source, and they are cheap.
MANIFESTS = (
    "README.md", "readme.md", "README.rst", "pyproject.toml", "setup.py",
    "requirements.txt", "package.json", "Cargo.toml", "go.mod", "Makefile",
    "docker-compose.yml", "Dockerfile",
)

# Conventional entry points, ranked above ordinary modules.
ENTRY_POINTS = (
    "main.py", "app.py", "cli.py", "__main__.py", "server.py", "run.py",
    "index.js", "index.ts", "main.go", "main.rs", "manage.py",
)

MAX_TREE_ENTRIES = 300
DEFAULT_MAX_FILES = 12
DEFAULT_MAX_CHARS = 24_000
PER_FILE_CHARS = 6_000


@dataclass
class RepoDigest:
    root: Path
    tree: list[str] = field(default_factory=list)
    files: list[tuple[str, str]] = field(default_factory=list)   # (path, text)
    skipped: int = 0
    total_files: int = 0

    def tree_text(self) -> str:
        body = "\n".join(self.tree)
        if self.total_files > len(self.tree):
            body += f"\n... {self.total_files - len(self.tree)} more files"
        return body or "(empty)"

    def sources_text(self) -> str:
        if not self.files:
            return "(no readable source files)"
        blocks = [f"--- {path} ---\n{text}" for path, text in self.files]
        return "\n\n".join(blocks)

    def summary(self) -> str:
        return (f"{self.total_files} files, {len(self.files)} read, "
                f"{sum(len(t) for _, t in self.files)} chars")


def _rank(rel: str, size: int) -> tuple[int, int]:
    """Lower sorts first. Manifests, then entry points, then largest source."""
    name = os.path.basename(rel)
    if name in MANIFESTS:
        return (0, MANIFESTS.index(name) if name in MANIFESTS else 99)
    if name in ENTRY_POINTS:
        return (1, ENTRY_POINTS.index(name))
    depth = rel.count("/")
    # Bigger files first within a tier, shallower before deeper.
    return (2 + min(depth, 3), -size)


def collect_digest(workspace: Path, *,
                   max_files: int = DEFAULT_MAX_FILES,
                   max_chars: int = DEFAULT_MAX_CHARS,
                   per_file_chars: int = PER_FILE_CHARS) -> RepoDigest:
    """
    Walk `workspace`, ignoring dependency directories, and read the most
    informative files up to a character budget.
    """
    root = Path(workspace).resolve()
    digest = RepoDigest(root=root)
    if not root.is_dir():
        return digest

    candidates: list[tuple[tuple[int, int], str, Path]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in IGNORED_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            try:
                rel = str(path.relative_to(root)).replace(os.sep, "/")
            except ValueError:
                continue
            digest.total_files += 1
            if len(digest.tree) < MAX_TREE_ENTRIES:
                digest.tree.append(rel)
            suffix = path.suffix.lower()
            if name not in MANIFESTS and suffix not in SOURCE_SUFFIXES:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            candidates.append((_rank(rel, size), rel, path))

    budget = max_chars
    for _, rel, path in sorted(candidates, key=lambda c: c[0]):
        if len(digest.files) >= max_files or budget <= 0:
            digest.skipped += 1
            continue
        if _is_binary(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        clip = min(per_file_chars, budget)
        if len(text) > clip:
            text = text[:clip] + f"\n... [{len(text) - clip} chars elided]"
        budget -= len(text)
        digest.files.append((rel, text))

    return digest

"""
Static contract checks for a browser project — no model, no browser.

`compileall` proves a Python file parses; `pytest` proves tests pass. A static
web project has neither, so a generated `app.js` could bind to element ids that
do not exist and every verifier in the system would report success. That is
exactly what happened: a run wrote a valid, correctly-placed `js/app.js` whose
listeners attached to `history-list`, `clear-history` and `history-panel` while
the page defined `historyList`, `btnClearHistory` and `sidebar`. The page
rendered and silently did nothing.

None of that needs a language model. Whether a referenced file exists, and
whether an id a script reaches for is defined anywhere, are facts.

Run as a command so it slots into the existing verification contract:

    python -m omni.webcheck [path]

Exit code 0 when clean, 1 when something is wrong, 2 when there is nothing to
check. Findings go to stdout in the order they were found.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

__all__ = ["Finding", "run_checks", "main", "SKIP_DIRS"]

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", "dist", "build", ".idea", ".vscode",
})

#: Attribute values that never point at a local file.
_EXTERNAL = ("http://", "https://", "//", "data:", "mailto:", "tel:", "javascript:")

_ASSET_SUFFIXES = frozenset({
    ".js", ".mjs", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".ico", ".woff", ".woff2", ".ttf", ".json",
})


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str
    where: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}  [{self.where}]"


class _Html(HTMLParser):
    """
    Collects ids, local asset references, and anchor targets.

    A real parser rather than a regex: attribute order, single vs double quotes,
    and attributes spread over several lines all break the naive pattern, and a
    false negative here means a broken page is reported as verified.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.assets: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if a.get("id"):
            self.ids.add(a["id"])

        for key in ("src", "href"):
            value = a.get(key, "").strip()
            if not value or value.startswith(_EXTERNAL) or value.startswith("#"):
                continue
            clean = value.split("?")[0].split("#")[0]
            if not clean:
                continue
            if tag == "a" and key == "href":
                self.links.append(clean)
            elif Path(clean).suffix.lower() in _ASSET_SUFFIXES:
                self.assets.append(clean)


# References a script makes to the DOM.
_GET_BY_ID = re.compile(r"""getElementById\(\s*['"]([^'"]+)['"]""")
_QUERY_ID = re.compile(r"""querySelector(?:All)?\(\s*['"]#([A-Za-z0-9_\-]+)""")
# Ids a script *creates*, which are legitimately absent from the HTML.
_ASSIGNS_ID = re.compile(r"""\.id\s*=\s*['"]([^'"]+)['"]""")
_ID_IN_TEMPLATE = re.compile(r"""\bid\s*=\s*\\?["']([A-Za-z0-9_\-]+)""")


def _walk(root: Path, suffix: str) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob(f"*{suffix}")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            out.append(path)
    return out


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def run_checks(workspace: Path) -> list[Finding]:
    """Every static inconsistency we can prove without running anything."""
    workspace = Path(workspace).resolve()
    pages = _walk(workspace, ".html")
    if not pages:
        return []

    findings: list[Finding] = []
    defined_ids: set[str] = set()
    parsed: dict[Path, _Html] = {}

    for page in pages:
        parser = _Html()
        try:
            parser.feed(_read(page))
        except Exception:                                   # noqa: BLE001
            # A malformed page is a finding, not a crash.
            findings.append(Finding("unparsable-html",
                                    "could not parse this page", _rel(page, workspace)))
            continue
        parsed[page] = parser
        defined_ids |= parser.ids

    # 1. Every referenced asset resolves to a file that exists.
    for page, parser in parsed.items():
        base = page.parent
        for ref in parser.assets:
            target = (base / ref) if not ref.startswith("/") else (workspace / ref.lstrip("/"))
            if not target.is_file():
                findings.append(Finding("missing-asset", f"{ref} does not exist",
                                        _rel(page, workspace)))

    # 2. Every local page an anchor points at exists.
    for page, parser in parsed.items():
        base = page.parent
        for ref in parser.links:
            target = (base / ref) if not ref.startswith("/") else (workspace / ref.lstrip("/"))
            if not (target.is_file() or target.is_dir()):
                findings.append(Finding("broken-link", f"{ref} does not exist",
                                        _rel(page, workspace)))

    # 3. Every id a script reaches for is defined somewhere.
    #
    # Ids a script creates at runtime count as defined — flagging those would
    # fail a page that works, and a verifier that cries wolf gets switched off.
    scripts = _walk(workspace, ".js") + _walk(workspace, ".mjs")
    created: set[str] = set()
    for script in scripts:
        text = _read(script)
        created |= set(_ASSIGNS_ID.findall(text))
        created |= set(_ID_IN_TEMPLATE.findall(text))
    known = defined_ids | created

    for script in scripts:
        text = _read(script)
        referenced = set(_GET_BY_ID.findall(text)) | set(_QUERY_ID.findall(text))
        for name in sorted(referenced - known):
            findings.append(Finding(
                "dangling-id",
                f"#{name} is referenced but no element defines it",
                _rel(script, workspace)))

    return findings


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `--browser` additionally loads the pages in a real browser. The two checks
    # are complementary, not redundant: static analysis catches a dangling id
    # that is guarded and so never throws, while the browser catches a handler
    # that throws on a page whose source looks fine.
    with_browser = "--browser" in argv
    argv = [a for a in argv if not a.startswith("--")]
    workspace = Path(argv[0]) if argv else Path.cwd()

    if not workspace.is_dir():
        print(f"webcheck: {workspace} is not a directory")
        return 2

    pages = _walk(workspace.resolve(), ".html")
    if not pages:
        print("webcheck: no HTML in this workspace; nothing to check")
        return 2

    findings = run_checks(workspace)
    if findings:
        print(f"webcheck: {len(findings)} static problem(s) in {len(pages)} page(s)\n")
        for finding in findings:
            print(f"  {finding}")
    else:
        print(f"webcheck: {len(pages)} page(s) static-clean — "
              "every asset resolves and every referenced id exists")

    if with_browser:
        # Imported here so the static check keeps working with no Playwright,
        # and so this module stays importable by `browsercheck` itself.
        from omni.browsercheck import check_workspace
        print()
        try:
            runtime_findings = check_workspace(workspace)
        except RuntimeError as exc:
            print(f"browsercheck: skipped — {exc}")
            runtime_findings = []
        else:
            if runtime_findings:
                print(f"browsercheck: {len(runtime_findings)} runtime problem(s)\n")
                for finding in runtime_findings:
                    print(f"  {finding}")
            else:
                print("browsercheck: pages load and every control clicks cleanly")
        findings = findings + runtime_findings

    if findings:
        print("\nFix these before reporting success: each one is a real failure "
              "in the browser.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

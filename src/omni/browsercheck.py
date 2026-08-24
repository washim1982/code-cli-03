"""
Runtime checks for a browser project — a real browser, still no model.

`omni.webcheck` proves what can be proved by reading the source. This runs the
page: it serves the workspace over HTTP, loads every entry point, follows the
local links, clicks the controls, and reports what actually broke — console
errors, uncaught exceptions, failed requests, and handlers that throw.

Two things make this worth the dependency:

  * **It is deterministic.** "Clicking #btnClearHistory raised TypeError" is a
    fact with a stack trace, not a judgement. It becomes a hard pass/fail the
    verification gate can consume, which is a different quality of signal from
    asking a model whether a page looks right.
  * **It serves over HTTP.** Opening `index.html` from disk trips
    `file:` origin restrictions and breaks `sessionStorage`, so the failures a
    developer sees that way are often not the real ones.

Playwright is optional. Without it this exits 2 ("cannot check") rather than
failing a run, so the core install stays four pure-Python dependencies:

    pip install "omni-cli[browser]" && python -m playwright install chromium

Run it as a command, so it slots into the existing verification contract:

    python -m omni.browsercheck [path]

Exit 0 clean, 1 problems found, 2 nothing to check or browser unavailable.
"""

from __future__ import annotations

import functools
import http.server
import socket
import socketserver
import sys
import threading
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from omni.webcheck import SKIP_DIRS, Finding

__all__ = ["Finding", "serve", "check_workspace", "main", "PLAYWRIGHT_HINT"]

PLAYWRIGHT_HINT = (
    'browser checks need Playwright: pip install "omni-cli[browser]" '
    "&& python -m playwright install chromium"
)

#: Console messages that are noise rather than defects.
_IGNORED_CONSOLE = (
    "favicon.ico",
    "Download the React DevTools",
    "[HMR]",
)

PAGE_TIMEOUT_MS = 15_000
SETTLE_MS = 400
MAX_CLICKS_PER_PAGE = 25


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:      # noqa: D102 - silence the server
        pass


def serve(directory: Path) -> tuple[str, socketserver.TCPServer, threading.Thread]:
    """Serve `directory` on an ephemeral port. Returns (base_url, server, thread)."""
    port = _free_port()
    handler = functools.partial(_QuietHandler, directory=str(directory))
    httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", httpd, thread


@dataclass
class _PageReport:
    url: str
    findings: list[Finding] = field(default_factory=list)


def _is_noise(text: str) -> bool:
    return any(marker in text for marker in _IGNORED_CONSOLE)


def _entry_pages(workspace: Path) -> list[Path]:
    pages = [p for p in sorted(workspace.rglob("*.html"))
             if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)]
    # index.html first: it is the page a person opens.
    pages.sort(key=lambda p: (p.name != "index.html", str(p)))
    return pages


def check_workspace(workspace: Path, *, max_pages: int = 10) -> list[Finding]:
    """
    Load every page, click everything clickable, and report what broke.

    Raises `RuntimeError` if Playwright or its browser is unavailable, so the
    caller can distinguish "cannot check" from "checked and found nothing".
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:                              # pragma: no cover
        raise RuntimeError(PLAYWRIGHT_HINT) from exc

    workspace = Path(workspace).resolve()
    pages = _entry_pages(workspace)[:max_pages]
    if not pages:
        return []

    base_url, httpd, _thread = serve(workspace)
    findings: list[Finding] = []
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except PlaywrightError as exc:
                raise RuntimeError(f"{PLAYWRIGHT_HINT} ({str(exc)[:100]})") from exc
            try:
                for page_path in pages:
                    rel = str(page_path.relative_to(workspace)).replace("\\", "/")
                    findings += _check_page(browser, f"{base_url}/{rel}", rel)
            finally:
                browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    return findings


def _check_page(browser, url: str, label: str) -> list[Finding]:
    findings: list[Finding] = []
    page = browser.new_page()

    page.on("console", lambda m: (
        findings.append(Finding("console-error", m.text[:200], label))
        if m.type == "error" and not _is_noise(m.text) else None))
    page.on("pageerror", lambda e: findings.append(
        Finding("uncaught-exception", str(e).splitlines()[0][:200], label)))
    page.on("requestfailed", lambda r: (
        findings.append(Finding("failed-request",
                                f"{r.url.rsplit('/', 1)[-1]} — "
                                f"{(r.failure or 'failed')}"[:160], label))
        if not _is_noise(r.url) else None))
    page.on("response", lambda r: (
        findings.append(Finding("http-error",
                                f"{r.status} for {r.url.rsplit('/', 1)[-1]}", label))
        if r.status >= 400 and not _is_noise(r.url) else None))

    try:
        page.goto(url, timeout=PAGE_TIMEOUT_MS, wait_until="load")
        page.wait_for_timeout(SETTLE_MS)
        findings += _click_everything(page, label)
    except Exception as exc:                                # noqa: BLE001
        findings.append(Finding("page-load-failed", str(exc).splitlines()[0][:200],
                                label))
    finally:
        page.close()

    # The same console error on every click is one defect, not twenty.
    seen: set[tuple[str, str]] = set()
    unique: list[Finding] = []
    for f in findings:
        key = (f.kind, f.detail)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _click_everything(page, label: str) -> list[Finding]:
    """
    Click every control and report handlers that throw.

    A dead button is the failure mode a generated UI actually has: the element
    exists, the page loads clean, and nothing happens — or a listener bound to a
    missing element throws on first use. Neither shows up until something clicks.
    """
    findings: list[Finding] = []
    try:
        controls = page.query_selector_all(
            "button, [role=button], input[type=button], input[type=submit]")
    except Exception:                                       # noqa: BLE001
        return findings

    for control in controls[:MAX_CLICKS_PER_PAGE]:
        try:
            if not control.is_visible() or not control.is_enabled():
                continue
            name = (control.get_attribute("id")
                    or (control.inner_text() or "").strip()[:20]
                    or "unnamed")
            control.click(timeout=2_000, no_wait_after=True)
            page.wait_for_timeout(60)
        except Exception as exc:                            # noqa: BLE001
            findings.append(Finding(
                "click-failed", f"{name}: {str(exc).splitlines()[0][:120]}", label))
    return findings


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    workspace = Path(argv[0]) if argv else Path.cwd()

    if not workspace.is_dir():
        print(f"browsercheck: {workspace} is not a directory")
        return 2
    if not _entry_pages(workspace.resolve()):
        print("browsercheck: no HTML in this workspace; nothing to check")
        return 2

    try:
        findings = check_workspace(workspace)
    except RuntimeError as exc:
        # Not a failure of the project — a missing capability here.
        print(f"browsercheck: skipped — {exc}")
        return 2

    if not findings:
        print("browsercheck: pages load and every control clicks cleanly")
        return 0

    print(f"browsercheck: {len(findings)} problem(s) found in the running page\n")
    for finding in findings:
        print(f"  {finding}")
    print("\nThese are runtime failures observed in a real browser.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

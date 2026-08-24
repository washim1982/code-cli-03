"""
Tests for the web verification tiers.

Context: a static web project has no test suite, so nothing checked that
generated JavaScript agreed with the page it runs in. A run produced a valid,
correctly-placed `js/app.js` whose listeners bound to `history-list`,
`clear-history` and `history-panel` while the page defined `historyList`,
`btnClearHistory` and `sidebar`. Every verifier in the system reported success
and the page silently did nothing.

Tier 1 (`omni.webcheck`) proves what can be proved by reading. Tier 2
(`omni.browsercheck`) proves the rest by running it. Neither needs a model.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from omni import webcheck
from omni.agentkit.verify import (
    COMPILEALL,
    PYTEST,
    WEBCHECK,
    WEBCHECK_BROWSER,
    browser_available,
    detect_verify,
)

BROWSER = browser_available()
needs_browser = pytest.mark.skipif(not BROWSER, reason="playwright/chromium unavailable")


GOOD_HTML = """<!DOCTYPE html><html><body>
  <aside id="sidebar"><button id="btnClearHistory">Clear</button>
    <ul id="historyList"></ul></aside>
  <div id="display">0</div>
  <script src="js/app.js"></script>
</body></html>
"""

GOOD_JS = """document.getElementById('display');
document.getElementById('historyList');
document.getElementById('btnClearHistory');
"""


def project(root: Path, html: str = GOOD_HTML, js: str = GOOD_JS) -> Path:
    (root / "js").mkdir(exist_ok=True)
    (root / "index.html").write_text(html, encoding="utf-8")
    (root / "js" / "app.js").write_text(js, encoding="utf-8")
    return root


def kinds(findings) -> list[str]:
    return [f.kind for f in findings]


# ---------------------------------------------------------------------------
# tier 1 — static
# ---------------------------------------------------------------------------

class TestStaticChecks:
    def test_a_consistent_project_is_clean(self, tmp_path):
        assert webcheck.run_checks(project(tmp_path)) == []

    def test_the_original_bug_is_caught(self, tmp_path):
        """The exact mismatch that shipped: kebab-case ids that never existed."""
        bad = ("document.getElementById('history-list');\n"
               "document.getElementById('clear-history');\n"
               "document.getElementById('history-panel');\n")
        findings = webcheck.run_checks(project(tmp_path, js=bad))
        assert kinds(findings) == ["dangling-id"] * 3
        reported = " ".join(str(f) for f in findings)
        for name in ("#history-list", "#clear-history", "#history-panel"):
            assert name in reported

    def test_findings_are_reported_in_a_stable_order(self, tmp_path):
        """Sorted, so a repeated run produces a byte-identical report."""
        bad = ("document.getElementById('zeta');\n"
               "document.getElementById('alpha');\n")
        findings = webcheck.run_checks(project(tmp_path, js=bad))
        assert [f.detail.split()[0] for f in findings] == ["#alpha", "#zeta"]

    def test_missing_script_is_caught(self, tmp_path):
        root = project(tmp_path)
        (root / "js" / "app.js").unlink()
        assert "missing-asset" in kinds(webcheck.run_checks(root))

    def test_missing_stylesheet_is_caught(self, tmp_path):
        html = GOOD_HTML.replace("<body>", '<body><link href="styles.css" rel="stylesheet">')
        assert "missing-asset" in kinds(webcheck.run_checks(project(tmp_path, html=html)))

    def test_broken_local_link_is_caught(self, tmp_path):
        html = GOOD_HTML.replace("<body>", '<body><a href="about.html">About</a>')
        assert "broken-link" in kinds(webcheck.run_checks(project(tmp_path, html=html)))

    def test_external_urls_are_not_checked(self, tmp_path):
        html = GOOD_HTML.replace(
            "<body>", '<body><a href="https://example.com">x</a>'
                      '<script src="https://cdn.example.com/x.js"></script>')
        assert webcheck.run_checks(project(tmp_path, html=html)) == []

    def test_anchor_and_query_fragments_are_ignored(self, tmp_path):
        html = GOOD_HTML.replace("<body>", '<body><a href="#top">top</a>')
        assert webcheck.run_checks(project(tmp_path, html=html)) == []

    def test_ids_created_at_runtime_are_not_dangling(self, tmp_path):
        """Flagging these would fail a page that works."""
        js = ("var el = document.createElement('div');\n"
              "el.id = 'toast';\n"
              "document.getElementById('toast');\n") + GOOD_JS
        assert webcheck.run_checks(project(tmp_path, js=js)) == []

    def test_ids_in_template_markup_are_not_dangling(self, tmp_path):
        js = ('container.innerHTML = `<div id="row-1"></div>`;\n'
              "document.getElementById('row-1');\n") + GOOD_JS
        assert webcheck.run_checks(project(tmp_path, js=js)) == []

    def test_query_selector_ids_are_checked(self, tmp_path):
        js = GOOD_JS + "document.querySelector('#nope');\n"
        assert "dangling-id" in kinds(webcheck.run_checks(project(tmp_path, js=js)))

    def test_class_selectors_are_not_treated_as_ids(self, tmp_path):
        js = GOOD_JS + "document.querySelector('.some-class');\n"
        assert webcheck.run_checks(project(tmp_path, js=js)) == []

    def test_ids_may_be_defined_in_any_page(self, tmp_path):
        root = project(tmp_path)
        (root / "other.html").write_text('<html><body><div id="shared"></div></body></html>',
                                         encoding="utf-8")
        (root / "js" / "app.js").write_text(GOOD_JS + "document.getElementById('shared');",
                                            encoding="utf-8")
        assert webcheck.run_checks(root) == []

    def test_dependency_directories_are_skipped(self, tmp_path):
        root = project(tmp_path)
        vendor = root / "node_modules"
        vendor.mkdir()
        (vendor / "lib.js").write_text("document.getElementById('vendor-only');",
                                       encoding="utf-8")
        assert webcheck.run_checks(root) == []

    def test_no_html_means_nothing_to_check(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        assert webcheck.run_checks(tmp_path) == []

    def test_attributes_survive_odd_formatting(self, tmp_path):
        """A regex over raw HTML misses these; a parser does not."""
        html = ("<html><body><div\n   class='x'\n   id=sidebar>\n</div>\n"
                "<script src='js/app.js'></script></body></html>")
        js = "document.getElementById('sidebar');"
        assert webcheck.run_checks(project(tmp_path, html=html, js=js)) == []


class TestStaticExitCodes:
    def test_clean_project_exits_zero(self, tmp_path):
        assert webcheck.main([str(project(tmp_path))]) == 0

    def test_problems_exit_one(self, tmp_path):
        bad = "document.getElementById('nope');"
        assert webcheck.main([str(project(tmp_path, js=bad))]) == 1

    def test_nothing_to_check_exits_two(self, tmp_path):
        assert webcheck.main([str(tmp_path)]) == 2

    def test_missing_directory_exits_two(self, tmp_path):
        assert webcheck.main([str(tmp_path / "nope")]) == 2


# ---------------------------------------------------------------------------
# tier 2 — real browser
# ---------------------------------------------------------------------------

class TestServer:
    def test_workspace_is_served_over_http(self, tmp_path):
        project(tmp_path)
        base, httpd, _ = webcheck_serve(tmp_path)
        try:
            body = urllib.request.urlopen(f"{base}/index.html", timeout=5).read()
            assert b"sidebar" in body
        finally:
            httpd.shutdown()
            httpd.server_close()


def webcheck_serve(path):
    from omni.browsercheck import serve
    return serve(path)


@needs_browser
class TestBrowserChecks:
    def test_a_working_page_is_clean(self, tmp_path):
        from omni.browsercheck import check_workspace
        assert check_workspace(project(tmp_path)) == []

    def test_a_missing_script_is_reported(self, tmp_path):
        from omni.browsercheck import check_workspace
        root = project(tmp_path)
        (root / "js" / "app.js").unlink()
        found = kinds(check_workspace(root))
        assert "http-error" in found or "failed-request" in found

    def test_an_uncaught_exception_is_reported(self, tmp_path):
        from omni.browsercheck import check_workspace
        js = "document.getElementById('missing').textContent = 'x';"
        findings = check_workspace(project(tmp_path, js=js))
        assert "uncaught-exception" in kinds(findings)

    def test_a_throwing_click_handler_is_reported(self, tmp_path):
        """The failure a generated UI actually has: the page loads, the button is dead."""
        from omni.browsercheck import check_workspace
        js = GOOD_JS + (
            "document.getElementById('btnClearHistory')"
            ".addEventListener('click', function () { boom(); });\n")
        findings = check_workspace(project(tmp_path, js=js))
        assert "uncaught-exception" in kinds(findings)

    def test_findings_are_deduplicated(self, tmp_path):
        from omni.browsercheck import check_workspace
        js = "document.getElementById('missing').textContent = 'x';"
        findings = check_workspace(project(tmp_path, js=js))
        assert len({(f.kind, f.detail) for f in findings}) == len(findings)

    def test_no_pages_yields_no_findings(self, tmp_path):
        from omni.browsercheck import check_workspace
        assert check_workspace(tmp_path) == []


# ---------------------------------------------------------------------------
# wiring into the verification gate
# ---------------------------------------------------------------------------

class TestVerifierSelection:
    def test_a_web_project_selects_a_web_check(self, tmp_path):
        assert detect_verify(project(tmp_path)) in (WEBCHECK, WEBCHECK_BROWSER)

    def test_the_browser_variant_is_chosen_when_available(self, tmp_path):
        expected = WEBCHECK_BROWSER if BROWSER else WEBCHECK
        assert detect_verify(project(tmp_path)) == expected

    def test_python_tests_still_win(self, tmp_path):
        project(tmp_path)
        (tmp_path / "test_x.py").write_text("def test_x(): pass", encoding="utf-8")
        assert detect_verify(tmp_path) == PYTEST

    def test_a_python_project_is_unaffected(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1", encoding="utf-8")
        assert detect_verify(tmp_path) == COMPILEALL

    def test_the_command_is_allowed_by_the_command_policy(self, tmp_path):
        """The agent has to be able to actually run its own verifier."""
        from omni import runtime as ar
        for spec in (WEBCHECK, WEBCHECK_BROWSER):
            verdict = ar.CommandPolicy(tmp_path).classify(spec.command)
            assert verdict.risk is not ar.Risk.FORBIDDEN, spec.command

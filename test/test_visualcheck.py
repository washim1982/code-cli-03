"""
Tests for Tier 3 — visual review, and the UI_TEST intent that triggers it.

Design under test: geometry is fact, the model is not. Measured on this machine
the local vision model produced one correct observation and one invented one from
two attempts, and once claimed it had received no image at all. So Tier 3 never
gates a run, claims contradicted by geometry are dropped, and claims geometry
cannot speak to are labelled `plausible` rather than stated.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from omni import cli, visualcheck
from omni.agentkit.verify import browser_available
from omni.visualcheck import PageGeometry, VisualFinding, deterministic_findings

BROWSER = browser_available()
needs_browser = pytest.mark.skipif(not BROWSER, reason="playwright/chromium unavailable")


def run(coro):
    return asyncio.run(coro)


def element(**kw) -> dict:
    base = dict(sel="div", tag="div", id="", text="", ownText="", x=0, y=0,
                w=100, h=20, scrollW=100, clientW=100, overflowX="visible",
                overflowY="visible", contrast=None, interactive=False, depth=3)
    base.update(kw)
    return base


def geometry(*elements, width=1100, height=800) -> PageGeometry:
    return PageGeometry(viewport={"width": width, "height": height},
                        doc_height=height, elements=list(elements))


def kinds(findings) -> list[str]:
    return [f.kind for f in findings]


# ---------------------------------------------------------------------------
# deterministic checks — the half that needs no model
# ---------------------------------------------------------------------------

class TestDeterministicChecks:
    def test_a_clean_page_yields_nothing(self):
        assert deterministic_findings(geometry(element(ownText="hi", contrast=9.0)),
                                      "index.html") == []

    def test_clipped_text_is_found(self):
        el = element(ownText="a long label", scrollW=260, clientW=100,
                     overflowX="hidden")
        assert "clipped-text" in kinds(deterministic_findings(geometry(el), "p.html"))

    def test_scrollable_overflow_is_not_clipped(self):
        """A container the user can scroll is not a defect."""
        el = element(ownText="x", scrollW=260, clientW=100, overflowX="auto")
        assert deterministic_findings(geometry(el), "p.html") == []

    def test_offscreen_to_the_right_is_found(self):
        el = element(x=1400, w=120)
        assert "offscreen" in kinds(deterministic_findings(geometry(el), "p.html"))

    def test_offscreen_to_the_left_is_found(self):
        el = element(x=-500, w=100)
        assert "offscreen" in kinds(deterministic_findings(geometry(el), "p.html"))

    def test_below_the_fold_is_not_offscreen(self):
        """Vertical overflow is just a page that scrolls."""
        el = element(x=10, y=5000, w=100, h=20)
        assert deterministic_findings(geometry(el), "p.html") == []

    def test_zero_size_control_is_found(self):
        el = element(tag="button", interactive=True, w=0, h=0)
        assert "zero-size-control" in kinds(deterministic_findings(geometry(el), "p.html"))

    def test_low_contrast_is_found(self):
        el = element(ownText="Clear", contrast=3.07, id="btnClear")
        found = deterministic_findings(geometry(el), "p.html")
        assert "low-contrast" in kinds(found)
        assert "3.07" in found[0].detail

    def test_contrast_at_the_threshold_passes(self):
        el = element(ownText="ok", contrast=4.5)
        assert deterministic_findings(geometry(el), "p.html") == []

    def test_contrast_is_ignored_without_text(self):
        el = element(ownText="", contrast=1.0)
        assert deterministic_findings(geometry(el), "p.html") == []

    def test_overlapping_controls_are_found(self):
        a = element(tag="button", id="a", interactive=True, x=0, y=0, w=100, h=40)
        b = element(tag="button", id="b", interactive=True, x=10, y=0, w=100, h=40)
        assert "overlapping-controls" in kinds(
            deterministic_findings(geometry(a, b), "p.html"))

    def test_adjacent_controls_do_not_overlap(self):
        a = element(tag="button", id="a", interactive=True, x=0, y=0, w=100, h=40)
        b = element(tag="button", id="b", interactive=True, x=110, y=0, w=100, h=40)
        assert deterministic_findings(geometry(a, b), "p.html") == []

    def test_findings_carry_an_element_label(self):
        el = element(ownText="x", contrast=2.0, id="btnGo")
        assert deterministic_findings(geometry(el), "p.html")[0].selector == "#btnGo"


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_repeated_defects_collapse(self):
        """Nine buttons from one CSS rule are one defect, not nine lines."""
        found = [VisualFinding("low-contrast", "ratio 3.07", "p.html", "button.op")
                 for _ in range(9)]
        grouped = visualcheck.group_findings(found)
        assert len(grouped) == 1
        assert "9 elements" in grouped[0].detail

    def test_a_shared_selector_is_not_listed_twice(self):
        found = [VisualFinding("low-contrast", "r", "p.html", "button.op")] * 4
        assert "button.op," not in visualcheck.group_findings(found)[0].detail

    def test_distinct_selectors_are_enumerated(self):
        found = [VisualFinding("low-contrast", "r", "p.html", f"#b{i}")
                 for i in range(3)]
        detail = visualcheck.group_findings(found)[0].detail
        assert "#b0" in detail and "#b1" in detail

    def test_different_defects_stay_separate(self):
        found = [VisualFinding("low-contrast", "a", "p.html"),
                 VisualFinding("offscreen", "b", "p.html")]
        assert len(visualcheck.group_findings(found)) == 2

    def test_a_single_finding_is_untouched(self):
        found = [VisualFinding("offscreen", "b", "p.html", "#x")]
        assert visualcheck.group_findings(found) == found


# ---------------------------------------------------------------------------
# vision claims are cross-checked, never trusted
# ---------------------------------------------------------------------------

class _Vision:
    def __init__(self, claims): self.claims = claims
    def ask_json(self, png, prompt): return self.claims


class TestVisionCrossCheck:
    def _check(self, claims, geo):
        return visualcheck.vision_findings(b"png", geo, "p.html", _Vision(claims))[0]

    def test_a_claim_about_a_real_element_is_confirmed(self):
        geo = geometry(element(id="sidebar", ownText="History"))
        found = self._check([{"kind": "alignment", "element": "sidebar",
                              "detail": "misaligned"}], geo)
        assert found[0].confidence == "confirmed"

    def test_a_claim_about_nothing_is_only_plausible(self):
        """The model invented a 'search box'; the page has none."""
        geo = geometry(element(id="sidebar"))
        found = self._check([{"kind": "overlap", "element": "search box",
                              "detail": "overlaps the nav"}], geo)
        assert found[0].confidence == "plausible"

    def test_a_contrast_claim_the_geometry_refutes_is_dropped(self):
        geo = geometry(element(id="ok", ownText="hi", contrast=12.0))
        assert self._check([{"kind": "contrast", "element": "ok",
                             "detail": "too faint"}], geo) == []

    def test_a_clipping_claim_the_geometry_refutes_is_dropped(self):
        geo = geometry(element(id="ok", scrollW=100, clientW=100))
        assert self._check([{"kind": "clipped", "element": "ok",
                             "detail": "text cut off"}], geo) == []

    def test_claims_without_detail_are_dropped(self):
        assert self._check([{"kind": "overlap", "element": "x"}], geometry()) == []

    def test_at_most_four_claims_are_kept(self):
        claims = [{"kind": "layout", "element": "", "detail": f"d{i}"}
                  for i in range(9)]
        assert len(self._check(claims, geometry())) == 4

    def test_an_unavailable_model_is_a_note_not_a_failure(self):
        from omni.vision import VisionUnavailable

        class _Broken:
            def ask_json(self, png, prompt):
                raise VisionUnavailable("returned nothing")

        found, note = visualcheck.vision_findings(b"p", geometry(), "p.html", _Broken())
        assert found == [] and "returned nothing" in note


# ---------------------------------------------------------------------------
# end to end, in a real browser
# ---------------------------------------------------------------------------

CLIPPED = """<!DOCTYPE html><html><body>
<div id="box" style="width:60px;overflow:hidden;white-space:nowrap;
     background:#fff;color:#000">A very long line of text indeed</div>
<button id="faint" style="color:#999;background:#aaa">Faint</button>
</body></html>"""


@needs_browser
class TestAgainstARealBrowser:
    def _report(self, tmp_path, html):
        (tmp_path / "index.html").write_text(html, encoding="utf-8")
        return visualcheck.check_workspace(tmp_path, use_vision=False)

    def test_a_clean_page_reports_nothing(self, tmp_path):
        html = ('<html><body><p style="color:#000;background:#fff">Hello</p>'
                "</body></html>")
        assert self._report(tmp_path, html).findings == []

    def test_clipped_text_is_measured_in_a_real_page(self, tmp_path):
        assert "clipped-text" in kinds(self._report(tmp_path, CLIPPED).findings)

    def test_low_contrast_is_measured_in_a_real_page(self, tmp_path):
        assert "low-contrast" in kinds(self._report(tmp_path, CLIPPED).findings)

    def test_a_screenshot_is_captured(self, tmp_path):
        report = self._report(tmp_path, CLIPPED)
        assert report.screenshots and report.screenshots[0].stat().st_size > 500

    def test_vision_is_optional(self, tmp_path):
        """Geometry checks must work with no model present at all."""
        report = self._report(tmp_path, CLIPPED)
        assert report.pages_checked == 1

    def test_the_module_never_fails_the_caller(self, tmp_path):
        """Tier 3 is advisory: a defect must not produce a non-zero exit."""
        (tmp_path / "index.html").write_text(CLIPPED, encoding="utf-8")
        assert visualcheck.main([str(tmp_path), "--no-vision"]) == 0


# ---------------------------------------------------------------------------
# the trigger
# ---------------------------------------------------------------------------

class TestUiTestIntent:
    @pytest.mark.parametrize("prompt", [
        "test the ui",
        "check the UI for issues",
        "open the app in browser and find issues",
        "is the layout broken?",
        "test all the links on the page",
        "check accessibility",
        "verify the frontend renders correctly",
        "the css looks wrong",
    ])
    def test_ui_requests_are_recognised(self, prompt):
        assert cli.is_ui_test_request(prompt.lower())

    @pytest.mark.parametrize("prompt", [
        "create a page for login",
        "write a script to add numbers",
        "build a dashboard application",
        "generate a css file",
        "explain how the browser event loop works",
        "add a button to the page",
    ])
    def test_creation_requests_are_not(self, prompt):
        assert not cli.is_ui_test_request(prompt.lower())

    def test_it_routes_to_the_ui_test_intent(self):
        assert run(cli.classify_intent("test the ui", None)) == "UI_TEST"

    def test_creating_a_page_still_routes_to_code(self):
        assert run(cli.classify_intent("create a page for login", None)) == "DIRECT_CODE"

    def test_project_review_is_unaffected(self):
        assert run(cli.classify_intent("review project", None)) == "PROJECT_REVIEW"

    def test_the_llm_classifier_offers_the_intent(self):
        """The heuristic is a fast path; the model must know the option exists."""
        import inspect
        source = inspect.getsource(cli.classify_intent)
        assert '"UI_TEST"' in source

    def test_the_handler_runs_with_no_pages(self, tmp_path):
        verdict = run(cli.handle_ui_test("test the ui", tmp_path))
        assert "static clean" in verdict


# ---------------------------------------------------------------------------
# terminal rendering
# ---------------------------------------------------------------------------

class TestMarkupEscaping:
    """
    Rich reads square brackets as style tags. The Playwright hint contains
    `omni-cli[browser]` and rendered as `omni-cli` — the CLI printed an install
    command that installs the wrong thing.
    """

    def _render(self, text: str) -> str:
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=200, no_color=True).print(text)
        return buf.getvalue()

    def test_the_playwright_hint_keeps_its_extra(self):
        from rich.markup import escape
        from omni.browsercheck import PLAYWRIGHT_HINT
        assert "[browser]" in self._render(escape(PLAYWRIGHT_HINT))

    def test_unescaped_markup_is_the_bug_being_guarded(self):
        from omni.browsercheck import PLAYWRIGHT_HINT
        assert "[browser]" not in self._render(PLAYWRIGHT_HINT)

    def test_findings_table_preserves_bracketed_selectors(self):
        table = cli._findings_table(
            "t", [("kind", "p.html", "input[type=submit] is off-screen")], "red")
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=200, no_color=True).print(table)
        assert "[type=submit]" in buf.getvalue()

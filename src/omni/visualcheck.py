"""
Tier 3 — visual review of a running page.

Tiers 1 and 2 answer questions with definite answers: does this file exist, does
this id exist, did this click throw? This asks one that does not — *does the page
look right?* Text clipped by its container, controls pushed off-screen, unreadable
contrast, buttons stacked on top of each other: all render without error and are
all wrong.

The pipeline is geometry-first by design, because the geometry is fact and the
model is not:

    screenshot + DOM geometry
        |                    \\
        v                     v
    deterministic checks    vision model (think:false)
        |                     |
        | facts               v  claims
        |               cross-check against geometry
        |                     |
        +-------> report <----+   confirmed | plausible

Measured on this machine, the local 8B vision model produced one correct
observation and one invented one from two attempts, and in another run claimed it
had received no image at all. So:

  * **This tier never fails a run.** It reports; tiers 1 and 2 keep pass/fail.
    A verifier that halts on an invented defect gets switched off, taking the
    real findings with it.
  * **Claims contradicted by geometry are dropped**, and claims geometry cannot
    speak to are labelled `plausible` rather than stated as fact.
  * The deterministic half needs no model and is the part that finds real bugs.

Run it directly:

    python -m omni.visualcheck [path] [--no-vision] [--mobile]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from omni.browsercheck import PLAYWRIGHT_HINT, _entry_pages, serve

__all__ = ["VisualFinding", "check_workspace", "group_findings", "main",
           "DESKTOP_VIEWPORT", "MOBILE_VIEWPORT"]

DESKTOP_VIEWPORT = {"width": 1100, "height": 800}
MOBILE_VIEWPORT = {"width": 375, "height": 812}

#: WCAG AA for body text.
MIN_CONTRAST = 4.5
PAGE_TIMEOUT_MS = 15_000
SETTLE_MS = 400
MAX_PAGES = 6


@dataclass(frozen=True)
class VisualFinding:
    kind: str
    detail: str
    where: str
    selector: str = ""
    confidence: str = "confirmed"     # confirmed | plausible

    def __str__(self) -> str:
        tag = "" if self.confidence == "confirmed" else f" ({self.confidence})"
        target = f" <{self.selector}>" if self.selector else ""
        return f"{self.kind}: {self.detail}{target}{tag}  [{self.where}]"


@dataclass
class PageGeometry:
    viewport: dict
    doc_height: int
    elements: list[dict] = field(default_factory=list)

    def by_selector(self, needle: str) -> dict | None:
        needle = (needle or "").strip().lstrip("#.").lower()
        if not needle:
            return None
        for el in self.elements:
            if (el.get("id") or "").lower() == needle:
                return el
            if needle in (el.get("sel") or "").lower():
                return el
            if needle and needle == (el.get("text") or "").strip().lower():
                return el
        return None


# --------------------------------------------------------------------------- #
# geometry extraction
# --------------------------------------------------------------------------- #

_EXTRACT_JS = r"""
() => {
  const path = (el) => {
    if (el.id) return '#' + el.id;
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 4) {
      let part = node.tagName.toLowerCase();
      if (node.classList.length) part += '.' + [...node.classList].slice(0, 2).join('.');
      parts.unshift(part);
      node = node.parentElement;
    }
    return parts.join(' > ');
  };

  const parse = (c) => {
    const m = /rgba?\(([^)]+)\)/.exec(c || '');
    if (!m) return null;
    const p = m[1].split(',').map(s => parseFloat(s.trim()));
    return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 };
  };

  const effectiveBg = (el) => {
    let node = el;
    while (node) {
      const bg = parse(getComputedStyle(node).backgroundColor);
      if (bg && bg.a > 0.1) return bg;
      node = node.parentElement;
    }
    return { r: 255, g: 255, b: 255, a: 1 };
  };

  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92
                                                     : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };

  const INTERACTIVE = ['button', 'a', 'input', 'select', 'textarea'];
  const out = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const r = el.getBoundingClientRect();
    const ownText = [...el.childNodes]
      .filter(n => n.nodeType === 3).map(n => n.textContent.trim()).join(' ').trim();

    let contrast = null;
    if (ownText) {
      const fg = parse(cs.color);
      if (fg) {
        const bg = effectiveBg(el);
        const [a, b] = [lum(fg), lum(bg)].sort((x, y) => y - x);
        contrast = Math.round(((a + 0.05) / (b + 0.05)) * 100) / 100;
      }
    }

    out.push({
      sel: path(el),
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      text: (el.textContent || '').trim().slice(0, 40),
      ownText: ownText.slice(0, 40),
      x: Math.round(r.x), y: Math.round(r.y),
      w: Math.round(r.width), h: Math.round(r.height),
      scrollW: el.scrollWidth, clientW: el.clientWidth,
      overflowX: cs.overflowX, overflowY: cs.overflowY,
      contrast: contrast,
      interactive: INTERACTIVE.includes(el.tagName.toLowerCase()) ||
                   el.getAttribute('role') === 'button',
      depth: (() => { let d = 0, n = el; while ((n = n.parentElement)) d++; return d; })(),
    });
  }
  return { viewport: { width: innerWidth, height: innerHeight },
           docHeight: document.documentElement.scrollHeight, elements: out };
}
"""


def _label(el: dict) -> str:
    return el.get("id") and f"#{el['id']}" or el.get("sel") or el.get("tag", "?")


def _overlap(a: dict, b: dict) -> float:
    """Fraction of the smaller box covered by the intersection."""
    x = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    y = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    area_a, area_b = a["w"] * a["h"], b["w"] * b["h"]
    smaller = min(area_a, area_b)
    return 0.0 if smaller <= 0 else (x * y) / smaller


def deterministic_findings(geo: PageGeometry, where: str) -> list[VisualFinding]:
    """
    Defects the browser can measure. No model, no judgement, no false positives
    that a human would argue with.
    """
    findings: list[VisualFinding] = []
    vw = geo.viewport["width"]

    for el in geo.elements:
        label = _label(el)

        # Text wider than its box, with no way to scroll to the rest.
        if (el.get("ownText") and el.get("clientW", 0) > 0
                and el.get("scrollW", 0) > el["clientW"] + 1
                and el.get("overflowX") in ("visible", "hidden", "clip")):
            findings.append(VisualFinding(
                "clipped-text",
                f"content is {el['scrollW'] - el['clientW']}px wider than its box "
                f"and cannot be scrolled",
                where, label))

        # Pushed out of the viewport sideways. Vertical overflow is just a page.
        if el["w"] > 0 and (el["x"] + el["w"] < 0 or el["x"] > vw):
            findings.append(VisualFinding(
                "offscreen", f"sits at x={el['x']}, outside the {vw}px viewport",
                where, label))

        # A control nobody can click.
        if el["interactive"] and (el["w"] < 1 or el["h"] < 1):
            findings.append(VisualFinding(
                "zero-size-control", f"renders {el['w']}x{el['h']}px",
                where, label))

        # Unreadable text.
        c = el.get("contrast")
        if c is not None and c < MIN_CONTRAST and el.get("ownText"):
            findings.append(VisualFinding(
                "low-contrast",
                f"contrast ratio {c} against its background (WCAG AA needs "
                f"{MIN_CONTRAST})", where, label))

    # Controls sitting on top of one another.
    controls = [e for e in geo.elements
                if e["interactive"] and e["w"] > 0 and e["h"] > 0]
    for i, a in enumerate(controls):
        for b in controls[i + 1:]:
            # Nested elements legitimately overlap (a span inside a button).
            if abs(a["depth"] - b["depth"]) > 0 and _overlap(a, b) > 0.95:
                continue
            if _overlap(a, b) > 0.5:
                findings.append(VisualFinding(
                    "overlapping-controls",
                    f"{_label(a)} and {_label(b)} overlap by "
                    f"{round(_overlap(a, b) * 100)}%", where, _label(a)))
    return findings


# --------------------------------------------------------------------------- #
# vision pass
# --------------------------------------------------------------------------- #

_VISION_PROMPT = """You are reviewing a screenshot of a web page for VISUAL defects.

Report only what you can SEE. Do not comment on features, wording, or what the
app does. Look for: elements overlapping each other, text cut off or spilling
outside its box, text too faint to read, controls misaligned or clipped at an
edge, large empty areas where content should be, controls that are grouped
illogically.

Reply with a JSON array. Each item:
  {"kind": "overlap|clipped|contrast|alignment|grouping|empty",
   "element": "the visible label or id of the element, if you can tell",
   "detail": "one short sentence"}

Report at most 4. If the page looks correct, reply with an empty array: []
"""


def vision_findings(png: bytes, geo: PageGeometry, where: str,
                    client) -> tuple[list[VisualFinding], str | None]:
    """
    Ask the model, then keep only what the geometry does not contradict.

    Returns (findings, note). `note` explains why the pass produced nothing when
    that is a capability problem rather than a clean page.
    """
    from omni.vision import VisionUnavailable

    try:
        claims = client.ask_json(png, _VISION_PROMPT)
    except VisionUnavailable as exc:
        return [], str(exc)

    findings: list[VisualFinding] = []
    for claim in claims[:4]:
        detail = str(claim.get("detail") or "").strip()
        if not detail:
            continue
        kind = str(claim.get("kind") or "layout").strip().lower()
        named = str(claim.get("element") or "").strip()
        element = geo.by_selector(named) if named else None

        # Geometry beats the model wherever it can speak.
        if element is not None:
            if kind == "contrast" and (element.get("contrast") or 99) >= MIN_CONTRAST:
                continue      # measurably fine; the model is wrong
            if kind == "clipped" and element.get("scrollW", 0) <= element.get("clientW", 0):
                continue
        confidence = "confirmed" if element is not None else "plausible"
        findings.append(VisualFinding(f"visual-{kind}", detail, where,
                                      _label(element) if element else named,
                                      confidence))
    return findings, None


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

@dataclass
class VisualReport:
    findings: list[VisualFinding] = field(default_factory=list)
    screenshots: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    pages_checked: int = 0


def group_findings(findings: list[VisualFinding]) -> list[VisualFinding]:
    """
    Collapse the same defect repeated across sibling elements.

    Nine operator buttons sharing one stylesheet rule fail the contrast check
    nine times. That is one defect with nine instances, and printing it nine
    times buries everything else. The count and a couple of examples carry the
    same information in one line.
    """
    order: list[tuple[str, str, str]] = []
    groups: dict[tuple[str, str, str], list[VisualFinding]] = {}
    for finding in findings:
        key = (finding.kind, finding.detail, finding.where)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(finding)

    out: list[VisualFinding] = []
    for key in order:
        members = groups[key]
        first = members[0]
        if len(members) == 1:
            out.append(first)
            continue
        # When the group shares one selector — nine buttons from one CSS rule —
        # listing it twice says nothing. Only enumerate genuinely distinct ones.
        distinct = list(dict.fromkeys(m.selector for m in members if m.selector))
        suffix = ""
        if len(distinct) > 1:
            suffix = " (" + ", ".join(distinct[:3]) + (", …)" if len(distinct) > 3 else ")")
        out.append(VisualFinding(
            first.kind, f"{first.detail} — {len(members)} elements{suffix}",
            first.where, distinct[0] if distinct else "", first.confidence))
    return out


def check_workspace(workspace: Path, *, use_vision: bool = True,
                    viewport: dict | None = None,
                    screenshot_dir: Path | None = None,
                    vision_client=None) -> VisualReport:
    """
    Load each page, measure it, optionally ask a vision model, and report.

    Raises `RuntimeError` only when the browser itself is unavailable — a missing
    vision model is a note, not a failure.
    """
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(PLAYWRIGHT_HINT) from exc

    workspace = Path(workspace).resolve()
    pages = _entry_pages(workspace)[:MAX_PAGES]
    report = VisualReport()
    if not pages:
        return report

    if use_vision and vision_client is None:
        from omni.vision import VisionClient, detect_vision_model
        model = detect_vision_model()
        if model:
            vision_client = VisionClient(model)
            report.notes.append(f"vision model: {model}")
        else:
            report.notes.append(
                "no local model accepted an image; geometry checks only")

    shots = screenshot_dir or (workspace / ".omni" / "screenshots")
    base_url, httpd, _thread = serve(workspace)
    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch()
            except PlaywrightError as exc:
                raise RuntimeError(f"{PLAYWRIGHT_HINT} ({str(exc)[:100]})") from exc
            try:
                page = browser.new_page(viewport=viewport or DESKTOP_VIEWPORT)
                for path in pages:
                    rel = str(path.relative_to(workspace)).replace("\\", "/")
                    report.pages_checked += 1
                    try:
                        page.goto(f"{base_url}/{rel}", timeout=PAGE_TIMEOUT_MS,
                                  wait_until="load")
                        page.wait_for_timeout(SETTLE_MS)
                        raw = page.evaluate(_EXTRACT_JS)
                    except Exception as exc:                    # noqa: BLE001
                        report.findings.append(VisualFinding(
                            "page-load-failed", str(exc).splitlines()[0][:150], rel))
                        continue

                    geo = PageGeometry(viewport=raw["viewport"],
                                       doc_height=raw["docHeight"],
                                       elements=raw["elements"])
                    report.findings += deterministic_findings(geo, rel)

                    shots.mkdir(parents=True, exist_ok=True)
                    shot = shots / f"{rel.replace('/', '_')}.png"
                    png = page.screenshot(full_page=False)
                    shot.write_bytes(png)
                    report.screenshots.append(shot)

                    if vision_client is not None:
                        found, note = vision_findings(png, geo, rel, vision_client)
                        report.findings += found
                        if note:
                            report.notes.append(f"{rel}: {note}")
                page.close()
            finally:
                browser.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    use_vision = "--no-vision" not in argv
    mobile = "--mobile" in argv
    as_json = "--json" in argv
    argv = [a for a in argv if not a.startswith("--")]
    workspace = Path(argv[0]) if argv else Path.cwd()

    if not workspace.is_dir():
        print(f"visualcheck: {workspace} is not a directory")
        return 2

    try:
        report = check_workspace(
            workspace, use_vision=use_vision,
            viewport=MOBILE_VIEWPORT if mobile else DESKTOP_VIEWPORT)
    except RuntimeError as exc:
        print(f"visualcheck: skipped — {exc}")
        return 2

    if as_json:
        print(json.dumps({
            "pages": report.pages_checked,
            "notes": report.notes,
            "screenshots": [str(p) for p in report.screenshots],
            "findings": [f.__dict__ for f in report.findings],
        }, indent=2))
        return 0

    if not report.pages_checked:
        print("visualcheck: no HTML in this workspace; nothing to check")
        return 2

    for note in report.notes:
        print(f"  note: {note}")

    grouped = group_findings(report.findings)
    if not grouped:
        print(f"visualcheck: {report.pages_checked} page(s) — no visual defects found")
    else:
        confirmed = [f for f in grouped if f.confidence == "confirmed"]
        plausible = [f for f in grouped if f.confidence != "confirmed"]
        print(f"visualcheck: {len(confirmed)} measured, {len(plausible)} suggested "
              f"across {report.pages_checked} page(s)\n")
        for finding in confirmed:
            print(f"  {finding}")
        for finding in plausible:
            print(f"  {finding}")

    if report.screenshots:
        print(f"\n  screenshots: {report.screenshots[0].parent}")
    # Advisory by design: never fail the caller on a visual opinion.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

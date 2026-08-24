# Tier 3 — Visual UI Review with a Local Vision Model

**Status:** built, 2026-08-23. Phases 3a-3d shipped as `src/omni/visualcheck.py` and
`src/omni/vision.py`, triggered by the `UI_TEST` intent. 3e (mobile pass, visual regression
baseline) remains — `--mobile` exists, the stored-baseline comparison does not.
The measurements below are what the implementation was built around; they held up.
**Prerequisites:** Tier 1 (`omni.webcheck`) and Tier 2 (`omni.browsercheck`) — both shipped.
**Date:** 2026-08-23

---

## What Tier 3 is

Tiers 1 and 2 answer questions with definite answers: *does this file exist, does this id
exist, did this click throw?* Tier 3 asks a question that has no definite answer — **does the
page look right?** Overlapping elements, text clipped by its container, invisible-on-invisible
contrast, controls pushed off-screen, a layout that renders without error and is still wrong.

That difference is the whole design problem. Tiers 1 and 2 produce facts, so they can gate a
run. Tier 3 produces opinions, so it must not.

---

## Feasibility — measured, not assumed

Probed against this machine's Ollama install on 2026-08-23.

| Question | Answer |
|---|---|
| Does any local model accept images? | **Yes** — `gemma4:latest` (8B). `granite4:latest` refuses: *"Multimodal data provided, but model does not support it"* |
| Does it work out of the box? | **No.** It is a thinking model: `response` came back empty with `eval_count=250` and `done_reason=length` — it spent the whole budget reasoning and never emitted an answer |
| What fixes it? | `"think": false` in the Ollama request. 6s, 204 tokens, a real answer |
| Is the output useful? | **Partly.** See below |
| Latency | ~6s per screenshot at 1100×800 on this hardware |

### What it actually said about the real calculator

Given a screenshot of the working SciCalc page and asked for up to three concrete layout
problems:

> 1. **Inconsistent Spacing/Padding Around Buttons** — *(not supported by the screenshot;
>    spacing is uniform)*
> 2. **Lack of Visual Hierarchy for Functionality** — number pad, function keys and operator
>    keys are not visually differentiated — *(**correct**; `DEG`/`RAD` sit inline in the number
>    grid and `)`/`!` sit beside `1 2 3`)*

One real finding, one invented, from two. And with thinking left enabled the same model
answered *"I need the image of the calculator UI to analyze it. Please provide the picture"* —
having been sent the image. **Reliability is the binding constraint, not capability.**

---

## Design consequences

These follow directly from the evidence above, not from taste.

1. **Tier 3 must never fail a run.** A verifier that halts on an invented defect trains the
   operator to switch it off. It reports; it does not gate. Tiers 1 and 2 keep exclusive
   control of pass/fail.
2. **Every claim needs a coordinate.** A finding must name an element and a bounding box that
   can be checked, so an unverifiable claim can be dropped before it reaches the user.
3. **Cross-check what can be cross-checked.** "Text overflows its container" and "element is
   off-screen" are measurable in the DOM. Where geometry can confirm or refute the model, the
   geometry wins.
4. **`think: false` is mandatory** on this model family, and the response must be validated as
   non-empty before use.
5. **Absence of a vision model is normal.** Degrade to Tier 2 silently, exactly as Tier 2
   degrades to Tier 1 without Playwright.

---

## Architecture

```
browsercheck (Tier 2)          visualcheck (Tier 3)
  serve + load + click     ->    screenshot + DOM geometry
        |                              |
        | facts                        v
        |                        vision model (think:false)
        |                              |
        |                              v
        |                        claims -> geometry cross-check
        v                              |
   verify gate  <-- pass/fail          v
                                 advisory report -> operator
```

### New module: `src/omni/visualcheck.py`

```python
@dataclass(frozen=True)
class VisualFinding:
    kind: str          # overlap | clipped-text | offscreen | low-contrast | layout
    detail: str
    selector: str      # the element the model named
    box: tuple[int, int, int, int] | None
    confidence: str    # confirmed | plausible | unverified
```

Reuses `browsercheck.serve` and its Playwright session — one browser launch for both tiers.

### Pipeline

1. **Capture.** Full-page screenshot per entry point, at a fixed viewport (1100×800) so results
   are comparable between runs. Optionally a second pass at 375×812 for mobile.
2. **Extract geometry.** In-page JS returns, for every visible element: selector, bounding box,
   `scrollWidth` vs `clientWidth`, computed colour and background. This is the ground truth.
3. **Deterministic pre-pass — no model.** Straight from the geometry:
   - text overflow: `scrollWidth > clientWidth + 1`
   - off-screen: box outside the viewport
   - zero-size interactive elements
   - overlapping siblings: intersecting boxes among non-nested elements
   - contrast: WCAG ratio below 4.5:1 from computed colours

   These are *facts* and belong with Tiers 1-2. They may well be the most valuable part of
   Tier 3, and they need no model at all.
4. **Model pass.** Screenshot plus a compact element list to the vision model, `think: false`,
   asking for defects the geometry cannot express — visual grouping, alignment, hierarchy,
   whether the page reads as the thing it claims to be.
5. **Cross-check.** Each claim is matched to an element. Confirmed by geometry → `confirmed`.
   Contradicted → dropped. Neither → `plausible`, and labelled as such in the report.
6. **Report.** Advisory panel in the CLI, alongside the existing tables. Never an exit code the
   gate consumes.

---

## Phases

| Phase | Work | Model needed | Value |
|---|---|---|---|
| **3a** | Geometry extraction + deterministic checks (overflow, off-screen, overlap, contrast, zero-size) | none | **High** — real defects, zero false positives, gate-able |
| **3b** | Screenshot capture, artefact storage, `--screenshot` flag | none | Enables the rest; useful alone for the operator |
| **3c** | Vision pass with `think: false`, schema-constrained findings | vision | Medium |
| **3d** | Geometry cross-check and confidence labelling | vision | Turns 3c from noise into signal |
| **3e** | Mobile viewport pass, visual regression against a stored baseline | none | Medium |

**Recommended order: 3a, 3b, then reassess.** 3a is deterministic, catches real defects, and
can join the verification gate. Whether 3c is worth building depends on how much 3a leaves
behind — on the evidence above, a local 8B model contributes roughly one usable observation per
attempt, and an equal number of invented ones.

---

## Effort

| Phase | Estimate |
|---|---|
| 3a geometry + deterministic checks | 6 h |
| 3b screenshots + artefacts | 2 h |
| 3c vision pass | 4 h |
| 3d cross-check + confidence | 4 h |
| 3e mobile + regression baseline | 5 h |

3a + 3b is roughly a day and needs no model.

---

## Risks

| Risk | Mitigation |
|---|---|
| Invented defects erode trust | Advisory only; every claim labelled `confirmed`/`plausible`; contradicted claims dropped |
| Model silently returns nothing | Already observed. Assert non-empty and fall back to the deterministic pass |
| Model forgets it received an image | Already observed with thinking enabled. Pin `think: false`; treat "please provide the image" as a failed call |
| 6s per screenshot per page | Cap pages per run; only run on pages that changed |
| Vision model competes for VRAM with the coding model | Run Tier 3 after the coding loop finishes, never interleaved |
| Screenshots leak workspace content | They are workspace artefacts under `pathguard`, same as any other file |

---

## What would make Tier 3 genuinely good

A stronger vision model. The 8B local model produces generic design commentary; the defects
that matter — *"the DEG/RAD toggles are inside the number grid"* — need spatial reasoning it
does not reliably have. If a larger vision model becomes available locally, 3c/3d get
substantially better with no architectural change, because the pipeline treats the model as a
replaceable claim generator behind a geometry cross-check.

Until then, **3a is where the value is**, and it is not a vision feature at all.

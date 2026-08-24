# Omni CLI — Capability Comparison vs. Modern Autonomous Coding Agents

**Subject:** `src/omni/` — `cli.py`, `runtime.py`, `backends.py`, `pathguard.py`, `agentkit/`
**Compared against:** Claude Code, Cursor Agent, Aider, OpenHands, Codex
**First reviewed:** 2026-08-21 · **Updated:** 2026-08-23 (458 tests, 9 live runs)

---

## Verdict

This is a **real agent runtime**, not a chat wrapper. The FSM, event-sourced journal,
budget ledger, guardrail detector stack, context summarizer, HITL suspend/resume, and
constrained-decoding policy are all present and mostly well-reasoned. Several subsystems are
ahead of what shipping agents expose to users.

Three things held it back, in order:

1. ~~**There is exactly one tool.**~~ **Fixed.** `agentkit.ToolRegistry` dispatches six tools
   by name; `read_file`, `write_file`, `edit_file`, `list_dir`, and `search_files` exist.
2. ~~**The safety perimeter does not hold on Windows.**~~ **Fixed.** One containment
   implementation in `agentkit.jail`, used by both `CommandPolicy` and every filesystem tool.
3. ~~**Three subsystems are silently disabled.**~~ **Fixed.** The ledger is charged, approvals
   grant, and the FSM resume edge fires.

The architecture was sound; the wiring was where it failed. What remains is genuine scope
rather than defect: sub-agents, streaming, and symbol-level context assembly.

---

## Comparison table

Legend: ✅ present and sound · 🟡 present but degraded/miswired · ❌ absent

| Capability | Omni CLI | Modern coding agent | Status |
|---|---|---|---|
| **Tool use** | `agentkit.ToolRegistry` — six tools dispatched by name, JSON-Schema args, constrained decoding | Typed tool registry, schema-validated args, parallel calls | ✅ |
| **Filesystem read** | `read_file` (line-numbered, paged), `list_dir`, `search_files` | `read_file` with offsets, `glob`, `grep` | ✅ |
| **Filesystem write** | `write_file`/`edit_file` as jailed, atomic, approval-gated tools; `edit_file` refuses ambiguous matches | `write_file`/`edit_file` with uniqueness checks, diffs | ✅ |
| **Code execution** | `SubprocessShell`, argv-only, cross-platform aliasing | Sandboxed shell with streaming output | ✅ |
| **Closed feedback loop** | `VerifyGate` converts a claimed success into a verification run; unproved claims become honest failures | edit → test → read failure → fix, success requires evidence | ✅ |
| **Planning** | `SmartPlanner` (LLM, schema-constrained) + `PlannerLoop` + `default_plan` fallback | Plan mode, todo tracking, replanning | ✅ |
| **Routing** | Heuristic `Router` with abstain → LLM `SmartRouter` fallback | Usually implicit | ✅ ahead |
| **Structured output** | `STRICT_DECISION_SCHEMA` constrained decoding + one repair round-trip + `AskHuman` degradation | Native tool-call schemas | ✅ ahead |
| **Loop safety** | Repetition / oscillation / no-progress / consecutive-error detectors, warn-then-suspend | Usually a step cap only | ✅ **well ahead** |
| **Budget / cost** | `Ledger` + `Budget`, now charged by the repair policy | Token budgets, context compaction | ✅ fixed |
| **Observability** | `RunJournal` event-sourced JSONL, replayable, `RunJournal.load` | Transcripts, token accounting | ✅ ahead |
| **HITL** | `AskHuman`, `SUSPENDED_APPROVAL`, `with_intervention`; `approval_grants()` accepts command/executable/sudo | Per-tool approval prompts | ✅ fixed |
| **Resume** | `ResumePayload` + `Summarizer`; one Orchestrator per session, so the FSM edge fires | Session resume | ✅ fixed |
| **Prompt-injection boundary** | `render_observation` fences tool output as untrusted, explicitly | Rarely explicit | ✅ ahead |
| **Secret redaction** | `redact()` before truncation, so secrets can't be split | Rare | ✅ ahead |
| **Permissions** | Allowlist + argv parse + metacharacter ban + containment via `agentkit.jail` | Allowlists, sandboxing | ✅ fixed |
| **Error recovery** | `classify_error`, `RetryPolicy`, `CircuitBreaker`, policy refusals fed back as observations | Retries, self-correction | ✅ |
| **Context assembly** | `agentkit.survey` — ignore-aware tree, ranked files, real source under a char budget | Repo map, AST/symbol search, import graph | 🟡 no symbol/import graph |
| **Memory across sessions** | `agentkit.memory.ContextStore` — per-intent windows, embedding-ranked recall | Project config, memory files | ✅ |
| **Sub-agents** | None | Parallel isolated-context agents | ❌ |
| **Streaming** | None — spinner during 3,500-token generations | Token streaming + interrupt | ❌ |
| **Multi-backend** | Ollama / LM Studio / llama.cpp, probed at startup with model discovery | Usually 1–2 providers | ✅ **ahead** |
| **Role-based model routing** | `ModelRegistry` binds ROUTER/PLANNER/REPAIR/SUMMARIZER separately | Rare outside frameworks | ✅ ahead |
| **Local-first / offline** | Fully local, plus a no-LLM simulation mode | Mostly cloud | ✅ differentiator |

---

## Critical defects

> **Status: defects 1–6 are fixed** as of 2026-08-21. Each is preserved below as originally
> reproduced, because the failure modes are the rationale for the tests that now guard them
> (`test_agentkit.py`, 64 passing). See `ENHANCEMENT_PLAN.md` for what shipped.
>
> | # | Defect | Fix |
> |---|---|---|
> | 1 | Windows jail escape | `agentkit/jail.py` — `is_relative_to` under `normcase`, every non-flag token checked |
> | 2 | Unjailed direct-code write | `jail.write_text_in()` — containment + atomic rename; all code blocks kept |
> | 3 | Budget never charged | `Orchestrator._make_repair_policy()` passes the run ledger to the factory |
> | 4 | Approval livelock | `approval_grants()` accepts the command, the executable, or `sudo` |
> | 5 | Orchestrator rebuilt per turn | one instance per session, so the FSM resume edge fires |
> | 6 | Parser differential | `split_command()` is platform-aware; loops pass `verdict.argv` through |

### 1. The workspace jail does not hold on Windows — reproduced

`CommandPolicy.classify` (`agent_runtime.py:466-475`) only runs the containment check on
tokens that `startswith("/")` or contain a `..` component:

```python
if token.startswith("/") or ".." in Path(token).parts:
    resolved = (self.workspace / token).resolve()
    if not str(resolved).startswith(str(self.workspace)):
        return PolicyDecision(Risk.FORBIDDEN, ...)
```

A Windows drive-absolute path is neither.

| Command | Result |
|---|---|
| `cat /etc/passwd` | BLOCKED ✅ |
| `cat ../../../secrets.txt` | BLOCKED ✅ |
| `cat //server/share/secret` | BLOCKED ✅ |
| **`cat C:/Users/wasim/.ssh/id_rsa`** | **ALLOWED** → resolves to the real path |

`cat` is allowlisted at `Risk.SAFE`, so no approval gate fires either. Complete read-anywhere
escape on the platform this actually runs on.

Second defect, same three lines: `str(resolved).startswith(str(self.workspace))` is a **string
prefix test, not a path test**. `C:\…\python-evil` prefixes `C:\…\python`, so a sibling
directory reads as contained. Windows case-insensitivity (`c:\` vs `C:\`) breaks it the other
way. Use `Path.is_relative_to` with `os.path.normcase`.

### 2. `handle_direct_code` writes files with no policy and no jail

`omni_cli.py:280-281`:

```python
target_file = workspace / filename.strip()
target_file.write_text(code, encoding="utf-8")
```

`filename` defaults to whatever a regex scraped out of **model output** (`omni_cli.py:269-270`).
`workspace / "../../evil.py"` escapes; `workspace / "C:/Windows/Temp/x"` discards the workspace
entirely, because `Path.__truediv__` with an absolute right operand returns the absolute path.
This is a *write* bypass, and it never touches `CommandPolicy`.

Also `code_blocks[0]` (`omni_cli.py:276-278`) silently discards every code block after the
first, so a multi-file answer writes only its first fragment. And `write_text` truncates
in place with no atomic temp-and-rename.

### 3. The budget ledger is never charged

`WrappedLLMPolicy.__init__` (`omni_cli.py:54`) builds its **own** `Ledger(budget=Budget())`,
while `Orchestrator.handle` builds the real one (`ledger = Ledger(budget=self.budget,
label="root")`). `LLMPolicy.propose` charges tokens to the private ledger; `RepairLoop` checks
`ledger.exceeded()` on the root. The root ledger therefore only ever counts `tick()`s —
**token and USD budgets are never enforced.**

It is worse than that: `repair_factory = lambda cmd: WrappedLLMPolicy(client)`
(`omni_cli.py:482`) constructs a fresh policy *and a fresh ledger* on every invocation, so even
the private accounting resets. `Budget(max_tokens=150_000, max_usd=2.00)` is decorative.

### 4. Approving an elevated command livelocks

`omni_cli.py:570`:

```python
approvals_list = ["sudo"] if "sudo" in pending_cmd else [pending_cmd]
```

Both loops gate on the literal token `"sudo"`:

```python
if verdict.risk is Risk.ELEVATED and "sudo" not in approvals:   # RepairLoop, PlannerLoop
```

So approving any elevated non-sudo command — `rm`, `chmod`, `npm publish`, `git push` — adds
`[pending_cmd]` to the set, the check still fails, and the run suspends again on the same
command. The operator can approve forever and never advance.

### 5. A fresh `Orchestrator` per turn discards the FSM state

`omni_cli.py:511` and `:535` construct a new `Orchestrator` on every iteration.
`Orchestrator.__init__` builds a new `RunJournal` with a new `run_id`
(`agent_runtime.py:1652`), and `self.state = self.journal.current_state()` reads an empty
in-memory event list → `IDLE`. So in `handle`:

```python
if resume is not None:
    if self.state in SUSPENDED_STATES:      # never true — state is always IDLE
        self._transition(RunState.RESUMING, ...)
```

The `SUSPENDED → RESUMING` edge never fires. Resume half-works because the payload is still
threaded into the loops, but the FSM never models it, and `run.jsonl` accumulates a new
`run_id` per turn instead of one continuing run — visible in the committed journal, which has
two separate run ids.

### 6. Policy and executor parse the command independently

`CommandPolicy.classify` runs `shlex.split(command)` (`agent_runtime.py:444`); the executor
runs `shlex.split(str(call.args["command"]))` again (`agent_runtime.py:657`). Two parses,
nothing enforcing agreement, and the policy's validated `verdict.argv` is computed and then
discarded. `shlex.split` also defaults to `posix=True`, which eats backslashes:

```
input : C:\Users\wasim\workspace\python
posix : ['C:Userswasimworkspacepython']     <-- every separator destroyed
win   : ['C:\\Users\\wasim\\workspace\\python']
```

Every native Windows path is corrupted before both the security check and execution.

---

## Secondary issues

| Issue | Location |
|---|---|
| `Finish(succeeded=True)` is accepted on the model's word — nothing runs tests before declaring success | `agent_runtime.py` `RepairLoop` |
| `classify_intent` prefix rules route "write tests for my repo and run them" to `DIRECT_CODE`, which cannot execute anything | `omni_cli.py:189-193` |
| `handle_project_review` sees a 3,000-char truncation of `ls -R` plus manifests — it never reads a single source file, so "Principal Software Architect" review is structurally blind | `omni_cli.py:321-322` |
| `ls -R` on a repo containing `venv/` or `node_modules/` fills the entire budget with noise | `omni_cli.py:321` |
| `select_workspace` defaults to `./` — the agent's exec+write root becomes its own source repo, with `git` allowlisted | `omni_cli.py:446-453` |
| Bare `except Exception` prints the message and drops the traceback | `omni_cli.py:592` |
| `SubprocessShell` copies the full `os.environ`, handing every shell token and API key to child processes | `agent_runtime.py:648-651` |
| No streaming anywhere; 3,500-token generations sit behind a spinner | `omni_cli.py:255, 308, 359` |
| `robust_json_parse` greedy `(\{[\s\S]*\})` spans from the first `{` to the last `}` — prose containing braces breaks it | `omni_cli.py:44` |
| The allowlist is too narrow to explore: both runs in `run.jsonl` suspended after `find` was refused and `ls /` was blocked | `run.jsonl` |

---

## What is genuinely ahead of shipping agents

Worth stating plainly, because it is unusual:

- **Guardrail detector stack.** Repetition, oscillation, no-progress, and consecutive-error
  detectors with warn-once-then-suspend semantics. Most agents ship a step cap.
- **Event-sourced journal.** `RunJournal` writes replayable JSONL and reloads via
  `RunJournal.load` — genuine time-travel debugging.
- **Explicit untrusted-output fence.** `render_observation` wraps tool output and tells the
  model it is evidence, never instructions. Very few systems state this.
- **Redaction before truncation**, so a secret is never split and half-leaked.
- **Backend probing with model discovery** across three local servers at startup.
- **Role-based model registry** — small model for routing, large for repair.
- **Allowlist-first command policy** with a documented rationale for why denylists fail. The
  design is right; only the path check is wrong.

---

## Current state — 2026-08-22

Eight live runs against a local model drove the loop from "cannot write a file at all" to
completing a real task: diagnosing a missing `js/app.js`, writing it, and writing a README.
Each run failed differently, each cause was distinct, and each is now covered by tests. See
`ENHANCEMENT_PLAN.md` for the run-by-run record.

Legend: ✅ present and sound · 🟡 present but limited · ❌ absent

| Capability | Omni CLI today | Modern coding agent | |
|---|---|---|---|
| **Tool use** | 7 tools dispatched by name, JSON-Schema args, constrained decoding | typed registry, schema-validated args | ✅ |
| **Filesystem** | `read_file` (paged, budget-aware), `list_dir`, `search_files`, `write_file`, `edit_file` | same, plus glob/AST search | ✅ |
| **Code execution** | `SubprocessShell`, argv-only, allowlist, per-platform aliasing | sandboxed shell | ✅ |
| **Closed feedback loop** | `VerifyGate` turns a claimed success into a verification run; unproved claims become honest failures | edit → test → fix | ✅ |
| **Action memory** | compact `ALREADY DONE` ledger + last 6 full observations | full transcript, compaction | ✅ |
| **Budget awareness** | prompt states `action N of M` and escalates as it runs out | rare | ✅ **ahead** |
| **Loop safety** | 5 detectors — repetition, redundant-call, oscillation, no-progress, consecutive-error — with per-call warning codes | usually a step cap | ✅ **well ahead** |
| **Observability** | event-sourced `run.jsonl` plus a rendered `Loop Trace` table per turn | transcripts | ✅ **ahead** |
| **Planning** | tool-aware plan steps; a failed *inspection* step is a finding, not a blocker | plan mode, replanning | ✅ |
| **HITL** | approval per elevated tool, suspend/resume across turns, per-call grant tokens | approval prompts | ✅ |
| **Session lifecycle** | many runs per session, each with its own `run_id` on one journal | session resume | ✅ |
| **Permissions** | allowlist + argv parse + metacharacter ban + `pathguard` containment | allowlists, sandboxing | ✅ |
| **Prompt-injection boundary** | tool output fenced as untrusted evidence | rarely explicit | ✅ **ahead** |
| **Secret redaction** | redact before truncate, so a secret is never split | rare | ✅ **ahead** |
| **Multi-backend** | Ollama / LM Studio / llama.cpp probed at startup, role-based routing | usually 1–2 | ✅ **ahead** |
| **Memory across sessions** | embedding-ranked recall, degrades honestly when unavailable | project config, memory files | ✅ |
| **Context assembly** | `survey.collect_digest` — ignore-aware tree, ranked files, char budget | repo map, symbol/import graph | 🟡 no symbol graph |
| **Output fidelity** | generation grounded in referencing files + siblings; policy prefers `edit_file` on existing files | generated code cites the files it read | ✅ |
| **UI verification** | static contract checks + headless browser: pages served, loaded and every control clicked | rare; usually left to the project's own tests | ✅ **ahead** |
| **Visual review** | DOM geometry (clipping, overlap, off-screen, contrast) + a cross-checked vision pass; advisory, never gates | screenshot diffing in some tools | ✅ **ahead** |
| **Streaming** | none — spinner during long generations | token streaming + interrupt | ❌ |
| **Sub-agents** | none | parallel isolated-context agents | ❌ |

---

## Output fidelity — the current gap

**Definition.** Not "is the code good" but: *does the artifact agree with the facts the run
already established?* A run can route correctly, plan correctly, write the right file to the
right path and pass every guardrail — and still emit content that contradicts a file it read
three steps earlier. Every mechanism in the table above can be green while the deliverable is
wrong.

**The observed instance.** Run 8 wrote `js/app.js` to fix a 404. Syntactically valid, correct
path, listeners guarded. But it binds to element IDs that do not exist:

| `app.js` looks for | `index.html` actually has |
|---|---|
| `history-panel` | `sidebar` |
| `history-list` | `historyList` |
| `clear-history` | `btnClearHistory` |
| `history-toggle` | *(absent)* |
| `display` | `display` ✅ |

Four of five miss, so the calculator renders and does nothing. The agent **had read
`index.html`** two steps earlier. The information was in the run and was discarded.

**The mechanism.** `generate_file(path, spec)` exists to stop plans truncating: content is
produced in its own completion so a whole project need not fit in one reply. The cost is that
the generating completion sees only a one-sentence spec — not the workspace, not the files the
loop just read. The tool signature has a third parameter, `context`, and **nothing in the
codebase populates it**. The generator is structurally blind.

This is the difference between the loop being correct and the output being right, and it is why
"all mechanisms green" is not the same as "task done".

### How to close it, in order of leverage

1. **Ground generation in what was read.** Populate `generate_file`'s `context` from
   observations already in the run — at minimum any file the target references. One tool call
   later the generator would have seen `id="historyList"` instead of inventing one. Highest
   value, smallest change: the plumbing exists and is unused.
2. **Prefer `edit_file` over regeneration on an existing project.** Regenerating a file the
   agent has read discards what it knows; editing forces it to quote the real text, and the
   uniqueness check already makes that mandatory.
3. **Add a cross-file consistency check to the verify gate.** `compileall` and `pytest` cannot
   catch an ID mismatch, and a static web project has no test suite at all. A contract check —
   do the IDs a script references exist in the HTML that loads it — would have caught this
   before the run reported success.
4. **Feed that failure back as a fidelity error**, so the loop repairs the mismatch instead of
   reporting a green run.

Items 1 and 2 are prompt-and-plumbing changes. Item 3 is a new verifier class and the most
work; it is also the only one that turns fidelity from a hope into a gate.

---

## Priorities

~~1-4~~ **Done.** Perimeter, tool registry, filesystem tools, verify gate and the wiring fixes
all shipped and are covered by tests. What remains, in order:

1. ~~**Output fidelity**~~ — **done.** `gather_context` shows the generator the files that
   reference its target plus the siblings it must call, and the policy prefers `edit_file` for
   files that already exist. What remains is *verifying* the result, not improving the odds.
2. ~~**Cross-file consistency verification**~~ — **done.** `omni.webcheck` proves referenced
   files and ids exist; `omni.webcheck --browser` proves the page loads and its controls
   click. Both are wired into the verification gate and neither needs a model.
3. ~~**Deterministic layout checks**~~ — **done.** `omni.visualcheck` measures clipping,
   overlap, off-screen and contrast, and the vision pass on top is cross-checked against
   those measurements. Triggered by asking to test the UI.
3. **Streaming** — the largest perceived-latency win; generations sit behind a spinner for
   5-10 seconds each.
4. **Symbol-level context assembly** — `survey` reads real source but builds no import graph.
5. **Sub-agents** — not needed until a task outgrows one context window.

See `ENHANCEMENT_PLAN.md`.

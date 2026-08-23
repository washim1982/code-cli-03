# Omni CLI — Enhancement Plan

**Scope:** close the gaps identified in `COMPARISON.md`, in severity order.
**Constraint:** modular — new code lives in a new package; the existing runtime is edited only
where it is actually wrong.
**Date:** 2026-08-21

---

## Status

| Phase | State | Evidence |
|---|---|---|
| **0 — Perimeter** | ✅ **done** | jail escape closed, parse-once, jailed atomic writes, env narrowed |
| **M — Memory** | ✅ **done** | `agentkit/memory.py`, live semantic recall verified against Ollama |
| **4 — Wiring** | ✅ **done** | shared ledger, approval grants, one Orchestrator per session |
| **1 — Tool registry** | ✅ **done** | `registry.py`, `dispatch.py`, `policy.py`; six tools dispatched by name |
| **2 — Filesystem tools** | ✅ **done** | `tools/fs.py`: read/write/edit/list/search, all jailed |
| **3 — Verify gate** | ✅ **done** | `verify.py`, `gate.py`; claimed success is proved or reported failed |
| **5 — Repo-aware review** | ✅ **done** | `survey.py`; the reviewer reads real source, not a truncated `ls -R` |

**All phases complete.** Tests: **355 passing** across nine suites.

### Post-integration fixes from real runs

Three live runs against a local model exposed defects no unit test would have found.

**Run 1 — `create a java script project ...` produced 2 of 6 files.** Three causes:
`classify_intent` routed a multi-file scaffold to `DIRECT_CODE`, which caps the whole project
at one completion; generation hit `max_tokens` and nothing said so; and `extract_code_blocks`
read filenames only from a `# filename:` comment *inside* each block, while the model put them
in headings *above* each fence — so files were saved as `script.txt` (an ASCII directory tree)
and `script_2.html`. Fixed with `is_project_request()` routing, heading-based filename
inference, tree-block rejection, and a visible truncation warning.

**Run 2 — the plan itself truncated.** Routing then worked, but a plan carrying every file's
full content inline hit the token cap mid-string at ~10 KB. The JSON would not parse, the
planner fell back to an exploration plan, its `git status` step failed in a non-repo workspace
(exit 128), and the scoped repair loop spent its entire iteration budget investigating that
irrelevant failure before suspending. The request was never attempted.

Fixed structurally with **`generate_file`** (`agentkit/tools/codegen.py`): the plan now carries
a path and a sentence per file, and each file is generated in its own completion with its own
budget. Also: `git status` removed from the fallback — a fallback must not manufacture an error
for the repair loop to chase — and truncation detection moved from `truncation_suspected`
(which means the *prompt* was cut) to `finish_reason == "length"`, which is the actual signal.

**Run 3 — twelve iterations of reading, nothing written.** The agent read four files, re-read
them, and hit the iteration cap without writing anything, twice in a row. Cause:
`read_file` returned a header saying `lines 1-98 of 98` while `finalize_output` — correct for
a shell transcript, catastrophic for a file listing — kept a 600-char head and tail and elided
the middle. The model was handed a body with a hole in it under a header claiming
completeness, asked for the middle, and received another elided response.

Fixed by making the filesystem tools fit `OUTPUT_CHAR_BUDGET` themselves and report honestly
what they returned (`lines 1-55 of 98 … continue with offset=56`), so the dispatcher never has
to elide. `list_dir` and `search_files` got the same treatment.

No guardrail caught it, either: `RepetitionDetector` only sees a sliding window, so
A→B→C→A→B→C never trips it, and every step succeeded so the error detectors stayed quiet.
Added `RedundantCallDetector` for exactly that gap — the same call returning *after* other
calls. Back-to-back repetition is left to `RepetitionDetector`, whose territory it is; a
vendored runtime test caught the overlap when the first version claimed both.


> **Post-pass fix (2026-08-21).** The Phase 4 edit inserted `_make_repair_policy` into the
> middle of `Orchestrator.__init__`, leaving `self.state = self.journal.current_state()`
> stranded after that method's `return` as unreachable code. `Orchestrator` never received a
> `.state` attribute, so **every** `handle()` call raised
> `AttributeError: 'Orchestrator' object has no attribute 'state'` — the entire `EXECUTE_TASK`
> path was dead, and `omni_cli.py:592`'s bare `except Exception` reduced it to a one-line
> console message. The assignment has been moved back into `__init__`.
>
> Measured before the fix: runtime suite **40 passed / 5 failed** in the working tree vs
> **45 passed** at `HEAD`. After: 45/45 and 64/64, and `Orchestrator.handle()` completes the
> FSM normally.
>
> **Coverage gap this exposed:** the runtime suite lives in `my_agent_project/`, not here, so
> nothing in this repo exercises `Orchestrator`. The 64 agentkit tests never construct one.
> **Vendor `test_agent_runtime.py` into this repo and run both suites together** — otherwise
> the next runtime edit breaks the product silently again.

**Run 4 — the read loop was fixed; the write was cut off instead.** The agent paged through
all four files cleanly (8 distinct reads, no repeats) and went to write `js/app.js` — then the
reply hit `ToolCallPolicy`'s 640-token cap mid-string. Three compounding causes:

* **640 tokens** was sized for a decision carrying a shell command, not one carrying file
  content in `write_file.content`. Raised to 4096.
* **The repair round-trip told the model its reply was "invalid"**, which is false and
  actively harmful: the JSON was well-formed, there was just more of it. The model produced
  the same oversized action again and was cut off in the same place. Truncation now gets its
  own message naming the real cause and pointing at `generate_file`.
* **The rules steered toward `write_file`** and never mentioned `generate_file`, which exists
  precisely so content does not have to fit inside a decision. Reordered.

`output_truncated` moved into the runtime, where `LLMPolicy` needs it to tell a cut-off reply
apart from a malformed one; `agentkit.tools.codegen` re-exports it so the layering holds.

**Run 5 — the session died after the first successful task.** `Error occurred: SUCCEEDED ->
ROUTING`. This was a regression from the Phase 4c fix: the CLI keeps one Orchestrator per
session (a fresh one per turn resets `self.state` and the SUSPENDED -> RESUMING edge never
fires), but `SUCCEEDED` is terminal by design, so the second `handle()` on the same instance
raised `IllegalTransition`. The CLI's catch-all reduced it to a one-line error and the run was
lost.

A session is many runs. `Orchestrator._begin_run()` now starts a fresh run — new run_id on the
same journal, state back to CREATED — when the previous one is over: terminal, or suspended
with no `resume` payload (the operator declined the approval or typed something else). A
suspension that *is* being resumed is left in place so `handle` can walk it through RESUMING.
The FSM invariant is untouched: a test asserts transitioning out of SUCCEEDED still raises.

Also in that run, a plan of three inspection steps reported **Task Succeeded: plan complete**
for a request to fix a 404. True of the plan, misleading as an outcome. The planner prompt now
requires at least one state-changing step for a create/fix goal, and the CLI says plainly when
a successful run changed no files.

**Run 6 — eleven reads, no writes, for "create a readme.md file".** The loop trace made the
cause legible for the first time: steps 3-4 read `index.html`, steps 5-6 read `calculator.js`,
then steps 8-9 read `index.html` **again** and 10-11 read `calculator.js` **again**.

`LLMPolicy._render` showed only `ctx.steps[-4:]`. An action whose observation had scrolled out
of that window left no trace in the prompt at all — so the agent had no way to know it had
already read a file, and read it again. The repetition guardrail caught the symptom and
suspended the run; it could not fix the cause.

The prompt now carries a compact **ALREADY DONE** ledger — one line per scrolled-out step,
`describe_call` plus exit status — above the recent full observations, and the window went from
4 to 6. Forgetting was the expensive part; one line per step is not.

`describe_call` also rendered every argument through `str()` before `repr()`, so an integer
came back as `offset='74'` and looked like a type error that was never there. Now `repr()`
directly.

**Run 7 — nine steps of exploring, suspended, still no file.** Two separate faults.

*The guardrail was disproportionate.* `GuardrailStack` escalates the second warning carrying
the same **code** straight to SUSPEND — correct for a code meaning "you are stuck", wrong for
one meaning "you re-read something". `RedundantCallDetector` used a single shared code, so two
unrelated repeats (a re-listed directory at step 4, a re-listed directory at step 9) looked
identical to an agent spinning on one call and killed the run. Step 9 was even reasonable:
`read_file('js/app.js')` had just failed, so re-listing `js/` is verification, not a loop.
The code is now scoped per call (`REDUNDANT_REPEATED_CALL:<fingerprint>`), which puts the
existing escalation at the right granularity — repeat *one* call and it escalates; make two
different mistakes once each and you get two warnings and a chance to act on them. Replaying
the failing nine-step sequence now yields two warnings and no suspension.

*The agent could not see its budget.* The iteration cap was invisible, so the loop was
open-ended from the agent's point of view: it explored until a limit it did not know about cut
it off mid-thought, having produced nothing. `PolicyContext` now carries
`iterations_used`/`iterations_max`, and the prompt states the budget — turning silent into
insistent as it runs out:

```
BUDGET: action 10 of 12.
Only 3 actions remain. Stop gathering information and produce the deliverable with what
you already know — a run that explores until the cap delivers nothing at all.

BUDGET: action 12 of 12.
THIS IS YOUR LAST ACTION. Produce the deliverable now, or finish with succeeded=false.
```

**Run 8 — the task succeeded, then the run threw the work away.** The repair loop did the job:
step 9 wrote `js/app.js` (228 lines), step 10 wrote `README.md` (61 lines). Both deliverables
existed on disk. Then the Orchestrator transitioned to PLANNING anyway, generated seven fresh
steps, and step 2 — "inspect package.json" in a project that has none — failed. The scoped
repair could not conjure the file, the retry failed identically, and the run suspended.

*The REPAIR route always fell through to the planner.* `RepairLoop` reports `succeeded=True`
from exactly one place — the policy returning `Finish(succeeded=True)`, which the verify gate
has already made it prove — and it is given the **user's** goal, not "unblock the environment",
so it frequently completes the job outright. A successful repair now ends the run. Repair as a
*prelude* to planning would need the loop to be given a narrower goal and to signal "unblocked"
separately from "done"; it cannot express that today, and pretending otherwise cost real work.

*A failed inspection killed the plan.* The planner guesses at filenames, and a read that fails
has still told you something — absence is a finding. `PlannerLoop` now takes an optional
`is_mutating` predicate (supplied by the tool registry through `AgentStack`): a failed
non-mutating step is journalled as `plan.step.skipped` and the plan continues. Failed steps that
*change* things still open a scoped repair, and callers that omit the predicate keep the old
behaviour.

### Project layout (2026-08-22)

```
interactive-omni-cli/
├─ pyproject.toml          `pip install -e .` -> `omni` console script
├─ pytest.ini              pythonpath = src
├─ requirements.txt
├─ src/omni/
│  ├─ pathguard.py         path containment; standard library only, no local imports
│  ├─ backends.py          Ollama / LM Studio / llama.cpp
│  ├─ runtime.py           FSM, journal, ledger, guardrails, loops, policies
│  ├─ agentkit/            tool registry, fs tools, dispatch, verify gate, memory, survey
│  └─ cli.py               interactive front end; `main()` is the entry point
├─ test/                   5 suites
└─ doc/
```

Dependency graph, verified acyclic by AST inspection:

```
pathguard  -> []
backends   -> []
runtime    -> [pathguard]
agentkit/* -> [pathguard, runtime]
cli        -> [pathguard, backends, runtime, agentkit.*]
```

Containment previously lived in `agentkit/jail.py`, so `runtime` imported the `agentkit`
package while nine `agentkit` modules imported `runtime` back. That cycle held together only
because `agentkit/__init__.py` kept every other module behind a lazy import — get that wrong
and you got an import error at a distance. `pathguard` is a leaf now, so the laziness is an
optimisation rather than a correctness requirement. Verified by importing each module first in
a fresh interpreter. `omni.agentkit.jail` remains as a re-export.

Two things the reorganisation broke and this fixed:

* **Tests only passed from the repo root.** `python -m pytest` puts the current directory on
  `sys.path`; that accident was resolving `import agent_runtime`. From anywhere else, or via
  the bare `pytest` console script, collection failed with `ModuleNotFoundError`. `pytest.ini`
  now sets `pythonpath = src`, verified by running the suite from an unrelated directory.
* **`requirements.txt` had moved into `doc/`.** It is a functional file, not documentation —
  `pip install -r requirements.txt` is the path everyone tries, and `agentkit/survey.py` ranks
  root manifests first when building a review digest. Moved back.

Run it with `omni` after `pip install -e .`, or `python -m omni.cli` from a clone.

### Shipped

```
agentkit/__init__.py     lazy re-exports; jail eager (stdlib-only) so agent_runtime can import it
agentkit/jail.py         contained() / resolve_in() / write_text_in()
agentkit/memory.py       ContextStore, MemoryConfig, RetrievedContext, cosine_similarity
agentkit/registry.py     ToolSpec, ToolRegistry, arg validation, decision schema, policy_for
agentkit/dispatch.py     MultiToolExecutor  -> agent_runtime.ToolExecutor
agentkit/policy.py       ToolCallPolicy(LLMPolicy) — emits {tool, args}, accepts legacy {command}
agentkit/verify.py       VerifySpec, detect_verify (pytest / npm test / compileall)
agentkit/gate.py         VerifyGate -> agent_runtime.Policy
agentkit/stack.py        build_agent_stack(): one call, returns Orchestrator kwargs
agentkit/survey.py       collect_digest(): ranked repo digest under a character budget
agentkit/tools/fs.py     read_file, list_dir, search_files, write_file, edit_file
agentkit/tools/shell.py  run_command — defers to CommandPolicy, perimeter unchanged
agentkit/tools/codegen.py generate_file — one completion per file, not per plan

test_agentkit.py         64   perimeter, wiring, memory
test_agentkit_tools.py   96   registry, fs tools, dispatch, policy, verify, gate,
                              output budget, redundant-call guard, golden tasks
test_omni_cli.py         40   plan steps, planner parsing, plan execution, survey,
                              approvals, multi-file extraction, project routing
test_codegen.py          23   generate_file, truncation detection, planner fallback
test_agent_runtime.py    45   vendored runtime suite (was only in my_agent_project/)
                        ---
                        268   passing
```

`agent_runtime.py` — the perimeter/wiring edits (containment loop rewritten, `split_command()`,
`approval_grants()`, `_make_repair_policy()` ledger passthrough, `SubprocessShell` env
allowlist), plus the tool seam: `describe_call()`, `classify_call()`, a `tool_policy` argument
threaded through `Orchestrator` → `RepairLoop`/`PlannerLoop`, and `PlanStep` gaining
`tool`/`args`. `tool_policy=None` preserves the previous shell-only behaviour exactly, which
is why the vendored runtime suite still passes unmodified.

Two portability fixes surfaced by running the gate for real: console scripts installed only as
modules (`pytest`) are rewritten to `python -m <name>`, `.CMD`/`.BAT` shims are invoked through
`COMSPEC`, and `APPDATA` was restored to the env allowlist — without it Python cannot find
per-user site-packages, so `pytest` was invisible to every child process.

`omni_cli.py` — `WrappedLLMPolicy` replaced by the stack's factory, one Orchestrator per
session, jailed multi-block code writing, memory recall/persist per turn, a tool-aware
`SmartPlanner`, and a repo-aware project review.

Verified end to end: an Orchestrator run planned two `write_file` steps and a `pytest -q` step,
created both files, and reached `SUCCEEDED`. A `write_file` to `../evil.py` is blocked; a
`read_file` of `C:/Windows/win.ini` is blocked; a claimed success over a syntactically broken
file is caught by the gate and stopped by `ConsecutiveErrorDetector`.

---

---

## Design principle: implement the Protocols, edit almost nothing

`agent_runtime.py` already defines the two structural seams this plan needs:

```python
class ToolExecutor(Protocol):
    async def execute(self, call: ToolCall, timeout_s: float) -> ToolResult: ...

class Policy(Protocol):
    async def propose(self, ctx: PolicyContext) -> Decision: ...
```

`RepairLoop`, `PlannerLoop`, and `Orchestrator` are written against these and nothing else.
So every new capability ships as **a new module satisfying one of them**, injected at
construction. `omni_cli.py` already passes both in (`executor=shell`,
`repair_policy_factory=...`), so integration is a constructor argument, not a rewrite.

```
        ┌─────────────────────────────────────────────┐
        │ agent_runtime.py    (Phase 0 fixes only)    │
        │ RepairLoop · PlannerLoop · Orchestrator     │
        │ GuardrailStack · Ledger · RunJournal        │
        └────────┬───────────────────────┬────────────┘
                 │ ToolExecutor          │ Policy
                 ▼                       ▼
      ┌────────────────────┐   ┌────────────────────┐
      │ agentkit/          │   │ agentkit/          │
      │   dispatch.py      │   │   gate.py          │
      │   tools/fs.py      │   │   (VerifyGate)     │
      │   tools/shell.py   │   │                    │
      └────────────────────┘   └────────────────────┘
            Phases 1-2                Phase 3
```

### Layout

```
interactive-omni-cli/
├─ agent_runtime.py           # Phase 0 only
├─ llm_backends.py            # untouched
├─ omni_cli.py                # Phase 4 wiring fixes + Phase 5 integration
└─ agentkit/                  # NEW
   ├─ __init__.py             # build_executor(), build_policy()
   ├─ jail.py                 # path containment, used by policy AND fs tools
   ├─ registry.py             # ToolSpec, ToolRegistry, decision schema
   ├─ dispatch.py             # MultiToolExecutor -> ToolExecutor
   ├─ policy.py               # ToolCallPolicy(LLMPolicy)
   ├─ verify.py               # VerifySpec, detect_verify, run_verify
   ├─ gate.py                 # VerifyGate -> Policy
   └─ tools/
      ├─ fs.py                # read_file, write_file, edit_file, list_dir, search_files
      └─ shell.py             # run_command (wraps CommandPolicy + SubprocessShell)
tests/
   test_jail.py  test_registry.py  test_fs_tools.py
   test_dispatch.py  test_gate.py  test_end_to_end.py
```

---

## Phase 0 — Close the perimeter (3 h) — **do this first, alone**

Real execution + real writes + a bypassable jail. Ship nothing else until this lands.

### 0a. Fix path containment — `agent_runtime.py:466-475`

The check only fires on `startswith("/")` or a `..` component, so Windows drive-absolute paths
are never examined, and `startswith` compares strings rather than paths.

```python
# agentkit/jail.py — single implementation, used by CommandPolicy and the fs tools
def contained(path: Path, root: Path) -> bool:
    a = Path(os.path.normcase(str(path.resolve())))
    b = Path(os.path.normcase(str(root.resolve())))
    return a == b or a.is_relative_to(b)

def resolve_in(root: Path, token: str) -> Path:
    """Raises JailBreak if token escapes root. Rejects reserved Windows device names."""
```

```python
# CommandPolicy.classify — check EVERY non-flag token
for token in argv[1:]:
    if token.startswith("-"):
        continue
    candidate = Path(token)
    resolved = (candidate if candidate.is_absolute() else self.workspace / token).resolve()
    if not contained(resolved, self.workspace):
        return PolicyDecision(Risk.FORBIDDEN, argv,
                              f"path '{token}' escapes the workspace root", violations)
```

### 0b. Parse once — `agent_runtime.py:444`, `:657`

The policy's validated `verdict.argv` is discarded and the executor re-splits the raw string.
Pass it through in `RepairLoop.run` and `PlannerLoop._execute`:

```python
call = decision.call.model_copy(update={
    "args": {**decision.call.args, "argv": verdict.argv}
})
```

and use `posix=(os.name != "nt")` at both `shlex.split` sites — the POSIX default destroys
every backslash in a Windows path.

### 0c. Route `handle_direct_code` writes through the jail — `omni_cli.py:280`

```python
target_file = agentkit.jail.resolve_in(workspace, filename.strip())   # raises on escape
tmp = target_file.with_suffix(target_file.suffix + ".tmp")
tmp.write_text(code, encoding="utf-8")
os.replace(tmp, target_file)          # atomic
```

Also stop discarding code blocks after the first (`omni_cli.py:276-278`): if the response has
multiple blocks, prompt per block or write each to its labelled filename.

### 0d. Narrow the child environment — `agent_runtime.py:648-651`

`os.environ.copy()` hands every shell token and API key to child processes. Pass an allowlist
(`PATH`, `SYSTEMROOT`, `TEMP`, `TMP`, `PATHEXT`, `HOME`/`USERPROFILE`, `LANG`) plus anything
the user opts into.

**Acceptance:** parametrised test asserting BLOCKED for `C:/Users/<user>/.ssh/id_rsa`,
`C:/Windows/System32/config/SAM`, `../../secrets`, `/etc/passwd`, `//server/share/x`, and the
sibling `…/python-evil`; ALLOWED for `./src/main.py` and a bare relative name. A Windows path
survives policy → executor with separators intact. `handle_direct_code` refuses
`../../evil.py` and `C:/Windows/Temp/x`.

---

## Phase 1 — Tool registry and dispatch (4 h)

### `agentkit/registry.py`

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str          # one line, rendered into the system prompt
    params: dict[str, Any]    # JSON Schema for args
    risk: Risk                # reuses agent_runtime.Risk
    mutating: bool            # drives the VerifyGate dirty flag
    handler: Callable[..., Awaitable[ToolOutcome]]

class ToolRegistry:
    def register(self, spec) -> None
    def specs(self) -> list[ToolSpec]
    def render_for_prompt(self) -> str
    def decision_schema(self) -> dict      # for constrained decoding
```

Handlers return a plain `ToolOutcome(ok, exit_code, output)` and stay ignorant of `call_id`,
redaction, and truncation — `dispatch.py` adds those. A new tool is ~15 lines.

### `agentkit/dispatch.py`

```python
class MultiToolExecutor:                      # satisfies ToolExecutor
    async def execute(self, call: ToolCall, timeout_s: float) -> ToolResult:
        spec = self.registry.get(call.tool)
        if spec is None:
            return ToolResult(..., ok=False, exit_code=127,
                              output=f"unknown tool {call.tool!r}; available: {names}",
                              error_class=ErrorClass.POLICY)
        # jsonschema-validate call.args; on failure return exit_code=22 with the
        # validation message as output, so the model self-corrects
        outcome = await asyncio.wait_for(spec.handler(**call.args), timeout_s)
        if spec.mutating and outcome.ok:
            self.dirty = True
        output, truncated, redactions = finalize_output(outcome.output)   # reused
        return ToolResult(...)
```

Three properties inherited free by reusing `ToolResult` and `finalize_output`:

- **Bad args never raise** — they come back as an observation, the same discipline the loop
  already applies to policy refusals.
- **Redaction-before-truncation applies to file reads too** — a `read_file` on a `.env` is
  redacted identically to shell output.
- **Guardrails keep working** — `ToolCall.fingerprint` is `digest(tool, args)`, so
  `RepetitionDetector` already catches "read the same file four times" with no change.

### `agentkit/policy.py`

```python
class ToolCallPolicy(LLMPolicy):          # subclass; do not edit LLMPolicy
    @property
    def SYSTEM(self) -> str: ...          # base contract + registry.render_for_prompt()

    def _parse(self, text: str) -> Decision:
        # {"kind":"act","thought":"...","tool":"write_file","args":{...}}
        # back-compat: {"command": "..."} -> tool="run_command", args={"command":...}
```

Pass `schema=registry.decision_schema()` so the inherited `propose` gets constrained decoding.
The existing one-shot repair round-trip and `AskHuman` degradation are kept as-is.

`SmartPlanner.SCHEMA` (`omni_cli.py:117`) needs the same treatment — plan steps become
`{title, tool, args}` rather than `{title, command}`, with the old shape still accepted.

**Acceptance:** a `ScriptedPolicy` emitting `Act(tool="list_dir")` runs through an unmodified
`RepairLoop` and produces a `StepRecord`. Unknown tool and bad args both yield observations,
not exceptions.

---

## Phase 2 — Filesystem tools (5 h)

| Tool | Args | Risk | Mutating | Notes |
|---|---|---|---|---|
| `read_file` | `path`, `offset?`, `limit?` | SAFE | no | line-numbered; refuses binary; 2,000-line cap |
| `list_dir` | `path?`, `glob?` | SAFE | no | skips `.git`, `venv`, `node_modules`; 200-entry cap |
| `search_files` | `pattern`, `glob?`, `max_results?` | SAFE | no | returns `path:line: text` |
| `write_file` | `path`, `content` | ELEVATED | yes | atomic temp + `os.replace`; creates parents |
| `edit_file` | `path`, `old`, `new`, `replace_all?` | ELEVATED | yes | **fails if `old` is not unique** |

Design notes:

- Every path goes through `agentkit/jail.py` — the same implementation Phase 0a gives
  `CommandPolicy`. One containment rule, one place to get it right.
- **fs tools bypass `CommandPolicy` deliberately.** They are not shell strings; the
  metacharacter ban is meaningless for them and the jail is the correct control. This is
  precisely why the agent cannot currently write a file through the loop at all: `>` is banned
  and `echo` alone cannot redirect.
- **`edit_file` failing on a non-unique `old` is intentional.** Silent multi-replace is the
  most common way an agent corrupts a file. The failure is a clean observation: "found 3
  matches, provide more context."
- `write_file`/`edit_file` at `Risk.ELEVATED` route through the **existing** approval path —
  but only after Phase 4b fixes the approval token, or every write will livelock.
- Config flag `auto_approve_writes` for unattended runs; off by default.

**Acceptance:** `test_jail.py` covers `..`, drive-absolute, UNC, symlink escape, and Windows
reserved device names. Round-trip: `write_file` then `read_file` returns the content;
`edit_file` with ambiguous `old` returns `ok=False` and does not modify the file.

---

## Phase 3 — Verify gate: make success require evidence (4 h)

Today `Finish(succeeded=True)` ends the run on the model's assertion. This is the gap between
"runs commands" and "gets things done".

### `agentkit/verify.py`

```python
@dataclass(frozen=True)
class VerifySpec:
    command: str            # "pytest -q" | "python -m compileall -q ."
    timeout_s: float = 120

def detect_verify(workspace: Path) -> VerifySpec | None
```

`detect_verify` picks by inspection: `pytest.ini`/`tests/` → `pytest -q`; `package.json` with a
`test` script → `npm test`; else `python -m compileall -q .`. A weak verifier beats none —
compile-checking catches the most common local-model failure, which is emitting code that does
not parse.

### `agentkit/gate.py`

```python
class VerifyGate:                          # satisfies Policy — wraps another Policy
    async def propose(self, ctx: PolicyContext) -> Decision:
        decision = await self.inner.propose(ctx)
        if not isinstance(decision, Finish) or not decision.succeeded:
            return decision
        if not self.executor.dirty or self.rounds >= self.max_rounds:
            return decision                # nothing was written, or budget spent
        result = await run_verify(self.spec, ...)
        if result.ok:
            return decision                # verified success
        self.rounds += 1
        return Act(thought="Verification failed; fixing before finishing.",
                   call=ToolCall(tool="run_command",
                                 args={"command": self.spec.command}))
```

**Why a `Policy` decorator rather than a change to `RepairLoop`:** it composes (stack a lint
gate on a test gate), it is testable with a `ScriptedPolicy`, and `RepairLoop` /
`PlannerLoop` / `Orchestrator` need zero edits. Every existing safety mechanism still applies
to the injected step — `Ledger.tick()` charges it, `NoProgressDetector` catches an agent that
keeps "fixing" without changing the observation hash, `ConsecutiveErrorDetector` suspends
after three failed rounds.

**Acceptance:** a `ScriptedPolicy` that writes a syntactically broken file and immediately
returns `Finish(succeeded=True)` must **not** terminate the run — the outcome is either a
repaired file or `StopReason.GUARDRAIL`, never a false success.

---

## Phase 4 — Repair the wiring (2 h)

Three sound subsystems that currently do nothing. Small diffs, high value.

### 4a. Charge the real ledger — `omni_cli.py:54, 482`

`WrappedLLMPolicy` builds its own `Ledger`, so `LLMPolicy` charges tokens somewhere
`RepairLoop` never reads, and `repair_factory` rebuilds it per call. Hoist one ledger per run
and inject it:

```python
run_ledger = agent_runtime.Ledger(budget=agent_runtime.Budget())
repair_factory = lambda cmd: agent_runtime.LLMPolicy(
    client=client, ledger=run_ledger, schema=llm_backends.STRICT_DECISION_SCHEMA)
```

Better still: let `Orchestrator.handle` own the ledger and pass it to the factory, so
`Budget(max_tokens=150_000, max_usd=2.00)` is actually enforced. `WrappedLLMPolicy` then has no
reason to exist.

### 4b. Fix the approval token — `omni_cli.py:570`

```python
approvals_list = ["sudo"] if "sudo" in pending_cmd else [pending_cmd]
```

Both loops gate on the literal string `"sudo"`, so approving `rm`, `chmod`, `npm publish`, or
`git push` re-suspends on the same command forever. Either grant the token the loop checks, or
— better — change the loops to check membership properly:

```python
if verdict.risk is Risk.ELEVATED and not approvals_grant(verdict, approvals):
```

where `approvals_grant` accepts `"sudo"`, the exact command, or the executable name. Add a
regression test that approves `rm build/` once and sees it execute.

### 4c. One `Orchestrator` per session — `omni_cli.py:511, 535`

Constructing it per turn creates a new `RunJournal` and a new `run_id`, so
`self.state` is always `IDLE` and the `SUSPENDED → RESUMING` edge never fires. Build it once
before the loop, or reload via `RunJournal.load(workspace / "run.jsonl")` when resuming.

---

## Phase M — Context memory ✅ done

`agentkit/memory.py` implements the specified `ContextStore` flow: `context/index.json`
initialisation, `add_context` append-and-save, `embed_text` that never raises, `_adaptive_n`
per-intent windows (coder 2 / chat 3 / architect·planner 5), `get_recent` with same-session
priority plus global back-fill and embedding auto-repair, and `retrieve_best_context` ranking
by cosine over the recent window.

Four corrections were applied, none of which change that flow:

1. **Atomic saves.** `open(path,"w")` truncated the whole index before writing; a crash
   destroyed it, the loader swallowed the `JSONDecodeError` and returned `[]`, and the next
   add overwrote the remains. Now temp-file + `os.replace`, and a corrupt index is preserved
   as `index.corrupt-<ts>.json`.

2. **Failed embeddings no longer win retrieval.** The zero-vector fallback is kept — callers
   still never see a missing embedding — but the entry is tagged `embedding_ok: False` and
   excluded from ranking. Previously every cosine was `0.0` while `best_score` started at
   `-1.0`, so the *first* candidate was returned as the "most similar" match: the store
   reported success while retrieving at random. With no usable embeddings it now degrades to
   most-recent and says so once in the UI.

3. **Dimension mismatch scores 0.0** instead of `zip()` truncating to the shorter vector,
   which turned a change of embedding model into confident nonsense.

4. **Timeouts** on the embedding call, which previously had none.

Two additions beyond the spec, both driven by the earlier review:

- **Both sides of the exchange are stored** (`USER: … ASSISTANT: …`). Storing only the model's
  output leaves a memory of answers with no questions — the half a follow-up query needs to
  match against.
- **Recalled text is fenced** by `with_context()` as `<<<RECALLED_CONTEXT>>>`, mirroring
  `render_observation`, so prior model output re-entering the prompt is read as reference
  rather than instruction.

Borrowed entries are tagged `from_other_session`, so the global back-fill — which does mix
unrelated work into a fresh session — is at least visible in the UI line.

**Verified live** against Ollama + `nomic-embed-text`: with "fastapi/uvicorn" and
"pytest/fixtures" stored, the query *"how do I test it?"* returned the pytest entry at
similarity 0.591.

**Known cost, unchanged from the spec:** embeddings are stored inline in a single JSON file
that is fully re-read and re-written on every add. Measured at **~21 KB per entry** (43 KB for
two). `MemoryConfig.max_entries` provides an opt-in cap; unbounded remains the default. If
this becomes a problem, move vectors to a sidecar `.npy`/JSONL and keep `index.json` as
metadata — the `ContextStore` interface does not need to change.

---

## Phase 5 — Make the agent see the repo (4 h)

`handle_project_review` (`omni_cli.py:318-362`) feeds the model a 3,000-char truncation of
`ls -R` plus a few manifests. It never reads a source file, so the "Principal Software
Architect" review is structurally blind — and `ls -R` on a repo with `venv/` spends the whole
budget on dependency paths.

Once Phase 2 lands, rewrite it on top of the tools:

1. `list_dir` with ignore rules → real file tree, dependencies excluded.
2. Rank candidates: entry points, largest source files, files touched most recently in git.
3. `read_file` the top N under an explicit token budget.
4. `search_files` for whatever the review prompt names.

This turns the review from "guess from filenames" into actual code review, and it exercises
the tool layer on a read-only task — a safe first integration.

**Optional, same phase:** streaming. All three handlers generate up to 3,500 tokens behind a
spinner. `llm_backends` already speaks to servers that stream; surfacing it via `rich.live`
would be the single largest perceived-latency win.

---

## Sequencing

| Phase | Delivers | Depends on | Effort |
|---|---|---|---|
| 0 | Perimeter holds | — | 3 h |
| 1 | Tool registry, dispatch, multi-tool policy | 0 | 4 h |
| 2 | Filesystem tools | 1 | 5 h |
| 3 | Verify gate — evidence-backed success | 1, 2 | 4 h |
| 4 | Ledger / approvals / orchestrator wiring | — (independent) | 2 h |
| 5 | Repo-aware review, streaming | 2 | 4 h |

**Total ≈ 22 h.** Phase 4 is independent and can be done any time — it is the best
value-per-hour in the plan. Phase 0 gates everything that writes or executes.

## Golden acceptance tasks

Run against a scratch workspace with a real local model. These are the honest measure; unit
tests will pass long before the agent works.

1. **Create + verify** — "Write `fizzbuzz.py` with a `fizzbuzz(n)` function and a pytest file
   covering n=3,5,15. Make the tests pass." Exercises `write_file` ×2, `run_command`, gate.
2. **Read + edit** — "Add a `--verbose` flag to `omni_cli.py` that prints the resolved intent."
   Exercises `read_file`, `edit_file` uniqueness, non-corruption.
3. **Debug loop** — seed a file with a deliberate `NameError`; "run the tests and fix what
   fails." This is the capability. Nothing else matters if this fails.
4. **Refusal** — "Read C:/Users/wasim/.ssh/id_rsa." Must be blocked, surface as an
   observation, and not end the run.
5. **False-success trap** — a task the model will claim to have finished without writing
   anything. `VerifyGate` must catch it.
6. **Approval** — an elevated command approved once must execute, not re-suspend.

## Risks

| Risk | Mitigation |
|---|---|
| Real exec + writes on a Windows host with no container | Phase 0 first; scratch workspace + git snapshot; stop defaulting the workspace to `./` |
| Small local models emit malformed tool calls | Constrained decoding via `registry.decision_schema()`; inherited repair round-trip; `AskHuman` degradation |
| Tool output blows the context window | `finalize_output` 4 KB cap inherited; `read_file` line caps; `Summarizer` handles long runs |
| Prompt injection via file contents | Route `read_file` results through `render_observation`, never raw concatenation |
| A 7B model cannot drive a 5-tool loop | `ModelRegistry` role bindings already exist — small model for routing, large for repair |
| Widening the command allowlist to compensate | Don't. `run.jsonl` shows both runs suspending on `find` and `ls /`; tools are the fix, not a bigger allowlist |

## Out of scope

Sub-agents, cross-session memory, and the retrieval/embedding work. Those come after the tool
layer and the verify gate exist.

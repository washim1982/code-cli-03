# Omni CLI

A local-first autonomous developer agent. It plans, writes files, runs commands, and will not
report success until a verification step actually passes.

Everything runs against a local inference server — Ollama, LM Studio, or llama.cpp. No API key,
no cloud round-trip.

---

## Requirements

| | |
|---|---|
| Python | 3.11 or newer |
| A local LLM server | Ollama, LM Studio, or llama.cpp — at least one running |
| Disk | the agent writes only inside the workspace you choose |

Runtime dependencies (four, all pure Python):

```
rich>=13.7.0          terminal rendering — panels, tables, markdown
questionary>=2.0.1    interactive prompts
pydantic>=2.5.0       typed tool calls, results, and resume payloads
httpx>=0.25.0         async HTTP to the inference backends
```

---

## Install

A virtual environment is strongly recommended — installing into system Python puts the `omni`
command and these four packages on your global interpreter.

```bash
python -m venv .venv
```

Activate it — **Windows PowerShell**:

```bash
.venv\Scripts\Activate.ps1
```

**macOS / Linux**:

```bash
source .venv/bin/activate
```

Then install the project in editable mode, which also installs the dependencies and puts an
`omni` command on your PATH:

```bash
pip install -e .
```

If you would rather not install the package at all, just fetch the dependencies:

```bash
pip install -r requirements.txt
```

---

## Set up a model

The CLI probes all three backends at startup and shows you which are online, so you only need
one of them.

**Ollama** — pull a chat model and, for the memory feature, the embedding model:

```bash
ollama pull qwen3:8b
```

```bash
ollama pull nomic-embed-text
```

**LM Studio** — load a model and start the local server from the Developer tab.

**llama.cpp** — run `llama-server` with your GGUF.

Default endpoints the CLI probes:

| Backend | URL |
|---|---|
| Ollama | `http://localhost:11434` |
| LM Studio | `http://localhost:1234` |
| llama.cpp | `http://localhost:8080` |

A different host or port can be entered at the model-selection prompt via
**`[Custom] Enter backend URL and model name manually`**.

`nomic-embed-text` is only needed for cross-session memory recall. Without it the agent works
normally; recall degrades to most-recent and says so on screen.

---

## Run

After `pip install -e .`:

```bash
omni
```

Straight from a clone, without installing:

```bash
python -m omni.cli
```

That form needs `src` on the import path. From the repo root:

```bash
PYTHONPATH=src python -m omni.cli
```

On **Windows PowerShell**:

```bash
$env:PYTHONPATH="src"; python -m omni.cli
```

### What it asks you at startup

1. **Select inference model** — a table of probed backends, then a list of their models.
   `[Simulation]` runs the whole loop with no LLM, useful for seeing the machinery work.
2. **Enter workspace directory** — the only directory the agent can read or write. It is
   created if missing. **Point this at a scratch directory, not at this repo.**
3. **Allow the agent to create and edit files without asking each time?** — `No` (the default)
   prompts for approval before every write. `Yes` is for unattended runs; containment still
   applies either way.

Then type a request at the `Omni>` prompt. `exit`, `quit`, or `q` to leave.

### Reading the output

After each task the CLI prints two tables.

**Execution History & Observations** — what the tools did: the thought behind each step, the
call, its exit status, and the first line of output.

**Loop Trace** — what the loop did around them, straight from the run journal: which route was
chosen and how confidently, the plan it produced, the policy verdict on each call (including
why anything was blocked), guardrail warnings, the suspension reason, and the run's iteration
/ token / time cost. Timings are milliseconds from the start of the run.

```
Loop Trace — run_e61a5a4079fb
 #    +ms   Phase        State            Detail
 1      0   state        ROUTING
 2      0   route                         PLAN (p=0.92) — build request
 3      0   state        PLANNING
 4      1   plan                          3 step(s): Create module; Escape attempt; Run tests
 5      1   state        EXECUTING
 6     36   step                          Create module -> exit 0
 7     37   step                          Escape attempt -> exit 126
 8     38   step failed                   Escape attempt (POLICY)
 9     38   state        SUSPENDED_HITL
10     38   suspended    SUSPENDED_HITL   repair of 'Escape attempt' failed: ...
```

Type `/trace` to toggle the loop trace off or on. The same events are always written to
`run.jsonl` in the workspace regardless, so a run can be inspected after the fact.

---

## What you can ask for

The prompt is classified into one of four modes:

| Mode | Triggered by | What happens |
|---|---|---|
| **Guided learning** | *"How do I build an agentic loop?"*, *"Explain FSM"* | A structured tutorial. No files touched. |
| **Direct code** | *"Write a python script to add two numbers"* | One file generated; you confirm the filename before it is saved. |
| **Project review** | *"Review project and suggest enhancements"* | Reads the real source of the workspace and writes an architectural report. |
| **Agent task** | *"Create a javascript project…"*, *"Fix the failing tests"* | Plans, writes files, runs commands, verifies. |

Anything spanning more than one file routes to **agent task** — a scaffold cannot be produced
as a single capped completion.

---

## What the agent can do

Seven tools. Reads are free; writes need approval unless you allowed them at startup.

| Tool | Risk |
|---|---|
| `read_file(path, offset?, limit?)` | safe |
| `list_dir(path?, glob?)` | safe |
| `search_files(pattern, glob?, max_results?)` | safe |
| `run_command(command)` | governed by the command allowlist |
| `write_file(path, content)` | needs approval |
| `edit_file(path, old, new, replace_all?)` | needs approval |
| `generate_file(path, spec)` | needs approval |

`generate_file` is preferred for substantial files: the plan carries a one-line description and
the content is produced in its own request, so a large project does not have to fit in a single
response.

### Command allowlist

`run_command` takes **one** command. Pipes, redirects, and chaining are rejected outright — not
escaped — which is why file creation goes through the filesystem tools rather than `echo >`.

```
cat  dir  echo  git  ls  mkdir  node  npm  npx  pip  pytest  python  python3  ruff  type
```

`rm` and `chmod` are allowed but always require approval. `git`, `npm`, and `pip` are further
restricted to a subcommand list (`git status/add/commit/diff/log/…`, `npm install/test/run/…`,
`pip install/list/show/…`).

Every path argument is confined to the workspace. Absolute paths, `..` traversal, UNC paths,
and symlinks pointing outside are all refused.

### Verification

When a task modifies the workspace, the agent is not allowed to simply declare success. It has
to run a check first, chosen from what is present:

| Found | Check |
|---|---|
| `pytest.ini`, `conftest.py`, `tests/`, or `test_*.py` | `pytest -q` |
| `package.json` with a `test` script | `npm test` |
| any `.py` file | `python -m compileall -q .` |
| nothing checkable | skipped |

If the check keeps failing, the run is reported as **failed** rather than succeeded.

---

## Tests

```bash
python -m pytest -q
```

311 tests across eight suites. `pytest.ini` sets `pythonpath = src`, so this works from any
directory without installing the package.

---

## Project layout

```
interactive-omni-cli/
├─ pyproject.toml          packaging; defines the `omni` command
├─ pytest.ini
├─ requirements.txt
├─ src/omni/
│  ├─ pathguard.py         workspace path containment; standard library only
│  ├─ backends.py          Ollama / LM Studio / llama.cpp clients
│  ├─ runtime.py           agent loop: FSM, journal, budget ledger, guardrails
│  ├─ agentkit/            tools, dispatch, verification gate, memory, repo survey
│  └─ cli.py               interactive front end
├─ test/
└─ doc/                    architecture notes, review, enhancement plan
```

Dependencies run strictly one way — `pathguard` ← `runtime` ← `agentkit` ← `cli` — with no
cycles.

---

## Files the agent writes into your workspace

| Path | What it is |
|---|---|
| `run.jsonl` | append-only event journal of every run; replayable |
| `context/index.json` | memory store, including embeddings |

Both are runtime state, not source. This repo's `.gitignore` already excludes them; add them to
your project's if you point the agent at a git repository.

---

## Troubleshooting

**All three backends show Offline.** Nothing is listening. Check with `ollama list`, or confirm
the LM Studio server is started. For a non-default port use the `[Custom]` option.

**`llamacpp: Client error '401 Unauthorized'`.** `llama-server` is running with an API key the
CLI is not sending. Use a different backend or restart the server without auth.

**Recall says `0/N ranked`.** The embedding model is unavailable, so retrieval fell back to
most-recent. `ollama pull nomic-embed-text` fixes it; deleting `context/` clears entries stored
before it was available.

**A run suspends with "requires operator approval".** Expected — an elevated tool or command
was proposed. Approve at the prompt and the run resumes from where it stopped.

**A run suspends with "the same call has now been issued N times".** A guardrail stopped the
agent going in circles. Re-prompt with a more specific instruction.

**`ModuleNotFoundError: No module named 'omni'`.** You are running `python -m omni.cli` without
`src` on the path and without having installed the package. Use `pip install -e .` or set
`PYTHONPATH=src`.

**A generated web project fails in the browser with `file:` origin errors.** That is a browser
restriction, not a bug in the generated code. Serve the directory over HTTP:

```bash
python -m http.server 8000
```

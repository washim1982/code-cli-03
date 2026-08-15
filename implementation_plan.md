# Interactive Omni CLI Implementation Plan

Provide an interactive CLI application utilizing the existing `agent_runtime.py` and `llm_backends.py`. This application will serve as the frontend for the v2 agent loop architecture, allowing users to effortlessly select workspaces, models, and initiate various types of tasks (chat, project creation, bug fixing, research, guided learning).

## User Review Required

> [!IMPORTANT]
> **External Dependencies**: To make the CLI truly interactive and beautifully formatted, I propose using two excellent Python libraries:
> - `rich`: For rendering beautiful markdown, tables, and colored terminal output.
> - `questionary`: For interactive prompts (dropdowns, selections for models and workspaces).
>
> Please confirm if adding these dependencies is acceptable, or if you prefer a strict standard-library-only approach (which would be less visually impressive).

## Open Questions

> [!WARNING]
> 1. **Entry Point Name**: I plan to name the main script `omni_cli.py`. Is this acceptable, or do you have a preferred name?
> 2. **Default Workspace**: Should the CLI default to the current working directory, or should it always prompt the user to enter a workspace path?
> 3. **Normal Chat vs. Tasks**: `agent_runtime.py` is heavily optimized for task execution (Sense-Think-Act). For "normal chat" or "guided learning", should the agent still use the full Orchestrator FSM, or should it bypass the FSM and perform direct LLM completion? (I recommend running it through the Orchestrator so it can still use tools if needed to answer questions).

## Proposed Changes

### Core CLI Application

#### [NEW] `omni_cli.py`
This will be the main entry point for the application.
- **Initialization**: Probe available models via `llm_backends.py` (Ollama, LMStudio, LlamaCpp) and let the user select their preferred backend/model using a dropdown.
- **Workspace Selection**: Prompt the user to enter or confirm the workspace directory.
- **Main Loop**: An interactive loop prompting the user for their request.
- **Request Handling**:
  - Parse the user's intent.
  - Instantiate `Orchestrator` and `RunJournal` from `agent_runtime.py`.
  - Provide a callback mechanism or rich live display to stream the agent's progress, state transitions, tool executions, and warnings to the console.
- **Formatting**: Use `rich.console.Console` and `rich.markdown.Markdown` to format the LLM's output, errors, and tool results elegantly.

#### [NEW] `requirements.txt`
To specify the required dependencies (`rich`, `questionary`, `httpx`, `pydantic`).

## Verification Plan

### Manual Verification
1. Run `python omni_cli.py`.
2. Verify that the CLI successfully probes local LLM backends and presents a selection menu.
3. Select a workspace and a model.
4. Input a simple "research" task and observe the formatted, interactive output.
5. Input a "create a new project" task and observe the agent scaffolding files in the workspace.
6. Trigger a guardrail or error and observe how the CLI presents the suspension (e.g., `SUSPENDED_APPROVAL`) and asks for user input.

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Enable UTF-8 encoding on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import agent_runtime
import llm_backends

console = Console(legacy_windows=False if sys.platform == "win32" else None)


def robust_json_parse(text: str) -> dict:
    """Extracts and parses JSON even if wrapped in markdown code fences or conversational text."""
    clean = text.strip()
    # Remove markdown code fences if present
    if "```" in clean:
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean, flags=re.MULTILINE).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: extract the outermost balanced/contiguous JSON object
        match = re.search(r"(\{[\s\S]*\})", clean)
        if match:
            return json.loads(match.group(1))
        raise


class WrappedLLMPolicy:
    """Wrapper to make LLMPolicy compatible with the factory pattern in Orchestrator."""
    def __init__(self, client: Any):
        self.client = client
        self.ledger = agent_runtime.Ledger(budget=agent_runtime.Budget())
        self.policy = agent_runtime.LLMPolicy(
            client=self.client, 
            ledger=self.ledger, 
            schema=llm_backends.STRICT_DECISION_SCHEMA
        )
        
    async def propose(self, ctx: agent_runtime.PolicyContext) -> agent_runtime.Decision:
        return await self.policy.propose(ctx)


class SmartRouter:
    """Heuristic router with LLM fallback for ambiguous or un-signaled intent."""
    SCHEMA = {
        "type": "object",
        "properties": {
            "route": {"type": "string", "enum": ["PLAN", "REPAIR", "CLARIFY"]},
            "rationale": {"type": "string"}
        },
        "required": ["route", "rationale"],
        "additionalProperties": False
    }

    SYSTEM = (
        "You are an intelligent task routing system for a software development agent.\n"
        "Analyze the user request and decide the appropriate route:\n"
        "- PLAN: Building, creating, project review/enhancement, guided learning, research, or running inspections.\n"
        "- REPAIR: Fixing errors, resolving stack traces, unblocking broken environments, or patching bugs.\n"
        "- CLARIFY: Completely meaningless, empty, or uninterpretable prompt.\n"
        "Return a JSON object with 'route' and 'rationale'."
    )

    def __init__(self, client: Any | None = None) -> None:
        self.heuristic = agent_runtime.Router()
        self.client = client

    async def route(self, prompt: str) -> agent_runtime.RouteDecision:
        decision = self.heuristic.route(prompt)
        if decision.route is not agent_runtime.Route.CLARIFY or self.client is None:
            return decision

        console.print("[dim]Heuristic router abstained. Asking LLM to analyze intent...[/dim]")
        try:
            completion = await self.client.complete(
                self.SYSTEM, 
                f"User request: {prompt}", 
                schema=self.SCHEMA,
                max_tokens=256
            )
            data = robust_json_parse(completion.text)
            route_str = str(data.get("route", "PLAN")).upper()
            rationale = str(data.get("rationale", "LLM determined intent"))
            route_val = getattr(agent_runtime.Route, route_str, agent_runtime.Route.PLAN)
            return agent_runtime.RouteDecision(route_val, 0.9, f"LLM Intent: {rationale}", decision.signals)
        except Exception as exc:
            console.print(f"[dim yellow]LLM router fallback notice ({exc}); defaulting to PLAN[/dim yellow]")
            return agent_runtime.RouteDecision(
                agent_runtime.Route.PLAN, 0.5, "Defaulted to PLAN after router clarify", decision.signals
            )


class SmartPlanner:
    """Dynamic LLM-based planner that constructs tailored PlanSteps for executable tasks."""
    SCHEMA = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "command": {"type": "string"}
                    },
                    "required": ["title", "command"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["steps"],
        "additionalProperties": False
    }

    SYSTEM = (
        "You are an autonomous engineering planner.\n"
        "Generate 1 to 4 concrete, safe command steps to execute the user's task.\n"
        "Allowed executables: ls, dir, cat, type, echo, mkdir, node, python, python3, pytest, npx, npm, git.\n"
        "For repository inspection, prefer git status, git ls-files, ls, or python scripts.\n"
        "No shell pipes (|), redirects (>), or chaining (&&). Single commands only.\n"
        "Return a JSON object: {\"steps\": [{\"title\": \"...\", \"command\": \"...\"}]}"
    )

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    async def plan(self, goal: str) -> list[agent_runtime.PlanStep]:
        if self.client is None:
            return agent_runtime.default_plan(goal)

        console.print("[dim]Generating customized execution plan with LLM...[/dim]")
        try:
            completion = await self.client.complete(
                self.SYSTEM,
                f"Goal: {goal}",
                schema=self.SCHEMA,
                max_tokens=512
            )
            data = robust_json_parse(completion.text)
            steps_data = data.get("steps", [])
            plan_steps = []
            for s in steps_data:
                title = str(s.get("title", "Execute command"))
                command = str(s.get("command", "")).strip()
                if command:
                    plan_steps.append(agent_runtime.PlanStep(title, command))
            if plan_steps:
                return plan_steps
        except Exception as exc:
            console.print(f"[dim yellow]LLM planner fallback ({exc}); using exploration plan[/dim yellow]")

        return [
            agent_runtime.PlanStep("Explore directory structure", "ls"),
            agent_runtime.PlanStep("Check git status", "git status")
        ]


async def classify_intent(prompt: str, client: llm_backends.ModelClient | None) -> str:
    """Classifies user interaction into LEARN_OR_CHAT, DIRECT_CODE, PROJECT_REVIEW, or EXECUTE_TASK."""
    p_lower = prompt.lower().strip()
    
    # 1. Fast heuristic checks for guided learning & Q&A
    if any(p_lower.startswith(k) for k in ["what is", "how do i", "how to", "explain", "teach me", "guide me", "i want to learn", "tell me about", "why does"]):
        return "LEARN_OR_CHAT"

    # 2. Fast heuristic checks for direct code generation / writing scripts
    if any(p_lower.startswith(k) for k in [
        "write ", "create a script", "generate a python", "write a function", "make a script",
        "generate script", "create script", "give me code", "give me a python", "write python",
        "code for", "script to"
    ]):
        return "DIRECT_CODE"

    # 3. Fast heuristic checks for repository reviews
    if any(k in p_lower for k in ["review project", "review the project", "project review", "suggest enhancement", "audit the codebase", "analyze repository", "code review"]):
        return "PROJECT_REVIEW"

    if client is None:
        if any(w in p_lower for w in ["write", "script", "create", "python", "code"]):
            return "DIRECT_CODE"
        return "EXECUTE_TASK"

    schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": ["LEARN_OR_CHAT", "DIRECT_CODE", "PROJECT_REVIEW", "EXECUTE_TASK"]
            },
            "rationale": {"type": "string"}
        },
        "required": ["intent"],
        "additionalProperties": False
    }

    system = (
        "You are an intent classifier for a developer AI assistant.\n"
        "Classify the user prompt into exactly ONE category:\n"
        "- LEARN_OR_CHAT: Asking a conceptual question, tutorial, or explanation.\n"
        "- DIRECT_CODE: The user wants a standalone script, snippet, function, or file written directly without executing shell terminal actions.\n"
        "- PROJECT_REVIEW: The user wants to review, audit, or analyze the codebase in the workspace.\n"
        "- EXECUTE_TASK: Complex operations requiring shell commands (running test suites, executing builds, fixing terminal errors, deploying).\n"
        "Return a JSON object: {\"intent\": \"LEARN_OR_CHAT\" | \"DIRECT_CODE\" | \"PROJECT_REVIEW\" | \"EXECUTE_TASK\"}"
    )
    
    try:
        completion = await client.complete(system, f"User request: {prompt}", schema=schema, max_tokens=150)
        data = robust_json_parse(completion.text)
        return data.get("intent", "EXECUTE_TASK")
    except Exception:
        if any(w in p_lower for w in ["write", "script", "create", "python", "code", "generate"]):
            return "DIRECT_CODE"
        return "EXECUTE_TASK"


async def handle_direct_code(prompt: str, client: llm_backends.ModelClient | None, workspace: Path) -> None:
    """Generates code directly from prompt and optionally saves it directly to disk."""
    if client is None:
        console.print(Panel(
            "[yellow]Simulation Mode:[/yellow] Connect an active LLM backend to generate code directly.",
            title="⚡ Direct Code Generation",
            border_style="yellow"
        ))
        return

    system = (
        "You are an expert Senior Software Engineer.\n"
        "Generate clean, robust, and production-ready code for the user's request.\n"
        "Format your output in clean Markdown with code blocks.\n"
        "Include concise explanations and indicate the suggested filename at the top of the code block if applicable."
    )

    with console.status("[bold cyan]Generating code directly with LLM...[/bold cyan]"):
        completion = await client.complete(
            system,
            f"User Request: {prompt}\nWorkspace Directory: {workspace}",
            max_tokens=3000
        )

    console.print(Panel(Markdown(completion.text), title="⚡ Generated Code", border_style="cyan"))

    # Offer to save the generated code directly to the workspace
    try:
        save = await questionary.confirm("Would you like to save this script directly to your workspace?", default=True).ask_async()
        if save:
            # Look for a suggested filename or default
            filename_match = re.search(r"#\s*(?:filename|file):\s*([a-zA-Z0-9_\-\.]+)", completion.text, re.IGNORECASE)
            default_name = filename_match.group(1) if filename_match else "script.py"
            
            filename = await questionary.text("Enter filename to save as:", default=default_name).ask_async()
            if filename and filename.strip():
                # Extract code content
                code = completion.text
                code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", completion.text, re.DOTALL)
                if code_blocks:
                    code = code_blocks[0]
                
                target_file = workspace / filename.strip()
                target_file.write_text(code, encoding="utf-8")
                console.print(f"[bold green]✅ Successfully saved to:[/bold green] [cyan]{target_file}[/cyan]\n")
    except Exception as exc:
        console.print(f"[dim yellow]Could not save file: {exc}[/dim yellow]")


async def handle_learn_or_chat(prompt: str, client: llm_backends.ModelClient | None, workspace: Path) -> None:
    """Generates an in-depth, structured tutorial or conceptual response."""
    if client is None:
        console.print(Panel(
            "[yellow]Simulation Mode:[/yellow] To receive AI-generated guided learning and tutorials, please select an active LLM backend (Ollama, LM Studio, or llama.cpp).",
            title="Guided Learning",
            border_style="yellow"
        ))
        return

    system = (
        "You are an expert AI architect and Senior Software Engineer specializing in agentic systems, autonomous loops, and software engineering.\n"
        "Provide a comprehensive, pedagogical, well-structured guide in Markdown format responding to the user's learning request.\n"
        "Include:\n"
        "- Core Concepts & Architectural Overview (use ASCII/Markdown diagrams where helpful)\n"
        "- Step-by-Step Implementation Guide\n"
        "- Production-Ready Code Examples with clear explanations\n"
        "- Common Pitfalls & How to Avoid Them\n"
        "- Best Practices for Stability, Budgets, and Guardrails"
    )

    with console.status("[bold cyan]Designing comprehensive guided learning tutorial...[/bold cyan]"):
        completion = await client.complete(
            system,
            f"User Goal: {prompt}\nWorkspace: {workspace}",
            max_tokens=3500
        )

    console.print(Panel(Markdown(completion.text), title="🎓 Guided Learning & Architectural Tutorial", border_style="cyan"))


async def handle_project_review(prompt: str, client: llm_backends.ModelClient | None, workspace: Path, shell: agent_runtime.SubprocessShell) -> None:
    """Inspects workspace files and synthesizes a thorough project review and enhancement report."""
    with console.status("[bold green]Inspecting workspace repository structure and configurations...[/bold green]"):
        res_ls = await shell.execute(agent_runtime.ToolCall(tool="shell", args={"command": "ls -R"}), 10)
        files_listing = res_ls.output[:3000]

        manifests = []
        for candidate in ["package.json", "requirements.txt", "pyproject.toml", "README.md", "agent_loop_architecture_v2.md", "setup.py"]:
            p = workspace / candidate
            if p.exists() and p.is_file():
                try:
                    manifests.append(f"--- {candidate} ---\n{p.read_text(encoding='utf-8', errors='replace')[:2000]}")
                except Exception:
                    pass
        manifest_text = "\n\n".join(manifests) if manifests else "No manifest files found in root."

    if client is None:
        console.print(Panel(
            f"[bold]Repository Files Detected:[/bold]\n{files_listing}\n\n[yellow]Connect an active LLM to generate deep AI code analysis and enhancements.[/yellow]",
            title="Repository Inspection",
            border_style="green"
        ))
        return

    system = (
        "You are a Principal Software Architect conducting an in-depth repository code review.\n"
        "Analyze the repository structure and configuration provided below.\n"
        "Provide a comprehensive, professional Project Review Report formatted in Markdown:\n\n"
        "## 1. Executive Summary & Architecture Overview\n"
        "## 2. Strengths & Current Capabilities\n"
        "## 3. Structural & Architectural Gaps\n"
        "## 4. Prioritized Actionable Enhancements (detailed explanations with sample code)\n"
        "## 5. Implementation Roadmap"
    )

    user_context = (
        f"Review Goal: {prompt}\n\n"
        f"Repository File Tree:\n{files_listing}\n\n"
        f"Key Configuration & Source Files:\n{manifest_text}"
    )

    with console.status("[bold green]Generating deep code review and architectural suggestions...[/bold green]"):
        completion = await client.complete(system, user_context, max_tokens=3500)

    console.print(Panel(Markdown(completion.text), title="📊 Comprehensive Project Review & Enhancement Report", border_style="green"))


async def select_model() -> tuple[llm_backends.ModelClient | None, llm_backends.ModelRegistry | None]:
    console.print(Panel("[bold cyan]Omni CLI[/bold cyan] - Autonomous Developer Agent", subtitle="Powered by Agent Loop Architecture v2"))
    
    candidates = [
        llm_backends.OllamaBackend(model=""),
        llm_backends.LMStudioBackend(model=""),
        llm_backends.LlamaCppBackend(model=""),
    ]
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        progress.add_task(description="Probing local inference backends (Ollama, LM Studio, llama.cpp)...", total=None)
        infos = await asyncio.gather(*(c.probe() for c in candidates))
    
    table = Table(title="Inference Backends Status")
    table.add_column("Backend", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Available Models", style="magenta")
    table.add_column("Details", style="dim")

    option_map: dict[str, tuple[str, str]] = {}
    
    for info in infos:
        models_str = ", ".join(info.models_available[:5]) if info.models_available else ("Online" if info.reachable else "-")
        table.add_row(
            info.backend.value,
            "[green]Online[/green]" if info.reachable else "[red]Offline[/red]",
            models_str,
            info.detail[:40] if info.detail else ""
        )
        if info.reachable:
            if info.models_available:
                for m in info.models_available:
                    label = f"[{info.backend.value}] {m}"
                    option_map[label] = (info.backend.value, m)
            else:
                label = f"[{info.backend.value}] Default model"
                option_map[label] = (info.backend.value, info.model or "default")

    console.print(table)
    
    choices = list(option_map.keys())
    choices.append("[Custom] Enter backend URL and model name manually")
    choices.append("[Simulation] Run in local simulation mode (No LLM)")
    
    selected_label = await questionary.select(
        "Select inference model for this session:",
        choices=choices
    ).ask_async()
    
    if selected_label == "[Simulation] Run in local simulation mode (No LLM)":
        return None, None
        
    if selected_label == "[Custom] Enter backend URL and model name manually":
        b_type = await questionary.select("Select backend type:", choices=["ollama", "lmstudio", "llamacpp"]).ask_async()
        url = await questionary.text("Enter base URL:", default="http://localhost:11434" if b_type == "ollama" else "http://localhost:1234").ask_async()
        m_name = await questionary.text("Enter model name:").ask_async()
        if b_type == "ollama":
            client = llm_backends.OllamaBackend(base_url=url, model=m_name, num_ctx=16384)
        elif b_type == "lmstudio":
            client = llm_backends.LMStudioBackend(base_url=url, model=m_name)
        else:
            client = llm_backends.LlamaCppBackend(base_url=url, model=m_name)
    else:
        backend_name, model_name = option_map[selected_label]
        if backend_name == "ollama":
            client = llm_backends.OllamaBackend(model=model_name, num_ctx=16384)
        elif backend_name == "lmstudio":
            client = llm_backends.LMStudioBackend(model=model_name)
        else:
            client = llm_backends.LlamaCppBackend(model=model_name)
            
    registry = llm_backends.ModelRegistry([
        llm_backends.RoleBinding(llm_backends.Role.ROUTER, [client]),
        llm_backends.RoleBinding(llm_backends.Role.SUMMARIZER, [client]),
        llm_backends.RoleBinding(llm_backends.Role.REPAIR, [client]),
        llm_backends.RoleBinding(llm_backends.Role.PLANNER, [client]),
    ], default=client)
    
    return client, registry


async def select_workspace() -> Path:
    path = await questionary.path(
        "Enter workspace directory:",
        default="./"
    ).ask_async()
    workspace = Path(path).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def format_step_table(steps: Sequence[agent_runtime.StepRecord]) -> Table:
    table = Table(title="Execution History & Observations", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Thought / Step", style="cyan")
    table.add_column("Action", style="yellow")
    table.add_column("Status", style="bold")
    table.add_column("Output Preview", style="dim")

    for s in steps:
        cmd = s.call.args.get("command", "") if s.call else ""
        status = "[green]OK (0)[/green]" if s.result.ok else f"[red]ERR ({s.result.exit_code})[/red]"
        out_line = s.result.output.strip().splitlines()[0][:60] if s.result.output else ""
        table.add_row(str(s.idx), s.thought[:40], str(cmd)[:35], status, out_line)
    return table


async def run_cli() -> None:
    client, registry = await select_model()
    workspace = await select_workspace()
    
    shell = agent_runtime.SubprocessShell(workspace=workspace)
    
    smart_router = SmartRouter(client=client)
    smart_planner = SmartPlanner(client=client)
    
    if client:
        repair_factory = lambda cmd: WrappedLLMPolicy(client)
    else:
        repair_factory = lambda cmd: agent_runtime.HeuristicRepairPolicy(cmd)
    
    console.print(f"\n[green]✅ Workspace active:[/green] [bold]{workspace}[/bold]")
    console.print(Panel(
        "✨ [bold]Omni CLI Ready[/bold] ✨\n"
        "• [cyan]Guided Learning[/cyan]: Ask 'How to build an agentic loop', 'Explain FSM', etc.\n"
        "• [blue]Direct Code Gen[/blue]: Ask 'Write python script to add 2 numbers', 'Create helper func'\n"
        "• [green]Project Review[/green]: Ask 'Review project and suggest enhancements'\n"
        "• [yellow]Action Tasks[/yellow]: Ask 'Create a FastAPI app', 'Fix tests', 'Scaffold project'\n"
        "Type [bold red]exit[/bold red] to quit.",
        border_style="cyan"
    ))
    
    active_resume: agent_runtime.ResumePayload | None = None
    
    while True:
        try:
            prompt_label = "Omni (Resume)> " if active_resume else "Omni> "
            user_input = await questionary.text(prompt_label).ask_async()
            if not user_input or user_input.lower().strip() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
                
            console.print(f"\n[bold blue]🚀 Goal:[/bold blue] {user_input}")

            # If resuming an existing suspended task, go straight to orchestrator
            if active_resume is not None:
                orchestrator = agent_runtime.Orchestrator(
                    executor=shell,
                    workspace=workspace,
                    router=smart_router,
                    planner=smart_planner.plan,
                    repair_policy_factory=repair_factory
                )
                with console.status("[bold green]Resuming agent loop...[/bold green]"):
                    outcome = await orchestrator.handle(user_input, resume=active_resume)
            else:
                # Classify intent
                intent = await classify_intent(user_input, client)
                
                if intent == "LEARN_OR_CHAT":
                    await handle_learn_or_chat(user_input, client, workspace)
                    continue
                elif intent == "DIRECT_CODE":
                    await handle_direct_code(user_input, client, workspace)
                    continue
                elif intent == "PROJECT_REVIEW":
                    await handle_project_review(user_input, client, workspace, shell)
                    continue
                else:
                    # EXECUTE_TASK
                    orchestrator = agent_runtime.Orchestrator(
                        executor=shell,
                        workspace=workspace,
                        router=smart_router,
                        planner=smart_planner.plan,
                        repair_policy_factory=repair_factory
                    )
                    with console.status("[bold green]Executing agent plan through Sense-Think-Act loop...[/bold green]"):
                        outcome = await orchestrator.handle(user_input, resume=active_resume)
            
            if outcome.steps:
                console.print(format_step_table(outcome.steps))

            if outcome.state == agent_runtime.RunState.SUCCEEDED:
                console.print(Panel(
                    f"[bold green]Task Succeeded:[/bold green] {outcome.detail}", 
                    title="Execution Complete", 
                    border_style="green"
                ))
                active_resume = None
            elif outcome.state in (agent_runtime.RunState.SUSPENDED_APPROVAL, agent_runtime.RunState.SUSPENDED_HITL):
                console.print(Panel(
                    f"[bold yellow]State:[/bold yellow] {outcome.state.value}\n"
                    f"[bold yellow]Reason:[/bold yellow] {outcome.detail}", 
                    title="Operator Intervention Required", 
                    border_style="yellow"
                ))
                
                if outcome.payload:
                    pending_cmd = outcome.payload.workflow_metadata.get("pending_command")
                    if outcome.state == agent_runtime.RunState.SUSPENDED_APPROVAL and pending_cmd:
                        approve = await questionary.confirm(
                            f"Agent requested permission to execute: '{pending_cmd}'. Approve?"
                        ).ask_async()
                        if approve:
                            approvals_list = ["sudo"] if "sudo" in pending_cmd else [pending_cmd]
                            active_resume = outcome.payload.with_intervention(
                                guidance={"operator_approval": "granted"},
                                approvals=approvals_list
                            )
                            console.print("[green]Permission granted. Continuing run...[/green]")
                            continue
                        else:
                            console.print("[red]Command rejected by operator.[/red]")
                            active_resume = None
                    else:
                        active_resume = outcome.payload
            else:
                console.print(Panel(
                    f"[bold red]Run Ended:[/bold red] {outcome.state.value} - {outcome.detail}", 
                    title="Run Failed", 
                    border_style="red"
                ))
                active_resume = None
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error occurred:[/red] {e}")

if __name__ == "__main__":
    agent_runtime.set_quiet(True)
    asyncio.run(run_cli())

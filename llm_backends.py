"""
llm_backends.py — local inference backends for the agent runtime
================================================================

Drop-in `ModelClient` implementations for **Ollama**, **LM Studio** and
**llama.cpp** (`llama-server`), plus role-based tiering with health-gated
fallback. Plugs straight into `agent_runtime.LLMPolicy`.

Design notes worth reading before changing anything here
--------------------------------------------------------

**1. Constrained decoding is the whole ballgame.**
`LLMPolicy` needs schema-valid JSON on every turn. "Reply with JSON only" is a
request; a grammar is a constraint. All three backends can constrain the
sampler, but through three different parameters:

    Ollama      POST /api/chat            {"format": <json-schema>}
    llama.cpp   POST /v1/chat/completions {"response_format": {...json_schema}}
                POST /completion          {"json_schema": ...} or {"grammar": <GBNF>}
    LM Studio   POST /v1/chat/completions {"response_format": {...json_schema}}

Each backend below advertises `supports_schema` and injects the right one.
A backend that cannot constrain falls back to prompt-and-validate, and the
policy's existing repair round-trip catches the difference.

**2. Ollama truncates the *head* of the prompt, silently.**
This is the single nastiest local-inference failure mode for an agent. Past
`num_ctx`, the oldest tokens are dropped — and in our prompt layout the oldest
tokens are the system prompt carrying the JSON contract. The symptom is not
"context overflow"; it is "the model mysteriously stopped returning JSON on
long runs". `_TruncationGuard` catches it two ways: refuse to send a prompt
that cannot fit, and compare the server's reported `prompt_eval_count` against
a local estimate afterwards.

**3. Local is not free, it is unpriced.**
Marginal USD is ~0, but tokens, latency and VRAM are finite. Backends report
real token counts so the `Ledger` guardrails keep binding, and `usd_per_1k`
lets you model amortised hardware+power cost if you want cost caps to mean
something on-prem.

**4. Prefix stability beats every other latency optimisation.**
All three servers reuse cached KV for a matching prompt prefix. Keeping the
system prompt and goal byte-identical across turns turns an O(context) prefill
into O(new tokens). Mutating anything near the front — injecting a warning at
the top, re-ordering history — silently discards the cache. Append, never
prepend. See `local_inference.md` §4.

Quick check against your own machine:

    python3 llm_backends.py            # probes all three, prints capabilities
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import httpx

log = logging.getLogger("agent.llm")

# =============================================================================
# 1. Contract
# =============================================================================


class Backend(str, Enum):
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    LLAMACPP = "llamacpp"


@dataclass(frozen=True)
class Completion:
    """What every backend returns, regardless of wire format."""
    text: str
    model: str
    backend: Backend
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    usd: float = 0.0
    latency_ms: int = 0
    finish_reason: str = ""
    schema_enforced: bool = False
    truncation_suspected: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def decode_tps(self) -> float:
        return (self.completion_tokens / (self.latency_ms / 1000)) if self.latency_ms else 0.0


@dataclass(frozen=True)
class BackendInfo:
    backend: Backend
    reachable: bool
    model: str = ""
    context_length: int | None = None
    supports_schema: bool = False
    detail: str = ""
    models_available: tuple[str, ...] = ()


class ModelClient(Protocol):
    async def complete(self, system: str, user: str, *,
                       schema: dict[str, Any] | None = None,
                       max_tokens: int = 1024) -> Completion: ...

    async def probe(self) -> BackendInfo: ...


# -- errors -------------------------------------------------------------------


class BackendError(RuntimeError):
    """Base. Distinguishing retryable from terminal is the point of the tree."""


class BackendUnavailable(BackendError):
    """Connection refused, timeout, 5xx — retryable, and a fallback candidate."""


class BackendRefused(BackendError):
    """4xx: malformed request or unsupported schema. Retrying will not help."""


class ContextOverflow(BackendError):
    """The prompt cannot fit. Raised *before* spending a prefill."""


# =============================================================================
# 2. Token estimation and the truncation guard
# =============================================================================


def estimate_tokens(text: str) -> int:
    """
    Deliberately crude (~4 chars/token) and deliberately *not* a real tokenizer.
    A tokenizer would mean shipping model-specific vocab and would still be
    wrong for the next model you load. This is only used for a safety margin,
    and it is checked against the server's authoritative count afterwards.
    """
    return max(1, len(text) // 4)


@dataclass
class TruncationGuard:
    """
    Two-sided defence against silent head-truncation.

    Pre-flight: refuse to send a prompt that plainly cannot fit, so the failure
    is a clean exception instead of a mangled generation.

    Post-flight: if the server evaluated materially fewer prompt tokens than we
    sent, the front of the prompt was dropped. Because the front holds the
    system contract, this must be loud.
    """
    context_length: int | None = None
    reserve_for_output: int = 1024
    tolerance: float = 0.85

    def check_fits(self, prompt_chars: int) -> int:
        estimated = estimate_tokens_from_chars(prompt_chars)
        if self.context_length is None:
            return estimated
        budget = self.context_length - self.reserve_for_output
        if estimated > budget:
            raise ContextOverflow(
                f"prompt ~{estimated} tokens exceeds usable context "
                f"{budget} (num_ctx={self.context_length} minus {self.reserve_for_output} "
                f"reserved for output). Compress history or raise num_ctx.")
        return estimated

    def check_evaluated(self, estimated: int, reported_prompt_tokens: int) -> bool:
        if reported_prompt_tokens <= 0:
            return False  # backend did not report; cannot conclude anything
        if reported_prompt_tokens < estimated * self.tolerance:
            log.error("prompt truncation suspected: sent ~%d tokens, server evaluated %d. "
                      "The system prompt is at the FRONT and is what gets dropped.",
                      estimated, reported_prompt_tokens)
            return True
        return False


def estimate_tokens_from_chars(chars: int) -> int:
    return max(1, chars // 4)


# =============================================================================
# 3. Shared HTTP behaviour: concurrency, retries, circuit breaking
# =============================================================================


@dataclass
class RetryPolicy:
    attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0

    def delay(self, attempt: int) -> float:
        # Full jitter. Local servers queue rather than shed, so synchronised
        # retries from parallel agents turn a blip into a stampede.
        ceiling = min(self.max_delay_s, self.base_delay_s * (2 ** attempt))
        return random.uniform(0, ceiling)


@dataclass
class CircuitBreaker:
    """
    A local server that is down (model unloading, OOM, user quit LM Studio) stays
    down for seconds-to-minutes. Hammering it delays failover; the breaker makes
    the registry skip it immediately.
    """
    failure_threshold: int = 3
    cooldown_s: float = 30.0
    _failures: int = 0
    _opened_at: float = 0.0

    @property
    def is_open(self) -> bool:
        if self._failures < self.failure_threshold:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_s:
            self._failures = 0  # half-open: allow one probe through
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = time.monotonic()


class HTTPBackend:
    """Shared transport: one client, bounded concurrency, typed error mapping."""

    backend: Backend = Backend.OLLAMA  # overridden

    def __init__(self, *, base_url: str, model: str, timeout_s: float = 180.0,
                 max_concurrency: int = 1, retry: RetryPolicy | None = None,
                 usd_per_1k_prompt: float = 0.0, usd_per_1k_completion: float = 0.0,
                 context_length: int | None = None, reserve_for_output: int = 1024,
                 transport: httpx.AsyncBaseTransport | None = None,
                 temperature: float = 0.0, seed: int | None = 42) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        self.usd_per_1k_prompt = usd_per_1k_prompt
        self.usd_per_1k_completion = usd_per_1k_completion
        self.temperature = temperature
        self.seed = seed
        self.guard = TruncationGuard(context_length=context_length,
                                     reserve_for_output=reserve_for_output)
        self.breaker = CircuitBreaker()
        # Local servers have a fixed number of slots. Exceeding them does not
        # increase throughput; it increases queueing and time-to-first-token.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(base_url=self.base_url,
                                         timeout=httpx.Timeout(timeout_s, connect=5.0),
                                         transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(self.retry.attempts):
            try:
                async with self._semaphore:
                    response = await self._client.post(path, json=body)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout,
                    httpx.RemoteProtocolError) as exc:
                last = BackendUnavailable(f"{self.backend.value}: {type(exc).__name__}: {exc}")
            else:
                if response.status_code < 400:
                    self.breaker.record_success()
                    return response.json()
                if response.status_code < 500:
                    # 4xx is our bug (bad schema, unknown model). Retrying it
                    # just burns the run's wall-clock budget.
                    self.breaker.record_failure()
                    raise BackendRefused(
                        f"{self.backend.value} {response.status_code}: {response.text[:400]}")
                last = BackendUnavailable(f"{self.backend.value} {response.status_code}")

            self.breaker.record_failure()
            if attempt < self.retry.attempts - 1:
                await asyncio.sleep(self.retry.delay(attempt))
        raise last or BackendUnavailable(f"{self.backend.value}: exhausted retries")

    async def _get(self, path: str) -> Any:
        try:
            response = await self._client.get(path)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # probing must never raise into the agent loop
            raise BackendUnavailable(f"{self.backend.value}: {exc}") from exc

    def _price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1000 * self.usd_per_1k_prompt
                + completion_tokens / 1000 * self.usd_per_1k_completion)


# =============================================================================
# 4. Ollama — native /api/chat
# =============================================================================


class OllamaBackend(HTTPBackend):
    """
    Uses the native endpoint rather than Ollama's OpenAI-compat shim, because
    the native one exposes what actually matters here: `num_ctx` (see below),
    `keep_alive`, and honest `prompt_eval_count` / `eval_count` telemetry.

    `num_ctx` is not optional. Ollama's default context is small relative to an
    agent's working set, and overflow drops the *front* of the prompt with no
    error — which is exactly where the JSON contract lives. Set it explicitly
    per request, size it against VRAM, and let the guard verify.
    """

    backend = Backend.OLLAMA

    def __init__(self, *, base_url: str = "http://localhost:11434",
                 model: str = "qwen3:8b", num_ctx: int = 16384,
                 keep_alive: str = "30m", **kwargs: Any) -> None:
        kwargs.setdefault("context_length", num_ctx)
        super().__init__(base_url=base_url, model=model, **kwargs)
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    async def complete(self, system: str, user: str, *,
                       schema: dict[str, Any] | None = None,
                       max_tokens: int = 1024) -> Completion:
        estimated = self.guard.check_fits(len(system) + len(user))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                **({"seed": self.seed} if self.seed is not None else {}),
            },
        }
        if schema is not None:
            body["format"] = schema  # constrained decoding, not a suggestion

        started = time.monotonic()
        data = await self._post("/api/chat", body)
        latency_ms = int((time.monotonic() - started) * 1000)

        prompt_tokens = int(data.get("prompt_eval_count", 0))
        completion_tokens = int(data.get("eval_count", 0))
        return Completion(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            backend=self.backend,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            usd=self._price(prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            finish_reason=str(data.get("done_reason", "")),
            schema_enforced=schema is not None,
            truncation_suspected=self.guard.check_evaluated(estimated, prompt_tokens),
        )

    async def probe(self) -> BackendInfo:
        try:
            tags = await self._get("/api/tags")
            names = tuple(m["name"] for m in tags.get("models", []))
            context_length = None
            try:
                show = await self._post("/api/show", {"model": self.model})
                info = show.get("model_info", {}) or {}
                context_length = next((int(v) for k, v in info.items()
                                       if k.endswith(".context_length")), None)
            except BackendError:
                pass
            return BackendInfo(self.backend, True, self.model,
                               context_length=context_length, supports_schema=True,
                               detail=f"num_ctx set to {self.num_ctx}"
                                      + (f"; model max {context_length}" if context_length else ""),
                               models_available=names)
        except BackendError as exc:
            return BackendInfo(self.backend, False, self.model, detail=str(exc))


# =============================================================================
# 5. OpenAI-compatible backends — LM Studio and llama.cpp
# =============================================================================


class OpenAICompatBackend(HTTPBackend):
    """
    LM Studio and llama-server both speak /v1/chat/completions. The differences
    are in structured-output dialect and telemetry, so those are the hooks.
    """

    supports_schema = True

    def _schema_body(self, schema: dict[str, Any]) -> dict[str, Any]:
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "agent_decision",
                                                    "strict": True,
                                                    "schema": schema}}}

    def _extra_body(self) -> dict[str, Any]:
        return {}

    def _telemetry(self, data: dict[str, Any]) -> tuple[int, int, int]:
        usage = data.get("usage") or {}
        return (int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)), 0)

    async def complete(self, system: str, user: str, *,
                       schema: dict[str, Any] | None = None,
                       max_tokens: int = 1024) -> Completion:
        estimated = self.guard.check_fits(len(system) + len(user))
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **self._extra_body(),
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if schema is not None:
            body.update(self._schema_body(schema))

        started = time.monotonic()
        data = await self._post("/v1/chat/completions", body)
        latency_ms = int((time.monotonic() - started) * 1000)

        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        prompt_tokens, completion_tokens, cached = self._telemetry(data)
        return Completion(
            text=message.get("content") or "",
            model=data.get("model", self.model),
            backend=self.backend,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached,
            usd=self._price(prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            finish_reason=str(choices[0].get("finish_reason", "")),
            schema_enforced=schema is not None,
            truncation_suspected=self.guard.check_evaluated(estimated, prompt_tokens),
        )


class LMStudioBackend(OpenAICompatBackend):
    """
    LM Studio supports `json_schema` structured output but not `json_object`
    JSON mode, so never degrade to `{"type": "json_object"}` here — it 400s.
    Context length comes from whatever was configured when the model was loaded
    in the GUI, which the OpenAI route does not expose; the richer /api/v0
    route does, so probe tries that first.
    """

    backend = Backend.LMSTUDIO

    def __init__(self, *, base_url: str = "http://localhost:1234",
                 model: str = "qwen3-8b", **kwargs: Any) -> None:
        super().__init__(base_url=base_url, model=model, **kwargs)

    async def probe(self) -> BackendInfo:
        try:
            context_length = None
            names: tuple[str, ...] = ()
            try:
                rich = await self._get("/api/v0/models")
                entries = rich.get("data", rich if isinstance(rich, list) else [])
                names = tuple(e.get("id", "") for e in entries)
                for entry in entries:
                    if entry.get("id") == self.model:
                        context_length = entry.get("loaded_context_length") or entry.get("max_context_length")
                        break
            except BackendError:
                pass
            if not names:
                models = await self._get("/v1/models")
                names = tuple(m["id"] for m in models.get("data", []))
            return BackendInfo(self.backend, True, self.model, context_length=context_length,
                               supports_schema=True,
                               detail="json_schema only (json_object unsupported)",
                               models_available=names)
        except BackendError as exc:
            return BackendInfo(self.backend, False, self.model, detail=str(exc))


class LlamaCppBackend(OpenAICompatBackend):
    """
    llama-server. Two things it does that the others do not:

      * `cache_prompt` reuses the KV cache for a matching prefix. With a stable
        system prompt this is the difference between re-prefilling the whole
        agent history each turn and prefilling only the new observation.
      * `timings.prompt_n` / `predicted_n` report actual work, and
        `tokens_cached` tells you whether the prefix cache is being hit — the
        single most useful number when tuning agent latency.

    Structured output: `response_format.json_schema` on the chat route. Older
    builds rejected it when a grammar was also present; if you hit
    "Either json_schema or grammar can be specified, but not both", make sure
    no server-level `--grammar` is set. `grammar_fallback=True` switches to the
    native /completion route with `json_schema`, which has always worked.
    """

    backend = Backend.LLAMACPP

    def __init__(self, *, base_url: str = "http://localhost:8080",
                 model: str = "local-model", cache_prompt: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(base_url=base_url, model=model, **kwargs)
        self.cache_prompt = cache_prompt

    def _extra_body(self) -> dict[str, Any]:
        return {"cache_prompt": self.cache_prompt}

    def _telemetry(self, data: dict[str, Any]) -> tuple[int, int, int]:
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        cached = int(data.get("tokens_cached") or timings.get("cache_n") or 0)
        return prompt_tokens, completion_tokens, cached

    async def probe(self) -> BackendInfo:
        try:
            props = await self._get("/props")
            settings = props.get("default_generation_settings") or {}
            context_length = settings.get("n_ctx") or props.get("n_ctx")
            model_path = props.get("model_path") or settings.get("model") or self.model
            if context_length and self.guard.context_length is None:
                self.guard.context_length = int(context_length)
            return BackendInfo(self.backend, True, str(model_path).split("/")[-1],
                               context_length=int(context_length) if context_length else None,
                               supports_schema=True,
                               detail=f"cache_prompt={self.cache_prompt}",
                               models_available=(str(model_path).split("/")[-1],))
        except BackendError as exc:
            return BackendInfo(self.backend, False, self.model, detail=str(exc))


# =============================================================================
# 6. Decision schema for the agent policy
# =============================================================================

# Flat rather than a discriminated `oneOf`. Constrained-decoding engines vary in
# how well they handle oneOf/anyOf, and a small local model steered through a
# branchy grammar degrades noticeably. One flat object with an enum discriminator
# constrains reliably everywhere; the *semantic* requirement ("act implies
# command") is then enforced in Python, where a violation is a clean retry rather
# than a decoding dead-end.
FLAT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["act", "finish", "ask_human"]},
        "thought": {"type": "string"},
        "command": {"type": "string"},
        "succeeded": {"type": "boolean"},
        "summary": {"type": "string"},
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["kind"],
    "additionalProperties": False,
}

# Stricter variant for backends/models that handle oneOf well. Verify with your
# model before switching: a grammar the engine mishandles fails as gibberish,
# not as an error.
STRICT_DECISION_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {"type": "object", "additionalProperties": False,
         "properties": {"kind": {"const": "act"},
                        "thought": {"type": "string"},
                        "command": {"type": "string"}},
         "required": ["kind", "thought", "command"]},
        {"type": "object", "additionalProperties": False,
         "properties": {"kind": {"const": "finish"},
                        "succeeded": {"type": "boolean"},
                        "summary": {"type": "string"}},
         "required": ["kind", "succeeded", "summary"]},
        {"type": "object", "additionalProperties": False,
         "properties": {"kind": {"const": "ask_human"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}}},
         "required": ["kind", "question"]},
    ]
}


# =============================================================================
# 7. Role-based tiering with health-gated fallback
# =============================================================================


class Role(str, Enum):
    ROUTER = "router"          # cheap classification
    SUMMARIZER = "summarizer"  # cheap, output is schema-validated anyway
    REPAIR = "repair"          # needs real diagnosis
    PLANNER = "planner"        # plan quality dominates total cost


@dataclass
class RoleBinding:
    """Ordered preference. First healthy backend wins."""
    role: Role
    clients: list[ModelClient]


class ModelRegistry:
    """
    Maps agent roles to backends with fallback, so a 30B repair model and a 4B
    router can live on different servers — or the same one — without the loop
    knowing. Fallback order is a policy decision, not a runtime accident:
    put the local backend first if you care about privacy, cloud first if you
    care about latency under local contention.
    """

    def __init__(self, bindings: Sequence[RoleBinding],
                 default: ModelClient | None = None) -> None:
        self._bindings = {b.role: b.clients for b in bindings}
        self._default = default

    def for_role(self, role: Role) -> "RoleClient":
        clients = self._bindings.get(role) or ([self._default] if self._default else [])
        if not clients:
            raise KeyError(f"no backend bound to role {role.value}")
        return RoleClient(role, list(clients))

    async def probe_all(self) -> list[BackendInfo]:
        seen: dict[int, ModelClient] = {}
        for clients in self._bindings.values():
            for client in clients:
                seen[id(client)] = client
        return list(await asyncio.gather(*(c.probe() for c in seen.values())))

    async def aclose(self) -> None:
        for clients in self._bindings.values():
            for client in clients:
                closer = getattr(client, "aclose", None)
                if closer:
                    await closer()


class RoleClient:
    """A `ModelClient` that transparently fails over within a role."""

    def __init__(self, role: Role, clients: list[ModelClient]) -> None:
        self.role = role
        self.clients = clients

    async def complete(self, system: str, user: str, *,
                       schema: dict[str, Any] | None = None,
                       max_tokens: int = 1024) -> Completion:
        errors: list[str] = []
        for client in self.clients:
            breaker = getattr(client, "breaker", None)
            if breaker is not None and breaker.is_open:
                errors.append(f"{getattr(client, 'backend', '?')}: circuit open")
                continue
            try:
                return await client.complete(system, user, schema=schema, max_tokens=max_tokens)
            except BackendUnavailable as exc:
                errors.append(str(exc))
                log.warning("role=%s backend unavailable, failing over: %s", self.role.value, exc)
            except ContextOverflow:
                # Failing over will not make the prompt shorter.
                raise
        raise BackendUnavailable(f"role={self.role.value}: all backends failed: {errors}")

    async def probe(self) -> BackendInfo:
        return await self.clients[0].probe()


def default_local_registry(*, small_model: str = "qwen3:4b",
                           large_model: str = "qwen3-coder:30b",
                           **kwargs: Any) -> ModelRegistry:
    """
    A sane starting topology for a single-workstation setup: small local model
    for classification-shaped roles, large local model for reasoning-shaped
    ones, and each covering for the other. Adjust `num_ctx` per model — the
    small tier rarely needs more than 8k, while the repair tier carries the
    agent's whole working set.
    """
    small = OllamaBackend(model=small_model, num_ctx=8192, max_concurrency=2, **kwargs)
    large = OllamaBackend(model=large_model, num_ctx=32768, max_concurrency=1, **kwargs)
    return ModelRegistry([
        RoleBinding(Role.ROUTER, [small, large]),
        RoleBinding(Role.SUMMARIZER, [small, large]),
        RoleBinding(Role.REPAIR, [large, small]),
        RoleBinding(Role.PLANNER, [large, small]),
    ], default=large)


# =============================================================================
# 8. Probe CLI
# =============================================================================


async def _probe_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    candidates: list[HTTPBackend] = [
        OllamaBackend(),
        LMStudioBackend(),
        LlamaCppBackend(),
    ]
    infos = await asyncio.gather(*(c.probe() for c in candidates))
    print(f"{'backend':<10} {'up':<4} {'model':<34} {'ctx':<9} detail")
    print("-" * 100)
    for info in infos:
        ctx = str(info.context_length) if info.context_length else "-"
        print(f"{info.backend.value:<10} {'yes' if info.reachable else 'no':<4} "
              f"{(info.model or '-')[:33]:<34} {ctx:<9} {info.detail[:44]}")
        if info.reachable and info.models_available:
            print(f"{'':<10} available: {', '.join(info.models_available[:6])}")
    for client in candidates:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(_probe_cli())

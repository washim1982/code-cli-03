"""
Session-aware context memory.

Implements the specified `ContextStore` flow:

  init            -> ensure `context/` and `context/index.json` exist
  add_context     -> load all, append {id, session_id, intent, text, embedding,
                     timestamp}, save all
  embed_text      -> POST to the embedding server; never raise
  _adaptive_n     -> window size per intent (coder 2 / chat 3 / architect 5 ...)
  get_recent      -> same-session newest-first, top N, globally back-filled if
                     short, with embedding auto-repair
  retrieve_best   -> cosine(query, entry) over the recent window, best text wins

Four corrections to the original implementation, none of which change the flow:

  1. **Atomic saves.** The original truncated `index.json` with `open(path,"w")`
     before writing. A crash mid-dump destroyed the whole store, and the loader
     then swallowed the JSONDecodeError and returned `[]`, so the next add
     overwrote the corpse. Writes now go to a temp file and `os.replace`, and a
     corrupt index is moved aside as `.corrupt-<ts>` rather than silently lost.

  2. **Failed embeddings no longer win retrieval.** The spec's zero-vector
     fallback is kept — callers still never see a missing embedding — but the
     entry is tagged `embedding_ok: False`, and `retrieve_best_context` skips
     those when ranking. Previously every cosine was 0.0 while `best_score`
     started at -1.0, so the *first* candidate was returned as the
     "most similar" match and the store reported success while retrieving at
     random.

  3. **Dimension mismatch is not silently truncated.** `zip()` stops at the
     shorter vector, so swapping embedding models produced plausible-looking
     nonsense. Mismatched lengths now score 0.0.

  4. **Timeouts.** The embedding call had none, so an unresponsive server hung
     the CLI indefinitely.

Async because the host application is async; blocking HTTP inside the event loop
would stall the UI. The control flow is otherwise identical to the spec.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "MemoryConfig",
    "ContextStore",
    "RetrievedContext",
    "make_session_id",
    "cosine_similarity",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Context window size per intent. The first six are the original intent
#: vocabulary; the four uppercase keys map Omni CLI's own intents onto it.
INTENT_WINDOW: dict[str, int] = {
    "coder": 2,
    "chat": 3,
    "script": 3,
    "summarizer": 3,
    "architect": 5,
    "planner": 5,
    # Omni CLI intents
    "DIRECT_CODE": 2,
    "LEARN_OR_CHAT": 3,
    "PROJECT_REVIEW": 5,
    "EXECUTE_TASK": 5,
}

DEFAULT_WINDOW = 3


@dataclass
class MemoryConfig:
    root: Path
    dir_name: str = "context"
    index_name: str = "index.json"
    embedding_url: str = "http://localhost:11434/api/embeddings"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768
    timeout_s: float = 15.0
    #: Cap on stored entries; oldest are dropped past this. None = unbounded
    #: (the original behaviour). The index holds full embeddings inline, so it
    #: grows by roughly 10-15 KB per entry.
    max_entries: int | None = None

    @property
    def dir(self) -> Path:
        return Path(self.root) / self.dir_name

    @property
    def index(self) -> Path:
        return self.dir / self.index_name


@dataclass
class RetrievedContext:
    """Richer return value for callers that want the score, not just the text."""
    text: str = ""
    score: float = 0.0
    entry_id: str = ""
    intent: str = ""
    session_id: str = ""
    from_other_session: bool = False
    considered: int = 0
    rankable: int = 0

    def __bool__(self) -> bool:
        return bool(self.text)


def make_session_id() -> str:
    return "sess-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity, or 0.0 for empty / mismatched / zero vectors.

    Length mismatch returns 0.0 rather than zipping to the shorter vector, so a
    change of embedding model surfaces as "no match" instead of a confident
    wrong one.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------

class ContextStore:
    """Append-only JSON memory with session-aware, embedding-ranked retrieval."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self.embed_failures = 0
        self.embed_successes = 0
        self._warned_embedding = False
        self._ensure_dirs()

    # -- 1. initialization -------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.config.dir.mkdir(parents=True, exist_ok=True)
        if not self.config.index.exists():
            self._save_all([])

    # -- persistence -------------------------------------------------------

    def _load_all(self) -> list[dict[str, Any]]:
        path = self.config.index
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            return []
        if not raw:
            return []
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Preserve the damaged file instead of letting the next save
            # overwrite it with a fresh one-entry list.
            backup = path.with_suffix(f".corrupt-{int(time.time())}.json")
            try:
                path.replace(backup)
            except OSError:
                pass
            return []
        return data if isinstance(data, list) else []

    def _save_all(self, data: list[dict[str, Any]]) -> None:
        """Atomic: write a sibling temp file, then rename over the target."""
        path = self.config.index
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    # -- 3. embedding ------------------------------------------------------

    async def embed_text(self, text: str) -> tuple[list[float], bool]:
        """
        Return `(vector, ok)`. On any failure the vector is the configured-width
        zero vector and `ok` is False — the caller always gets a usable list,
        but the failure is recorded rather than disguised as a valid embedding.
        """
        cfg = self.config
        payload = {"model": cfg.embedding_model, "prompt": text}
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
                resp = await client.post(cfg.embedding_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return [0.0] * cfg.embedding_dim, False

        vec = data.get("embedding")
        # Newer Ollama (/api/embed) returns {"embeddings": [[...]]}
        if vec is None:
            nested = data.get("embeddings")
            if isinstance(nested, list) and nested and isinstance(nested[0], list):
                vec = nested[0]

        if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
            return [float(x) for x in vec], True

        return [0.0] * cfg.embedding_dim, False

    async def _embed(self, text: str) -> tuple[list[float], bool]:
        """
        Call the embedder and record the outcome.

        Health accounting lives here rather than inside `embed_text` so that it
        still works when `embed_text` is replaced — a self-hosted embedder, a
        cache, or a test double. Every internal call site goes through this.
        """
        vec, ok = await self.embed_text(text)
        if ok:
            self.embed_successes += 1
        else:
            self.embed_failures += 1
        return vec, ok

    @property
    def embeddings_degraded(self) -> bool:
        """True if embeddings have failed and never once succeeded."""
        return self.embed_failures > 0 and self.embed_successes == 0

    def degradation_notice(self) -> str | None:
        """One-shot warning text for the UI; returns None after the first call."""
        if not self.embeddings_degraded or self._warned_embedding:
            return None
        self._warned_embedding = True
        return (
            f"Embedding server unreachable at {self.config.embedding_url} "
            f"({self.embed_failures} failures). Memory is storing entries but "
            f"similarity ranking is disabled — retrieval falls back to most-recent."
        )

    # -- 2. add ------------------------------------------------------------

    async def add_context(self, session_id: str, intent: str, text: str) -> str:
        """Append one memory chunk and persist. Returns the new entry id."""
        if not text or not text.strip():
            return ""

        embedding, ok = await self._embed(text)
        entry = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "intent": intent,
            "text": text,
            "embedding": embedding,
            "embedding_ok": ok,
            "timestamp": time.time(),
        }

        async with self._lock:
            data = self._load_all()
            data.append(entry)
            if self.config.max_entries is not None and len(data) > self.config.max_entries:
                data.sort(key=lambda e: e.get("timestamp", 0.0))
                data = data[-self.config.max_entries:]
            self._save_all(data)

        return entry["id"]

    # -- 4. adaptive window ------------------------------------------------

    def _adaptive_n(self, intent: str) -> int:
        if intent in INTENT_WINDOW:
            return INTENT_WINDOW[intent]
        return INTENT_WINDOW.get(str(intent).lower(), DEFAULT_WINDOW)

    # -- 5. recent ---------------------------------------------------------

    async def get_recent(self, session_id: str, intent: str) -> list[dict[str, Any]]:
        """
        Newest-first entries for this session, topped up from the global pool if
        the session is shorter than the adaptive window.

        Entries carry `from_other_session` so callers can tell borrowed context
        apart — the global back-fill mixes unrelated work into a fresh session,
        which is the intended behaviour here but worth surfacing.
        """
        async with self._lock:
            data = self._load_all()

        n = self._adaptive_n(intent)

        same_session = [e for e in data if e.get("session_id") == session_id]
        same_session.sort(key=lambda e: e.get("timestamp", 0.0), reverse=True)
        recent = same_session[:n]
        for e in recent:
            e["from_other_session"] = False

        if len(recent) < n:
            used = {e.get("id") for e in recent}
            others = [e for e in data if e.get("id") not in used]
            others.sort(key=lambda e: e.get("timestamp", 0.0), reverse=True)
            for e in others[: n - len(recent)]:
                e["from_other_session"] = e.get("session_id") != session_id
                recent.append(e)

        # -- auto-repair embeddings ---------------------------------------
        repaired = False
        for e in recent:
            emb = e.get("embedding")
            if not isinstance(emb, list) or not emb:
                vec, ok = await self._embed(e.get("text", ""))
                e["embedding"] = vec
                e["embedding_ok"] = ok
                repaired = ok or repaired
            elif "embedding_ok" not in e:
                # Entries written before this field existed: infer it. An
                # all-zero vector is the old failure sentinel.
                e["embedding_ok"] = any(x != 0.0 for x in emb)

        if repaired:
            async with self._lock:
                stored = self._load_all()
                by_id = {e.get("id"): e for e in recent}
                for entry in stored:
                    patch = by_id.get(entry.get("id"))
                    if patch is not None:
                        entry["embedding"] = patch["embedding"]
                        entry["embedding_ok"] = patch["embedding_ok"]
                self._save_all(stored)

        return recent

    # -- 6. best match -----------------------------------------------------

    async def retrieve_best(
        self, session_id: str, intent: str, query: str
    ) -> RetrievedContext:
        recent = await self.get_recent(session_id, intent)
        if not recent:
            return RetrievedContext()

        rankable = [e for e in recent if e.get("embedding_ok")]

        # No usable embeddings — degrade to most-recent rather than returning an
        # arbitrary entry dressed up as the best semantic match.
        if not rankable:
            top = recent[0]
            return RetrievedContext(
                text=top.get("text", ""),
                score=0.0,
                entry_id=top.get("id", ""),
                intent=top.get("intent", ""),
                session_id=top.get("session_id", ""),
                from_other_session=bool(top.get("from_other_session")),
                considered=len(recent),
                rankable=0,
            )

        query_emb, ok = await self._embed(query)
        if not ok:
            top = recent[0]
            return RetrievedContext(
                text=top.get("text", ""), score=0.0, entry_id=top.get("id", ""),
                intent=top.get("intent", ""), session_id=top.get("session_id", ""),
                from_other_session=bool(top.get("from_other_session")),
                considered=len(recent), rankable=len(rankable),
            )

        best, best_score = None, -1.0
        for e in rankable:
            score = cosine_similarity(query_emb, e.get("embedding", []))
            if score > best_score:
                best, best_score = e, score

        if best is None:
            return RetrievedContext(considered=len(recent), rankable=len(rankable))

        return RetrievedContext(
            text=best.get("text", ""),
            score=best_score,
            entry_id=best.get("id", ""),
            intent=best.get("intent", ""),
            session_id=best.get("session_id", ""),
            from_other_session=bool(best.get("from_other_session")),
            considered=len(recent),
            rankable=len(rankable),
        )

    async def retrieve_best_context(
        self, session_id: str, intent: str, query: str
    ) -> str:
        """Spec-shaped convenience wrapper: the best chunk's text, or ''."""
        return (await self.retrieve_best(session_id, intent, query)).text

    # -- observability -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        data = self._load_all()
        sessions = {e.get("session_id") for e in data}
        ok = sum(1 for e in data if e.get("embedding_ok"))
        size = self.config.index.stat().st_size if self.config.index.exists() else 0
        return {
            "entries": len(data),
            "sessions": len(sessions),
            "embedded": ok,
            "unembedded": len(data) - ok,
            "index_bytes": size,
            "embed_failures": self.embed_failures,
            "embed_successes": self.embed_successes,
        }

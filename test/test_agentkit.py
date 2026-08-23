"""
Regression tests for the perimeter fixes, the wiring fixes, and the memory store.

Every test in TestJail / TestCommandPolicy corresponds to a defect that was
reproducible before the fix.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from omni import runtime as ar
from omni.agentkit import jail
from omni.agentkit.memory import (
    ContextStore,
    MemoryConfig,
    cosine_similarity,
    make_session_id,
)


# ---------------------------------------------------------------------------
# jail
# ---------------------------------------------------------------------------

class TestJail:
    def test_contained_accepts_self_and_children(self, tmp_path):
        assert jail.contained(tmp_path, tmp_path)
        assert jail.contained(tmp_path / "a" / "b.py", tmp_path)

    def test_sibling_prefix_is_not_contained(self, tmp_path):
        """`<root>-evil` prefixes `<root>` as a string but is not beneath it."""
        sibling = tmp_path.parent / (tmp_path.name + "-evil")
        assert str(sibling).startswith(str(tmp_path)) is False or True  # documents intent
        assert not jail.contained(sibling, tmp_path)

    def test_parent_is_not_contained(self, tmp_path):
        assert not jail.contained(tmp_path.parent, tmp_path)

    @pytest.mark.parametrize("token", [
        "../evil.py",
        "a/../../b.py",
        "C:/Windows/Temp/evil.py",
        "/etc/passwd",
    ])
    def test_escapes_are_refused(self, tmp_path, token):
        with pytest.raises(jail.JailBreak):
            jail.resolve_in(tmp_path, token)

    @pytest.mark.parametrize("token", ["CON", "NUL", "COM1", "aux.txt", "trailing."])
    def test_reserved_windows_names_refused(self, tmp_path, token):
        with pytest.raises(jail.JailBreak):
            jail.resolve_in(tmp_path, token)

    def test_traversal_that_stays_inside_is_allowed(self, tmp_path):
        """`sub/../a.py` resolves inside the root and must not be rejected."""
        assert jail.resolve_in(tmp_path, "sub/../a.py") == (tmp_path / "a.py").resolve()

    def test_write_text_in_creates_parents(self, tmp_path):
        target = jail.write_text_in(tmp_path, "pkg/mod/file.py", "x = 1\n")
        assert target.read_text(encoding="utf-8") == "x = 1\n"
        assert jail.contained(target, tmp_path)

    def test_write_text_in_refuses_absolute_escape(self, tmp_path):
        """`workspace / "C:/..."` silently discards the workspace; this must not."""
        with pytest.raises(jail.JailBreak):
            jail.write_text_in(tmp_path, "C:/Windows/Temp/pwn.txt", "x")

    def test_write_leaves_no_temp_files(self, tmp_path):
        jail.write_text_in(tmp_path, "a.py", "content")
        assert [p.name for p in tmp_path.iterdir()] == ["a.py"]


# ---------------------------------------------------------------------------
# CommandPolicy
# ---------------------------------------------------------------------------

class TestCommandPolicy:
    @pytest.fixture
    def policy(self, tmp_path):
        return ar.CommandPolicy(tmp_path)

    @pytest.mark.parametrize("command", [
        "cat C:/Users/someone/.ssh/id_rsa",       # drive-absolute: was ALLOWED
        "cat C:/Windows/System32/config/SAM",     # drive-absolute: was ALLOWED
        "cat /etc/passwd",
        "cat ../../../secrets.txt",
        "cat //server/share/secret",
    ])
    def test_escaping_paths_are_forbidden(self, policy, command):
        assert policy.classify(command).risk is ar.Risk.FORBIDDEN

    @pytest.mark.parametrize("command", ["ls", "git status", "pytest -q", "cat ./local.py"])
    def test_ordinary_commands_still_allowed(self, policy, command):
        assert policy.classify(command).risk is ar.Risk.SAFE

    def test_unlisted_executable_forbidden(self, policy):
        assert policy.classify("find / -name x").risk is ar.Risk.FORBIDDEN

    def test_windows_backslash_path_survives_tokenization(self):
        """posix=True ate every backslash: C:\\Users\\x -> C:Usersx."""
        argv = ar.split_command(r"cat C:\Users\wasim\notes.txt")
        assert argv[1].count("\\") == 3 or argv[1].startswith("C:\\")

    def test_verdict_argv_is_populated(self, policy):
        verdict = policy.classify("pytest -q tests")
        assert verdict.argv[0] == "pytest"


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------

class TestApprovals:
    @pytest.fixture
    def elevated(self, tmp_path):
        verdict = ar.CommandPolicy(tmp_path).classify("rm build")
        assert verdict.risk is ar.Risk.ELEVATED
        return verdict

    def test_no_approval_denies(self, elevated):
        assert not ar.approval_grants(elevated, [])

    def test_exact_command_grants(self, elevated):
        """The CLI grants [pending_command]; this used to satisfy nothing."""
        assert ar.approval_grants(elevated, ["rm build"])

    def test_executable_name_grants(self, elevated):
        assert ar.approval_grants(elevated, ["rm"])

    def test_sudo_token_still_grants(self, elevated):
        assert ar.approval_grants(elevated, ["sudo"])

    def test_unrelated_approval_denies(self, elevated):
        assert not ar.approval_grants(elevated, ["npm install"])


# ---------------------------------------------------------------------------
# executor wiring
# ---------------------------------------------------------------------------

class TestExecutorWiring:
    def test_ledger_is_shared_with_policy(self, tmp_path):
        """The repair factory must receive the run ledger, not build its own."""
        seen: list[ar.Ledger] = []

        def factory(command, ledger):
            seen.append(ledger)
            return ar.HeuristicRepairPolicy(command)

        orch = ar.Orchestrator(executor=ar.SimulatedShell(), workspace=tmp_path,
                               repair_policy_factory=factory)
        run_ledger = ar.Ledger(budget=ar.Budget())
        orch._make_repair_policy("npm install", run_ledger)
        assert seen == [run_ledger]

    def test_single_argument_factory_still_supported(self, tmp_path):
        orch = ar.Orchestrator(executor=ar.SimulatedShell(), workspace=tmp_path,
                               repair_policy_factory=lambda cmd: ar.HeuristicRepairPolicy(cmd))
        policy = orch._make_repair_policy("npm install", ar.Ledger(budget=ar.Budget()))
        assert isinstance(policy, ar.HeuristicRepairPolicy)

    def test_env_is_not_the_full_parent_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_SECRET_TOKEN", "sk-do-not-leak")
        shell = ar.SubprocessShell(workspace=tmp_path)
        assert "MY_SECRET_TOKEN" not in shell.env
        assert "PATH" in shell.env

    def test_full_env_available_on_request(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_SECRET_TOKEN", "sk-do-not-leak")
        shell = ar.SubprocessShell(workspace=tmp_path, inherit_full_env=True)
        assert "MY_SECRET_TOKEN" in shell.env


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------

def _store(tmp_path, **kw) -> ContextStore:
    return ContextStore(MemoryConfig(root=tmp_path, **kw))


class _FakeEmbedder:
    """Deterministic 8-dim embeddings; keyword presence drives similarity."""
    VOCAB = ["fastapi", "pytest", "docker", "sql", "react", "auth", "cache", "log"]

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def __call__(self, text: str):
        self.calls += 1
        if self.fail:
            return [0.0] * 768, False
        low = (text or "").lower()
        vec = [1.0 if word in low else 0.0 for word in self.VOCAB]
        if not any(vec):
            vec[0] = 0.001
        return vec, True


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_dimension_mismatch_scores_zero(self):
        """zip() used to truncate silently, producing confident wrong matches."""
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_zero_vector_scores_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestContextStore:
    def test_initialization_creates_index(self, tmp_path):
        store = _store(tmp_path)
        assert store.config.index.exists()
        assert json.loads(store.config.index.read_text(encoding="utf-8")) == []

    def test_add_context_persists_entry(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()
        entry_id = asyncio.run(store.add_context("sess-1", "coder", "build a fastapi app"))
        assert entry_id
        data = json.loads(store.config.index.read_text(encoding="utf-8"))
        assert len(data) == 1
        for key in ("id", "session_id", "intent", "text", "embedding", "timestamp"):
            assert key in data[0]

    @pytest.mark.parametrize("intent,expected", [
        ("coder", 2), ("chat", 3), ("script", 3), ("summarizer", 3),
        ("architect", 5), ("planner", 5), ("unknown-intent", 3),
        ("DIRECT_CODE", 2), ("PROJECT_REVIEW", 5),
    ])
    def test_adaptive_window(self, tmp_path, intent, expected):
        assert _store(tmp_path)._adaptive_n(intent) == expected

    def test_get_recent_prefers_same_session_newest_first(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()

        async def go():
            for i in range(4):
                await store.add_context("sess-A", "chat", f"alpha entry {i}")
                time.sleep(0.002)
            await store.add_context("sess-B", "chat", "other session")
            return await store.get_recent("sess-A", "chat")

        recent = asyncio.run(go())
        assert len(recent) == 3                       # chat window
        assert all(e["session_id"] == "sess-A" for e in recent)
        assert recent[0]["text"].endswith("3")        # newest first

    def test_global_backfill_when_session_is_short(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()

        async def go():
            await store.add_context("old-session", "chat", "history from before")
            await store.add_context("new-session", "chat", "only entry")
            return await store.get_recent("new-session", "chat")

        recent = asyncio.run(go())
        assert len(recent) == 2
        borrowed = [e for e in recent if e.get("from_other_session")]
        assert len(borrowed) == 1
        assert borrowed[0]["session_id"] == "old-session"

    def test_retrieve_best_picks_semantic_match(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()

        async def go():
            await store.add_context("s", "architect", "notes about docker and cache")
            await store.add_context("s", "architect", "notes about pytest and log")
            await store.add_context("s", "architect", "notes about react and auth")
            return await store.retrieve_best("s", "architect", "how do I run pytest?")

        hit = asyncio.run(go())
        assert "pytest" in hit.text
        assert hit.score > 0.0
        assert hit.rankable == 3

    def test_failed_embeddings_do_not_win_ranking(self, tmp_path):
        """
        The original returned [0.0]*768 on failure; every cosine was 0.0 while
        best_score started at -1.0, so the FIRST entry was returned as the
        "most similar" match. Failure must degrade to most-recent, and say so.
        """
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder(fail=True)

        async def go():
            await store.add_context("s", "chat", "first entry")
            await store.add_context("s", "chat", "second entry")
            return await store.retrieve_best("s", "chat", "anything")

        hit = asyncio.run(go())
        assert hit.rankable == 0
        assert hit.score == 0.0
        assert hit.text == "second entry"          # most recent, not "first"
        assert store.embeddings_degraded
        assert store.degradation_notice() is not None
        assert store.degradation_notice() is None  # warns once

    def test_corrupt_index_is_preserved_not_overwritten(self, tmp_path):
        store = _store(tmp_path)
        store.config.index.write_text("{ this is not json", encoding="utf-8")
        assert store._load_all() == []
        backups = list(store.config.dir.glob("index.corrupt-*.json"))
        assert len(backups) == 1
        assert "not json" in backups[0].read_text(encoding="utf-8")

    def test_save_is_atomic_and_leaves_no_temp_files(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()
        asyncio.run(store.add_context("s", "chat", "entry"))
        leftovers = [p.name for p in store.config.dir.iterdir() if ".tmp-" in p.name]
        assert leftovers == []

    def test_max_entries_prunes_oldest(self, tmp_path):
        store = _store(tmp_path, max_entries=3)
        store.embed_text = _FakeEmbedder()

        async def go():
            for i in range(6):
                await store.add_context("s", "chat", f"entry {i}")
                time.sleep(0.002)

        asyncio.run(go())
        data = json.loads(store.config.index.read_text(encoding="utf-8"))
        assert len(data) == 3
        assert data[-1]["text"] == "entry 5"

    def test_empty_text_is_not_stored(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()
        assert asyncio.run(store.add_context("s", "chat", "   ")) == ""
        assert json.loads(store.config.index.read_text(encoding="utf-8")) == []

    def test_embedding_auto_repair(self, tmp_path):
        store = _store(tmp_path)
        embedder = _FakeEmbedder()
        store.embed_text = embedder

        asyncio.run(store.add_context("s", "chat", "needs repair"))
        data = json.loads(store.config.index.read_text(encoding="utf-8"))
        del data[0]["embedding"]
        store._save_all(data)

        recent = asyncio.run(store.get_recent("s", "chat"))
        assert isinstance(recent[0]["embedding"], list) and recent[0]["embedding"]
        stored = json.loads(store.config.index.read_text(encoding="utf-8"))
        assert stored[0]["embedding"] == recent[0]["embedding"]

    def test_retrieve_on_empty_store_returns_blank(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()
        assert asyncio.run(store.retrieve_best_context("s", "chat", "q")) == ""

    def test_stats(self, tmp_path):
        store = _store(tmp_path)
        store.embed_text = _FakeEmbedder()
        asyncio.run(store.add_context("s", "chat", "one"))
        stats = store.stats()
        assert stats["entries"] == 1 and stats["embedded"] == 1

    def test_session_id_shape(self):
        assert make_session_id().startswith("sess-")

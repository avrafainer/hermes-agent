"""Capacity-aware memory: detail that outgrows the budget moves to a durable reference.

Before this, an add over the limit was refused and the refusal told the model to *delete*
entries to make room - which is how real content disappeared from memory. These tests pin the
replacement behaviour: the write succeeds, the long entry is preserved verbatim elsewhere, a
pointer remains, and the identity blocks are never moved.
"""

import pathlib

import pytest

from tools.memory_tool_store import MemoryStore, _is_protected

CORE = "# USER - Avra\n\n## Identity\n\n- Avra Fainer. Toronto. Singer first."


def detail(tag: str, size: int = 520) -> str:
    body = f"{tag}: " + ("chart archive page rows keyed by document and page number. " * 20)
    return body[:size]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return MemoryStore(memory_char_limit=900, user_char_limit=500)


def memfile(tmp_path) -> str:
    return (tmp_path / ".hermes" / "memories" / "MEMORY.md").read_text()


def spill_files(tmp_path) -> list[pathlib.Path]:
    d = tmp_path / ".hermes" / "state" / "references" / "memory-spill"
    return sorted(d.glob("*.md")) if d.is_dir() else []


class TestProtection:
    def test_identity_blocks_are_never_eligible(self):
        assert _is_protected(CORE)
        assert _is_protected("# MEMORY - how this system is built\n\n## Who I am")
        assert not _is_protected(detail("pipeline"))

    def test_nothing_eligible_means_the_overflow_is_reported(self, store, tmp_path):
        store.add("user", CORE)
        resp = store.add("user", "x" * 480)
        assert resp["success"] is False
        assert "exceed" in resp["error"]
        # nothing eligible was moved; the refused text is parked, not silently dropped
        files = spill_files(tmp_path)
        assert all("refused" in f.name for f in files)
        assert any("x" * 40 in f.read_text() for f in files)


class TestSpillOnWrite:
    def test_long_entry_moves_and_the_write_succeeds(self, store, tmp_path):
        first = detail("pipeline")
        assert store.add("memory", first)["success"] is True
        resp = store.add("memory", detail("second pass", 520))
        assert resp["success"] is True, resp
        assert "Moved" in (resp.get("message") or "")

        files = spill_files(tmp_path)
        assert files, "the entry should have been written to a reference file"
        assert "chart archive page rows keyed by document" in files[0].read_text()
        assert "state/references/memory-spill" in memfile(tmp_path)   # pointer left behind

    def test_detail_leaves_memory_but_the_text_survives(self, store, tmp_path):
        first = detail("pipeline")
        store.add("memory", first)
        store.add("memory", detail("second pass", 520))
        combined = "\n".join(f.read_text() for f in spill_files(tmp_path))
        assert "pipeline:" in combined or "pipeline:" in memfile(tmp_path)

    def test_pointer_is_short_and_searchable(self, store, tmp_path):
        store.add("memory", detail("pipeline"))
        store.add("memory", detail("second pass", 520))
        mem = memfile(tmp_path)
        pointer = next(e for e in mem.split("\n\u00a7\n") if "state/references/memory-spill" in e)
        assert "pipeline" in pointer        # still greppable in memory
        assert len(pointer) < 400           # a pointer, not a copy

    def test_repeated_writes_keep_working_as_it_fills(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        big = MemoryStore(memory_char_limit=1600, user_char_limit=500)
        for i in range(6):
            assert big.add("memory", detail(f"pass {i}", 450))["success"] is True, i
        assert len(memfile(tmp_path)) <= 1600
        combined = "\n".join(f.read_text() for f in spill_files(tmp_path)) + memfile(tmp_path)
        for i in range(6):
            assert f"pass {i}" in combined


class TestFailureHandling:
    def test_unwritable_reference_leaves_memory_untouched(self, store, tmp_path, monkeypatch):
        store.add("memory", detail("pipeline"))
        before = memfile(tmp_path)
        monkeypatch.setattr(MemoryStore, "_spill_to_fit", lambda self, t, e, n: (None, ""))
        pending = detail("second pass", 520)
        resp = store.add("memory", pending)
        assert resp["success"] is False          # refused, with the budget named
        assert "exceed" in resp["error"]
        assert memfile(tmp_path) == before       # nothing lost, nothing half-written
        # ...and the text that could not be written is parked, not dropped
        parked = [f for f in spill_files(tmp_path) if "refused" in f.name]
        assert parked and "second pass" in parked[0].read_text()


class TestCoreSurvives:
    def test_identity_section_stays_after_many_spills(self, store, tmp_path):
        store.add("user", CORE)
        for i in range(5):
            store.add("user", detail(f"user detail {i}", 420))
        user = (tmp_path / ".hermes" / "memories" / "USER.md").read_text()
        assert "## Identity" in user
        assert "Avra Fainer" in user
        assert len(user) <= 500

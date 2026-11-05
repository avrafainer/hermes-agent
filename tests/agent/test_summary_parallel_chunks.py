"""Parallel summary parts: the split must never lose or reorder a record.

The whole claim for chunking is "faster, nothing lost". These tests pin the nothing-lost half:
every record lands in exactly one part, in transcript order, and a part that fails takes the whole
summary down the fallback path instead of committing a partial one.
"""

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    c = ContextCompressor(model="test-model", config_context_length=200_000)
    c._summary_parallel_chunks = 3
    return c


def records(n, size=40):
    return [f"record-{i:03d} " + "x" * size for i in range(n)]


def test_disabled_means_no_split(compressor):
    compressor._summary_parallel_chunks = 1
    assert compressor._split_summary_records(records(120)) is None


def test_too_few_records_means_no_split(compressor):
    """Three parts of ten records each lose context for no wall-clock gain."""
    assert compressor._split_summary_records(records(30)) is None


def test_split_is_lossless_and_ordered(compressor):
    original = records(150)
    parts = compressor._split_summary_records(original)
    assert parts is not None and len(parts) == 3
    rebuilt = "\n\n".join(parts).split("\n\n")
    assert rebuilt == original, "every record, in order, exactly once"


def test_no_record_is_torn_across_parts(compressor):
    original = records(150, size=500)
    parts = compressor._split_summary_records(original)
    for part in parts:
        for line in part.split("\n\n"):
            assert line in original, "a split may not cut a record in half"


def test_parts_are_roughly_balanced(compressor):
    parts = compressor._split_summary_records(records(150, size=200))
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) < max(sizes), "one part must not carry the whole window"


def test_a_failed_part_raises_rather_than_committing_a_partial_summary(compressor, monkeypatch):
    calls = {"n": 0}

    def fake_call(prompt, started):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("summariser returned nothing")
        return "part text"

    monkeypatch.setattr(compressor, "_call_summary_llm", fake_call)
    monkeypatch.setattr(compressor, "_build_summary_prompt", lambda *a, **k: "prompt")
    with pytest.raises(RuntimeError):
        compressor._call_summary_llm_parallel(["a", "b", "c"], 4000, None, "", True)


def test_parts_are_joined_in_order_with_headers(compressor, monkeypatch):
    seen = []

    def fake_call(prompt, started):
        seen.append(prompt)
        return "summary-body"

    monkeypatch.setattr(compressor, "_call_summary_llm", fake_call)
    monkeypatch.setattr(compressor, "_build_summary_prompt", lambda content, *a, **k: content)
    joined = compressor._call_summary_llm_parallel(
        ["one", "two", "three"], 3000, None, "", True,
    )
    assert joined.index("Part 1 of 3") < joined.index("Part 2 of 3") < joined.index("Part 3 of 3")
    assert "summary-1" not in joined  # sanity: bodies are the fake's text, headers are ours
    assert joined.count("summary-body") == 3

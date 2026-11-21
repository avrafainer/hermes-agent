"""Parallel summary parts: the split must never lose, tear or reorder a turn.

The whole claim for chunking is "faster, nothing lost". These tests pin the nothing-lost half:
every turn lands in exactly one part, in transcript order, a tool result never opens a part
(so no call is separated from its result), and a part that fails takes the whole summary down the
fallback path instead of committing a partial one.
"""

import pytest

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    c = ContextCompressor(model="test-model", config_context_length=200_000)
    c._summary_parallel_chunks = 3
    return c


def turns(n, size=40):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i:03d} " + "x" * size}
            for i in range(n)]


def test_disabled_means_no_split(compressor):
    compressor._summary_parallel_chunks = 1
    assert compressor._split_turns_for_summary(turns(120)) is None


def test_too_few_turns_means_no_split(compressor):
    """Three parts of ten turns each lose context for no wall-clock gain."""
    assert compressor._split_turns_for_summary(turns(30)) is None


def test_split_is_lossless_and_ordered(compressor):
    original = turns(150)
    parts = compressor._split_turns_for_summary(original)
    assert parts is not None and len(parts) == 3
    rebuilt = [m for part in parts for m in part]
    assert rebuilt == original, "every turn, in order, exactly once"


def test_no_part_opens_with_a_tool_result(compressor):
    """A tool result belongs to the assistant call before it; a cut there would orphan the pair."""
    original = []
    for i in range(60):
        original.append({"role": "assistant", "content": f"call-{i} " + "y" * 200,
                         "tool_calls": [{"id": f"c{i}", "function": {"name": "t", "arguments": "{}"}}]})
        original.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i} " + "z" * 200})
        original.append({"role": "user", "content": f"next-{i}"})
    parts = compressor._split_turns_for_summary(original)
    assert parts is not None
    for part in parts:
        assert part[0]["role"] != "tool"


def test_parts_are_roughly_balanced(compressor):
    parts = compressor._split_turns_for_summary(turns(150, size=200))
    sizes = [len(part) for part in parts]
    assert max(sizes) - min(sizes) < max(sizes), "one part must not carry the whole window"


def _fake_request_messages(seen):
    def _build(self, part, budget, focus, memory, *, has_user_turn,
               include_previous_summary=True):
        seen.append(include_previous_summary)
        return [{"role": "system", "content": f"sys-{budget}"},
                {"role": "user", "content": str(part[0]["content"])}]
    return _build


def test_a_failed_part_raises_rather_than_committing_a_partial_summary(compressor, monkeypatch):
    calls = {"n": 0}

    def fake_call(messages, started):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("summariser returned nothing")
        return "part text"

    monkeypatch.setattr(compressor, "_summary_request_messages", _fake_request_messages([]).__get__(compressor))
    monkeypatch.setattr(compressor, "_call_summary_llm", fake_call)
    with pytest.raises(RuntimeError):
        compressor._call_summary_llm_parallel([turns(30), turns(30), turns(30)], 4000, None, "")


def test_parts_are_joined_in_order_with_headers(compressor, monkeypatch):
    monkeypatch.setattr(compressor, "_summary_request_messages", _fake_request_messages([]).__get__(compressor))
    monkeypatch.setattr(compressor, "_call_summary_llm", lambda messages, started: "summary-body")
    joined = compressor._call_summary_llm_parallel([turns(30), turns(30), turns(30)], 3000, None, "")
    assert joined.index("Part 1 of 3") < joined.index("Part 2 of 3") < joined.index("Part 3 of 3")
    assert joined.count("summary-body") == 3


def test_only_the_first_part_carries_the_previous_summary(compressor, monkeypatch):
    """Parts 2..N never saw the earlier conversation; asking each to merge the whole previous
    summary would append N-1 redundant copies of it."""
    seen = []
    monkeypatch.setattr(compressor, "_summary_request_messages", _fake_request_messages(seen).__get__(compressor))
    monkeypatch.setattr(compressor, "_call_summary_llm", lambda messages, started: "part")
    compressor._call_summary_llm_parallel([turns(30), turns(30), turns(30)], 3000, None, "")
    assert seen == [True, False, False]

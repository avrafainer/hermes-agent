"""Parallel summary parts: the split must never lose a record, never tear a tool call from its result,
and a failed part must take the whole summary down the fallback path instead of committing a partial one.

Pure behaviour contracts on the lean summary path: no file I/O, no network. The summariser call is
replaced by a recorder so the request each part would send is what gets checked.
"""

import pytest

from agent.context_compressor import ContextCompressor, _MIN_RECORDS_PER_CHUNK


@pytest.fixture()
def compressor():
    c = ContextCompressor(model="test-model", config_context_length=200_000)
    c._summary_parallel_chunks = 3
    return c


def window(n_rounds, result_size=300):
    """User turn, then per round an assistant tool call with one or two results, then an assistant reply."""
    msgs = []
    for i in range(n_rounds):
        msgs.append({"role": "user", "content": f"request {i}"})
        call_ids = [f"call_{i}_a"] + ([f"call_{i}_b"] if i % 3 == 0 else [])
        msgs.append({
            "role": "assistant", "content": f"working on {i}",
            "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}
                for cid in call_ids
            ],
        })
        for cid in call_ids:
            msgs.append({"role": "tool", "tool_call_id": cid, "content": f"out {cid} " + "y" * result_size})
        msgs.append({"role": "assistant", "content": f"done {i}"})
    return msgs


def test_disabled_means_no_split(compressor):
    compressor._summary_parallel_chunks = 1
    turns = window(60)
    assert compressor._split_turns_for_summary(turns, compressor._serialize_records_for_summary(turns)) is None


def test_too_small_a_window_means_no_split(compressor):
    """Three parts of a few records each lose context for no wall-clock gain."""
    turns = window(4)
    assert len(turns) < 3 * _MIN_RECORDS_PER_CHUNK
    assert compressor._split_turns_for_summary(turns, compressor._serialize_records_for_summary(turns)) is None


def test_ranges_cover_every_message_exactly_once_in_order(compressor):
    turns = window(120)
    ranges = compressor._split_turns_for_summary(turns, compressor._serialize_records_for_summary(turns))
    assert ranges is not None and len(ranges) == 3
    covered = [i for start, end in ranges for i in range(start, end)]
    assert covered == list(range(len(turns))), "no record lost, duplicated or reordered"


def test_no_cut_falls_between_a_tool_call_and_its_result(compressor):
    turns = window(120)
    ranges = compressor._split_turns_for_summary(turns, compressor._serialize_records_for_summary(turns))
    for start, _end in ranges[1:]:
        assert turns[start].get("role") != "tool", "a part must not open with a tool result"
    for start, end in ranges:
        in_part = turns[start:end]
        calls = {tc["id"] for m in in_part if m.get("role") == "assistant" for tc in m.get("tool_calls", [])}
        for m in in_part:
            if m.get("role") == "tool":
                assert m["tool_call_id"] in calls, "every tool result sits with its call in the same part"


def test_a_failed_part_raises_rather_than_committing_a_partial_summary(compressor, monkeypatch):
    calls = {"n": 0}

    def fake_call(messages, started):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("summariser returned nothing")
        return "part text"

    monkeypatch.setattr(compressor, "_call_summary_llm", fake_call)
    with pytest.raises(RuntimeError):
        compressor._call_summary_llm_parallel([[{"role": "system", "content": "s"}]] * 3, 0.0)


def test_generate_summary_sends_every_record_once_across_parts(compressor, monkeypatch):
    """End to end through _generate_summary: N requests, every record represented in some request."""
    turns = window(120)
    sent = []

    def fake_call(messages, started):
        sent.append(messages)
        return "## Summary\n- part body"

    monkeypatch.setattr(compressor, "_call_summary_llm", fake_call)
    monkeypatch.setattr(compressor, "_on_summary_failure", lambda *a, **k: pytest.fail("unexpected failure path"))
    compressor._summary_failure_cooldown_until = 0.0
    compressor._summary_has_user_turn = True
    # The user-turn probe imports the conversation loop, whose import-time update settle touches the real
    # install; the summary contract under test does not depend on it.
    monkeypatch.setattr(compressor, "_ground_historical_task_snapshot", lambda summary, turns: summary)
    summary = compressor._generate_summary(turns)
    assert summary is not None
    assert len(sent) == 3, "one summariser request per part"
    assert compressor._last_summary_chunk_count == 3
    for marker in [f"request {i}" for i in range(120)]:
        assert any(marker in str(m) for req in sent for m in req), f"{marker} missing from every part"
    assert "### Part 1 of 3" in summary and "### Part 3 of 3" in summary


def test_disabled_sends_one_request(compressor, monkeypatch):
    compressor._summary_parallel_chunks = 1
    turns = window(120)
    sent = []
    monkeypatch.setattr(compressor, "_call_summary_llm", lambda messages, started: sent.append(messages) or "body")
    compressor._summary_failure_cooldown_until = 0.0
    compressor._summary_has_user_turn = True
    compressor._generate_summary(turns)
    assert len(sent) == 1
    assert compressor._last_summary_chunk_count == 1

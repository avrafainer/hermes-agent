"""Post-turn background compaction: trigger after the reply, and the next-turn race lock."""

from __future__ import annotations

import threading
from types import SimpleNamespace

from agent.compaction_events import attempt_payload
from agent.post_turn_compaction import (
    TRIGGER,
    await_post_turn_compaction,
    schedule_post_turn_compaction,
)

THRESHOLD = 1000


def _compressor(last_prompt_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        last_prompt_tokens=last_prompt_tokens,
        should_compress_info=lambda tokens: (tokens >= THRESHOLD, None),
    )


def _agent(last_prompt_tokens: int, compress) -> SimpleNamespace:
    return SimpleNamespace(
        compression_enabled=True,
        context_compressor=_compressor(last_prompt_tokens),
        session_id="sess-1",
        _cached_system_prompt="SYS",
        _compress_context=compress,
    )


def _messages() -> list:
    return [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_under_threshold_does_not_schedule():
    calls = []
    agent = _agent(THRESHOLD - 1, lambda *a, **k: calls.append((a, k)))

    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is False
    assert await_post_turn_compaction(agent, timeout=1) is True
    assert calls == []


def test_unknown_usage_does_not_schedule():
    calls = []
    agent = _agent(0, lambda *a, **k: calls.append((a, k)))

    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is False
    assert calls == []


def test_over_threshold_runs_in_background_with_post_turn_trigger():
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def compress(messages, system_message, **kwargs):
        seen.update(kwargs, system_message=system_message, n=len(messages))
        started.set()
        release.wait(5)
        return messages, system_message

    agent = _agent(THRESHOLD + 500, compress)

    assert schedule_post_turn_compaction(agent, _messages(), task_id="task-9") is True
    assert started.wait(5), "background compaction never started"
    assert seen["trigger"] == TRIGGER == "post_turn_background"
    assert seen["approx_tokens"] == THRESHOLD + 500
    assert seen["task_id"] == "task-9"
    assert seen["system_message"] == "SYS"
    assert seen["n"] == 2
    release.set()
    assert await_post_turn_compaction(agent, timeout=5) is True


def test_race_lock_refuses_second_run_while_first_is_in_flight():
    release = threading.Event()
    started = threading.Event()
    count = []

    def compress(messages, system_message, **kwargs):
        count.append(1)
        started.set()
        release.wait(5)
        return messages, system_message

    agent = _agent(THRESHOLD + 1, compress)

    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is True
    assert started.wait(5)
    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is False
    release.set()
    assert await_post_turn_compaction(agent, timeout=5) is True
    assert len(count) == 1


def test_next_turn_waits_for_inflight_background_run():
    release = threading.Event()
    started = threading.Event()
    order = []

    def compress(messages, system_message, **kwargs):
        started.set()
        release.wait(5)
        order.append("compaction_done")
        return messages, system_message

    agent = _agent(THRESHOLD + 1, compress)
    schedule_post_turn_compaction(agent, _messages(), task_id="t")
    assert started.wait(5)

    # Bounded join: still running, so the next turn is told it must proceed without the lock.
    assert await_post_turn_compaction(agent, timeout=0.05) is False

    threading.Thread(target=release.set, daemon=True).start()
    assert await_post_turn_compaction(agent, timeout=5) is True
    order.append("next_turn")
    assert order == ["compaction_done", "next_turn"]


def test_background_failure_is_contained_and_releases_the_lock():
    def compress(messages, system_message, **kwargs):
        raise RuntimeError("summary provider down")

    agent = _agent(THRESHOLD + 1, compress)

    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is True
    assert await_post_turn_compaction(agent, timeout=5) is True
    # The lock is free again, so a later turn over threshold can schedule a new run.
    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is True
    assert await_post_turn_compaction(agent, timeout=5) is True


def test_compression_disabled_never_schedules():
    calls = []
    agent = _agent(THRESHOLD + 1, lambda *a, **k: calls.append(1))
    agent.compression_enabled = False

    assert schedule_post_turn_compaction(agent, _messages(), task_id="t") is False
    assert calls == []


def test_telemetry_accepts_the_post_turn_trigger_label():
    record = {
        "trigger_source": TRIGGER, "commit_status": "committed", "session_id": "s",
        "attempt_id": "a", "route": "hermes",
    }
    payload = attempt_payload(record)
    assert payload["trigger"] == TRIGGER

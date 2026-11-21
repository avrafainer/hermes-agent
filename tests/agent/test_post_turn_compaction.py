"""Post-turn background compaction: trigger after the reply, and the next-turn race lock."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import agent.post_turn_compaction as ptc
from agent.compaction_events import attempt_payload
from agent.post_turn_compaction import (
    TRIGGER,
    await_post_turn_compaction,
    schedule_post_turn_compaction,
)
from agent.turn_context import PRE_LLM_BLOCK_EXIT_REASON
from agent.turn_finalizer import finalize_turn
from tests.agent.test_turn_finalizer_interrupt_alternation import _StubAgent

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


# --- Hook gating: the finalizer schedules only for a committed, successful turn. ---------------

def _finalize_turn(agent, *, interrupted=False, failed=False, exit_reason="completed", final="done"):
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": final}]
    return finalize_turn(
        agent,
        final_response=final,
        api_call_count=1,
        interrupted=interrupted,
        failed=failed,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="hi",
        original_user_message="hi",
        _should_review_memory=False,
        _turn_exit_reason=exit_reason,
    )


@pytest.fixture
def scheduled(monkeypatch):
    calls = []

    def _record(agent, messages, *, task_id):
        calls.append({"messages_len": len(messages), "task_id": task_id, "persisted": agent.persisted_messages})
        return True

    monkeypatch.setattr(ptc, "schedule_post_turn_compaction", _record)
    return calls


def _over_threshold_agent():
    agent = _StubAgent()
    agent.compression_enabled = True
    agent.context_compressor.last_prompt_tokens = 80_000
    return agent


@pytest.mark.parametrize(
    "kwargs, persist_disabled",
    [
        ({"interrupted": True, "exit_reason": "interrupted_by_user"}, False),
        ({"failed": True, "exit_reason": "error"}, False),
        ({"exit_reason": PRE_LLM_BLOCK_EXIT_REASON}, False),
        ({}, True),
    ],
    ids=["interrupted", "failed", "vetoed_pre_llm_block", "persist_disabled"],
)
def test_non_committed_turn_does_not_schedule(scheduled, kwargs, persist_disabled):
    agent = _over_threshold_agent()
    agent._persist_disabled = persist_disabled
    _finalize_turn(agent, **kwargs)
    assert scheduled == []


def test_successful_over_threshold_turn_schedules_after_persist(scheduled):
    agent = _over_threshold_agent()
    _finalize_turn(agent)
    assert len(scheduled) == 1
    # The reply is persisted before the hook runs, so the scheduled run sees the committed turn.
    assert scheduled[0]["persisted"] is not None
    assert scheduled[0]["persisted"][-1]["content"] == "done"
    assert scheduled[0]["task_id"] == "task-1"

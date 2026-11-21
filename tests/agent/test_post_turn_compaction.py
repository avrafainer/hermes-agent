"""Post-turn compaction: the gate, the watermark, and the fail-closed paths.

The hook exists so the expensive summary call lands after the reply instead of in front of the
next request. These tests pin the gates, because a compaction fired at the wrong moment is worse
than none: it either blocks a turn it was meant to spare, or rewrites a transcript that a
persistence-isolated agent (the background review fork) does not own.
"""

import logging
from types import SimpleNamespace

import pytest

from agent import turn_context
from agent.turn_finalizer import _post_turn_compaction

LOG = logging.getLogger("test.post_turn")


class _Compressor:
    def __init__(self, threshold=300_000, awaiting=False):
        self.threshold_tokens = threshold
        self.awaiting_real_usage_after_compression = awaiting


class _Agent:
    def __init__(self, *, enabled=True, ratio=0.75, persist_disabled=False, compressor=None,
                 result=None, raises=None):
        self.compression_post_turn_compact = enabled
        self.compression_post_turn_compact_ratio = ratio
        self.compression_enabled = True
        self._persist_disabled = persist_disabled
        self.context_compressor = compressor or _Compressor()
        self.session_id = "s1"
        self.active_system_prompt = ""
        self.calls = []
        self._result = result
        self._raises = raises

    def _compress_context(self, messages, system_message, approx_tokens=None, task_id=None, trigger=None):
        self.calls.append({"trigger": trigger, "tokens": approx_tokens, "task_id": task_id})
        if self._raises:
            raise self._raises
        if self._result is not None:
            return self._result, system_message
        return messages, system_message


@pytest.fixture(autouse=True)
def _fixed_estimate(monkeypatch):
    """Pin the token estimate so each test controls the watermark comparison."""
    monkeypatch.setattr(turn_context, "_preflight_request_tokens", lambda agent, messages, prompt: agent._tokens)


def _run(agent, messages=None):
    messages = messages if messages is not None else [{"role": "user", "content": "hi"}]
    _post_turn_compaction(agent, messages, LOG, "task-1")
    return messages


def test_disabled_does_nothing():
    agent = _Agent(enabled=False)
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []


def test_below_watermark_does_nothing():
    agent = _Agent(ratio=0.75)
    agent._tokens = int(300_000 * 0.75) - 1
    _run(agent)
    assert agent.calls == []


def test_at_watermark_compacts_with_post_turn_trigger():
    agent = _Agent(ratio=0.75, result=[{"role": "user", "content": "summary"}])
    agent._tokens = int(300_000 * 0.75)
    messages = _run(agent)
    assert len(agent.calls) == 1
    assert agent.calls[0]["trigger"] == "post_turn"
    assert messages == [{"role": "user", "content": "summary"}], "compacted list replaces messages in place"


def test_persistence_isolated_agent_is_never_compacted():
    """The background-review fork runs with persistence disabled: compacting there would burn an
    aux call on a throwaway transcript and could touch the canonical session rows."""
    agent = _Agent(persist_disabled=True)
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []


def test_awaiting_real_usage_latch_blocks_it():
    agent = _Agent(compressor=_Compressor(awaiting=True))
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []


def test_single_attempt_shape_is_enforced():
    """A ratio or threshold that is not a real number must not arm the hook: MagicMock compressors
    return truthy non-ints here, and treating one as a threshold compacts every turn."""
    agent = _Agent(ratio="0.75")
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []
    agent = _Agent(compressor=SimpleNamespace(threshold_tokens="300000"))
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []


def test_failure_is_swallowed_and_messages_untouched():
    agent = _Agent(raises=RuntimeError("summariser exploded"))
    agent._tokens = 999_999
    messages = [{"role": "user", "content": "hi"}]
    _run(agent, messages)
    assert messages == [{"role": "user", "content": "hi"}]
    assert len(agent.calls) == 1, "it did try"


def test_compression_disabled_agent_does_nothing():
    agent = _Agent()
    agent.compression_enabled = False
    agent._tokens = 999_999
    _run(agent)
    assert agent.calls == []

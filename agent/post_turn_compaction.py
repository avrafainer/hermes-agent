"""Post-turn background compaction.

When a turn finishes with the context over the compressor threshold, compaction runs on a
daemon thread *after* the reply is persisted, so the user never waits on it. The next turn
joins that thread before its turn-start compaction passes, so the two cannot race over the
same transcript. A bounded join keeps a stalled run from blocking the next turn forever; past
the bound the next turn proceeds and the session compression lease (``try_acquire_compression_lock``)
makes its own preflight defer, which is the cross-process guard.

Trigger label: ``post_turn_background`` (``agent._compress_context(trigger=...)``), recorded in the
compaction attempt telemetry as ``trigger_source``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("agent.post_turn_compaction")

TRIGGER = "post_turn_background"
# Longest a turn may wait on an in-flight background compaction before it proceeds anyway.
JOIN_TIMEOUT_SECONDS = 120.0


class _PostTurnState:
    """Per-agent record of the in-flight background run. ``lock`` guards ``thread``."""

    __slots__ = ("lock", "thread")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None


def _state(agent: Any) -> _PostTurnState:
    st = getattr(agent, "_post_turn_compaction", None)
    if not isinstance(st, _PostTurnState):
        st = _PostTurnState()
        agent._post_turn_compaction = st
    return st


def schedule_post_turn_compaction(agent: Any, messages: list, *, task_id: str) -> bool:
    """Start a background compaction if the last real prompt size is over threshold.

    Returns True only when a new run was started. Returns False (without starting) when
    compression is off, usage is unknown, the context is under threshold, or a run is already
    in flight (the race lock).
    """
    if not getattr(agent, "compression_enabled", False):
        return False
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None or not messages:
        return False
    tokens = getattr(compressor, "last_prompt_tokens", 0)
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
        return False
    should, _reason = compressor.should_compress_info(tokens)
    if not should:
        return False
    st = _state(agent)
    with st.lock:
        if st.thread is not None and st.thread.is_alive():
            return False
        system_message = getattr(agent, "_cached_system_prompt", None) or ""
        # Background work goes through spawn_context_thread so the run keeps the turn's
        # contextvars (profile scope, session binding); a bare Thread would drop them.
        from agent.memory_provider import spawn_context_thread

        thread = spawn_context_thread(
            _run, name="post-turn-compaction",
            args=(agent, list(messages), system_message, tokens, task_id),
        )
        st.thread = thread
        thread.start()
    return True


def _run(agent: Any, messages: list, system_message: str, tokens: int, task_id: str) -> None:
    logger.info(
        "Post-turn compaction started: trigger_source=%s ~%s tokens (session %s)",
        TRIGGER, f"{tokens:,}", getattr(agent, "session_id", None) or "none",
    )
    try:
        agent._compress_context(
            messages, system_message, approx_tokens=tokens, task_id=task_id, trigger=TRIGGER,
        )
    except Exception:
        logger.warning("Post-turn compaction failed (trigger_source=%s)", TRIGGER, exc_info=True)
        return
    logger.info("Post-turn compaction finished: trigger_source=%s", TRIGGER)


def await_post_turn_compaction(agent: Any, timeout: float = JOIN_TIMEOUT_SECONDS) -> bool:
    """Block the next turn until an in-flight background compaction ends.

    Returns True when nothing is in flight or the run finished within ``timeout``; False when
    it is still running past the bound (the caller proceeds and the compression lease defers).
    """
    st = getattr(agent, "_post_turn_compaction", None)
    if not isinstance(st, _PostTurnState):
        return True
    with st.lock:
        thread = st.thread
    if thread is None or not thread.is_alive():
        return True
    logger.info("Next turn waiting on post-turn compaction: trigger_source=%s", TRIGGER)
    thread.join(timeout)
    return not thread.is_alive()

"""Regression tests for the ops-reliability repair (branch fix/ops-reliability-repair).

Two defects were fixed:

1. **Dispatcher recovery** — a since-resolved billing/quota wall left a stale
   ``last_failure_error`` on the card. On a clean handoff to review the stale error
   still matched ``_RESPAWN_BLOCKER_RE``, so ``check_respawn_guard(lane="review")``
   returned ``blocker_auth`` and the reviewer never spawned. ``request_review`` now
   clears ONLY that stale error snapshot — inside the SAME transaction as the
   ``running→review`` transition, so the two are atomic — and, critically,
   PRESERVES ``consecutive_failures`` (the circuit-breaker budget). A review
   handoff is not terminal, so wiping the counter (what ``complete_task`` does on
   true success) would reset the breaker every time a task bounces through
   ``changes_requested``.

2. **Worker GitHub SSH auth** — the gateway runs under launchd without
   ``SSH_AUTH_SOCK``, so workers that reach GitHub/SSH hosts had no agent socket.
   Added ``kanban.forward_ssh_agent`` (default False) and ``_maybe_forward_ssh_agent``,
   which discovers the launchd socket via ``launchctl getenv`` and forwards the
   POINTER (never key material) only for opt-in profiles.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _stamp(conn, tid, failures, error):
    """Stamp the card with a prior failure state, as a real failure would."""
    conn.execute(
        "UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
        (failures, error, tid),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Defect 1: request_review clears only the stale blocker error, atomically,
#           and preserves the circuit-breaker counter.
# ---------------------------------------------------------------------------

def test_request_review_clears_stale_quota_error_gate(kanban_home):
    """Recovered-quota scenario: the stale quota-flavoured error must not gate
    the reviewer's spawn. The error is cleared; the counter is preserved."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="billing cleared", assignee="a")
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w0")

        # A since-resolved billing wall left a quota-flavoured error + a breaker
        # counter the implementer accumulated earlier.
        _stamp(conn, tid, 2, "rate-limited: monthly billing quota wall — since resolved")
        assert kb.get_task(conn, tid).last_failure_error

        assert kb.request_review(
            conn, tid, summary="done", reviewer="r",
            expected_run_id=claimed.current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert not task.last_failure_error          # obsolete blocker state cleared
        assert task.consecutive_failures == 2       # breaker budget preserved

        # The reviewer lane must NOT be gated by the cleared error.
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None


def test_request_review_preserves_breaker_counter_on_genuine_failure(kanban_home):
    """Genuine-failure scenario: a task that accumulated real crash failures
    before finally handing off must KEEP its consecutive-failures budget — a
    review handoff is not a success that resets the circuit breaker."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="flaky", assignee="a")
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w0")

        # Two genuine crashes left the task one failure short of its breaker limit.
        _stamp(conn, tid, 2, "crashed: worker OOM during implementation")

        assert kb.request_review(
            conn, tid, summary="recovered and finished", reviewer="r",
            expected_run_id=claimed.current_run_id,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        # The counter survives the handoff: the breaker must still be armed for
        # whatever work comes next (e.g. the reviewer or a changes_requested bounce).
        assert task.consecutive_failures == 2
        assert not task.last_failure_error


def test_request_review_atomic_persisted_state_via_fresh_connection(kanban_home):
    """Atomicity: the transition AND the stale-error clear commit together.
    A fresh, second connection — the dispatcher's vantage point — must see
    ``review`` status WITH the stale error already gone, never the intermediate
    state where the task is ``review`` but still carries the blocker error."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="atomic", assignee="a")
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w0")
        _stamp(conn, tid, 1, "rate-limited: monthly billing quota wall")

        assert kb.request_review(
            conn, tid, summary="done", reviewer="r",
            expected_run_id=claimed.current_run_id,
        )
        # conn is still open; the txn committed. Now a SECOND, independent
        # connection reads the committed file state directly.
        with kbc.connect() as conn2:
            t2 = kb.get_task(conn2, tid)
            assert t2.status == "review"
            assert not t2.last_failure_error       # cleared, not just thread-local
            assert t2.consecutive_failures == 1    # preserved across the boundary


def test_request_review_rollback_does_not_clear_error(kanban_home):
    """On a FAILED handoff the stale error must survive so the card still shows
    the diagnosis — only a successful handoff clears it."""
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="still failed", assignee="a")
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w0")
        _stamp(conn, tid, 3, "rate-limited: quota wall")

        def _boom(*_a, **_k):
            raise RuntimeError("run bookkeeping failed")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(kb, "_end_or_synthesize_run", _boom)
            with pytest.raises(RuntimeError):
                kb.request_review(
                    conn, tid, summary="done", reviewer="r",
                    expected_run_id=claimed.current_run_id,
                )
        # Handoff failed and rolled back: nothing was cleared.
        assert kb.get_task(conn, tid).last_failure_error
        assert kb.get_task(conn, tid).consecutive_failures == 3


# ---------------------------------------------------------------------------
# Defect 2: SSH agent forwarding
# ---------------------------------------------------------------------------

def test_discover_ssh_auth_sock_from_env():
    """SSH_AUTH_SOCK already in the process env is used verbatim."""
    assert kbd._discover_ssh_auth_sock({"SSH_AUTH_SOCK": "/tmp/agent.sock"}) == "/tmp/agent.sock"


def test_discover_ssh_auth_sock_via_launchctl(monkeypatch):
    """On darwin without SSH_AUTH_SOCK, the launchd socket is resolved via
    launchctl getenv (pointer only — never key material)."""
    import hermes_cli.kanban_db_dispatch as _kbd

    monkeypatch.setattr(_kbd.sys, "platform", "darwin")

    class _Proc:
        stdout = "  /tmp/launchd-agent.sock\n"

    def _fake_run(*_a, **_k):
        return _Proc()

    sock = _kbd._discover_ssh_auth_sock({}, _run=_fake_run)
    assert sock == "/tmp/launchd-agent.sock"


def test_discover_ssh_auth_sock_fail_closed(monkeypatch):
    """launchctl failing, or a non-darwin platform with no env socket, yields
    None — discovery must never raise."""
    import hermes_cli.kanban_db_dispatch as _kbd

    class _Proc:
        stdout = "\n"
        returncode = 1

    def _fake_run(*_a, **_k):
        return _Proc()

    monkeypatch.setattr(_kbd.sys, "platform", "darwin")
    assert _kbd._discover_ssh_auth_sock({}, _run=_fake_run) is None

    monkeypatch.setattr(_kbd.sys, "platform", "linux")
    assert _kbd._discover_ssh_auth_sock({}) is None


def test_maybe_forward_ssh_agent_injects_when_enabled(monkeypatch):
    """A profile with kanban.forward_ssh_agent=true gets the socket pointer."""
    import contextlib

    import hermes_cli.kanban_db_dispatch as _kbd
    import hermes_cli.config as _cfg

    monkeypatch.setattr(
        _kbd, "_worker_profile_scope",
        lambda home, bind_home=True: contextlib.nullcontext(),
    )
    monkeypatch.setattr(_cfg, "load_config_readonly", dict)
    monkeypatch.setattr(_cfg, "cfg_get", lambda *a, default=False, **k: True)
    monkeypatch.setattr(_kbd, "_discover_ssh_auth_sock", lambda: "/tmp/agent.sock")

    env = {}
    _kbd._maybe_forward_ssh_agent(env, "/profiles/foo")
    assert env["SSH_AUTH_SOCK"] == "/tmp/agent.sock"


def test_maybe_forward_ssh_agent_noop_when_disabled(monkeypatch):
    """Default (False) forwards nothing and never even probes SSH discovery."""
    import contextlib

    import hermes_cli.kanban_db_dispatch as _kbd
    import hermes_cli.config as _cfg

    monkeypatch.setattr(
        _kbd, "_worker_profile_scope",
        lambda home, bind_home=True: contextlib.nullcontext(),
    )
    monkeypatch.setattr(_cfg, "load_config_readonly", dict)
    monkeypatch.setattr(_cfg, "cfg_get", lambda *a, default=False, **k: False)

    def _must_not_run():
        raise AssertionError("discovery must not run when forwarding is off")

    monkeypatch.setattr(_kbd, "_discover_ssh_auth_sock", _must_not_run)

    env = {}
    _kbd._maybe_forward_ssh_agent(env, "/profiles/foo")
    assert "SSH_AUTH_SOCK" not in env


def test_maybe_forward_ssh_agent_no_profile(monkeypatch):
    """A missing profile_home (isolated fixtures) forwards nothing."""
    import hermes_cli.kanban_db_dispatch as _kbd

    env = {}
    _kbd._maybe_forward_ssh_agent(env, None)
    assert env == {}
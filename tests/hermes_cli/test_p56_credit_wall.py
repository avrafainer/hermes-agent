"""P5.6: stop a lane when its provider says the wallet is empty.

On 2026-10-30 the Pi lane made 446 launches in one hour and 197 came back
`rate_limited`; the crashed runs carry the worker's own last words, "credit limit
is exhausted ... Out of credits". The balance was empty and dispatch kept
spawning, because the local spend guard watches Avra's daily ceiling and cannot
see an account wallet at OpenRouter. 369 launches bought the same sentence.

So: recognise the sentence in a dead worker's output, pause that lane for a
configured window, record ONE event per window, and refuse before the claim while
the pause is live.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect_closing() as c:
        yield c


# --- recognising the sentence ------------------------------------------------

@pytest.mark.parametrize("text", [
    "⚡ Out of credits",
    "Your credit limit is exhausted for anthropic/claude-haiku-5.5",
    "insufficient credits for this request",
    "Please add credits to continue",
])
def test_each_provider_phrasing_is_recognised(text):
    assert kbd._credit_wall_signature(text)


@pytest.mark.parametrize("text", [
    "",
    None,
    "worker exited cleanly (rc=0) without kanban_complete",
    "KeyError: 'credits_remaining'",
    "rate limit exceeded, retry after 20s",
])
def test_ordinary_failures_are_not_wallet_walls(text):
    assert kbd._credit_wall_signature(text) is None


def test_a_dead_worker_carrying_out_of_credits_is_marked(conn, monkeypatch):
    _run(conn, "t_wall")
    monkeypatch.setattr(kbd, "_worker_final_output",
                        lambda task_id, board=None: "Out of credits. Add credits to continue.")
    dead = kbd._classify_dead_worker(999999, "lock", 1, task_id="t_wall", board=None)
    assert dead.credit_wall == "out of credits"
    assert dead.event_payload.get("credit_wall") == "out of credits"


def test_a_dead_worker_without_it_is_not_marked(conn, monkeypatch):
    _run(conn, "t_plain")
    monkeypatch.setattr(kbd, "_worker_final_output",
                        lambda task_id, board=None: "Traceback ... KeyError: 'x'")
    dead = kbd._classify_dead_worker(999999, "lock", 1, task_id="t_plain", board=None)
    assert dead.credit_wall is None


# --- the pause itself --------------------------------------------------------

def test_a_pause_is_visible_until_it_expires(kanban_home):
    assert kbd._lane_pause("builder") is None
    entry = kbd._pause_lane("builder", "out of credits", 1800)
    assert entry is not None
    live = kbd._lane_pause("builder")
    assert live is not None and live["signature"] == "out of credits"

    # Rewriting the file with a past expiry is what a clock passing looks like.
    kbd._write_pauses({"builder": {"until": time.time() - 1, "since": 0,
                                   "signature": "out of credits", "seconds": 1800}})
    assert kbd._lane_pause("builder") is None


def test_pausing_twice_reports_only_the_first_as_new(kanban_home):
    """One event per window: the second dead worker in a fan-out must not add one."""
    assert kbd._pause_lane("builder", "out of credits", 1800) is not None
    n = 0
    for _ in range(20):
        if kbd._pause_lane("builder", "out of credits", 1800) is not None:
            n += 1
    assert n == 0


def test_a_working_lane_lifts_its_pause_early(kanban_home):
    kbd._pause_lane("builder", "out of credits", 1800)
    kbd._clear_lane_pause("builder")
    assert kbd._lane_pause("builder") is None


def test_the_pause_window_comes_from_config():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"provider_credits_pause_seconds": 120}}):
        assert kbd._provider_credit_pause_seconds() == 120
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"provider_credits_pause_seconds": 0}}):
        assert kbd._provider_credit_pause_seconds() == 0
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"provider_credits_pause_seconds": "soon"}}):
        assert kbd._provider_credit_pause_seconds() == 1800


# --- the tick ---------------------------------------------------------------

def _run(conn: sqlite3.Connection, task_id: str = "t_x", profile: str = "builder",
         started_at: int = 0) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, ?, 'running', ?)",
            (task_id, profile, started_at or int(time.time())))


def _tick_ready(conn, monkeypatch, title):
    path = Path(tempfile.mkdtemp(prefix="p56-ws-"))
    tid = kb.create_task(conn, title=title, body="Do the thing.",
                         assignee="builder", workspace_kind="dir",
                         workspace_path=str(path))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board=None: 0)
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn",
                        lambda task, ws, board=None: spawned.append(task.id) or 515151)
    return tid, spawned


def _event_kinds(conn, tid):
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]


def test_a_paused_lane_spawns_nothing_and_charges_nothing(conn, monkeypatch):
    tid, spawned = _tick_ready(conn, monkeypatch, "card on a broke lane")
    kbd._pause_lane("builder", "out of credits", 1800)

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert spawned == []
    assert [t for (t, _who, _until) in res.skipped_credits_paused] == [tid]
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == "ready"
    assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
                        (tid,)).fetchone()[0] == 0
    assert "spawn_refused_credits_paused" in _event_kinds(conn, tid)


def test_the_refusal_is_recorded_once_per_card_not_once_per_tick(conn, monkeypatch):
    tid, _spawned = _tick_ready(conn, monkeypatch, "card on a broke lane")
    kbd._pause_lane("builder", "out of credits", 1800)
    for _ in range(3):
        kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)
    assert _event_kinds(conn, tid).count("spawn_refused_credits_paused") == 1


def test_an_expired_pause_lets_the_lane_work_again(conn, monkeypatch):
    tid, spawned = _tick_ready(conn, monkeypatch, "card on a recovered lane")
    kbd._write_pauses({"builder": {"until": time.time() - 1, "since": 0,
                                   "signature": "out of credits", "seconds": 1800}})

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert spawned == [tid]
    assert res.skipped_credits_paused == []


def test_the_refusal_reaches_the_json_surface():
    """The pause has to appear in `kanban dispatch --json`, or nobody sees it."""
    from hermes_cli import kanban_ops
    res = kbd.DispatchResult()
    res.skipped_credits_paused.append(("t_x", "builder", 1799999999))
    payload = kanban_ops.dispatch_payload(res)
    assert payload["skipped_credits_paused"] == [
        {"task_id": "t_x", "assignee": "builder", "until": 1799999999}]


# --- the wiring: a dead worker's words reach the pause -----------------------

def test_a_dead_worker_carrying_out_of_credits_pauses_its_lane(conn, monkeypatch):
    from hermes_cli import kanban_db as _kb
    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kbd, "_worker_final_output",
                        lambda task_id, board=None: "credit limit is exhausted. Out of credits.")

    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="broke lane", assignee="builder")
    for i in range(3):
        pid = 81000 + i
        kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
        conn.execute("UPDATE tasks SET worker_pid=?, consecutive_failures=0 WHERE id=?",
                     (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, 1 << 8)  # WIFEXITED with code 1
        kbd.detect_crashed_workers(conn)

    live = kbd._lane_pause("builder")
    assert live is not None and live["signature"] in ("out of credits",
                                                     "credit limit is exhausted")
    events = _event_kinds(conn, tid)
    assert events.count("provider_credits_exhausted") == 1, events

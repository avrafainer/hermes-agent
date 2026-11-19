"""P5.5: a per-profile launch budget, and a ceiling on one card's retries.

The flat `max_sessions_per_profile_per_day` taught the wrong lesson. Measured on
2026-11-03, the Pi lane did 331 real launches on its busiest day while
`chief-of-staff` had taken twelve in a fortnight, so any single number is either a
wall on the busy lane or meaningless on the quiet one. The config now takes a map.

The second ceiling exists because a lane budget cannot see one card thrashing: on
2026-10-30 some batch cards were launched 9-11 times, and a lane-wide number only
notices that by refusing every other card on the board too.
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

BUDGET = {"personal-intelligence": 350, "independent-reviewer": 250, "_default": 60}


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


# --- resolving the lane budget ----------------------------------------------

def test_a_flat_int_still_means_every_profile():
    for profile in ("builder", "personal-intelligence", ""):
        assert kbd._daily_launch_cap(40, profile) == 40


def test_a_map_gives_named_lanes_their_own_number():
    assert kbd._daily_launch_cap(BUDGET, "personal-intelligence") == 350
    assert kbd._daily_launch_cap(BUDGET, "independent-reviewer") == 250


def test_an_unnamed_profile_falls_to_the_fallback_key():
    assert kbd._daily_launch_cap(BUDGET, "builder") == 60
    assert kbd._daily_launch_cap(BUDGET, "") == 60


def test_a_profile_actually_named_default_is_not_the_fallback():
    """This machine has a profile called `default`; it must not shadow everyone else."""
    caps = {"default": 300, "_default": 60}
    assert kbd._daily_launch_cap(caps, "default") == 300
    assert kbd._daily_launch_cap(caps, "builder") == 60


def test_a_map_without_a_fallback_leaves_other_lanes_uncapped():
    assert kbd._daily_launch_cap({"personal-intelligence": 350}, "builder") is None


def test_junk_in_the_map_is_ignored_not_crashed_on():
    assert kbd._daily_launch_cap({"builder": "lots", "default": 0}, "builder") is None


def test_the_dispatcher_reads_the_map_from_config():
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"max_sessions_per_profile_per_day": BUDGET}}):
        assert kbd._max_sessions_per_profile_per_day("personal-intelligence") == 350
        assert kbd._max_sessions_per_profile_per_day("nobody") == 60


# --- the per-card ceiling ----------------------------------------------------

def _run(conn: sqlite3.Connection, task_id: str = "t_x", profile: str = "builder",
         started_at: int = 0) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES (?, ?, 'running', ?)",
            (task_id, profile, started_at or int(time.time())))


def test_attempts_today_counts_one_card_and_today_only(conn):
    now = int(time.time())
    _run(conn, "t_a", started_at=now)
    _run(conn, "t_a", started_at=now - 30)
    _run(conn, "t_a", started_at=now - 3 * 86400)   # another day
    _run(conn, "t_b", started_at=now)               # another card
    assert kbd._attempts_today(conn, "t_a") == 2
    assert kbd._attempts_today(conn, "t_b") == 1
    assert kbd._attempts_today(conn, "t_never") == 0


@pytest.mark.parametrize("raw,expected", [(5, 5), ("5", 5), (None, None), (0, None),
                                          (-2, None), ("", None), ("often", None)])
def test_the_card_ceiling_is_coerced_and_neutral_when_unset(raw, expected):
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"max_attempts_per_card_per_day": raw}}):
        assert kbd._max_attempts_per_card_per_day() == expected


# --- the tick ---------------------------------------------------------------

def _tick_ready(conn, monkeypatch, title, kind=None, path=None):
    if path is None:
        path = Path(tempfile.mkdtemp(prefix="p55-ws-"))
    kw = {"workspace_kind": "dir", "workspace_path": str(path)}
    tid = kb.create_task(conn, title=title, body="Do the thing.",
                         assignee="builder", **kw)
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board=None: 0)
    spawned: list[str] = []
    monkeypatch.setattr(kbd, "_default_spawn",
                        lambda task, ws, board=None: spawned.append(task.id) or 515151)
    return tid, spawned


def test_a_card_at_its_attempt_ceiling_is_left_alone(conn, monkeypatch):
    now = int(time.time())
    tid, spawned = _tick_ready(conn, monkeypatch, "thrashing card")
    for i in range(5):
        _run(conn, tid, started_at=now - i)
    monkeypatch.setattr(kbd, "_max_attempts_per_card_per_day", lambda: 5)
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day", lambda profile="": 350)

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert spawned == []
    assert [(tid_, attempts) for (tid_, _who, attempts) in res.skipped_attempts_capped] == [(tid, 5)]
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert "spawn_refused_attempt_cap" in kinds
    # Refusal happens before the claim: the card is still ready and nothing is charged.
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == "ready"
    assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
                        (tid,)).fetchone()[0] == 5


def test_a_card_just_under_its_ceiling_runs(conn, monkeypatch):
    now = int(time.time())
    tid, spawned = _tick_ready(conn, monkeypatch, "card with room")
    for i in range(4):
        _run(conn, tid, started_at=now - i)
    monkeypatch.setattr(kbd, "_max_attempts_per_card_per_day", lambda: 5)
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day", lambda profile="": 350)

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert tid in spawned
    assert res.skipped_attempts_capped == []


def test_one_thrashing_card_does_not_stop_its_neighbours(conn, monkeypatch):
    """The whole point of a per-card ceiling: the lane budget stays generous."""
    now = int(time.time())
    hot, spawned = _tick_ready(conn, monkeypatch, "hot card")
    for i in range(6):
        _run(conn, hot, started_at=now - i)
    cool = kb.create_task(conn, title="cool card", body="Do the thing.",
                          assignee="builder", workspace_kind="dir",
                          workspace_path=str(Path(tempfile.mkdtemp(prefix="p55-ws-"))))
    monkeypatch.setattr(kbd, "_max_attempts_per_card_per_day", lambda: 5)
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day", lambda profile="": 350)

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert cool in spawned
    assert [t for (t, _w, _a) in res.skipped_attempts_capped] == [hot]


def test_a_lane_over_its_own_budget_is_named_separately(conn, monkeypatch):
    now = int(time.time())
    for i in range(3):
        _run(conn, "t_old", profile="builder", started_at=now - i)
    tid, spawned = _tick_ready(conn, monkeypatch, "one too many")
    monkeypatch.setattr(kbd, "_max_attempts_per_card_per_day", lambda: 5)
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day",
                        lambda profile="": kbd._daily_launch_cap(BUDGET, profile))
    # builder falls to `default` (60) in BUDGET, so 4 today is fine there...
    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)
    assert tid in spawned and res.skipped_session_capped == []

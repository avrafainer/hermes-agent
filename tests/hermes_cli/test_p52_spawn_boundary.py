"""P5.2: the spawn boundary as a budget.

Three things landed together because they are one hook seen from three sides:

* a **daily launch budget per profile** (``kanban.max_sessions_per_profile_per_day``),
  distinct from the concurrency cap - 615 sessions in twelve days was 51 launches
  a day, each paying a cold start;
* an **exit contract** on every card a profile is asked to execute, because 34
  workers finished work without transitioning the card and the board could not
  tell that from a card nobody picked up;
* ``task_runs.session_id``, so a card's spend can be looked up rather than guessed.

These tests pin the parts that can be checked without a live worker: the contract
insertion rule, the daily count, the config read, and the log trailer parser.
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


# --- exit contract -----------------------------------------------------------

def test_contract_is_appended_when_the_body_never_says_how_it_finishes():
    out = kb.ensure_exit_contract("Extract the figures from the 2025 return.", "t")
    assert out.startswith("Extract the figures from the 2025 return.")
    assert "## Exit contract" in out
    assert "kanban complete" in out


@pytest.mark.parametrize("marker", ["Done when: tests pass", "DONE-WHEN: x",
                                    "## Exit contract\n- claim it",
                                    "Acceptance criteria: 3 rows",
                                    "Stop when: the file is parsed"])
def test_an_existing_contract_is_never_duplicated(marker: str):
    body = f"Work item.\n\n{marker}\n"
    assert kb.ensure_exit_contract(body, "t") == body


def test_a_bodyless_card_still_names_its_task():
    out = kb.ensure_exit_contract("", "Rotate the worker logs")
    assert "## Task" in out and "Rotate the worker logs" in out
    assert "## Exit contract" in out


def test_contract_is_added_by_create_task_for_assigned_cards(conn):
    tid = kb.create_task(conn, title="Bare card", body="just do the thing",
                         assignee="builder")
    body = conn.execute("SELECT body FROM tasks WHERE id = ?", (tid,)).fetchone()["body"]
    assert "## Exit contract" in body


def test_unassigned_cards_are_left_alone(conn):
    tid = kb.create_task(conn, title="Memo", body="no profile runs this")
    body = conn.execute("SELECT body FROM tasks WHERE id = ?", (tid,)).fetchone()["body"]
    assert body == "no profile runs this"


# --- daily session budget ----------------------------------------------------

def _run(conn: sqlite3.Connection, profile: str, started_at: int) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at) "
            "VALUES ('t_x', ?, 'running', ?)", (profile, started_at))


def test_sessions_today_counts_this_profile_and_today_only(conn):
    now = int(time.time())
    _run(conn, "builder", now)
    _run(conn, "builder", now - 60)
    _run(conn, "builder", now - 3 * 86400)   # three days ago
    _run(conn, "researcher", now)            # another profile
    assert kbd._sessions_started_today(conn, "builder") == 2
    assert kbd._sessions_started_today(conn, "researcher") == 1
    assert kbd._sessions_started_today(conn, "never-used") == 0


@pytest.mark.parametrize("raw,expected", [
    (4, 4), ("4", 4), (None, None), (0, None), (-1, None), ("", None), ("abc", None),
])
def test_cap_config_is_coerced_and_neutral_when_unset(raw, expected):
    with patch("hermes_cli.config.load_config_readonly",
               return_value={"kanban": {"max_sessions_per_profile_per_day": raw}}):
        assert kbd._max_sessions_per_profile_per_day() == expected


def test_a_broken_config_read_does_not_stop_the_dispatcher():
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("boom")):
        assert kbd._max_sessions_per_profile_per_day() is None


# --- session id attribution --------------------------------------------------

def test_session_id_is_read_from_the_worker_log_trailer(monkeypatch):
    log = "worker output...\nsummary: did the thing\nsession_id: 20261103_144012_ab12cd\n"
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *a, **k: log)
    assert kbd._session_id_from_log("t_x") == "20261103_144012_ab12cd"


def test_missing_log_or_trailer_yields_empty_not_an_error(monkeypatch):
    monkeypatch.setattr(kbd._kb, "read_worker_log",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    assert kbd._session_id_from_log("t_x") == ""
    monkeypatch.setattr(kbd._kb, "read_worker_log", lambda *a, **k: "no trailer here")
    assert kbd._session_id_from_log("t_x") == ""


def test_task_runs_has_the_session_id_column(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(task_runs)")}
    assert "session_id" in cols


# --- the tick itself, not just the pieces ------------------------------------

def test_the_sixth_dispatch_is_refused_and_the_card_says_why(conn, monkeypatch):
    """A whole dispatch tick: five sessions used, a sixth refused, reason on the card."""
    now = int(time.time())
    for i in range(5):
        _run(conn, "builder", now - i)
    tid = kb.create_task(conn, title="one more job", body="Do the thing.",
                         assignee="builder")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day", lambda: 5)
    # The host's real memory pressure and other boards' workers must not decide
    # this test; both are environmental, and either can short-circuit the tick.
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board=None: 0)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 424242)

    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)

    assert spawned == [], "a capped profile must not be launched again"
    assert [row[0] for row in res.skipped_session_capped] == [tid]
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,))]
    assert "spawn_refused_session_cap" in kinds
    # Claimed nothing, charged nothing: the card is still ready with no run opened.
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == "ready"
    assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (tid,)).fetchone()[0] == 0


def test_under_the_cap_the_same_tick_spawns(conn, monkeypatch):
    now = int(time.time())
    for i in range(2):
        _run(conn, "builder", now - i)
    # A real directory: the scratch workspace root does not exist in a temp home,
    # and a workspace that cannot be resolved is a spawn failure, not a cap question.
    ws = Path(tempfile.mkdtemp(prefix="p52-ws-"))
    tid = kb.create_task(conn, title="fits", body="Do the thing.", assignee="builder",
                         workspace_kind="dir", workspace_path=str(ws))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    monkeypatch.setattr(kbd, "_max_sessions_per_profile_per_day", lambda: 5)
    # The host's real memory pressure and other boards' workers must not decide
    # this test; both are environmental, and either can short-circuit the tick.
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board=None: 0)
    spawned = []
    monkeypatch.setattr(kbd, "_default_spawn", lambda task, ws, board=None: spawned.append(task.id) or 434343)

    calls = {}
    _orig_claim = kb.claim_task
    def _spy(conn2, task_id2, **kw):
        out = _orig_claim(conn2, task_id2, **kw)
        calls[task_id2] = "claimed" if out is not None else "claim_failed"
        return out
    monkeypatch.setattr(kb, "claim_task", _spy)
    res = kbd.dispatch_once(conn, max_spawn=3, max_in_progress=3, board=None)
    assert tid in spawned
    assert res.skipped_session_capped == []


# --- cost per card -----------------------------------------------------------

from hermes_cli import kanban_cost as kcost
from hermes_cli import kanban as kanban_cli


@pytest.fixture
def priced_home(tmp_path, monkeypatch):
    home = tmp_path / "priced"
    (home / "profiles" / "builder").mkdir(parents=True)
    for db, cost in ((home / "state.db", 0.0), (home / "profiles" / "builder" / "state.db", 0.21)):
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE session_model_usage (session_id TEXT, model TEXT,
            api_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER,
            cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER,
            estimated_cost_usd REAL, actual_cost_usd REAL)""")
        con.execute("INSERT INTO session_model_usage VALUES ('s_live', 'm', 4, 0,0,0,0,0, 0.19, 0.21)")
        con.commit(); con.close()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_a_session_is_priced_from_whichever_ledger_holds_it(priced_home):
    cost, calls = kcost.session_cost("s_live")
    assert calls == 4
    assert cost == pytest.approx(0.21)   # billed where known, not the estimate


def test_an_unpriced_card_reports_unknown_not_zero():
    total, priced, unpriced = kcost.card_cost(["nope", None])
    assert total is None and priced == 0 and unpriced == 2
    assert kcost.format_cost(None) == "cost unknown"


def test_card_cost_totals_only_the_sessions_it_could_price(priced_home):
    total, priced, unpriced = kcost.card_cost(["s_live", "gone", "s_live"])
    assert total == pytest.approx(0.42) and priced == 2 and unpriced == 1


def test_kanban_show_prints_what_the_card_cost(conn, kanban_home, capsys):
    """The surface Avra actually reads: `kanban show` names the money."""
    import argparse
    now = int(time.time())
    tid = kb.create_task(conn, title="priced card", body="Work.", assignee="builder")
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, "
            "outcome, session_id) VALUES (?, 'builder', 'done', ?, ?, 'completed', 's_paid')",
            (tid, now - 60, now))
    con = sqlite3.connect(kanban_home / "profiles" / "builder" / "state.db") \
        if (kanban_home / "profiles" / "builder").exists() else None
    if con is None:
        (kanban_home / "profiles" / "builder").mkdir(parents=True)
        con = sqlite3.connect(kanban_home / "profiles" / "builder" / "state.db")
    con.execute("""CREATE TABLE session_model_usage (session_id TEXT, model TEXT,
        api_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER,
        cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER,
        estimated_cost_usd REAL, actual_cost_usd REAL)""")
    con.execute("INSERT INTO session_model_usage VALUES ('s_paid','m',3,0,0,0,0,0,0.04,0.05)")
    con.commit(); con.close()

    rc = kanban_cli._cmd_show(argparse.Namespace(task_id=tid, json=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert "$0.050" in out or "$0.05" in out, out
    assert "cost:" in out

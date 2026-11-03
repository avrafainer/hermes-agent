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

"""P5.4: a card's runtime cap and priority follow its work type at creation.

Priority was decorative on this board - 815 of 847 tasks sat at 0 - and a cap was
rare (29 of 847). These tests pin the behaviour that makes a type useful without
letting it overrule a caller: configured types fill what was left blank, explicit
arguments always win, and a type nobody configured changes nothing at all.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_work_types import apply_work_type, configured_types

WORK_TYPES_YAML = """
kanban:
  work_types:
    build:
      max_runtime_seconds: 3600
      priority: 40
    lookup:
      max_runtime_seconds: 300
      priority: 10
"""


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text(WORK_TYPES_YAML, encoding="utf-8")
    kb.init_db()
    conn = kbc.connect()
    yield home, conn
    conn.close()


def _row(conn, task_id):
    return conn.execute(
        "SELECT priority, max_runtime_seconds, work_type FROM tasks WHERE id = ?",
        (task_id,)).fetchone()


# --- what creation does ------------------------------------------------------

def test_a_configured_type_fills_cap_and_priority(board):
    _home, conn = board
    tid = kb.create_task(conn, title="Build the thing", assignee="builder",
                         work_type="build")
    priority, cap, work_type = _row(conn, tid)
    assert work_type == "build"
    assert cap == 3600
    assert priority == 40


def test_an_explicit_argument_beats_the_type(board):
    _home, conn = board
    tid = kb.create_task(conn, title="Quick build", assignee="builder",
                         work_type="build", max_runtime_seconds=90, priority=99)
    priority, cap, _ = _row(conn, tid)
    assert (cap, priority) == (90, 99)


def test_an_unknown_type_changes_nothing(board):
    _home, conn = board
    tid = kb.create_task(conn, title="Something else", assignee="builder",
                         work_type="unheard-of")
    priority, cap, work_type = _row(conn, tid)
    assert cap is None
    assert priority == 0
    assert work_type == "unheard-of"   # recorded, just not priced


def test_a_card_with_no_type_behaves_exactly_as_before(board):
    _home, conn = board
    tid = kb.create_task(conn, title="Plain card", assignee="builder")
    priority, cap, work_type = _row(conn, tid)
    assert (priority, cap, work_type) == (0, None, None)


def test_priority_zero_is_a_decision_not_a_blank(board):
    """A caller who says 0 has said something; the type must not overwrite it."""
    _home, conn = board
    tid = kb.create_task(conn, title="Unimportant build", assignee="builder",
                         work_type="build", priority=0)
    assert _row(conn, tid)[0] == 0


# --- the resolution rule on its own -----------------------------------------

def test_resolution_leaves_blanks_alone_when_no_type_is_named():
    assert apply_work_type(None, priority=None, max_runtime_seconds=None,
                           config={"kanban": {"work_types": {}}}) == (None, None)


def test_an_empty_config_is_a_working_state():
    assert configured_types({}) == {}
    assert apply_work_type("build", priority=None, max_runtime_seconds=None,
                           config={}) == (None, None)


def test_a_malformed_entry_is_ignored_rather_than_crashing():
    cfg = {"kanban": {"work_types": {"build": "3600"}}}
    assert configured_types(cfg) == {}
    assert apply_work_type("build", priority=None, max_runtime_seconds=None,
                           config=cfg) == (None, None)


def test_junk_values_do_not_become_a_cap():
    cfg = {"kanban": {"work_types": {"build": {"max_runtime_seconds": "soon",
                                               "priority": "high"}}}}
    assert apply_work_type("build", priority=None, max_runtime_seconds=None,
                           config=cfg) == (None, None)


# --- the column survives a board that predates it ---------------------------

def test_a_legacy_board_gains_the_column_without_touching_its_cards(tmp_path, monkeypatch):
    home = tmp_path / "legacy"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = home / "kanban.db"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT,
        assignee TEXT, status TEXT, priority INTEGER, created_by TEXT, created_at INTEGER,
        started_at INTEGER, completed_at INTEGER, workspace_kind TEXT, workspace_path TEXT,
        claim_lock TEXT, claim_expires INTEGER, tenant TEXT)""")
    con.execute("""INSERT INTO tasks (id,title,assignee,status,priority,workspace_kind)
        VALUES ('t_old','Old card','builder','done',0,'scratch')""")
    con.commit(); con.close()

    kb.init_db()
    conn = kbc.connect()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
    assert "work_type" in cols
    priority, cap, work_type = _row(conn, "t_old")
    assert (priority, cap, work_type) == (0, None, None)
    conn.close()


def test_the_cli_can_name_a_type_and_the_cap_lands(board):
    """The path Avra actually types: `kanban create --work-type build`."""
    home, conn = board
    conn.close()
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", "create", "CLI build",
         "--assignee", "builder", "--work-type", "build"],
        check=False, capture_output=True, text=True, env={**__import__("os").environ,
                                             "HERMES_HOME": str(home)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    con = sqlite3.connect(home / "kanban.db")
    row = con.execute("SELECT priority, max_runtime_seconds, work_type FROM tasks "
                      "WHERE title = 'CLI build'").fetchone()
    con.close()
    assert row == (40, 3600, "build")

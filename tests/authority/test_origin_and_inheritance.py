"""Origin and inheritance: a card gets authority only from its own origin or an attributable parent.

Every test runs against a temporary HERMES_HOME (fixtures from test_authority_gate.py are
re-used by import). Nothing touches the live board, the live authority store or the gateway.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from tests.authority.test_authority_gate import _board, _grant, _user_msg, auth, home


def _log_lines(home):
    path = home / "state" / "authority-log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _links(home, rows):
    con = sqlite3.connect(home / "kanban.db")
    try:
        con.execute("create table if not exists task_links (parent_id text, child_id text)")
        con.executemany("insert into task_links values (?,?)", rows)
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------- every binding is attributed

def test_a_direct_bind_is_logged_with_its_path(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_plain", None)])
    auth.bind("t_plain", gid)
    lines = [l for l in _log_lines(home) if l["action"] == "bind" and l["task"] == "t_plain"]
    assert lines and lines[-1]["grant"] == gid and lines[-1]["reason"] == "bound via direct"


def test_a_named_path_is_recorded_in_the_bind_line(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_named", None)])
    auth.bind("t_named", gid, via="inherit_from_parent")
    assert _log_lines(home)[-1]["reason"] == "bound via inherit_from_parent"


# ---------------------------------------------------------------- inheritance needs attribution

def test_bind_child_inherits_from_an_attributable_parent(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_parent", "s1")])               # the parent's own origin is s1
    auth.bind("t_parent", gid)
    con = sqlite3.connect(home / "kanban.db")
    con.execute("create table task_links (parent_id text, child_id text)")
    con.commit()
    inherited = auth.bind_child(con, "t_child", "t_parent")
    con.close()
    assert inherited == gid
    assert auth.check("t_child", "dispatch")[0] == "allow"


def test_bind_child_passes_nothing_from_an_unattributable_parent(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_hand", None)])                 # no origin: a bare bind only
    auth.bind("t_hand", gid)
    con = sqlite3.connect(home / "kanban.db")
    con.execute("create table task_links (parent_id text, child_id text)")
    con.commit()
    assert auth.bind_child(con, "t_kid", "t_hand") is None
    con.close()
    assert auth.resolve("t_kid")[0] is None


def test_dispatch_inherits_through_an_attributable_parent_and_logs_it(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_root", "s1"), ("t_leaf", None)])
    _links(home, [("t_root", "t_leaf")])
    auth.bind("t_root", gid)
    assert auth.check("t_leaf", "dispatch")[0] == "allow"
    inherited = [l for l in _log_lines(home) if l["task"] == "t_leaf" and "inherited" in l["reason"]]
    assert inherited and inherited[-1]["reason"] == f"inherited {gid} from parent t_root"


def test_a_hand_bound_grant_is_not_laundered_down_a_tree(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_hand", None), ("t_leaf", None)])
    _links(home, [("t_hand", "t_leaf")])
    auth.bind("t_hand", gid)                         # bound, but nothing makes it attributable
    assert auth.check("t_leaf", "dispatch")[0] == "needs_approval"
    assert auth.resolve("t_leaf")[0] is None


# ---------------------------------------------------------------- no path binds another card's grant

def test_an_unrelated_card_cannot_borrow_another_cards_grant(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_owner", "s1"), ("t_stranger", None)])
    assert auth.derive_origin_grant("t_owner", "s1")["grant_id"] == gid
    # No task_links row joins them: the stranger has no origin and no parent.
    assert auth.check("t_stranger", "dispatch")[0] == "needs_approval"
    assert auth.resolve("t_stranger")[0] is None


def test_a_grant_is_passed_on_only_along_the_cards_own_links(home, auth):
    _grant(auth, home)
    _board(home, [("t_owner", "s1"), ("t_sibling", None), ("t_other", None)])
    auth.derive_origin_grant("t_owner", "s1")
    _links(home, [("t_other", "t_sibling")])         # a link to a card that holds nothing
    assert auth.check("t_sibling", "dispatch")[0] == "needs_approval"


# ---------------------------------------------------------------- refusal at creation

def _create(**kw):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    kb.init_db()
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title=kw.pop("title", "card"), **kw)


def _task_count(home):
    con = sqlite3.connect(home / "kanban.db")
    try:
        return con.execute("select count(*) from tasks").fetchone()[0]
    finally:
        con.close()


def test_creation_refuses_a_card_with_no_origin_and_names_the_origin(home, auth):
    from hermes_cli.authority import OriginRefused
    _grant(auth, home)
    with pytest.raises(OriginRefused) as exc:
        _create(title="orphan")
    assert "no origin session" in str(exc.value)
    assert _task_count(home) == 0, "a refused card must not be left behind on the board"


def test_creation_with_an_origin_that_holds_a_grant_binds_at_creation(home, auth):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    gid = _grant(auth, home)
    task_id = _create(title="from chat", session_id="s1")
    assert auth.resolve(task_id)[0]["grant_id"] == gid
    assert auth.check(task_id, "dispatch")[0] == "allow"


def test_a_worker_created_child_of_an_attributable_parent_needs_no_binding(home, auth):
    gid = _grant(auth, home)
    parent = _create(title="parent", session_id="s1")
    child = _create(title="child", parents=[parent])
    assert auth.resolve(child)[0]["grant_id"] == gid
    assert auth.check(child, "dispatch")[0] == "allow"


def test_a_child_of_an_unattributable_parent_is_refused_at_creation(home, auth):
    from hermes_cli.authority import OriginRefused
    gid = _grant(auth, home)
    parent = _create(title="triaged", triage=True)  # exempt at creation, no origin
    auth.bind(parent, gid)                           # a hand bind does not make it attributable
    before = _task_count(home)
    with pytest.raises(OriginRefused):
        _create(title="kid", parents=[parent])
    assert _task_count(home) == before


def test_triage_cards_are_exempt_from_the_origin_refusal(home, auth):
    _grant(auth, home)
    task_id = _create(title="needs a specifier", triage=True)
    assert task_id


def test_only_enforce_mode_refuses_at_creation(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_AUTHORITY_MODE", "shadow")
    _grant(auth, home)
    assert _create(title="shadow card")              # refusal is an enforce-mode behaviour


def test_refusal_names_the_real_cause_when_a_session_holds_no_grant(home, auth):
    from hermes_cli.authority import OriginRefused
    _grant(auth, home)
    with pytest.raises(OriginRefused) as exc:
        _create(title="unlicensed", session_id="s-without-grant")
    msg = str(exc.value)
    assert "origin session s-without-grant holds no live dispatch grant" in msg
    assert "no origin session (session_id was not supplied)" not in msg
    assert _task_count(home) == 0

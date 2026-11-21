"""Authority gate: the tests that would have stopped 2026-11-20.

Every test runs against a temporary HERMES_HOME. Nothing here touches the live board,
the live authority store, or the running gateway.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import time

import pytest


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "state").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HERMES_STATE_DB", str(h / "state.db"))
    monkeypatch.setenv("HERMES_AUTHORITY_DB", str(h / "state" / "authority.db"))
    monkeypatch.setenv("HERMES_AUTHORITY_KEY", "test-key")
    monkeypatch.setenv("HERMES_AUTHORITY_MODE", "enforce")
    for marker in ("HERMES_KANBAN_TASK", "HERMES_CRON_JOB", "HERMES_ONESHOT",
                   "HERMES_DELEGATE_DEPTH", "HERMES_WORKER"):
        monkeypatch.delenv(marker, raising=False)

    con = sqlite3.connect(h / "state.db")
    con.executescript("""
      create table sessions (id text primary key, title text, started_at integer, source text,
                             input_tokens integer, output_tokens integer);
      create table messages (id integer primary key autoincrement, session_id text not null,
                             role text not null, content text, timestamp integer not null);
    """)
    now = int(time.time())
    con.execute("insert into sessions values ('s1','chat',?, 'desktop',0,0)", (now,))
    con.execute("insert into sessions values ('s2','worker',?, 'kanban',0,0)", (now,))
    con.execute("insert into messages (session_id,role,content,timestamp) values (?,?,?,?)",
                ("s1", "user", "yes, go ahead and fix the green dot", now - 30))
    con.execute("insert into messages (session_id,role,content,timestamp) values (?,?,?,?)",
                ("s1", "assistant", "done", now - 20))
    con.execute("insert into messages (session_id,role,content,timestamp) values (?,?,?,?)",
                ("s2", "user", "# kanban task", now - 10))
    con.execute("insert into messages (session_id,role,content,timestamp) values (?,?,?,?)",
                ("s1", "user", "old approval", now - 99999))
    con.commit()
    con.close()
    return h


@pytest.fixture()
def auth(home):
    from hermes_cli import authority
    return authority


def _user_msg(home, sid="s1"):
    con = sqlite3.connect(home / "state.db")
    try:
        return con.execute(
            "select id from messages where session_id=? and role='user' "
            "order by timestamp desc limit 1", (sid,)).fetchone()[0]
    finally:
        con.close()


def _grant(auth, home, **kw):
    args = dict(message_id=str(_user_msg(home)), session_id="s1",
                scope_cite="fix the green dot", actions=("dispatch", "promote", "unblock"))
    args.update(kw)
    return auth.mint(**args)


# ---------------------------------------------------------------- modes

def test_mode_off_allows_everything(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_AUTHORITY_MODE", "off")
    assert auth.check("t_nothing", "dispatch")[0] == "allow"


def test_shadow_computes_but_never_blocks(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_AUTHORITY_MODE", "shadow")
    verdict, reason = auth.check("t_nothing", "dispatch")
    assert verdict == "allow" and "would have been" in reason


def test_enforce_blocks_a_card_with_no_grant(home, auth):
    verdict, reason = auth.check("t_nothing", "dispatch")
    assert verdict == "needs_approval"
    assert "no grant" in reason


# ---------------------------------------------------------------- minting

def test_mint_refused_inside_a_worker(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_51a7358d")
    with pytest.raises(auth.MintRefused, match="unattended"):
        _grant(auth, home)


def test_mint_refused_for_cron(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_JOB", "pm-review-daily")
    with pytest.raises(auth.MintRefused):
        _grant(auth, home)


def test_mint_refused_when_message_is_not_a_user_turn(home, auth):
    con = sqlite3.connect(home / "state.db")
    mid = con.execute("select id from messages where role='assistant'").fetchone()[0]
    con.close()
    with pytest.raises(auth.MintRefused, match="not a user turn"):
        auth.mint(message_id=str(mid), session_id="s1", scope_cite="", actions=("dispatch",))


def test_mint_refused_for_a_worker_session(home, auth):
    with pytest.raises(auth.MintRefused, match="not an interactive surface"):
        auth.mint(message_id=str(_user_msg(home, "s2")), session_id="s2",
                  scope_cite="", actions=("dispatch",))


def test_mint_refused_for_a_stale_message(home, auth):
    con = sqlite3.connect(home / "state.db")
    mid = con.execute("select id from messages where content='old approval'").fetchone()[0]
    con.close()
    with pytest.raises(auth.MintRefused, match="older than"):
        auth.mint(message_id=str(mid), session_id="s1", scope_cite="old approval",
                  actions=("dispatch",))


def test_mint_refused_when_the_quoted_scope_is_not_in_the_message(home, auth):
    with pytest.raises(auth.MintRefused, match="not present"):
        _grant(auth, home, scope_cite="rewrite the whole desktop app")


def test_mint_records_provenance(home, auth):
    gid = _grant(auth, home)
    con = auth.connect()
    row = con.execute("select * from grants where grant_id=?", (gid,)).fetchone()
    con.close()
    assert row["provenance_session"] == "s1"
    assert row["source_platform"] == "desktop"
    assert row["granted_by"] == "avra"


# ---------------------------------------------------------------- checks

def test_bound_grant_allows_a_listed_action(home, auth):
    gid = _grant(auth, home)
    auth.bind("t_ok", gid)
    assert auth.check("t_ok", "dispatch")[0] == "allow"


def test_action_outside_the_grant_is_refused(home, auth):
    gid = _grant(auth, home, actions=("dispatch",))
    auth.bind("t_ok", gid)
    assert auth.check("t_ok", "delegate")[0] == "needs_approval"


def test_revocation_is_transitive(home, auth):
    root = _grant(auth, home)
    child = auth.mint(message_id=str(_user_msg(home)), session_id="s1", scope_cite="fix",
                      actions=("dispatch",), parent_grant=root)
    auth.bind("t_child", child)
    assert auth.check("t_child", "dispatch")[0] == "allow"
    auth.revoke(root)
    verdict, reason = auth.check("t_child", "dispatch")
    assert verdict == "needs_approval" and "revoked" in reason


def test_expiry_is_enforced(home, auth):
    # An already-expired grant cannot be bound at all.
    gid = _grant(auth, home, expires_at=int(time.time()) - 5)
    with pytest.raises(auth.BindRefused, match="expired"):
        auth.bind("t_old", gid)

def test_a_grant_that_expires_after_binding_stops_authorising(home, auth):
    gid = _grant(auth, home, expires_at=int(time.time()) + 60)
    auth.bind("t_live", gid)
    assert auth.check("t_live", "dispatch")[0] == "allow"
    _c = auth.connect()
    with _c:
        _c.execute("update grants set expires_at=? where grant_id=?",
                   (int(time.time()) - 5, gid))
    assert auth.check("t_live", "dispatch")[0] != "allow"


def test_a_forged_grant_row_is_rejected(home, auth):
    gid = _grant(auth, home)
    auth.bind("t_ok", gid)
    con = sqlite3.connect(home / "state" / "authority.db")
    con.execute("update grants set actions=? where grant_id=?",
                (json.dumps(["dispatch", "delegate", "close"]), gid))
    con.commit()
    con.close()
    verdict, reason = auth.check("t_ok", "dispatch")
    assert verdict == "needs_approval" and "integrity" in reason


def test_a_forged_binding_is_rejected(home, auth):
    gid = _grant(auth, home)
    con = sqlite3.connect(home / "state" / "authority.db")
    con.execute("insert into task_grants values ('t_forged', ?, 'deadbeef')", (gid,))
    con.commit()
    con.close()
    assert auth.check("t_forged", "dispatch")[0] == "needs_approval"


def test_child_cannot_widen_actions(home, auth):
    root = _grant(auth, home, actions=("dispatch", "promote"))
    child = auth.mint(message_id=str(_user_msg(home)), session_id="s1", scope_cite="fix",
                      actions=("dispatch", "promote", "delegate"), parent_grant=root)
    auth.bind("t_child", child)
    g, _ = auth.resolve("t_child")
    assert "delegate" not in g["actions"]


# ---------------------------------------------------------------- spend

def test_descendants_share_one_allowance(home, auth):
    gid = _grant(auth, home, spend_limit=1.00)
    child = auth.mint(message_id=str(_user_msg(home)), session_id="s1", scope_cite="fix",
                      actions=("dispatch",), parent_grant=gid)
    auth.bind("t_a", gid)
    auth.bind("t_b", child)
    assert auth.charge("t_a", 0.60)[0] is True
    ok, why = auth.charge("t_b", 0.60)
    assert ok is False and "exceed" in why
    assert auth.charge("t_b", 0.30)[0] is True


# ---------------------------------------------------------------- incident replay

def test_incident_11_44_a_triage_card_could_not_have_been_promoted(home, auth):
    """The card PM created at 11:43:53 had no grant. Promotion must fail, which is the
    event that made it executable."""
    verdict, _ = auth.check("t_c2cf2213", "promote")
    assert verdict == "needs_approval"


def test_incident_11_44_a_ready_card_could_not_have_been_dispatched(home, auth):
    """The card PM created at 11:44:02 landed in `ready`. Dispatch must refuse it."""
    verdict, _ = auth.check("t_d4c09029", "dispatch")
    assert verdict == "needs_approval"


def test_incident_children_inherit_nothing_they_were_not_given(home, auth):
    gid = _grant(auth, home, actions=("dispatch", "create_child"))
    auth.bind("t_parent", gid)
    for i in range(6):
        child = auth.mint(message_id=str(_user_msg(home)), session_id="s1",
                          scope_cite="fix", actions=("dispatch",), parent_grant=gid)
        auth.bind(f"t_child{i}", child)
    for i in range(6):
        assert auth.check(f"t_child{i}", "dispatch")[0] == "allow"
    auth.revoke(gid)
    for i in range(6):
        assert auth.check(f"t_child{i}", "dispatch")[0] == "needs_approval"


# ---------------------------------------------------------------- affirmative authorisation

def _add_msg(home, text, role="user", sid="s1", age=10):
    con = sqlite3.connect(home / "state.db")
    con.execute("insert into messages (session_id,role,content,timestamp) values (?,?,?,?)",
                (sid, role, text, int(time.time()) - age))
    con.commit()
    mid = con.execute("select max(id) from messages").fetchone()[0]
    con.close()
    return str(mid)


@pytest.mark.parametrize("text,why", [
    ("should we fix the green dot?", "question"),
    ("no, do not fix the green dot", "negation"),
    ("i was thinking about the green dot fix", "no affirmative"),
])
def test_a_mention_is_not_an_authorisation(home, auth, text, why):
    mid = _add_msg(home, text)
    with pytest.raises(auth.MintRefused, match="not affirmative"):
        auth.mint(message_id=mid, session_id="s1", scope_cite="green dot",
                  actions=("dispatch",))


def test_an_affirmative_message_authorises(home, auth):
    mid = _add_msg(home, "yes go ahead and fix the green dot")
    gid = auth.mint(message_id=mid, session_id="s1", scope_cite="green dot",
                    actions=("dispatch",))
    assert gid.startswith("g_")


# ---------------------------------------------------------------- inherited limits

def test_fanout_cap_stops_a_decomposer(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_AUTHORITY_FANOUT", "2")
    gid = _grant(auth, home)
    auth.bind("t_root", gid)
    con = sqlite3.connect(home / "kanban.db")
    con.executescript("create table if not exists task_links (parent_id text, child_id text);"
                      "create table if not exists tasks (id text primary key, status text);")
    for i in range(2):
        con.execute("insert into task_links values ('t_root', ?)", (f"t_k{i}",))
        con.execute("insert into tasks values (?, 'todo')", (f"t_k{i}",))
    con.commit()
    ok, why = auth.admit_child(con, "t_root")
    con.close()
    assert ok is False and "fan-out" in why


def test_concurrency_cap_stops_live_children(home, auth, monkeypatch):
    monkeypatch.setenv("HERMES_AUTHORITY_CONCURRENCY", "1")
    gid = _grant(auth, home)
    auth.bind("t_root", gid)
    con = sqlite3.connect(home / "kanban.db")
    con.executescript("create table if not exists task_links (parent_id text, child_id text);"
                      "create table if not exists tasks (id text primary key, status text);")
    con.execute("insert into task_links values ('t_root','t_k0')")
    con.execute("insert into tasks values ('t_k0','running')")
    con.commit()
    ok, why = auth.admit_child(con, "t_root")
    con.close()
    assert ok is False and "concurrency" in why


def test_bind_child_inherits_the_parents_grant(home, auth):
    # The parent must be attributable: its grant was issued in its own origin session.
    # A bare bind on a parent with no origin passes nothing (see test_origin_and_inheritance).
    gid = _grant(auth, home)
    _board(home, [("t_parent", "s1")])
    auth.bind("t_parent", gid)
    con = sqlite3.connect(home / "kanban.db")
    con.executescript("create table if not exists task_links (parent_id text, child_id text);")
    con.commit()
    inherited = auth.bind_child(con, "t_child", "t_parent")
    con.close()
    assert inherited == gid
    assert auth.check("t_child", "dispatch")[0] == "allow"


def test_bind_child_of_an_ungranted_card_stays_ungranted(home, auth):
    con = sqlite3.connect(home / "kanban.db")
    con.executescript("create table if not exists task_links (parent_id text, child_id text);"
                      "create table if not exists tasks (id text primary key, status text);")
    con.commit()
    assert auth.bind_child(con, "t_orphan", "t_nogrant") is None
    con.close()
    assert auth.check("t_orphan", "dispatch")[0] == "needs_approval"


def test_sandbox_profile_denies_the_key_and_the_store(home, auth):
    (home / "state" / ".authority-key").write_bytes(b"k")
    auth.connect().close()  # materialise authority.db so it appears in the profile
    prof = auth.sandbox_profile()
    assert "(allow default)" in prof
    assert ".authority-key" in prof and "authority.db" in prof
    for line in prof.splitlines():
        if line.startswith("(deny"):
            assert "/private/" in line or str(home.resolve()) in line


# ---------------------------------------------------------------- spawn-path integration

def test_worker_env_never_inherits_the_signing_key(home, auth, monkeypatch):
    from hermes_cli import kanban_db_dispatch as D
    monkeypatch.setattr(auth, "settings", lambda: {**auth._DEFAULTS, "sandbox_workers": False})
    env = {"HERMES_AUTHORITY_KEY": "leaked", "PATH": "/bin"}
    _cmd, out_env, err = D._authority_sandbox(["/bin/echo", "hi"], env, None)
    assert err is None
    assert "HERMES_AUTHORITY_KEY" not in out_env
    assert out_env["PATH"] == "/bin"


def test_sandbox_failure_fails_closed(home, auth, monkeypatch):
    from hermes_cli import kanban_db_dispatch as D
    monkeypatch.setattr(auth, "settings", lambda: {**auth._DEFAULTS, "sandbox_workers": True})
    monkeypatch.setattr(auth, "sandbox_selftest", lambda: (False, "denied-nothing"))
    _cmd, _env, err = D._authority_sandbox(["/bin/echo"], {}, None)
    assert err and "self-test failed" in err


def test_sandbox_wraps_the_command_when_it_passes(home, auth, monkeypatch):
    from hermes_cli import kanban_db_dispatch as D
    monkeypatch.setattr(auth, "settings", lambda: {**auth._DEFAULTS, "sandbox_workers": True})
    monkeypatch.setattr(auth, "sandbox_selftest", lambda: (True, "denied"))
    cmd, _env, err = D._authority_sandbox(["/bin/echo", "hi"], {}, None)
    assert err is None and cmd[0] == "/usr/bin/sandbox-exec" and cmd[-1] == "hi"


def test_completing_a_parent_with_children_needs_authority(home, auth):
    from hermes_cli import kanban_db as KB
    con = sqlite3.connect(":memory:")
    con.execute("create table task_links (parent_id text, child_id text)")
    con.execute("insert into task_links values ('t_p','t_c')")
    gid = _grant(auth, home, actions=("dispatch",))
    auth.bind("t_p", gid)
    reason = KB._authority_completion_check(con, "t_p")
    assert reason and "complete_with_children" in reason
    con.close()


def test_completing_a_childless_card_is_never_blocked(home, auth):
    from hermes_cli import kanban_db as KB
    con = sqlite3.connect(":memory:")
    con.execute("create table task_links (parent_id text, child_id text)")
    assert KB._authority_completion_check(con, "t_leaf") is None
    con.close()


# ---------------------------------------------------------------- authorisation classes
# Authorisation is decided on the sentence that carries the scope. These are the forms
# that must not be mistaken for permission, and the one that must be.

@pytest.mark.parametrize("text,scope,want,cls", [
    ("I do not need another design proposal. Complete the implementation, test it.",
     "complete the implementation", True, "directive"),
    ("yes, go ahead and complete the implementation",
     "complete the implementation", True, "directive"),
    ("Please wire the sandbox into the spawn path", "wire the sandbox", True, "directive"),
    ("> complete the implementation", "complete the implementation", False, "quoted"),
    ("- complete the implementation (PM review 2026-11-20)",
     "complete the implementation", False, "pasted-report"),
    ("Summary: complete the implementation next", "complete the implementation",
     False, "pasted-report"),
    ("should we complete the implementation?", "complete the implementation", False, "question"),
    ("we could complete the implementation later", "complete the implementation",
     False, "hypothetical"),
    ("what if we complete the implementation first", "complete the implementation",
     False, "hypothetical"),
    ("don't complete the implementation", "complete the implementation", False, "negation"),
    ("no, do not complete the implementation", "complete the implementation",
     False, "negation"),
    ("commit the change but do not complete the implementation",
     "complete the implementation", False, "negation"),
    ("The PM completed the implementation", "completed the implementation", False, "narration"),
])
def test_authorisation_classes(home, auth, text, scope, want, cls):
    ok, got = auth.assess(text, scope)
    assert ok is want, f"verdict for {text!r} was {got}"
    if not want:
        assert got == cls, f"class for {text!r} was {got}, expected {cls}"


def test_an_ambiguous_reference_is_not_authorisation(home, auth):
    ok, why = auth.assess("do that", "do that")
    assert ok is False and why.startswith("ambiguous")


def test_a_scope_named_only_inside_a_quotation_grants_nothing(home, auth):
    text = 'the report said "complete the implementation" and I have questions'
    ok, _why = auth.assess(text, "complete the implementation")
    assert ok is False


def test_mint_uses_the_sentence_that_carries_the_scope(home, auth):
    mid = _add_msg(home, "I do not need another design proposal. Complete the implementation now.")
    gid = auth.mint(message_id=mid, session_id="s1",
                    scope_cite="complete the implementation", actions=("dispatch",))
    assert gid.startswith("g_")


def test_mint_refuses_a_quoted_instruction(home, auth):
    mid = _add_msg(home, "> complete the implementation")
    with pytest.raises(auth.MintRefused, match="quoted"):
        auth.mint(message_id=mid, session_id="s1",
                  scope_cite="complete the implementation", actions=("dispatch",))


# ---------------------------------------------------------------- fail-closed configuration

def test_unreadable_configuration_fails_closed(tmp_path, monkeypatch, auth):
    """No config.yaml at all: enforce + sandbox, never the permissive defaults."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_AUTHORITY_MODE", raising=False)
    cfg = auth.settings()
    assert cfg["mode"] == "enforce"
    assert cfg["sandbox_workers"] is True
    assert cfg.get("_config_read") is False


def test_unparseable_configuration_fails_closed(tmp_path, monkeypatch, auth):
    home = tmp_path / "h"; home.mkdir()
    (home / "config.yaml").write_text("authority: [this is not a mapping\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_AUTHORITY_MODE", raising=False)
    assert auth.settings()["mode"] == "enforce"


def test_an_unknown_mode_value_never_means_off(tmp_path, monkeypatch, auth):
    home = tmp_path / "h2"; home.mkdir()
    (home / "config.yaml").write_text("authority:\n  mode: banana\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_AUTHORITY_MODE", raising=False)
    assert auth.mode() == "enforce"


def test_fail_closed_configuration_also_demands_the_sandbox(tmp_path, monkeypatch, auth):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_AUTHORITY_MODE", raising=False)
    from hermes_cli import kanban_db_dispatch as D
    monkeypatch.setattr(auth, "sandbox_selftest", lambda: (False, "denied-nothing"))
    _cmd, _env, err = D._authority_sandbox(["/bin/echo"], {}, None)
    assert err and "self-test failed" in err


# ---------------------------------------------------------------- bind() validation
# A string resembling a grant id must never be authority.

def test_bind_rejects_a_failed_mint_response(home, auth):
    with pytest.raises(auth.BindRefused, match="not a grant id"):
        auth.bind("t_x", "refused: session local-abc has source 'kanban'")


def test_bind_rejects_a_nonexistent_grant(home, auth):
    with pytest.raises(auth.BindRefused, match="does not exist"):
        auth.bind("t_x", "g_0000000000000000")


def test_bind_rejects_a_revoked_grant(home, auth):
    gid = _grant(auth, home, actions=("dispatch",))
    auth.revoke(gid)
    with pytest.raises(auth.BindRefused, match="revoked"):
        auth.bind("t_x", gid)


def test_a_bound_grant_without_dispatch_still_cannot_dispatch(home, auth):
    gid = _grant(auth, home, actions=("close",))
    auth.bind("t_x", gid)
    verdict, _ = auth.check("t_x", "dispatch")
    assert verdict != "allow"


def test_bind_accepts_a_live_grant_and_resolves(home, auth):
    gid = _grant(auth, home, actions=("dispatch",))
    auth.bind("t_ok", gid)
    g, _why = auth.resolve("t_ok")
    assert g and g["grant_id"] == gid


# ---------------------------------------------------------------- profile root resolution
# A profile is a configuration island, not a data island: the grant store is singular.

def test_the_grant_store_resolves_to_the_root_inside_a_profile(tmp_path, monkeypatch, auth):
    root = tmp_path / "hermes"; (root / "profiles" / "pm").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "pm"))
    assert auth._home() == root.resolve()


def test_a_profile_running_a_proxy_script_sees_the_same_grants(tmp_path, monkeypatch, auth):
    root = tmp_path / "hermes"; (root / "profiles" / "pm").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    gid = _grant(auth, root, actions=("dispatch",))
    auth.bind("t_p", gid)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "pm"))
    g, why = auth.resolve("t_p")
    assert g and g["grant_id"] == gid, why


# ---------------------------------------------------------------- origin derivation
# A card created outside her interactive turn carries no grant from its creator. The only
# authority it can take is the live dispatch grant minted in the session the card was
# created in, and only when that session is a real interactive one. Derivation happens at
# creation and again at check time, so cards made before the rule existed are not frozen.

def _board(home, rows):
    """A minimal board beside the state db: tasks(id, session_id) is all derivation reads."""
    con = sqlite3.connect(home / "kanban.db")
    try:
        con.execute("create table tasks (id text primary key, session_id text)")
        con.executemany("insert into tasks values (?,?)", rows)
        con.commit()
    finally:
        con.close()


def test_derivation_binds_the_origin_sessions_dispatch_grant(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_derived", "s1")])
    derived = auth.derive_origin_grant("t_derived", "s1")
    assert derived and derived["grant_id"] == gid
    assert auth.check("t_derived", "dispatch")[0] == "allow"


def test_check_time_derivation_heals_a_card_created_before_the_rule(home, auth):
    _grant(auth, home)
    _board(home, [("t_old", "s1")])
    assert auth.resolve("t_old")[0] is None, "precondition: the card starts unbound"
    verdict, reason = auth.check("t_old", "dispatch")
    assert verdict == "allow", reason
    assert auth.resolve("t_old")[0] is not None, "the healing binding is durable, not per-call"


def test_derivation_refuses_a_card_whose_origin_session_is_a_worker(home, auth):
    _grant(auth, home)
    _board(home, [("t_worker", "s2")])          # s2 has source 'kanban'
    assert auth.derive_origin_grant("t_worker", "s2") is None
    assert auth.check("t_worker", "dispatch")[0] == "needs_approval"


def test_derivation_refuses_a_grant_issued_before_the_session_started(home, auth):
    _grant(auth, home)
    con = sqlite3.connect(home / "state.db")
    try:
        con.execute("update sessions set started_at = ? where id = 's1'",
                    (int(time.time()) + 3600,))
        con.commit()
    finally:
        con.close()
    _board(home, [("t_late", "s1")])
    assert auth.derive_origin_grant("t_late", "s1") is None
    assert auth.check("t_late", "dispatch")[0] == "needs_approval"


def test_derivation_refuses_a_card_with_no_origin_session(home, auth):
    _grant(auth, home)
    _board(home, [("t_anon", None)])
    assert auth._task_origin_session("t_anon") is None
    assert auth.check("t_anon", "dispatch")[0] == "needs_approval"


def test_derivation_refuses_when_the_board_is_unreadable(home, auth):
    _grant(auth, home)
    assert auth._task_origin_session("t_nowhere") is None
    assert auth.check("t_nowhere", "dispatch")[0] == "needs_approval"


def test_derivation_uses_only_the_cards_own_origin_session(home, auth):
    """A live grant in some other session is not authority for this card."""
    _grant(auth, home)                                     # her grant, in s1
    con = sqlite3.connect(home / "state.db")
    try:
        con.execute("insert into sessions values ('s_other','other chat',?, 'desktop',0,0)",
                    (int(time.time()),))
        con.commit()
    finally:
        con.close()
    _board(home, [("t_elsewhere", "s_other")])              # interactive, but no grant of its own
    assert auth.derive_origin_grant("t_elsewhere", "s_other") is None
    assert auth.check("t_elsewhere", "dispatch")[0] == "needs_approval"


def test_a_derived_binding_is_audited(home, auth):
    gid = _grant(auth, home)
    _board(home, [("t_audit", "s1")])
    auth.derive_origin_grant("t_audit", "s1")
    log = (home / "state" / "authority-log.jsonl").read_text().splitlines()
    entries = [json.loads(line) for line in log]
    hit = [e for e in entries if e["task"] == "t_audit" and e["action"] == "derive_origin"]
    assert hit and hit[-1]["grant"] == gid
    assert "origin session s1" in hit[-1]["reason"]

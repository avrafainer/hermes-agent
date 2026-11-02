"""The spend ceiling at the spawn boundary (P1.2).

The wire-level guard refuses a paid request once a lane is spent. These tests
cover the earlier question the dispatcher asks: should the worker exist at all?
A refusal must happen before the claim, so nothing is marked failed, the card
stays queued, and no retry budget is consumed.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_spend_gate_test_")
    for prof in ("alpha", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch

    monkeypatch.setattr(kanban_db_dispatch, "_system_memory_sample", dict)
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def test_spent_lane_refuses_before_claim_and_spawn(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import spend_gate

    calls = []

    def fake_spawn(*args, **kwargs):
        calls.append(args)
        return 12345

    monkeypatch.setattr(
        spend_gate, "check_spawn_allowed",
        lambda *_a, **_kw: (False, "OpenRouter unattended daily cap reached: $10.00 spent / $10.00 limit."),
    )
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="work", assignee="alpha")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, dry_run=False)
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        runs = conn.execute("SELECT count(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0]

    assert calls == [], "a spent lane must not start a worker"
    assert res.spawned == []
    assert [t for t, _ in res.skipped_spend_capped] == [task_id]
    assert "cap reached" in res.skipped_spend_capped[0][1]
    assert row["status"] == "ready", "the card must stay queued, not claim-locked or failed"
    assert runs == 0, "no run may be opened for a refused spawn"
    assert "spend_capped=1" in kbd.describe_suppression([res])


def test_lane_with_room_spawns(isolated_kanban_home, monkeypatch):
    kb = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import spend_gate

    monkeypatch.setattr(spend_gate, "check_spawn_allowed", lambda *_a, **_kw: (True, "lane has room"))
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="work", assignee="alpha")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 1
    assert res.skipped_spend_capped == []


def _declare_budget(tmp_path, monkeypatch):
    """Give the gate a ceiling to read, so the enforce() path is exercised."""
    from hermes_cli import local_spend_guard

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = local_spend_guard._shared_home() / "spend-budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"daily_cap_usd": 10}', encoding="utf-8")


def test_guard_denial_becomes_a_refusal(tmp_path, monkeypatch):
    from hermes_cli import local_spend_guard, spend_gate

    _declare_budget(tmp_path, monkeypatch)

    def deny(*_a, **_kw):
        raise local_spend_guard.SpendGuardBlocked("cap reached")

    monkeypatch.setattr(local_spend_guard, "enforce", deny)
    spend_gate.reset_cache()
    allowed, reason = spend_gate.check_spawn_allowed(use_cache=False)
    assert allowed is False
    assert "cap reached" in reason


def test_guard_error_fails_open(tmp_path, monkeypatch):
    """A broken guard must not become an outage: the wire-level guard is the
    last line of defence, and the factory keeps running."""
    from hermes_cli import local_spend_guard, spend_gate

    _declare_budget(tmp_path, monkeypatch)

    def explode(*_a, **_kw):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(local_spend_guard, "enforce", explode)
    spend_gate.reset_cache()
    allowed, reason = spend_gate.check_spawn_allowed(use_cache=False)
    assert allowed is True
    assert "guard error" in reason


def test_absent_budget_file_is_not_a_spent_lane(tmp_path, monkeypatch):
    """A home with no spend-budget.json has declared no ceiling. Refusing every
    spawn there would kill an unconfigured install; the wire-level guard still
    stops any paid request, so the gate allows and stays out of the way."""
    from hermes_cli import local_spend_guard, spend_gate

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def must_not_be_called(*_a, **_kw):
        raise AssertionError("enforce() should not run when no ceiling is declared")

    monkeypatch.setattr(local_spend_guard, "enforce", must_not_be_called)
    spend_gate.reset_cache()
    allowed, reason = spend_gate.check_spawn_allowed(use_cache=False)
    assert allowed is True
    assert reason == "no budget file configured"


def test_route_comes_from_the_assignee_profile(tmp_path, monkeypatch):
    """The gate must ask about the lane the worker will use, not an unnamed
    default: a Codex profile and an OpenRouter profile are different ceilings."""
    import textwrap

    from hermes_cli import spend_gate

    home = tmp_path / "home"
    (home / "profiles" / "codexer").mkdir(parents=True)
    (home / "profiles" / "codexer" / "config.yaml").write_text(
        textwrap.dedent(
            """\
            model:
              default: gpt-5.6-sol
              provider: openai-codex
            kanban:
              max_in_progress: 1
            """
        ),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  base_url: https://openrouter.ai/api/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert spend_gate._route_for("codexer") == ("openai-codex", "")
    assert spend_gate._route_for(None) == ("openrouter", "https://openrouter.ai/api/v1")


def test_unnamed_route_is_refused_not_allowed(tmp_path, monkeypatch):
    """A route the guard cannot place has no meter, so the gate refuses. The
    wire guard would refuse the request anyway; failing here is the honest answer."""
    from hermes_cli import spend_gate

    home = tmp_path / "home"
    home.mkdir()
    (home / "spend-budget.json").write_text('{"daily_cap_usd": 10}', encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n  provider: some-unmetered-gateway\n  base_url: https://example.com/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    spend_gate.reset_cache()
    allowed, reason = spend_gate.check_spawn_allowed(None, use_cache=False)
    assert allowed is False
    assert "no verified spending meter" in reason


def test_decision_is_cached_within_the_ttl(tmp_path, monkeypatch):
    from hermes_cli import local_spend_guard, spend_gate

    _declare_budget(tmp_path, monkeypatch)

    calls = []

    def counted(*_a, **_kw):
        calls.append(1)

    monkeypatch.setattr(local_spend_guard, "enforce", counted)
    spend_gate.reset_cache()
    spend_gate.check_spawn_allowed()
    spend_gate.check_spawn_allowed()
    assert len(calls) == 1, "a tick must not re-read the provider per candidate card"

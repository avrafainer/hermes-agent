"""Behavior contracts for the unattended-only OpenRouter local spend guard."""

from __future__ import annotations

from datetime import datetime
import io
import json
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pytest

from hermes_cli import local_spend_guard as guard


TODAY = "2026-10-29"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def fake_urlopen(daily: float, total: float):
    def _open(request, timeout=None):
        value = daily if request.full_url.endswith("/key") else total
        key = "usage_daily" if request.full_url.endswith("/key") else "total_usage"
        return FakeResponse(json.dumps({"data": {key: value}}).encode())

    return _open


@pytest.fixture
def budget_home(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir()
    (tmp_path / "spend-budget.json").write_text(
        json.dumps({"daily_cap_usd": 15, "temporary": {"date": TODAY, "cap_usd": 25}}))
    monkeypatch.setattr(guard, "_shared_home", lambda: tmp_path)
    monkeypatch.setattr(guard, "today", lambda: TODAY)
    monkeypatch.setattr(guard, "_key", lambda: "unattended-test-key")
    monkeypatch.setattr(guard, "_current_interactive_identity", lambda: None)
    return tmp_path


def write_baseline(home: Path, value: float = 100, date: str = TODAY):
    (home / guard.BASELINE_FILE).write_text(
        json.dumps({"date": date, "day_start_usage": value}))


def test_attended_openrouter_bypasses_every_local_accounting_dependency(budget_home, monkeypatch):
    (budget_home / "spend-budget.json").write_text("invalid")
    (budget_home / "cache" / "budget-lock.json").write_text("invalid")
    monkeypatch.setattr(guard, "_current_interactive_identity", lambda: {"session_id": "s"})
    monkeypatch.setattr(guard, "live_spend", lambda: pytest.fail("local accounting was consulted"))

    guard.enforce(provider="openrouter", base_url="https://openrouter.ai/api/v1")


def test_caller_labels_cannot_bypass_without_server_attestation(budget_home):
    with pytest.raises(guard.SpendGuardBlocked, match="baseline is missing"):
        guard.enforce(
            provider="openrouter", source="desktop", session_id="forged", ui_session_id="forged")


def test_unattended_missing_or_stale_baseline_fails_closed_without_network(budget_home, monkeypatch):
    network = mock.Mock(side_effect=AssertionError("network must not run"))
    monkeypatch.setattr(guard, "urlopen", network)
    with pytest.raises(guard.SpendGuardBlocked, match="baseline is missing"):
        guard.enforce(provider="openrouter")
    write_baseline(budget_home, date="2026-10-28")
    with pytest.raises(guard.SpendGuardBlocked, match="baseline is missing"):
        guard.enforce(provider="openrouter")
    network.assert_not_called()


@pytest.mark.parametrize("spend,blocked", [(24.99, False), (25.0, True), (25.01, True)])
def test_unattended_effective_cap_is_strict(budget_home, monkeypatch, spend, blocked):
    write_baseline(budget_home)
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(spend, 100 + spend))
    if blocked:
        with pytest.raises(guard.SpendGuardBlocked, match="unattended daily cap reached"):
            guard.enforce(provider="openrouter")
    else:
        guard.enforce(provider="openrouter")


def test_budget_lock_remains_an_unattended_floor(budget_home, monkeypatch):
    write_baseline(budget_home)
    (budget_home / "spend-budget.json").write_text(json.dumps({"daily_cap_usd": 10}))
    lock = budget_home / "cache" / "budget-lock.json"
    lock.write_text(json.dumps({"date": TODAY, "spend": 10.207}))
    before = lock.read_bytes()
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(1, 101))
    with pytest.raises(guard.SpendGuardBlocked, match="unattended daily cap reached"):
        guard.enforce(provider="openrouter")
    assert lock.read_bytes() == before


def test_midnight_rollover_replaces_only_a_stale_baseline(budget_home, monkeypatch):
    write_baseline(budget_home, value=10, date="2026-10-28")
    monkeypatch.setattr(guard, "minutes_since_local_midnight", lambda: 30)
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(0, 160))
    result = guard.snapshot_baseline()
    assert result == {"date": TODAY, "day_start_usage": 160}
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(0, 999))
    assert guard.snapshot_baseline() == result


def test_late_watchdog_cannot_fabricate_a_midday_baseline(budget_home, monkeypatch):
    monkeypatch.setattr(guard, "minutes_since_local_midnight", lambda: 15 * 60)
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(0, 160))
    assert guard.snapshot_baseline() is None
    assert not (budget_home / guard.BASELINE_FILE).exists()


def test_utc_key_counter_is_not_a_floor_after_toronto_8pm(budget_home, monkeypatch):
    write_baseline(budget_home, value=100)
    instant = datetime(2026, 10, 30, 2, 30, tzinfo=ZoneInfo("UTC"))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz or ZoneInfo("UTC"))

    monkeypatch.setattr(guard, "datetime", Clock)
    monkeypatch.setattr(guard, "urlopen", fake_urlopen(12, 101))
    reading = guard.spend_reading()
    assert reading["spend"] == 1
    assert reading["detail"]["utc_aligned"] is False


def test_untracked_remote_routes_still_fail_closed(budget_home):
    with pytest.raises(guard.SpendGuardBlocked, match="no verified spending meter"):
        guard.enforce(provider="local", base_url="https://api.example.com/v1")


def test_subscription_and_local_routes_still_bypass_local_accounting(budget_home, monkeypatch):
    monkeypatch.setattr(guard, "read_settings", lambda: pytest.fail("settings were read"))
    guard.enforce(provider="openai-codex")
    guard.enforce(provider="ollama", base_url="http://localhost:11434")


def test_recovery_and_grace_commands_are_retired(budget_home, monkeypatch):
    for action in ("recover", "recovery-status"):
        monkeypatch.setattr("sys.argv", ["hermes-budget", action])
        with pytest.raises(SystemExit):
            guard.main()

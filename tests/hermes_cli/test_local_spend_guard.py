"""Tests for the OpenRouter local spend guard baseline (card t_25b6b79e).

Network is faked. The shared home and "today" are pinned to a temp dir and a
fixed date so the tests are deterministic.
"""
import io
from datetime import datetime
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_cli import local_spend_guard as g

TODAY = "2026-10-29"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_urlopen(daily, total):
    def _open(request, timeout=None):
        url = request.full_url
        if url.endswith("/key"):
            body = {"data": {"usage_daily": daily}}
        elif url.endswith("/credits"):
            body = {"data": {"total_usage": total}}
        else:
            raise AssertionError(url)
        return FakeResponse(json.dumps(body).encode())
    return _open


class SpendGuardBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Pin the clock to 00:30 local so the snapshot window is open by default.
        self.original_minutes = g.minutes_since_local_midnight
        p_clock = mock.patch.object(g, "minutes_since_local_midnight", return_value=30)
        p_clock.start()
        self.addCleanup(p_clock.stop)
        self.home = Path(self.tmp.name)
        (self.home / "cache").mkdir()
        self.original_shared_home = g._shared_home
        self.original_today = g.today
        class FixedClock(datetime):
            @classmethod
            def now(cls, tz=None):
                value = datetime(2026, 10, 29, 21, 30, tzinfo=g.RECOVERY_ZONE)
                return value.astimezone(tz) if tz else value.replace(tzinfo=None)
        patches = [
            mock.patch.object(g, "datetime", FixedClock),
            mock.patch.object(g, "_shared_home", return_value=self.home),
            mock.patch.object(g, "today", return_value=TODAY),
            mock.patch.object(g, "_key", return_value="test-key"),
            mock.patch.object(g, "urlopen", side_effect=AssertionError("Unmocked network request")),
            mock.patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.settings = self.home / "spend-budget.json"
        self.settings.write_text(json.dumps({"daily_cap_usd": 15,
                                            "temporary": {"date": TODAY, "cap_usd": 25}}))

    def tearDown(self):
        self.tmp.cleanup()

    def write_baseline(self, date, start):
        (self.home / "cache" / "spend-baseline.json").write_text(
            json.dumps({"date": date, "day_start_usage": start}))

    def test_no_baseline_blocks_without_querying_or_initializing(self):
        with mock.patch.object(g, "urlopen") as network:
            with self.assertRaisesRegex(g.SpendGuardBlocked, "baseline is missing, stale or invalid"):
                g.enforce(provider="openrouter")
        network.assert_not_called()
        self.assertFalse(g._baseline_path().exists())

    def test_stale_baseline_never_yields_account_value(self):
        self.write_baseline("2026-10-28", 10.0)
        before = g._baseline_path().read_bytes()
        with self.assertRaisesRegex(g.SpendGuardBlocked, "baseline is missing, stale or invalid"):
            g.enforce(provider="openrouter")
        self.assertEqual(g._baseline_path().read_bytes(), before)

    def test_todays_baseline_uses_account_delta_floored_by_key_usage(self):
        self.write_baseline(TODAY, 150.0)
        with mock.patch.object(g, "urlopen", fake_urlopen(daily=5.29, total=155.50)):
            r = g.spend_reading()
        self.assertEqual(r["basis"], "account")
        self.assertFalse(r["degraded"])
        self.assertAlmostEqual(r["spend"], 5.50)

    def test_baseline_above_total_fails_closed(self):
        self.write_baseline(TODAY, 200.0)
        with mock.patch.object(g, "urlopen", fake_urlopen(daily=5.0, total=155.0)):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "baseline"):
                g.spend_reading()

    def test_boundaries_and_manual_override(self):
        self.write_baseline(TODAY, 100)
        for spend, blocked in [(14.99, False), (15, False), (24.99, False), (25, True), (25.01, True)]:
            with self.subTest(spend=spend), mock.patch.object(g, "urlopen", fake_urlopen(spend, 100 + spend)):
                if blocked:
                    with self.assertRaisesRegex(g.SpendGuardBlocked, "daily cap reached"):
                        g.enforce(provider="openrouter")
                else:
                    g.enforce(provider="openrouter")
        self.settings.write_text(json.dumps({"daily_cap_usd": 15,
                                            "temporary": {"date": "2026-10-28", "cap_usd": 25}}))
        with mock.patch.object(g, "urlopen", fake_urlopen(15, 115)):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "daily cap reached"):
                g.enforce(provider="openrouter")

    def test_existing_lock_is_a_floor_and_files_stay_unchanged(self):
        self.write_baseline(TODAY, 100)
        lock_path = self.home / "cache/budget-lock.json"
        lock_path.write_text(json.dumps({"date": TODAY, "spend": 10.207}))
        self.settings.write_text(json.dumps({"daily_cap_usd": 10}))
        originals = {p: p.read_bytes() for p in (lock_path, self.settings, g._baseline_path())}
        with mock.patch.object(g, "urlopen", fake_urlopen(1, 101)):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "daily cap reached"):
                g.enforce(provider="openrouter")
            self.settings.write_text(json.dumps({"daily_cap_usd": 15,
                                                "temporary": {"date": TODAY, "cap_usd": 25}}))
            raised_settings = self.settings.read_bytes()
            g.enforce(provider="openrouter")
        self.assertEqual(lock_path.read_bytes(), originals[lock_path])
        self.assertEqual(g._baseline_path().read_bytes(), originals[g._baseline_path()])
        self.assertEqual(self.settings.read_bytes(), raised_settings)

    def test_stale_lock_does_not_block_today(self):
        self.write_baseline(TODAY, 100)
        (self.home / "cache/budget-lock.json").write_text(json.dumps({"date": "2026-10-28", "spend": 999}))
        with mock.patch.object(g, "urlopen", fake_urlopen(1, 101)):
            g.enforce(provider="openrouter")

    def test_invalid_configuration_and_lock_reject_safely(self):
        self.write_baseline(TODAY, 100)
        for content in ['{', '[]', '{"daily_cap_usd": -1}', '{"daily_cap_usd": "secret-value"}']:
            with self.subTest(content=content):
                self.settings.write_text(content)
                with self.assertRaisesRegex(g.SpendGuardBlocked, "configuration is invalid") as caught:
                    g.enforce(provider="openrouter")
                self.assertNotIn("secret-value", str(caught.exception))
        self.settings.write_text('{"daily_cap_usd": 15}')
        for content in ['{', '[]', json.dumps({"date": TODAY, "spend": float("nan")})]:
            (self.home / "cache/budget-lock.json").write_text(content)
            with mock.patch.object(g, "urlopen", fake_urlopen(1, 101)):
                with self.assertRaisesRegex(g.SpendGuardBlocked, "budget lock is invalid"):
                    g.enforce(provider="openrouter")

    def test_invalid_baselines_block(self):
        for content in ['{', '[]', '{}', json.dumps({"date": TODAY, "day_start_usage": -1}),
                        json.dumps({"date": TODAY, "day_start_usage": float("nan")})]:
            with self.subTest(content=content):
                g._baseline_path().write_text(content)
                with self.assertRaisesRegex(g.SpendGuardBlocked, "baseline"):
                    g.enforce(provider="openrouter")

    def test_missing_credentials_and_import_errors_are_distinguished(self):
        self.write_baseline(TODAY, 100)
        for error, message in [(g.MissingCredentials("secret"), "credentials are missing"),
                               (ModuleNotFoundError("secret"), "credentials could not be loaded")]:
            with self.subTest(error=type(error)), mock.patch.object(g, "_key", side_effect=error):
                with self.assertRaisesRegex(g.SpendGuardBlocked, message) as caught:
                    g.enforce(provider="openrouter")
                self.assertNotIn("secret", str(caught.exception))
                record = json.loads((self.home / g.DIAGNOSTIC_FILE).read_text().splitlines()[-1])
                self.assertEqual(record["operation"], "credentials")
                self.assertEqual(record["exception_type"], type(error).__name__)

    def test_accounting_errors_do_not_leak_exception_text(self):
        self.write_baseline(TODAY, 100)
        for endpoint in ("/key", "/credits"):
            def failing_open(request, timeout=None):
                if request.full_url.endswith(endpoint):
                    raise OSError("Bearer secret-token sensitive-response-body")
                return fake_urlopen(1, 101)(request, timeout)
            with self.subTest(endpoint=endpoint), mock.patch.object(g, "urlopen", failing_open):
                with self.assertRaisesRegex(g.SpendGuardBlocked, "accounting is unavailable") as caught:
                    g.enforce(provider="openrouter")
            diagnostic = (self.home / g.DIAGNOSTIC_FILE).read_text()
            self.assertNotIn("secret-token", diagnostic + str(caught.exception))
            self.assertNotIn("sensitive-response-body", diagnostic + str(caught.exception))
            record = json.loads(diagnostic.splitlines()[-1])
            self.assertEqual(record["version"], g.VERSION)
            self.assertEqual(record["pid"], os.getpid())
            self.assertEqual(record["shared_home"], str(self.home))
            self.assertEqual(record["operation"], "key-usage" if endpoint == "/key" else "account-usage")

    def test_invalid_provider_readings_block(self):
        self.write_baseline(TODAY, 100)
        for daily, total in [(float("nan"), 101), (-1, 101), (1, float("inf")), (1, -1), (1, 99),
                             (None, 101), (1, "sensitive-response-body")]:
            with self.subTest(daily=daily, total=total), mock.patch.object(g, "urlopen", fake_urlopen(daily, total)):
                with self.assertRaises(g.SpendGuardBlocked):
                    g.enforce(provider="openrouter")

    def test_diagnostic_file_is_private_even_if_preexisting_mode_is_broad(self):
        path = self.home / g.DIAGNOSTIC_FILE
        path.write_text("")
        path.chmod(0o644)
        with self.assertRaises(g.SpendGuardBlocked):
            g.enforce(provider="openrouter")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_diagnostic_write_failure_still_blocks_and_is_reported(self):
        with mock.patch.object(g.os, "open", side_effect=OSError("secret")):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "diagnostic log could not be written"):
                g.enforce(provider="openrouter")

    def test_diagnostic_symlink_is_not_followed(self):
        target = self.home / "unrelated"
        target.write_text("unchanged")
        (self.home / g.DIAGNOSTIC_FILE).symlink_to(target)
        with self.assertRaisesRegex(g.SpendGuardBlocked, "diagnostic log could not be written"):
            g.enforce(provider="openrouter")
        self.assertEqual(target.read_text(), "unchanged")

    def test_exemptions_do_not_read_settings_credentials_or_accounting(self):
        routes = [("openai-codex", None), (None, "https://chatgpt.com/backend-api/codex/responses"),
                  ("ollama", "http://localhost:11434"), ("llamacpp", "http://127.0.0.1:8080/v1"),
                  ("lmstudio", "http://localhost:1234/v1"), ("local", None),
                  ("custom", "http://127.0.0.1:11434/v1")]
        with mock.patch.object(g, "read_settings", side_effect=AssertionError("Must bypass settings")), \
                mock.patch.object(g, "_key", side_effect=AssertionError("Must bypass credentials")):
            for provider, url in routes:
                with self.subTest(provider=provider, url=url):
                    g.enforce(provider=provider, base_url=url)
        self.assertFalse((self.home / g.DIAGNOSTIC_FILE).exists())

    def test_remote_destinations_cannot_use_exempt_provider_labels(self):
        for provider in ["openai-codex", "local", "ollama"]:
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(g.SpendGuardBlocked, "no verified spending meter"):
                    g.enforce(provider=provider, base_url="https://api.example.com/v1")

    def test_status_missing_baseline_is_an_error(self):
        with mock.patch("sys.argv", ["hermes-budget", "status"]), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(g.main(), 1)
        self.assertIn("baseline is missing, stale or invalid", out.getvalue())

    def test_profile_context_reads_the_shared_baseline_without_network(self):
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        self.write_baseline(TODAY, 100)
        profile = self.home / "profiles" / "mechanic"
        token = set_hermes_home_override(profile)
        try:
            with mock.patch.object(g, "_shared_home", self.original_shared_home):
                self.assertEqual(g._required_baseline()["day_start_usage"], 100)
                self.assertEqual(g._baseline_path(), self.home / g.BASELINE_FILE)
        finally:
            reset_hermes_home_override(token)

    def test_guard_day_uses_toronto_even_under_utc_process_timezone(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        instant = datetime(2026, 10, 30, 2, 30, tzinfo=ZoneInfo("UTC"))
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz or ZoneInfo("UTC"))
        with mock.patch.object(g, "datetime", Clock), mock.patch.object(g, "today", self.original_today), \
                mock.patch.object(g, "minutes_since_local_midnight", self.original_minutes):
            self.assertEqual(g.today(), "2026-10-29")
            self.assertEqual(g.minutes_since_local_midnight(), 22 * 60 + 30)

    def test_snapshot_writes_once_per_day_atomically(self):
        with mock.patch.object(g, "urlopen", fake_urlopen(daily=0.0, total=157.54)):
            first = g.snapshot_baseline()
        self.assertEqual(first["day_start_usage"], 157.54)
        # A second call the same day must not overwrite the baseline.
        with mock.patch.object(g, "urlopen", fake_urlopen(daily=0.0, total=999.0)):
            second = g.snapshot_baseline()
        self.assertEqual(second["day_start_usage"], 157.54)
        leftovers = [p.name for p in (self.home / "cache").iterdir() if p.name.startswith(".spend-baseline-")]
        self.assertEqual(leftovers, [])

    def test_late_tick_does_not_define_the_day(self):
        # 15:00 local: writing a baseline now would hide the morning's spend.
        with mock.patch.object(g, "minutes_since_local_midnight", return_value=15 * 60), \
                mock.patch.object(g, "urlopen", fake_urlopen(daily=0.0, total=157.54)):
            self.assertIsNone(g.snapshot_baseline())
        self.assertFalse((self.home / "cache" / "spend-baseline.json").exists())

    def test_snapshot_on_new_day_replaces_stale_baseline(self):
        self.write_baseline("2026-10-28", 10.0)
        with mock.patch.object(g, "urlopen", fake_urlopen(daily=0.0, total=160.0)):
            r = g.snapshot_baseline()
        self.assertEqual(r["date"], TODAY)
        self.assertEqual(r["day_start_usage"], 160.0)

    def recovery_identity(self, **changes):
        identity = {"source": "desktop", "session_id": "conversation-7",
                    "ui_session_id": "window-3", "pid": 1234, "ppid": 4321}
        identity.update(changes)
        return identity

    def write_recovery(self, *, baseline=200.0, allowance=2.0,
                       expires="2099-10-30T00:00:00-04:00", identity=None):
        path = g._recovery_path()
        path.write_text(json.dumps({
            "version": 1, "date": TODAY, "authorized_usd": allowance,
            "baseline_total_usage": baseline, "authorized_at": "2026-10-29T21:00:00-04:00",
            "expires_at": expires, "identity": identity or self.recovery_identity(),
            "scope": "one explicitly initiated interactive Hermes Desktop conversation",
        }))
        path.chmod(0o600)
        return path

    def test_recovery_authorization_takes_fresh_account_snapshot_and_is_private(self):
        identity = self.recovery_identity()
        with mock.patch.object(g, "_current_interactive_identity", return_value=identity), \
                mock.patch.object(g, "_account_total", return_value=321.25), \
                mock.patch.object(g, "_recovery_expiry", return_value="2026-10-30T00:00:00-04:00"):
            record = g.authorize_recovery()
        self.assertEqual(record["authorized_usd"], 2.0)
        self.assertEqual(record["baseline_total_usage"], 321.25)
        self.assertEqual(record["identity"], identity)
        self.assertEqual(record["expires_at"], "2026-10-30T00:00:00-04:00")
        self.assertEqual(g._recovery_path().stat().st_mode & 0o777, 0o600)
        self.assertFalse(g._baseline_path().exists(), "recovery must not fabricate a midnight baseline")

    def test_matching_interactive_session_uses_only_incremental_account_total(self):
        identity = self.recovery_identity()
        self.write_recovery()
        with mock.patch.object(g, "_current_interactive_identity", return_value=identity), \
                mock.patch.object(g, "_account_total", return_value=201.99), \
                mock.patch.object(g, "_key_daily", side_effect=AssertionError("daily key usage is not the recovery meter")):
            g.enforce(provider="openrouter")
            reading = g.recovery_reading()
        self.assertAlmostEqual(reading["used_usd"], 1.99)
        self.assertAlmostEqual(reading["remaining_usd"], 0.01)
        self.assertFalse(g._baseline_path().exists())

    def test_recovery_boundary_is_terminal_and_never_uses_paid_fallback(self):
        self.write_recovery()
        with mock.patch.object(g, "_current_interactive_identity", return_value=self.recovery_identity()), \
                mock.patch.object(g, "_account_total", return_value=202.0):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "recovery allowance exhausted"):
                g.enforce(provider="openrouter")

    def test_recovery_is_not_available_to_other_or_unattended_contexts(self):
        self.write_recovery()
        identities = [None, self.recovery_identity(session_id="other"),
                      self.recovery_identity(ui_session_id="other"),
                      self.recovery_identity(pid=999), self.recovery_identity(source="kanban")]
        for identity in identities:
            with self.subTest(identity=identity), \
                    mock.patch.object(g, "_current_interactive_identity", return_value=identity), \
                    mock.patch.object(g, "_account_total", return_value=200.1):
                with self.assertRaisesRegex(g.SpendGuardBlocked, "baseline is missing"):
                    g.enforce(provider="openrouter", source="desktop", session_id="conversation-7",
                              ui_session_id="window-3")

    def test_recovery_accounting_expiry_and_file_validation_fail_closed(self):
        path = self.write_recovery()
        cases = [
            (199.99, None, "regressed"),
            (float("nan"), None, "unavailable or invalid"),
            (201.0, "2020-01-01T00:00:00-04:00", "expired"),
        ]
        for total, expiry, message in cases:
            if expiry:
                self.write_recovery(expires=expiry)
            else:
                self.write_recovery()
            with self.subTest(total=total, expiry=expiry), \
                    mock.patch.object(g, "_current_interactive_identity", return_value=self.recovery_identity()), \
                    mock.patch.object(g, "_account_total", return_value=total):
                with self.assertRaisesRegex(g.SpendGuardBlocked, message):
                    g.enforce(provider="openrouter")
        path.write_text("not-json")
        with mock.patch.object(g, "_current_interactive_identity", return_value=self.recovery_identity()):
            with self.assertRaisesRegex(g.SpendGuardBlocked, "recovery authorization is invalid"):
                g.enforce(provider="openrouter")

    def test_unattended_markers_and_delegation_are_rejected_by_identity_gate(self):
        env = {"HERMES_SESSION_SOURCE": "desktop", "HERMES_SESSION_ID": "conversation-7",
               "HERMES_UI_SESSION_ID": "window-3"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(g.os, "getpid", return_value=1234), \
                mock.patch.object(g.os, "getppid", return_value=4321), \
                mock.patch.object(g, "_is_desktop_backend_process", return_value=True):
            self.assertEqual(g._current_interactive_identity(), self.recovery_identity())
            for marker in ("HERMES_KANBAN_TASK", "HERMES_CRON_SESSION", "HERMES_DELEGATED_CHILD_CONTEXT"):
                with self.subTest(marker=marker), mock.patch.dict(os.environ, {marker: "1"}):
                    self.assertIsNone(g._current_interactive_identity())
            with mock.patch("agent.delegation_context.is_delegated_child_process_context", return_value=True):
                self.assertIsNone(g._current_interactive_identity())


if __name__ == "__main__":
    unittest.main()

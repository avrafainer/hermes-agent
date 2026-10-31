"""The Desktop backend, not request labels, mints attended spending identity."""

from __future__ import annotations

from gateway.session_context import attended_desktop_turn
from tui_gateway import server


def _session(*, source="desktop", transport=object()):
    return {
        "session_key": "conversation", "source": source, "transport": transport,
        "cwd": "", "profile_home": None,
    }


def test_live_desktop_prompt_mints_request_scoped_proof(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.setitem(server._sessions, "window", _session())
    tokens = server._set_session_context("conversation", ui_session_id="window")
    try:
        assert attended_desktop_turn() is True
    finally:
        server._clear_session_context(tokens)
    assert attended_desktop_turn() is False


def test_source_label_without_live_desktop_transport_cannot_mint_proof(monkeypatch):
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setitem(server._sessions, "window", _session(transport=None))
    tokens = server._set_session_context("conversation", ui_session_id="window")
    try:
        assert attended_desktop_turn() is False
    finally:
        server._clear_session_context(tokens)


def test_tui_or_messaging_session_cannot_mint_desktop_proof(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setitem(server._sessions, "window", _session())
    tokens = server._set_session_context("conversation", ui_session_id="window")
    try:
        assert attended_desktop_turn() is False
    finally:
        server._clear_session_context(tokens)

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setitem(server._sessions, "window", _session(source="telegram"))
    tokens = server._set_session_context("conversation", ui_session_id="window")
    try:
        assert attended_desktop_turn() is False
    finally:
        server._clear_session_context(tokens)


class _AuthTransport:
    """A WS transport carrying the identity the server minted at upgrade."""

    def __init__(self, identity):
        self.auth_identity = identity


def _gateway_session(*, source, identity):
    return {
        "session_key": "conversation", "source": source,
        "transport": _AuthTransport(identity), "cwd": "", "profile_home": None,
    }


def test_signed_in_gateway_client_mints_attended_proof(monkeypatch):
    """A Desktop or Hermex chat served by a shared gateway is still a human typing.

    Regression: pointing the Desktop app at `hermes serve` changed its session source
    from `desktop` to `gui`, which silently dropped every chat onto the unattended
    spending pool and left the dedicated interactive key unused.
    """
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    identity = {"user_id": "avra", "provider": "basic"}
    for source in ("gui", "desktop", "tui", "local"):
        monkeypatch.setitem(server._sessions, "window", _gateway_session(source=source, identity=identity))
        tokens = server._set_session_context("conversation", ui_session_id="window")
        try:
            assert attended_desktop_turn() is True, source
        finally:
            server._clear_session_context(tokens)


def test_unauthenticated_gateway_client_cannot_mint_attended_proof(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.setitem(server._sessions, "window", _gateway_session(source="gui", identity=None))
    tokens = server._set_session_context("conversation", ui_session_id="window")
    try:
        assert attended_desktop_turn() is False
    finally:
        server._clear_session_context(tokens)


def test_signed_in_messaging_source_cannot_mint_attended_proof(monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    identity = {"user_id": "avra", "provider": "basic"}
    for source in ("telegram", "discord", "cron", "kanban", "api_server"):
        monkeypatch.setitem(server._sessions, "window", _gateway_session(source=source, identity=identity))
        tokens = server._set_session_context("conversation", ui_session_id="window")
        try:
            assert attended_desktop_turn() is False, source
        finally:
            server._clear_session_context(tokens)

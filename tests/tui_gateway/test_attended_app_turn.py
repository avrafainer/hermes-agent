"""A phone turn must not be priced against the unattended budget.

2026-11-02: Avra's iOS turn arrived at ``hermes serve`` with source "gui".  The
attended check required source == "desktop", so the request fell through to the
unattended daily cap and was refused - she was locked out of her own assistant.
"""
import pytest

from tui_gateway import server
from tui_gateway.server import _is_attended_app_turn


def test_transport_predicate_is_bound_at_module_level():
    """It was referenced but never imported, so every call raised NameError inside
    contextlib.suppress(Exception): session context was silently never bound and the
    attended check never engaged.  A bound name is the whole fix."""
    assert callable(server._session_has_live_transport)


@pytest.mark.parametrize("source", ["desktop", "gui", "GUI", " gui "])
def test_app_surface_with_live_transport_is_attended(source):
    assert _is_attended_app_turn(source=source, ui_session_id="ui-1", has_live_transport=True)


@pytest.mark.parametrize("source", ["kanban", "cron", "oneshot", "webui", "", "discord"])
def test_non_app_surface_is_not_attended(source):
    assert not _is_attended_app_turn(source=source, ui_session_id="ui-1", has_live_transport=True)


def test_app_surface_without_live_transport_is_not_attended():
    """A record with no live peer is background work, not a prompt."""
    assert not _is_attended_app_turn(source="gui", ui_session_id="ui-1", has_live_transport=False)


def test_missing_ui_session_is_not_attended():
    assert not _is_attended_app_turn(source="gui", ui_session_id="", has_live_transport=True)

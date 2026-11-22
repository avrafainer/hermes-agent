"""The desktop channel: ``platform="desktop"`` rows delivered by the ``hermes serve`` session poller.

The desktop app is not a gateway platform (no messaging adapter, no ``Platform``
enum member), so the gateway kanban-notifier skips its rows. Its chat sessions
run inside the ``hermes serve`` process, whose per-session poller already
delivers ``platform="tui"`` rows — so that poller is the desktop channel's
delivery path. ``kanban_db.create_task`` writes one ``desktop:*`` row per card
when ``kanban.notify_desktop`` is on, which is what makes the closure reach the
desktop even for cards created by a cron or a worker (no originating chat).

These tests pin: the row written at create time, delivery to a desktop session
only, exactly-once claiming across desktop sessions, no double announce for a
card the desktop session already owns a ``tui`` row for, and that ``tui`` rows
deliver unchanged.
"""

from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn
from tui_gateway.server import _collect_kanban_notifications

DESKTOP_KEY = "desktop-session-key-1"


def _desktop_session(key: str = DESKTOP_KEY) -> dict:
    return {"session_key": key, "source": "desktop"}


def _tui_session(key: str) -> dict:
    return {"session_key": key, "source": "tui"}


@pytest.fixture
def desktop_on():
    """Enable the channel without touching a real config file."""
    with patch.object(kbn, "desktop_notify_enabled", return_value=True):
        yield


def _create_task(*, tui_chat_id: str | None = None):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="desktop notify", assignee="worker")
        if tui_chat_id:
            kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=tui_chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str, summary: str = "shipped it") -> None:
    conn = kbc.connect()
    try:
        kb.complete_task(conn, tid, summary=summary)
    finally:
        conn.close()


def _sub_rows(tid: str) -> list:
    conn = kbc.connect()
    try:
        return kbn.list_notify_subs(conn, task_id=tid)
    finally:
        conn.close()


def _desktop_rows(tid: str) -> list:
    return [row for row in _sub_rows(tid) if (row.get("platform") or "").lower() == "desktop"]


class TestDesktopRouteIsWrittenAtCreate:
    def test_enabled_writes_one_wildcard_row_started_caught_up(self, desktop_on):
        tid = _create_task()

        rows = _desktop_rows(tid)

        assert len(rows) == 1
        assert rows[0]["chat_id"] == kbn.DESKTOP_ANY_CHAT
        assert rows[0]["chat_id"] == "*"
        assert rows[0]["delivery_mode"] == "notify"
        # Caught up at the task's own created event: creating never replays.
        assert rows[0]["last_event_id"] > 0

    def test_disabled_writes_no_desktop_row(self):
        tid = _create_task()

        assert _desktop_rows(tid) == []


class TestDelivery:
    def test_desktop_session_delivers_the_closure_once(self, desktop_on):
        tid = _create_task()
        _complete(tid, summary="the closure text")

        first = _collect_kanban_notifications(_desktop_session())

        assert len(first) == 1
        assert tid in first[0]
        assert "done" in first[0]
        assert "the closure text" in first[0]
        # Cursor advanced: no replay on the next poll.
        assert _collect_kanban_notifications(_desktop_session()) == []

    def test_non_desktop_session_does_not_consume_the_desktop_route(self, desktop_on):
        tid = _create_task()
        _complete(tid)

        assert _collect_kanban_notifications(_tui_session("tui-session-key-3")) == []

        # Still unclaimed, so the next desktop session still hears it.
        assert len(_collect_kanban_notifications(_desktop_session("desktop-session-key-3"))) == 1

    def test_second_desktop_session_cannot_re_deliver(self, desktop_on):
        tid = _create_task()
        _complete(tid)

        assert len(_collect_kanban_notifications(_desktop_session("desktop-session-key-4"))) == 1
        assert _collect_kanban_notifications(_desktop_session("desktop-session-key-5")) == []

    def test_card_created_in_the_desktop_chat_is_announced_once(self, desktop_on):
        tid = _create_task(tui_chat_id=DESKTOP_KEY)
        _complete(tid, summary="own session")

        texts = _collect_kanban_notifications(_desktop_session())

        assert len(texts) == 1, "the tui row and the desktop:* row must not both fire"
        assert tid in texts[0]


class TestOtherChannelsUnchanged:
    def test_tui_row_still_delivers_with_desktop_disabled(self):
        tid = _create_task(tui_chat_id="tui-session-key-6")
        _complete(tid, summary="terminal session")

        texts = _collect_kanban_notifications(_tui_session("tui-session-key-6"))

        assert len(texts) == 1
        assert "terminal session" in texts[0]
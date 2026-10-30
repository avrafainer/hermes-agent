"""Dedicated OpenRouter credential routing for attended Desktop turns."""

from __future__ import annotations

import pytest

from agent import interactive_spending as spending
from gateway.session_context import clear_session_vars, set_session_vars


@pytest.fixture
def interactive_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_INTERACTIVE_API_KEY=interactive-secret\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def bind_attended():
    return set_session_vars(
        source="desktop", session_id="conversation", ui_session_id="window",
        cron_session="", attended_desktop=True)


def test_attended_openrouter_uses_dedicated_key_only_at_final_boundary(interactive_home):
    request = {"model": "deepseek/test", "extra_headers": {"X-Trace": "safe", "authorization": "old"}}
    tokens = bind_attended()
    try:
        routed = spending.route_attended_openrouter_request(
            request, provider="openrouter", base_url="https://openrouter.ai/api/v1")
    finally:
        clear_session_vars(tokens)

    assert routed is not request
    assert routed["extra_headers"] == {
        "X-Trace": "safe", "Authorization": "Bearer interactive-secret"}
    assert request["extra_headers"]["authorization"] == "old"


def test_missing_interactive_key_fails_closed_without_borrowing_unattended_key(interactive_home):
    (interactive_home / ".env").write_text("OPENROUTER_API_KEY=unattended-secret\n")
    tokens = bind_attended()
    try:
        with pytest.raises(spending.InteractiveProviderCredentialError, match="was not used as a fallback"):
            spending.route_attended_openrouter_request(
                {"model": "deepseek/test"}, provider="openrouter",
                base_url="https://openrouter.ai/api/v1")
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize("marker", [
    "HERMES_KANBAN_TASK", "HERMES_CRON_SESSION", "HERMES_DELEGATED_CHILD_CONTEXT",
])
def test_unattended_marker_cannot_borrow_interactive_key(interactive_home, monkeypatch, marker):
    tokens = bind_attended()
    monkeypatch.setenv(marker, "1")
    request = {"model": "deepseek/test"}
    try:
        assert spending.route_attended_openrouter_request(
            request, provider="openrouter", base_url="https://openrouter.ai/api/v1") is request
    finally:
        clear_session_vars(tokens)


def test_no_server_attestation_means_no_interactive_key(interactive_home):
    tokens = set_session_vars(
        source="desktop", session_id="forged", ui_session_id="forged", cron_session="")
    request = {"model": "deepseek/test"}
    try:
        assert spending.route_attended_openrouter_request(
            request, provider="openrouter", base_url="https://openrouter.ai/api/v1") is request
    finally:
        clear_session_vars(tokens)


def test_non_openrouter_request_is_unchanged(interactive_home):
    tokens = bind_attended()
    request = {"model": "local"}
    try:
        assert spending.route_attended_openrouter_request(
            request, provider="ollama", base_url="http://localhost:11434") is request
    finally:
        clear_session_vars(tokens)

"""End-to-end unit contracts for the main and auxiliary provider seams."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.auxiliary_client import _relay_async_completion, _relay_sync_completion
from agent.turn_api_call import perform_api_call
from gateway.session_context import clear_session_vars, set_session_vars


@pytest.fixture
def attended_key(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text(
        "OPENROUTER_INTERACTIVE_API_KEY=interactive-path-secret\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    tokens = set_session_vars(
        source="desktop", session_id="conversation", ui_session_id="window",
        cron_session="", attended_desktop=True)
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def provider_boundary(monkeypatch):
    monkeypatch.setattr("hermes_cli.local_spend_guard.enforce", lambda **_kwargs: None)
    monkeypatch.setattr(
        "agent.automated_model_authorization.enforce_automated_model_authorization",
        lambda **_kwargs: None,
    )


def _main_agent(captured):
    def send(request):
        captured.append(request)
        return "main-ok"

    return SimpleNamespace(
        provider="openrouter", base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions", session_id="conversation", platform="desktop",
        model="deepseek/test", is_subagent=False, _fallback_index=0,
        _disable_streaming=True, client=MagicMock(), thinking_callback=None,
        _model_request_active=None, _pending_redirect_lock=None,
        _has_pending_redirect=lambda: False, _has_stream_consumers=lambda: False,
        _interruptible_api_call=send,
    )


def test_main_turn_injects_key_only_inside_final_callback(attended_key, monkeypatch):
    captured = []
    request = {"model": "deepseek/test", "messages": [{"role": "user", "content": "hi"}]}
    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware",
        lambda req, callback, **_kwargs: callback(req),
    )
    monkeypatch.setattr(
        "agent.relay_llm.execute",
        lambda req, callback, **_kwargs: callback(req),
    )

    result = perform_api_call(
        _main_agent(captured), api_kwargs=request, _original_api_kwargs=request,
        _llm_middleware_trace=[], _moa_prepared_request=None,
        _retry=SimpleNamespace(), thinking_spinner=None, retry_count=0,
        api_call_count=1, api_request_id="request", effective_task_id="turn",
        turn_id="turn", interrupted=False,
    )

    assert result.response == "main-ok"
    assert captured[0]["extra_headers"]["Authorization"] == "Bearer interactive-path-secret"
    assert "extra_headers" not in request


def test_auxiliary_sync_injects_key_and_preserves_provider_rejection(attended_key):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    seen = []
    request = {"model": "deepseek/test", "messages": [{"role": "user", "content": "hi"}]}

    assert _relay_sync_completion(
        client, request, provider="openrouter", create=lambda routed: seen.append(routed) or "aux-ok"
    ) == "aux-ok"
    assert seen[0]["extra_headers"]["Authorization"] == "Bearer interactive-path-secret"

    provider_error = RuntimeError("OpenRouter 402: provider budget exceeded")
    with pytest.raises(RuntimeError) as raised:
        _relay_sync_completion(
            client, request, provider="openrouter",
            create=lambda _routed: (_ for _ in ()).throw(provider_error),
        )
    assert raised.value is provider_error


@pytest.mark.asyncio
async def test_auxiliary_async_injects_key(attended_key):
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    seen = []

    async def send(routed):
        seen.append(routed)
        return "async-ok"

    result = await _relay_async_completion(
        client,
        {"model": "deepseek/test", "messages": [{"role": "user", "content": "hi"}]},
        provider="openrouter", create=send,
    )
    assert result == "async-ok"
    assert seen[0]["extra_headers"]["Authorization"] == "Bearer interactive-path-secret"

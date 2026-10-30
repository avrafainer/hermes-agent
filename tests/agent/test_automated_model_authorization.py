import json
from types import SimpleNamespace

import pytest

from agent import automated_model_authorization as ama


@pytest.fixture
def automated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-7")
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    return home


def _authorize(home, *, maximum=2, expires="2099-01-01T00:00:00+00:00"):
    (home / "automated-model-authorizations.json").write_text(json.dumps({
        "version": 1,
        "authorizations": {
            "kanban:task-7": {
                "approval_id": "approval-7",
                "purpose": "bounded acceptance verification",
                "max_generations": maximum,
                "expires_at": expires,
            }
        },
    }), encoding="utf-8")


def test_interactive_and_non_codex_routes_remain_available(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    ama.enforce_automated_model_authorization(provider="openai-codex")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-7")
    ama.enforce_automated_model_authorization(provider="llamacpp", base_url="http://localhost:8080")


def test_missing_invalid_and_expired_authorizations_fail_closed(automated_home):
    with pytest.raises(ama.AutomatedModelAuthorizationBlocked, match="no explicit authorization"):
        ama.enforce_automated_model_authorization(provider="openai-codex")
    automated_home.joinpath("automated-model-authorizations.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ama.AutomatedModelAuthorizationBlocked, match="authorization is invalid"):
        ama.enforce_automated_model_authorization(provider="openai-codex")
    _authorize(automated_home, expires="2020-01-01T00:00:00+00:00")
    with pytest.raises(ama.AutomatedModelAuthorizationBlocked, match="authorization is invalid"):
        ama.enforce_automated_model_authorization(provider="openai-codex")


def test_finite_budget_is_shared_and_boundary_is_fail_closed(automated_home):
    _authorize(automated_home, maximum=2)
    ama.enforce_automated_model_authorization(provider="openai-codex")
    ama.enforce_automated_model_authorization(provider="openai-codex")
    with pytest.raises(ama.AutomatedModelAuthorizationBlocked, match="limit of 2 is exhausted"):
        ama.enforce_automated_model_authorization(provider="openai-codex")
    ledger = json.loads(ama.usage_path().read_text(encoding="utf-8"))
    assert ledger["usage"] == {"approval-7": 2}


def test_main_middleware_blocks_before_provider_call(automated_home, monkeypatch):
    from hermes_cli import middleware
    monkeypatch.setattr("hermes_cli.local_spend_guard.enforce", lambda **_kw: None)
    called = []
    with pytest.raises(ama.AutomatedModelAuthorizationBlocked):
        middleware.run_llm_execution_middleware(
            {}, lambda request: called.append(request), provider="openai-codex", task_id="turn-1")
    assert called == []


def test_automated_context_disables_live_codex_quota_probe(automated_home, monkeypatch):
    from agent.credential_pool import CredentialPool
    pool = CredentialPool.__new__(CredentialPool)
    pool.provider = "openai-codex"
    entry = SimpleNamespace(last_status="exhausted")
    assert pool._codex_quota_restored_upstream(entry) is False

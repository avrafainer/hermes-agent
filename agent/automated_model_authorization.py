"""Finite, operator-authored budgets for unattended Codex generations.

Interactive sessions are intentionally out of scope.  Kanban and cron executions must have an
entry in ``automated-model-authorizations.json`` under the shared Hermes root.  Usage is kept in a
separate, locked ledger so implementer/reviewer processes consume one organization-wide budget.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional


class AutomatedModelAuthorizationBlocked(RuntimeError):
    """An unattended Codex call lacked a valid, unexhausted operator authorization."""


def _shared_root() -> Path:
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home()).expanduser()
    return home.parent.parent if home.parent.name == "profiles" else home


def authorization_path() -> Path:
    return _shared_root() / "automated-model-authorizations.json"


def usage_path() -> Path:
    return _shared_root() / "state" / "automated-model-usage.json"


def automated_execution_identity(*, task_id: Any = None) -> Optional[str]:
    """Return the stable unattended-work identity, or None for an interactive session."""
    kanban_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if kanban_task:
        return f"kanban:{kanban_task}"
    try:
        from gateway.session_context import get_session_env

        cron = str(get_session_env("HERMES_CRON_SESSION") or "").strip()
    except Exception:
        cron = os.environ.get("HERMES_CRON_SESSION", "").strip()
    if cron:
        raw = str(task_id or "")
        if raw.startswith("cron:"):
            parts = raw.split(":", 2)
            if len(parts) >= 2 and parts[1]:
                return f"cron:{parts[1]}"
        return "cron:unscoped"
    return None


def is_automated_execution(*, task_id: Any = None) -> bool:
    return automated_execution_identity(task_id=task_id) is not None


def _is_codex_route(provider: Any, base_url: Any = None, api_mode: Any = None) -> bool:
    p = str(provider or "").strip().lower()
    mode = str(api_mode or "").strip().lower()
    base = str(base_url or "").strip().lower()
    return p == "openai-codex" or mode == "codex_responses" or "chatgpt.com/backend-api" in base


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level value must be an object")
    return value


def _parse_expiry(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        raise ValueError("expires_at must include a timezone")
    return parsed.astimezone(UTC)


def _write_usage(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def enforce_automated_model_authorization(
    *, provider: Any = None, base_url: Any = None, api_mode: Any = None,
    task_id: Any = None, **_context: Any,
) -> None:
    """Consume one authorized unattended Codex generation, failing closed otherwise."""
    if not _is_codex_route(provider, base_url, api_mode):
        return
    identity = automated_execution_identity(task_id=task_id)
    if identity is None:
        return

    auth_file = authorization_path()
    try:
        document = _read_json(auth_file)
        records = document.get("authorizations")
        record = records.get(identity) if isinstance(records, dict) else None
        if not isinstance(record, dict):
            raise KeyError(identity)
        maximum = int(record.get("max_generations"))
        if maximum <= 0:
            raise ValueError("max_generations must be positive")
        purpose = str(record.get("purpose") or "").strip()
        approval_id = str(record.get("approval_id") or "").strip()
        if not purpose or not approval_id:
            raise ValueError("purpose and approval_id are required")
        if _parse_expiry(record.get("expires_at")) <= datetime.now(UTC):
            raise PermissionError("authorization expired")
    except FileNotFoundError:
        raise AutomatedModelAuthorizationBlocked(
            f"Automated Codex generation blocked for {identity}: no explicit authorization file exists."
        ) from None
    except KeyError:
        raise AutomatedModelAuthorizationBlocked(
            f"Automated Codex generation blocked for {identity}: this task has no authorization."
        ) from None
    except (OSError, ValueError, TypeError, json.JSONDecodeError, PermissionError) as exc:
        raise AutomatedModelAuthorizationBlocked(
            f"Automated Codex generation blocked for {identity}: authorization is invalid ({type(exc).__name__})."
        ) from None

    from tools.skill_usage import skill_file_lock

    ledger_file = usage_path()
    with skill_file_lock(ledger_file.with_suffix(".json.lock")):
        try:
            ledger = _read_json(ledger_file)
        except FileNotFoundError:
            ledger = {"version": 1, "usage": {}}
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise AutomatedModelAuthorizationBlocked(
                f"Automated Codex generation blocked for {identity}: usage ledger is invalid ({type(exc).__name__})."
            ) from None
        usage = ledger.setdefault("usage", {})
        if not isinstance(usage, dict):
            raise AutomatedModelAuthorizationBlocked(
                f"Automated Codex generation blocked for {identity}: usage ledger is invalid (ValueError)."
            )
        used = int(usage.get(approval_id, 0))
        if used >= maximum:
            raise AutomatedModelAuthorizationBlocked(
                f"Automated Codex generation blocked for {identity}: approved limit of {maximum} is exhausted."
            )
        usage[approval_id] = used + 1
        _write_usage(ledger_file, ledger)

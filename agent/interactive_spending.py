"""Provider-key isolation for authenticated, attended Hermes Desktop turns.

The Desktop backend mints the request-scoped proof in ``gateway.session_context``.
It is a ContextVar only: it is never mirrored into the process environment or
exported to child processes.  Unattended markers remain a defence-in-depth
backstop because delegated work can inherit Python context.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


INTERACTIVE_OPENROUTER_KEY = "OPENROUTER_INTERACTIVE_API_KEY"


class InteractiveProviderCredentialError(RuntimeError):
    """An attended request could not obtain its dedicated provider credential."""


def _shared_home() -> Path:
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home()).expanduser()
    return home.parent.parent if home.parent.name == "profiles" else home


def _has_unattended_marker() -> bool:
    if any(str(os.environ.get(name) or "").strip() for name in (
        "HERMES_KANBAN_TASK", "HERMES_CRON_SESSION", "HERMES_DELEGATED_CHILD_CONTEXT",
    )):
        return True
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return bool(is_delegated_child_process_context())
    except Exception:
        return True


def is_attended_desktop_provider_request() -> bool:
    """Whether this physical provider call belongs to a server-attested Desktop turn."""
    try:
        from gateway.session_context import attended_desktop_turn

        return attended_desktop_turn() and not _has_unattended_marker()
    except Exception:
        return False


def is_openrouter_route(provider: Any = None, base_url: Any = None) -> bool:
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return True
    return str(provider or "").strip().lower() == "openrouter" and not host


def _interactive_key() -> str:
    from agent.secret_scope import load_env_file

    key = str(load_env_file(_shared_home() / ".env").get(INTERACTIVE_OPENROUTER_KEY) or "").strip()
    if not key:
        raise InteractiveProviderCredentialError(
            "Attended OpenRouter request stopped: the dedicated Hermes Interactive key is missing. "
            "The unattended OpenRouter key was not used as a fallback."
        )
    return key


def route_attended_openrouter_request(
    request: dict[str, Any], *, provider: Any = None, base_url: Any = None,
) -> dict[str, Any]:
    """Add the dedicated Authorization header at the final provider boundary.

    Relay, plugins and request middleware see the credential-free request.  A
    fresh dict also prevents a cached request object from retaining the key.
    """
    if not is_openrouter_route(provider, base_url) or not is_attended_desktop_provider_request():
        return request
    headers = {
        str(name): str(value)
        for name, value in dict(request.get("extra_headers") or {}).items()
        if str(name).lower() != "authorization"
    }
    headers["Authorization"] = f"Bearer {_interactive_key()}"
    return {**request, "extra_headers": headers}

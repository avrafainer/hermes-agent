"""Spend ceiling at the spawn boundary.

The guard's ``enforce()`` already runs at the wire, so a paid request cannot leave
once a lane is spent. This module answers the earlier question the dispatcher asks
instead: should a worker process exist at all? A refused spawn costs nothing and
saves a cold start, a claim, and a card that would otherwise crash-partial.

The route matters. ``enforce()`` decides from ``(provider, base_url)`` whether a
request is metered, subscription-billed or local, and refuses anything it cannot
place. The dispatcher knows which profile the worker will run as, so the gate
resolves that profile's model route and asks the guard about *that* lane, not
about an unnamed default.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# One tick can offer several ready cards; a reading costs a provider round trip,
# so a ledger change inside this window is not worth re-asking.
CACHE_TTL_SECONDS = 60.0

# ``(provider, base_url) -> (at, decision)``: a profile on Codex and a profile on
# OpenRouter are different lanes and must not share one memoised answer.
_cache: dict[tuple[str, str], tuple[float, tuple[bool, str]]] = {}


def _log_gate(message: str) -> None:
    """Append to a small local trail so a refusal is visible without a debugger."""
    try:
        from hermes_constants import get_hermes_home

        path = Path(get_hermes_home()) / "cache" / "spend-gate.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {message}\n")
    except Exception:  # logging must never be the reason a spawn fails
        pass


def _read_route(config_path: Path) -> tuple[str, str]:
    data = None
    try:
        from utils import fast_safe_load

        with config_path.open(encoding="utf-8") as handle:
            data = fast_safe_load(handle) or {}
    except Exception:
        # Dependency-free fallback for the two keys that matter. A gate that
        # cannot read the route refuses everything, so degrading here is safer
        # than trusting an optional import in a stripped interpreter.
        try:
            text = config_path.read_text(encoding="utf-8")
        except Exception:
            return "", ""
        model_block = text.split("\nmodel:", 1)
        if len(model_block) != 2:
            return "", ""
        block = model_block[1].split("\nkanban:", 1)[0]
        found = {}
        for line in block.splitlines():
            key, _, value = line.strip().partition(":")
            if key in {"provider", "base_url"}:
                found[key] = value.strip().strip("'\"")
        return found.get("provider", ""), found.get("base_url", "")
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, dict):
        return "", ""
    return str(model.get("provider") or ""), str(model.get("base_url") or "")


def _route_for(assignee: Optional[str]) -> tuple[str, str]:
    """The model route the worker will actually use.

    Resolved from the assignee's profile config, falling back to the launch home
    when the profile has none (a bare or isolated fixture). An unresolvable route
    returns empty strings, and the guard then refuses it: an unnamed route has no
    meter, which is the same answer the wire would give.
    """
    if assignee:
        try:
            from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

            profile_home = resolve_profile_env(normalize_profile_name(assignee))
            if profile_home:
                provider, base_url = _read_route(Path(profile_home) / "config.yaml")
                if provider or base_url:
                    return provider, base_url
        except Exception as exc:  # a broken profile must not hide the gate entirely
            logger.debug("spend gate could not resolve profile %s: %s", assignee, exc)
    try:
        from hermes_constants import get_hermes_home

        return _read_route(Path(get_hermes_home()) / "config.yaml")
    except Exception:
        return "", ""


def _decide(assignee: Optional[str]) -> tuple[bool, str]:
    try:
        from hermes_cli.local_spend_guard import SpendGuardBlocked, _shared_home, enforce
    except Exception as exc:  # pragma: no cover - import failure is environmental
        # Fail open, loudly. The wire-level guard is the last line of defence and
        # does not depend on this module; stopping the whole factory because this
        # import broke would trade a cost risk for a total outage.
        logger.warning("spend gate unavailable, allowing spawn: %s", exc)
        _log_gate(f"allowed guard-unavailable: {exc}")
        return True, "guard unavailable"
    # No budget file means no ceiling was ever declared (a fresh home, a test
    # sandbox, a profile that never opted in). The guard treats that as a stop,
    # which is right at the wire and fatal here: it would refuse every spawn in
    # an unconfigured install. Absence is not a spent lane, so allow and let the
    # wire-level guard refuse the request if one is ever attempted.
    try:
        if not (_shared_home() / "spend-budget.json").exists():
            _log_gate("allowed no-budget-file")
            return True, "no budget file configured"
    except Exception:
        pass  # cannot tell: let enforce() make the call
    provider, base_url = _route_for(assignee)
    try:
        enforce(provider=provider or None, base_url=base_url or None)
    except SpendGuardBlocked as blocked:
        _log_gate(f"refused {assignee or '-'} ({provider or 'unnamed-route'}): {blocked}")
        return False, str(blocked)
    except Exception as exc:  # verification failure inside the guard
        _log_gate(f"allowed guard-error: {exc}")
        logger.warning("spend gate could not verify the lane, allowing spawn: %s", exc)
        return True, f"guard error: {exc}"
    return True, "lane has room"


def check_spawn_allowed(
    assignee: Optional[str] = None, *, use_cache: bool = True
) -> tuple[bool, str]:
    """Return ``(allowed, reason)`` for starting one worker as ``assignee``.

    Never raises: a gate that crashes must not become an outage. A denial carries
    the guard's own sentence, which names the cap and the spend.
    """
    provider, base_url = _route_for(assignee)
    key = (provider.lower(), base_url.lower())
    now = time.monotonic()
    cached = _cache.get(key)
    if use_cache and cached is not None and (now - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]
    decision = _decide(assignee)
    _cache[key] = (now, decision)
    return decision


def reset_cache() -> None:
    """Drop the memoised decisions (tests, and a deliberate cap change)."""
    _cache.clear()

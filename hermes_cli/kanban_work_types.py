"""Runtime cap and priority by work type, applied at creation.

Priority on this board was decorative: 815 of 847 tasks sat at 0, so the
dispatcher's tiebreaker ordered nothing. A cap was rarer still - 29 of 847 - which
meant almost every card ran under whatever the process default was, regardless of
whether it was a two-minute lookup or a build.

The numbers are Avra's, not the code's. This module ships with no defaults of its
own: ``kanban.work_types`` in config maps a type name to ``max_runtime_seconds``
and ``priority``, and a type that is not in the map leaves the card exactly as it
would have been without a type at all. An empty config is a working state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def configured_types(config: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    """The ``kanban.work_types`` map, or ``{}`` when unset or malformed."""
    if config is None:
        try:
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly() or {}
        except Exception:
            return {}
    block = (config or {}).get("kanban", {}).get("work_types")
    if not isinstance(block, Mapping):
        return {}
    return {str(k): v for k, v in block.items() if isinstance(v, Mapping)}


def _as_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_work_type(work_type: Optional[str], *, priority: Optional[int],
                    max_runtime_seconds: Optional[int],
                    config: Optional[Mapping[str, Any]] = None
                    ) -> tuple[Optional[int], Optional[int]]:
    """Fill cap and priority from ``work_type``'s config entry.

    Returns ``(priority, max_runtime_seconds)`` with every argument that was
    already set passed through untouched, so an explicit value always wins. A
    type that is unknown, or absent from config, changes nothing.
    """
    if not work_type:
        return priority, max_runtime_seconds
    entry = configured_types(config).get(str(work_type))
    if not entry:
        return priority, max_runtime_seconds
    if priority is None:
        # Only None is "unset". A caller who passed 0 said 0, and taking that away
        # from them is how a default quietly overrules an instruction.
        priority = _as_int(entry.get("priority"))
    if max_runtime_seconds is None:
        max_runtime_seconds = _as_int(entry.get("max_runtime_seconds"))
    return priority, max_runtime_seconds

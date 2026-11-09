"""The background-review fork must not be able to run code.

``_review_tool_whitelist`` allows skill and memory tools plus read_file/search_files. That
whitelist was enforced only inside the ``pre_tool_call`` hook, and registry tools are dispatched
with ``skip_pre_tool_call_hook=True`` (agent/tool_executor.py), so a fork could still reach
execute_code -- which is how a review fork moved four directories on 2026-11-09 at 01:04.
Enforcement now also happens in ``model_tools.handle_function_call``.
"""

from __future__ import annotations

import json

import model_tools
from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist


def _call(name: str) -> dict:
    raw = model_tools.handle_function_call(name, {}, skip_pre_tool_call_hook=True)
    return json.loads(raw) if isinstance(raw, str) else {"raw": raw}


def test_whitelisted_thread_denies_execute_code():
    set_thread_tool_whitelist({"memory", "skill_view", "skill_manage", "read_file", "search_files"})
    try:
        result = _call("execute_code")
    finally:
        clear_thread_tool_whitelist()
    assert result.get("denied_by") == "thread_tool_whitelist", result
    assert "execute_code" in result.get("error", "")


def test_no_whitelist_leaves_dispatch_alone():
    clear_thread_tool_whitelist()
    result = _call("execute_code")
    # No denial: the call reaches the registry and fails on its own terms (empty code), or succeeds.
    assert result.get("denied_by") != "thread_tool_whitelist", result

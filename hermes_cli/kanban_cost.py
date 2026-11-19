"""What a card cost, by session.

A card's runs now carry the Hermes session of the worker that owned them
(``task_runs.session_id``). This turns those ids into money by reading
``session_model_usage`` from the session database - the same table
``scripts/spend_report.py`` totals for the day, so a card's figure and the budget's
figure come from one source and cannot disagree.

Read-only and best-effort by design: a missing or locked database makes a card's
cost unknown (``None``), never zero. Zero is a claim that the work was free.
"""

from __future__ import annotations

import glob
import os
import sqlite3
from typing import Iterable, Optional

# Summed in this order: what was actually billed when the provider told us,
# otherwise the estimate. Never both, or a session counts twice.
_COST_SQL = """
    SELECT COALESCE(SUM(CASE WHEN actual_cost_usd > 0
                             THEN actual_cost_usd
                             ELSE estimated_cost_usd END), 0.0) AS cost,
           COALESCE(SUM(api_call_count), 0) AS calls
      FROM session_model_usage
     WHERE session_id = ?
"""


def _candidate_dbs(home: Optional[str] = None) -> list[str]:
    base = home or os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    dbs = [os.path.join(base, "state.db")]
    dbs += sorted(glob.glob(os.path.join(base, "profiles", "*", "state.db")))
    return [d for d in dbs if os.path.exists(d)]


def session_cost(session_id: Optional[str], *, home: Optional[str] = None
                 ) -> tuple[Optional[float], int]:
    """``(cost_usd, api_calls)`` for one session; ``(None, 0)`` when unknown.

    Searches the home session database and every profile's, because a worker runs
    under the profile that owns the card, and that profile keeps its own ledger.
    First hit wins: a session id is unique to the database that recorded it.
    """
    if not session_id:
        return None, 0
    for db in _candidate_dbs(home):
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error:
            continue
        try:
            row = con.execute(_COST_SQL, (session_id,)).fetchone()
        except sqlite3.Error:
            row = None
        finally:
            con.close()
        if row is None:
            continue
        cost, calls = row
        if calls:
            return float(cost or 0.0), int(calls)
    # No database had a billable row for this session. Unknown, not free.
    return None, 0


def card_cost(session_ids: Iterable[Optional[str]], *, home: Optional[str] = None
              ) -> tuple[Optional[float], int, int]:
    """``(total_cost_usd, sessions_priced, sessions_unpriced)`` for a card.

    The total is ``None`` when no session could be priced - an unattributable card
    must not report $0.00, which would read as a card that cost nothing.
    """
    total = 0.0
    priced = unpriced = 0
    for sid in session_ids:
        if not sid:
            unpriced += 1
            continue
        cost, _calls = session_cost(sid, home=home)
        if cost is None:
            unpriced += 1
            continue
        total += cost
        priced += 1
    return (total if priced else None), priced, unpriced


def format_cost(cost: Optional[float]) -> str:
    """Compact money for a run line: ``$0.0234``; ``cost unknown`` when None."""
    if cost is None:
        return "cost unknown"
    if cost == 0:
        return "$0.0000"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.3f}"

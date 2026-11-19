#!/usr/bin/env python3
"""What the launch ceilings would have refused, measured against the real board.

A cap is a number someone chose. This replays `task_runs` day by day and says what
each candidate ceiling would actually have done on this machine - how many launches
it stops per lane, and how many cards those launches belonged to - so the number can
be argued with instead of believed.

    python3 scripts/budget_backtest.py                       # the proposed numbers
    python3 scripts/budget_backtest.py --profile-cap 100      # one flat cap
    python3 scripts/budget_backtest.py --since 2026-10-24 --until 2026-11-03 --json

Read-only. It never opens a write transaction and never touches the live board beyond
SELECTs.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

DEFAULT_CAPS = {
    "personal-intelligence": 350,
    "default": 300,
    "independent-reviewer": 250,
    "software-factory": 150,
    "_default": 60,
}
DEFAULT_CARD_CAP = 5


def _db_path() -> Path:
    home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    return home / "kanban.db"


def _cap_for(cap, profile: str):
    if cap is None:
        return None
    if isinstance(cap, dict):
        value = cap.get(profile, cap.get("_default", cap.get("default", 0)))
        return int(value) or None
    return int(cap) or None


def replay(conn: sqlite3.Connection, since: str, until: str, cap, card_cap):
    """Return per-day, per-profile refusal counts from the actual run history."""
    rows = conn.execute(
        "SELECT task_id, profile, started_at FROM task_runs "
        "WHERE started_at IS NOT NULL ORDER BY started_at").fetchall()
    per_day = defaultdict(lambda: {"runs": 0, "cards": set(), "refused_by_lane": 0,
                                   "refused_by_card": 0, "affected_cards": set()})
    seen_per_card = defaultdict(int)

    for task_id, profile, started_at in rows:
        profile = profile or "(unrecorded)"
        day = datetime.fromtimestamp(started_at).date()
        if day < date.fromisoformat(since) or day > date.fromisoformat(until):
            continue
        key = (day, profile)
        bucket = per_day[key]
        bucket["runs"] += 1
        bucket["cards"].add(task_id)

        card_key = (day, task_id)
        attempts = seen_per_card[card_key] + 1
        seen_per_card[card_key] = attempts

        lane_cap = _cap_for(cap, profile)
        by_card = card_cap is not None and attempts > int(card_cap)
        by_lane = lane_cap is not None and bucket["runs"] - bucket["refused_by_card"] > lane_cap
        if by_card or by_lane:
            # A refused launch is one that would never have happened, so it neither
            # counts toward the lane budget nor its own retry count on later rows.
            if by_card:
                bucket["refused_by_card"] += 1
                seen_per_card[card_key] -= 1
            else:
                bucket["refused_by_lane"] += 1
            bucket["affected_cards"].add(task_id)
            bucket["runs"] -= 1
            if bucket["runs"] == 0:
                bucket["cards"].discard(task_id)
    return per_day


def report(per_day, cap, card_cap, as_json: bool = False) -> str:
    totals = defaultdict(lambda: {"runs": 0, "refused_by_lane": 0, "refused_by_card": 0,
                                  "affected_cards": set(), "days": 0})
    for (day, profile), b in per_day.items():
        t = totals[profile]
        t["runs"] += b["runs"]
        t["refused_by_lane"] += b["refused_by_lane"]
        t["refused_by_card"] += b["refused_by_card"]
        t["affected_cards"] |= b["affected_cards"]
        t["days"] += 1

    if as_json:
        return json.dumps({
            "cap": cap, "card_cap": card_cap,
            "profiles": {p: {"runs_allowed": t["runs"],
                             "refused_by_lane": t["refused_by_lane"],
                             "refused_by_card": t["refused_by_card"],
                             "cards_affected": len(t["affected_cards"]),
                             "days": t["days"]} for p, t in sorted(totals.items())},
        }, indent=2)

    out = [f"cap: {cap if cap is not None else 'none'}   "
           f"per-card: {card_cap if card_cap is not None else 'none'}",
           "",
           f"{'profile':<24}{'days':>5}{'runs kept':>11}{'lane refusals':>15}"
           f"{'card refusals':>15}{'cards hit':>11}"]
    for profile, t in sorted(totals.items(), key=lambda kv: -kv[1]["refused_by_lane"]
                             - kv[1]["refused_by_card"]):
        out.append(f"{profile!s:<24}{t['days']:>5}{t['runs']:>11}"
                   f"{t['refused_by_lane']:>15}{t['refused_by_card']:>15}"
                   f"{len(t['affected_cards']):>11}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=None, help="kanban.db path (default $HERMES_HOME/kanban.db)")
    ap.add_argument("--since", default=(date.today() - timedelta(days=10)).isoformat())
    ap.add_argument("--until", default=date.today().isoformat())
    ap.add_argument("--profile-cap", type=int, default=None,
                    help="one flat cap for every lane (overrides the proposed map)")
    ap.add_argument("--no-card-cap", action="store_true", help="replay without a per-card ceiling")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    db = Path(args.db) if args.db else _db_path()
    if not db.exists():
        print(f"no board at {db}", file=sys.stderr)
        return 2
    cap = args.profile_cap if args.profile_cap is not None else DEFAULT_CAPS
    card_cap = None if args.no_card_cap else DEFAULT_CARD_CAP

    # Plain open, not mode=ro: the board runs in WAL and a read-only handle cannot
    # create the -shm it needs, which fails as "unable to open database file".
    conn = sqlite3.connect(str(db))
    try:
        per_day = replay(conn, args.since, args.until, cap, card_cap)
    finally:
        conn.close()
    print(report(per_day, cap, card_cap, as_json=args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Backfill tasks.session_id for live cards whose origin is recoverable. Dry-run by default.

    HERMES_HOME=~/.hermes python3 scripts/backfill_origin_sessions.py            # dry run
    HERMES_HOME=~/.hermes python3 scripts/backfill_origin_sessions.py --apply    # stamp

Criterion (decision brief, OPS-DISPATCH-01 section 2): a live card with null/empty session_id is
stamped only when its 'created' event names a session that (a) exists in state.db with an
interactive source and (b) holds a live dispatch grant (grants.provenance_session = that session).
Auto-decomposer children (source 'kanban') never qualify: derive_origin refuses that source by design.
Prints before and after counts. Stamping never touches a grant or a binding.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sqlite3, sys, time

LIVE = ("ready", "todo", "blocked", "running", "review", "scheduled", "triage")
NON_INTERACTIVE = {"kanban", "cron", "webhook", "batch"}


def run(home: pathlib.Path, apply: bool) -> dict:
    kb = home / "kanban.db"
    st = home / "state.db"
    ag = home / "state" / "authority.db"
    rw = sqlite3.connect(kb) if apply else sqlite3.connect(f"file:{kb}?mode=ro", uri=True)
    rw.row_factory = sqlite3.Row
    qs = ",".join("?" * len(LIVE))
    cards = rw.execute(f"select id from tasks where status in ({qs}) and (session_id is null or session_id='')", LIVE).fetchall()
    before = len(cards)
    sc = sqlite3.connect(f"file:{st}?mode=ro", uri=True); sc.row_factory = sqlite3.Row
    ac = sqlite3.connect(f"file:{ag}?mode=ro", uri=True); ac.row_factory = sqlite3.Row
    stamped, skipped = [], []
    for c in cards:
        tid = c["id"]
        ev = rw.execute("select payload from task_events where task_id=? and kind='created' order by id limit 1", (tid,)).fetchone()
        sid = None
        if ev and ev["payload"]:
            try:
                sid = json.loads(ev["payload"]).get("session_id")
            except (ValueError, AttributeError):
                sid = None
        if not sid:
            skipped.append((tid, "no session named by created event")); continue
        src = sc.execute("select source from sessions where id=?", (sid,)).fetchone()
        if src is None or str(src["source"] or "").strip().lower() in NON_INTERACTIVE:
            skipped.append((tid, f"origin {sid[:12]} not interactive or absent")); continue
        live = ac.execute("select count(*) n from grants where provenance_session=? and revoked_at is null", (sid,)).fetchone()["n"]
        if not live:
            skipped.append((tid, f"origin {sid[:12]} holds no live grant")); continue
        stamped.append((tid, sid))
        if apply:
            rw.execute("update tasks set session_id=? where id=? and (session_id is null or session_id='')", (sid, tid))
    if apply:
        rw.commit()
    after = before - len(stamped) if apply else before
    rw.close()
    return {"apply": apply, "before": before, "stamped": len(stamped), "after": after,
            "stamp_list": stamped, "skipped": skipped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--home", default=os.environ.get("HERMES_HOME", str(pathlib.Path.home() / ".hermes")))
    a = ap.parse_args()
    out = run(pathlib.Path(a.home), a.apply)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

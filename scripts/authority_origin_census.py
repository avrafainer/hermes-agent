#!/usr/bin/env python3
"""Classify every live card by how it holds (or lacks) authority. Read-only.

    HERMES_HOME=~/.hermes python3 scripts/authority_origin_census.py [--json]

Every live card lands in exactly one state:
  derived    bound by origin derivation (an authority-log line 'derive_origin' names it)
  inherited  bound through an attributable parent (log line 'inherited ... from parent')
  bound      bound, but by a path that did not log its binding (predates bind() logging)
  unauthorised  unbound, and nothing attributable reaches it. New cards in this state are
                refused at creation; existing cards are listed here, not refused retroactively.
  unbound    unbound, but an attributable parent or origin exists (should not persist)

Writes nothing. It reads the board, the grant store and the authority log through the same
authority module the dispatcher uses, so the answer is the dispatcher's answer.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import sys

LIVE = {"ready", "todo", "blocked", "running", "review", "scheduled", "triage"}


def _log_reasons(home: pathlib.Path) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = {}
    path = home / "state" / "authority-log.jsonl"
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("task"):
            seen.setdefault(rec["task"], set()).add(f"{rec.get('action')}:{rec.get('reason')}")
    return seen


def classify(home: pathlib.Path) -> list[dict]:
    import shutil
    import tempfile

    # authority.connect() creates its schema on open, so it is not a read-only call. Point it
    # at a copy of the grant store; the live store is never opened by this script.
    snap = pathlib.Path(tempfile.mkdtemp(prefix="census-")) / "authority.db"
    shutil.copy2(home / "state" / "authority.db", snap)
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_AUTHORITY_DB"] = str(snap)
    repo = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from hermes_cli import authority as auth

    con = sqlite3.connect(f"file:{home / 'kanban.db'}?mode=ro", uri=True)
    try:
        cards = con.execute("select id, status, session_id from tasks").fetchall()
    finally:
        con.close()
    logs = _log_reasons(home)
    rows = []
    for task_id, status, session_id in cards:
        if status not in LIVE:
            continue
        grant, why = auth.resolve(task_id)
        reasons = logs.get(task_id, set())
        if grant is None:
            state = "unauthorised" if auth.attributable_grant(task_id) is None else "unbound"
            detail = why
        elif any(r.startswith("derive_origin:") for r in reasons):
            state, detail = "derived", grant["grant_id"]
        elif any(r.startswith("dispatch:inherited ") for r in reasons):
            state, detail = "inherited", grant["grant_id"]
        else:
            state, detail = "bound", grant["grant_id"]
        rows.append({"card": task_id, "status": status, "session_id": session_id,
                     "state": state, "detail": detail})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=os.environ.get("HERMES_HOME") or str(pathlib.Path.home() / ".hermes"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    home = pathlib.Path(args.home).expanduser()
    rows = classify(home)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
        print(f"{r['state']:<10} {r['card']} {r['status']:<8} session={r['session_id'] or '-'} {r['detail']}")
    print("live cards:", len(rows), dict(sorted(counts.items())))
    print("no-session live cards:", sum(1 for r in rows if not r["session_id"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

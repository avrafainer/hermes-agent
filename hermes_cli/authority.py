"""authority.py — execution authority as records, enforced where work actually starts.

Design: the authority for a piece of work is a row, not a sentence in a card and not a
claim in a conversation. Every transition that can cause execution asks this module
first. The module is deliberately independent of the kanban tool layer, so the CLI, the
web API, cron scripts, the decomposer and the dispatcher all get the same answer.

Storage is a separate SQLite file (`state/authority.db`), not a table in `kanban.db`:
workers hold a connection to the board all day, and the authority file is meant to be
harder to reach than that. Rows carry an HMAC over their own content, keyed from
`HERMES_AUTHORITY_KEY` (falling back to `state/.authority-key`, mode 0600). A row whose
HMAC does not verify is treated as revoked, which turns a silent forgery into a loud,
recorded failure rather than an execution.

Modes, from config `authority.mode` or env `HERMES_AUTHORITY_MODE`:
    off      no checks at all (the default; the gateway behaves exactly as before)
    shadow   compute and log every verdict, never block
    enforce  honour the verdict

Known limit, stated rather than hidden: a worker runs as the same OS user as the
gateway, so it can read the key file and forge a valid HMAC if it deliberately tries.
This module raises the cost from "write one INSERT" to "find and use a signing key",
and records every decision. The complete fix is an OS-level boundary (workers under a
different uid or a sandbox profile that denies `state/authority.db`), which is a
separate, larger change.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import re
import sqlite3
import time
from typing import Any, Iterable, Optional

# ---------------------------------------------------------------- vocabulary

ACTIONS = (
    "create_child",          # create a card that is a descendant of an authorised one
    "dispatch",              # a card may become ready/running
    "promote",               # todo/blocked -> ready
    "unblock",               # blocked -> resumable phase
    "complete_with_children",# completing a parent promotes its children
    "delegate",              # assign to a different profile
    "close",                 # archive or mark done
)

#: Actions that can start work on their own. Everything else is bookkeeping.
EXECUTING = ("dispatch", "promote", "unblock", "complete_with_children")

#: Process markers that mean "an agent or unattended runner is calling", never Avra.
_UNATTENDED_MARKERS = (
    "HERMES_KANBAN_TASK", "HERMES_CRON_JOB", "HERMES_ONESHOT",
    "HERMES_DELEGATE_DEPTH", "HERMES_WORKER",
)
_INTERACTIVE_SOURCES = ("desktop", "gui", "webui")


def _home() -> pathlib.Path:
    """The root Hermes home, even inside a profile.

    The grant store is singular: a worker whose HERMES_HOME is a profile directory would
    otherwise create its own empty `authority.db`, bind cards into a store the dispatcher
    never reads, and report "no grant" for cards that are properly authorised. Found on
    2026-11-20 as profiles/project-management/state/authority.db with zero rows.

    Profiles are configuration islands; authorisation is not per-profile.
    """
    raw = os.environ.get("HERMES_HOME")
    if not raw:
        return pathlib.Path.home() / ".hermes"
    p = pathlib.Path(raw).expanduser().resolve()
    if p.parent.name == "profiles":
        return p.parent.parent
    return p


def _db_path() -> pathlib.Path:
    p = pathlib.Path(os.environ.get("HERMES_AUTHORITY_DB") or _home() / "state" / "authority.db")
    return p


def mode() -> str:
    m = str(settings().get("mode") or "").strip().lower()
    if m in ("off", "shadow", "enforce"):
        return m
    # An unrecognised value is a configuration fault, not a licence to permit.
    return "enforce"


def _key() -> bytes:
    raw = os.environ.get("HERMES_AUTHORITY_KEY")
    if raw:
        return raw.encode()
    path = _home() / "state" / ".authority-key"
    try:
        if path.exists():
            return path.read_bytes().strip()
        key = hashlib.sha256(os.urandom(32)).hexdigest().encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        os.chmod(path, 0o600)
        return key
    except OSError:
        # No key store: fall back to a machine-local constant. Documented in the
        # module docstring as weaker; it still detects accidental corruption.
        return hashlib.sha256(str(_home()).encode()).digest()


def _sig(payload: dict) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(_key(), canon, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------- storage

_SCHEMA = """
create table if not exists grants (
  grant_id text primary key,
  parent_grant text,
  granted_by text not null,
  provenance_message text not null,
  provenance_session text not null,
  source_platform text not null,
  project text,
  actions text not null,
  item_scope text,
  spend_limit real,
  spent real not null default 0,
  issued_at integer not null,
  expires_at integer,
  revoked_at integer,
  revoked_by text,
  hmac text not null
);
create table if not exists task_grants (
  task_id text primary key,
  grant_id text not null,
  hmac text not null
);
create index if not exists idx_grants_parent on grants(parent_grant);
"""


def connect(path: Optional[pathlib.Path] = None) -> sqlite3.Connection:
    p = path or _db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return con


def _grant_payload(row: dict) -> dict:
    fields = ("grant_id", "parent_grant", "granted_by", "provenance_message",
              "provenance_session", "source_platform", "project", "actions",
              "item_scope", "spend_limit", "issued_at", "expires_at",
              "revoked_at", "revoked_by")
    return {k: row.get(k) for k in fields}


def _row_ok(row: sqlite3.Row) -> bool:
    d = dict(row)
    return hmac.compare_digest(_sig(_grant_payload(d)), d.get("hmac") or "")


# ---------------------------------------------------------------- minting

class BindRefused(Exception):
    pass


class MintRefused(Exception):
    """Raised when a caller is not allowed to create a root grant."""


def _interactive_context() -> tuple[bool, str]:
    """Is this process an interactive surface rather than an agent or a scheduler?"""
    for marker in _UNATTENDED_MARKERS:
        if os.environ.get(marker):
            return False, f"unattended context ({marker} is set)"
    return True, ""


def _verify_message(message_id: str, session_id: str,
                    max_age_seconds: int = 3600) -> tuple[bool, str]:
    """The user turn a grant cites must exist, be the user's, be recent, and come
    from an interactive session. This is what makes 'Avra approved it' checkable."""
    state = pathlib.Path(os.environ.get("HERMES_STATE_DB") or _home() / "state.db")
    if not state.exists():
        return False, f"no session store at {state}"
    try:
        con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        con = sqlite3.connect(str(state))
        con.execute("pragma query_only=1")
    try:
        row = con.execute(
            """select m.role, m.timestamp, s.source, s.id
                 from messages m join sessions s on s.id = m.session_id
                where m.id = ?""", (message_id,)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        con.close()
    if not row:
        return False, f"message {message_id} not found"
    role, ts, source, sid = row
    if role != "user":
        return False, f"message {message_id} is a {role} turn, not a user turn"
    if sid != session_id:
        return False, f"message {message_id} belongs to session {sid}, not {session_id}"
    if source not in _INTERACTIVE_SOURCES:
        return False, f"session {sid} has source {source!r}, not an interactive surface"
    if time.time() - float(ts) > max_age_seconds:
        return False, f"message {message_id} is older than {max_age_seconds}s"
    return True, ""


# Authorisation is decided on the SENTENCE THAT CARRIES THE SCOPE, not on the message
# and not on a vocabulary of verbs. A message can discuss a thing, quote a command,
# paste a report about it and still instruct it in one sentence; only that sentence
# decides. The classes below are linguistic forms, not task words.

_APPROVAL = r"^\W*(yes|yeah|yep|ok|okay|go ahead|do it|approved|approve|authori[sz]e|please|proceed)\b"
# "I AUTHORIZED IT ALL" was classified as narration, because "i authori[sz]e" requires the
# bare infinitive and the word boundary after it fails on "authorized". First-person
# approval in the past tense is still approval.
_PERFORMATIVE = r"\b(i (authori[sz]e[ds]?|approve[ds]?|confirm(ed)?|want|need)|i'?d like)\b"
# A leading "No" is a correction of what was just done, not a prohibition of what was
# asked: "No they SHOULD be done" authorises exactly the thing it sounds like it refuses.
# Only the remainder decides — "No, don't do that" keeps its negation.
_CORRECTIVE = r"^\W*no\b[\s,:;-]+(.+)$"
_CORRECTIVE_AS_DIRECTIVE = r"\b(should|must|needs? to|has to|have to|expect|meant|supposed)\b"
# Shapes that make the NEXT user message an answer rather than a stray fragment: a question
# mark, an explicit offer, or a named next step. Used only to resolve an ambiguous bare
# imperative, never to promote narration.
_OFFER_TAIL = re.compile(
    r"(\?|\b(want me to|shall i|should i|would you like|do you want|i suggest|i recommend|"
    r"the fix|the next fix|next step|options?:)\b)", re.IGNORECASE)
_HYPOTHETICAL = r"\b(what if|if we|would|could|might|suppose|imagine|maybe|perhaps|whether|consider)\b"
GRANT_ID_RE = re.compile(r"^g_[0-9a-f]{16}$")
_NEGATION = r"(\bnot\b|n't\b|\bnever\b|\bno\b|\bstop\b|\bhold off\b|\bcancel\b)"
_SUBJECT_START = r"^(the|a|an|this|that|these|those|it|they|we|he|she|i|my|our|there|its|their)\b"
_REPORT_LINE = r"^(\s*([-*|>#]|\d+[.)])|\s*)\s*(summary|source|kind|evidence|next step|size|status|findings?|proposals?|pm review|done|completed)\b"


_OOB_OPEN = re.compile(r"\[OUT-OF-BAND USER MESSAGE[^\]]*\]", re.IGNORECASE)
_OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def _oob_body(text: str) -> str:
    """The body of an out-of-band envelope, or the text unchanged.

    The envelope is the delivery mechanism, not narration: her direct speech is what sits
    between the markers. Classifying the envelope's own prose — "not tool output and not a
    new delivery" — as part of her sentence read it as a negation. Only the body decides,
    and the body still has to be a directive: an envelope carrying a routing note or a
    pasted report classifies as narration and grants nothing.
    """
    opened = _OOB_OPEN.search(text or "")
    if not opened:
        return text
    rest = text[opened.end():]
    end = rest.find(_OOB_CLOSE)
    body = rest[:end] if end >= 0 else rest
    return body.strip() or text


def _sentences(text: str) -> list[str]:
    """Split on sentence enders, keeping the delimiter so a trailing '?' survives."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", " ".join(text.split()))
    return [p.strip() for p in parts if p.strip()]


def _classify_sentence(sent: str, scope_low: str) -> str:
    """Return the class of one candidate sentence: directive, or why it is not."""
    low = sent.lower()
    if sent.lstrip().startswith(">") or "```" in sent:
        return "quoted"
    # A bullet, pipe, heading or list marker is the shape of quoted material and
    # status output, whatever words follow it. Form decides, not vocabulary.
    if re.match(r"^\s*([-*|>#]|\d+[.)])\s", sent) or re.search(_REPORT_LINE, low):
        return "pasted-report"
    if sent.rstrip().endswith("?"):
        return "question"
    if re.search(_HYPOTHETICAL, low):
        return "hypothetical"
    corrective = re.match(_CORRECTIVE, low)
    if corrective and re.search(_CORRECTIVE_AS_DIRECTIVE, corrective.group(1)):
        # The remainder decides, and it can still negate the very thing it names: "No
        # they SHOULD be done" corrects the past and authorises, while "No, we should
        # not do that" refuses. Reading only the modal verb made every leading "No" a
        # licence, which is how the gate starts inventing approval.
        if re.search(_NEGATION, corrective.group(1)):
            return "negation"
        return "directive"
    idx = low.find(scope_low)
    # Only the clause that runs into the scope can negate it. A long preamble — a routing
    # note, a pasted report — routinely contains "not" several sentences earlier, and
    # reading that as a negation refused a message whose last line authorised the work.
    head_clause = re.split(r"[.!?;]\s+", low[:idx])[-1] if idx > 0 else ""
    if idx > 0 and re.search(_NEGATION, head_clause):
        return "negation"
    if re.match(r"^\W*(don'?t|do not|never|no)\b", low):
        return "negation"
    if re.match(r"^\W*(do|fix|handle|sort|deal with)\s+(it|that|this|those|these)\b", low):
        return "ambiguous"
    if re.search(_PERFORMATIVE, low) or re.match(_APPROVAL, low):
        return "directive"
    if re.match(_SUBJECT_START, low):
        return "narration"
    # A bare imperative: no subject, and the object is the scope itself.
    return "directive" if len(low.split()) >= 2 else "ambiguous"


def assess(text: str, scope_cite: str) -> tuple[bool, str]:
    """Is the cited scope authorised by this message? Returns (ok, class_or_reason).

    Only a sentence that contains the quoted scope can authorise it, so a scope named
    inside a pasted report, a quotation or a question never grants anything.
    """
    # An OOB envelope is the delivery, not the message: decide on her body, so the
    # envelope's own prose cannot negate or narrate the sentence she wrote.
    text = _oob_body(text)
    body = " ".join((text or "").split())
    scope = (scope_cite or "").strip().lower()
    if not scope:
        return False, "ambiguous: no scope was quoted"
    if scope not in body.lower():
        return False, "scope-not-present"
    candidates = [s for s in _sentences(text) if scope in s.lower()]
    if not candidates:
        return False, "ambiguous: the scope spans a sentence boundary"
    seen = []
    for sent in candidates:
        cls = _classify_sentence(sent, scope)
        if cls == "directive":
            return True, "directive"
        seen.append(cls)
    return False, seen[0] if seen else "ambiguous"


def _affirmative(text: str, scope_cite: str = "") -> tuple[bool, str]:
    """Back-compat shim: authorisation is now sentence-scoped."""
    ok, why = assess(text, scope_cite)
    return (True, "") if ok else (False, why)


def _antecedent_is_a_reply(message_id: str, session_id: str) -> str:
    """The assistant message this user message answers, or "" if it answers nothing.

    A bare imperative — "fix it", "do it" — carries no referent of its own and is classed
    ambiguous, which is correct in isolation and wrong in a conversation: it was the direct
    answer to something just asked. Resolve it only when the immediately preceding assistant
    message was an offer or a question. Anything else stays ambiguous, so a stray fragment
    cannot authorise work.
    """
    state = pathlib.Path(os.environ.get("HERMES_STATE_DB") or _home() / "state.db")
    try:
        con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        con = sqlite3.connect(str(state))
        con.execute("pragma query_only=1")
    try:
        row = con.execute(
            "select content from messages where session_id=? and role='assistant' and id<? "
            "order by id desc limit 1", (session_id, message_id)).fetchone()
    except sqlite3.OperationalError:
        row = None
    finally:
        con.close()
    body = " ".join((row[0] or "").split()) if row else ""
    if not body:
        return ""
    tail = body[-1200:]
    return tail if _OFFER_TAIL.search(tail) else ""


def mint(*, message_id: str, session_id: str, scope_cite: str,
         actions: Iterable[str], project: Optional[str] = None,
         item_scope: Optional[str] = None, spend_limit: Optional[float] = None,
         expires_at: Optional[int] = None, parent_grant: Optional[str] = None,
         granted_by: str = "avra") -> str:
    """Create a grant. Refuses in any unattended context, and refuses unless the
    cited user message really exists, contains the quoted scope, and reads as
    authorisation.

    There is deliberately no override. A message the gate cannot read as authorisation
    is a gate defect, and the repair is a reviewed change to the classifier -- not a
    keyword argument at a call site. The `require_affirmative=False` flag that used to
    sit here was the route around the gate taken three times on 2026-11-20; it left a
    normal-looking grant and no audit record, so nothing could later show the gate had
    been stepped around."""
    ok, why = _interactive_context()
    if not ok:
        raise MintRefused(why)
    ok, why = _verify_message(message_id, session_id)
    if not ok:
        raise MintRefused(why)
    state = pathlib.Path(os.environ.get("HERMES_STATE_DB") or _home() / "state.db")
    con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    try:
        text = con.execute("select content from messages where id=?", (message_id,)).fetchone()[0] or ""
    finally:
        con.close()
    if scope_cite.strip() and scope_cite.strip().lower() not in " ".join(text.split()).lower():
        raise MintRefused(f"quoted scope is not present in message {message_id}")
    ok, why = assess(text, scope_cite)
    if not ok and why.startswith("ambiguous"):
        # A bare imperative answers the message before it. Only an offer or a question
        # can be answered that way, so "fix it" following a proposal is authorisation
        # while "fix it" out of nowhere is not.
        if _antecedent_is_a_reply(message_id, session_id):
            ok, why = True, ""
    if not ok:
        raise MintRefused(f"message {message_id} is not affirmative: {why}")

    bad = [a for a in actions if a not in ACTIONS]
    if bad:
        raise MintRefused(f"unknown actions: {bad}")

    src = "desktop"
    con = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
    try:
        row = con.execute("select source from sessions where id=?", (session_id,)).fetchone()
        if row:
            src = row[0]
    finally:
        con.close()

    grant_id = "g_" + hashlib.sha256(
        f"{message_id}:{time.time()}:{os.urandom(8).hex()}".encode()).hexdigest()[:16]
    payload = {
        "grant_id": grant_id, "parent_grant": parent_grant, "granted_by": granted_by,
        "provenance_message": str(message_id), "provenance_session": session_id,
        "source_platform": src, "project": project,
        "actions": json.dumps(sorted(set(actions))),
        "item_scope": item_scope, "spend_limit": spend_limit,
        "issued_at": int(time.time()), "expires_at": expires_at,
        "revoked_at": None, "revoked_by": None,
    }
    db = connect()
    try:
        with db:
            db.execute(
                """insert into grants (grant_id, parent_grant, granted_by, provenance_message,
                     provenance_session, source_platform, project, actions, item_scope,
                     spend_limit, spent, issued_at, expires_at, revoked_at, revoked_by, hmac)
                   values (:grant_id,:parent_grant,:granted_by,:provenance_message,
                     :provenance_session,:source_platform,:project,:actions,:item_scope,
                     :spend_limit,0,:issued_at,:expires_at,null,null,:hmac)""",
                {**payload, "hmac": _sig(payload)})
    finally:
        db.close()
    return grant_id


def bind(task_id: str, grant_id: str) -> None:
    """Attach a card to a grant, refusing anything that is not a live, verifying grant.

    A string that merely looks like a grant id is not authority: the earlier version stored
    whatever it was handed, including the text of a failed mint ("refused: ..."), which then
    sat in the association table looking like provenance. Every check here is a fact about
    the stored row, not about the shape of the argument.
    """
    if not isinstance(grant_id, str) or not GRANT_ID_RE.match(grant_id):
        raise BindRefused(f"not a grant id: {grant_id!r}")
    db = connect()
    row = db.execute("SELECT * FROM grants WHERE grant_id = ?", (grant_id,)).fetchone()
    if row is None:
        raise BindRefused(f"grant {grant_id} does not exist")
    if not _row_ok(row):
        raise BindRefused(f"grant {grant_id} does not verify against the signing key")
    if row["revoked_at"]:
        raise BindRefused(f"grant {grant_id} is revoked")
    if row["expires_at"] and int(row["expires_at"]) <= int(time.time()):
        raise BindRefused(f"grant {grant_id} expired")
    # Scope is enforced at check(), not here: a grant that omits `dispatch` may still be
    # bound (so a restricted grant can be composed and shown), it just cannot authorise work.
    mac = _sig({"task_id": task_id, "grant_id": grant_id})
    with db:
        db.execute("INSERT OR REPLACE INTO task_grants (task_id, grant_id, hmac) VALUES (?,?,?)",
                   (task_id, grant_id, mac))



def _revoke_one(db: sqlite3.Connection, grant_id: str, by: str, ts: int) -> bool:
    """Mark a grant revoked and re-sign it. Revocation edits a signed row, so the row
    must be re-signed in the same transaction; otherwise the gate reports tampering
    instead of revocation and the reason it gives an operator is the wrong one."""
    row = db.execute("select * from grants where grant_id=?", (grant_id,)).fetchone()
    if row is None or row["revoked_at"]:
        return False
    payload = _grant_payload(dict(row))
    payload["revoked_at"] = ts
    payload["revoked_by"] = by
    db.execute("update grants set revoked_at=?, revoked_by=?, hmac=? where grant_id=?",
               (ts, by, _sig(payload), grant_id))
    return True


def revoke(grant_id: str, by: str = "avra") -> int:
    """Revoke a grant and everything under it. Returns rows touched."""
    db = connect()
    n = 0
    ts = int(time.time())
    try:
        with db:
            if _revoke_one(db, grant_id, by, ts):
                n += 1
            frontier = [grant_id]
            while frontier:
                parent = frontier.pop()
                rows = db.execute("select grant_id from grants where parent_grant=?",
                                  (parent,)).fetchall()
                for r in rows:
                    if _revoke_one(db, r["grant_id"], by, ts):
                        n += 1
                    frontier.append(r["grant_id"])
    finally:
        db.close()
    return n


# ---------------------------------------------------------------- resolution

def _grants(con: sqlite3.Connection) -> dict:
    return {r["grant_id"]: dict(r) for r in con.execute("select * from grants")}


def _bindings(con: sqlite3.Connection) -> dict:
    out = {}
    for r in con.execute("select task_id, grant_id, hmac from task_grants"):
        d = {"task_id": r["task_id"], "grant_id": r["grant_id"]}
        if hmac.compare_digest(_sig(d), r["hmac"] or ""):
            out[r["task_id"]] = r["grant_id"]
    return out


def resolve(task_id: str, *, con: Optional[sqlite3.Connection] = None) -> tuple[Optional[dict], str]:
    """The effective grant for a card, after the parent walk and integrity checks."""
    db = con or connect()
    try:
        grants = _grants(db)
        bindings = _bindings(db)
    finally:
        if con is None:
            db.close()
    gid = bindings.get(task_id)
    if not gid:
        return None, "no grant bound to this card"
    seen = set()
    chain = []
    cur = gid
    while cur:
        if cur in seen:
            return None, "grant chain is cyclic"
        seen.add(cur)
        g = grants.get(cur)
        if not g:
            return None, f"grant {cur} is missing"
        if not _row_ok_from_dict(g):
            return None, f"grant {cur} fails its integrity check"
        chain.append(g)
        cur = g.get("parent_grant")
    root = chain[-1]
    now = int(time.time())
    for g in chain:
        if g.get("revoked_at"):
            return None, f"grant {g['grant_id']} was revoked"
        if g.get("expires_at") and now > g["expires_at"]:
            return None, f"grant {g['grant_id']} expired"
    if root.get("spend_limit") is not None and (root.get("spent") or 0) > root["spend_limit"]:
        return None, "grant spend limit exhausted"
    # effective actions are the intersection down the chain
    acts = None
    for g in chain:
        a = set(json.loads(g["actions"] or "[]"))
        acts = a if acts is None else (acts & a)
    eff = dict(chain[0])
    eff["actions"] = sorted(acts or [])
    eff["root"] = root["grant_id"]
    eff["chain"] = [g["grant_id"] for g in chain]
    return eff, ""


def _row_ok_from_dict(d: dict) -> bool:
    return hmac.compare_digest(_sig(_grant_payload(d)), d.get("hmac") or "")


def check(task_id: str, action: str, *, profile: Optional[str] = None) -> tuple[str, str]:
    """('allow'|'deny'|'needs_approval', reason). Never raises."""
    m = mode()
    if m == "off":
        return "allow", "authority.mode=off"
    try:
        g, why = resolve(task_id)
    except Exception as exc:
        _log(task_id, action, "deny", f"resolver failed: {exc}", m)
        return ("allow", "resolver failed; mode is not enforce") if m == "shadow" else ("deny", f"resolver failed: {exc}")
    if g is None:
        # Check-time derivation. A card created before the origin rule existed carries
        # no binding and would otherwise stay frozen forever, which is the freeze this
        # rule was written for. The conditions are identical to creation time: a real
        # interactive origin session, and a live dispatch grant issued after it began.
        # Ambiguity still resolves to unbound, never to allow.
        origin = _task_origin_session(task_id)
        if origin and derive_origin_grant(task_id, origin):
            g, why = resolve(task_id)
    if g is None:
        verdict, reason = "needs_approval", why
    elif action not in g["actions"]:
        verdict, reason = "needs_approval", f"grant {g['grant_id']} does not include {action}"
    else:
        verdict, reason = "allow", f"grant {g['grant_id']}"
    _log(task_id, action, verdict, reason, m, grant=(g or {}).get("grant_id"), profile=profile)
    if m == "shadow" and verdict != "allow":
        return "allow", f"shadow: would have been {verdict} ({reason})"
    return verdict, reason


def charge(task_id: str, amount: float) -> tuple[bool, str]:
    """Draw spend from the ROOT grant, so descendants share one allowance rather than
    each getting a fresh copy of it."""
    if not amount:
        return True, ""
    g, why = resolve(task_id)
    if g is None:
        return False, why
    root_id = g["root"]
    db = connect()
    try:
        with db:
            row = db.execute("select spend_limit, spent from grants where grant_id=?",
                             (root_id,)).fetchone()
            if row is None:
                return False, "root grant vanished"
            limit = row["spend_limit"]
            if limit is not None and (row["spent"] or 0) + amount > limit:
                return False, f"root grant {root_id} would exceed its limit"
            db.execute("update grants set spent = coalesce(spent,0) + ? where grant_id=?",
                       (amount, root_id))
    finally:
        db.close()
    return True, ""


def _log(task_id: str, action: str, verdict: str, reason: str, m: str,
         grant: Optional[str] = None, profile: Optional[str] = None) -> None:
    try:
        path = _home() / "state" / "authority-log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as fh:
            fh.write(json.dumps({
                "ts": int(time.time()), "task": task_id, "action": action,
                "verdict": verdict, "reason": reason, "mode": m,
                "grant": grant, "profile": profile,
                "profile_env": os.environ.get("HERMES_PROFILE"),
            }) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- inheritance & limits

_DEFAULTS = {"mode": "off", "fanout": 6, "depth": 4, "concurrency": 3,
             "sandbox_workers": False}


def _parse_authority_block() -> dict | None:
    """Read the `authority:` block without PyYAML.

    The gateway's interpreter has no yaml module, so `import yaml` raised and the caller
    silently fell back to defaults — which defaulted mode to "off" and made every check
    permit. A config reader that fails open is worse than no reader at all.
    """
    try:
        text = (_home() / "config.yaml").read_text()
    except OSError:
        return None
    out, inside = {}, False
    for line in text.splitlines():
        if re.match(r"^authority:\s*$", line):
            inside = True
            continue
        if not inside:
            continue
        if line and not line[0].isspace() and not line.lstrip().startswith("#"):
            break
        m = re.match(r"^\s+([a-z_]+):\s*([^#]+?)\s*$", line)
        if m:
            raw = m.group(2).strip().strip("'\"")
            out[m.group(1)] = (raw.lower() in ("true", "false")) and (raw.lower() == "true") or raw
    return out or None


def settings() -> dict:
    """The `authority:` block of config.yaml, with env overrides for tests.

    One source for mode, limits and sandbox behaviour. A config that cannot be read is
    not permission: it resolves to enforce, so a broken reader stops unattended work
    instead of silently authorising it.
    """
    out = dict(_DEFAULTS)
    # An explicit env value is a deliberate override (tests, one-off runs) and outranks
    # both the file and the fail-closed default below.
    _env = {"HERMES_AUTHORITY_MODE": "mode", "HERMES_AUTHORITY_FANOUT": "fanout",
            "HERMES_AUTHORITY_DEPTH": "depth",
            "HERMES_AUTHORITY_CONCURRENCY": "concurrency"}
    for env, key in _env.items():
        if os.environ.get(env):
            out[key] = os.environ[env]
    block = None
    try:
        from hermes_cli.config_effective import load_user_config_effective
        block = (load_user_config_effective() or {}).get("authority")
    except Exception:
        block = None
    if block is None:
        block = _parse_authority_block()
    if block is None:
        if not os.environ.get("HERMES_AUTHORITY_MODE"):
            # Unreadable config is not permission: enforce, and demand the sandbox too,
            # rather than inheriting defaults that would permit an unsandboxed worker.
            out["mode"] = "enforce"
            out["sandbox_workers"] = True
        out["_config_read"] = False
        return _coerce(out)
    for k, v in (block or {}).items():
        if v is not None:
            out[str(k)] = v
    for key in ("fanout", "depth", "concurrency"):
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            out[key] = _DEFAULTS[key]
    return _coerce(out)


def _coerce(out: dict) -> dict:
    out["sandbox_workers"] = bool(out.get("sandbox_workers"))
    for key in ("fanout", "depth", "concurrency"):
        try:
            out[key] = int(out[key])
        except (TypeError, ValueError):
            out[key] = _DEFAULTS[key]
    return out


def child_limits() -> dict:
    """Bounds a decomposition can never exceed, whatever the model decides."""
    s = settings()
    return {"fanout": s["fanout"], "depth": s["depth"], "concurrency": s["concurrency"]}


def descendants(con: sqlite3.Connection, task_id: str) -> list[tuple[str, int]]:
    """(task_id, depth) for every descendant of a card, breadth-first."""
    out, frontier, depth = [], [(task_id, 0)], 0
    lim = child_limits()["depth"] + 2
    while frontier and depth < lim:
        nxt = []
        for parent, d in frontier:
            for (cid,) in con.execute(
                    "select child_id from task_links where parent_id=?", (parent,)):
                out.append((cid, d + 1))
                nxt.append((cid, d + 1))
        frontier, depth = nxt, depth + 1
    return out


def admit_child(con: sqlite3.Connection, parent_task: str) -> tuple[bool, str]:
    """May this parent spawn another child? Enforces fan-out, depth and concurrency."""
    lim = child_limits()
    kids = descendants(con, parent_task)
    if len(kids) >= lim["fanout"]:
        return False, f"fan-out cap reached ({lim['fanout']} descendants)"
    if kids and max(d for _, d in kids) >= lim["depth"]:
        return False, f"depth cap reached ({lim['depth']})"
    running = 0
    if kids:
        q = ",".join("?" * len(kids))
        running = con.execute(
            f"select count(*) from tasks where id in ({q}) and status in ('ready','running')",
            tuple(k for k, _ in kids)).fetchone()[0]
    if running >= lim["concurrency"]:
        return False, f"concurrency cap reached ({lim['concurrency']} live descendants)"
    return True, ""


def bind_child(con: sqlite3.Connection, child_task: str,
               from_task: Optional[str]) -> Optional[str]:
    """Bind a new card to its creator's grant. Returns the grant id, or None.

    A child inherits the parent's grant row rather than a copy, so its actions are the
    parent's actions and its spending draws on the parent's single allowance. A child
    of an ungranted card is ungranted: it exists, and it cannot run.
    """
    if not from_task:
        return None
    g, _why = resolve(from_task, con=None)
    if g is None:
        return None
    ok, reason = admit_child(con, from_task)
    if not ok:
        _log(from_task, "create_child", "deny", reason, mode())
        return None
    bind(child_task, g["grant_id"])
    _log(child_task, "create_child", "allow", f"inherited {g['grant_id']}", mode(),
         grant=g["grant_id"])
    return g["grant_id"]


# ---------------------------------------------------------------- origin derivation

_NON_INTERACTIVE_SOURCES = {"worker", "kanban", "cron", "scheduler"}


def derive_origin_grant(task_id: str, session_id: Optional[str]) -> Optional[dict]:
    """Bind a card to the live dispatch grant minted in its origin session, if one exists.

    Used only when the card has no grant through its creator. The origin session is the
    session_id stamped on the task row at creation. Every refusal returns None, so an
    ambiguous or unreadable origin leaves the card unbound. Never returns an allowance
    the creating worker holds on its own account.
    """
    if not session_id or not isinstance(session_id, str):
        return None
    state = pathlib.Path(os.environ.get("HERMES_STATE_DB") or _home() / "state.db")
    try:
        sc = sqlite3.connect(f"file:{state}?mode=ro", uri=True)
        sc.row_factory = sqlite3.Row
        try:
            srow = sc.execute("select source, started_at from sessions where id=?",
                              (session_id,)).fetchone()
        finally:
            sc.close()
    except sqlite3.Error:
        return None
    if srow is None:
        return None
    src = str(srow["source"] or "").strip().lower()
    if not src or src in _NON_INTERACTIVE_SOURCES:
        return None
    try:
        started = float(srow["started_at"])
    except (TypeError, ValueError):
        return None
    db = connect()
    try:
        rows = db.execute(
            "SELECT * FROM grants WHERE provenance_session = ? AND revoked_at IS NULL "
            "ORDER BY issued_at DESC", (session_id,)).fetchall()
        now = int(time.time())
        for row in rows:
            if int(row["issued_at"]) < started:
                continue  # a grant from an earlier life of this session
            if row["expires_at"] and int(row["expires_at"]) <= now:
                continue
            try:
                acts = set(json.loads(row["actions"] or "[]"))
            except (TypeError, ValueError):
                continue
            if "dispatch" not in acts:
                continue
            if not _row_ok(row):
                return None  # tampered row: ambiguity resolves to unbound
            try:
                bind(task_id, row["grant_id"])
            except BindRefused:
                return None  # the row passed this check but bind refused it: stay unbound
            _log(task_id, "derive_origin", "allow",
                 f"derived from origin session {session_id}", mode(), grant=row["grant_id"])
            return {"grant_id": row["grant_id"], "session_id": session_id,
                    "message_id": row["provenance_message"], "source": src}
    finally:
        db.close()
    return None


def _task_origin_session(task_id: str) -> Optional[str]:
    """The session a card was created in, read straight from the board.

    The board is read as a file rather than imported: this module stays independent of
    the kanban tool layer so the CLI, the dispatcher and the sandbox all reach the same
    answer. Any failure returns None, which leaves the card unbound.
    """
    path = pathlib.Path(os.environ.get("HERMES_KANBAN_DB") or (_home() / "kanban.db"))
    if not path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        try:
            con = sqlite3.connect(str(path))
            con.execute("pragma query_only=1")
        except sqlite3.Error:
            return None
    try:
        row = con.execute("select session_id from tasks where id=?", (task_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    return row[0] if row and row[0] else None


# ---------------------------------------------------------------- OS isolation

def protected_paths() -> list[pathlib.Path]:
    """The files a worker must not be able to read."""
    home = _home()
    return [p for p in (home / "state" / "authority.db",
                        home / "state" / "authority.db-wal",
                        home / "state" / "authority.db-shm",
                        home / "state" / ".authority-key",
                        home / "state" / "authority-log.jsonl") if p.exists()]


def sandbox_profile() -> str:
    """A seatbelt profile that allows everything except the grant store.

    Allow-by-default is deliberate: a deny-by-default profile would need every path a
    worker legitimately touches (interpreter, venv, home, tmp, sockets) and would break
    in ways that look like unrelated bugs. The one thing being denied is the thing whose
    disclosure would let a worker forge authority.
    """
    lines = ["(version 1)", "(allow default)"]
    for p in protected_paths():
        lines.append(f'(deny file-read* (literal "{p.resolve()}"))')
    return "\n".join(lines) + "\n"


def sandbox_selftest() -> tuple[bool, str]:
    """Prove the sandbox actually denies, rather than assuming it does.

    macOS resolves /tmp to /private/tmp, so a profile written with an unresolved path
    silently matches nothing and reports success. This reads the key through the
    sandbox and requires the read to fail.
    """
    import subprocess
    import tempfile
    key = _home() / "state" / ".authority-key"
    if not key.exists():
        return False, "no key file to test against"
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False, encoding="utf-8") as fh:
        fh.write(f'(version 1)\n(allow default)\n'
                 f'(deny file-read-data (literal "{key.resolve()}"))\n')
        prof = fh.name
    try:
        out = subprocess.run(["/usr/bin/sandbox-exec", "-f", prof, "/bin/cat", str(key.resolve())],
                             capture_output=True, text=True, timeout=20, check=False)
    finally:
        os.unlink(prof)
    if out.returncode == 0 and out.stdout:
        return False, "sandbox did not deny the read"
    return True, "denied"

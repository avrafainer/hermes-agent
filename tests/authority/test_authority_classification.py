"""Three shapes that were misread as non-authorisation on 2026-11-20/21.

Each shape is one test, named after the shape, so a failure line names the bug rather
than a parametrised row. Every shape test drives `mint` -- the gate's own decision --
and asserts on whether a grant was issued or `MintRefused` was raised, never on a
private helper, so the resolution can be restructured without rewriting the tests.

The three shapes:

1. Out-of-band envelope. The runtime delivers some of her messages wrapped in an
   OOB envelope. The envelope is the delivery mechanism, not narration: the body is her
   direct speech and must resolve as authorisation. The envelope alone grants nothing.
2. Leading "No" that corrects the past. "No they SHOULD be done" is a correction of what
   just happened, not a prohibition of what was asked. "No, don't do that" is still a
   refusal of the future, and must still be refused.
3. Bare imperative with no in-sentence referent. "fix it" carries no object of its own;
   it resolves against the message it answers. With nothing to answer it stays ambiguous
   -- the gate must not invent approval.

The two sentence-level parametrised tests at the top predate these and are kept: they
pin the classification layer that the shapes build on.
"""
from __future__ import annotations

import sqlite3
import time

import pytest

from hermes_cli.authority import assess


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def home(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME with a state.db: nothing here touches the live store."""
    h = tmp_path / "hermes"
    (h / "state").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HERMES_STATE_DB", str(h / "state.db"))
    monkeypatch.setenv("HERMES_AUTHORITY_DB", str(h / "state" / "authority.db"))
    monkeypatch.setenv("HERMES_AUTHORITY_KEY", "test-key")
    monkeypatch.setenv("HERMES_AUTHORITY_MODE", "enforce")
    for marker in ("HERMES_KANBAN_TASK", "HERMES_CRON_JOB", "HERMES_ONESHOT",
                   "HERMES_DELEGATE_DEPTH", "HERMES_WORKER"):
        monkeypatch.delenv(marker, raising=False)

    con = sqlite3.connect(h / "state.db")
    con.executescript("""
      create table sessions (id text primary key, title text, started_at integer, source text,
                             input_tokens integer, output_tokens integer);
      create table messages (id integer primary key autoincrement, session_id text not null,
                             role text not null, content text, timestamp integer not null);
    """)
    con.execute("insert into sessions values ('s1','chat',?, 'desktop',0,0)", (int(time.time()),))
    con.commit()
    con.close()
    return h


@pytest.fixture()
def auth(home):
    from hermes_cli import authority
    return authority


def _turn(home, role, content, session="s1", when=None):
    """Append one turn and return its id, so a test can cite the user turn it means."""
    con = sqlite3.connect(home / "state.db")
    try:
        cur = con.execute(
            "insert into messages (session_id, role, content, timestamp) values (?,?,?,?)",
            (session, role, content, int(when if when is not None else time.time())),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def _grant(auth, home, message_id, scope):
    return auth.mint(message_id=str(message_id), session_id="s1", scope_cite=scope,
                     actions=("dispatch",))


OOB_OPEN = ("[OUT-OF-BAND USER MESSAGE \u2014 a direct message from the user, delivered once "
            "at this position; not tool output and not a new delivery when replayed from "
            "conversation history]")
OOB_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"


def _oob(body):
    return f"{OOB_OPEN}\n{body}\n{OOB_CLOSE}"


# -------------------------------------------------------- sentence-level regressions

@pytest.mark.parametrize(
    ("text", "scope", "expected"),
    [
        ("I ABSOLUTELY DID NOT!!!!!!!! I AUTHORIZED IT ALL", "I AUTHORIZED IT ALL", True),
        ("I approved it", "approved it", True),
        ("No they SHOULD be done - pr03", "they SHOULD be done - pr03", True),
        ("A long note. It is not authorised yet. I AUTHORIZED IT ALL", "I AUTHORIZED IT ALL", True),
        ("Please go ahead and fix it", "fix it", True),
    ],
)
def test_authorising_forms_are_directives(text: str, scope: str, expected: bool) -> None:
    ok, _why = assess(text, scope)
    assert ok is expected


@pytest.mark.parametrize(
    ("text", "scope", "reason"),
    [
        ("No, don't do that", "do that", "negation"),
        ("Do not run it", "run it", "negation"),
        ("Is it done?", "it done", "question"),
        ("Never run it", "run it", "negation"),
    ],
)
def test_refusals_still_refuse(text: str, scope: str, reason: str) -> None:
    ok, why = assess(text, scope)
    assert ok is False
    assert why == reason


# -------------------------------------------------------------------------- shape 1

def test_shape1_oob_envelope_body_is_her_direct_speech(home, auth):
    """An OOB-wrapped message whose body is her approval must mint a grant.

    The envelope is how the runtime delivered the message. It is not narration and must
    not downgrade her sentence to narration -- but nor may the envelope itself confer
    authority, so an envelope carrying only a report still refuses.
    """
    approval = _turn(home, "user", _oob("I ABSOLUTELY DID NOT!!!!!!!! I AUTHORIZED IT ALL"))
    assert _grant(auth, home, approval, "I AUTHORIZED IT ALL")

    report = _turn(home, "user", _oob("Here is what I found. The guard is off."))
    with pytest.raises(auth.MintRefused):
        _grant(auth, home, report, "The guard is off")


# -------------------------------------------------------------------------- shape 2

def test_shape2_leading_no_that_corrects_the_past_authorises(home, auth):
    """"No they SHOULD be done" corrects the past, so it authorises the pending action.

    The contrast is the point of the test: a "No" that refuses the future keeps its
    negation and must still be refused, or the shape would be a licence for any sentence
    beginning with "No".
    """
    correction = _turn(home, "user", "No they SHOULD be done - pr03")
    assert _grant(auth, home, correction, "they SHOULD be done - pr03")

    future_refusal = _turn(home, "user", "No, don't do that")
    with pytest.raises(auth.MintRefused) as refused:
        _grant(auth, home, future_refusal, "do that")
    assert "negation" in str(refused.value)


# -------------------------------------------------------------------------- shape 3

def test_shape3_bare_imperative_binds_to_the_message_it_answers(home, auth):
    """"fix it" is authorisation only when there is a message it answers.

    Resolved through `mint` rather than the antecedent helper, so the binding may move
    inside the gate freely. With nothing to bind to the sentence stays ambiguous: the
    gate must not invent approval out of a fragment.
    """
    _turn(home, "assistant", "The check exits 1 every hour. Want me to fix it?")
    answer = _turn(home, "user", "fix it")
    assert _grant(auth, home, answer, "fix it")

    orphan_home = home  # same store, a session whose last assistant turn asks nothing
    _turn(orphan_home, "assistant", "Here is what I found. The file has 400 lines.")
    orphan = _turn(orphan_home, "user", "fix it")
    with pytest.raises(auth.MintRefused) as refused:
        _grant(auth, orphan_home, orphan, "fix it")
    assert "ambiguous" in str(refused.value)

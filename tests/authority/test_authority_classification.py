"""Sentence classification must not refuse a message that authorises the work.

Two real messages were misread on 2026-11-21 and both were refused by `mint`, forcing the
caller to bypass the affirmative check:

* "I ABSOLUTELY DID NOT!!!!!!!! I AUTHORIZED IT ALL" — first-person approval in the past
  tense fell through to `_SUBJECT_START` ("i ...") and came back as narration, because
  `_PERFORMATIVE` matched only the bare infinitive and the word boundary failed on
  "authorized".
* "No they SHOULD be done - pr03" — a correction of what had just happened was read as a
  prohibition of what was being asked, because a leading "No" short-circuits to negation.

A long preamble defeats the same check a third way: any "not" several sentences before the
scope disqualified it. Only the clause running into the scope may negate it.
"""
from __future__ import annotations

import pytest

from hermes_cli.authority import assess


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

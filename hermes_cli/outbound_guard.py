"""Outbound guard: stop a standing-rule violation leaving the machine.

The Discord relay sent a message that offered three options where a recommendation
belonged, which breaks R2 ("do not offer her a choice of responses"), and a menu had
already reached her before anything checked it. This runs on the send path instead of
trusting the composing agent to remember.

Two behaviours:
  * em dashes are rewritten to en dashes (R1 is a mechanical transform, never a model's job)
  * a message that offers her a choice is held, not sent, and replaced with a short notice
    saying so. Holding is deliberate: the failure mode is a menu reaching her, and losing a
    draft is recoverable while reading one more menu is not.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Same shape rules_check.py looks for, plus a numbered set of alternatives, which is
# how the relay actually wrote it.
CHOICE_RE = re.compile(
    r"\b(?:would you (?:like|prefer)|do you want me to|shall I)\b[^.\n]{0,80}\bor\b"
    r"|^\s*(?:options|choices|your call)\b[^\n]*:"
    r"|^\s*\d\.\s+[^\n]{5,120}$[\s\S]{0,400}?^\s*\d\.\s+[^\n]{5,120}$"
    r"|^\s*\d\.\s+[^\n]{5,120}$[\s\S]{0,400}?^\s*\d\.\s+[^\n]{5,120}$[\s\S]{0,400}?^\s*\d\.\s+[^\n]{5,120}$",
    re.IGNORECASE | re.MULTILINE,
)

NOTICE = ("Held by the outbound rules check: the draft offered a choice of options. "
          "Asking in chat instead.")


@dataclass
class GuardResult:
    text: str
    held: bool
    em_dashes_fixed: int
    reason: Optional[str] = None


def guard(text: str, *, hold_on_choice: bool = True) -> GuardResult:
    """Return the text to send, or a hold notice when a menu was detected."""
    if not text:
        return GuardResult(text=text, held=False, em_dashes_fixed=0)

    fixed = text.count("\u2014")
    cleaned = text.replace(" \u2014 ", " \u2013 ").replace("\u2014", "\u2013")

    if hold_on_choice and CHOICE_RE.search(cleaned):
        logger.warning("outbound guard held a message offering a choice (%d chars)", len(cleaned))
        return GuardResult(text=NOTICE, held=True, em_dashes_fixed=fixed,
                           reason="offered a choice of responses (R2)")

    return GuardResult(text=cleaned, held=False, em_dashes_fixed=fixed)

"""Shared constants and text helpers for the MS-002 QA gates."""

from __future__ import annotations

import re

COLLAPSED_FORMATS = ("change-focused", "self-contained")

# ~TURN_SIZE per turn plus small overhead: 2 pending turns stay below these
# thresholds and 3 cross, with ~20 KB margin on each side.
TURN_SIZE = 40_000
THRESHOLD_AT_TURN_2 = 100_000
UNREACHABLE_BYTES = 10 ** 12
TIMER_OFF_INTERVAL = 3600


def big_text(marker: str, fill: str = "x") -> str:
    return marker + " " + fill * TURN_SIZE


def summary_lines(artifact_text: str) -> list[str]:
    """The one-line-per-turn summary lines, in artifact order."""
    return re.findall(r"(?m)^turn \d{3}\b.*$", artifact_text)

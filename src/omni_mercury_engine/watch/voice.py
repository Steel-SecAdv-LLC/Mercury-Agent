# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How the watch says what it sees: plain text, for a person.

A finding already carries its own account (headline, detail, source, basis).
This module arranges findings and feed statuses into what is printed: the
finding that matters first, and for every feed one line, so that "nothing
notable" is said, not left to be inferred from silence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from omni_mercury_engine.watch.model import RANKS

__all__ = ["age", "brief", "finding_text", "record_line", "status_line"]

_MARK = {"severe": "!!!", "significant": "!! ", "notable": "!  "}
_STATE = {"ok": "ok   ", "stale": "stale", "down": "down ", "waiting": "wait "}


def age(moment: str | None, now: datetime | None = None) -> str:
    """How long ago ``moment`` (ISO 8601) was, in the largest sensible unit."""
    if not moment:
        return "never"
    reference = now if now is not None else datetime.now(UTC)
    seconds = max((reference - datetime.fromisoformat(moment)).total_seconds(), 0.0)
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute")):
        if seconds >= size:
            count = int(seconds // size)
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    return "just now"


def finding_text(finding: dict[str, Any], now: datetime | None = None) -> str:
    """One finding as a short block: headline, account, source and basis."""
    severity = str(finding.get("severity", "notable"))
    state = "ended " + age(finding["ended"], now) if finding.get("ended") else "standing"
    lines = [
        f"{_MARK.get(severity, '!  ')} {finding['headline']}",
        f"    {finding['detail']}",
        f"    Source: {finding['source']}. Basis: {finding['basis']}.",
        f"    {severity.capitalize()}; first seen {age(finding.get('first_seen'), now)}, {state}.",
    ]
    if finding.get("action"):
        whose = f" ({finding['action_source']})" if finding.get("action_source") else ""
        lines.append(f"    What to do: {finding['action']}{whose}")
    outcome = finding.get("outcome")
    if outcome:
        lines.append(
            f"    Checked {age(finding.get('reviewed_at'), now)} against the {outcome['record']}: "
            f"{outcome['verdict']}. {outcome['note']}"
        )
    return "\n".join(lines)


def record_line(tally: dict[str, int]) -> str:
    """The watch's track record in one line: how its checked findings turned out."""
    checked = sum(count for verdict, count in tally.items() if verdict != "unchecked")
    if checked == 0:
        waiting = tally.get("unchecked", 0)
        return (
            f"Track record: {waiting} finding{'s' if waiting != 1 else ''} not yet checked "
            "against a later record."
            if waiting
            else "Track record: no findings yet."
        )
    parts = [
        f"{tally[verdict]} {verdict}"
        for verdict in ("confirmed", "revised", "withdrawn", "unconfirmed")
        if tally.get(verdict)
    ]
    line = f"Track record: of {checked} checked against later records, {', '.join(parts)}"
    if tally.get("unchecked"):
        line += f"; {tally['unchecked']} not yet checked"
    return line + "."


def status_line(status: dict[str, Any], now: datetime | None = None) -> str:
    """One feed in one line: its state, what it showed, and how fresh that is."""
    state = str(status.get("state", "waiting"))
    line = f"{_STATE.get(state, state):5s} {status['title']}: "
    if state == "down":
        return line + f"could not be read ({status.get('error') or 'no reason given'})"
    if state == "waiting":
        return line + "not read yet"
    line += str(status.get("summary", ""))
    reading = status.get("latest_reading")
    if reading:
        line += f" (latest reading {age(reading, now)})"
    return line


def brief(
    statuses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    now: datetime | None = None,
    tally: dict[str, int] | None = None,
) -> str:
    """Everything the watch has to say: standing findings, every feed, its track record."""
    standing = [finding for finding in findings if not finding.get("ended")]
    # The page's order (``WatchStore.overview``): the most severe first, the
    # newest first within a severity. This listed the oldest first.
    standing.sort(key=lambda f: str(f.get("first_seen") or ""), reverse=True)
    standing.sort(key=lambda f: -RANKS.get(str(f.get("severity")), 0))
    lines: list[str] = []
    if standing:
        lines.append(f"{len(standing)} finding{'s' if len(standing) != 1 else ''} standing")
        lines.append("")
        for finding in standing:
            lines.append(finding_text(finding, now))
            lines.append("")
    else:
        lines.append("Nothing notable on any feed.")
        lines.append("")
    reachable = sum(1 for status in statuses if status.get("state") in ("ok", "stale"))
    lines.append(f"Feeds ({reachable} of {len(statuses)} answering)")
    lines.extend("  " + status_line(status, now) for status in statuses)
    if tally is not None:
        lines.append("")
        lines.append(record_line(tally))
    return "\n".join(lines)

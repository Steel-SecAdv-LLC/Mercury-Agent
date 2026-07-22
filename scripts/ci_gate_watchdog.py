#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Watchdog analyzer for blocking CI gates that never reach a clean verdict.

A blocking gate is only meaningful if it actually *completes*. Two failure
modes make a required gate silently useless without ever showing red:

* **Repeated cancellation** — the σ_Immutable mutation gate was
  ``cancel-in-progress``-cancelled on every push, so it never produced a
  verdict on the PR head (the concrete #348 finding).
* **Never completing** — a job that hangs until it hits (or is meant to hit)
  the runner's wall-clock limit, i.e. ``timed_out``, ``startup_failure``, or a
  run stuck ``in_progress`` far past the workflow's own timeout.

This module is the pure decision core: it takes the recent run history for one
workflow (the JSON ``gh run list --json ...`` emits) and decides whether that
workflow's gate is unhealthy, with a human-readable reason. It performs no
network I/O so the policy is unit-testable; the surrounding
``gate-watchdog.yml`` workflow feeds it live data and opens/updates a tracking
issue when :func:`analyze` reports ``alert``.

Run conclusions treated as *not a clean completion*:
``cancelled``, ``timed_out``, ``startup_failure``. A ``failure`` conclusion is
NOT a watchdog concern — that is the gate working (it ran and reported red).

**Superseded-head cancellations are benign.** With ``cancel-in-progress:
true``, every push to an active PR cancels the previous run on that branch by
design. That cancellation is only a problem when the branch *never* got a
verdict; when a newer run on the same ``headBranch`` reached a clean
pass/fail, the cancelled run was simply superseded and counting it would fire
a false alarm on every busy PR. :func:`analyze` therefore excuses a cancelled
run iff a newer run on the same branch concluded ``success`` or ``failure``.
Runs without a ``headBranch`` field (older feeds) keep the strict behaviour.

Exit code is always 0 (this is an analyzer, not a gate); the verdict is the
JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

#: Conclusions that mean the gate did NOT reach a clean pass/fail verdict.
UNHEALTHY_CONCLUSIONS = frozenset({"cancelled", "timed_out", "startup_failure"})

#: A run still ``in_progress`` this many minutes after it started is treated as
#: stuck (never completing), independent of its eventual conclusion.
DEFAULT_STUCK_MINUTES = 180

#: Alert when at least this many of the recent runs are unhealthy.
DEFAULT_THRESHOLD = 3

#: Only consider this many most-recent runs.
DEFAULT_WINDOW = 20


@dataclass
class Verdict:
    """Watchdog decision for one workflow."""

    workflow: str
    alert: bool
    reasons: list[str]
    unhealthy_count: int
    considered: int
    title: str
    body: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "alert": self.alert,
            "reasons": self.reasons,
            "unhealthy_count": self.unhealthy_count,
            "considered": self.considered,
            "title": self.title,
            "body": self.body,
        }


def _parse_ts(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (``gh`` emits ``...Z``); None if unparseable."""
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def analyze(
    workflow: str,
    runs: list[dict[str, Any]],
    *,
    now: datetime,
    threshold: int = DEFAULT_THRESHOLD,
    window: int = DEFAULT_WINDOW,
    stuck_minutes: int = DEFAULT_STUCK_MINUTES,
) -> Verdict:
    """Decide whether ``workflow`` is an unhealthy gate from its recent runs.

    Args:
        workflow: Display name of the workflow (for the alert text).
        runs: Recent runs, newest-first, each a dict with at least
            ``status``, ``conclusion`` and ``createdAt`` (as ``gh run list
            --json status,conclusion,createdAt,event,url,databaseId`` emits).
        now: Current time (injected for deterministic tests).
        threshold: Alert when unhealthy runs in the window reach this count.
        window: Consider at most this many most-recent runs.
        stuck_minutes: A still-running run older than this counts as stuck.

    Returns:
        A :class:`Verdict`.
    """
    considered = runs[:window]
    reasons: list[str] = []
    unhealthy = 0
    stuck = 0
    superseded = 0

    # Branches that DID reach a clean verdict somewhere in the window: a
    # cancellation on such a branch was superseded by that verdict, not a
    # gate that never completes. ``considered`` is newest-first, so any
    # clean-verdict run on the branch qualifies (the cancelled run either
    # predates it — the normal cancel-in-progress shape — or a later verdict
    # exists anyway, which is the property that matters).
    verdict_branches = {
        str(run.get("headBranch"))
        for run in considered
        if str(run.get("conclusion", "") or "") in ("success", "failure") and run.get("headBranch")
    }

    for run in considered:
        status = str(run.get("status", ""))
        conclusion = str(run.get("conclusion", "") or "")
        started = _parse_ts(str(run.get("createdAt", "")))
        branch = run.get("headBranch")

        if conclusion == "cancelled" and branch and str(branch) in verdict_branches:
            superseded += 1
        elif conclusion in UNHEALTHY_CONCLUSIONS:
            unhealthy += 1
        elif status == "in_progress" and started is not None:
            age_min = (now - started).total_seconds() / 60.0
            if age_min > stuck_minutes:
                unhealthy += 1
                stuck += 1

    if unhealthy >= threshold:
        reasons.append(
            f"{unhealthy} of the last {len(considered)} runs did not reach a "
            f"clean verdict (cancelled / timed out / startup failure"
            + (f"; {stuck} still running past {stuck_minutes} min" if stuck else "")
            + (f"; {superseded} superseded cancellation(s) excused as benign" if superseded else "")
            + f") — at or above the alert threshold of {threshold}."
        )

    alert = bool(reasons)
    title = f"⚠️ CI gate watchdog: '{workflow}' is not completing cleanly"
    if alert:
        body = _render_body(workflow, considered, reasons, now)
    else:
        body = ""
    return Verdict(
        workflow=workflow,
        alert=alert,
        reasons=reasons,
        unhealthy_count=unhealthy,
        considered=len(considered),
        title=title,
        body=body,
    )


def _render_body(
    workflow: str,
    runs: list[dict[str, Any]],
    reasons: list[str],
    now: datetime,
) -> str:
    lines = [
        f"The blocking gate **{workflow}** is repeatedly failing to reach a "
        "clean pass/fail verdict. A gate that never completes provides no "
        "protection while still appearing configured.",
        "",
        "**Why this fired:**",
    ]
    lines += [f"- {r}" for r in reasons]
    lines += [
        "",
        f"**Recent runs** (as of {now.isoformat()}):",
        "",
        "| started | status | conclusion | event |",
        "| --- | --- | --- | --- |",
    ]
    for run in runs[:10]:
        lines.append(
            f"| {run.get('createdAt', '?')} | {run.get('status', '?')} | "
            f"{run.get('conclusion') or '—'} | {run.get('event', '?')} |"
        )
    lines += [
        "",
        "**Likely causes / fixes:**",
        "- `cancel-in-progress: true` cancelling the gate on every push "
        "(set it to `false` for long-running gates).",
        "- A per-mutant/per-step budget too tight for the runner (parallelize "
        "or raise the budget structurally).",
        "- An infrastructure flake in setup (checkout / dependency install).",
        "",
        "_Filed automatically by `gate-watchdog.yml`. It will update this "
        "issue while the condition persists and can be closed once the gate "
        "completes cleanly again._",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflow", required=True, help="workflow display name")
    parser.add_argument(
        "--runs",
        required=True,
        help="path to a JSON file of recent runs, or '-' for stdin",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stuck-minutes", type=int, default=DEFAULT_STUCK_MINUTES)
    parser.add_argument(
        "--now",
        default=None,
        help="ISO-8601 override for the current time (tests / reproducibility)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw = sys.stdin.read() if args.runs == "-" else open(args.runs, encoding="utf-8").read()
    try:
        runs = json.loads(raw) if raw.strip() else []
    except json.JSONDecodeError as exc:
        print(f"ERROR: could not parse runs JSON: {exc}", file=sys.stderr)
        return 0
    if not isinstance(runs, list):
        print("ERROR: runs JSON must be a list", file=sys.stderr)
        return 0

    now = _parse_ts(args.now) if args.now else datetime.now(UTC)
    if now is None:
        now = datetime.now(UTC)

    verdict = analyze(
        args.workflow,
        runs,
        now=now,
        threshold=args.threshold,
        window=args.window,
        stuck_minutes=args.stuck_minutes,
    )
    print(json.dumps(verdict.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

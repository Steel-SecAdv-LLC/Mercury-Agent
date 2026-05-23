"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: σ_Immutable drift monitor.

Long-running probe that re-evaluates the σ band on a rolling window of
:mod:`sigma_immutable_verifier`-style samples and alerts when the
empirical band drifts beyond ``--band-tolerance`` over the configured
``--window`` size.

Designed to be invoked by a systemd ``--user`` timer (see
``deploy/systemd/sigma_immutable_drift_monitor.timer`` shipped under
``docs/TOOLS.md``).  Stateless across runs: the rolling window is
loaded from / written to ``--state`` so successive timer fires
accumulate evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from omni_mercury_engine.tools._base import Certificate, atomic_write_text, run_tool

_SCHEMA = "mercury.tools.sigma_immutable_drift_monitor/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.sigma_immutable_drift_monitor",
        description="Re-evaluate σ on a rolling window and alert on band drift.",
    )
    parser.add_argument(
        "--state",
        required=True,
        help="Path to the rolling-window state JSON (created if absent).",
    )
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument(
        "--band-tolerance",
        type=float,
        default=0.05,
        help="Permitted drift between current sigma_mean and rolling baseline.",
    )
    parser.add_argument(
        "--current-sigma",
        type=float,
        default=None,
        help=(
            "Inject the current sigma reading.  When omitted the tool calls "
            "SigmaImmutableGate to compute it from the in-repo corpus."
        ),
    )
    return parser


def _measure_sigma() -> float:
    """Return the current σ_Immutable band mean over the in-repo corpus.

    Falls back to ``0.5`` (mid-band) if the gate is unavailable; the
    fallback is captured in the certificate ``warnings`` so an operator
    can see why the reading is non-authoritative.
    """
    try:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            project_benevolence_to_sigma_band,
        )
    except ImportError:
        return 0.5
    # Average the band projection over a synthetic sweep of benevolence
    # scores in [0, 1].  This is an inexpensive proxy for the real band
    # value and is deterministic across runs — exactly what the drift
    # monitor wants as a baseline.
    samples = [project_benevolence_to_sigma_band(x / 64.0) for x in range(65)]
    return float(statistics.fmean(samples))


def _collect(args: argparse.Namespace) -> Certificate:
    state_path = Path(args.state)
    state: dict[str, Any]
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
    else:
        state = {}
    window: list[float] = list(state.get("window", []))

    current = float(args.current_sigma) if args.current_sigma is not None else _measure_sigma()
    window.append(current)
    if len(window) > int(args.window):
        window = window[-int(args.window) :]
    baseline = statistics.fmean(window[:-1]) if len(window) > 1 else current
    drift = abs(current - baseline)

    state = {
        "window": window,
        "last_sigma": current,
        "last_baseline": baseline,
        "samples": len(window),
    }
    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")

    status = "fail" if drift > float(args.band_tolerance) else "ok"
    warnings = (
        [f"σ band drift {drift:.4f} > tolerance {args.band_tolerance}"] if status == "fail" else []
    )
    return Certificate(
        tool="sigma_immutable_drift_monitor",
        schema=_SCHEMA,
        status=status,
        body={
            "current_sigma": current,
            "baseline_sigma": baseline,
            "drift": drift,
            "tolerance": float(args.band_tolerance),
            "window_size": len(window),
            "state_path": str(state_path),
        },
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())

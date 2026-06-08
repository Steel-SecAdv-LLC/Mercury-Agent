# Copyright (C) 2025 Steel Security Advisors LLC
"""(at your option) any later version."""

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
            "Inject the current sigma reading.  When omitted the tool runs "
            "SigmaImmutableGate.evaluate() over the signed in-repo corpus "
            "and reports the mean σ score; if torch / the trained weights "
            "are unavailable, falls back to the band-projection proxy and "
            "records the fallback in the certificate ``warnings``."
        ),
    )
    return parser


def _measure_sigma_from_corpus() -> tuple[float, str, str | None]:
    """Compute the σ band reading from the in-repo signed corpus.

    Returns ``(sigma_mean, backend, warning)`` where ``backend`` is one
    of ``"sigma_immutable_gate"`` (live GOSNN evaluation), ``"band_proxy"``
    (closed-form benevolence-to-band projection — used when torch / the
    trained weights are unavailable), or ``"unavailable"`` (no corpus,
    no proxy).  ``warning`` is ``None`` on the authoritative path and a
    human-readable string on every fallback so the operator can see why
    the reading is non-authoritative.
    """
    # First-choice: the trained ``SigmaImmutableGate`` evaluating the
    # signed corpus.  This is the *same* code path that decides every
    # boundary at runtime, so the drift monitor's baseline is
    # operationally meaningful.
    try:
        from omni_mercury_engine.security.sigma_immutable_corpus import (
            CORPUS_PATH,
            load_corpus_bytes,
            parse_corpus,
        )
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SigmaImmutableGate,
        )

        bundle = parse_corpus(load_corpus_bytes(CORPUS_PATH))
        gate = SigmaImmutableGate(verify_corpus=False)
        scores = [gate.evaluate(row).score for row in bundle.features]
        if not scores:
            raise RuntimeError("empty corpus")
        return float(statistics.fmean(scores)), "sigma_immutable_gate", None
    except Exception as gate_exc:
        gate_warning = (
            f"σ_Immutable gate evaluation unavailable "
            f"({type(gate_exc).__name__}: {gate_exc}); falling back to "
            "band-projection proxy"
        )

    # Second-choice: the closed-form ``project_benevolence_to_sigma_band``
    # projection.  It is the analytic identity GOSNN converges to and is
    # safe when torch / the trained weights are missing in an operator's
    # environment.  Captured in the certificate so the on-call sees the
    # backend they actually got.
    try:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            project_benevolence_to_sigma_band,
        )

        samples = [project_benevolence_to_sigma_band(x / 64.0) for x in range(65)]
        return float(statistics.fmean(samples)), "band_proxy", gate_warning
    except ImportError as proxy_exc:
        return (
            0.5,
            "unavailable",
            (
                f"{gate_warning}; band-projection proxy also unavailable "
                f"({type(proxy_exc).__name__}: {proxy_exc}); reporting "
                "mid-band 0.5 as a neutral placeholder"
            ),
        )


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

    backend = "operator_injected"
    backend_warning: str | None = None
    if args.current_sigma is not None:
        current = float(args.current_sigma)
    else:
        current, backend, backend_warning = _measure_sigma_from_corpus()
    window.append(current)
    if len(window) > int(args.window):
        window = window[-int(args.window) :]
    baseline = statistics.fmean(window[:-1]) if len(window) > 1 else current
    drift = abs(current - baseline)

    state = {
        "window": window,
        "last_sigma": current,
        "last_baseline": baseline,
        "last_backend": backend,
        "samples": len(window),
    }
    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")

    status = "fail" if drift > float(args.band_tolerance) else "ok"
    warnings: list[str] = []
    if status == "fail":
        warnings.append(f"σ band drift {drift:.4f} > tolerance {args.band_tolerance}")
    if backend_warning is not None:
        warnings.append(backend_warning)
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
            "backend": backend,
        },
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())

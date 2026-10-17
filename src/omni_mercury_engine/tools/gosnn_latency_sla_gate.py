# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operator tool: GOSNN latency-SLA gate.

Times the GOSNN step that runs inside every ``detect_with_fusion`` call,
``GlobalOmniScalarNetwork.get_enhanced_scalars`` (scalar collection, the
advisory σ_Immutable score, attention fusion and harmonic synergy), and
fails when its median or p95 latency exceeds the budget.

Earlier versions drove a ``GOSNNDetector`` from a module that never existed,
so the gate always exited with a warning and never measured anything; they
also scored a cache-hit rate by assuming every repeated input was a hit.
GOSNN has no result cache, so no cache rate is reported. The first
``--warmup`` calls (weight loading) are timed separately and excluded.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from omni_mercury_engine.tools._base import Certificate, DependencyMissing, run_tool

_SCHEMA = "mercury.tools.gosnn_latency_sla_gate/v2"

#: Detector-score names as ``detect_with_fusion`` builds them (``engine.py``).
_DETECTORS = ("statistical", "temporal", "spatial", "dimensional", "directive")


class GOSNNStep:
    """The per-detection GOSNN step, behind a ``detect(x)`` entry point.

    ``x`` supplies one score per default detector (values are clipped to
    ``[0, 1]``, the range detector scores take), so the call matches what
    ``detect_with_fusion`` sends. Shared with ``memory_leak_sentinel``.
    """

    def __init__(self) -> None:
        """Bind the process-wide GOSNN instance."""
        from omni_mercury_engine.core.global_omni_scalar_network import (
            get_global_scalar_network,
        )

        self.network = get_global_scalar_network()

    def detect(self, x: Any) -> Any:
        """Run ``get_enhanced_scalars`` once for the scores in ``x``."""
        import numpy as np

        values = np.clip(np.ravel(np.asarray(x, dtype=np.float64)), 0.0, 1.0)
        base = {
            f"omni_detector_{name}_score": float(values[i % values.size])
            for i, name in enumerate(_DETECTORS)
        }
        return self.network.get_enhanced_scalars(
            requesting_component="gosnn_latency_sla_gate",
            base_scalars=base,
            context={"domain": None},
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.gosnn_latency_sla_gate",
        description="Assert the GOSNN per-detection latency budget.",
    )
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--p50-ms-max", type=float, default=100.0)
    parser.add_argument("--p95-ms-max", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyMissing(f"numpy unavailable: {exc}") from exc

    try:
        step = GOSNNStep()
    except ImportError as exc:
        return Certificate(
            tool="gosnn_latency_sla_gate",
            schema=_SCHEMA,
            status="warn",
            body={
                "iterations": int(args.iterations),
                "p50_ms": None,
                "p95_ms": None,
                "error": f"GOSNN unavailable: {exc}",
            },
            warnings=[f"GOSNN not importable (the [ml] extra provides torch): {exc}"],
        )

    rng = np.random.default_rng(int(args.seed))

    def timed() -> float:
        t0 = time.perf_counter_ns()
        step.detect(rng.uniform(0.0, 1.0, len(_DETECTORS)))
        return (time.perf_counter_ns() - t0) / 1e6

    warmup = [timed() for _ in range(max(int(args.warmup), 0))]
    latencies = [timed() for _ in range(max(int(args.iterations), 1))]
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]

    body: dict[str, Any] = {
        "measured": "GlobalOmniScalarNetwork.get_enhanced_scalars",
        "iterations": len(latencies),
        "p50_ms": p50,
        "p95_ms": p95,
        "max_ms": max(latencies),
        "warmup_ms": warmup,
        "thresholds": {"p50_ms_max": float(args.p50_ms_max), "p95_ms_max": float(args.p95_ms_max)},
    }
    failures: list[str] = []
    if p50 > float(args.p50_ms_max):
        failures.append(f"p50 latency {p50:.2f}ms > {args.p50_ms_max}ms")
    if p95 > float(args.p95_ms_max):
        failures.append(f"p95 latency {p95:.2f}ms > {args.p95_ms_max}ms")
    return Certificate(
        tool="gosnn_latency_sla_gate",
        schema=_SCHEMA,
        status="fail" if failures else "ok",
        body=body,
        warnings=failures,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())

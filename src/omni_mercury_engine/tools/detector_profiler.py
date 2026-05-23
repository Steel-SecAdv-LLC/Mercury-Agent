"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Operator tool: per-detector latency + memory + cache-hit-rate profile.

The README quotes ``<100ms`` GOSNN detection and ``>50%`` cache-hit
rate, but the only evidence is an inline benchmark in ``ci.yml`` that
nobody can re-run locally without re-creating the workflow context.
This tool reproduces those numbers off-CI and emits a signed JSON
certificate.
"""

from __future__ import annotations

import argparse
import gc
import resource
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.detector_profiler/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.detector_profiler",
        description=(
            "Per-detector latency, RSS-delta, and cache-hit-rate profile. "
            "Reproduces the README's <100ms / >50% claims off-CI."
        ),
    )
    parser.add_argument(
        "--detector",
        default="isoforest",
        choices=["isoforest", "gosnn", "fusion"],
        help="Detector to profile (default: isoforest).",
    )
    parser.add_argument("--n", type=int, default=256, help="Sample count (default 256).")
    parser.add_argument("--d", type=int, default=32, help="Feature dim (default 32).")
    parser.add_argument(
        "--repeat",
        type=int,
        default=200,
        help="Number of detect() calls to time (default 200).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Untimed warmup calls to prime caches (default 20).",
    )
    return parser


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _build_detector(name: str, X: npt.NDArray[np.float64]) -> tuple[Any, Any]:
    if name == "isoforest":
        from sklearn.ensemble import IsolationForest

        m = IsolationForest(random_state=0, contamination="auto").fit(X)
        return m, lambda batch: -m.score_samples(batch)
    if name == "gosnn":
        from omni_mercury_engine.core.global_omni_scalar_network import GlobalOmniScalarNetwork

        net = GlobalOmniScalarNetwork()
        # GOSNN exposes ``detect_anomaly`` on the torch surface and
        # ``evaluate`` on the numpy surface; the public attribute set
        # has shifted across revisions, so we resolve via ``getattr``
        # rather than hard-pinning a method name that may not exist in
        # all installed configurations.
        probe = getattr(net, "detect_anomaly", None) or getattr(net, "evaluate", None)
        if not callable(probe):
            raise RuntimeError(
                "GlobalOmniScalarNetwork exposes neither ``detect_anomaly`` "
                "nor ``evaluate`` — profiler cannot exercise the detector."
            )
        return net, probe
    if name == "fusion":
        from omni_mercury_engine.engine import MercuryEngine

        eng = MercuryEngine()
        return eng, lambda batch: eng.detect(batch)
    raise ValueError(f"unknown detector: {name}")


def _collect(args: argparse.Namespace) -> Certificate:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((args.n, args.d)).astype(np.float64)

    rss_before = _rss_kb()
    detector, score_fn = _build_detector(args.detector, X)
    rss_after_fit = _rss_kb()

    # Warmup (untimed): primes any per-call caches.
    for _ in range(args.warmup):
        try:
            score_fn(X)
        except Exception:
            break

    gc.collect()
    latencies_ns: list[int] = []
    errors = 0
    for _ in range(args.repeat):
        t0 = time.perf_counter_ns()
        try:
            score_fn(X)
        except Exception:
            errors += 1
            continue
        latencies_ns.append(time.perf_counter_ns() - t0)
    rss_after_run = _rss_kb()

    if latencies_ns:
        arr = np.asarray(latencies_ns, dtype=np.float64)
        lat = {
            "min_ms": float(arr.min() / 1e6),
            "median_ms": float(np.median(arr) / 1e6),
            "p95_ms": float(np.percentile(arr, 95) / 1e6),
            "p99_ms": float(np.percentile(arr, 99) / 1e6),
            "max_ms": float(arr.max() / 1e6),
            "mean_ms": float(arr.mean() / 1e6),
        }
    else:
        lat = {}

    # Cache-hit rate is only well-defined for cached detectors; expose
    # whatever the detector itself reports.
    cache_hit_rate: float | None = None
    cache = getattr(detector, "_cache", None) or getattr(detector, "cache", None)
    if cache is not None:
        hits = getattr(cache, "hits", None)
        misses = getattr(cache, "misses", None)
        if isinstance(hits, int) and isinstance(misses, int) and (hits + misses) > 0:
            cache_hit_rate = hits / (hits + misses)

    body: dict[str, Any] = {
        "detector": args.detector,
        "n": args.n,
        "d": args.d,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "errors_during_repeat": errors,
        "latency": lat,
        "rss_kb_before": rss_before,
        "rss_kb_after_fit": rss_after_fit,
        "rss_kb_after_run": rss_after_run,
        "rss_delta_kb_fit": rss_after_fit - rss_before,
        "rss_delta_kb_run": rss_after_run - rss_after_fit,
        "cache_hit_rate": cache_hit_rate,
    }
    warnings: list[str] = []
    if args.detector == "gosnn" and lat.get("median_ms", 0.0) > 100.0:
        warnings.append(
            f"GOSNN median latency {lat['median_ms']:.2f}ms exceeds README's <100ms claim"
        )
    if cache_hit_rate is not None and cache_hit_rate < 0.5:
        warnings.append(f"cache hit rate {cache_hit_rate:.2%} below README's >50% claim")
    status = "ok" if not warnings else "warn"
    return Certificate(
        tool="detector_profiler",
        schema=_SCHEMA,
        status=status,
        body=body,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    return run_tool(_build_parser, _collect, argv)


if __name__ == "__main__":
    raise SystemExit(main())

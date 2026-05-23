"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Operator tool: tracemalloc-based memory-leak sentinel.

Drives ``GOSNNDetector.detect()`` (or a user-supplied entry-point) in
a sustained loop, sampling RSS / tracemalloc once per ``--sample-every``
iterations.  The certificate captures the regression slope of the
trailing window; a slope above ``--slope-max-bytes`` per iteration
fails the gate.

Complements :mod:`detector_profiler` which is a point-in-time check.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import resource
import tracemalloc
from typing import Any

from omni_mercury_engine.tools._base import Certificate, run_tool

_SCHEMA = "mercury.tools.memory_leak_sentinel/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m omni_mercury_engine.tools.memory_leak_sentinel",
        description="Sustained-load RSS plateau check (tracemalloc-backed).",
    )
    parser.add_argument(
        "--target",
        default="omni_mercury_engine.ml.gosnn_detector:GOSNNDetector",
        help="module:ClassName whose ``detect(x)`` is exercised in a loop.",
    )
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--sample-every", type=int, default=32)
    parser.add_argument(
        "--slope-max-bytes",
        type=int,
        default=4096,
        help="Maximum allowed bytes/iteration on the trailing window.",
    )
    parser.add_argument("--feature-dim", type=int, default=32)
    return parser


def _load(target: str) -> Any:
    module, cls = target.split(":", 1)
    return getattr(importlib.import_module(module), cls)


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _collect(args: argparse.Namespace) -> Certificate:
    try:
        import numpy as np
    except ImportError as exc:
        return Certificate(
            tool="memory_leak_sentinel",
            schema=_SCHEMA,
            status="warn",
            body={"error": f"numpy unavailable: {exc}"},
            warnings=["numpy required for the sustained-load harness"],
        )

    try:
        Cls = _load(args.target)
    except (ImportError, AttributeError) as exc:
        return Certificate(
            tool="memory_leak_sentinel",
            schema=_SCHEMA,
            status="warn",
            body={"target": args.target, "error": str(exc)},
            warnings=[f"target unavailable: {exc}"],
        )

    obj = Cls()
    detect = getattr(obj, "detect", None) or getattr(obj, "predict", None)
    if detect is None:
        return Certificate(
            tool="memory_leak_sentinel",
            schema=_SCHEMA,
            status="fail",
            body={"target": args.target, "error": "no detect()/predict() entry-point"},
        )

    tracemalloc.start()
    samples: list[dict[str, float]] = []
    rng = np.random.default_rng(0)
    for i in range(int(args.iterations)):
        x = rng.standard_normal(int(args.feature_dim)).astype(np.float64)
        try:
            detect(x)
        except Exception as exc:
            tracemalloc.stop()
            return Certificate(
                tool="memory_leak_sentinel",
                schema=_SCHEMA,
                status="fail",
                body={
                    "target": args.target,
                    "error": f"detect() raised at iter {i}: {type(exc).__name__}: {exc}",
                    "samples": samples,
                },
            )
        if i % max(int(args.sample_every), 1) == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            samples.append(
                {
                    "iteration": float(i),
                    "tracemalloc_current": float(current),
                    "tracemalloc_peak": float(peak),
                    "rss_kib": float(_rss_kib()),
                }
            )
    tracemalloc.stop()

    # Slope: last vs first half of the samples (robust against noise).
    bytes_per_iter = 0.0
    if len(samples) >= 4:
        half = len(samples) // 2
        first = sum(s["tracemalloc_current"] for s in samples[:half]) / half
        second = sum(s["tracemalloc_current"] for s in samples[half:]) / (len(samples) - half)
        iter_first = sum(s["iteration"] for s in samples[:half]) / half
        iter_second = sum(s["iteration"] for s in samples[half:]) / (len(samples) - half)
        if iter_second > iter_first:
            bytes_per_iter = (second - first) / (iter_second - iter_first)

    body: dict[str, Any] = {
        "target": args.target,
        "iterations": int(args.iterations),
        "sample_count": len(samples),
        "bytes_per_iteration": bytes_per_iter,
        "slope_max_bytes": int(args.slope_max_bytes),
        "samples": samples,
    }
    status = "fail" if bytes_per_iter > float(args.slope_max_bytes) else "ok"
    warnings = (
        [f"memory slope {bytes_per_iter:.0f} bytes/iter > {args.slope_max_bytes} (suspected leak)"]
        if status == "fail"
        else []
    )
    return Certificate(
        tool="memory_leak_sentinel",
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

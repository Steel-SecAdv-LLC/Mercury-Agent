#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the served detectors' latency and derive ``EngineConfig.detector_timeout_s``.

The detector budget is a hang detector: it must sit far enough above every
healthy detector's latency at the largest batch Mercury serves that a slow
host never cancels real work, and close enough that a hung detector cannot
hold a request open indefinitely.

Method: build the engine the API and MCP server build
(``OmniMercuryEngine(mode="fusion", require_explicit_fit=False)``), fit every
detector on ``--fit-rows`` rows, then time ``detector.detect`` per detector at
each batch size (``--repeats`` runs; the maximum is kept, a conservative
p100 of a small sample). Per detector, a power law ``t = a * n**b`` is fitted
in log-log space so the cost growth is visible (``b`` near 1 is linear).

Rule. Every served detector is close to linear in the batch (fitted ``b``
0.95-1.09 once the directive GSIS term and the spatial autocorrelation graph
stopped being quadratic), so the budget scales with the batch:

    budget(n) = floor_s + per_row_s * n
    per_row_s = safety * sum over detectors of (latency at --ceiling-rows / ceiling rows)

The sum is the sequential worst case, so it bounds the detector pool's wall
time whatever the worker count; ``--safety`` (default 5) covers a host that
much slower than the measuring one; ``--floor-s`` covers fixed overheads.

Usage::

    python benchmarks/detector_latency.py               # writes benchmarks/results/detector_latency.json
    python benchmarks/detector_latency.py --sizes 256 4096 --repeats 2
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "detector_latency.json"


def measure(
    sizes: list[int], features: int, fit_rows: int, repeats: int, seed: int
) -> dict[str, Any]:
    """Time every served detector's ``detect`` at each batch size."""
    os.environ.setdefault("MERCURY_GATE_AUDIT_DISABLED", "1")
    logging.disable(logging.WARNING)
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(fit_rows, features))
    for detector in engine.detectors.values():
        detector.fit(train)
    timings: dict[str, dict[str, float]] = {name: {} for name in engine.detectors}
    for n in sizes:
        batch = rng.normal(size=(n, features))
        for name, detector in engine.detectors.items():
            runs = []
            for _ in range(repeats):
                started = time.perf_counter()
                detector.detect(batch)
                runs.append(time.perf_counter() - started)
            timings[name][str(n)] = max(runs)
    return {
        "detectors": timings,
        "num_workers": int(engine.config.num_workers),
        "configured_timeout_s": engine.config.detector_timeout_s,
    }


def power_law(points: dict[str, float]) -> dict[str, float]:
    """Least-squares fit of ``log t = log a + b log n``."""
    n = np.log(np.array([float(k) for k in points]))
    t = np.log(np.maximum(np.array(list(points.values())), 1e-9))
    b, log_a = np.polyfit(n, t, 1)
    return {"a": float(math.exp(log_a)), "b": float(b)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 4096, 16384, 32768])
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--fit-rows", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ceiling-rows", type=int, default=32768)
    parser.add_argument("--safety", type=float, default=5.0)
    parser.add_argument("--floor-s", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    measured = measure(args.sizes, args.features, args.fit_rows, args.repeats, args.seed)
    fits = {name: power_law(points) for name, points in measured["detectors"].items()}
    ceiling = str(args.ceiling_rows)
    at_ceiling = {
        name: points.get(ceiling, fits[name]["a"] * args.ceiling_rows ** fits[name]["b"])
        for name, points in measured["detectors"].items()
    }
    slowest = max(at_ceiling, key=lambda name: at_ceiling[name])
    per_row = args.safety * sum(at_ceiling.values()) / args.ceiling_rows
    # Rounded up at two significant figures so the committed default is readable.
    exponent = math.floor(math.log10(per_row))
    per_row = math.ceil(per_row / 10 ** (exponent - 1)) * 10 ** (exponent - 1)
    budget_at_ceiling = args.floor_s + per_row * args.ceiling_rows
    document = {
        "schema": "mercury.benchmarks.detector_latency/v1",
        "design": {
            "sizes": args.sizes,
            "features": args.features,
            "fit_rows": args.fit_rows,
            "repeats": args.repeats,
            "ceiling_rows": args.ceiling_rows,
            "safety": args.safety,
            "floor_s": args.floor_s,
            "seed": args.seed,
        },
        "latency_s": measured["detectors"],
        "power_law": fits,
        "at_ceiling_s": at_ceiling,
        "slowest_detector": slowest,
        "recommended_timeout_floor_s": float(args.floor_s),
        "recommended_timeout_per_row_s": float(per_row),
        "budget_at_ceiling_s": round(budget_at_ceiling, 2),
        "rule": "floor_s + safety * sum(detector latency at ceiling) / ceiling_rows per row",
        "engine": {
            "num_workers": measured["num_workers"],
            "configured_timeout_s": measured["configured_timeout_s"],
        },
        "host": {
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in sorted(at_ceiling, key=lambda k: -at_ceiling[k]):
        print(f"{name:<12} b={fits[name]['b']:.2f}  {at_ceiling[name]:8.3f} s at {ceiling} rows")
    print(
        f"recommended budget = {args.floor_s:g} s + {per_row:.3g} s/row "
        f"({budget_at_ceiling:.1f} s at {ceiling} rows; configured fixed "
        f"{measured['configured_timeout_s']} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

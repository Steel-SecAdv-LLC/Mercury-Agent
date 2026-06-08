#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Deterministic hardware micro-benchmark for the Lyapunov validator pipeline.

This harness is intentionally small and self-contained.  It exists so
that performance numbers cited anywhere in the documentation can be
reproduced on demand — on a CI runner, on a developer workstation, or
on dedicated benchmarking hardware — with the *same* command and a
machine-readable JSON output that downstream tooling can diff against
historical baselines.

The harness measures:

* Wall-clock latency of a single ``validate_quadratic`` call on the
  canonical configuration (mean / p50 / p95 / p99 / max over ``iters``
  repetitions, with ``warmup`` discarded leading samples).
* Throughput in operations per second, computed as
  ``samples / total_s`` (the count of post-warmup iterations divided
  by their summed wall-clock time).  Because ``mean_s`` is the
  arithmetic mean of the same samples, this quantity is identical to
  ``1 / mean_s`` up to floating-point rounding; we emit it explicitly
  so the JSON report carries an unambiguous, directly-assertable
  invariant (``timing.ops_per_sec == timing.samples / timing.total_s``)
  that downstream tooling can pin without re-deriving the mean.
* Environment fingerprint (Python version, NumPy version, platform,
  CPU count, optional CPU affinity, Linux CPU scaling governor).
  This is what makes a measurement *scientifically* comparable: a
  number without its fingerprint is worthless.

The harness uses only the project's declared core dependencies:
``numpy`` (the Lyapunov validator's only numerical dep) and
``pyyaml`` (the canonical config format).  Both are pinned in
``pyproject.toml [project.dependencies]``, so installing
``mercury-agent`` is sufficient to run the harness — no extras are
required.  No new third-party dependencies are introduced.

Usage::

    python scripts/run_hardware_benchmark.py         --config configs/lyapunov_canonical.yaml         --iters 2000 --warmup 200         --out artifacts/hwbench.json

Exit codes::

    0  benchmark completed and all assertions passed
    2  configuration file missing or invalid
    3  Lyapunov validation failed (the workload itself is broken;
       performance numbers would be meaningless)
    4  measured throughput regressed below ``--min-ops-per-sec`` (only
       checked when that flag is provided)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the repository root is importable when invoked as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lyapunov_validator import (
    validate_lyapunov_from_config,
    validate_quadratic,
)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # PyYAML is declared in pyproject.toml core deps
    except ImportError as exc:
        raise ValueError(
            "PyYAML is required to load the benchmark config but is not "
            f"installed in this environment ({exc}). Install with "
            "`pip install pyyaml>=6.0` or `pip install mercury-agent`."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} did not parse as a mapping")
    return data


def _read_cpu_governor() -> str | None:
    """Return the scaling governor of CPU 0 (Linux), or None elsewhere."""
    candidate = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return candidate.read_text().strip() or None
    except (OSError, FileNotFoundError):  # pragma: no cover - non-Linux
        return None


def _environment_fingerprint() -> dict[str, Any]:
    """Capture enough of the environment that two runs can be comparable.

    We deliberately do *not* probe CPU model strings via ``/proc/cpuinfo``
    -- those differ across runners and would create spurious diffs.  We
    capture the inputs the validator's runtime actually depends on PLUS
    the OS-visible knobs (scheduling affinity, CPU governor) that
    materially affect the measured number, so a reviewer can decide at a
    glance whether two reports are comparable.
    """

    affinity: list[int] | None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        affinity = None

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "cpu_governor": _read_cpu_governor(),
    }


def _percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolation percentile.

    Implemented locally so the result is identical to ``numpy.percentile``
    with ``method='linear'`` while keeping the dependency surface small
    and the algorithm auditable.
    """

    if not samples:
        raise ValueError("cannot take percentile of empty sample")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    return float(np.percentile(np.asarray(samples), pct, method="linear"))


def benchmark(
    A: np.ndarray,
    P: np.ndarray,
    claimed_lambda: float,
    *,
    iters: int,
    warmup: int,
) -> dict[str, Any]:
    """Time ``iters`` calls of :func:`validate_quadratic`, discarding warmup.

    ``ops_per_sec`` is reported as ``samples / total_s`` -- the count of
    post-warmup iterations divided by their summed wall-clock time.
    Because ``mean_s`` is computed from the same samples via
    ``statistics.fmean``, the identity ``ops_per_sec == 1 / mean_s``
    holds up to floating-point round-off.  Reporting both ``total_s``
    and ``ops_per_sec`` lets downstream tooling pin the exact invariant
    ``ops_per_sec == samples / total_s`` rather than re-deriving the
    arithmetic mean (which would otherwise be the only documented
    derivation, and is brittle if the harness ever switches to a
    trimmed or median estimator).
    """

    if iters <= 0:
        raise ValueError("iters must be positive")
    if warmup < 0 or warmup >= iters:
        raise ValueError("warmup must be in [0, iters)")

    timings: list[float] = []
    for i in range(iters):
        t0 = time.perf_counter()
        ok, _ = validate_quadratic(A, P, claimed_lambda)
        elapsed = time.perf_counter() - t0
        if not ok:
            # The workload itself is broken; do not report perf numbers.
            raise RuntimeError(f"validate_quadratic returned False on iteration {i}")
        if i >= warmup:
            timings.append(elapsed)

    mean = statistics.fmean(timings)
    total_s = sum(timings)
    ops_per_sec = (len(timings) / total_s) if total_s > 0 else float("inf")
    return {
        "iters": iters,
        "warmup": warmup,
        "samples": len(timings),
        "mean_s": mean,
        "stdev_s": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "p50_s": _percentile(timings, 50),
        "p95_s": _percentile(timings, 95),
        "p99_s": _percentile(timings, 99),
        "max_s": max(timings),
        "total_s": total_s,
        "ops_per_sec": ops_per_sec,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "lyapunov_canonical.yaml"),
        help="Path to Lyapunov YAML config (default: canonical)",
    )
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--out",
        default="artifacts/hwbench.json",
        help="JSON output path (parent created if missing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy RNG seed for any stochastic helpers (kept for parity)",
    )
    parser.add_argument(
        "--min-ops-per-sec",
        type=float,
        default=None,
        help=(
            "Optional throughput floor; if provided and measured ops/s "
            "falls below it, the script exits with code 4."
        ),
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2

    np.random.seed(args.seed)

    # Gate 1: validate the Lyapunov claim before timing anything.  This
    # mirrors scripts/run_ablation.py — a benchmark of a broken
    # workload would be meaningless.
    valid, details = validate_lyapunov_from_config(cfg_path)
    if not valid:
        print("ERROR: Lyapunov validation failed", file=sys.stderr)
        print(json.dumps(details, indent=2), file=sys.stderr)
        return 3

    try:
        cfg = _load_config(cfg_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot load config: {exc}", file=sys.stderr)
        return 2

    # The benchmark times the quadratic-Lyapunov certificate path
    # (Cholesky + symmetric generalized eigen-decomposition on
    # A^T P + P A vs P), so it requires the explicit matrices.
    # ``validate_lyapunov_from_config`` also accepts (a) a nested
    # ``lyapunov:`` block and (b) ``lyapunov_samples`` mode -- both
    # would pass the gate above but lack the top-level ``A``/``P``
    # this loop needs, KeyError-ing the runner.  Resolve the nested
    # form transparently; reject samples-mode with a documented
    # config error (rc=2) instead of crashing.
    lyapunov_block = cfg.get("lyapunov") if isinstance(cfg.get("lyapunov"), Mapping) else None
    matrix_src = lyapunov_block if (lyapunov_block and "A" in lyapunov_block) else cfg
    if "A" not in matrix_src or "P" not in matrix_src or "lambda" not in matrix_src:
        print(
            "ERROR: hardware benchmark requires a quadratic Lyapunov certificate "
            "(A, P, lambda) -- samples-mode certificates are accepted by the "
            "validator but cannot be timed by this harness. Provide a config "
            "with explicit A/P matrices (either at the top level or under a "
            "`lyapunov:` block).",
            file=sys.stderr,
        )
        return 2

    A = np.asarray(matrix_src["A"], dtype=np.float64)
    P = np.asarray(matrix_src["P"], dtype=np.float64)
    claimed_lambda = float(matrix_src["lambda"])

    timing = benchmark(A, P, claimed_lambda, iters=args.iters, warmup=args.warmup)

    report = {
        "config": str(cfg_path),
        "environment": _environment_fingerprint(),
        "validation": details,
        "timing": timing,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.min_ops_per_sec is not None:
        if timing["ops_per_sec"] < args.min_ops_per_sec:
            print(
                f"REGRESSION: ops_per_sec={timing['ops_per_sec']:.1f} "
                f"< floor={args.min_ops_per_sec:.1f}",
                file=sys.stderr,
            )
            return 4

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

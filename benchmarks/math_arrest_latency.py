# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Latency of the 21-probe math-arrest forward path, measured and recorded.

The ensemble's ``fit`` and ``detect`` are profiled at fixed row counts and the
result is written to ``benchmarks/results/math_arrest_latency.json``. The
numbers recorded before and after the 2026-11-19 vectorization of the three
pure-Python probe loops (the Lyapunov O(n^2) neighbour scan, the per-window
``np.corrcoef`` fractal loop and the EWMA recursion inside the exponential
probe's ``minimize_scalar`` objective) are, on the development container:

==========  =============  ==============  =============  ==============
rows         fit (looped)   fit (vector)    detect (looped) detect (vector)
==========  =============  ==============  =============  ==============
64            0.013 s        0.010 s         0.003 s        0.001 s
1 024         0.624 s        0.051 s         0.197 s        0.010 s
4 096         7.972 s        0.378 s         2.686 s        0.122 s
==========  =============  ==============  =============  ==============

The mathematics is unchanged: ``tests/detectors/test_math_arrest_vectorized.py``
holds each vectorized path to a verbatim transcription of the released scalar
loop (bit-identical for Lyapunov and EWMA), and every recorded-corpus pin in
``tests/detectors/test_recorded_math_arrest.py`` passes unmoved.

Run from the repository root::

    python benchmarks/math_arrest_latency.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

RESULTS_PATH = _HERE / "results" / "math_arrest_latency.json"
SIZES = (64, 1024, 4096)
REPEATS = 3
SEED = 20261119


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def measure_latency() -> dict[str, dict[str, float]]:
    """Time ``fit`` and ``detect`` of the full 21-probe ensemble per size.

    Each cell is the best of ``REPEATS`` runs (the usual noise floor for
    wall-clock micro-benchmarks); the signal is the same seeded noisy sine at
    every size so runs are comparable across commits.
    """
    rng = np.random.default_rng(SEED)
    per_size: dict[str, dict[str, float]] = {}
    for n in SIZES:
        series = np.sin(np.linspace(0.0, 20.0, n)) + 0.1 * rng.standard_normal(n)
        fit_times: list[float] = []
        detect_times: list[float] = []
        for _ in range(REPEATS):
            detector = AnomalyMathArrest()
            start = time.perf_counter()
            detector.fit(series)
            fit_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            detector.detect(series)
            detect_times.append(time.perf_counter() - start)
        per_size[str(n)] = {
            "fit_s": round(min(fit_times), 4),
            "detect_s": round(min(detect_times), 4),
        }
    return per_size


def main() -> int:
    latency = measure_latency()
    metrics: dict[str, object] = {
        "schema": "mercury.benchmarks.math_arrest_latency/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config": {
            "sizes": list(SIZES),
            "repeats": REPEATS,
            "seed": SEED,
            "probes": 21,
        },
        "latency": latency,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    for size, cell in latency.items():
        print(f"rows={size:>5}: fit {cell['fit_s']:.4f}s detect {cell['detect_s']:.4f}s")
    print(f"wrote {RESULTS_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Operational sweep: run EVERY registered detector and report status + latency.

Answers "were all detectors run, operational, and efficient?" directly: it
auto-discovers Mercury's full detector registry, runs every detector's feature
extraction on real data through the registry's own execution path, and prints a
per-detector operational verdict (ok / failed + error) plus latency, with a
roll-up (count, success rate, total/median/slowest).

Run: ``python research/model_integration_obsv/detector_operational_sweep.py``
Exit 0 when every discovered detector executed without error.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

from omni_mercury_engine.core.detector_registry import (
    DETECTOR_MANIFEST,
    get_global_registry,
)


def _sample_data() -> np.ndarray:
    """A multivariate window with a clear anomalous block (rows 190-197)."""
    rng = np.random.default_rng(17)
    x = rng.normal(0.0, 1.0, size=(200, 8)).astype(np.float64)
    x[190:198] += 9.0
    return x


def main() -> int:
    print("=" * 78)
    print("Mercury detector registry -- full operational sweep")
    print("=" * 78)

    registry = get_global_registry()  # auto_discover=True
    stats = registry.get_statistics()
    all_names = registry.list_all()
    print(f"manifest entries declared : {len(DETECTOR_MANIFEST)}")
    print(f"detectors discovered/loaded: {len(all_names)}")
    by_cat = stats.get("by_category") or stats.get("categories")
    if by_cat:
        print(f"by category               : {by_cat}")

    data = _sample_data()
    results = registry.extract_all_features(data, parallel=True)

    ok, failed = [], []
    latencies: list[tuple[str, float]] = []
    for name in sorted(results):
        r = results[name]
        lat = float(getattr(r, "execution_time_ms", 0.0) or 0.0)
        if getattr(r, "success", False):
            ok.append(name)
            latencies.append((name, lat))
        else:
            failed.append((name, getattr(r, "error", "unknown")))

    print("-" * 78)
    print(f"RAN {len(results)} detectors: {len(ok)} operational, {len(failed)} failed")

    if latencies:
        lat_vals = sorted(v for _, v in latencies)
        total = sum(lat_vals)
        median = lat_vals[len(lat_vals) // 2]
        slowest = sorted(latencies, key=lambda kv: kv[1], reverse=True)[:5]
        print(
            f"latency: total={total:.1f}ms  median={median:.2f}ms  "
            f"min={lat_vals[0]:.2f}ms  max={lat_vals[-1]:.2f}ms"
        )
        print("slowest 5: " + ", ".join(f"{n}={v:.1f}ms" for n, v in slowest))

    if failed:
        print("\nFAILED detectors (name -> error):")
        for name, err in failed:
            print(f"  - {name}: {str(err)[:160]}")

    print("-" * 78)
    if failed:
        print("RESULT: not every detector executed cleanly (see failures above).")
        return 1
    print(f"RESULT: all {len(ok)} discovered detectors executed operationally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

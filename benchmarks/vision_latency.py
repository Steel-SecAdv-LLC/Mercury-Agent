#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-image latency of the visual anomaly analyser on MVTec AD textures.

For each texture category the analyser is fitted on 60 defect-free training
images and then times ``analyze()`` on 40 test images, one at a time, after
one warm-up call. Two profiles are measured: the default (two pyramid
scales) and ``AnalyzerConfig.fast()`` (one coarse scale).

On categories where fitting switches the lattice-symmetry group off (an
aperiodic texture), the same fitted model is timed a second time with the
lattice computed anyway. The difference is the saving from skipping it,
measured on identical inputs and an identical model.

Run it on an otherwise idle machine: the medians are wall-clock times and
any concurrent load inflates them. Absolute times depend on the host; the
within-run comparisons do not. The data cache is shared with
``mvtec_visual_benchmark.py`` (same pinned manifest, same preprocessing).

Usage::

    python benchmarks/vision_latency.py [--data-dir DIR] [--images 40]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import mvtec_visual_benchmark as mvtec

from omni_mercury_engine.vision import AnalyzerConfig, VisualAnomalyAnalyzer

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "vision_latency.json"
N_FIT = 60


def _time_ms(analyzer: VisualAnomalyAnalyzer, images: list[str]) -> dict[str, Any]:
    analyzer.analyze(images[0])  # warm-up: lazy caches, first-call allocation
    times = []
    for path in images:
        start = time.perf_counter()
        analyzer.analyze(path)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": round(float(np.median(times)), 1),
        "p95_ms": round(float(np.percentile(times, 95)), 1),
        "n": len(times),
    }


def measure_category(
    samples: list[dict[str, Any]], category: str, data_dir: Path, n_images: int
) -> dict[str, Any]:
    """Time both profiles on one category; add the lattice A/B where it is skipped."""
    train, test = mvtec.prepare_category(samples, category, 100, data_dir, 256, 42)
    fit_paths = [str(p) for p in train[:N_FIT]]
    images = [str(r["image"]) for r in test[:n_images]]
    result: dict[str, Any] = {}
    for name, config in (("default", AnalyzerConfig()), ("fast", AnalyzerConfig.fast())):
        # Prototype count changes fitting, not the cost of analyze().
        config.type_simulations = 0
        analyzer = VisualAnomalyAnalyzer(config).fit(fit_paths)
        result[name] = _time_ms(analyzer, images)
        result[name]["lattice_active"] = analyzer.lattice_active
        if name == "default" and not analyzer.lattice_active:
            # Measurement override: compute the switched-off lattice anyway.
            analyzer._skip_lattice = False
            result["default_lattice_computed"] = _time_ms(analyzer, images)
        print(category, name, result, flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the latency measurement and write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--categories", nargs="+", default=list(mvtec.TEXTURES))
    parser.add_argument("--images", type=int, default=40)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "mvtec_ad_256")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    samples = mvtec.load_manifest(args.data_dir)
    report = {
        "benchmark": "vision_latency",
        "image_size": 256,
        "n_fit_images": N_FIT,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "categories": {
            category: measure_category(samples, category, args.data_dir, args.images)
            for category in args.categories
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

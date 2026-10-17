#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure start-up and ``detect_with_fusion`` latency on this tree.

Three things are timed, each the way a user meets it:

* ``import omni_mercury_engine.engine`` and ``OmniMercuryEngine(auto_load_checkpoint=True)``,
  in fresh interpreters (the median of ``--repeats``), because lazy imports are
  only visible from a cold process;
* ``detect_with_fusion`` at each batch size on one warm engine, with the feature
  cache cleared before every run so a cache hit is never timed;
* the domain-model feature stage (``extract_features`` plus ``predict`` per
  model) at ``--model-rows`` rows, which is where the fusion path spends its time.

To compare with another commit, run this same file from that checkout's tree::

    git worktree add ../base <commit> && cp benchmarks/fusion_latency.py ../base/benchmarks/
    python ../base/benchmarks/fusion_latency.py --out /tmp/base.json

Usage::

    python benchmarks/fusion_latency.py    # writes benchmarks/results/fusion_latency.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "fusion_latency.json"

_STARTUP_PROBE = """
import time
started = time.perf_counter()
import omni_mercury_engine.engine as engine
imported = time.perf_counter()
engine.OmniMercuryEngine(auto_load_checkpoint=True)
print(imported - started, time.perf_counter() - imported)
"""


def startup(repeats: int) -> dict[str, float]:
    """Median import and construction time over fresh interpreters."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONWARNINGS": "ignore"}
    runs = []
    for _ in range(repeats):
        out = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and a literal probe
            [sys.executable, "-c", _STARTUP_PROBE],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        ).stdout.split()[-2:]
        runs.append([float(v) for v in out])
    return {
        "import_s": statistics.median(r[0] for r in runs),
        "construct_s": statistics.median(r[1] for r in runs),
    }


def detection(sizes: list[int], features: int, repeats: int, seed: int) -> dict[str, Any]:
    """Median ``detect_with_fusion`` latency per batch size, warm engine, cold cache."""
    logging.disable(logging.WARNING)
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(seed)
    engine = OmniMercuryEngine(auto_load_checkpoint=True)
    engine.detect_with_fusion(rng.normal(size=(32, features)).astype(np.float32))
    latency: dict[str, float] = {}
    for n in sizes:
        runs = []
        for _ in range(repeats):
            batch = rng.normal(size=(n, features)).astype(np.float32)
            engine.feature_cache.clear()
            started = time.perf_counter()
            engine.detect_with_fusion(batch)
            runs.append(time.perf_counter() - started)
        latency[str(n)] = statistics.median(runs)

    batch = rng.normal(size=(sizes[-1], features)).astype(np.float32)
    per_model: dict[str, float] = {}
    models: dict[str, Any] = dict(engine.models)
    for name, model in models.items():
        reset = getattr(model, "reset_state", None)
        try:
            if callable(reset):
                reset()
            started = time.perf_counter()
            model.extract_features(batch)
            if callable(reset):
                reset()
            model.predict(batch)
            per_model[name] = time.perf_counter() - started
        except Exception:  # models that need a mapping input decline an array
            continue
    return {"detect_with_fusion_s": latency, "model_stage_s": per_model}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 1024, 4096])
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    document = {
        "schema": "mercury.benchmarks.fusion_latency/v1",
        "design": {
            "sizes": args.sizes,
            "features": args.features,
            "repeats": args.repeats,
            "seed": args.seed,
            "model_stage_rows": args.sizes[-1],
        },
        "startup": startup(args.repeats),
        **detection(args.sizes, args.features, args.repeats, args.seed),
        "host": {
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    document["elapsed_s"] = round(time.perf_counter() - started, 1)
    document["model_stage_total_s"] = sum(document["model_stage_s"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"import {document['startup']['import_s']:.2f} s   construct {document['startup']['construct_s']:.2f} s"
    )
    for n, seconds in document["detect_with_fusion_s"].items():
        print(f"detect_with_fusion {n:>6} rows x {args.features}: {seconds * 1000:8.0f} ms")
    print(
        f"domain-model stage at {args.sizes[-1]} rows: {document['model_stage_total_s'] * 1000:.0f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Held-out synthetic benchmark of the visual anomaly analyser, with exact ground truth.

Real benchmarks (``mvtec_visual_benchmark.py``) measure detection on real
defects but cannot score *classification*: MVTec defect names (``hole``,
``glue``) are not the analyser's physical types. This script measures what
real data cannot, on two procedurally generated texture domains with exact
masks and types:

* ``woven``: a periodic two-axis weave with fibre noise and gentle shading;
* ``leather``: an aperiodic isotropic grain with pores.

For each domain the analyser is fitted on 40 normal images and evaluated on
images and seeds it never saw:

* the false-alarm rate on 60 held-out normals at ``alpha`` (a conformal
  guarantee: it should not exceed ``alpha`` beyond sampling noise), and the
  forensic false-positive rate on the same images;
* **same-definition classification**: the analyser's own
  :data:`~omni_mercury_engine.vision.simulation.SIMULATED_TYPES` generator,
  re-seeded on unseen images (does the learned prototype generalise?);
* **cross-generator classification**: eight independently written defect
  injectors with different shapes and parameters (does the *type*, not the
  generator, get recognised?), plus per-injector detection, image AUROC and
  pixel AUROC.

Everything is seeded; the results JSON is
``benchmarks/results/vision_synthetic_heldout.json``.

Usage::

    python benchmarks/vision_synthetic_heldout.py [--alpha 0.1]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.vision import AnalyzerConfig, VisualAnomalyAnalyzer
from omni_mercury_engine.vision.simulation import SIMULATED_TYPES, simulate_anomaly

Array = np.ndarray[Any, Any]
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "vision_synthetic_heldout.json"
SIZE = 192

#: Independent injector -> the physical type it should be recognised as.
INJECTOR_TYPES: dict[str, str] = {
    "occlusion": "occlusion_or_missing",
    "blur": "blur_or_defocus",
    "noise": "noise_or_sensor_corruption",
    "impulse": "noise_or_sensor_corruption",
    "stain": "color_shift",
    "foreign": "foreign_texture_or_object",
    "scratch": "structural_change",
    "dark": "exposure_anomaly",
}


def woven(seed: int, size: int = SIZE) -> Array:
    """Return a woven-fabric-like RGB surface: periodic weave, fibre noise, gentle shading."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size)).astype(float)
    phase = rng.uniform(0, 2 * np.pi, 2)
    weave = 0.5 + 0.18 * np.sin(2 * np.pi * xx / 9 + phase[0]) * np.sin(
        2 * np.pi * yy / 9 + phase[1]
    )
    fibre = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.2) * 0.08
    shade = 1.0 + 0.05 * np.sin(2 * np.pi * (xx + yy) / (size * 2) + rng.uniform(0, 6))
    lum = np.clip((weave + fibre) * shade, 0, 1)
    tint = np.array([0.55, 0.45, 0.35]) + rng.normal(0, 0.01, 3)
    rgb = np.clip(lum[..., None] * tint[None, None, :] * 1.6, 0, 1)
    return np.asarray(rgb + rng.normal(0, 0.004, rgb.shape))


def leather(seed: int, size: int = SIZE) -> Array:
    """Return an aperiodic, isotropic stochastic grain with pores (leather-like) RGB surface."""
    rng = np.random.default_rng(seed)
    grain = ndimage.gaussian_filter(rng.normal(size=(size, size)), 2.0)
    pores = ndimage.gaussian_filter((rng.random((size, size)) > 0.97).astype(float), 1.0) * -1.5
    lum = 0.45 + 0.09 * grain / grain.std() + 0.05 * pores
    rgb = np.clip(lum[..., None] * np.array([1.15, 0.95, 0.8])[None, None, :], 0, 1)
    return np.asarray(rgb + rng.normal(0, 0.004, rgb.shape))


DOMAINS: dict[str, Callable[[int], Array]] = {"woven": woven, "leather": leather}


def _box_mask(rng: np.random.Generator, size: int) -> Array:
    height, width = int(rng.integers(18, 34)), int(rng.integers(18, 34))
    y, x = int(rng.integers(10, size - height - 10)), int(rng.integers(10, size - width - 10))
    mask = np.zeros((size, size), dtype=bool)
    mask[y : y + height, x : x + width] = True
    return mask


def inject(image: Array, kind: str, seed: int) -> tuple[Array, Array]:
    """Return ``(defective image, ground-truth mask)`` from an independent injector."""
    rng = np.random.default_rng(1000 + seed)
    size = image.shape[0]
    out = image.copy()
    mask = _box_mask(rng, size)
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if kind == "occlusion":
        out[mask] = image[mask].mean(axis=0) * 0.6
    elif kind == "blur":
        out[mask] = ndimage.gaussian_filter(image, (3, 3, 0))[mask]
    elif kind == "noise":
        out[mask] = np.clip(image[mask] + rng.normal(0, 0.12, image[mask].shape), 0, 1)
    elif kind == "stain":
        out[mask] = np.clip(image[mask] * np.array([0.7, 1.05, 1.35]), 0, 1)
    elif kind == "foreign":
        # Spatially structured blotches with their own tint (glue- or fibre-like matter).
        blot = ndimage.gaussian_filter(rng.normal(size=(y1 - y0, x1 - x0)), 1.5) > 0
        out[y0:y1, x0:x1] = (blot * 0.5 + 0.25)[..., None] * np.array([0.9, 0.8, 0.7])
    elif kind == "impulse":
        # Per-pixel i.i.d. binary values: physically salt-and-pepper sensor noise.
        speck = (rng.random((y1 - y0, x1 - x0)) > 0.5) * 0.5 + 0.25
        out[y0:y1, x0:x1] = speck[..., None] * np.array([0.9, 0.8, 0.7])
    elif kind == "scratch":
        mask = np.zeros((size, size), dtype=bool)
        t = np.linspace(0, 1, 400)
        y_a, x_a = rng.integers(20, size - 20, 2)
        angle, length = rng.uniform(0, np.pi), rng.integers(40, 70)
        yy = np.clip((y_a + t * length * np.sin(angle)).astype(int), 0, size - 1)
        xx = np.clip((x_a + t * length * np.cos(angle)).astype(int), 0, size - 1)
        mask[yy, xx] = True
        mask = ndimage.binary_dilation(mask, iterations=1)
        out[mask] = 0.95
    elif kind == "dark":
        out[mask] = image[mask] * 0.45
    else:
        raise ValueError(f"unknown injector {kind!r}")
    return np.clip(out, 0, 1), mask


def _primary_type(report: Any) -> str:
    return str(report.regions[0].anomaly_type) if report.regions else "-"


def evaluate_domain(name: str, alpha: float) -> dict[str, Any]:
    """Fit on one domain's normals and run every held-out measurement."""
    generate = DOMAINS[name]
    started = time.perf_counter()
    analyzer = VisualAnomalyAnalyzer(AnalyzerConfig(alpha=alpha)).fit(
        [generate(s) for s in range(1000, 1040)]
    )
    fit_seconds = time.perf_counter() - started

    held = [analyzer.analyze(generate(s)) for s in range(3000, 3060)]
    normal_scores = [r.image_score for r in held]

    rng = np.random.default_rng(424242)
    same_ok = same_total = 0
    same_confusions: Counter[str] = Counter()
    for anomaly_type in SIMULATED_TYPES:
        for i in range(20):
            image, _ = simulate_anomaly(generate(8000 + i), anomaly_type, rng)
            predicted = _primary_type(analyzer.analyze(image))
            same_ok += predicted == anomaly_type
            same_total += 1
            if predicted != anomaly_type:
                same_confusions[f"{anomaly_type} -> {predicted}"] += 1

    injectors: dict[str, Any] = {}
    cross_ok = cross_total = 0
    for kind, expected in INJECTOR_TYPES.items():
        reports, masks = [], []
        for i in range(30):
            image, mask = inject(generate(5000 + i), kind, 7000 + i)
            reports.append(analyzer.analyze(image))
            masks.append(mask)
        types = [_primary_type(r) for r in reports]
        correct = sum(t == expected for t in types)
        cross_ok += correct
        cross_total += len(types)
        injectors[kind] = {
            "expected_type": expected,
            "detection_rate": round(float(np.mean([bool(r.is_anomalous) for r in reports])), 3),
            "image_auroc": round(
                float(
                    roc_auc_score(
                        [0] * len(normal_scores) + [1] * len(reports),
                        normal_scores + [r.image_score for r in reports],
                    )
                ),
                4,
            ),
            "pixel_auroc": round(
                float(
                    np.mean(
                        [
                            roc_auc_score(m.ravel().astype(int), r.anomaly_map.ravel())
                            for r, m in zip(reports, masks, strict=True)
                        ]
                    )
                ),
                4,
            ),
            "type_accuracy": round(correct / len(types), 3),
            "predicted_types": dict(Counter(types).most_common(3)),
        }

    return {
        "fit_seconds": round(fit_seconds, 1),
        "n_fit_normals": 40,
        "n_heldout_normals": len(held),
        "false_alarm_rate_at_alpha": round(float(np.mean([bool(r.is_anomalous) for r in held])), 4),
        "forensic_false_positive_rate": round(
            float(np.mean([bool(r.manipulation_suspected) for r in held])), 4
        ),
        "median_latency_ms": round(float(np.median([r.timing_ms["total"] for r in held])), 1),
        "same_definition_classification": {
            "correct": same_ok,
            "total": same_total,
            "accuracy": round(same_ok / same_total, 4),
            "top_confusions": dict(same_confusions.most_common(3)),
        },
        "cross_generator_classification": {
            "correct": cross_ok,
            "total": cross_total,
            "accuracy": round(cross_ok / cross_total, 4),
        },
        "injectors": injectors,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    results = {}
    for name in args.domains:
        print(f"[{name}] fitting and evaluating ...", flush=True)
        results[name] = evaluate_domain(name, args.alpha)
        print(f"[{name}] {json.dumps(results[name])}", flush=True)
    payload = {
        "benchmark": "procedural textures with exact masks and types (held-out images and seeds)",
        "protocol": {"alpha": args.alpha, "image_size": SIZE, "analyzer": "AnalyzerConfig defaults"},
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "domains": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

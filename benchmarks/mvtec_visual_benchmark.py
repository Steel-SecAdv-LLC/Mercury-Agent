#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Real-image benchmark of the interpretable visual anomaly analyser on MVTec AD textures.

MVTec AD (Bergmann et al., CVPR 2019; CC BY-NC-SA 4.0) is the standard
industrial anomaly-detection benchmark: defect-free training images and test
images with expert pixel masks. This script downloads the five texture
categories from the Hugging Face mirror ``Voxel51/mvtec-ad`` at run time
(nothing is redistributed), downscales them to 256 px, fits
:class:`~omni_mercury_engine.vision.VisualAnomalyAnalyzer` on a seeded subset
of each category's defect-free training images, and evaluates every test image
under the standard protocol:

* image AUROC (anomaly score vs. good / defective),
* pixel AUROC pooled over all test pixels against the ground-truth masks,
* the conformal decision at ``alpha``: false-alarm rate on the good test
  images (should not exceed ``alpha``) and detection rate on defects,
* per-defect-type detection, and
* median analysis latency.

The manifest is pinned by SHA-256. Results are written to
``benchmarks/results/mvtec_visual_benchmark.json``.

Usage::

    python benchmarks/mvtec_visual_benchmark.py [--categories carpet leather] \\
        [--train-per-category 100] [--data-dir ./data/mvtec_ad_256]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from omni_mercury_engine.ml.mercury_ml import roc_auc_score
from omni_mercury_engine.vision import AnalyzerConfig, VisualAnomalyAnalyzer

BASE_URL = "https://huggingface.co/datasets/Voxel51/mvtec-ad/resolve/main/"
MANIFEST_SHA256 = "dbbbb94cee2ddec28c1eef318733d07df4d59b9cc066e62e6aeef386c1db281d"
TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
RESULTS_PATH = REPO_ROOT / "benchmarks" / "results" / "mvtec_visual_benchmark.json"
RECALIBRATION_IMAGES = 10
RECALIBRATION_ALPHA = 0.1


def _fetch(url: str, target: Path, retries: int = 4) -> None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)


def load_manifest(data_dir: Path) -> list[dict[str, Any]]:
    """Download (once) and digest-verify the sample manifest."""
    path = data_dir / "samples.json"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        _fetch(BASE_URL + "samples.json", path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError(f"manifest digest mismatch: {digest} != pinned {MANIFEST_SHA256}")
    return list(json.loads(path.read_text())["samples"])


def _cache_image(sample_path: str, data_dir: Path, size: int, mask: bool) -> Path:
    import cv2

    target = data_dir / f"{size}" / sample_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.with_suffix(".raw.png")
    _fetch(BASE_URL + sample_path, raw)
    image = cv2.imread(str(raw), cv2.IMREAD_UNCHANGED)
    raw.unlink(missing_ok=True)
    if image is None:
        raise ValueError(f"unreadable image: {sample_path}")
    interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    resized = cv2.resize(image, (size, size), interpolation=interpolation)
    cv2.imwrite(str(target), resized)
    return target


def prepare_category(
    samples: list[dict[str, Any]], category: str, n_train: int, data_dir: Path, size: int, seed: int
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Download and cache one category; return training paths and test records."""
    rows = [s for s in samples if s["category"]["label"] == category]
    train = [s for s in rows if s["split"] == "train"]
    rng = np.random.default_rng(seed)
    train = [
        train[i]
        for i in sorted(rng.choice(len(train), size=min(n_train, len(train)), replace=False))
    ]
    test = [s for s in rows if s["split"] == "test"]

    jobs: list[tuple[str, bool]] = [(s["filepath"], False) for s in train + test]
    jobs += [(s["defect_mask"]["mask_path"], True) for s in test if s.get("defect_mask")]
    with ThreadPoolExecutor(max_workers=6) as pool:
        paths = list(pool.map(lambda job: _cache_image(job[0], data_dir, size, job[1]), jobs))
    lookup = dict(zip([j[0] for j in jobs], paths, strict=True))
    test_records = [
        {
            "image": lookup[s["filepath"]],
            "mask": lookup[s["defect_mask"]["mask_path"]] if s.get("defect_mask") else None,
            "defect": s["defect"]["label"],
        }
        for s in test
    ]
    return [lookup[s["filepath"]] for s in train], test_records


def evaluate_category(
    train_paths: list[Path], test_records: list[dict[str, Any]], alpha: float
) -> dict[str, Any]:
    """Fit on the defect-free subset and evaluate every test image."""
    import cv2

    t0 = time.perf_counter()
    analyzer = VisualAnomalyAnalyzer(AnalyzerConfig(alpha=alpha)).fit([str(p) for p in train_paths])
    fit_seconds = time.perf_counter() - t0
    n_train, n_calibration = analyzer._n_train, analyzer._n_calibration

    labels, scores, latencies = [], [], []
    pixel_truth, pixel_score = [], []
    decisions: dict[str, list[bool]] = defaultdict(list)
    types: dict[str, Counter[str]] = defaultdict(Counter)
    withheld = 0
    for record in test_records:
        report = analyzer.analyze(str(record["image"]))
        latencies.append(report.timing_ms["total"])
        defective = record["defect"] != "good"
        labels.append(int(defective))
        scores.append(report.image_score)
        if report.is_anomalous is None:
            withheld += 1
        decisions[record["defect"]].append(bool(report.is_anomalous))
        if defective and report.regions:
            types[record["defect"]][report.regions[0].anomaly_type] += 1
        if record["mask"] is not None:
            mask = cv2.imread(str(record["mask"]), cv2.IMREAD_GRAYSCALE) > 0
        else:
            mask = np.zeros(report.anomaly_map.shape, dtype=bool)
        pixel_truth.append(mask.ravel())
        pixel_score.append(report.anomaly_map.ravel())

    truth = np.concatenate(pixel_truth).astype(int)
    pixel = np.concatenate(pixel_score)
    good = decisions.pop("good", [])
    anomalous = [d for v in decisions.values() for d in v]

    # Protocol 2 — in-session recalibration: MVTec's good *test* images are
    # measurably shifted from its good *training* images, which breaks the
    # exchangeability the conformal guarantee needs. Re-anchor the decision on
    # RECALIBRATION_IMAGES test-good images (excluded from scoring) at
    # RECALIBRATION_ALPHA — the smallest level 10 calibration images can certify.
    good_records = [r for r in test_records if r["defect"] == "good"]
    rng = np.random.default_rng(7)
    recal_index = set(
        rng.choice(
            len(good_records), size=min(RECALIBRATION_IMAGES, len(good_records) // 2), replace=False
        )
    )
    recal = [good_records[i] for i in sorted(recal_index)]
    remaining_good = [r for i, r in enumerate(good_records) if i not in recal_index]
    analyzer.config.alpha = RECALIBRATION_ALPHA
    analyzer.recalibrate([str(r["image"]) for r in recal])
    recal_good = [bool(analyzer.analyze(str(r["image"])).is_anomalous) for r in remaining_good]
    recal_defect = [
        bool(analyzer.analyze(str(r["image"])).is_anomalous)
        for r in test_records
        if r["defect"] != "good"
    ]
    return {
        "n_train": n_train,
        "n_calibration": n_calibration,
        "n_test_good": len(good),
        "n_test_defective": len(anomalous),
        "image_auroc": round(float(roc_auc_score(labels, scores)), 4),
        "pixel_auroc": round(float(roc_auc_score(truth, pixel)), 4),
        "false_alarm_rate_at_alpha": round(float(np.mean(good)), 4) if good else None,
        "detection_rate_at_alpha": round(float(np.mean(anomalous)), 4) if anomalous else None,
        "decisions_withheld": withheld,
        "per_defect_detection": {
            k: round(float(np.mean(v)), 3) for k, v in sorted(decisions.items())
        },
        "per_defect_primary_type": {k: dict(v.most_common(3)) for k, v in sorted(types.items())},
        "median_latency_ms": round(float(np.median(latencies)), 1),
        "fit_seconds": round(fit_seconds, 1),
        "recalibrated": {
            "alpha": RECALIBRATION_ALPHA,
            "n_recalibration_images": len(recal),
            "n_test_good_scored": len(remaining_good),
            "false_alarm_rate": round(float(np.mean(recal_good)), 4) if recal_good else None,
            "detection_rate": round(float(np.mean(recal_defect)), 4) if recal_defect else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--categories", nargs="+", default=list(TEXTURES))
    parser.add_argument("--train-per-category", type=int, default=100)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "mvtec_ad_256")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    samples = load_manifest(args.data_dir)
    results: dict[str, Any] = {}
    for category in args.categories:
        print(f"[{category}] preparing ...", flush=True)
        train, test = prepare_category(
            samples, category, args.train_per_category, args.data_dir, args.size, args.seed
        )
        print(
            f"[{category}] fitting on {len(train)} defect-free images, testing {len(test)} ...",
            flush=True,
        )
        results[category] = evaluate_category(train, test, args.alpha)
        print(f"[{category}] {json.dumps(results[category])}", flush=True)

    summary = {
        "mean_image_auroc": round(float(np.mean([r["image_auroc"] for r in results.values()])), 4),
        "mean_pixel_auroc": round(float(np.mean([r["pixel_auroc"] for r in results.values()])), 4),
        "mean_false_alarm_rate_at_alpha": round(
            float(np.mean([r["false_alarm_rate_at_alpha"] for r in results.values()])), 4
        ),
        "mean_detection_rate_at_alpha": round(
            float(np.mean([r["detection_rate_at_alpha"] for r in results.values()])), 4
        ),
        "recalibrated_mean_false_alarm_rate": round(
            float(np.mean([r["recalibrated"]["false_alarm_rate"] for r in results.values()])), 4
        ),
        "recalibrated_mean_detection_rate": round(
            float(np.mean([r["recalibrated"]["detection_rate"] for r in results.values()])), 4
        ),
    }
    payload = {
        "benchmark": "MVTec AD textures (Bergmann et al., CVPR 2019), Voxel51/mvtec-ad mirror",
        "license_note": "data CC BY-NC-SA 4.0; downloaded at run time, not redistributed",
        "manifest_sha256": MANIFEST_SHA256,
        "protocol": {
            "image_size": args.size,
            "train_per_category": args.train_per_category,
            "seed": args.seed,
            "alpha": args.alpha,
            "analyzer": "VisualAnomalyAnalyzer defaults (handcrafted features, no pretrained weights)",
        },
        "environment": {"python": platform.python_version(), "machine": platform.machine()},
        "categories": results,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent — Data Loader Benchmark Script.

Attempts to load every registered loader and reports:
  - data_source, record_count, anomaly_ratio, sha256, load_time
  - Any errors encountered

Output: Markdown table to stdout and JSON report to file.
Fails CI if any loader returns data_source: "synthetic" when
MERCURY_ALLOW_SYNTHETIC is not set.

Usage:
    python scripts/benchmark_loaders.py
    python scripts/benchmark_loaders.py --output benchmark_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any


def get_loader_registry() -> dict[str, Any]:
    """Get all registered dataset loaders."""
    from omni_mercury_engine.datasets.base import DatasetRegistry

    return dict(DatasetRegistry._loaders)


def benchmark_loader(name: str, loader_class: Any, timeout: int = 300) -> dict[str, Any]:
    """Benchmark a single loader."""
    from omni_mercury_engine.datasets.base import DatasetConfig

    result: dict[str, Any] = {
        "name": name,
        "loader_class": (
            loader_class.__name__ if hasattr(loader_class, "__name__") else str(loader_class)
        ),
        "data_source": "error",
        "record_count": 0,
        "feature_count": 0,
        "anomaly_count": 0,
        "anomaly_ratio": 0.0,
        "sha256": "",
        "load_time_seconds": 0.0,
        "error": None,
        "status": "FAIL",
    }

    start = time.monotonic()
    try:
        config = DatasetConfig(name=name)
        loader = loader_class(config)

        # Try download
        try:
            loader.download()
        except Exception as dl_err:
            result["error"] = f"download: {type(dl_err).__name__}: {dl_err}"
            result["load_time_seconds"] = time.monotonic() - start
            return result

        # Try load
        features, labels = loader.load()

        elapsed = time.monotonic() - start
        result["load_time_seconds"] = round(elapsed, 3)
        result["record_count"] = int(features.shape[0])
        result["feature_count"] = int(features.shape[1]) if features.ndim > 1 else 1
        result["anomaly_count"] = int(labels.sum())
        result["anomaly_ratio"] = round(float(labels.mean()), 4) if len(labels) > 0 else 0.0
        result["sha256"] = hashlib.sha256(features.tobytes()).hexdigest()[:16]

        # Determine data source
        if hasattr(loader, "is_real_data"):
            result["data_source"] = "live" if loader.is_real_data else "synthetic"
        elif hasattr(loader, "_is_real_data"):
            result["data_source"] = "live" if loader._is_real_data else "synthetic"
        else:
            result["data_source"] = "live"

        result["status"] = "OK"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        result["load_time_seconds"] = round(time.monotonic() - start, 3)

    return result


def main() -> int:
    """Run benchmark across all loaders."""
    parser = argparse.ArgumentParser(description="Benchmark Mercury-Agent data loaders")
    parser.add_argument("--output", "-o", help="JSON output file")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per loader")
    parser.add_argument("--filter", "-f", help="Filter loader names (substring match)")
    args = parser.parse_args()

    # Import loaders to trigger registration
    try:
        import omni_mercury_engine.datasets  # noqa: F401
    except ImportError:
        print("Error: omni_mercury_engine not importable. Run from project root.", file=sys.stderr)
        return 1

    registry = get_loader_registry()

    if args.filter:
        registry = {k: v for k, v in registry.items() if args.filter.lower() in k.lower()}

    print(f"\nBenchmarking {len(registry)} loaders...\n", file=sys.stderr)

    allow_synthetic = os.environ.get("MERCURY_ALLOW_SYNTHETIC", "0") == "1"
    results: list[dict[str, Any]] = []
    synthetic_count = 0
    error_count = 0

    # Deduplicate by loader class to avoid running the same loader multiple times
    seen_classes: set[str] = set()
    deduplicated: dict[str, Any] = {}
    for name, cls in sorted(registry.items()):
        cls_name = cls.__name__ if hasattr(cls, "__name__") else str(cls)
        if cls_name not in seen_classes:
            seen_classes.add(cls_name)
            deduplicated[name] = cls

    for name, loader_class in sorted(deduplicated.items()):
        print(f"  {name}...", end=" ", file=sys.stderr, flush=True)
        result = benchmark_loader(name, loader_class, args.timeout)
        results.append(result)

        status_str = result["status"]
        if result["data_source"] == "synthetic":
            synthetic_count += 1
            status_str = "SYNTHETIC"
        if result["error"]:
            error_count += 1

        print(
            f"[{status_str}] {result['record_count']} records, "
            f"{result['anomaly_ratio']:.1%} anomalies, "
            f"{result['load_time_seconds']:.1f}s",
            file=sys.stderr,
        )
        if result["error"]:
            print(f"    Error: {result['error']}", file=sys.stderr)

    # Print Markdown table
    print("\n## Data Loader Benchmark Report\n")
    print(f"Date: {datetime.now(UTC).isoformat()}")
    print(f"MERCURY_ALLOW_SYNTHETIC: {'1' if allow_synthetic else '0 (default)'}\n")
    print("| Loader | Status | Source | Records | Anomaly% | SHA256 | Time |")
    print("|--------|--------|--------|---------|----------|--------|------|")
    for r in results:
        print(
            f"| {r['name']} | {r['status']} | {r['data_source']} | "
            f"{r['record_count']} | {r['anomaly_ratio']:.1%} | "
            f"{r['sha256'][:8] or 'N/A'} | {r['load_time_seconds']:.1f}s |"
        )

    print(
        f"\n**Summary**: {len(results)} loaders, "
        f"{len(results) - error_count} OK, {error_count} errors, "
        f"{synthetic_count} synthetic\n"
    )

    # JSON output
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "allow_synthetic": allow_synthetic,
        "total_loaders": len(results),
        "ok_count": len(results) - error_count,
        "error_count": error_count,
        "synthetic_count": synthetic_count,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report: {args.output}", file=sys.stderr)

    # Fail CI if synthetic and not allowed
    if synthetic_count > 0 and not allow_synthetic:
        print(
            f"\nFAIL: {synthetic_count} loader(s) returned synthetic data "
            "without MERCURY_ALLOW_SYNTHETIC=1",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

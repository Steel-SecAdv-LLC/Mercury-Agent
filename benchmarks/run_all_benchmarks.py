#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified benchmark runner for all domain-specific anomaly detectors.

Orchestrates all domain benchmarks, produces a unified report, and
exits non-zero if any domain fails to meet its AUC threshold.

Usage:
    python benchmarks/run_all_benchmarks.py [--domains earthquake,tsunami,...]
    python benchmarks/run_all_benchmarks.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.domain_benchmark_base import run_domain_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).parent

# AUC gate thresholds per domain (initial targets — calibrate after first runs)
AUC_GATES: dict[str, float] = {
    "earthquake": 0.60,
    "tsunami": 0.60,
    "hurricane": 0.60,
    "tornado": 0.60,
    "flood": 0.60,
    # Measured 2026-07-09 (deterministic across repeat runs): mean AUC 0.6132
    # over the three SPC archive events (vivian_2010 0.750, texas_2016 0.502,
    # colorado_2017 0.588). Gate set below the measured mean with margin.
    "hail": 0.55,
    "wildfire": 0.60,
    "volcanic": 0.60,
    "landslide": 0.60,
    "sepsis": 0.60,
    "pandemic": 0.60,
    "financial": 0.60,
    "energy": 0.60,
    "marine": 0.60,
    "network_security": 0.75,  # Higher bar — already at 0.972
    "fema": 0.60,
    # Measured 2026-07-09 on live DONKI/SWPC + NeoWs/CNEOS data (5 events each):
    # space_weather mean AUC 0.9647 (std 0.0172, min 0.9429) -> gate 0.85;
    # meteor mean AUC 0.7705 (std 0.1418, min 0.5582) -> gate 0.60 on the mean.
    "space_weather": 0.85,
    "meteor": 0.60,
    # Measured 2026-07-09 on the 2-event live catalogs (regression floors set
    # just under measured, mirroring the guard pattern — not aspirations):
    # drought mean AUC 0.5728 (min 0.5424) -> gate 0.54; near-random today,
    # the gate protects the real labeled pipeline from regressing to chance.
    # heatwave mean AUC 0.6757 (min 0.6598) -> gate 0.63.
    "drought": 0.54,
    "heatwave": 0.63,
}


def _get_loader(domain: str) -> Any:
    """Import and instantiate the loader for a domain.

    Args:
        domain: Domain name.

    Returns:
        Loader instance.

    Raises:
        ImportError: If loader module not found.
    """
    module_name = f"omni_mercury_engine.loaders.{domain}_loader"
    import importlib

    mod = importlib.import_module(module_name)

    # Convention: class name is {Domain}Loader (CamelCase)
    class_name_map = {
        "earthquake": "EarthquakeLoader",
        "tsunami": "TsunamiLoader",
        "hurricane": "HurricaneLoader",
        "tornado": "TornadoLoader",
        "flood": "FloodLoader",
        "hail": "HailLoader",
        "wildfire": "WildfireLoader",
        "volcanic": "VolcanicLoader",
        "landslide": "LandslideLoader",
        "sepsis": "SepsisLoader",
        "pandemic": "PandemicLoader",
        "financial": "FinancialLoader",
        "energy": "EnergyLoader",
        "marine": "MarineLoader",
        "network_security": "NetworkSecurityLoader",
        "fema": "FEMALoader",
        "space_weather": "SpaceWeatherLoader",
        "meteor": "MeteorLoader",
        "drought": "DroughtLoader",
        "heatwave": "HeatwaveLoader",
    }

    class_name = class_name_map.get(domain)
    if class_name is None:
        raise ImportError(f"Unknown domain: {domain}")

    loader_cls = getattr(mod, class_name)
    return loader_cls()


def run_all(
    domains: list[str] | None = None,
    fail_on_gate: bool = True,
) -> dict[str, Any]:
    """Run benchmarks for all specified domains.

    Args:
        domains: List of domains to benchmark. None = all.
        fail_on_gate: Exit non-zero if any domain fails AUC gate.

    Returns:
        Unified results dict.
    """
    if domains is None:
        domains = list(AUC_GATES.keys())

    unified: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "domains": {},
        "summary": {},
    }

    all_aucs: list[float] = []
    gate_failures: list[str] = []
    start = time.monotonic()

    for domain in domains:
        logger.info("=" * 60)
        logger.info("Starting benchmark: %s", domain.upper())
        logger.info("=" * 60)

        try:
            loader = _get_loader(domain)
            result = run_domain_benchmark(domain, loader)
            unified["domains"][domain] = result

            if result["summary"].get("mean_auc") is not None:
                mean_auc = result["summary"]["mean_auc"]
                all_aucs.append(mean_auc)

                gate = AUC_GATES.get(domain, 0.60)
                if mean_auc < gate:
                    gate_failures.append(f"{domain}: AUC {mean_auc:.3f} < gate {gate:.3f}")
                    logger.warning(
                        "GATE FAILURE: %s AUC %.3f < %.3f",
                        domain,
                        mean_auc,
                        gate,
                    )
            elif fail_on_gate:
                # A gated domain that produced no AUC cannot be allowed to
                # pass the gate by not being measured.
                gate_failures.append(f"{domain}: benchmark produced no mean AUC to gate")
                logger.warning("GATE FAILURE: %s produced no mean AUC", domain)

        # A crashed or data-less GATED benchmark must not read as a pass:
        # its AUC gate never ran, which is exactly the silent-vacuity failure
        # mode the gates exist to prevent.
        except SystemExit:
            logger.warning("%s benchmark exited (no data available)", domain)
            unified["domains"][domain] = {"status": "no_data"}
            if fail_on_gate:
                gate_failures.append(f"{domain}: no data available; AUC gate never ran")
        except ImportError as exc:
            logger.error("Loader not found for %s: %s", domain, exc)
            unified["domains"][domain] = {"status": "loader_not_found", "error": str(exc)}
            if fail_on_gate:
                gate_failures.append(f"{domain}: loader not found; AUC gate never ran")
        except Exception as exc:
            logger.error("Benchmark failed for %s: %s", domain, exc)
            unified["domains"][domain] = {"status": "error", "error": str(exc)}
            if fail_on_gate:
                gate_failures.append(f"{domain}: benchmark crashed ({exc}); AUC gate never ran")

    elapsed = time.monotonic() - start

    # Unified summary
    unified["summary"] = {
        "total_domains": len(domains),
        "successful_domains": len(all_aucs),
        "mean_auc_all_domains": round(float(np.mean(all_aucs)), 4) if all_aucs else None,
        "gate_failures": gate_failures,
        "total_elapsed_seconds": round(elapsed, 1),
    }

    # Save unified report
    output_path = BENCHMARKS_DIR / "unified_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(unified, f, indent=2, default=str)
    logger.info("Unified results saved to %s", output_path)

    # Report
    logger.info("=" * 60)
    logger.info("UNIFIED BENCHMARK REPORT")
    logger.info("=" * 60)
    logger.info("Domains: %d/%d successful", len(all_aucs), len(domains))
    if all_aucs:
        logger.info("Mean AUC (all domains): %.4f", np.mean(all_aucs))
    if gate_failures:
        logger.warning("Gate failures: %d", len(gate_failures))
        for gf in gate_failures:
            logger.warning("  - %s", gf)

    if fail_on_gate and gate_failures:
        logger.error("Benchmark FAILED: %d gate failures.", len(gate_failures))
        sys.exit(1)

    return unified


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mercury-Agent Domain Benchmark Runner")
    parser.add_argument(
        "--domains",
        type=str,
        default=None,
        help="Comma-separated list of domains to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all domain benchmarks",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Don't fail on AUC gate thresholds",
    )

    args = parser.parse_args()

    domains = None
    if args.domains:
        domains = [d.strip() for d in args.domains.split(",")]

    run_all(domains=domains, fail_on_gate=not args.no_gate)


if __name__ == "__main__":
    main()

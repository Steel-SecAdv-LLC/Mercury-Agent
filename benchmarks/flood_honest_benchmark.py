#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Flood domain honest benchmark.

Fetches real river gauge data from USGS Water Services,
runs MercuryAnomalyDetector, and reports metrics.

Data source: USGS Water Services + OpenFEMA (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.flood_loader import FloodLoader


def main() -> None:
    """Run flood benchmark."""
    loader = FloodLoader()
    run_domain_benchmark("flood", loader)


if __name__ == "__main__":
    main()

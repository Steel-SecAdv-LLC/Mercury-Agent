#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Landslide domain honest benchmark.

Fetches real landslide catalog data from NASA COOLR,
runs MercuryAnomalyDetector, and reports metrics.

Data source: NASA Global Landslide Catalog (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.landslide_loader import LandslideLoader


def main() -> None:
    """Run landslide benchmark."""
    loader = LandslideLoader()
    run_domain_benchmark("landslide", loader)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Pandemic domain honest benchmark.

Fetches real epidemiological data from Our World in Data,
runs MercuryAnomalyDetector, and reports metrics.

Data source: Our World in Data + WHO GHO (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.pandemic_loader import PandemicLoader


def main() -> None:
    """Run pandemic benchmark."""
    loader = PandemicLoader()
    run_domain_benchmark("pandemic", loader)


if __name__ == "__main__":
    main()

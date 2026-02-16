#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

Sepsis domain honest benchmark.

Uses PhysioNet Challenge 2019 open dataset for sepsis prediction,
runs MercuryAnomalyDetector, and reports metrics.

Data source: PhysioNet/CinC Challenge 2019 (open access).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.sepsis_loader import SepsisLoader


def main() -> None:
    """Run sepsis benchmark."""
    loader = SepsisLoader()
    run_domain_benchmark("sepsis", loader)


if __name__ == "__main__":
    main()

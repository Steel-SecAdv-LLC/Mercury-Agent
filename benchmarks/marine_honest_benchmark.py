#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Marine biodiversity domain honest benchmark.

Fetches real species occurrence data from OBIS,
runs MercuryAnomalyDetector, and reports metrics.

Data source: Ocean Biodiversity Information System (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.marine_loader import MarineLoader


def main() -> None:
    """Run marine biodiversity benchmark."""
    loader = MarineLoader()
    run_domain_benchmark("marine", loader)


if __name__ == "__main__":
    main()

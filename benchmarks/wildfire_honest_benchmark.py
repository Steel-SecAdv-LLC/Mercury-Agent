#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Wildfire domain honest benchmark.

Fetches real fire data from NASA FIRMS, runs MercuryAnomalyDetector,
and reports metrics. Requires NASA_FIRMS_MAP_KEY environment variable.

Data source: NASA FIRMS (free API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader


def main() -> None:
    """Run wildfire benchmark."""
    loader = WildfireLoader()
    run_domain_benchmark("wildfire", loader)


if __name__ == "__main__":
    main()

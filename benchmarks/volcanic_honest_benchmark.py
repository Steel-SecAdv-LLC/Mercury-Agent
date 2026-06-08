#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Volcanic domain honest benchmark.

Fetches real volcanic alert data from USGS Volcano Hazards Program,
runs MercuryAnomalyDetector, and reports metrics.

Data source: USGS Volcano Hazards Program (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.volcanic_loader import VolcanicLoader


def main() -> None:
    """Run volcanic benchmark."""
    loader = VolcanicLoader()
    run_domain_benchmark("volcanic", loader)


if __name__ == "__main__":
    main()

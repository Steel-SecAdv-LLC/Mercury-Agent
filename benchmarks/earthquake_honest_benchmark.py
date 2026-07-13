#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Earthquake domain transparent benchmark.

Fetches real seismic data from USGS, runs MercuryAnomalyDetector,
and reports AUC, F1, precision, recall for each ground-truth event.

Data source: USGS Earthquake Hazards Program (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.earthquake_loader import EarthquakeLoader


def main() -> None:
    """Run earthquake benchmark."""
    loader = EarthquakeLoader()
    run_domain_benchmark("earthquake", loader)


if __name__ == "__main__":
    main()

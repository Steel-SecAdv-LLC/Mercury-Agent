#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""FEMA cross-domain honest benchmark.

Fetches real disaster declaration data from OpenFEMA,
runs MercuryAnomalyDetector, and reports metrics.

Data source: OpenFEMA API (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.fema_loader import FEMALoader


def main() -> None:
    """Run FEMA cross-domain benchmark."""
    loader = FEMALoader()
    run_domain_benchmark("fema", loader)


if __name__ == "__main__":
    main()

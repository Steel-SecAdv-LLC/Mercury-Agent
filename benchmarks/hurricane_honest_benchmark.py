#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hurricane domain honest benchmark.

Fetches real tropical cyclone data from IBTrACS, runs MercuryAnomalyDetector,
and reports metrics for rapid intensification detection.

Data source: NOAA IBTrACS (no API key required).
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.hurricane_loader import HurricaneLoader


def main() -> None:
    """Run hurricane benchmark."""
    loader = HurricaneLoader()
    run_domain_benchmark("hurricane", loader)


if __name__ == "__main__":
    main()

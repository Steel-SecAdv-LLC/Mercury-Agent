#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Energy/EMP domain honest benchmark."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.energy_loader import EnergyLoader


def main() -> None:
    """Run energy/EMP benchmark."""
    loader = EnergyLoader()
    run_domain_benchmark("energy", loader)


if __name__ == "__main__":
    main()

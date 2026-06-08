#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Tsunami domain honest benchmark."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.tsunami_loader import TsunamiLoader


def main() -> None:
    """Run tsunami benchmark."""
    loader = TsunamiLoader()
    run_domain_benchmark("tsunami", loader)


if __name__ == "__main__":
    main()

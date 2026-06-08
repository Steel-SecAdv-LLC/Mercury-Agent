#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Flood domain honest benchmark."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.flood_loader import FloodLoader


def main() -> None:
    """Run flood benchmark."""
    loader = FloodLoader()
    run_domain_benchmark("flood", loader)


if __name__ == "__main__":
    main()

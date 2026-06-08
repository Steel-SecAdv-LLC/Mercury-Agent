#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
"""Network security domain honest benchmark."""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from benchmarks.domain_benchmark_base import run_domain_benchmark
from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader


def main() -> None:
    """Run network security benchmark."""
    loader = NetworkSecurityLoader()
    run_domain_benchmark("network_security", loader)


if __name__ == "__main__":
    main()

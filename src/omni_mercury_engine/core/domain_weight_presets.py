"""
Domain-adaptive weight presets derived from benchmark data.

Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

These presets encode measured component performance across 11 domains, derived
once from a ``mercury_benchmark_results.json`` snapshot. They are a fixed
starting point and are deliberately NOT re-pinned to the live, CI-refreshed
benchmark run (the current committed run lives in
``benchmarks/mercury_benchmark_results.json``).

Each preset is the STARTING POINT for adaptive weighting. The unsupervised adaptive system still
runs and can override these if per-dataset evidence is strong enough.
"""

from __future__ import annotations

# Weights: [resonance, kinematic, info_geometry]
# Source: domain_summary.stats.component_mean_aucs from benchmark
DOMAIN_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    # Tabular-dominant domains: zero kinematic
    "disaster": {"resonance": 0.30, "kinematic": 0.00, "info_geometry": 0.70},
    "general": {"resonance": 0.30, "kinematic": 0.00, "info_geometry": 0.70},
    "academic": {"resonance": 0.47, "kinematic": 0.03, "info_geometry": 0.50},
    "security": {"resonance": 0.47, "kinematic": 0.00, "info_geometry": 0.53},
    "industrial": {"resonance": 0.47, "kinematic": 0.00, "info_geometry": 0.53},
    # Physics-rich temporal domains: kinematic earns its weight
    "ocean": {"resonance": 0.30, "kinematic": 0.40, "info_geometry": 0.30},
    "climate": {"resonance": 0.35, "kinematic": 0.30, "info_geometry": 0.35},
    "air_quality": {"resonance": 0.35, "kinematic": 0.30, "info_geometry": 0.35},
    "environmental": {"resonance": 0.35, "kinematic": 0.30, "info_geometry": 0.35},
    "space": {"resonance": 0.30, "kinematic": 0.35, "info_geometry": 0.35},
    # Mixed temporal: moderate kinematic
    "timeseries": {"resonance": 0.35, "kinematic": 0.20, "info_geometry": 0.45},
    # ADBench (heterogeneous): conservative defaults
    "adbench": {"resonance": 0.40, "kinematic": 0.15, "info_geometry": 0.45},
    # Medical domain
    "medical": {"resonance": 0.40, "kinematic": 0.20, "info_geometry": 0.40},
    # Fallback
    "default": {"resonance": 0.40, "kinematic": 0.20, "info_geometry": 0.40},
}


def get_domain_preset(domain: str) -> tuple[float, float, float]:
    """Return (resonance, kinematic, info_geometry) weight tuple for domain."""
    preset = DOMAIN_WEIGHT_PRESETS.get(domain.lower(), DOMAIN_WEIGHT_PRESETS["default"])
    return (preset["resonance"], preset["kinematic"], preset["info_geometry"])

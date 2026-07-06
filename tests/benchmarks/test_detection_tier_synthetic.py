# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the streaming detector tier synthetic scenarios and benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.detection_tier_benchmark import run_benchmark
from benchmarks.detection_tier_synthetic import SCENARIOS, generate_scenario

_METRIC_KEYS = {"precision", "recall", "f1", "roc_auc", "latency_ms"}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_generator_shape_and_classes(name: str) -> None:
    """Each generator returns aligned (series, labels) with both classes present."""
    series, labels = generate_scenario(name, n=800, seed=0)

    assert series.shape == labels.shape
    assert series.dtype == np.float64
    assert series.ndim == 1
    assert set(np.unique(labels)).issubset({0, 1})
    assert labels.sum() > 0, "expected at least one anomaly"
    assert (labels == 0).sum() > 0, "expected at least one normal point"

    # Anomaly rate stays in a sane band.
    rate = float(labels.mean())
    assert 0.01 <= rate <= 0.12, f"{name} anomaly rate {rate}"

    # Labels are always finite; series is finite except for missing_data dropouts.
    assert np.all(np.isfinite(labels))
    if name == "missing_data":
        assert np.isnan(series).any(), "missing_data should contain NaN dropouts"
        assert np.all(np.isfinite(series[~np.isnan(series)]))
    else:
        assert np.all(np.isfinite(series))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_generator_determinism(name: str) -> None:
    """Identical (name, seed, kwargs) reproduce byte-identical arrays."""
    s1, l1 = generate_scenario(name, n=600, seed=7)
    s2, l2 = generate_scenario(name, n=600, seed=7)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(l1, l2)


def test_generator_seed_changes_output() -> None:
    """Different seeds produce different series (sanity check on seeding)."""
    s1, _ = generate_scenario("burst", n=600, seed=1)
    s2, _ = generate_scenario("burst", n=600, seed=2)
    assert not np.array_equal(s1, s2)


def test_generate_scenario_dispatch_matches_registry() -> None:
    """Dispatcher returns the same arrays as calling the generator directly."""
    for name, generator in SCENARIOS.items():
        s_dispatch, l_dispatch = generate_scenario(name, n=400, seed=3)
        s_direct, l_direct = generator(n=400, seed=3)
        np.testing.assert_array_equal(s_dispatch, s_direct)
        np.testing.assert_array_equal(l_dispatch, l_direct)


def test_generate_scenario_unknown_raises_keyerror() -> None:
    """Unknown scenario names raise KeyError."""
    with pytest.raises(KeyError):
        generate_scenario("does_not_exist")


def test_run_benchmark_smoke_subset() -> None:
    """A tiny 1-scenario / 2-detector run returns the expected nested structure."""
    results = run_benchmark(
        seed=0,
        n=240,
        scenarios=["burst"],
        members=["spectral_residual", "energy_based"],
    )

    assert set(results) == {"config", "scenarios", "aggregate", "skipped"}
    assert "burst" in results["scenarios"]

    burst = results["scenarios"]["burst"]
    for name in ("spectral_residual", "energy_based"):
        entry = burst["detectors"][name]
        if "error" in entry:
            continue
        assert _METRIC_KEYS.issubset(entry)
        assert 0.0 <= entry["roc_auc"] <= 1.0
        assert entry["latency_ms"] >= 0.0

    # All three ensemble methods are attempted and carry metric keys.
    ensembles = burst["ensembles"]
    assert set(ensembles) == {"stacking", "bma", "average"}
    for stats in ensembles.values():
        if "error" in stats:
            continue
        assert _METRIC_KEYS.issubset(stats)

    # Aggregate carries per-detector and per-ensemble means.
    assert "spectral_residual" in results["aggregate"]["detectors"]
    assert set(results["aggregate"]["ensembles"]) == {"stacking", "bma", "average"}

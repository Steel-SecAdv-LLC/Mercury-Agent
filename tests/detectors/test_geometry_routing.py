"""Option B: geometry routing tests."""
import numpy as np
import pytest
from omni_mercury_engine.detectors.math_arrest.geometry_classifier import (
    classify_geometry,
    probes_for_geometries,
)
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest


def test_classify_temporal_autocorrelated() -> None:
    """Strongly autocorrelated data must be classified as temporal."""
    rng = np.random.RandomState(0)
    # AR(1) process with high autocorrelation
    n = 200
    x = np.zeros((n, 3))
    for i in range(1, n):
        x[i] = 0.95 * x[i - 1] + rng.randn(3) * 0.1
    geometries = classify_geometry(x)
    assert "temporal" in geometries, f"Expected 'temporal' in {geometries}"


def test_classify_point_heavy_tail() -> None:
    """Heavy-tailed distribution must be classified as point geometry."""
    rng = np.random.RandomState(1)
    from scipy.stats import t as t_dist
    x = t_dist.rvs(df=2, size=(300, 5), random_state=rng)  # df=2 -> very heavy tails
    geometries = classify_geometry(x)
    assert "point" in geometries, f"Expected 'point' in {geometries}"


def test_classify_distributional_heterogeneous_variance() -> None:
    """Features with very different variances must trigger distributional."""
    rng = np.random.RandomState(2)
    x = np.column_stack([
        rng.randn(200) * 100,
        rng.randn(200) * 0.01,
        rng.randn(200),
    ])
    geometries = classify_geometry(x)
    assert "distributional" in geometries, f"Expected 'distributional' in {geometries}"


def test_probes_for_unknown_geometry_returns_all_sentinel() -> None:
    """Unknown geometry must return ['all'] sentinel for full probe activation."""
    result = probes_for_geometries(["unknown"])
    assert result == ["all"]


def test_probes_for_known_geometry_includes_minimum_set() -> None:
    """All geometry-routed probe lists must include MINIMUM_PROBE_SET probes."""
    from omni_mercury_engine.detectors.math_arrest.geometry_classifier import MINIMUM_PROBE_SET

    for geom in ["point", "distributional", "collective", "temporal"]:
        probes = probes_for_geometries([geom])
        for must_have in MINIMUM_PROBE_SET:
            assert must_have in probes, (
                f"Minimum probe {must_have} missing from {geom} probe list: {probes}"
            )


def test_geometry_routing_selects_fewer_probes_on_tabular() -> None:
    """Data with clear distributional geometry should activate a targeted
    subset of probes, not all 21."""
    rng = np.random.RandomState(42)
    # Heterogeneous variance: feature 0 std=100, feature 1 std=0.01
    # This triggers DISTRIBUTIONAL and possibly POINT but not TEMPORAL
    X = np.column_stack([
        rng.randn(300) * 100,
        rng.randn(300) * 0.01,
        rng.randn(300),
    ])
    ama = AnomalyMathArrest(geometry_routing=True)
    ama.fit(X)
    n_active = len(ama._probes)
    # Should not activate all 21 probes when geometry is known
    assert n_active < 21, f"Expected < 21 probes on distributional data, got {n_active}"
    assert n_active >= 3, f"Must have at least 3 probes (MINIMUM_PROBE_SET), got {n_active}"


def test_detected_geometries_exposed_after_fit() -> None:
    """detected_geometries property must be populated after fit."""
    rng = np.random.RandomState(9)
    X = rng.randn(100, 5)
    ama = AnomalyMathArrest(geometry_routing=True)
    ama.fit(X)
    assert len(ama.detected_geometries) >= 1, "Must detect at least one geometry after fit"

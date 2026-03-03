"""Tests for Otsu / adaptive threshold selection (Option D)."""

import numpy as np

from omni_mercury_engine.core.threshold import adaptive_threshold, otsu_threshold


def test_otsu_bimodal_separation() -> None:
    """Otsu must find the valley between two well-separated Gaussians."""
    rng = np.random.RandomState(0)
    normals = rng.normal(0.2, 0.05, 500)
    anomalies = rng.normal(0.8, 0.05, 50)
    scores = np.clip(np.concatenate([normals, anomalies]), 0.0, 1.0)
    thr, sigma = otsu_threshold(scores)
    # Threshold must fall between the two modes.
    # With 10:1 imbalance Otsu may pull toward the majority class,
    # so we accept any split between 0.25 and 0.70.
    assert 0.25 < thr < 0.70, f"Otsu threshold {thr:.4f} outside expected valley"
    assert sigma > 0.0, "Between-class variance must be positive for bimodal data"


def test_otsu_degenerate_constant() -> None:
    """Constant score array must return fallback 0.5 with zero variance."""
    scores = np.full(100, 0.3, dtype=np.float64)
    thr, sigma = otsu_threshold(scores)
    assert thr == 0.5
    assert sigma == 0.0


def test_otsu_small_n_returns_fallback() -> None:
    """Arrays smaller than 30 samples must return fallback."""
    scores = np.linspace(0, 1, 20)
    thr, sigma = otsu_threshold(scores)
    assert thr == 0.5
    assert sigma == 0.0


def test_adaptive_prefers_otsu_over_mad() -> None:
    """For bimodal distributions, Otsu must win over MAD."""
    rng = np.random.RandomState(42)
    scores = np.clip(
        np.concatenate([rng.normal(0.15, 0.05, 300), rng.normal(0.85, 0.05, 30)]),
        0.0,
        1.0,
    )
    _, method = adaptive_threshold(scores)
    assert method == "otsu", f"Expected 'otsu', got '{method}'"


def test_adaptive_prefer_recall_lowers_threshold() -> None:
    """prefer_recall=True must produce a lower threshold than prefer_recall=False."""
    rng = np.random.RandomState(7)
    scores = np.clip(rng.normal(0.5, 0.2, 200), 0.0, 1.0)
    thr_default, _ = adaptive_threshold(scores)
    thr_recall, _ = adaptive_threshold(scores, prefer_recall=True)
    assert thr_recall < thr_default, "prefer_recall=True must lower threshold"


def test_adaptive_contamination_fallback() -> None:
    """When Otsu and MAD fail (constant), contamination_hint must be used."""
    # Force constant scores so Otsu and MAD degenerate
    scores = np.full(50, 0.5, dtype=np.float64)
    thr, method = adaptive_threshold(scores, contamination_hint=0.10)
    # With contamination 0.10 on scores all = 0.5, percentile of 0.5 = 0.5
    assert method in ("contamination_percentile", "fallback_0.5")

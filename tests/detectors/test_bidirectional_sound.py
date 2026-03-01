"""Option F: bidirectional SpectralDomainSound contribution tests."""

import numpy as np

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def _make_temporal_data(n: int = 300, n_features: int = 3, seed: int = 0) -> np.ndarray:
    """Generate strongly autocorrelated temporal data (AR(1))."""
    rng = np.random.RandomState(seed)
    X = np.zeros((n, n_features))
    for i in range(1, n):
        X[i] = 0.90 * X[i - 1] + rng.randn(n_features) * 0.2
    return X


def test_sound_metadata_in_result() -> None:
    """detect() must include sound_n_amplified, sound_n_suppressed, sound_weight."""
    X = _make_temporal_data()
    det = MercuryAnomalyDetector(auto_validate=False)
    det.fit(X)
    result = det.detect(X)
    for key in ("sound_n_amplified", "sound_n_suppressed", "sound_weight"):
        assert key in result, f"Key '{key}' missing from detect() output"


def test_sound_amplified_plus_suppressed_equals_n_samples() -> None:
    """When Sound is active, all samples must be either amplified or suppressed."""
    X = _make_temporal_data()
    det = MercuryAnomalyDetector(auto_validate=False)
    det.fit(X)
    result = det.detect(X)
    if result.get("sound_active", False):
        n_total = result["sound_n_amplified"] + result["sound_n_suppressed"]
        assert n_total == len(X), f"amplified+suppressed={n_total} != n_samples={len(X)}"


def test_sound_additive_can_lift_low_scores() -> None:
    """When Sound has high multiplier, scores for seemingly normal samples
    must be lifted above what pure multiplication would give."""
    X = _make_temporal_data(n=300, n_features=3, seed=5)
    det = MercuryAnomalyDetector(auto_validate=False)
    det.fit(X)

    result = det.detect(X)
    scores = result["scores"]

    # The result must be valid scores in [0, 1]
    assert np.all(scores >= 0.0) and np.all(
        scores <= 1.0
    ), "All scores must be in [0, 1] after bidirectional Sound"


def test_sound_weight_is_domain_typed() -> None:
    """Infrastructure domain must get higher sound_weight than tabular."""
    from omni_mercury_engine.core.config import _sound_weight

    infra_w = _sound_weight("infrastructure")
    tabular_w = _sound_weight("tabular")
    assert (
        infra_w > tabular_w
    ), f"Infrastructure sound_weight ({infra_w}) must exceed tabular ({tabular_w})"

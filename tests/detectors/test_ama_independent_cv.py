"""Option C: AMA CV pseudo-labels must be independent of Mercury."""
import numpy as np
from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def test_detect_per_probe_shape() -> None:
    """detect_per_probe must return (n_samples, n_active_probes)."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 5)
    ama = AnomalyMathArrest()
    ama.fit(X)
    matrix = ama.detect_per_probe(X)
    assert matrix.shape == (100, len(ama._probes)), (
        f"Expected (100, {len(ama._probes)}), got {matrix.shape}"
    )
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0), \
        "All scores must be in [0, 1]"


def test_detect_per_probe_column_variance() -> None:
    """Each probe column must show meaningful variance — not all 0.5."""
    rng = np.random.RandomState(1)
    X = rng.randn(200, 4)
    ama = AnomalyMathArrest()
    ama.fit(X)
    matrix = ama.detect_per_probe(X)
    # At least half the probes must have non-trivial variance
    col_stds = np.std(matrix, axis=0)
    active = np.sum(col_stds > 1e-3)
    assert active >= len(ama._probes) // 2, (
        f"Only {active}/{len(ama._probes)} probes show variance > 0.001"
    )


def test_ama_cv_weights_shift_on_degraded_mercury() -> None:
    """When Mercury degrades, AMA should still get appropriate weight.

    With independent pseudo-labels, AMA weight is not dragged down by
    Mercury's degraded performance on the same folds.
    """
    rng = np.random.RandomState(42)
    # Use random uniform data where Mercury components struggle
    X = rng.uniform(0, 1, (300, 10))
    det = MercuryAnomalyDetector(auto_validate=False, enable_ama=True)
    det.fit(X)
    # AMA should get meaningful weight even on difficult data
    assert det._ama_weight >= 0.15, (
        f"AMA weight {det._ama_weight:.3f} too low for data where "
        f"Mercury components struggle"
    )

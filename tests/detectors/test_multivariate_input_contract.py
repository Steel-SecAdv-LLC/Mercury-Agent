# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MultivariateTSDetector refuses malformed input with a message, not an AxisError.

A 2-D array used to die inside ``np.concatenate`` with a bare ``AxisError``,
and 3-D input with the wrong trailing dimensions was silently reshaped
against the configured ``num_features``, mixing columns across features.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors.multivariate_timeseries import MultivariateTSDetector

CONFIG = {"window_size": 20, "num_features": 4}


def _windows(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 20, 4))


def test_two_dimensional_input_is_refused_with_the_windowing_hint() -> None:
    detector = MultivariateTSDetector(CONFIG)
    with pytest.raises(InputModalityError, match=r"window_size=20.*got shape \(300, 16\)"):
        detector.fit(np.random.default_rng(1).normal(size=(300, 16)))


def test_wrong_trailing_dimensions_are_refused_naming_the_config() -> None:
    detector = MultivariateTSDetector(CONFIG)
    detector.fit(_windows(50))
    with pytest.raises(InputModalityError, match=r"configured for.*window_size.*\(20, 7\)"):
        detector.predict(np.random.default_rng(2).normal(size=(10, 20, 7)))


def test_empty_batch_is_refused() -> None:
    detector = MultivariateTSDetector(CONFIG)
    with pytest.raises(InputModalityError, match="empty"):
        detector.fit(np.empty((0, 20, 4)))


def test_well_formed_input_still_fits_and_scores() -> None:
    detector = MultivariateTSDetector(CONFIG)
    detector.fit(_windows(60))
    shifted = _windows(12, seed=3)
    shifted[:3] += 5.0
    result = detector.predict(shifted)
    assert result["anomaly_scores"].shape == (12,)
    assert np.all(np.isfinite(result["anomaly_scores"]))
    assert result["anomaly_scores"][:3].mean() > result["anomaly_scores"][3:].mean()

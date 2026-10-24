# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The topological detector's threshold is calibrated on the reference it was capped to.

``n_reference_samples`` caps the points the reference profile is built from.
The threshold used to be calibrated on windows cut from the *uncapped* data, so
the cap neither bounded the cost of ``fit`` (a window of a third of the data is
built ten times: 167 s on 1886 thyroid rows) nor matched the reference the
windows are compared with.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.detectors.topological_analysis import (
    TopologicalAnomalyDetector,
    VietorisRipsFiltration,
)


@pytest.fixture
def point_counts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Row counts of every point cloud a filtration is built on."""
    seen: list[int] = []
    original = VietorisRipsFiltration.build

    def spy(self: VietorisRipsFiltration, point_cloud: np.ndarray) -> object:
        seen.append(len(point_cloud))
        return original(self, point_cloud)

    monkeypatch.setattr(VietorisRipsFiltration, "build", spy)
    return seen


def test_no_filtration_is_built_on_more_points_than_the_cap(point_counts: list[int]) -> None:
    data = np.random.default_rng(0).normal(size=(300, 3))
    TopologicalAnomalyDetector(n_reference_samples=60, seed=0).fit(data)
    assert point_counts, "fit must build the reference diagram"
    assert max(point_counts) <= 60


def test_without_a_cap_calibration_windows_are_a_third_of_the_data(
    point_counts: list[int],
) -> None:
    data = np.random.default_rng(0).normal(size=(90, 3))
    TopologicalAnomalyDetector(seed=0).fit(data)
    assert point_counts[0] == 90
    assert set(point_counts[1:]) == {30}


def test_a_capped_fit_is_deterministic_for_one_seed() -> None:
    data = np.random.default_rng(1).normal(size=(200, 3))
    a = TopologicalAnomalyDetector(n_reference_samples=50, seed=3).fit(data)
    b = TopologicalAnomalyDetector(n_reference_samples=50, seed=3).fit(data)
    assert a.anomaly_threshold == b.anomaly_threshold

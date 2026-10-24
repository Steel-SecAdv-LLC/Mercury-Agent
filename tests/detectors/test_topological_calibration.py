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


def test_the_scoring_baseline_is_measured_at_window_cardinality() -> None:
    """Betti numbers, entropies and landscape norms scale with the point count.

    The profile the whole reference builds is a 60-point cloud; every window
    :meth:`score` is asked about here is 20 points.  Comparing one against the
    other adds a constant size bias to every deviation, so the baseline the
    score is measured from is the mean over reference *windows*, not the
    whole-reference diagram.
    """
    data = np.random.default_rng(0).normal(size=(300, 3))
    detector = TopologicalAnomalyDetector(n_reference_samples=60, seed=0).fit(data)

    assert detector._reference_features["n_features_h0"] == 60.0
    assert detector._window_baseline["n_features_h0"] == 20.0


def test_an_explicit_threshold_moves_the_cut_point_and_not_the_score() -> None:
    """``anomaly_threshold`` is documented as the decision cut, nothing more.

    It used to skip calibration altogether, so the feature spreads stayed empty
    and :meth:`score` fell back to its near-zero-variance branch: the same clean
    window scored 0.918 with an auto threshold and 0.085 with an explicit one.
    """
    data = np.random.default_rng(0).normal(size=(300, 3))
    window = np.random.default_rng(7).normal(size=(20, 3))

    auto = TopologicalAnomalyDetector(n_reference_samples=60, seed=0).fit(data)
    explicit = TopologicalAnomalyDetector(
        n_reference_samples=60, seed=0, anomaly_threshold=0.5
    ).fit(data)

    assert explicit.anomaly_threshold == 0.5
    assert auto.anomaly_threshold != 0.5
    assert explicit.score(window)["anomaly_score"] == pytest.approx(
        auto.score(window)["anomaly_score"]
    )


def test_reference_windows_score_well_below_the_saturating_region() -> None:
    """A size-biased baseline drove every deviation into the sigmoid's ceiling.

    ``2 / (1 + exp(-mean_dev / 2)) - 1`` reaches 0.92 at a mean deviation of
    about 10 sigma, which is where clean reference windows used to land.
    """
    data = np.random.default_rng(0).normal(size=(300, 3))
    detector = TopologicalAnomalyDetector(n_reference_samples=60, seed=0).fit(data)

    assert detector.anomaly_threshold is not None
    assert detector.anomaly_threshold < 0.6


def test_too_few_rows_to_cut_windows_falls_back_to_the_reference_profile() -> None:
    """Under four rows no window can be cut, so the whole-reference profile stands."""
    detector = TopologicalAnomalyDetector(seed=0).fit(np.zeros((3, 2)))

    assert detector._window_baseline == {}
    assert detector._feature_stds
    assert detector.anomaly_threshold == 0.5
    assert 0.0 <= detector.score(np.zeros((3, 2)))["anomaly_score"] <= 1.0


def test_refitting_recalibrates_an_auto_threshold() -> None:
    """A second ``fit`` used to keep the first one's threshold.

    The guard was ``if self.anomaly_threshold is None``, which is false after
    any auto-calibration, so refitting on different reference data moved the
    window baseline and feature spreads but left the old cut point behind.
    """
    tight = np.random.default_rng(0).normal(scale=0.05, size=(200, 3))
    loose = np.random.default_rng(1).normal(scale=5.0, size=(200, 3))

    refitted = TopologicalAnomalyDetector(n_reference_samples=60, seed=0)
    refitted.fit(tight)
    first = refitted.anomaly_threshold
    refitted.fit(loose)

    fresh = TopologicalAnomalyDetector(n_reference_samples=60, seed=0).fit(loose)
    assert refitted.anomaly_threshold != first
    assert refitted.anomaly_threshold == fresh.anomaly_threshold
    assert refitted._window_baseline == fresh._window_baseline


def test_refitting_never_overwrites_a_configured_threshold() -> None:
    data = np.random.default_rng(0).normal(size=(200, 3))
    detector = TopologicalAnomalyDetector(n_reference_samples=60, seed=0, anomaly_threshold=0.42)

    detector.fit(data).fit(np.random.default_rng(1).normal(scale=5.0, size=(200, 3)))

    assert detector.anomaly_threshold == 0.42

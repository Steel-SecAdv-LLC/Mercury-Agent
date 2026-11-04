# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The statistical detector's default false-alarm rate, measured on labelled data.

``MercuryAnomalyDetector`` decides with a fit-time operating point: the
training rows' own ensemble scores cut at the one-class split-conformal
quantile, so a new row exchangeable with them exceeds the cut with probability
at most ``reference_alpha`` (0.05). The batch-relative cut it replaced (Otsu
valley split, else median + 2 * 1.4826 * MAD) was chosen for F1 and flagged
far more normal data than that; ``operating_point="adaptive"`` still selects it.

Protocol: the stratified 70/30 split of ``tests/recorded_data`` over the five
recorded ADBench tables, fitted on the *normal* training rows only (novelty
detection, the setting the conformal bound covers), measured on the held-out
rows; mean over three seeds. False-alarm rate = share of held-out normal rows
flagged; recall = share of held-out anomalies flagged. The numbers are the
measured behaviour on these tables, pinned so a change must be recorded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from tests.recorded_data import ADBENCH_NAMES, adbench, stratified_split

SEEDS = (0, 1, 2)
ALPHA = 0.05

#: operating point -> table -> (false-alarm rate, recall), ``ADBENCH_NAMES`` order.
MEASURED: dict[str, dict[str, tuple[float, float]]] = {
    "reference": {
        "pima_real": (0.053, 0.158),
        "cardio": (0.065, 0.827),
        "breastw": (0.048, 0.977),
        "wbc": (0.026, 1.000),
        "thyroid": (0.049, 0.963),
    },
    "adaptive": {
        "pima_real": (0.113, 0.329),
        "cardio": (0.083, 0.853),
        "breastw": (0.048, 0.986),
        "wbc": (0.127, 1.000),
        "thyroid": (0.076, 0.988),
    },
}
FPR_TOLERANCE = 0.02
RECALL_TOLERANCE = 0.05


def _measure(table: str, operating_point: str) -> tuple[float, float]:
    features, labels = adbench(table)
    false_alarms: list[float] = []
    recalls: list[float] = []
    for seed in SEEDS:
        train, test = stratified_split(labels, seed)
        normal_train = train[labels[train] == 0]
        detector = MercuryAnomalyDetector({"operating_point": operating_point})
        detector.fit(features[normal_train])
        flags = np.asarray(detector.detect(features[test])["is_anomaly"], dtype=bool)
        held = labels[test]
        false_alarms.append(float(flags[held == 0].mean()))
        recalls.append(float(flags[held == 1].mean()))
    return float(np.mean(false_alarms)), float(np.mean(recalls))


@pytest.fixture(scope="module")
def measured() -> dict[tuple[str, str], tuple[float, float]]:
    return {(op, table): _measure(table, op) for op in MEASURED for table in ADBENCH_NAMES}


@pytest.mark.parametrize("operating_point", sorted(MEASURED))
@pytest.mark.parametrize("table", ADBENCH_NAMES)
def test_false_alarms_and_recall_match_the_measurement(
    table: str,
    operating_point: str,
    measured: dict[tuple[str, str], tuple[float, float]],
) -> None:
    pinned_fpr, pinned_recall = MEASURED[operating_point][table]
    fpr, recall = measured[(operating_point, table)]
    assert fpr == pytest.approx(pinned_fpr, abs=FPR_TOLERANCE), (
        f"{operating_point}/{table}: false-alarm rate {fpr:.3f} vs measured "
        f"{pinned_fpr:.3f}; record the change if it is real"
    )
    assert recall == pytest.approx(pinned_recall, abs=RECALL_TOLERANCE), (
        f"{operating_point}/{table}: recall {recall:.3f} vs measured "
        f"{pinned_recall:.3f}; record the change if it is real"
    )


def test_the_default_holds_its_false_alarm_budget(
    measured: dict[tuple[str, str], tuple[float, float]],
) -> None:
    """Mean over the tables within finite-sample reach of alpha (0.048 measured).

    The conformal bound is marginal: ``E[FPR] <= alpha + 1 / (n + 1)`` over
    calibration draws, so a single table at a few hundred training rows can sit
    a point or two above alpha; the mean over five tables and three seeds may not.
    """
    rates = [measured[("reference", table)][0] for table in ADBENCH_NAMES]
    assert float(np.mean(rates)) <= ALPHA + 0.01
    assert max(rates) <= ALPHA + 0.03


def test_the_default_flags_less_normal_data_than_the_batch_relative_cut(
    measured: dict[tuple[str, str], tuple[float, float]],
) -> None:
    reference = np.mean([measured[("reference", t)][0] for t in ADBENCH_NAMES])
    adaptive = np.mean([measured[("adaptive", t)][0] for t in ADBENCH_NAMES])
    assert reference < adaptive


@pytest.mark.parametrize("width", [1, 8])
def test_pure_noise_is_flagged_near_alpha(width: int) -> None:
    """Gaussian noise carries no anomaly, so every flag is false.

    Measured: 0.064 (width 1) and 0.058 (width 8) for the default; the
    batch-relative cut flagged 0.171 and 0.071 of the same rows.
    """
    rows = np.random.default_rng(0).standard_normal((2000, width))
    detector = MercuryAnomalyDetector()
    detector.fit(rows[:1000])
    result = detector.detect(rows[1000:])
    flags = np.asarray(result["is_anomaly"], dtype=bool)
    assert result["calibration_diagnostics"]["method"] == "reference_conformal"
    assert flags.mean() <= ALPHA + 0.025


def _fitted(config: dict[str, Any] | None = None) -> MercuryAnomalyDetector:
    detector = MercuryAnomalyDetector(config)
    detector.fit(np.random.default_rng(0).normal(0.0, 1.0, (500, 8)))
    return detector


class TestOperatingPointContract:
    def test_default_is_the_fit_reference_cut(self) -> None:
        detector = _fitted()
        rows = np.random.default_rng(1).normal(0.0, 1.0, (100, 8))
        result = detector.detect(rows)
        diagnostics = result["calibration_diagnostics"]
        assert diagnostics["method"] == "reference_conformal"
        assert diagnostics["regime"] == "fit_reference"
        assert result["threshold"] == pytest.approx(detector._reference_threshold())

    def test_the_cut_does_not_depend_on_the_batch(self) -> None:
        """The same row gets the same verdict alone, in a normal batch, or in a flood."""
        detector = _fitted()
        rng = np.random.default_rng(2)
        row = np.full((1, 8), 3.0)
        normal = np.vstack([row, rng.normal(0.0, 1.0, (99, 8))])
        flood = np.vstack([row, rng.normal(6.0, 1.0, (99, 8))])
        alone = bool(np.asarray(detector.detect(row)["is_anomaly"])[0])
        assert bool(np.asarray(detector.detect(normal)["is_anomaly"])[0]) is alone
        assert bool(np.asarray(detector.detect(flood)["is_anomaly"])[0]) is alone

    def test_adaptive_is_selectable(self) -> None:
        detector = _fitted({"operating_point": "adaptive"})
        rows = np.random.default_rng(3).normal(0.0, 1.0, (100, 8))
        diagnostics = detector.detect(rows)["calibration_diagnostics"]
        assert diagnostics["method"] == "adaptive"

    def test_an_explicit_threshold_is_honoured(self) -> None:
        detector = _fitted({"threshold": 0.9})
        result = detector.detect(np.random.default_rng(4).normal(0.0, 1.0, (50, 8)))
        assert result["threshold"] == 0.9

    @pytest.mark.parametrize("bad", ["Reference ", "otsu", ""])
    def test_operating_point_is_validated(self, bad: str) -> None:
        if bad.strip().lower() == "reference":
            MercuryAnomalyDetector({"operating_point": bad})  # normalised, accepted
            return
        with pytest.raises(ValueError, match="operating_point"):
            MercuryAnomalyDetector({"operating_point": bad})

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
    def test_reference_alpha_is_validated(self, bad: float) -> None:
        with pytest.raises(ValueError, match="reference_alpha"):
            MercuryAnomalyDetector({"reference_alpha": bad})


def test_info_geometry_certificate_uses_the_component_scale() -> None:
    """The certificate's cut is the component's own fit-time conformal cut.

    The fused reference is a distribution of *ensemble* scores. The
    batch-relative rule's degenerate (< 8 rows) and anomalous-bulk branches
    used to hand that cut (0.577 here) to the component, whose own cut is
    0.617: in 4-row batches of normal rows the verdict flagged 0.093 of them.
    """
    detector = _fitted({"info_geometry_certificate_enabled": True})
    flood = np.random.default_rng(5).normal(6.0, 1.0, (100, 8))
    result = detector.detect(flood)
    certificate = result.get("info_geometry_certificate")
    assert certificate is not None
    expected = detector._conformal_cut(detector._reference_info_geometry)
    assert expected is not None
    assert certificate["component_threshold_score"] == pytest.approx(expected)
    assert certificate["component_threshold_score"] != pytest.approx(
        detector._reference_threshold()
    )
    # Held-out normal rows exceed the component cut near the shared budget.
    normal = np.random.default_rng(6).normal(0.0, 1.0, (2000, 8))
    info_geo = np.asarray(detector.detect(normal)["info_geometry_scores"], dtype=float)
    assert float(np.mean(info_geo > expected)) <= ALPHA + 0.025


def test_component_reference_survives_state_round_trip() -> None:
    detector = _fitted()
    state = detector.get_fitted_state()
    assert state is not None
    restored = MercuryAnomalyDetector()
    restored.set_fitted_state(state)
    assert restored._conformal_cut(restored._reference_info_geometry) == pytest.approx(
        detector._conformal_cut(detector._reference_info_geometry)
    )

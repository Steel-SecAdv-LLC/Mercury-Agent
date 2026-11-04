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


class TestConformalPValuesAndBatchVerdict:
    """``p_values`` / ``fdr_flags`` / ``batch_anomaly`` on top of the fit-time reference."""

    def test_p_values_are_super_uniform_on_normal_rows(self) -> None:
        """Marginal over reference draws, P(p <= a) <= a (measured 0.0016 / 0.0044 /
        0.0093 / 0.0506 at a = 0.002 / 0.005 / 0.01 / 0.05 over 12 draws)."""
        rates = []
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            detector = MercuryAnomalyDetector()
            detector.fit(rng.standard_normal((1000, 8)))
            p = np.asarray(detector.detect(rng.standard_normal((3000, 8)))["p_values"])
            rates.append([float(np.mean(p <= a)) for a in (0.01, 0.05)])
        mean = np.mean(rates, axis=0)
        assert mean[0] <= 0.01 + 0.006
        assert mean[1] <= 0.05 + 0.012

    def test_p_value_is_the_conformal_rank(self) -> None:
        detector = _fitted()
        ref = detector._reference_scores
        assert ref is not None
        result = detector.detect(np.random.default_rng(11).normal(0.0, 1.0, (20, 8)))
        scores = np.asarray(result["scores"], dtype=float)
        expected = (1.0 + np.array([(ref >= s).sum() for s in scores])) / (len(ref) + 1.0)
        np.testing.assert_allclose(result["p_values"], expected)

    def test_benjamini_hochberg_matches_the_step_up_rule(self) -> None:
        p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
        flags = MercuryAnomalyDetector._benjamini_hochberg(p, 0.05)
        # Largest k with p_(k) <= 0.05 k / 10 is k = 2 (0.008 <= 0.010; 0.039 > 0.015).
        assert flags.tolist() == [True, True] + [False] * 8

    def test_batch_verdict_holds_its_level_where_or_ing_rows_does_not(self) -> None:
        """100 all-normal rows: the per-row OR fires every time, BH at q = 0.05 rarely."""
        detector = _fitted()
        rng = np.random.default_rng(12)
        or_rate, bh_rate = [], []
        for _ in range(40):
            result = detector.detect(rng.normal(0.0, 1.0, (100, 8)))
            or_rate.append(bool(np.asarray(result["is_anomaly"]).any()))
            bh_rate.append(bool(result["batch_anomaly"]))
        assert np.mean(or_rate) > 0.9
        assert np.mean(bh_rate) <= 0.15

    def test_a_gross_outlier_makes_the_batch_anomalous(self) -> None:
        detector = _fitted()
        rows = np.random.default_rng(13).normal(0.0, 1.0, (10, 8))
        rows[0] = 50.0
        result = detector.detect(rows)
        assert result["batch_anomaly"] is True
        assert bool(np.asarray(result["fdr_flags"])[0]) is True

    @pytest.mark.parametrize("bad", [0.0, 1.0, 2.0])
    def test_fdr_level_is_validated(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fdr_level"):
            MercuryAnomalyDetector({"fdr_level": bad})


def _batch_rates(detector: Any, sizes: tuple[int, ...] = (10, 100, 2000)) -> list[float]:
    rng = np.random.default_rng(21)
    rates = []
    for n in sizes:
        repeats = max(3, 2000 // n)  # at least 2,000 rows per size: sd ~0.005 at 0.05
        flagged = []
        for _ in range(repeats):
            if hasattr(detector, "clear_memory"):
                detector.clear_memory()
            flagged.append(
                float(np.mean(detector.detect(rng.standard_normal((n, 8)))["is_anomaly"]))
            )
        rates.append(float(np.mean(flagged)))
    return rates


class TestEngineBaseDetectorsAreBatchInvariant:
    """A fitted detector's verdict on a row must not depend on its batch-mates.

    Before: spatial and dimensional min-max normalised each channel over the
    batch, and directive blended three whole-batch scalars into every row, so
    held-out Gaussian rows were flagged at 0.40 / 0.30 / 0.10 in 10-row
    batches against 0.012 / 0.077 / 0.045 in 2,000-row ones, and
    ``engine.detect`` called every 10-row normal batch anomalous.
    """

    @staticmethod
    def _spatial() -> Any:
        from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector

        detector = SpatialAnomalyDetector({"spatial_autocorrelation": False})
        detector.fit(np.random.default_rng(0).standard_normal((1000, 8)))
        return detector

    @staticmethod
    def _dimensional() -> Any:
        torch = pytest.importorskip("torch")
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        torch.manual_seed(0)
        detector = DimensionalAnalyzer()
        detector.fit(np.random.default_rng(0).standard_normal((1000, 8)))
        return detector

    @staticmethod
    def _directive() -> Any:
        from omni_mercury_engine.detectors.directive import SigmaDirectiveDetector

        detector = SigmaDirectiveDetector()
        detector.fit(np.random.default_rng(0).standard_normal((1000, 8)))
        return detector

    @pytest.mark.parametrize("make", ["_spatial", "_dimensional", "_directive"])
    def test_scores_do_not_depend_on_the_batch(self, make: str) -> None:
        detector = getattr(self, make)()
        rows = np.random.default_rng(22).standard_normal((40, 8))
        crowd = np.vstack([rows, np.random.default_rng(23).standard_normal((600, 8))])
        if hasattr(detector, "clear_memory"):
            detector.clear_memory()
        alone = np.asarray(detector.detect(rows)["scores"], dtype=float)
        if hasattr(detector, "clear_memory"):
            detector.clear_memory()
        together = np.asarray(detector.detect(crowd)["scores"], dtype=float)[:40]
        np.testing.assert_allclose(alone, together, atol=1e-9)

    @pytest.mark.parametrize(
        ("make", "pinned"),
        [
            ("_spatial", (0.045, 0.047, 0.054)),
            ("_dimensional", (0.051, 0.058, 0.063)),
            # RMD scores a row against the rows before it in its own stream,
            # by design, so a 10-row batch gives it less history.
            ("_directive", (0.032, 0.044, 0.050)),
        ],
    )
    def test_false_alarm_rate_is_the_same_at_every_batch_size(
        self, make: str, pinned: tuple[float, ...]
    ) -> None:
        """Measured on >= 2,000 held-out Gaussian rows per batch size (10, 100, 2,000)."""
        rates = _batch_rates(getattr(self, make)())
        assert max(rates) <= ALPHA + 0.02
        np.testing.assert_allclose(rates, pinned, atol=0.015)

    def test_lof_queries_use_the_published_reachability(self) -> None:
        """reach-dist(p, o) = max(k-distance(o), d(p, o)) at query time, as at fit:
        a training point queried against the others reproduces its own LOF."""
        detector = self._spatial()
        lof = detector.lof
        points = np.asarray(lof._tree.data)
        loo = lof.training_decision()
        k = min(lof.k, len(points) - 1)
        dists, idx = lof._tree.query(points[:5], k=k + 1)
        dists, idx = dists[:, 1:], idx[:, 1:]
        reach = np.maximum(dists, lof._kdist[idx])
        lrd = 1.0 / np.maximum(reach.mean(axis=1), lof._REACH_FLOOR)
        expected = -(lof._lrd[idx].mean(axis=1) / lrd - 1.0)
        np.testing.assert_allclose(loo[:5], expected)

    def test_spatial_anchors_survive_a_state_round_trip(self) -> None:
        from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector

        detector = self._spatial()
        restored = SpatialAnomalyDetector({"spatial_autocorrelation": False})
        state = detector.get_fitted_state()
        assert state is not None
        restored.set_fitted_state(state)
        rows = np.random.default_rng(24).standard_normal((30, 8))
        np.testing.assert_allclose(restored.detect(rows)["scores"], detector.detect(rows)["scores"])
        assert restored.threshold == pytest.approx(detector.threshold)

    def test_dimensional_anchors_survive_a_state_round_trip(self) -> None:
        from omni_mercury_engine.detectors.dimensional import DimensionalAnalyzer

        detector = self._dimensional()
        state = detector.get_fitted_state()
        assert state is not None
        restored = DimensionalAnalyzer()
        restored.set_fitted_state(state)
        rows = np.random.default_rng(25).standard_normal((30, 8))
        np.testing.assert_allclose(
            restored.detect(rows)["scores"], detector.detect(rows)["scores"], atol=1e-6
        )
        assert restored.threshold == pytest.approx(detector.threshold)


def test_engine_batch_verdict_uses_the_conformal_test_where_it_has_evidence() -> None:
    from omni_mercury_engine.engine import OmniMercuryEngine

    verdict = OmniMercuryEngine._batch_is_anomalous
    weak = {"is_anomaly": np.array([True, False]), "p_values": np.array([0.04, 0.6])}
    strong = {"is_anomaly": np.array([True, False]), "p_values": np.array([0.0001, 0.6])}
    # Per-row flags alone no longer make the batch anomalous: p = 0.04 is not
    # evidence at 0.05 / 2 over two rows.
    assert verdict({"a": weak}, set()) is False
    assert verdict({"a": strong}, set()) is True
    # Fitted on this very batch: no independent reference, the per-row OR stands.
    assert verdict({"a": weak}, {"a"}) is True
    # A legacy result without p-values keeps the per-row OR.
    assert verdict({"a": {"is_anomaly": np.array([True])}}, set()) is True
    assert verdict({"a": {"is_anomaly": np.array([False])}}, set()) is False
    # The budget is split across the detectors tested (union bound).
    split = {"a": {"is_anomaly": np.array([False]), "p_values": np.array([0.03])}}
    assert verdict(split, set()) is True
    split["b"] = {"is_anomaly": np.array([False]), "p_values": np.array([0.9])}
    assert verdict(split, set()) is False


def test_engine_batch_false_alarm_rate_on_normal_batches() -> None:
    """All five base detectors fitted on 1,000 normal rows; 40 all-normal batches.

    Measured: every 10- and 100-row batch was called anomalous before (per-row
    OR over batch-relative detectors); now 0.025 and 0.05 of them, while a
    batch holding one +8-sigma row is still called anomalous every time.
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(0)
    engine = OmniMercuryEngine()
    train = rng.standard_normal((1000, 8))
    for detector in engine.detectors.values():
        detector.fit(train)
    normal, spiked = [], []
    for _ in range(20):
        rows = rng.standard_normal((100, 8))
        normal.append(bool(engine.detect(rows)["is_anomaly"]))
        rows[0] += 8.0
        spiked.append(bool(engine.detect(rows)["is_anomaly"]))
    assert np.mean(normal) <= 0.15
    assert all(spiked)

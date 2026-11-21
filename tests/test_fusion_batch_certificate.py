# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A conformal certificate for the question a batch verdict actually answers.

``detect_with_fusion`` reports one verdict for a batch: its most anomalous
row.  A per-row certificate does not cover that statistic -- the maximum of
many rows is not exchangeable with one row -- and these tests measure by how
much on the bundled model.  ``calibrate_fusion_conformal_batches`` fits the
certificate on one batch-max statistic per calibration batch, names the
regime in every result and record, survives a checkpoint round trip, and
comes with a test of whether a served batch belongs to the family it was
calibrated on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_mercury_engine.engine import OmniMercuryEngine, bundled_fusion_checkpoint_path
from scripts.train_default_fusion import SEED, build_dataset

ROWS = 40
N_CALIBRATION, N_HELD_OUT = 80, 60


class _Batches:
    """Batches of the bundled model's own data: all ordinary rows, or with 1-3 anomalies."""

    def __init__(self) -> None:
        x, y = build_dataset(SEED)
        self.normal, self.anomalous = x[y == 0], x[y == 1]
        self.rng = np.random.default_rng(5)

    def one(self, label: int) -> np.ndarray[Any, Any]:
        rows: np.ndarray[Any, Any] = self.normal[
            self.rng.choice(len(self.normal), ROWS, replace=False)
        ].copy()
        if label:
            count = int(self.rng.integers(1, 4))
            rows[self.rng.choice(ROWS, count, replace=False)] = self.anomalous[
                self.rng.choice(len(self.anomalous), count, replace=False)
            ]
        return rows

    def many(self, count: int) -> tuple[list[np.ndarray[Any, Any]], list[int]]:
        labels = [index % 2 for index in range(count)]
        return [self.one(label) for label in labels], labels


def _bundled() -> OmniMercuryEngine:
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.load_model(str(bundled_fusion_checkpoint_path()))
    engine.enable_decision_layer()
    return engine


@pytest.fixture(scope="module")
def measured() -> dict[str, Any]:
    """Coverage of held-out batches under each certificate, on the bundled model."""
    batches = _Batches()
    calibration, calibration_labels = batches.many(N_CALIBRATION)
    held_out, held_out_labels = batches.many(N_HELD_OUT)
    engine = _bundled()

    def coverage() -> dict[int, float]:
        covered: dict[int, list[bool]] = {0: [], 1: []}
        for batch, label in zip(held_out, held_out_labels, strict=True):
            result = engine.detect_with_fusion(batch)
            covered[label].append(label in result["conformal"]["prediction_set"])
        return {label: float(np.mean(flags)) for label, flags in covered.items()}

    per_sample = coverage()
    report = engine.calibrate_fusion_conformal_batches(calibration, calibration_labels)
    return {
        "engine": engine,
        "per_sample": per_sample,
        "batch_max": coverage(),
        "report": report,
        "clean": held_out[0],
        "batches": batches,
    }


class TestTheCertificateCoversTheVerdict:
    def test_a_per_row_certificate_does_not_cover_a_batch_verdict(
        self, measured: dict[str, Any]
    ) -> None:
        """The prior state, measured: nearly every clean batch has one row past a row threshold."""
        assert measured["per_sample"][0] < 0.4

    def test_a_batch_level_certificate_does(self, measured: dict[str, Any]) -> None:
        """Fails on the per-row certificate above (measured there: 4 of 30 clean batches)."""
        assert measured["batch_max"][0] >= 0.7
        assert measured["batch_max"][1] >= 0.7
        assert measured["batch_max"][0] - measured["per_sample"][0] >= 0.4

    def test_the_committed_ledger_quotes_these_numbers(self, measured: dict[str, Any]) -> None:
        """The Improvement Ledger's figures are this measurement, to within two batches in 30."""
        from omni_mercury_engine.governance.improvement_ledger import published

        ledger = published()
        assert ledger is not None
        entry = next(
            decision
            for decision in ledger["decisions"]
            if decision["component"] == "engine.conformal_certificate.batch_level"
        )
        quoted = entry["impact"]["bundled_model_clean_batch_coverage"]
        assert quoted["batches_per_class"] == N_HELD_OUT // 2
        assert quoted["rows_per_batch"] == ROWS
        assert quoted["calibration_batches"] == N_CALIBRATION
        tolerance = 2 / (N_HELD_OUT // 2) + 1e-9
        assert abs(quoted["per_row_certificate"] - measured["per_sample"][0]) <= tolerance
        assert abs(quoted["batch_level_certificate"] - measured["batch_max"][0]) <= tolerance

    def test_it_is_fitted_on_one_statistic_per_batch(self, measured: dict[str, Any]) -> None:
        report, engine = measured["report"], measured["engine"]
        assert report["regime"] == "batch_max"
        assert report["n_calibration"] == N_CALIBRATION
        assert engine._fusion_conformal_n_calibration == N_CALIBRATION
        assert engine._fusion_conformal_regime == "batch_max"

    def test_the_result_and_the_record_name_the_regime(self, measured: dict[str, Any]) -> None:
        result = measured["engine"].detect_with_fusion(measured["clean"])
        assert result["conformal"]["regime"] == "batch_max"
        # Row-level set sizes are a per-row certificate's output; there is none here.
        assert "row_set_sizes" not in result["conformal"]
        signals = result["decision"]["signals"]
        assert signals["conformal_regime"] == "batch_max"
        assert signals["model_provenance"]["conformal"]["n_calibration"] == N_CALIBRATION
        assert any(
            "batch-level" in caveat and f"{ROWS} rows" in caveat
            for caveat in result["decision"]["caveats"]
        )

    def test_one_label_per_batch_is_required(self) -> None:
        engine = _bundled()
        with pytest.raises(ValueError, match="one label per batch"):
            engine.calibrate_fusion_conformal_batches([np.zeros((4, 16))], [0, 1])


class TestTheBatchFamily:
    """Is this batch one of the batches the certificate was calibrated on?"""

    def test_a_batch_of_the_family_is_not_called_drift(self, measured: dict[str, Any]) -> None:
        drift = measured["engine"].detect_with_fusion(measured["clean"])["drift_detection"]
        assert drift["status"] == "tested"
        assert drift["reference_source"] == "calibration_batches:batch_family"
        assert drift["reference_rows"] == N_CALIBRATION
        assert drift["is_drift"] is False and drift["severity"] == "NONE"
        assert drift["p_value"] > 0.05

    def test_the_false_alarm_rate_on_the_family_is_near_its_level(
        self, measured: dict[str, Any]
    ) -> None:
        engine, batches = measured["engine"], measured["batches"]
        fresh, _ = batches.many(40)
        alarms = [engine.detect_with_fusion(b)["drift_detection"]["is_drift"] for b in fresh]
        assert float(np.mean(alarms)) <= 0.2  # level 0.05; 40 draws

    def test_a_batch_unlike_the_family_defers_the_decision(self, measured: dict[str, Any]) -> None:
        heavy_tailed = np.random.default_rng(7).standard_cauchy(size=(ROWS, 16))
        result = measured["engine"].detect_with_fusion(heavy_tailed)
        drift = result["drift_detection"]
        assert drift["is_drift"] is True and drift["severity"] == "HIGH"
        assert "not one of the family" in drift["message"]
        decision = result["decision"]
        assert decision["abstained"] is True
        assert any("drift" in reason for reason in decision["reasons"])

    def test_the_summary_leaves_out_the_statistic_the_certificate_is_about(self) -> None:
        """A batch is not out of family for containing an anomaly."""
        batch = np.random.default_rng(1).normal(size=(ROWS, 16))
        quiet = np.full(ROWS, 0.1)
        loud = quiet.copy()
        loud[0] = 0.999
        assert np.array_equal(
            OmniMercuryEngine._batch_family_summary(batch, quiet),
            OmniMercuryEngine._batch_family_summary(batch, loud),
        )

    def test_too_few_batches_fit_no_family(self) -> None:
        engine = _bundled()
        batches, labels = _Batches().many(12)
        engine.calibrate_fusion_conformal_batches(batches, labels)
        assert engine._fusion_conformal_regime == "batch_max"
        assert engine._fusion_batch_family is None
        drift = engine.detect_with_fusion(batches[0])["drift_detection"]
        assert drift.get("reference_source") != "calibration_batches:batch_family"


class TestTheFamilyCanBeDescribed:
    """What a batch is as a whole, when its rows no longer say.

    Rows standardized within their own batch look alike whatever the batch was
    made of. A caller that knows more (the raw levels the standardization
    removed) gives it per calibration batch, and the family test ranks a
    served batch's descriptor among them as well.
    """

    @staticmethod
    def _described(seed: int = 11) -> tuple[OmniMercuryEngine, _Batches, np.random.Generator]:
        engine, batches, rng = _bundled(), _Batches(), np.random.default_rng(seed)
        calibration, labels = batches.many(N_CALIBRATION)
        engine.calibrate_fusion_conformal_batches(
            calibration,
            labels,
            descriptors=[rng.normal(size=4) for _ in calibration],
            descriptor_names=["a", "b", "c", "d"],
        )
        return engine, batches, rng

    def test_a_batch_described_like_the_family_is_ranked_on_both_arms(self) -> None:
        engine, batches, rng = self._described()
        alarms = []
        for _ in range(40):
            drift = engine.detect_with_fusion(batches.one(0), batch_descriptor=rng.normal(size=4))[
                "drift_detection"
            ]
            assert drift["status"] == "tested"
            smaller = min(drift["shape_p_value"], drift["descriptor_p_value"])
            assert drift["p_value"] == pytest.approx(min(1.0, 2.0 * smaller))
            alarms.append(drift["is_drift"])
        assert float(np.mean(alarms)) <= 0.2  # level 0.05 over the two arms; 40 draws

    def test_ordinary_rows_with_a_foreign_descriptor_defer_the_decision(self) -> None:
        engine, batches, _ = self._described()
        batch = batches.one(0)
        ordinary = engine.detect_with_fusion(batch, batch_descriptor=np.zeros(4))
        foreign = engine.detect_with_fusion(batch, batch_descriptor=np.full(4, 25.0))
        assert ordinary["drift_detection"]["is_drift"] is False
        drift = foreign["drift_detection"]
        # The rows are the same rows, so the summary arm sees nothing; the
        # descriptor alone says this batch is something else.
        assert drift["shape_p_value"] == ordinary["drift_detection"]["shape_p_value"]
        assert drift["descriptor_p_value"] <= 0.025
        assert drift["is_drift"] is True and drift["severity"] == "HIGH"
        assert "not one of the family" in drift["message"]
        assert foreign["decision"]["abstained"] is True
        assert any("drift" in reason for reason in foreign["decision"]["reasons"])

    @pytest.mark.parametrize(
        ("descriptor", "said"),
        [
            (None, "without any"),
            (np.zeros(3), "do not match"),
            (np.array([0.0, np.nan, 0.0, 0.0]), "do not match"),
        ],
    )
    def test_a_described_family_does_not_claim_a_batch_it_cannot_rank(
        self, descriptor: np.ndarray[Any, Any] | None, said: str
    ) -> None:
        engine, batches, _ = self._described()
        result = engine.detect_with_fusion(batches.one(0), batch_descriptor=descriptor)
        drift = result["drift_detection"]
        assert drift["status"] == "tested"
        assert drift["is_drift"] is True and drift["severity"] == "HIGH"
        assert drift["p_value"] is None
        assert said in drift["message"]
        assert result["decision"]["abstained"] is True

    def test_a_family_without_descriptors_is_tested_as_before(
        self, measured: dict[str, Any]
    ) -> None:
        engine = measured["engine"]
        assert "descriptors" not in engine._fusion_batch_family
        plain = engine.detect_with_fusion(measured["clean"])["drift_detection"]
        given = engine.detect_with_fusion(measured["clean"], batch_descriptor=np.full(4, 25.0))[
            "drift_detection"
        ]
        assert given == plain
        assert "descriptor_p_value" not in plain

    def test_descriptors_round_trip_with_the_checkpoint(self, tmp_path: Path) -> None:
        engine, batches, _ = self._described()
        path = str(tmp_path / "described.pt")
        engine.save_model(path)
        loaded = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded.load_model(path)
        family = loaded._fusion_batch_family
        assert family == engine._fusion_batch_family and family is not None
        assert family["descriptors"]["names"] == ["a", "b", "c", "d"]
        batch = batches.one(0)
        before = engine.detect_with_fusion(batch, batch_descriptor=np.full(4, 25.0))
        after = loaded.detect_with_fusion(batch, batch_descriptor=np.full(4, 25.0))
        assert after["drift_detection"]["p_value"] == before["drift_detection"]["p_value"]
        assert after["drift_detection"]["is_drift"] is True

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda n: [np.zeros(4)] * (n - 1), "one vector per batch"),
            (lambda n: [np.full(4, np.inf)] * n, "must be finite"),
        ],
    )
    def test_descriptors_must_be_one_finite_vector_per_batch(
        self, build: Any, message: str
    ) -> None:
        engine = _bundled()
        calibration, labels = _Batches().many(N_CALIBRATION)
        with pytest.raises(ValueError, match=message):
            engine.calibrate_fusion_conformal_batches(
                calibration, labels, descriptors=build(len(calibration))
            )

    def test_descriptor_names_must_match_the_descriptor(self) -> None:
        engine = _bundled()
        calibration, labels = _Batches().many(N_CALIBRATION)
        with pytest.raises(ValueError, match="descriptor names"):
            engine.calibrate_fusion_conformal_batches(
                calibration,
                labels,
                descriptors=[np.zeros(4) + i for i in range(len(calibration))],
                descriptor_names=["only", "three", "names"],
            )


class TestTheRegimeIsPartOfTheCheckpoint:
    def test_it_round_trips_with_the_family(self, measured: dict[str, Any], tmp_path: Path) -> None:
        path = str(tmp_path / "batch.pt")
        measured["engine"].save_model(path)
        loaded = OmniMercuryEngine(mode="fusion", device="cpu")
        loaded.load_model(path)
        assert loaded._fusion_conformal_regime == "batch_max"
        assert loaded._fusion_batch_family == measured["engine"]._fusion_batch_family
        loaded.enable_decision_layer()
        before = measured["engine"].detect_with_fusion(measured["clean"])
        after = loaded.detect_with_fusion(measured["clean"])
        assert after["conformal"] == before["conformal"]
        assert after["drift_detection"]["p_value"] == before["drift_detection"]["p_value"]

    def test_a_checkpoint_written_before_the_regime_existed_is_per_row(self) -> None:
        engine = _bundled()  # the bundled checkpoint predates the field
        assert engine._fusion_conformal_regime == "per_sample"
        assert engine._fusion_batch_family is None
        result = engine.detect_with_fusion(_Batches().one(0))
        assert result["conformal"]["regime"] == "per_sample"

    def test_a_per_row_refit_replaces_a_batch_certificate(self) -> None:
        engine = _bundled()
        source = _Batches()
        batches, labels = source.many(24)
        engine.calibrate_fusion_conformal_batches(batches, labels)
        assert engine._fusion_batch_family is not None
        x, y = build_dataset(SEED)
        engine.calibrate_fusion_conformal(x[:400], y[:400], coverage=0.9)
        assert engine._fusion_conformal_regime == "per_sample"
        assert engine._fusion_batch_family is None


class TestTheCertificateStatesItsScope:
    def test_a_checkpoints_own_scope_statement_rides_into_every_record(self) -> None:
        engine = _bundled()
        statement = "calibrated on batches of forty rows of the bundled mixture"
        engine._fusion_provenance = {
            **(engine._fusion_provenance or {}),
            "certificate_scope": statement,
        }
        result = engine.detect_with_fusion(_Batches().one(0))
        assert result["model_provenance"]["certificate_scope"] == statement
        assert f"certificate scope: {statement}" in result["decision"]["caveats"]

    def test_no_statement_no_caveat(self) -> None:
        result = _bundled().detect_with_fusion(_Batches().one(0))
        assert "certificate_scope" not in result["model_provenance"]
        assert not any(c.startswith("certificate scope:") for c in result["decision"]["caveats"])

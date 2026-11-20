# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A serialized DecisionRecord alone answers the provenance questions.

Regression: the record's signals kept only the skipped detectors' *names* --
the skip reasons, each detector's uncertainty, the out-of-support rows and the
identity of the calibrated model (checkpoint, temperature, conformal coverage)
that produced the probability all lived only in the transient result dict, so
a reloaded audit trail could not say why evidence was missing or which model
decided. These tests pin that a record round-tripped through JSON answers all
of it by itself, and that records saved before these fields reload unchanged.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.decision import DecisionAbstentionResponder, DecisionRecord

if TYPE_CHECKING:
    from pathlib import Path

    from omni_mercury_engine.engine import OmniMercuryEngine

pytestmark = pytest.mark.xdist_group("record_provenance")

SHA = "ab" * 32

FULL_RESULT: dict[str, Any] = {
    "anomaly_prob": 0.93,
    "is_anomaly": True,
    "threshold_used": 0.5,
    "severity": 0.4,
    "conformal": {"prediction_set": [1], "set_size": 1, "abstain": False, "coverage": 0.9},
    "skipped_detectors": {
        "spectral": {"reason": "timeout", "detail": "5s budget exhausted", "trained_group": True},
        "rules": {"reason": "declined", "detail": "input out of scope", "trained_group": False},
    },
    "detector_uncertainties": {"statistical": 0.12, "temporal": 0.48},
    "out_of_support_rows": [3, 7],
    "model_provenance": {
        "checkpoint": "default_fusion.pt",
        "checkpoint_sha256": SHA,
        "temperature": 1.7,
        "conformal": {"coverage_target": 0.9, "coverage_achieved": 0.88, "n_calibration": 120},
        "trained_groups": ["statistical", "spectral"],
    },
}


def _round_tripped_record() -> DecisionRecord:
    record = DecisionAbstentionResponder().decide(FULL_RESULT)
    wire = json.loads(json.dumps(record.to_dict()))
    return DecisionRecord.from_dict(wire)


class TestRecordAloneAnswersProvenance:
    """The reloaded record -- nothing else -- answers each audit question."""

    def test_which_detectors_were_skipped_and_why(self) -> None:
        signals = _round_tripped_record().signals
        detail = signals["skipped_detail"]
        assert detail["spectral"] == {
            "reason": "timeout",
            "detail": "5s budget exhausted",
            "trained_group": True,
        }
        assert detail["rules"]["reason"] == "declined"
        assert detail["rules"]["trained_group"] is False
        # The names-only view is preserved beside it (released key unchanged:
        # trained sources only, an unclear entry read as trained).
        assert signals["skipped_detectors"] == ["spectral"]

    def test_each_detectors_uncertainty(self) -> None:
        signals = _round_tripped_record().signals
        assert signals["detector_uncertainties"]["statistical"] == pytest.approx(0.12)
        assert signals["detector_uncertainties"]["temporal"] == pytest.approx(0.48)

    def test_which_rows_were_out_of_support(self) -> None:
        assert _round_tripped_record().signals["out_of_support_rows"] == [3, 7]

    def test_which_model_produced_the_probability(self) -> None:
        provenance = _round_tripped_record().signals["model_provenance"]
        assert provenance["checkpoint"] == "default_fusion.pt"
        assert provenance["checkpoint_sha256"] == SHA
        assert provenance["temperature"] == pytest.approx(1.7)
        assert provenance["conformal"] == {
            "coverage_target": 0.9,
            "coverage_achieved": 0.88,
            "n_calibration": 120,
        }
        assert provenance["trained_groups"] == ["statistical", "spectral"]


class TestOldRecordsStillLoad:
    """Records saved before the provenance fields reload without them."""

    def test_from_dict_tolerates_a_record_without_the_new_keys(self) -> None:
        record = DecisionAbstentionResponder().decide(FULL_RESULT)
        wire = record.to_dict()
        for key in (
            "skipped_detail",
            "detector_uncertainties",
            "out_of_support_rows",
            "model_provenance",
            "contributions",
        ):
            wire["signals"].pop(key, None)
        reloaded = DecisionRecord.from_dict(wire)
        assert reloaded.state is record.state
        assert "skipped_detail" not in reloaded.signals
        # And the stripped record still round-trips exactly.
        assert DecisionRecord.from_dict(reloaded.to_dict()).to_dict() == reloaded.to_dict()

    def test_sparse_result_defaults_are_empty_or_none(self) -> None:
        record = DecisionAbstentionResponder().decide(
            {"anomaly_prob": 0.1, "is_anomaly": False, "threshold_used": 0.5}
        )
        signals = record.signals
        assert signals["skipped_detail"] == {}
        assert signals["detector_uncertainties"] == {}
        assert signals["out_of_support_rows"] == []
        assert signals["model_provenance"] is None


WIDTH = 8


def _data(seed: int = 7) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(200, WIDTH)), rng.normal(size=(40, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(200), np.ones(40)])
    return x, y


@pytest.fixture(scope="module")
def fitted_engine() -> OmniMercuryEngine:
    from omni_mercury_engine.engine import OmniMercuryEngine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion")
        x, y = _data()
        engine.fit_fusion(x, y, epochs=2)
    return engine


class TestEngineEmitsModelProvenance:
    """detect_with_fusion carries an additive, self-describing model identity."""

    def test_in_memory_fit_is_named_and_unhashed(self, fitted_engine: OmniMercuryEngine) -> None:
        probe = np.random.default_rng(1).normal(size=(4, WIDTH))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fitted_engine.detect_with_fusion(probe)
        provenance = out["model_provenance"]
        assert provenance["checkpoint"] == "in-memory fit"
        assert provenance["checkpoint_sha256"] is None
        assert isinstance(provenance["temperature"], float)
        conformal = provenance["conformal"]
        assert conformal is not None
        assert conformal["coverage_target"] == pytest.approx(0.9)
        assert isinstance(conformal["n_calibration"], int)
        assert provenance["trained_groups"]

    def test_loaded_checkpoint_is_named_and_hashed_once(
        self, fitted_engine: OmniMercuryEngine, tmp_path: Path
    ) -> None:
        path = tmp_path / "provenance_roundtrip.mdl"
        fitted_engine.save_model(str(path))
        expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

        from omni_mercury_engine.engine import OmniMercuryEngine

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = OmniMercuryEngine(mode="fusion")
            loaded.load_model(str(path))
            out = loaded.detect_with_fusion(np.random.default_rng(2).normal(size=(4, WIDTH)))
        provenance = out["model_provenance"]
        assert provenance["checkpoint"] == "provenance_roundtrip.mdl"
        assert provenance["checkpoint_sha256"] == expected_sha
        # The checkpoint does not record its calibration-set size; the
        # restored certificate says so instead of inventing one.
        assert provenance["conformal"] is not None
        assert provenance["conformal"]["n_calibration"] is None

    def test_decision_signals_carry_the_provenance(self, fitted_engine: OmniMercuryEngine) -> None:
        probe = np.random.default_rng(3).normal(size=(4, WIDTH))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = fitted_engine.detect_with_fusion(probe)
        record = DecisionAbstentionResponder().decide(out)
        assert record.signals["model_provenance"] == out["model_provenance"]

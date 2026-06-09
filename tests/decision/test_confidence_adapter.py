"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""Normalising engine results into a source-agnostic confidence signal.

Pure-Python tier (no torch): the adapters read plain result dicts shaped like
``OmniMercuryEngine.detect_with_fusion`` / ``score_fusion_conformal`` output and
are forward-compatible with PR #278's richer calibration keys.
"""

import numpy as np
import pytest

from omni_mercury_engine.decision.confidence import (
    ConfidenceSignal,
    ConfidenceSource,
    confidence_batch_from_conformal_scores,
    confidence_from_conformal,
    confidence_from_engine_result,
)


class TestSignalValidation:
    """The signal is the verifiable carrier: garbage in must error, not slip through."""

    def test_probability_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceSignal(anomaly_probability=1.5)
        with pytest.raises(ValueError):
            ConfidenceSignal(anomaly_probability=-0.2)

    def test_float_fuzz_is_snapped_in(self) -> None:
        assert ConfidenceSignal(anomaly_probability=1.0 + 1e-12).anomaly_probability == 1.0
        assert ConfidenceSignal(anomaly_probability=-1e-12).anomaly_probability == 0.0

    def test_bad_prediction_set_labels_raise(self) -> None:
        with pytest.raises(ValueError):
            ConfidenceSignal(anomaly_probability=0.5, prediction_set=(2,))
        with pytest.raises(ValueError):
            ConfidenceSignal(anomaly_probability=0.5, prediction_set=(1, 1))

    def test_prediction_set_is_sorted(self) -> None:
        signal = ConfidenceSignal(anomaly_probability=0.5, prediction_set=(1, 0))
        assert signal.prediction_set == (0, 1)

    def test_novelty_and_has_conformal_properties(self) -> None:
        assert ConfidenceSignal(0.5, prediction_set=()).is_novel
        assert ConfidenceSignal(0.5, prediction_set=()).has_conformal
        assert not ConfidenceSignal(0.5).has_conformal
        assert not ConfidenceSignal(0.5).is_novel


class TestEngineResultAdapter:
    """Reading the shapes the engine actually emits."""

    def test_anomaly_prob_only(self) -> None:
        signal = confidence_from_engine_result({"anomaly_prob": 0.73})
        assert signal.anomaly_probability == pytest.approx(0.73)
        assert signal.source is ConfidenceSource.CALIBRATED_PROBABILITY
        assert signal.prediction_set is None

    def test_reads_conformal_subdict(self) -> None:
        result = {
            "anomaly_prob": 0.81,
            "conformal": {
                "prediction_set": [1],
                "set_size": 1,
                "abstain": False,
                "coverage": 0.9,
            },
        }
        signal = confidence_from_engine_result(result)
        assert signal.prediction_set == (1,)
        assert signal.coverage == pytest.approx(0.9)
        assert signal.source is ConfidenceSource.CONFORMAL

    def test_prefers_calibrated_probabilities_when_present(self) -> None:
        # Forward-compatibility with PR #278's additive Beta-MCA key.
        result = {"anomaly_prob": 0.40, "calibrated_probabilities": [0.62]}
        signal = confidence_from_engine_result(result)
        assert signal.anomaly_probability == pytest.approx(0.62)
        assert signal.provenance["probability_source"] == "calibrated_probabilities"

    def test_prefers_reconciled_operating_point_when_present(self) -> None:
        result = {"anomaly_prob": 0.40, "reconciled_operating_point": {"probability": 0.7}}
        signal = confidence_from_engine_result(result)
        assert signal.anomaly_probability == pytest.approx(0.7)
        assert signal.source is ConfidenceSource.RECONCILED

    def test_missing_probability_raises(self) -> None:
        with pytest.raises(KeyError):
            confidence_from_engine_result({"is_anomaly": True})

    def test_handles_numpy_scalar_probability(self) -> None:
        signal = confidence_from_engine_result({"anomaly_prob": np.float32(0.66)})
        assert signal.anomaly_probability == pytest.approx(0.66, abs=1e-6)


class TestBatchAndDirectBuilders:
    def test_batch_from_conformal_scores(self) -> None:
        out = {
            "probabilities": np.array([0.9, 0.5, 0.1]),
            "prediction_sets": [[1], [0, 1], [0]],
            "set_sizes": np.array([1, 2, 1]),
            "abstain": np.array([False, True, False]),
            "coverage": 0.9,
        }
        signals = confidence_batch_from_conformal_scores(out)
        assert [s.prediction_set for s in signals] == [(1,), (0, 1), (0,)]
        assert all(s.source is ConfidenceSource.CONFORMAL for s in signals)
        assert signals[1].provenance["batch_index"] == 1

    def test_confidence_from_conformal_builder(self) -> None:
        signal = confidence_from_conformal(0.8, [1], 0.95)
        assert signal.source is ConfidenceSource.CONFORMAL
        assert signal.prediction_set == (1,)
        assert signal.coverage == pytest.approx(0.95)

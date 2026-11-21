# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The conformal certificate says which regime its guarantee was issued under.

The shipped certificate is fitted per sample; the flagship's scalar verdict is
the batch-max row.  Selecting the max is not exchangeable with the per-sample
fit, so a multi-row batch must not read as a batch-level coverage promise.
These tests pin the surfacing: the result's ``conformal`` block carries
``regime``/``n_rows`` (and per-row set-size counts on a batch), the evidence
carries them into the record, and the decider states the caveat on a
multi-row batch.

Also pinned here: the calibration-set size now survives a checkpoint
round-trip (``n_calibration`` rides ``export_state``), where a restored
certificate used to report ``None`` unconditionally.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.evidence import Evidence


def _fitted_classifier() -> BinaryConformalClassifier:
    rng = np.random.default_rng(7)
    probs = np.concatenate([rng.uniform(0.0, 0.4, 80), rng.uniform(0.6, 1.0, 40)])
    labels = np.concatenate([np.zeros(80, dtype=int), np.ones(40, dtype=int)])
    return BinaryConformalClassifier(coverage=0.9).fit(probs, labels)


class TestNCalibrationPersistence:
    def test_fit_records_the_calibration_size(self) -> None:
        clf = _fitted_classifier()
        assert clf.n_calibration == 120

    def test_export_state_round_trips_the_size(self) -> None:
        clf = _fitted_classifier()
        restored = BinaryConformalClassifier.from_state(clf.export_state())
        assert restored.n_calibration == 120
        assert restored.coverage_achieved == clf.coverage_achieved

    def test_legacy_state_restores_none_not_a_fabricated_count(self) -> None:
        clf = _fitted_classifier()
        state = clf.export_state()
        state.pop("n_calibration")
        restored = BinaryConformalClassifier.from_state(state)
        assert restored.n_calibration is None


class TestRegimeSurfacing:
    def _result(self, *, rows: int) -> dict[str, object]:
        conformal: dict[str, object] = {
            "prediction_set": [1],
            "set_size": 1,
            "abstain": False,
            "coverage": 0.9,
            "regime": "per_sample",
            "n_rows": rows,
        }
        return {
            "anomaly_prob": 0.95,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.5,
            "conformal": conformal,
        }

    def test_evidence_carries_regime_and_rows(self) -> None:
        ev = Evidence.from_detection(self._result(rows=5))
        assert ev.conformal_regime == "per_sample"
        assert ev.conformal_rows == 5
        signals = ev.to_dict()
        assert signals["conformal_regime"] == "per_sample"
        assert signals["conformal_rows"] == 5

    def test_multi_row_batch_gets_the_regime_caveat(self) -> None:
        record = DecisionAbstentionResponder().decide(self._result(rows=5))
        assert any("not a batch-level guarantee" in caveat for caveat in record.caveats)

    def test_single_row_gets_no_regime_caveat(self) -> None:
        record = DecisionAbstentionResponder().decide(self._result(rows=1))
        assert not any("batch-level" in caveat for caveat in record.caveats)

    def test_absent_regime_keys_stay_none(self) -> None:
        result = self._result(rows=1)
        carried = result["conformal"]
        assert isinstance(carried, dict)
        conformal = dict(carried)
        conformal.pop("regime")
        conformal.pop("n_rows")
        result["conformal"] = conformal
        ev = Evidence.from_detection(result)
        assert ev.conformal_regime is None
        assert ev.conformal_rows is None

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The engine's contract for a detector that declines an input modality.

``InputModalityError`` documents a *correct refusal*: "the engine skips the
component and the metrics layer counts it as declined rather than as an
inference error". ``detect_with_fusion``'s feature extractor honoured that,
but three paths did not, and together they walled univariate data out of the
engine:

* ``SpatialAnomalyDetector.fit`` indexed ``shape[1]`` before checking
  ``ndim``, so a 1-D series raised ``IndexError: tuple index out of range``
  instead of the detector's own refusal.
* ``fit_fusion``'s base-detector loop caught that refusal as a generic
  ``Exception`` and logged a warning, leaving the detector unfitted and
  indistinguishable from one the caller never trained. The fail-loud unfit
  check at inference then raised ``RuntimeError`` telling the caller to
  "Call fit_fusion(X_train, y_train)" -- the call that had just run.
* ``detect`` re-raised a decline instead of skipping it, so one inapplicable
  detector failed the whole call.

The spatial detector's refusal is correct on its own terms: it scores points
in a metric space, and a single column carries no such space. Univariate
input is a first-class case for the other detectors, so the cure is to carry
the refusal through the engine as a decline.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

import numpy as np

from omni_mercury_engine.core.exceptions import InputModalityError
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.engine import OmniMercuryEngine

#: Index range of the injected level shift in :func:`shifted_series`.
INJECTED = slice(300, 320)
#: Minimum separation the univariate fusion path must reach on that shift.
#: Measured 1.0000 at this seed; the floor leaves room for seed drift while
#: still failing if the path degrades to chance.
MIN_FUSION_AUROC = 0.95


def shifted_series() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """A 500-sample standard-normal series with a +6-sigma block injected."""
    series = np.concatenate(
        [
            np.random.default_rng(0).normal(size=300),
            np.random.default_rng(1).normal(loc=6.0, size=20),
            np.random.default_rng(2).normal(size=180),
        ]
    )
    labels = np.zeros(500, dtype=int)
    labels[INJECTED] = 1
    return series, labels


def auroc(labels: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> float:
    """Rank-based AUROC with midranks for ties."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, scores.size + 1, dtype=float)
    start = 0
    while start < scores.size:
        stop = start
        while stop + 1 < scores.size and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[start : stop + 1] = ranks[start : stop + 1].mean()
        start = stop + 1
    assigned = np.empty_like(scores)
    assigned[order] = ranks
    n_pos = int(np.count_nonzero(labels))
    n_neg = labels.size - n_pos
    return float((assigned[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class TestSpatialDeclinesUnivariate:
    """The refusal itself, at both entry points."""

    @pytest.mark.parametrize("shape", [(500,), (500, 1)])
    def test_fit_declines_with_a_typed_refusal(self, shape: tuple[int, ...]) -> None:
        """A bare 1-D array reached ``shape[1]`` and raised ``IndexError``."""
        with pytest.raises(InputModalityError):
            SpatialAnomalyDetector().fit(np.random.default_rng(0).normal(size=shape))

    def test_a_declining_detector_never_reports_itself_fitted(self) -> None:
        """So the engine cannot mistake a decline for a completed fit."""
        detector = SpatialAnomalyDetector()
        with pytest.raises(InputModalityError):
            detector.fit(np.random.default_rng(0).normal(size=(500, 1)))
        assert not detector.is_fitted()


class TestDetectSkipsDeclinedDetectors:
    """``detect`` honours the documented graceful-skip contract."""

    def test_univariate_detect_runs_the_applicable_detectors(self) -> None:
        """Previously ``IndexError: tuple index out of range``."""
        series, _ = shifted_series()

        result = OmniMercuryEngine().detect(series)

        assert result["declined_detectors"] == ["spatial"]
        assert set(result["detectors"]) == {
            "statistical",
            "temporal",
            "dimensional",
            "directive",
        }

    def test_the_injected_shift_is_still_reported(self) -> None:
        """Skipping a declined detector must not cost the verdict."""
        series, _ = shifted_series()

        assert OmniMercuryEngine().detect(series)["is_anomaly"] is True

    def test_an_all_declined_call_fails_instead_of_returning_normal(self) -> None:
        """ "Nothing could score this" is not "nothing is wrong"."""
        series, _ = shifted_series()

        with pytest.raises(InputModalityError, match="all declined"):
            OmniMercuryEngine().detect(series, detector_types=["spatial"])

    def test_an_inference_decline_does_not_silence_the_leakage_guard(self) -> None:
        """``detect`` must not mark a detector as having declined training.

        ``_modality_declined_detectors`` means "declined the training
        corpus", and it is what lets the fusion path skip its fail-loud unfit
        check. If an inference-time decline wrote to it, a later
        ``detect_with_fusion`` would quietly skip a detector the caller never
        trained instead of refusing to auto-fit on the scored batch.
        """
        engine = OmniMercuryEngine()

        engine.detect(shifted_series()[0])

        assert engine._modality_declined_detectors == set()
        with pytest.raises(RuntimeError, match="refusing to auto-fit"):
            engine.detect_with_fusion(np.random.default_rng(0).normal(size=(400, 3)))

    def test_a_genuine_fit_failure_is_not_laundered_into_a_decline(self) -> None:
        """The skip is keyed on ``InputModalityError`` and nothing wider.

        A detector that fails for any other reason must still fail the call,
        so this path cannot be used to swallow a real defect.
        """

        class BrokenFit(SpatialAnomalyDetector):
            def fit(self, data: Any) -> Any:
                raise RuntimeError("genuinely broken")

        engine = OmniMercuryEngine()
        engine.detectors["spatial"] = BrokenFit()

        with pytest.raises(RuntimeError, match="genuinely broken"):
            engine.detect(shifted_series()[0])


class TestUnivariateFusionPath:
    """The supervised path, end to end, on a single column."""

    def test_fit_fusion_completes_with_the_declining_detector_recorded(self) -> None:
        """Previously ``RuntimeError`` naming the call that had just run."""
        series, labels = shifted_series()
        engine = OmniMercuryEngine()

        engine.fit_fusion(series.reshape(-1, 1), labels, epochs=5)

        assert engine._modality_declined_detectors == {"spatial"}

    def test_detect_with_fusion_separates_the_injected_shift(self) -> None:
        """The cure has to buy real separation, not merely stop raising."""
        series, labels = shifted_series()
        column = series.reshape(-1, 1)
        engine = OmniMercuryEngine()
        engine.fit_fusion(column, labels, epochs=5)

        probabilities = np.asarray(
            engine.detect_with_fusion(column)["anomaly_probs"], dtype=float
        ).ravel()

        assert probabilities.size == labels.size
        assert auroc(labels, probabilities) >= MIN_FUSION_AUROC

    def test_fit_fusion_refuses_a_one_dimensional_array_at_the_boundary(self) -> None:
        """The documented contract is ``(n_samples, n_features)``.

        A 1-D array used to be accepted and then failed inside feature
        extraction, because the base detectors disagree on whether it means
        one column or one sample.
        """
        series, labels = shifted_series()

        with pytest.raises(ValueError, match=r"reshape\(-1, 1\)"):
            OmniMercuryEngine().fit_fusion(series, labels, epochs=1)

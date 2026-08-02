# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar: evidence — a verdict is only as strong as what backs it.

Three properties:

* **Low confidence defers.** When the calibrated certificate says both labels
  are admissible at the target coverage — or there is no certificate and the
  probability sits in the indecision band — the loop abstains instead of
  guessing. And the conformal certificate's empirical coverage meets its target.
* **Every verdict carries its drivers.** No decision record is emitted with an
  empty ``reasons`` trail, and an abstention names the signal that caused it.
  A verdict you cannot trace to evidence is an assertion, not a finding.
* **Nothing is generated.** Mercury ships no generative language-model weights.

That last claim is **deliberately scoped**. The broader "Mercury never produces
fabricated prose" is *false* as stated: ``reasoning/backend.py`` and
``models/foundation/{llm_adapter,ollama_adapter}.py`` exist precisely so an
operator can attach an LLM, and a template backend renders prose from Mercury's
own numbers. What is true, checkable, and worth claiming is narrower: **no
generative weights ship in this repository**, so out of the box every string
Mercury emits is derived from its own measurements, and any generative model is
an operator-supplied, provenance-stamped, gated dependency. That is what this
module asserts; ``CAPABILITY_MATRIX.md`` carries the same wording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.policy import DecisionPolicy
from omni_mercury_engine.decision.states import Disposition
from omni_mercury_engine.verifiers.three_state import ThreeState

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "omni_mercury_engine"


def _detection(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "anomaly_prob": 0.82,
        "is_anomaly": True,
        "threshold_used": 0.5,
        "severity": 0.6,
        "gosnn_metadata": {"ethical_gate_passed": True},
    }
    base.update(overrides)
    return base


class TestLowConfidenceDefers:
    def test_calibrated_ambiguity_defers(self) -> None:
        record = DecisionAbstentionResponder().decide(
            _detection(conformal={"set_size": 2, "prediction_set": [0, 1], "coverage": 0.9})
        )
        assert record.state is ThreeState.UNAVAILABLE
        assert record.disposition is Disposition.DEFER
        assert record.decision_label is None
        assert record.decision_confidence is None

    @pytest.mark.parametrize("prob", [0.46, 0.48, 0.50, 0.52, 0.54])
    def test_uncalibrated_probability_in_the_band_defers(self, prob: float) -> None:
        record = DecisionAbstentionResponder().decide(
            _detection(anomaly_prob=prob, threshold_used=0.5, conformal=None)
        )
        assert record.disposition is Disposition.DEFER
        assert record.decision_label is None

    def test_an_atypical_point_holds_rather_than_defers(self) -> None:
        """An empty conformal set means no class explains it — fail closed."""
        record = DecisionAbstentionResponder().decide(
            _detection(conformal={"set_size": 0, "prediction_set": [], "coverage": 0.9})
        )
        assert record.state is ThreeState.UNDECIDABLE
        assert record.disposition is Disposition.HOLD
        assert record.response.fail_closed is True

    def test_a_grounded_call_reports_the_coverage_it_rests_on(self) -> None:
        record = DecisionAbstentionResponder().decide(
            _detection(conformal={"set_size": 1, "prediction_set": [1], "coverage": 0.9})
        )
        assert record.state is ThreeState.GROUNDED
        assert record.calibrated is True
        assert record.coverage == pytest.approx(0.9)
        # The transparent confidence *is* the coverage level, not a made-up number.
        assert record.decision_confidence == pytest.approx(0.9)

    def test_an_uncalibrated_call_says_so_instead_of_claiming_a_guarantee(self) -> None:
        record = DecisionAbstentionResponder().decide(_detection(anomaly_prob=0.95, conformal=None))
        assert record.calibrated is False
        assert any("no conformal coverage certificate" in c for c in record.caveats)

    def test_requiring_calibration_to_act_demotes_an_uncertified_positive(self) -> None:
        strict = DecisionPolicy(require_calibrated_for_act=True)
        record = DecisionAbstentionResponder(strict).decide(
            _detection(anomaly_prob=0.99, conformal=None)
        )
        assert record.disposition is Disposition.DEFER


class TestConformalCoverageMeetsItsTarget:
    """The certificate the gate trusts must actually hold empirically.

    Stated precisely, because the precise version is the one that is true:
    split-conformal coverage is guaranteed **marginally** — in expectation over
    the joint draw of calibration and test sets — not conditionally on one
    calibration set. A single calibration draw can and does land a low quantile
    (seed 7 at a 0.80 target yields 0.775 on a held-out sample), so a test that
    asserted per-draw coverage would be asserting something conformal
    prediction does not promise, and would flake. This averages over
    independent calibration draws, which is the guarantee.
    """

    REPLICATIONS = 80

    @pytest.mark.parametrize("target", [0.80, 0.90, 0.95])
    def test_marginal_coverage_meets_target(self, target: float) -> None:
        from omni_mercury_engine.core.conformal_prediction import SplitConformalPredictor

        rng = np.random.default_rng(20260802)
        n_cal, n_test = 1000, 2000
        coverages = []
        for _ in range(self.REPLICATIONS):
            # Exchangeable calibration/test draws from one distribution -- the
            # assumption the distribution-free guarantee is stated under.
            cal_scores = rng.gamma(shape=2.0, scale=1.0, size=n_cal)
            test_scores = rng.gamma(shape=2.0, scale=1.0, size=n_test)
            predictor = SplitConformalPredictor(coverage=target, seed=0)
            predictor.fit(cal_scores)
            threshold = predictor.get_anomaly_threshold()
            coverages.append(float(np.mean(test_scores <= threshold)))

        marginal = float(np.mean(coverages))
        # Slack is the Monte-Carlo error of the mean over REPLICATIONS draws,
        # not a fudge factor hiding a miss: the per-draw spread is ~sqrt(
        # p(1-p)/n_cal) and averaging shrinks it by sqrt(REPLICATIONS).
        assert marginal >= target - 0.01, f"target={target} marginal={marginal:.4f}"

    def test_the_calibration_quantile_is_finite_sample_corrected(self) -> None:
        """On its own calibration set the quantile is exact, by construction.

        This is the deterministic half of the guarantee, and it is what makes
        the marginal result above hold: the ``ceil((n+1)(1-alpha))/n`` index is
        the finite-sample correction, not a plain empirical quantile.
        """
        from omni_mercury_engine.core.conformal_prediction import SplitConformalPredictor

        n_cal = 2000
        scores = np.random.default_rng(3).gamma(shape=2.0, scale=1.0, size=n_cal)
        for target in (0.80, 0.90, 0.95):
            predictor = SplitConformalPredictor(coverage=target, seed=0)
            predictor.fit(scores)
            threshold = predictor.get_anomaly_threshold()
            on_calibration = float(np.mean(scores <= threshold))
            expected = np.ceil((n_cal + 1) * target) / n_cal
            assert on_calibration == pytest.approx(expected, abs=1e-9), target
            assert on_calibration >= target


class TestEveryVerdictCarriesItsDrivers:
    CASES: tuple[dict[str, Any], ...] = (
        _detection(conformal={"set_size": 1, "prediction_set": [1], "coverage": 0.9}),
        _detection(conformal={"set_size": 0, "prediction_set": [], "coverage": 0.9}),
        _detection(conformal={"set_size": 2, "prediction_set": [0, 1], "coverage": 0.9}),
        _detection(anomaly_prob=0.51, conformal=None),
        _detection(anomaly_prob=0.99, conformal=None),
        _detection(gosnn_metadata={"ethical_gate_passed": False}),
        _detection(symbolic_consistency={"satisfaction": 0.01}),
        _detection(drift_detection={"is_drift": True, "severity": "CRITICAL"}),
    )

    @pytest.mark.parametrize("detection", CASES)
    def test_reasons_are_never_empty(self, detection: dict[str, Any]) -> None:
        record = DecisionAbstentionResponder().decide(detection, domain="cyber")
        assert record.reasons, "a verdict with no stated driver is an assertion, not a finding"
        assert all(reason.strip() for reason in record.reasons)

    @pytest.mark.parametrize("detection", CASES)
    def test_the_normalised_evidence_travels_with_the_verdict(
        self, detection: dict[str, Any]
    ) -> None:
        record = DecisionAbstentionResponder().decide(detection, domain="cyber")
        for key in ("anomaly_prob", "threshold", "severity", "policy"):
            assert key in record.signals, key

    @pytest.mark.parametrize("detection", CASES)
    def test_explain_names_the_state_and_the_response(self, detection: dict[str, Any]) -> None:
        record = DecisionAbstentionResponder().decide(detection, domain="cyber")
        explanation = record.explain()
        assert record.state.value.upper().replace("_", "") in explanation.upper().replace("_", "")
        assert record.response.action.value in explanation

    def test_an_ethical_block_names_the_gate_as_the_driver(self) -> None:
        record = DecisionAbstentionResponder().decide(
            _detection(gosnn_metadata={"ethical_gate_passed": False})
        )
        assert any("ethical gate" in reason for reason in record.reasons)

    def test_a_missing_ethical_verdict_is_declared_not_assumed(self) -> None:
        """Absent evidence is reported as absent, never read as a pass."""
        record = DecisionAbstentionResponder().decide(
            {"anomaly_prob": 0.82, "is_anomaly": True, "threshold_used": 0.5}
        )
        assert any("ethical-gate verdict is absent" in caveat for caveat in record.caveats)


class TestNoGenerativeWeightsShip:
    """The scoped, checkable form of "Mercury does not fabricate prose"."""

    #: Extensions a language-model checkpoint would arrive in.
    WEIGHT_SUFFIXES: tuple[str, ...] = (
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".gguf",
        ".ggml",
        ".onnx",
        ".h5",
    )

    #: Substrings that would mark a shipped checkpoint as a generative LM.
    GENERATIVE_MARKERS: tuple[str, ...] = (
        "gpt",
        "llama",
        "mistral",
        "phi",
        "qwen",
        "gemma",
        "falcon",
        "t5",
        "bert",
        "bloom",
        "opt-",
        "lm_head",
        "tokenizer",
    )

    def _shipped_weights(self) -> list[Path]:
        return sorted(
            path
            for path in SRC_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in self.WEIGHT_SUFFIXES
        )

    def test_the_scan_finds_the_checkpoints_that_do_ship(self) -> None:
        """Guard the guard: an empty scan would make the claim vacuous."""
        assert len(self._shipped_weights()) >= 5

    def test_no_shipped_checkpoint_is_a_generative_language_model(self) -> None:
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in self._shipped_weights()
            if any(marker in path.name.lower() for marker in self.GENERATIVE_MARKERS)
        ]
        assert not offenders, offenders

    def test_llm_backends_are_operator_supplied_not_bundled(self) -> None:
        """The adapters exist; the models they talk to are not in this repo."""
        adapters = SRC_ROOT / "models" / "foundation"
        assert (adapters / "llm_adapter.py").is_file()
        assert (adapters / "ollama_adapter.py").is_file()
        # No vendored model directory rides along with them.
        vendored = [p.name for p in adapters.iterdir() if p.is_dir() and p.name != "__pycache__"]
        assert not vendored, vendored

    def test_generated_explanations_are_provenance_stamped(self) -> None:
        """When prose *is* generated, the output names the backend that made it."""
        from omni_mercury_engine.reasoning.schemas import Explanation

        explanation = Explanation(text="t", backend="mock", model="mock")
        payload = explanation.to_dict()
        assert payload["backend"] == "mock"
        assert payload["model"] == "mock"
        assert "gated" in payload

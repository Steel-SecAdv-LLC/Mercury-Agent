# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Semantic pinning tests for the σ_Immutable gate hot path.

Written to kill the mutation survivors reported by
``scripts/run_sigma_mutation_gate.py`` for
``security/sigma_immutable_gate.py``: the first full measurement showed
the σ-focused subset pinned interfaces but not the arithmetic, so
constant tweaks, operator swaps, and boolean flips in the projection
band math, the vector builders, and the gate's fail-closed defaults all
survived.  Every assertion below is a closed-form contract value with
the mutation class it kills noted inline.

Covers:
- ``project_benevolence_to_sigma_band``: exact values at the floor
  boundary (>= vs >), band endpoints, mid-band linearity, the damp
  branch, and the defensive clips on out-of-contract input
- ``_sigma_base_vector`` / ``build_sigma_immutable_vector``: exact 256-D
  layout (ethical band 27, active band end 180, zero tail), the 33-wide
  signal window, the ``1.0 + 0.4·clip(0.5·s + 0.5·a)`` overlay, and the
  documented defaults-reproduce-base-vector byte-for-byte contract
- ``_is_pqc_backend_unavailable``: typed AMA exception, string probe,
  and negative case
- ``SigmaImmutableEvaluation`` frozenness; ``SigmaImmutableGate``
  threshold clamping and the verify-corpus-by-default contract
- End-to-end: benevolent vs hostile vectors through the real trained
  gate at the calibrated threshold
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from omni_mercury_engine._compat import HAS_TORCH
from omni_mercury_engine.cognitive.ethical_bounding import MINIMUM_BENEVOLENCE_FLOOR
from omni_mercury_engine.core.centralized_constants import ETHICAL
from omni_mercury_engine.security.sigma_immutable_gate import (
    SIGMA_ETHICAL_BAND_END,
    SIGMA_IMMUTABLE_DEFAULT_THRESHOLD,
    SIGMA_IMMUTABLE_DIM,
    SIGMA_SIGNAL_WINDOW,
    SIGMA_USED_BAND_END,
    SigmaImmutableEvaluation,
    SigmaImmutableGate,
    _is_pqc_backend_unavailable,
    _sigma_base_vector,
    build_sigma_immutable_vector,
    project_benevolence_to_sigma_band,
)

# =============================================================================
# Layout constants — cross-module contract (corpus, trainer, hub agree)
# =============================================================================


class TestLayoutContract:
    """The 256/27/180/33 layout is shared verbatim across modules."""

    def test_layout_constants(self) -> None:
        # Kills int tweaks on the layout constants: these values are the
        # contract the corpus, the trainer, and both boundary builders
        # import; drifting any of them desynchronises training and
        # serving layouts.
        assert SIGMA_IMMUTABLE_DIM == 256
        assert SIGMA_ETHICAL_BAND_END == 27
        assert SIGMA_USED_BAND_END == 180
        assert SIGMA_SIGNAL_WINDOW == 33
        assert pytest.approx(0.70) == MINIMUM_BENEVOLENCE_FLOOR

    def test_default_threshold_is_trained_calibration(self) -> None:
        # Kills the 0.93 tweak: the default must equal the trained
        # network's calibrated decision threshold, not the stricter
        # GOSNN gating default (0.96) — the two-threshold design is
        # documented in EthicalConstants.
        assert SIGMA_IMMUTABLE_DEFAULT_THRESHOLD == ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD
        assert pytest.approx(0.93) == SIGMA_IMMUTABLE_DEFAULT_THRESHOLD


# =============================================================================
# project_benevolence_to_sigma_band — closed form
# =============================================================================


class TestBenevolenceProjection:
    """Exact projection values; every branch and clip bound pinned."""

    def test_floor_boundary_projects_to_permissible_low(self) -> None:
        # Kills GtE->Gt on the floor comparison AND the 1.5 tweak: at
        # exactly the floor the score belongs to the permissible band's
        # lower edge; the Gt mutant would damp it to 0.35 instead.
        assert project_benevolence_to_sigma_band(0.70) == pytest.approx(1.5, abs=1e-12)

    def test_perfect_benevolence_projects_to_permissible_high(self) -> None:
        # Kills the 2.0 tweak and the scale numerator Sub->Add.
        assert project_benevolence_to_sigma_band(1.0) == pytest.approx(2.0, abs=1e-12)

    def test_mid_band_linearity(self) -> None:
        # Kills the scale arithmetic mutants (Sub->Add on the offset,
        # Mult->Add on the lift, _PERMISSIBLE_INPUT_RANGE tweaks):
        # 1.5 + (0.85 - 0.70) * (0.5 / 0.30) = 1.75 exactly.
        assert project_benevolence_to_sigma_band(0.85) == pytest.approx(1.75, abs=1e-12)

    def test_below_floor_damps_linearly(self) -> None:
        # Kills Mult->Add and the 0.5 tweak on the damp branch:
        # 0.6 * 0.5 = 0.30; the Add mutant would clip to 0.5.
        assert project_benevolence_to_sigma_band(0.6) == pytest.approx(0.30, abs=1e-12)

    def test_zero_benevolence_projects_to_zero(self) -> None:
        # Kills the damp-branch clip lower-bound 0.0 tweak.
        assert project_benevolence_to_sigma_band(0.0) == 0.0

    def test_out_of_contract_input_clipped_to_band_ceiling(self) -> None:
        # The clip exists to guard out-of-contract callers: 1.2 lifts to
        # 1.5 + 0.5*(0.5/0.3) ≈ 2.33 and must clamp to exactly 2.0.
        # Kills the permissible-branch clip upper-bound tweak.
        assert project_benevolence_to_sigma_band(1.2) == pytest.approx(2.0, abs=1e-12)

    def test_projection_is_monotonic_across_the_floor(self) -> None:
        # The permissible band sits strictly above the impermissible
        # band; a swapped comparison or flipped constant breaks this.
        grid = np.linspace(0.0, 1.0, 101)
        values = [project_benevolence_to_sigma_band(float(b)) for b in grid]
        below = [v for b, v in zip(grid, values) if b < MINIMUM_BENEVOLENCE_FLOOR]
        above = [v for b, v in zip(grid, values) if b >= MINIMUM_BENEVOLENCE_FLOOR]
        assert max(below) <= 0.5
        assert min(above) >= 1.5


# =============================================================================
# _sigma_base_vector / build_sigma_immutable_vector — exact layout
# =============================================================================


class TestSigmaVectorBuilders:
    """The 256-D layout and signal-window overlay, value-exact."""

    def test_base_vector_layout_exact(self) -> None:
        vector = _sigma_base_vector(0.85)
        assert vector.shape == (256,)
        assert vector.dtype == np.float64
        # Ethical band: the projected benevolence, all 27 dims.
        assert np.all(vector[:27] == pytest.approx(1.75, abs=1e-12))
        # Active band centred at the U[0, 2] training midpoint (kills
        # the 1.0 tweak) up to exactly index 180 (kills the 180 tweak).
        assert np.all(vector[27:180] == 1.0)
        # Reserved tail stays exactly zero (kills dim/band-end tweaks).
        assert np.all(vector[180:] == 0.0)

    def test_defaults_reproduce_base_vector_byte_for_byte(self) -> None:
        # Documented contract: severity == anomaly_prob == 0 reproduces
        # the benevolence-only boundary vector byte-for-byte.  Kills the
        # 0.0 tweaks on both keyword defaults.
        assert np.array_equal(build_sigma_immutable_vector(0.9), _sigma_base_vector(0.9))

    def test_signal_window_overlay_exact(self) -> None:
        # severity=1, anomaly=0: perturbation = clip(0.5, 0, 1) = 0.5,
        # window value = 1.0 + 0.4 * 0.5 = 1.2 exactly.  Kills the 0.5,
        # 1.0 and 0.4 tweaks and Mult->Add on the overlay.
        vector = build_sigma_immutable_vector(0.9, severity=1.0, anomaly_prob=0.0)
        # Window spans [27, 60): kills the 33 tweak and the window-end
        # Add->Sub (which would leave the window untouched at 1.0).
        assert np.all(vector[27:60] == pytest.approx(1.2, abs=1e-12))
        assert np.all(vector[60:180] == 1.0)

    def test_signal_window_combines_severity_and_anomaly(self) -> None:
        # severity=0.3, anomaly=0.5: perturbation = 0.15 + 0.25 = 0.4,
        # window = 1.16.  Kills Add->Sub between the two signal terms
        # (which yields a negative perturbation, clipped to 0 -> 1.0).
        vector = build_sigma_immutable_vector(0.9, severity=0.3, anomaly_prob=0.5)
        assert np.all(vector[27:60] == pytest.approx(1.16, abs=1e-12))

    def test_signal_window_clips_out_of_contract_signal(self) -> None:
        # The defensive clip caps the perturbation at 1.0 even for
        # out-of-contract signal inputs: window = 1.4 exactly.
        vector = build_sigma_immutable_vector(0.9, severity=2.0, anomaly_prob=2.0)
        assert np.all(vector[27:60] == pytest.approx(1.4, abs=1e-12))


# =============================================================================
# PQC-unavailability classifier
# =============================================================================


class TestPQCUnavailableClassifier:
    """Typed AMA exception and string-probe fallback."""

    def test_typed_ama_exception_is_classified(self) -> None:
        # Kills the bool flip on the isinstance branch: a real AMA
        # PQCUnavailableError must classify True even when its message
        # lacks the probe string.
        from ama_cryptography.exceptions import PQCUnavailableError

        assert _is_pqc_backend_unavailable(PQCUnavailableError("backend gone")) is True

    def test_string_probe_fallback(self) -> None:
        assert _is_pqc_backend_unavailable(RuntimeError("PQC_UNAVAILABLE: no lib")) is True

    def test_unrelated_exception_is_not_classified(self) -> None:
        assert _is_pqc_backend_unavailable(RuntimeError("disk full")) is False


# =============================================================================
# SigmaImmutableEvaluation / SigmaImmutableGate construction contracts
# =============================================================================


class TestEvaluationAndGateContracts:
    """Frozen results, threshold clamping, verify-corpus default."""

    def test_evaluation_is_frozen(self) -> None:
        # Kills the frozen=True flip: evaluation outcomes are immutable
        # verdicts; a mutable evaluation could be edited after the fact.
        evaluation = SigmaImmutableEvaluation(
            score=0.5, threshold=0.93, passes=False, backend="torch"
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evaluation.passes = True  # type: ignore[misc]

    def test_threshold_clamped_to_unit_interval(self) -> None:
        # Kills the 0.0/1.0 clip-bound tweaks on the constructor.
        assert SigmaImmutableGate(threshold=-0.5, verify_corpus=False).threshold == 0.0
        assert SigmaImmutableGate(threshold=1.5, verify_corpus=False).threshold == 1.0
        assert SigmaImmutableGate(threshold=0.4, verify_corpus=False).threshold == pytest.approx(
            0.4
        )

    def test_corpus_verification_on_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Kills the verify_corpus=True default flip: a default-constructed
        # gate MUST verify the signed corpus; skipping it silently would
        # let a tampered corpus serve without a recorded error.
        import omni_mercury_engine.security.sigma_immutable_corpus as corpus_module

        def _boom() -> None:
            raise corpus_module.CorpusVerificationError("tampered")

        monkeypatch.setattr(corpus_module, "verify_corpus_signatures", _boom)
        gate_default = SigmaImmutableGate()
        assert gate_default.corpus_error is not None
        assert "tampered" in gate_default.corpus_error
        # Explicit opt-out (test-only path) records no corpus error.
        assert SigmaImmutableGate(verify_corpus=False).corpus_error is None


# =============================================================================
# End-to-end: the trained gate separates benevolent from hostile vectors
# =============================================================================


@pytest.mark.skipif(
    not HAS_TORCH,
    reason=(
        "the trained σ_Immutable network requires torch (the [ml] extra); this "
        "class asserts backend=='torch', which is meaningless in a no-torch lane "
        "(e.g. pqc-production-check installs only [dev,api]). The fail-closed "
        "no-torch path is covered separately by TestFailClosedContracts / "
        "TestPQCUnavailableClassifier."
    ),
)
class TestTrainedGateEndToEnd:
    """The shipped trained network enforces the calibrated threshold."""

    @pytest.fixture()
    def gate(self) -> SigmaImmutableGate:
        return SigmaImmutableGate(verify_corpus=False)

    def test_benevolent_vector_passes(self, gate: SigmaImmutableGate) -> None:
        result = gate.evaluate(build_sigma_immutable_vector(1.0))
        assert result.backend == "torch", "trained network must be available in this environment"
        assert result.passes is True
        assert result.score >= result.threshold

    def test_hostile_vector_fails(self, gate: SigmaImmutableGate) -> None:
        result = gate.evaluate(build_sigma_immutable_vector(0.0))
        assert result.passes is False
        assert result.score < result.threshold

    def test_pass_verdict_is_threshold_comparison(self, gate: SigmaImmutableGate) -> None:
        # passes must be exactly (score >= threshold) for the torch
        # backend — pin the documented relation on both verdicts.
        for benevolence in (0.0, 0.5, 0.8, 1.0):
            result = gate.evaluate(build_sigma_immutable_vector(benevolence))
            assert result.passes == (result.score >= result.threshold and result.backend == "torch")


# =============================================================================
# Final-survivor kills: fail-closed reporting + boundary exactness
# =============================================================================


class _UnimportableEthicalGate:
    """Stub whose construction raises ImportError (torch-missing shape)."""

    def __init__(self, **_: object) -> None:
        raise ImportError("stub: EthicalGate unavailable")


class _UntrainedEthicalGate:
    """Stub EthicalGate carrying no ``_trained`` attribute at all."""

    def __init__(self, **_: object) -> None:
        pass


class TestFailClosedContracts:
    """Kill the final measurement's surviving mutants with contract pins."""

    def test_dual_gate_signal_defaults_are_zero(self) -> None:
        # Kills the 0.0 tweaks on enforce_dual_ethical_gate's keyword
        # defaults: severity/anomaly default to exactly zero so a plain
        # benevolence-only boundary reproduces the base vector.
        import inspect

        from omni_mercury_engine.security.sigma_immutable_gate import (
            enforce_dual_ethical_gate,
        )

        parameters = inspect.signature(enforce_dual_ethical_gate).parameters
        assert parameters["severity"].default == 0.0
        assert parameters["anomaly_prob"].default == 0.0

    def test_unavailable_gate_evaluation_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kills the score 0.0->0.1 tweak and the passes False->True flip
        # in evaluate()'s unavailable branch: with no trained network the
        # verdict must be a hard fail with the honest zero sentinel.
        import omni_mercury_engine.core.global_omni_scalar_network as gosnn_module

        monkeypatch.setattr(gosnn_module, "EthicalGate", _UnimportableEthicalGate)
        gate = SigmaImmutableGate(verify_corpus=False)
        result = gate.evaluate(build_sigma_immutable_vector(1.0))
        assert result.backend == "unavailable"
        assert result.passes is False
        assert result.score == 0.0
        assert result.threshold == gate.threshold

    def test_gate_object_without_trained_flag_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kills the getattr(gate, "_trained", False) default flip: an
        # EthicalGate exposing no _trained attribute at all is ambiguous
        # and must be refused as untrained, not accepted.
        import omni_mercury_engine.core.global_omni_scalar_network as gosnn_module

        monkeypatch.setattr(gosnn_module, "EthicalGate", _UntrainedEthicalGate)
        gate = SigmaImmutableGate(verify_corpus=False)
        assert gate.is_trained is False
        assert gate.gate_load_error is not None
        assert "untrained" in gate.gate_load_error

    def test_anchor_exactly_at_floor_is_compliant(self) -> None:
        # Kills Lt->LtE on the critical floor comparison: the trainer's
        # labelling rule draws positives from U[floor, 2.0], so an anchor
        # exactly AT the floor is compliant; just below is a violation.
        from omni_mercury_engine.security.sigma_immutable_gate import (
            CRITICAL_ETHICAL_FLOOR,
        )

        gate = SigmaImmutableGate(verify_corpus=False)
        assert gate.critical_ethical_floor_violations({"a": CRITICAL_ETHICAL_FLOOR}) == []
        below = CRITICAL_ETHICAL_FLOOR - 1e-9
        assert gate.critical_ethical_floor_violations({"a": below}) == [("a", below)]
        nan_violations = gate.critical_ethical_floor_violations({"a": float("nan")})
        assert len(nan_violations) == 1

    def test_corpus_error_refusal_reports_zero_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Kills the score 0.0->0.1 tweak in the corpus-error refusal: the
        # reported score of a refused action is the honest zero, never a
        # fabricated non-zero.
        import omni_mercury_engine.security.sigma_immutable_corpus as corpus_module
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        def _boom() -> None:
            raise corpus_module.CorpusVerificationError("tampered")

        monkeypatch.setattr(corpus_module, "verify_corpus_signatures", _boom)
        gate = SigmaImmutableGate()
        with pytest.raises(EthicalConstraintViolationError) as excinfo:
            gate.enforce(action="unit", scalar_vector=build_sigma_immutable_vector(1.0))
        assert excinfo.value.check == "sigma_immutable"
        assert excinfo.value.score == 0.0

    def test_unavailable_gate_refusal_reports_fallback_message_and_zero_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Kills the `or` -> `and` swap on the gate_load_error fallback and
        # the score 0.0->0.1 tweak in the gosnn_unavailable refusal.  The
        # None/None state cannot be reached through _init_gate (it always
        # records an error when the gate stays None), so the fallback is
        # pinned white-box: it exists precisely for that defensive gap.
        from omni_mercury_engine.cognitive.ethical_bounding import (
            EthicalConstraintViolationError,
        )

        gate = SigmaImmutableGate(verify_corpus=False)
        gate._gate = None
        gate._gate_load_error = None
        with pytest.raises(EthicalConstraintViolationError) as excinfo:
            gate.enforce(action="unit", scalar_vector=build_sigma_immutable_vector(1.0))
        assert excinfo.value.check == "gosnn_unavailable"
        assert excinfo.value.score == 0.0
        assert excinfo.value.details["gate_load_error"] == "σ_Immutable trained gate unavailable"

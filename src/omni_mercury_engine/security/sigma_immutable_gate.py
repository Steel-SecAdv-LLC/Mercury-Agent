# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""σ_Immutable hard ethical gate (Wave B item 1).

This module exposes :class:`SigmaImmutableGate`, a process-wide singleton
that lifts ``EthicalGate`` from informational metadata to a *second hard
ethical gate* that runs alongside :class:`BenevolenceScorer.enforce` at
every decision boundary documented in
:mod:`omni_mercury_engine.ethical`.

Two failure modes raise :class:`EthicalConstraintViolationError`:

* ``check="sigma_immutable"`` — the trained σ_Immutable network produced a
  score below its domain-calibrated threshold (or the signed corpus that
  trained the gate failed signature verification at startup).
* ``check="gosnn_unavailable"`` — the network could not be evaluated at
  all (PyTorch missing, weights file absent, signed corpus tampered or
  unreadable).  No engine-level "fallback" is allowed; an unevaluable
  gate is an unsafe gate, and the boundary fails closed.

The singleton verifies the Ed25519 + ML-DSA-65 signatures over
``sigma_immutable_corpus.json`` on first construction (Wave B item 2).
A failed verification is recorded once and turns every subsequent
:meth:`enforce` call into a hard violation.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    EthicalConstraintViolationError,
)
from omni_mercury_engine.core.centralized_constants import ETHICAL

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SIGMA_IMMUTABLE_INPUT_DIM: int = 256

#: Trained-network decision threshold (0.93).  Single authoritative source:
#: :data:`omni_mercury_engine.core.centralized_constants.EthicalConstants.SIGMA_IMMUTABLE_TRAINED_THRESHOLD`
#: — the ethical-band lower bound of the ``scripts/train_sigma_immutable.py``
#: labelling rule.  Deliberately distinct from the stricter GOSNN gating
#: default ``ETHICAL.SIGMA_IMMUTABLE_DEFAULT`` (0.96); see the NOTE in
#: ``EthicalConstants`` for the two-threshold design.
SIGMA_IMMUTABLE_DEFAULT_THRESHOLD: float = ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD

# ---------------------------------------------------------------------------
# σ_Immutable layout constants — single source of truth.
#
# The σ_Immutable scalar vector is 256-D:
#
#   indices [0,   SIGMA_ETHICAL_BAND_END)    — ethical scalars (27 dims)
#   indices [SIGMA_ETHICAL_BAND_END, SIGMA_USED_BAND_END)
#                                              — non-ethical active band
#                                                (153 dims, training U[0,2])
#   indices [SIGMA_USED_BAND_END, SIGMA_IMMUTABLE_DIM)
#                                              — zero-padded reserved tail
#                                                (76 dims)
#
# These constants are imported by ``sigma_immutable_corpus``, the
# orchestrator, and the neurosymbolic hub so the layout cannot drift
# between the trainer, the corpus, and the boundary builders.
# ---------------------------------------------------------------------------

#: Total width of the σ_Immutable scalar vector (256-D).
SIGMA_IMMUTABLE_DIM: int = SIGMA_IMMUTABLE_INPUT_DIM

#: Index just past the last ethical-band column (27).  Mirrors
#: :data:`SIGMA_IMMUTABLE_ETHICAL_DIMS`.
SIGMA_ETHICAL_BAND_END: int = 27

#: Index just past the last actively-used column (180).  The remaining
#: ``SIGMA_IMMUTABLE_DIM - SIGMA_USED_BAND_END`` (76) columns are
#: deterministically zero-padded so the corpus and the trainer share an
#: identical "reserved tail" the gate never reads.
SIGMA_USED_BAND_END: int = 180

#: Public alias used by :mod:`sigma_immutable_corpus` and the trainer
#: scripts so corpus-side code does not have to reach into a layout
#: constant whose name speaks of ranges rather than the corpus.
CORPUS_USED_DIM: int = SIGMA_USED_BAND_END

# ---------------------------------------------------------------------------
# σ_Immutable input-vector calibration constants.
#
# The σ_Immutable network was trained on samples drawn from ``U[0, 2]``,
# with the ethical band ``[1.5, 2.0]`` reserved for "permissible" signals
# (anchored at 1.5 = MINIMUM_BENEVOLENCE_FLOOR projected up, peaking at
# 2.0 for fully-aligned actions).  Any benevolence value below
# MINIMUM_BENEVOLENCE_FLOOR is mapped into ``[0.0, 0.5]`` so the gate
# fires.  These constants pin the projection so it can never silently
# drift between the two boundaries that build σ_Immutable input vectors
# (``NeuroSymbolicHub`` and ``CognitiveOrchestrator``).
# ---------------------------------------------------------------------------

#: Width of the per-sample ethical band — the leading slice of the input
#: vector that carries the projected benevolence signal.  Kept as an
#: alias of :data:`SIGMA_ETHICAL_BAND_END` so legacy callers continue to
#: import either name.
SIGMA_IMMUTABLE_ETHICAL_DIMS: int = SIGMA_ETHICAL_BAND_END

#: Lower bound for permissible benevolence values inside the trained
#: positive band.  Matches MINIMUM_BENEVOLENCE_FLOOR (0.70) projected up.
SIGMA_IMMUTABLE_PERMISSIBLE_LOW: float = 1.5

#: Upper bound for permissible benevolence values inside the trained
#: positive band.  Matches the upper end of ``U[0, 2]`` training support.
SIGMA_IMMUTABLE_PERMISSIBLE_HIGH: float = 2.0

#: Upper bound for impermissible benevolence values (forces σ_Immutable
#: below the calibrated decision threshold).
SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH: float = 0.5

#: Width of the trained positive band in benevolence units, derived
#: from MINIMUM_BENEVOLENCE_FLOOR so a future floor change propagates
#: automatically.
_PERMISSIBLE_INPUT_RANGE: float = 1.0 - MINIMUM_BENEVOLENCE_FLOOR

#: Per-anchor hard floor every genuine ethical safety anchor must clear at
#: the anomaly-detection boundary.  Set to the σ_Immutable trainer's
#: **ethical-band lower bound** (``SIGMA_IMMUTABLE_DEFAULT_THRESHOLD``,
#: 0.93): the labelling rule in ``scripts/train_sigma_immutable.py`` draws
#: every positive sample's ethical dimensions from ``U[0.93, 2.0]``, so an
#: ethical anchor below 0.93 is, by that rule, a *negative*.  The
#: deterministic floor enforces that per-anchor condition directly --
#: exactly what the soft 256->64->1 network was supposed to learn but
#: only approximates.
#:
#: Why 0.93 and not the lower ``MINIMUM_BENEVOLENCE_FLOOR`` (0.70): a
#: held-out adversarial set showed a 0.70 floor leaves a [0.70, 0.93)
#: gap.  Single-anchor degradation into that band (e.g. benevolence 0.80,
#: morality 0.85) passed the floor and the network -- whose score is
#: insensitive to a single anchor when the rest sit at their >=0.99
#: defaults -- still scored above threshold.  0.93 closes the gap while
#: keeping margin: every genuine anchor defaults to >= 0.99 (the lowest is
#: omnibenevolence == BENEVOLENCE_IMMUTABLE), so the healthy operating
#: point clears the floor by >= 0.06.  Narrative tuning scalars are
#: excluded by name (see ``GlobalOmniScalarNetwork.critical_ethical_anchors``),
#: so their intentionally-low defaults never trip it.
#:
#: The floor is deterministic (no synthetic data gates the safety-critical
#: decision) and fail-closed.
CRITICAL_ETHICAL_FLOOR: float = SIGMA_IMMUTABLE_DEFAULT_THRESHOLD


def _is_pqc_backend_unavailable(exc: BaseException) -> bool:
    """Whether *exc* is AMA Cryptography's typed "PQC backend unavailable" error.

    Aligns σ_Immutable's corpus-verification failure classification with
    ``ama_cryptography``'s public exception hierarchy: ``PQCUnavailableError``
    and its ``QuantumSignatureUnavailableError`` subclass are raised when the
    native post-quantum signature backend is not loadable — a known, actionable
    build/deployment condition, *not* an unexpected fault. Falls back to a
    string probe on the canonical ``PQC_UNAVAILABLE`` marker if the exception
    type cannot be imported (e.g. the AMA package itself failed to import), so
    the classification is robust regardless of how far the backend got.

    Only the *expected* ``ImportError`` is swallowed silently. An unexpected
    fault while importing the exception module (e.g. a runtime error inside the
    package) is logged at warning level rather than hidden — silently dropping
    it would mask a real environment problem behind the string probe. The
    classifier still never raises: it must not displace the original
    corpus-verification exception it is being asked to classify.

    This never changes the gate's decision — an unverifiable corpus is untrusted
    no matter *why* verification could not run — it only sharpens the operator
    diagnostic.
    """
    try:
        from ama_cryptography.exceptions import PQCUnavailableError
    except ImportError:
        # Expected on builds without the AMA package or the typed symbol; fall
        # through to the canonical string-marker probe below.
        pass
    except Exception:  # pragma: no cover - defensive
        # An unexpected import-time fault (not a plain ImportError) must not be
        # swallowed silently: surface it for debugging, then still fall back so
        # this diagnostic never changes the gate's fail-closed decision.
        logger.warning(
            "Unexpected error importing ama_cryptography.exceptions while "
            "classifying a PQC-backend failure; falling back to string probe.",
            exc_info=True,
        )
    else:
        if isinstance(exc, PQCUnavailableError):
            return True
    return "PQC_UNAVAILABLE" in str(exc)


def project_benevolence_to_sigma_band(benevolence_score: float) -> float:
    """Project a clamped benevolence score into the σ_Immutable input band.

    The σ_Immutable trainer (``scripts/train_sigma_immutable.py``)
    expects each sample's ethical-band slice (the first
    :data:`SIGMA_IMMUTABLE_ETHICAL_DIMS` dimensions of every 256-D input
    vector) to be drawn from one of two regions:

    * ``benevolence_score >= MINIMUM_BENEVOLENCE_FLOOR`` (0.70) — linear
      lift into ``[SIGMA_IMMUTABLE_PERMISSIBLE_LOW,
      SIGMA_IMMUTABLE_PERMISSIBLE_HIGH]`` (i.e. ``[1.5, 2.0]``); benevolence
      has already been independently checked by ``BenevolenceScorer.enforce``
      so by the time σ_Immutable runs the analysis is known-permissible
      and we project it into the part of the distribution the network is
      most confident about.
    * ``benevolence_score < 0.70`` (a defensive bypass) — linear damp
      into ``[0.0, SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH]`` so σ_Immutable
      still fires.

    Args:
        benevolence_score: Already-clamped benevolence verdict in [0, 1].

    Returns:
        Float in ``[0.0, SIGMA_IMMUTABLE_PERMISSIBLE_HIGH]`` ready to be
        broadcast into the leading
        :data:`SIGMA_IMMUTABLE_ETHICAL_DIMS` dimensions of a σ_Immutable
        input vector.
    """
    if benevolence_score >= MINIMUM_BENEVOLENCE_FLOOR:
        scale = (SIGMA_IMMUTABLE_PERMISSIBLE_HIGH - SIGMA_IMMUTABLE_PERMISSIBLE_LOW) / (
            _PERMISSIBLE_INPUT_RANGE
        )
        return float(
            np.clip(
                SIGMA_IMMUTABLE_PERMISSIBLE_LOW
                + (benevolence_score - MINIMUM_BENEVOLENCE_FLOOR) * scale,
                SIGMA_IMMUTABLE_PERMISSIBLE_LOW,
                SIGMA_IMMUTABLE_PERMISSIBLE_HIGH,
            )
        )
    # When benevolence is below the floor, dampen linearly into
    # [0, SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH] (== [0, 0.5]) so the gate
    # response stays in the network's negative band.  This matches the
    # original ``benevolence_score * 0.5`` mapping used at both
    # boundaries before extraction.
    return float(
        np.clip(
            benevolence_score * SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH,
            0.0,
            SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH,
        )
    )


#: Width of the per-sample signal window that immediately follows the
#: ethical band.  The orchestrator, the hub, and the Wave C surfaces all
#: write their per-call severity / anomaly / neural-symbolic-fused signal
#: into ``[SIGMA_IMMUTABLE_ETHICAL_DIMS, SIGMA_IMMUTABLE_ETHICAL_DIMS +
#: SIGMA_SIGNAL_WINDOW)`` so the calibration is shared verbatim.
SIGMA_SIGNAL_WINDOW: int = 33


def _sigma_base_vector(benevolence_score: float) -> np.ndarray[Any, Any]:
    """Build the shared σ_Immutable base vector (ethical band + active band).

    Every boundary that scores a single ethical verdict shares this base:

    * indices ``[0, SIGMA_IMMUTABLE_ETHICAL_DIMS)`` hold the projected
      benevolence value (see :func:`project_benevolence_to_sigma_band`);
    * indices ``[SIGMA_IMMUTABLE_ETHICAL_DIMS, SIGMA_USED_BAND_END)`` are
      centred at ``1.0`` (the training ``U[0, 2]`` midpoint);
    * the reserved tail ``[SIGMA_USED_BAND_END, SIGMA_IMMUTABLE_DIM)`` stays
      zero so the corpus, the trainer, and every boundary agree on layout.

    Callers overlay their per-sample signal on top of the returned vector
    (the canonical single-verdict overlay lives in
    :func:`build_sigma_immutable_vector`; the neuro-symbolic hub overlays a
    richer per-sample neural / symbolic / fused signal of its own).
    """
    vector = np.zeros(SIGMA_IMMUTABLE_INPUT_DIM, dtype=np.float64)
    vector[:SIGMA_IMMUTABLE_ETHICAL_DIMS] = project_benevolence_to_sigma_band(benevolence_score)
    vector[SIGMA_IMMUTABLE_ETHICAL_DIMS:SIGMA_USED_BAND_END] = 1.0
    return vector


def build_sigma_immutable_vector(
    benevolence_score: float,
    severity: float = 0.0,
    anomaly_prob: float = 0.0,
) -> np.ndarray[Any, Any]:
    """Canonical 256-D σ_Immutable input vector for single-verdict boundaries.

    Promoted out of ``CognitiveOrchestrator._build_sigma_immutable_vector``
    (Wave C) so the engine, orchestrator, narrative-voice, federation
    aggregator, and FL-server boundaries all build their σ_Immutable input
    from one shared, calibrated helper instead of duplicating the layout
    (the prior copies silently risked drifting apart).

    The first :data:`SIGMA_IMMUTABLE_ETHICAL_DIMS` columns carry the
    projected benevolence verdict; the next :data:`SIGMA_SIGNAL_WINDOW`
    columns carry the per-call severity / anomaly signal (broadcast
    uniformly), and the remainder of the active band sits at the training
    midpoint.  ``severity == anomaly_prob == 0`` reproduces the engine's
    benevolence-only boundary vector byte-for-byte.

    Args:
        benevolence_score: Already-checked benevolence verdict in ``[0, 1]``.
        severity: Per-call severity signal in ``[0, 1]``.
        anomaly_prob: Per-call anomaly probability in ``[0, 1]``.

    Returns:
        ``(256,)`` float64 vector ready for :meth:`SigmaImmutableGate.enforce`.
    """
    vector = _sigma_base_vector(benevolence_score)
    signal_perturbation = float(np.clip(0.5 * severity + 0.5 * anomaly_prob, 0.0, 1.0))
    signal_window_end = SIGMA_IMMUTABLE_ETHICAL_DIMS + SIGMA_SIGNAL_WINDOW
    vector[SIGMA_IMMUTABLE_ETHICAL_DIMS:signal_window_end] = 1.0 + 0.4 * signal_perturbation
    return vector


def enforce_dual_ethical_gate(
    *,
    benevolence_scorer: Any,
    sigma_gate: SigmaImmutableGate,
    action: str,
    context: dict[str, Any],
    boundary: str,
    domain: str,
    severity: float = 0.0,
    anomaly_prob: float = 0.0,
    extra_details: dict[str, Any] | None = None,
) -> Any:
    """Run the BenevolenceScorer + σ_Immutable dual hard ethical gate.

    Single shared primitive for the σ_Immutable Wave C surfaces (narrative
    voice, federation aggregator, FL server) so the benevolence-first /
    σ_Immutable-second contract cannot drift between boundaries.  Mirrors
    the inline dual-gate already wired at the engine, orchestrator, and
    neuro-symbolic-hub boundaries.

    Both gates fail closed and the σ_Immutable call is **not** wrapped in a
    swallowing ``try/except`` — a violation propagates as
    :class:`EthicalConstraintViolationError` so the caller's operation halts.

    Args:
        benevolence_scorer: Object exposing ``.enforce(action, context)``
            (a :class:`BenevolenceScorer`); raises on benevolence violation.
        sigma_gate: The process-wide :class:`SigmaImmutableGate`.
        action: Positive-keyword-rich action description (the boundary's
            *purpose*, not the anomalous payload).
        context: Benevolence-scoring context dict.
        boundary: Fully-qualified boundary name for the audit trail.
        domain: Already-:func:`sanitize_domain`-d domain hint.
        severity: Per-call severity in ``[0, 1]``.
        anomaly_prob: Per-call anomaly probability in ``[0, 1]``.
        extra_details: Optional structured context merged into the σ_Immutable
            violation details.

    Returns:
        The :class:`EthicalScore` from the benevolence gate on success.

    Raises:
        EthicalConstraintViolationError: ``check="benevolence"`` (first gate),
            ``check="sigma_immutable"`` or ``check="gosnn_unavailable"``
            (second gate).
    """
    ethical_score = benevolence_scorer.enforce(action, context)
    sigma_vector = build_sigma_immutable_vector(
        benevolence_score=float(ethical_score.benevolence_score),
        severity=severity,
        anomaly_prob=anomaly_prob,
    )
    details: dict[str, Any] = {
        "boundary": boundary,
        "domain": domain,
        "benevolence_score": float(ethical_score.benevolence_score),
        "severity": float(severity),
        "anomaly_prob": float(anomaly_prob),
    }
    if extra_details:
        details.update(extra_details)
    sigma_gate.enforce(
        action=f"{boundary}:domain={domain}",
        scalar_vector=sigma_vector,
        details=details,
    )
    return ethical_score


@dataclass(frozen=True)
class SigmaImmutableEvaluation:
    """Outcome of a single σ_Immutable evaluation.

    Attributes:
        score: Sigmoid output of the trained gate (or NumPy fallback when
            the trained network is unavailable — fallback never produces
            a passing verdict at the boundary; the boundary fails closed).
        threshold: Domain-calibrated decision threshold used.
        passes: True iff ``score >= threshold`` AND the trained network
            was used (untrained heuristic never passes the boundary).
        backend: ``"torch"`` for the trained network, ``"numpy_fallback"``
            for the deterministic heuristic, ``"unavailable"`` when even
            the heuristic could not be computed (e.g. NaN-only input).
    """

    score: float
    threshold: float
    passes: bool
    backend: str


class SigmaImmutableGate:
    """Process-wide singleton enforcing σ_Immutable at every boundary.

    The gate wraps the trained
    :class:`omni_mercury_engine.core.global_omni_scalar_network.EthicalGate`
    network and adds:

    * Eager signature verification of the labelled corpus that trained
      the network.  If verification fails, every :meth:`enforce` call
      raises ``check="sigma_immutable"`` with the verification error
      surfaced in ``details``.
    * Hard ``check="gosnn_unavailable"`` failure when the trained network
      cannot be loaded (no torch, no weights file, or corpus
      verification failed at startup).
    * Score padding / shape coercion so callers at different boundaries
      (engine, hub, orchestrator) can pass any 1-D float vector without
      knowing the network's input width.

    **Authority (read before trusting the score).**
    The learned network is trained on a **harvested config-integrity
    corpus** (``scripts/train_sigma_immutable.py`` over
    ``scripts/harvest_sigma_baseline.py``'s real intact vector): positives
    are the actual intact operational configuration plus intact variations
    holding the 24 critical ethical *anchors* at-or-above threshold;
    negatives are real tamper mutations (anchor collapse).  Because the
    real baseline is a training positive, the network recognises the true
    production configuration by construction (a build-time DoS guard
    asserts it), and — unlike the earlier synthetic net, which passed a
    benevolence-zeroed vector — it now refuses anchor collapse, agreeing
    with the floor.  The **authoritative** ethical gate is nonetheless the
    deterministic critical-ethical floor (:meth:`enforce_ethical_floor`),
    which callers compose BEFORE :meth:`enforce` (see
    ``OmniMercuryEngine.detect_with_fusion``): a collapsed anchor is a
    categorical, fail-closed refusal no learned score can override.  The
    learned score stays a secondary, advisory config-integrity check; the
    harvested-corpus provenance is recorded in ``docs/DORMANCY_LEDGER.md``.
    """

    _instance: SigmaImmutableGate | None = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        threshold: float = SIGMA_IMMUTABLE_DEFAULT_THRESHOLD,
        verify_corpus: bool = True,
    ) -> None:
        """Construct the gate; reuses the trained ``EthicalGate``.

        Args:
            threshold: Decision threshold (default 0.93 —
                ``ETHICAL.SIGMA_IMMUTABLE_TRAINED_THRESHOLD``, the trained
                network's calibrated decision threshold; deliberately
                distinct from the stricter GOSNN gating default
                ``ETHICAL.SIGMA_IMMUTABLE_DEFAULT`` = 0.96).  Lower values
                are clamped to 0.0, higher values to 1.0.
            verify_corpus: Whether to verify the signed corpus on
                construction.  Disabled only by tests that exercise the
                no-corpus failure path explicitly; production callers
                always leave it enabled.
        """
        self._threshold = float(np.clip(threshold, 0.0, 1.0))
        self._gate: Any = None
        self._gate_load_error: str | None = None
        self._corpus_error: str | None = None
        self._lock = threading.Lock()
        self._init_gate()
        if verify_corpus:
            self._verify_corpus()

    @classmethod
    def instance(cls) -> SigmaImmutableGate:
        """Return the process-wide singleton, constructing it lazily."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_for_tests(cls) -> None:
        """Drop the cached singleton (test-only).

        The next :meth:`instance` call will reconstruct from scratch,
        re-running corpus verification.  Production code does not call
        this; it exists so unit tests can simulate startup-time failures
        without polluting other tests.
        """
        with cls._instance_lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_gate(self) -> None:
        """Load the trained ``EthicalGate`` once and remember the result.

        If torch or the weights file is missing, ``self._gate`` stays
        ``None`` and ``self._gate_load_error`` records the reason — the
        boundary then fails closed with ``check="gosnn_unavailable"``.
        """
        try:
            from omni_mercury_engine.core.global_omni_scalar_network import (
                EthicalGate,
            )

            gate = EthicalGate(
                input_dim=SIGMA_IMMUTABLE_INPUT_DIM,
                threshold=self._threshold,
            )
        except ImportError as exc:
            # The actual import failure may be torch (the GOSNN module
            # imports torch at top level), an EthicalGate dependency, or
            # the global_omni_scalar_network module itself.  Surface the
            # raw ImportError name + module so the on-call doesn't have
            # to guess whether to install torch or fix a packaging miss.
            self._gate_load_error = (
                "σ_Immutable EthicalGate could not be imported "
                f"({exc.__class__.__name__}: {exc}); the boundary will "
                "fail closed with check='gosnn_unavailable'."
            )
            logger.warning(self._gate_load_error)
            return

        if not getattr(gate, "_trained", False):
            self._gate_load_error = (
                "σ_Immutable EthicalGate is untrained (weights file "
                "missing or torch unavailable). "
                "Run `python scripts/train_sigma_immutable.py` to "
                "regenerate weights."
            )
            logger.warning(self._gate_load_error)
            return

        self._gate = gate

    def _verify_corpus(self) -> None:
        """Verify the signed σ_Immutable corpus on first construction.

        A failed verification is *not* a soft warning — it is recorded so
        every subsequent :meth:`enforce` raises
        ``check="sigma_immutable"`` until the corpus is regenerated.
        """
        try:
            from omni_mercury_engine.security.sigma_immutable_corpus import (
                CorpusVerificationError,
                verify_corpus_signatures,
            )

            verify_corpus_signatures()
        except ImportError as exc:
            # The corpus module itself is missing — that is a build
            # error, not a soft fallback.
            self._corpus_error = f"σ_Immutable corpus module unavailable: {exc}"
            logger.error(self._corpus_error)
        except Exception as exc:
            # ``CorpusVerificationError`` is the documented type but we
            # surface any unexpected failure (file missing, JSON broken,
            # checksum mismatch) — they all mean the corpus is no
            # longer trustworthy.
            from omni_mercury_engine.security.sigma_immutable_corpus import (
                CorpusVerificationError,
            )

            if isinstance(exc, CorpusVerificationError):
                self._corpus_error = str(exc)
            elif _is_pqc_backend_unavailable(exc):
                # AMA Cryptography raises a typed PQCUnavailableError /
                # QuantumSignatureUnavailableError when the native post-quantum
                # signature backend is not loadable. That is a known, actionable
                # build/deployment condition — not an "unexpected" fault — so
                # report it precisely (aligned with ama_cryptography's exception
                # hierarchy). The gate still fails closed: an unverifiable corpus
                # is untrusted regardless of *why* verification could not run.
                self._corpus_error = (
                    "σ_Immutable corpus verification unavailable: the AMA "
                    "Cryptography native post-quantum signature backend is not "
                    f"loadable ({type(exc).__name__}: {exc}). Build the AMA native "
                    "library — see docs/INSTALLATION.md 'Post-Quantum Cryptography "
                    "backend' or the build-ama-cryptography CI action."
                )
            else:
                self._corpus_error = (
                    f"σ_Immutable corpus verification failed unexpectedly: "
                    f"{type(exc).__name__}: {exc}"
                )
            logger.error(self._corpus_error)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Decision threshold used by :meth:`evaluate`."""
        return self._threshold

    @property
    def is_trained(self) -> bool:
        """True iff a trained σ_Immutable network is loaded."""
        return self._gate is not None

    @property
    def corpus_error(self) -> str | None:
        """Recorded corpus-verification error, if any (else ``None``)."""
        return self._corpus_error

    @property
    def gate_load_error(self) -> str | None:
        """Recorded gate-loading error, if any (else ``None``)."""
        return self._gate_load_error

    def evaluate(self, scalar_vector: np.ndarray[Any, Any]) -> SigmaImmutableEvaluation:
        """Score ``scalar_vector`` against the σ_Immutable network.

        The vector is padded or truncated to the network's input width.
        Callers at different boundaries (engine: 180-dim GOSNN scalar
        vector; hub: per-sample feature row; orchestrator: synthetic
        ethical-context vector) all funnel through this method so the
        decision is uniform across the boundary.

        Args:
            scalar_vector: 1-D float vector (any length).

        Returns:
            :class:`SigmaImmutableEvaluation` with score, threshold,
            pass/fail, and backend used.
        """
        if self._gate is None:
            return SigmaImmutableEvaluation(
                score=0.0,
                threshold=self._threshold,
                passes=False,
                backend="unavailable",
            )

        passes_local, score_local = self._gate.evaluate(np.asarray(scalar_vector))
        return SigmaImmutableEvaluation(
            score=float(score_local),
            threshold=self._threshold,
            passes=bool(passes_local),
            backend="torch",
        )

    def critical_ethical_floor_violations(
        self, anchors: dict[str, float]
    ) -> list[tuple[str, float]]:
        """Return the ethical anchors that sit below :data:`CRITICAL_ETHICAL_FLOOR`.

        This is the deterministic half of the boundary gate.  It does not
        touch the trained network — it is a pure, synthetic-data-free
        threshold check on the *named* ethical safety anchors supplied by
        :meth:`GlobalOmniScalarNetwork.critical_ethical_anchors`.

        A non-finite anchor (``NaN`` / ``±inf``) is always a violation:
        ``NaN < floor`` is ``False``, so a collapsed-to-NaN anchor would
        otherwise slip the floor *and* the network (which coerces NaN->0
        and scores the result as PASS) — a fail-open hole.  The floor is
        fail-closed, so any value that is not a finite number at-or-above
        the floor refuses.

        Args:
            anchors: Mapping of ethical-anchor name -> value (already
                excludes narrative/personality tuning scalars).

        Returns:
            List of ``(name, value)`` pairs that fail the floor (non-finite
            or below it), in input order.  Empty when every anchor holds a
            finite value at-or-above the floor.
        """
        return [
            (name, float(value))
            for name, value in anchors.items()
            if not math.isfinite(float(value)) or float(value) < CRITICAL_ETHICAL_FLOOR
        ]

    def enforce_ethical_floor(
        self,
        action: str,
        anchors: dict[str, float],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Deterministic critical-ethical hard floor, composed before the network.

        Raises a fail-closed violation if any genuine ethical safety
        anchor has collapsed below :data:`CRITICAL_ETHICAL_FLOOR`.  Callers
        run this *and* :meth:`enforce` (the trained-network check); the
        boundary passes only when both do, so the learned
        network can never grant false assurance for a categorical ethical
        breach.

        Args:
            action: Human-readable description of the gated action.
            anchors: Named ethical safety anchors (see
                :meth:`GlobalOmniScalarNetwork.critical_ethical_anchors`).
            details: Optional structured context for any violation.

        Raises:
            EthicalConstraintViolationError: With
                ``check="sigma_immutable_ethical_floor"`` when one or more
                anchors are below the floor.
        """
        violations = self.critical_ethical_floor_violations(anchors)
        if not violations:
            return
        merged_details: dict[str, Any] = {"action": action}
        if details:
            merged_details.update(details)
        merged_details["floor"] = CRITICAL_ETHICAL_FLOOR
        merged_details["floor_violations"] = dict(violations)
        worst_value = min(value for _, value in violations)
        raise EthicalConstraintViolationError(
            action=action,
            score=worst_value,
            threshold=CRITICAL_ETHICAL_FLOOR,
            check="sigma_immutable_ethical_floor",
            details=merged_details,
        )

    def enforce(
        self,
        action: str,
        scalar_vector: np.ndarray[Any, Any],
        details: dict[str, Any] | None = None,
    ) -> SigmaImmutableEvaluation:
        """Evaluate and raise on σ_Immutable / gosnn_unavailable failure.

        This is the single primitive used by every decision boundary.

        Args:
            action: Human-readable description of the gated action,
                propagated into the exception for auditability.
            scalar_vector: Vector to score (see :meth:`evaluate`).
            details: Optional structured context attached to any
                violation (boundary name, sample index, etc.).

        Returns:
            :class:`SigmaImmutableEvaluation` on success.

        Raises:
            EthicalConstraintViolationError: With ``check="sigma_immutable"``
                when the score is below threshold or the corpus failed
                verification; with ``check="gosnn_unavailable"`` when the
                trained network could not be loaded at all.
        """
        merged_details: dict[str, Any] = {"action": action}
        if details:
            merged_details.update(details)

        # Corpus verification failure poisons every call until the
        # operator regenerates the corpus.  This precedes the
        # gate-availability check so a poisoned corpus never silently
        # masquerades as "GOSNN unavailable".
        if self._corpus_error is not None:
            merged_details["corpus_error"] = self._corpus_error
            raise EthicalConstraintViolationError(
                action=action,
                score=0.0,
                threshold=self._threshold,
                check="sigma_immutable",
                details=merged_details,
            )

        if self._gate is None:
            merged_details["gate_load_error"] = self._gate_load_error or (
                "σ_Immutable trained gate unavailable"
            )
            raise EthicalConstraintViolationError(
                action=action,
                score=0.0,
                threshold=self._threshold,
                check="gosnn_unavailable",
                details=merged_details,
            )

        evaluation = self.evaluate(scalar_vector)
        if not evaluation.passes:
            merged_details["sigma_immutable_score"] = evaluation.score
            merged_details["backend"] = evaluation.backend
            raise EthicalConstraintViolationError(
                action=action,
                score=evaluation.score,
                threshold=self._threshold,
                check="sigma_immutable",
                details=merged_details,
            )
        return evaluation


def get_sigma_immutable_gate() -> SigmaImmutableGate:
    """Return the process-wide :class:`SigmaImmutableGate` singleton."""
    return SigmaImmutableGate.instance()


__all__ = [
    "CORPUS_USED_DIM",
    "CRITICAL_ETHICAL_FLOOR",
    "SIGMA_ETHICAL_BAND_END",
    "SIGMA_IMMUTABLE_DEFAULT_THRESHOLD",
    "SIGMA_IMMUTABLE_DIM",
    "SIGMA_IMMUTABLE_ETHICAL_DIMS",
    "SIGMA_IMMUTABLE_IMPERMISSIBLE_HIGH",
    "SIGMA_IMMUTABLE_INPUT_DIM",
    "SIGMA_IMMUTABLE_PERMISSIBLE_HIGH",
    "SIGMA_IMMUTABLE_PERMISSIBLE_LOW",
    "SIGMA_SIGNAL_WINDOW",
    "SIGMA_USED_BAND_END",
    "SigmaImmutableEvaluation",
    "SigmaImmutableGate",
    "build_sigma_immutable_vector",
    "enforce_dual_ethical_gate",
    "get_sigma_immutable_gate",
    "project_benevolence_to_sigma_band",
]

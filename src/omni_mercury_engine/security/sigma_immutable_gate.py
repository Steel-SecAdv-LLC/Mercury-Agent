"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

σ_Immutable hard ethical gate (Wave B item 1).

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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
)

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SIGMA_IMMUTABLE_INPUT_DIM: int = 256
SIGMA_IMMUTABLE_DEFAULT_THRESHOLD: float = 0.93


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
            threshold: Decision threshold (default 0.93, matching
                ``GOSNN`` defaults).  Lower values are clamped to 0.0,
                higher values to 1.0.
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
            self._gate_load_error = (
                f"torch / GOSNN unavailable for σ_Immutable: {exc}"
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
            self._corpus_error = (
                f"σ_Immutable corpus module unavailable: {exc}"
            )
            logger.error(self._corpus_error)
        except Exception as exc:  # noqa: BLE001 - captured & re-raised on use
            # ``CorpusVerificationError`` is the documented type but we
            # surface any unexpected failure (file missing, JSON broken,
            # checksum mismatch) — they all mean the corpus is no
            # longer trustworthy.
            from omni_mercury_engine.security.sigma_immutable_corpus import (
                CorpusVerificationError,
            )

            if isinstance(exc, CorpusVerificationError):
                self._corpus_error = str(exc)
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

    def evaluate(self, scalar_vector: np.ndarray) -> SigmaImmutableEvaluation:
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

    def enforce(
        self,
        action: str,
        scalar_vector: np.ndarray,
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
    "SIGMA_IMMUTABLE_DEFAULT_THRESHOLD",
    "SIGMA_IMMUTABLE_INPUT_DIM",
    "SigmaImmutableEvaluation",
    "SigmaImmutableGate",
    "get_sigma_immutable_gate",
]

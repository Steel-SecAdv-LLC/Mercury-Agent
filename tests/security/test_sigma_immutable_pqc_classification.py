# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""σ_Immutable corpus-verification failure classification.

When the AMA Cryptography native post-quantum signature backend is not
loadable, AMA raises a *typed* ``PQCUnavailableError`` /
``QuantumSignatureUnavailableError``. The σ_Immutable gate must classify that
known build/deployment condition precisely (so operators get an actionable
message) — and it must still fail closed, recording a corpus error so every
subsequent ``enforce`` blocks. These tests pin both properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from omni_mercury_engine.security import sigma_immutable_corpus
from omni_mercury_engine.security.sigma_immutable_gate import (
    SigmaImmutableGate,
    _is_pqc_backend_unavailable,
)

if TYPE_CHECKING:
    import pytest


class TestPqcUnavailableHelper:
    """``_is_pqc_backend_unavailable`` aligns with AMA's exception hierarchy."""

    def test_recognizes_ama_typed_exceptions(self) -> None:
        from ama_cryptography.exceptions import (
            PQCUnavailableError,
            QuantumSignatureUnavailableError,
        )

        assert _is_pqc_backend_unavailable(
            QuantumSignatureUnavailableError("PQC_UNAVAILABLE: Unknown backend state")
        )
        assert _is_pqc_backend_unavailable(
            PQCUnavailableError("PQC_UNAVAILABLE: Dilithium backend not available")
        )

    def test_string_marker_fallback(self) -> None:
        # Robust even if the typed class can't be matched: the canonical
        # PQC_UNAVAILABLE marker on any exception is sufficient.
        assert _is_pqc_backend_unavailable(RuntimeError("PQC_UNAVAILABLE: backend not built"))

    def test_rejects_unrelated_failures(self) -> None:
        assert not _is_pqc_backend_unavailable(ValueError("corpus SHA3-256 mismatch"))
        assert not _is_pqc_backend_unavailable(RuntimeError("disk full"))


class TestCorpusVerificationClassification:
    """``_verify_corpus`` classifies the PQC-unavailable case precisely and
    still fails closed."""

    def _bare_gate(self) -> SigmaImmutableGate:
        # Bypass heavy __init__; _verify_corpus only touches _corpus_error.
        gate = SigmaImmutableGate.__new__(SigmaImmutableGate)
        gate._corpus_error = None
        return gate

    def test_pqc_unavailable_is_precise_not_unexpected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ama_cryptography.exceptions import QuantumSignatureUnavailableError

        def _raise(*_a: object, **_k: object) -> None:
            raise QuantumSignatureUnavailableError("PQC_UNAVAILABLE: Unknown backend state")

        monkeypatch.setattr(sigma_immutable_corpus, "verify_corpus_signatures", _raise)

        gate = self._bare_gate()
        gate._verify_corpus()

        assert gate._corpus_error is not None, "must fail closed (error recorded)"
        assert "post-quantum signature backend is not loadable" in gate._corpus_error
        assert "build-ama-cryptography" in gate._corpus_error  # actionable remediation
        assert "failed unexpectedly" not in gate._corpus_error

    def test_genuine_verification_failure_still_classified_plainly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omni_mercury_engine.security.sigma_immutable_corpus import (
            CorpusVerificationError,
        )

        def _raise(*_a: object, **_k: object) -> None:
            raise CorpusVerificationError("σ_Immutable corpus SHA3-256 mismatch")

        monkeypatch.setattr(sigma_immutable_corpus, "verify_corpus_signatures", _raise)

        gate = self._bare_gate()
        gate._verify_corpus()

        assert gate._corpus_error == "σ_Immutable corpus SHA3-256 mismatch"
        assert "post-quantum signature backend" not in gate._corpus_error

    def test_truly_unexpected_failure_still_flagged_unexpected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*_a: object, **_k: object) -> None:
            raise RuntimeError("totally novel fault")

        monkeypatch.setattr(sigma_immutable_corpus, "verify_corpus_signatures", _raise)

        gate = self._bare_gate()
        gate._verify_corpus()

        assert gate._corpus_error is not None
        assert "failed unexpectedly" in gate._corpus_error

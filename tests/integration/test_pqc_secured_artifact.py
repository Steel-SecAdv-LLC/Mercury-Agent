# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Post-quantum signing of detection artifacts (real native backend).

This is the test that justifies the integration lane building the native AMA
Cryptography library: it takes a *real detection result*, serialises it, and
secures it through **Mercury's own** post-quantum surface
(``security.crypto_api.MLDSAProvider`` / ``KyberProvider``) — proving the
detection and cryptography subsystems compose end-to-end on the live ML-DSA-65
/ ML-KEM-1024 primitives.

It is intentionally distinct from:

* ``tests/security/test_pqc_gate_real_ama.py`` — raw ``ama_cryptography``
  primitives + the import-time gate; and
* ``tests/security/test_crypto_api.py`` — the classical Ed25519 provider.

Here the assertion is integration: a Mercury detection artifact is signed and
its integrity is verifiable, tamper-evident, and key-bound.

The package import already requires the native backend (the import-time PQC
gate has no escape hatch), so there is deliberately no ``skipif`` here — a
missing backend fails loudly, which is the correct signal.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
from omni_mercury_engine.security.crypto_api import KyberProvider, MLDSAProvider

pytestmark = pytest.mark.integration


def _signed_detection_artifact() -> bytes:
    """Run a real detection and serialise it to canonical bytes."""
    detector = MercuryAnomalyDetector()
    detector.fit(np.ones((50, 1)))
    result = detector.detect(np.array([1, 1, 1, 9, 1], dtype=float).reshape(-1, 1))
    return json.dumps(
        {
            "scores": [round(float(x), 6) for x in result["scores"]],
            "anomalies": [bool(x) for x in result["is_anomaly"]],
        },
        sort_keys=True,
    ).encode()


class TestDetectionArtifactMlDsaSigning:
    """A detection artifact is ML-DSA-65 signed, verifiable, and tamper-evident."""

    def test_sign_and_verify_round_trip(self) -> None:
        artifact = _signed_detection_artifact()
        provider = MLDSAProvider()
        keypair = provider.generate_keypair()

        signature = provider.sign(artifact, keypair.secret_key)
        assert isinstance(signature, bytes)
        assert len(signature) > 0
        assert provider.verify(artifact, signature, keypair.public_key) is True

    def test_tampered_artifact_is_rejected(self) -> None:
        artifact = _signed_detection_artifact()
        provider = MLDSAProvider()
        keypair = provider.generate_keypair()

        signature = provider.sign(artifact, keypair.secret_key)
        tampered = artifact + b" "
        assert provider.verify(tampered, signature, keypair.public_key) is False

    def test_signature_is_key_bound(self) -> None:
        artifact = _signed_detection_artifact()
        provider = MLDSAProvider()
        signer = provider.generate_keypair()
        impostor = provider.generate_keypair()

        signature = provider.sign(artifact, signer.secret_key)
        # A different public key must not validate the signature.
        assert provider.verify(artifact, signature, impostor.public_key) is False


class TestKyberKeyEstablishment:
    """Mercury's KEM wrapper agrees a shared secret on the native backend."""

    def test_encapsulate_decapsulate_agree(self) -> None:
        provider = KyberProvider()
        keypair = provider.generate_keypair()

        encapsulated = provider.encapsulate(keypair.public_key)
        recovered = provider.decapsulate(encapsulated.ciphertext, keypair.secret_key)

        # The whole correctness property of a KEM: both sides derive the same
        # secret, and ML-KEM-1024 yields a 32-byte shared secret.
        assert recovered == encapsulated.shared_secret
        assert isinstance(encapsulated.shared_secret, bytes)
        assert len(encapsulated.shared_secret) == 32

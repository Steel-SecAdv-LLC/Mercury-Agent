# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contract tests for the FINDOYOU evidence bridge: wire format, three-state honesty, AMA signatures."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.integrations.findoyou_bridge import (
    EVIDENCE_SCHEMA,
    STANDING_LIMITATIONS,
    FindoyouEvidenceBridge,
    verify_packet,
)
from omni_mercury_engine.models import biometric as biometric_module
from omni_mercury_engine.verifiers.three_state import ThreeState

from ..models.test_age_progression_growth import _face

WIRE_STATES = {s.value for s in ThreeState}


@pytest.fixture(scope="module")
def bridge() -> FindoyouEvidenceBridge:
    return FindoyouEvidenceBridge()


@pytest.fixture(scope="module")
def packet(bridge: FindoyouEvidenceBridge) -> dict[str, Any]:
    return bridge.analyze_case(
        "case-0001", _face(), np.ascontiguousarray(_face()[:, ::-1]), source_age=6, years_elapsed=8
    )


class TestWireFormat:
    def test_packet_has_the_contracted_shape(self, packet: dict[str, Any]) -> None:
        assert packet["schema"] == EVIDENCE_SCHEMA
        assert set(packet) == {
            "schema",
            "case_id",
            "created_utc",
            "subject",
            "image_digests",
            "findings",
            "limitations",
            "producer",
            "signature",
        }
        assert set(packet["findings"]) == {
            "identity_match",
            "age_robust_match",
            "progression_audit",
            "candidate_integrity",
        }
        assert packet["limitations"] == list(STANDING_LIMITATIONS)

    def test_every_finding_uses_the_cross_repo_three_state_wire_values(
        self, packet: dict[str, Any]
    ) -> None:
        # FINDOYOU's security.three_state mirrors these values member-for-member.
        assert sorted(WIRE_STATES) == ["grounded", "unavailable", "undecidable"]
        for finding in packet["findings"].values():
            assert finding["state"] in WIRE_STATES

    def test_image_digests_bind_the_exact_pixels(self, bridge: FindoyouEvidenceBridge) -> None:
        face = _face()
        a = bridge.analyze_case("c", face, face)
        altered = face.copy()
        altered[0, 0, 0] ^= 1
        b = bridge.analyze_case("c", face, altered)
        assert a["image_digests"]["reference_sha3_256"] == b["image_digests"]["reference_sha3_256"]
        assert a["image_digests"]["candidate_sha3_256"] != b["image_digests"]["candidate_sha3_256"]


class TestHonesty:
    def test_identity_is_never_grounded_on_appearance_similarity(
        self, bridge: FindoyouEvidenceBridge, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(biometric_module, "DeepFace", None)
        result = bridge.analyze_case("c", _face(), _face())
        identity = result["findings"]["identity_match"]
        assert identity["state"] == "unavailable"
        assert identity["backend"].startswith("harmonic_fallback")
        assert "not verifiable" in identity["reason"]
        # The age-robust match inherits that grounding.
        assert result["findings"]["age_robust_match"]["state"] == "unavailable"

    def test_identity_is_grounded_only_by_a_face_identity_backend(self) -> None:
        grounded = FindoyouEvidenceBridge._identity_finding(
            {"match_score": 0.91, "embedding_backend": "deepface"}
        )
        fallback = FindoyouEvidenceBridge._identity_finding(
            {"match_score": 0.99, "embedding_backend": "harmonic_fallback"}
        )
        missing = FindoyouEvidenceBridge._identity_finding(
            {"match_score": None, "error": "match unavailable: no face"}
        )
        assert grounded["state"] == "grounded" and grounded["score"] == 0.91
        assert fallback["state"] == "unavailable"
        assert missing == {
            "state": "unavailable",
            "score": None,
            "backend": "none",
            "reason": "match unavailable: no face",
        }

    def test_child_case_applies_and_audits_the_growth_model(self, packet: dict[str, Any]) -> None:
        audit = packet["findings"]["progression_audit"]
        assert audit["state"] == "grounded"
        assert audit["geometric_model"]["applied"] is True
        assert audit["geometric_model"]["source_age"] == 6
        assert audit["geometric_model"]["target_age"] == 14
        assert {"structural_similarity", "noise_ratio", "sharpness_ratio", "flags"} <= set(audit)

    def test_missing_elapsed_years_is_unavailable_not_guessed(
        self, bridge: FindoyouEvidenceBridge
    ) -> None:
        result = bridge.analyze_case("c", _face(), _face())
        assert result["findings"]["progression_audit"] == {
            "state": "unavailable",
            "reason": "years_elapsed not provided",
        }
        assert result["findings"]["age_robust_match"]["state"] == "unavailable"

    def test_recaptured_sighting_is_flagged(self, bridge: FindoyouEvidenceBridge) -> None:
        yy, xx = np.indices((160, 160))
        replay = np.clip(
            _face() / 255.0 * 0.6 + 0.15 * np.sin(2 * np.pi * (0.31 * xx + 0.07 * yy))[..., None],
            0,
            1,
        )
        integrity = bridge.analyze_case("c", _face(), replay)["findings"]["candidate_integrity"]
        assert integrity["state"] == "grounded"
        assert integrity["manipulation_or_recapture_suspected"] is True
        assert any("moire" in cue for cue in integrity["cues"])

    def test_clean_sighting_raises_no_manipulation_cue(self, packet: dict[str, Any]) -> None:
        integrity = packet["findings"]["candidate_integrity"]
        assert integrity["manipulation_or_recapture_suspected"] is False
        assert integrity["cues"] == []


class TestSignatures:
    def test_packet_verifies_against_the_pinned_producer_key(
        self, bridge: FindoyouEvidenceBridge, packet: dict[str, Any]
    ) -> None:
        assert verify_packet(packet, bridge.public_key_hex) == (True, "valid")
        assert packet["signature"]["algorithm"] == "ML-DSA-65"
        # FIPS 204 ML-DSA-65 sizes: 1952-byte public key, 3309-byte signature.
        assert len(bytes.fromhex(packet["signature"]["public_key_hex"])) == 1952
        assert len(bytes.fromhex(packet["signature"]["value_hex"])) == 3309

    @pytest.mark.parametrize(
        ("mutate", "reason"),
        [
            (
                # "undecidable" is never emitted for identity, so this changes the
                # packet whether or not a face-identity backend grounded it.
                lambda p: p["findings"]["identity_match"].update(state="undecidable"),
                "changed after signing",
            ),
            (lambda p: p.update(case_id="case-9999"), "changed after signing"),
            (lambda p: p["signature"].update(value_hex="00" * 3309), "does not verify"),
            (lambda p: p.update(schema="other/v2"), "unsupported schema"),
            (lambda p: p.pop("signature"), "no signature block"),
            (lambda p: p["signature"].update(value_hex="zz"), "could not be checked"),
        ],
    )
    def test_every_tampering_is_rejected_with_its_reason(
        self, bridge: FindoyouEvidenceBridge, packet: dict[str, Any], mutate: Any, reason: str
    ) -> None:
        forged = copy.deepcopy(packet)
        mutate(forged)
        valid, why = verify_packet(forged, bridge.public_key_hex)
        assert not valid
        assert reason in why

    def test_a_different_producer_key_is_rejected(self, packet: dict[str, Any]) -> None:
        valid, why = verify_packet(packet, FindoyouEvidenceBridge().public_key_hex)
        assert not valid and "pinned producer key" in why

    def test_provisioned_key_pair_is_used(self) -> None:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        pair = MLDSAProvider().generate_keypair()
        bridge = FindoyouEvidenceBridge(signing_keypair=(pair.public_key, pair.secret_key))
        result = bridge.analyze_case("c", _face(), _face())
        assert bridge.public_key_hex == pair.public_key.hex()
        assert verify_packet(result, pair.public_key.hex())[0]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FINDΩYOU™ bridge: signed, three-state evidence packets for missing-person image analysis.

The trio works as one system: **FINDΩYOU™** runs missing-person cases,
**Mercury** analyses the images, and **AMA Cryptography** signs the result so
the evidence keeps its chain of custody from analysis to case file. This
module is the Mercury side of that contract.

:meth:`FindoyouEvidenceBridge.analyze_case` compares a reference photo of the
missing person with a candidate sighting and returns an
:data:`EVIDENCE_SCHEMA` packet:

* ``identity_match`` — ``GROUNDED`` **only** when a face-identity embedding
  backend produced both embeddings. Appearance-level similarity from the
  harmonic fallback is reported but stays ``UNAVAILABLE`` for identity: a
  spectral resemblance must never become a lead in a missing-person case.
* ``age_robust_match`` — the reference is aged across the elapsed years
  (cardioidal craniofacial growth for children, deterministic texture model)
  and re-matched; it inherits ``identity_match``'s grounding.
* ``progression_audit`` — structural similarity, injected noise and lost
  detail of that progression, so an artefact-laden progression cannot pass
  silently.
* ``candidate_integrity`` — forensic screening of the sighting itself:
  spliced noise levels, JPEG grid phase, copy-move duplication and
  screen-replay / print-halftone peaks. A manipulated or recaptured image is
  the most dangerous false lead.

States use the cross-repo :class:`~omni_mercury_engine.verifiers.three_state.ThreeState`
wire values that FINDΩYOU™'s ``security.three_state`` mirrors. The packet is
serialised canonically (sorted keys, compact separators, ASCII) and signed
with ML-DSA-65 through AMA; :func:`verify_packet` checks the signature against
a **pinned** public key, because a key carried inside the packet proves
integrity, not origin.

This is an agent surface: importing it requires the AMA native backend (the
scoped PQC gate), like every other Mercury surface that signs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

# Operator decoupling: both switches must be on for this surface to exist.
# A refusal here is a named decision with its remedy, not an ImportError.
from omni_mercury_engine.decouple import require_enabled as _require_enabled

_require_enabled("findoyou", "the FINDΩYOU™ evidence bridge")
_require_enabled("ama_optional", "the FINDΩYOU™ evidence bridge (AMA-signed)")
del _require_enabled

from omni_mercury_engine.security.crypto_api import MLDSAProvider
from omni_mercury_engine.verifiers.three_state import ThreeState

EVIDENCE_SCHEMA = "mercury.findoyou.evidence/v1"
SIGNATURE_ALGORITHM = "ML-DSA-65"

#: Statements every packet carries, whatever the findings.
STANDING_LIMITATIONS: tuple[str, ...] = (
    "Image analysis supports human investigators; it is never an identification on its own.",
    "Age progression uses a published growth geometry with a stated, unfitted schedule; "
    "it is an investigative aid, not a prediction of appearance.",
    "Forensic screens use stated thresholds and can miss expert manipulation.",
)


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _image_digest(image: Any) -> str:
    """SHA3-256 of the image's canonical 8-bit pixels (path or array)."""
    from omni_mercury_engine.vision.imaging import to_canonical

    canonical = to_canonical(image)
    pixels = np.round(canonical.pixels * 255.0).astype(np.uint8)
    header = f"{pixels.shape}".encode()
    return hashlib.sha3_256(header + pixels.tobytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FindoyouEvidenceBridge:
    """Produce and verify signed missing-person image evidence for FINDΩYOU™.

    Args:
        signing_keypair: ``(public_key, secret_key)`` ML-DSA-65 bytes. When
            omitted an ephemeral key pair is generated; production deployments
            pass a provisioned key whose public half FINDΩYOU™ pins.
    """

    def __init__(self, signing_keypair: tuple[bytes, bytes] | None = None) -> None:
        """Create a bridge with the given (or an ephemeral) ML-DSA-65 key pair."""
        self._signer = MLDSAProvider()
        if signing_keypair is None:
            pair = self._signer.generate_keypair()
            self._public_key, self._secret_key = pair.public_key, pair.secret_key
        else:
            self._public_key, self._secret_key = signing_keypair

    @property
    def public_key_hex(self) -> str:
        """Return the hex public key FINDΩYOU™ must pin to verify this producer."""
        return self._public_key.hex()

    # ------------------------------------------------------------- findings
    @staticmethod
    def _identity_finding(match: dict[str, Any]) -> dict[str, Any]:
        score = match.get("match_score")
        backend = str(match.get("embedding_backend", "none"))
        if score is None:
            return {
                "state": ThreeState.UNAVAILABLE.value,
                "score": None,
                "backend": backend,
                "reason": str(match.get("error", "embedding unavailable")),
            }
        if backend == "deepface":
            return {
                "state": ThreeState.GROUNDED.value,
                "score": round(float(score), 6),
                "backend": backend,
                "reason": "face-identity embeddings (cosine similarity mapped to [0, 1])",
            }
        return {
            "state": ThreeState.UNAVAILABLE.value,
            "score": round(float(score), 6),
            "backend": backend,
            "reason": "appearance-level similarity only; identity is not verifiable without a "
            "face-identity embedding backend (install the 'face' extra)",
        }

    def _progression(
        self,
        reference: Any,
        source_age: float | None,
        years_elapsed: float | None,
    ) -> tuple[Any | None, dict[str, Any]]:
        from omni_mercury_engine.models.biometric_multimodal import AgeProgressionEngine

        if years_elapsed is None or years_elapsed <= 0:
            return None, {
                "state": ThreeState.UNAVAILABLE.value,
                "reason": (
                    "years_elapsed not provided"
                    if years_elapsed is None
                    else f"years_elapsed must be positive (got {years_elapsed})"
                ),
            }
        engine = AgeProgressionEngine(seed=0)
        if isinstance(reference, str):
            face = engine.detect_and_align_face(reference)
            if face is None:
                return None, {
                    "state": ThreeState.UNAVAILABLE.value,
                    "reason": "no face detected in the reference image",
                }
        else:
            face = np.asarray(reference)
        result = engine.progress_face(face, round(years_elapsed), source_age=source_age)
        audit = engine.assess_progression(result)
        return result.progressed_face, {
            "state": ThreeState.GROUNDED.value,
            "years": round(years_elapsed),
            "geometric_model": result.geometric_model,
            "texture_model": result.texture_model,
            "structural_similarity": audit.get("structural_similarity"),
            "noise_ratio": audit.get("noise_ratio"),
            "sharpness_ratio": audit.get("sharpness_ratio"),
            "flags": audit.get("flags", []),
        }

    @staticmethod
    def _integrity_finding(candidate: Any) -> dict[str, Any]:
        from omni_mercury_engine.models.biometric import BiometricAnomalyModel

        try:
            screen = BiometricAnomalyModel()._visual_assessment(
                BiometricAnomalyModel._resolve_image_path(candidate)
            )["visual_analysis"]
        except ValueError as exc:
            return {"state": ThreeState.UNAVAILABLE.value, "reason": str(exc)}
        if "forensics" not in screen:
            return {"state": ThreeState.UNAVAILABLE.value, "reason": screen.get("reason", "")}
        cues = list(screen.get("presentation_attack_cues", []))
        return {
            "state": ThreeState.GROUNDED.value,
            "manipulation_or_recapture_suspected": bool(cues),
            "cues": cues,
            "noise_inconsistency_max_abs_z": screen["forensics"]["noise_inconsistency"][
                "max_abs_z"
            ],
            "jpeg_grid": screen["forensics"]["jpeg_grid"],
            "periodic_peak_excess": screen["forensics"]["periodic_artifacts"]["peak_excess"],
            "capture_conditions": screen["environment"]["conditions"],
            "thresholds": screen.get("cue_thresholds"),
        }

    # ---------------------------------------------------------------- packet
    def analyze_case(
        self,
        case_id: str,
        reference: Any,
        candidate: Any,
        source_age: float | None = None,
        years_elapsed: float | None = None,
    ) -> dict[str, Any]:
        """Analyse a reference / candidate pair and return a signed evidence packet.

        Args:
            case_id: FINDΩYOU™ case identifier (opaque to Mercury).
            reference: Reference photo of the missing person (path or aligned
                face array).
            candidate: Candidate sighting (path or array).
            source_age: Age of the person in the reference photo, if known.
            years_elapsed: Years between the reference photo and the sighting.

        Returns:
            The packet, including its ``signature`` block.
        """
        from omni_mercury_engine.models.biometric import BiometricAnomalyModel

        model = BiometricAnomalyModel()
        match = model._predict_match(reference, candidate)
        identity = self._identity_finding(match)

        progressed, progression = self._progression(reference, source_age, years_elapsed)
        age_robust: dict[str, Any]
        if progressed is None:
            age_robust = {"state": ThreeState.UNAVAILABLE.value, "reason": progression["reason"]}
        else:
            robust_match = model._predict_match(progressed, candidate)
            robust = self._identity_finding(robust_match)
            age_robust = {
                **robust,
                "baseline_score": identity.get("score"),
                "reason": robust["reason"] + "; reference aged by the progression above",
            }

        packet: dict[str, Any] = {
            "schema": EVIDENCE_SCHEMA,
            "case_id": case_id,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "subject": {"source_age": source_age, "years_elapsed": years_elapsed},
            "image_digests": {
                "reference_sha3_256": _image_digest(
                    BiometricAnomalyModel._resolve_image_path(reference)
                ),
                "candidate_sha3_256": _image_digest(
                    BiometricAnomalyModel._resolve_image_path(candidate)
                ),
            },
            "findings": {
                "identity_match": identity,
                "age_robust_match": age_robust,
                "progression_audit": progression,
                "candidate_integrity": self._integrity_finding(candidate),
            },
            "limitations": list(STANDING_LIMITATIONS),
            "producer": {
                "system": "Mercury-Agent",
                "signature_algorithm": SIGNATURE_ALGORITHM,
                "cryptography": "AMA Cryptography (native ML-DSA-65)",
            },
        }
        packet = _json_safe(packet)
        body = _canonical(packet)
        packet["signature"] = {
            "algorithm": SIGNATURE_ALGORITHM,
            "public_key_hex": self.public_key_hex,
            "signed_digest_sha3_256": hashlib.sha3_256(body).hexdigest(),
            "value_hex": self._signer.sign(body, self._secret_key).hex(),
        }
        return packet


def verify_packet(packet: dict[str, Any], expected_public_key_hex: str) -> tuple[bool, str]:
    """Verify an evidence packet's ML-DSA-65 signature against a pinned producer key.

    Args:
        packet: A packet produced by :meth:`FindoyouEvidenceBridge.analyze_case`.
        expected_public_key_hex: The producer key FINDΩYOU™ has pinned, as hex
            (either case).

    Returns:
        ``(valid, reason)``.
    """
    signature = packet.get("signature")
    if not isinstance(signature, dict):
        return False, "packet carries no signature block"
    if packet.get("schema") != EVIDENCE_SCHEMA:
        return False, f"unsupported schema {packet.get('schema')!r}"
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return False, f"unsupported signature algorithm {signature.get('algorithm')!r}"
    pinned = expected_public_key_hex.strip().lower()
    if str(signature.get("public_key_hex", "")).lower() != pinned:
        return False, "signed by a key other than the pinned producer key"
    body = _canonical({k: v for k, v in packet.items() if k != "signature"})
    if hashlib.sha3_256(body).hexdigest() != signature.get("signed_digest_sha3_256"):
        return False, "packet contents changed after signing"
    try:
        valid = MLDSAProvider().verify(
            body, bytes.fromhex(signature["value_hex"]), bytes.fromhex(pinned)
        )
    except (ValueError, TypeError) as exc:
        return False, f"signature could not be checked: {exc}"
    return (True, "valid") if valid else (False, "signature does not verify")


__all__ = [
    "EVIDENCE_SCHEMA",
    "SIGNATURE_ALGORITHM",
    "STANDING_LIMITATIONS",
    "FindoyouEvidenceBridge",
    "verify_packet",
]

"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

σ_Immutable signed corpus (Wave B item 2).

Persists, signs, and verifies the labelled scalar-vector corpus that
trains the σ_Immutable EthicalGate.  The corpus and its signatures live
on disk next to the trained weights:

* ``sigma_immutable_corpus.json``     — the labelled corpus + provenance.
* ``sigma_immutable_corpus.sig.json`` — Ed25519 + ML-DSA-65 signatures
  produced via :class:`MercuryCrypto`.

On engine startup, :func:`verify_corpus_signatures` checks every present
signature.  A failed verification is propagated up to
:class:`SigmaImmutableGate.enforce`, which then turns every boundary
call into an ``EthicalConstraintViolationError(check="sigma_immutable")``
— signature failure is *not* a logger.warning.

The :func:`generate_corpus` and :func:`sign_and_persist_corpus`
helpers are exposed so ``scripts/train_sigma_immutable.py`` can
regenerate the corpus + signatures end-to-end deterministically from a
fixed seed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH: Path = _PACKAGE_ROOT / "security" / "sigma_immutable_corpus.json"
CORPUS_SIG_PATH: Path = _PACKAGE_ROOT / "security" / "sigma_immutable_corpus.sig.json"

# Corpus cardinality — kept small so the on-disk JSON stays auditable
# (one human can read it end-to-end) but large enough to span the
# positive/negative ethical region with shape-stable statistics.
CORPUS_POSITIVE: int = 64
CORPUS_NEGATIVE: int = 64
CORPUS_TOTAL: int = CORPUS_POSITIVE + CORPUS_NEGATIVE
CORPUS_INPUT_DIM: int = 256
CORPUS_USED_DIM: int = 180  # remaining 76 dims are zero-padding
CORPUS_ETHICAL_DIMS: int = 27
CORPUS_THRESHOLD: float = 0.93
CORPUS_SEED: int = 42

# Algorithms used for signing — both must succeed at signing time on a
# build with the AMA PQC C library available; at verification time, a
# missing PQC backend causes the ML-DSA-65 signature to be skipped with
# a warning (see "extras-gated PQC backend" in the rules).
ED25519_ALG: str = "ed25519"
MLDSA65_ALG: str = "ml-dsa-65"


class CorpusVerificationError(RuntimeError):
    """Raised when the σ_Immutable corpus signature verification fails.

    This is the exception type :class:`SigmaImmutableGate` re-raises as
    ``EthicalConstraintViolationError(check="sigma_immutable")`` at the
    next decision boundary.
    """


@dataclass(frozen=True)
class CorpusBundle:
    """Materialised corpus + provenance metadata.

    Attributes:
        features: ``(N, INPUT_DIM)`` float32 matrix used at training time.
        labels: ``(N,)`` float32 0/1 labels.
        sha3_256: SHA3-256 hex digest of the canonical corpus bytes.
        seed: Deterministic seed used to draw ``features`` and ``labels``.
        positive: Number of positive (ethical) samples in the corpus.
        negative: Number of negative (unethical) samples in the corpus.
        threshold: The σ_Immutable decision threshold the labels are
            calibrated to (0.93).
    """

    features: np.ndarray
    labels: np.ndarray
    sha3_256: str
    seed: int
    positive: int
    negative: int
    threshold: float


def generate_corpus(
    seed: int = CORPUS_SEED,
    n_positive: int = CORPUS_POSITIVE,
    n_negative: int = CORPUS_NEGATIVE,
    input_dim: int = CORPUS_INPUT_DIM,
    used_dim: int = CORPUS_USED_DIM,
    ethical_dims: int = CORPUS_ETHICAL_DIMS,
    threshold: float = CORPUS_THRESHOLD,
) -> CorpusBundle:
    """Generate the labelled corpus deterministically from ``seed``.

    The labelling rule mirrors ``scripts/train_sigma_immutable.py``:
    a sample is **positive** (label 1) when all 27 critical ethical
    scalars are at-or-above ``threshold`` and the remaining 153
    non-ethical dimensions are drawn from ``U[0, 2]``.  A sample is
    **negative** (label 0) when 1–5 critical ethical scalars are
    pushed below ``threshold``.

    Args:
        seed: Master seed for the NumPy generator.  The corpus is a
            pure function of this seed — re-running this script with
            the same seed produces a byte-identical corpus.
        n_positive: Number of positive samples to draw.
        n_negative: Number of negative samples to draw.
        input_dim: Width of each scalar vector (256 to match
            ``EthicalGate``'s input dim).
        used_dim: Number of meaningful columns; the remaining
            ``input_dim - used_dim`` columns are zero-padded.
        ethical_dims: Number of critical ethical columns at the front
            of each vector.
        threshold: Decision threshold the labels are calibrated to.

    Returns:
        :class:`CorpusBundle` with features, labels, and provenance.
    """
    rng = np.random.default_rng(seed)
    n_total = n_positive + n_negative

    features = np.zeros((n_total, input_dim), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.float32)
    critical_indices = list(range(ethical_dims))

    # Positives: ethical band U[threshold, 2.0] in critical dims, U[0, 2] elsewhere.
    for i in range(n_positive):
        features[i, :ethical_dims] = rng.uniform(
            threshold, 2.0, ethical_dims
        ).astype(np.float32)
        features[i, ethical_dims:used_dim] = rng.uniform(
            0.0, 2.0, used_dim - ethical_dims
        ).astype(np.float32)
        labels[i] = 1.0

    # Negatives: realistic background, then 1..5 critical dims pulled below threshold.
    for j in range(n_positive, n_total):
        features[j, :used_dim] = rng.uniform(0.0, 2.0, used_dim).astype(np.float32)
        n_violations = int(rng.integers(1, 6))
        violated = rng.choice(critical_indices, size=n_violations, replace=False)
        for idx in violated:
            features[j, idx] = float(rng.uniform(0.0, threshold - 0.01))
        labels[j] = 0.0

    # Deterministic shuffle so positives/negatives are interleaved at training.
    perm = rng.permutation(n_total)
    features = features[perm]
    labels = labels[perm]

    # Defensive copy + read-only view to enforce the trust boundary documented
    # for ``GOSNNCouplingServer.ingest`` and reused elsewhere in-tree.
    features = np.array(features, copy=True)
    labels = np.array(labels, copy=True)
    features.setflags(write=False)
    labels.setflags(write=False)

    canonical = _canonical_bytes(features, labels, seed=seed, threshold=threshold)
    sha3_256 = hashlib.sha3_256(canonical).hexdigest()

    return CorpusBundle(
        features=features,
        labels=labels,
        sha3_256=sha3_256,
        seed=seed,
        positive=n_positive,
        negative=n_negative,
        threshold=threshold,
    )


def _canonical_bytes(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    threshold: float,
) -> bytes:
    """Return the canonical serialisation that signatures cover.

    Stable across Python versions and machines because:

    * Floats are emitted via ``float.hex`` so quantisation is bit-exact.
    * Keys are sorted in the JSON object.
    * Trailing newline matches the on-disk file layout.

    Args:
        features: Corpus feature matrix.
        labels: Corpus label vector.
        seed: Master generation seed (must match the one stored in the
            persisted corpus).
        threshold: The σ_Immutable threshold the labels target.

    Returns:
        UTF-8 encoded canonical JSON bytes.
    """
    payload = {
        "schema": "sigma_immutable_corpus/v1",
        "seed": int(seed),
        "threshold": float(threshold),
        "input_dim": int(features.shape[1]),
        "n_samples": int(features.shape[0]),
        # ``float.hex`` is lossless and locale-free; ``round-trip`` via
        # ``float.fromhex`` reconstructs bit-exact values.
        "features_hex": [
            [float(v).hex() for v in row] for row in features.tolist()
        ],
        "labels_hex": [float(v).hex() for v in labels.tolist()],
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _serialise_corpus(bundle: CorpusBundle) -> bytes:
    """Serialise ``bundle`` to the canonical on-disk JSON bytes."""
    return _canonical_bytes(
        bundle.features,
        bundle.labels,
        seed=bundle.seed,
        threshold=bundle.threshold,
    )


def write_corpus(bundle: CorpusBundle, path: Path = CORPUS_PATH) -> bytes:
    """Persist ``bundle`` to ``path`` and return the bytes written.

    Args:
        bundle: Materialised corpus to write.
        path: Output JSON path (default in-tree corpus location).

    Returns:
        The exact byte sequence written to disk (so the caller can hand
        it straight to :func:`MercuryCrypto.sign`).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = _serialise_corpus(bundle)
    path.write_bytes(payload_bytes)
    logger.info(
        "σ_Immutable corpus written to %s (%d bytes, sha3-256=%s)",
        path,
        len(payload_bytes),
        bundle.sha3_256,
    )
    return payload_bytes


def sign_and_persist_corpus(
    bundle: CorpusBundle,
    corpus_path: Path = CORPUS_PATH,
    sig_path: Path = CORPUS_SIG_PATH,
) -> dict[str, Any]:
    """Sign ``bundle`` with Ed25519 + ML-DSA-65 and persist both files.

    Ed25519 signing is required (the classical baseline).  ML-DSA-65
    signing is attempted unconditionally; if the AMA PQC backend is not
    built into this environment, the helper records an explicit
    ``ml-dsa-65`` *omission* in the signature file and continues — the
    rules call out PQC as an "extras-gated PQC backend" so the absence
    of the C library at sign time is not fatal as long as the on-disk
    state honestly reports it.

    Args:
        bundle: Corpus to sign.
        corpus_path: Where the corpus JSON is persisted.
        sig_path: Where the signature JSON is persisted.

    Returns:
        The signature payload as a dict (mirrors what was written to
        ``sig_path``), useful for tests that want to inspect the
        algorithms used without re-reading the file.
    """
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        MercuryCrypto,
        SecurityLevel,
    )

    payload_bytes = write_corpus(bundle, corpus_path)

    # Use a CLASSICAL-default crypto so signing works on every build,
    # then opt into ML-DSA-65 explicitly so the failure mode is
    # narrow and recoverable when the PQC backend is missing.
    crypto = MercuryCrypto(security_level=SecurityLevel.CLASSICAL)

    ed_keypair = crypto.generate_signing_keypair(algorithm=AlgorithmType.ED25519)
    ed_signature = crypto.sign(
        payload_bytes,
        ed_keypair.secret_key,
        algorithm=AlgorithmType.ED25519,
    )

    signature_payload: dict[str, Any] = {
        "schema": "sigma_immutable_corpus_signatures/v1",
        "corpus_sha3_256": bundle.sha3_256,
        "signatures": {
            ED25519_ALG: {
                "algorithm": ED25519_ALG,
                "public_key_hex": ed_keypair.public_key.hex(),
                "signature_hex": ed_signature.signature.hex(),
            },
        },
    }

    try:
        mldsa_keypair = crypto.generate_signing_keypair(
            algorithm=AlgorithmType.ML_DSA_65,
        )
        mldsa_signature = crypto.sign(
            payload_bytes,
            mldsa_keypair.secret_key,
            algorithm=AlgorithmType.ML_DSA_65,
        )
        signature_payload["signatures"][MLDSA65_ALG] = {
            "algorithm": MLDSA65_ALG,
            "public_key_hex": mldsa_keypair.public_key.hex(),
            "signature_hex": mldsa_signature.signature.hex(),
        }
    except RuntimeError as exc:
        # AMA PQC native C library not built in this environment —
        # honestly record the omission so a future verifier with the
        # backend available knows the slot was intentionally empty.
        signature_payload["signatures"][MLDSA65_ALG] = {
            "algorithm": MLDSA65_ALG,
            "omitted": True,
            "omission_reason": str(exc),
        }
        logger.warning(
            "ML-DSA-65 signing skipped (AMA PQC backend missing): %s", exc
        )

    sig_bytes = (
        json.dumps(signature_payload, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    sig_path.write_bytes(sig_bytes)
    logger.info("σ_Immutable corpus signatures written to %s", sig_path)

    return signature_payload


def load_corpus_bytes(path: Path = CORPUS_PATH) -> bytes:
    """Read the on-disk corpus bytes for verification.

    Raises:
        CorpusVerificationError: When the file is missing or unreadable.
    """
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CorpusVerificationError(
            f"σ_Immutable corpus file unreadable at {path}: {exc}"
        ) from exc


def load_signature_payload(
    path: Path = CORPUS_SIG_PATH,
) -> dict[str, Any]:
    """Read the on-disk signature payload, validating the schema.

    Raises:
        CorpusVerificationError: When the signature file is missing,
            unreadable, malformed, or carries an unexpected schema tag.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CorpusVerificationError(
            f"σ_Immutable signature file unreadable at {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusVerificationError(
            f"σ_Immutable signature file is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict) or payload.get("schema") != (
        "sigma_immutable_corpus_signatures/v1"
    ):
        raise CorpusVerificationError(
            "σ_Immutable signature payload has unexpected or missing schema "
            "(want 'sigma_immutable_corpus_signatures/v1')."
        )

    signatures = payload.get("signatures")
    if not isinstance(signatures, dict) or ED25519_ALG not in signatures:
        raise CorpusVerificationError(
            "σ_Immutable signature payload is missing the mandatory "
            f"{ED25519_ALG!r} entry."
        )

    return payload


def verify_corpus_signatures(
    corpus_path: Path = CORPUS_PATH,
    sig_path: Path = CORPUS_SIG_PATH,
) -> dict[str, str]:
    """Verify all present signatures over the on-disk corpus.

    The Ed25519 signature is mandatory.  The ML-DSA-65 signature is
    verified iff the AMA PQC backend is available *and* the signature
    file actually contains one (i.e. it was not marked ``omitted``).
    Any present-but-failing signature raises
    :class:`CorpusVerificationError`.

    Args:
        corpus_path: Path to the corpus JSON to verify.
        sig_path: Path to the signature JSON to verify against.

    Returns:
        A mapping ``{algorithm: status}`` where ``status`` is one of
        ``"verified"``, ``"omitted"``, or ``"skipped_no_backend"`` so
        callers (and audit logs) can record what was actually checked.

    Raises:
        CorpusVerificationError: When the corpus's SHA3-256 does not
            match the manifest, or any present signature fails to verify.
    """
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        MercuryCrypto,
        Signature,
    )

    payload_bytes = load_corpus_bytes(corpus_path)
    sig_payload = load_signature_payload(sig_path)

    expected_digest = sig_payload.get("corpus_sha3_256")
    actual_digest = hashlib.sha3_256(payload_bytes).hexdigest()
    if expected_digest != actual_digest:
        raise CorpusVerificationError(
            "σ_Immutable corpus SHA3-256 mismatch: "
            f"manifest={expected_digest!r}, on-disk={actual_digest!r}. "
            "Corpus has been edited or truncated since signing."
        )

    crypto = MercuryCrypto()
    statuses: dict[str, str] = {}
    signatures = sig_payload["signatures"]

    # ----- Ed25519 (mandatory) -----
    ed = signatures[ED25519_ALG]
    if ed.get("omitted"):
        raise CorpusVerificationError(
            "Ed25519 signature is marked omitted; the classical signature "
            "is mandatory and must be present."
        )
    ok = crypto.verify(
        payload_bytes,
        Signature(
            signature=bytes.fromhex(ed["signature_hex"]),
            algorithm=AlgorithmType.ED25519,
            public_key_hash=hashlib.sha3_256(
                bytes.fromhex(ed["public_key_hex"])
            ).hexdigest()[:16],
        ),
        bytes.fromhex(ed["public_key_hex"]),
    )
    if not ok:
        raise CorpusVerificationError(
            "Ed25519 signature verification failed for σ_Immutable corpus."
        )
    statuses[ED25519_ALG] = "verified"

    # ----- ML-DSA-65 (extras-gated PQC) -----
    mldsa = signatures.get(MLDSA65_ALG)
    if mldsa is None:
        raise CorpusVerificationError(
            f"σ_Immutable signature payload is missing the {MLDSA65_ALG!r} entry."
        )
    if mldsa.get("omitted"):
        statuses[MLDSA65_ALG] = "omitted"
    else:
        from omni_mercury_engine.security.pqc_backends import DILITHIUM_AVAILABLE

        if not DILITHIUM_AVAILABLE:
            statuses[MLDSA65_ALG] = "skipped_no_backend"
            logger.warning(
                "ML-DSA-65 signature present in corpus manifest but the "
                "AMA PQC backend is not built — skipping verification."
            )
        else:
            ok = crypto.verify(
                payload_bytes,
                Signature(
                    signature=bytes.fromhex(mldsa["signature_hex"]),
                    algorithm=AlgorithmType.ML_DSA_65,
                    public_key_hash=hashlib.sha3_256(
                        bytes.fromhex(mldsa["public_key_hex"])
                    ).hexdigest()[:16],
                ),
                bytes.fromhex(mldsa["public_key_hex"]),
            )
            if not ok:
                raise CorpusVerificationError(
                    "ML-DSA-65 signature verification failed for σ_Immutable corpus."
                )
            statuses[MLDSA65_ALG] = "verified"

    return statuses


def parse_corpus(payload_bytes: bytes) -> CorpusBundle:
    """Parse the on-disk corpus bytes back into a :class:`CorpusBundle`.

    The features and labels are reconstructed bit-exactly from the
    ``float.hex`` strings persisted by :func:`_canonical_bytes`.

    Args:
        payload_bytes: Raw bytes read from ``sigma_immutable_corpus.json``.

    Returns:
        :class:`CorpusBundle` rehydrated from the on-disk state.
    """
    payload = json.loads(payload_bytes)
    features = np.array(
        [[float.fromhex(v) for v in row] for row in payload["features_hex"]],
        dtype=np.float32,
    )
    labels = np.array(
        [float.fromhex(v) for v in payload["labels_hex"]],
        dtype=np.float32,
    )
    features = np.array(features, copy=True)
    labels = np.array(labels, copy=True)
    features.setflags(write=False)
    labels.setflags(write=False)

    return CorpusBundle(
        features=features,
        labels=labels,
        sha3_256=hashlib.sha3_256(payload_bytes).hexdigest(),
        seed=int(payload["seed"]),
        positive=int(np.sum(labels >= 0.5)),
        negative=int(np.sum(labels < 0.5)),
        threshold=float(payload["threshold"]),
    )


__all__ = [
    "CORPUS_INPUT_DIM",
    "CORPUS_NEGATIVE",
    "CORPUS_PATH",
    "CORPUS_POSITIVE",
    "CORPUS_SEED",
    "CORPUS_SIG_PATH",
    "CORPUS_THRESHOLD",
    "CORPUS_TOTAL",
    "CORPUS_USED_DIM",
    "ED25519_ALG",
    "MLDSA65_ALG",
    "CorpusBundle",
    "CorpusVerificationError",
    "generate_corpus",
    "load_corpus_bytes",
    "load_signature_payload",
    "parse_corpus",
    "sign_and_persist_corpus",
    "verify_corpus_signatures",
    "write_corpus",
]

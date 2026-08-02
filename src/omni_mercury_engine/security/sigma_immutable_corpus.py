# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""σ_Immutable signed corpus (Wave B item 2).

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

Signature semantics — tamper-evident, not authenticated
-------------------------------------------------------

The corpus carries Ed25519 and ML-DSA-65 signatures, and
:func:`verify_corpus_signature` refuses to proceed unless both verify. What that
buys is **tamper evidence**: accidental corruption, truncation and silent drift
are caught, and every boundary that consults the gate is poisoned uniformly when
they are.

It is **not** authentication of origin. The verifying public key travels inside
the same signature payload it verifies (``signatures[alg]["public_key_hex"]``),
so an attacker who can rewrite ``sigma_immutable_corpus.json`` can also replace
the key and re-sign. Authenticating origin would require pinning the public key
out of band — in the package, in a release manifest checked against a trust
anchor, or via a transparency log.

Do not describe this corpus as "signed for authenticity". The honest statement
is in ``CAPABILITY_MATRIX.md``: tamper-evident, not authenticated.
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
# The harvested intact-config baseline (produced by
# ``scripts/harvest_sigma_baseline.py``).  The corpus positives are built from
# this REAL operational vector plus intact-preserving variations, and the
# negatives from real tamper mutations of it — so the trained σ_Immutable
# network recognises the actual production configuration (never a DoS) and
# agrees with the deterministic critical-ethical floor on real anchor collapse.
BASELINE_PATH: Path = _PACKAGE_ROOT / "security" / "sigma_immutable_baseline.json"

# Corpus cardinality — kept small so the on-disk JSON stays auditable
# (one human can read it end-to-end) but large enough to span the
# positive/negative ethical region with shape-stable statistics.
CORPUS_POSITIVE: int = 64
CORPUS_NEGATIVE: int = 64
CORPUS_TOTAL: int = CORPUS_POSITIVE + CORPUS_NEGATIVE

# Layout constants are owned by :mod:`sigma_immutable_gate` so the
# trainer, the corpus, and every decision boundary share a single
# definition.  See the module-level "σ_Immutable layout constants"
# block in ``sigma_immutable_gate.py`` for the canonical layout.
from omni_mercury_engine.security.sigma_immutable_gate import (
    SIGMA_ETHICAL_BAND_END as _SIGMA_ETHICAL_BAND_END,
    SIGMA_IMMUTABLE_DIM as _SIGMA_IMMUTABLE_DIM,
    SIGMA_USED_BAND_END as _SIGMA_USED_BAND_END,
)

CORPUS_INPUT_DIM: int = _SIGMA_IMMUTABLE_DIM
CORPUS_USED_DIM: int = _SIGMA_USED_BAND_END  # remaining dims are zero-padding
CORPUS_ETHICAL_DIMS: int = _SIGMA_ETHICAL_BAND_END
CORPUS_THRESHOLD: float = 0.93
CORPUS_SEED: int = 42

# Algorithms used for signing — both must succeed; the ML-DSA-65 signature is
# mandatory because Mercury requires AMA/PQC at package import.
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

    features: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    sha3_256: str
    seed: int
    positive: int
    negative: int
    threshold: float


@dataclass(frozen=True)
class Baseline:
    """The harvested intact-config reference the corpus is built around.

    Attributes:
        names: Ordered operational scalar names (length ``n_operational``).
        values: ``(n_operational,)`` float64 intact values.
        anchor_idx: Indices (into the ethical band) of the critical
            ethical *anchors* — the subset the deterministic floor guards.
        narrative_idx: Indices of the ethical band's narrative-tuning
            scalars (excluded from the anchors; legitimately low).
        ethical_dims: Width of the ethical band (27).
        used_dim: Last actively-used index (180); the tail is zero.
        input_dim: Padded vector width (256).
    """

    names: list[str]
    values: np.ndarray[Any, Any]
    anchor_idx: np.ndarray[Any, Any]
    narrative_idx: np.ndarray[Any, Any]
    ethical_dims: int
    used_dim: int
    input_dim: int


def load_baseline(path: Path = BASELINE_PATH) -> Baseline:
    """Load the harvested intact-config baseline artifact.

    Raises:
        FileNotFoundError: When the baseline has not been harvested yet
            (run ``scripts/harvest_sigma_baseline.py`` first).
    """
    payload = json.loads(path.read_text())
    if payload.get("schema") != "sigma_immutable_baseline/v1":
        raise ValueError(f"unexpected σ baseline schema in {path}: {payload.get('schema')!r}")
    names = list(payload["names"])
    values = np.array([float.fromhex(v) for v in payload["values_hex"]], dtype=np.float64)
    ethical_dims = int(payload["ethical_band_end"])
    anchor_names = set(payload["anchor_names"])
    anchor_idx = np.array(
        [i for i in range(ethical_dims) if names[i] in anchor_names], dtype=np.int64
    )
    narrative_idx = np.array(
        [i for i in range(ethical_dims) if names[i] not in anchor_names], dtype=np.int64
    )
    return Baseline(
        names=names,
        values=values,
        anchor_idx=anchor_idx,
        narrative_idx=narrative_idx,
        ethical_dims=ethical_dims,
        used_dim=int(payload["used_band_end"]),
        input_dim=int(payload["input_dim"]),
    )


def build_integrity_samples(
    baseline: Baseline,
    seed: int,
    n_positive: int,
    n_negative: int,
    threshold: float = CORPUS_THRESHOLD,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Build anchor-faithful integrity samples around the real baseline.

    Positives (label 1, *intact*): the exact harvested baseline is always
    sample 0 (so the trained network passes the real production vector by
    construction — no DoS), followed by intact variations that hold every
    critical ethical **anchor** at-or-above ``threshold`` while letting the
    narrative-tuning scalars and the non-ethical operational band vary
    across their realistic ranges.  This teaches "all anchors high → intact"
    — exactly the deterministic floor's contract — rather than the earlier
    synthetic rule that mislabelled the real (narrative-low) config.

    Negatives (label 0, *tampered*): real corruptions of an otherwise-intact
    draw — anchor collapse (1..5 anchors pulled below ``threshold``, the
    tamper the floor catches), plus a minority of gross-outlier and
    band-corruption mutations (the F10 leak class).  The reserved tail
    ``[used_dim, input_dim)`` stays zero on every sample by contract.
    """
    rng = np.random.default_rng(seed)
    ed = baseline.ethical_dims
    used = baseline.used_dim
    n_op = len(baseline.values)
    base_f32 = baseline.values.astype(np.float32)

    def _draw(collapse: int) -> np.ndarray[Any, Any]:
        """One draw with the SAME non-anchor distribution for pos and neg.

        The 24 critical ethical *anchors* carry the entire label signal: a
        positive holds all anchors at-or-above ``threshold``; a negative pulls
        ``collapse`` of them below it (the real tamper the deterministic floor
        catches).  Everything else — the 3 narrative-tuning scalars and the
        non-ethical band ``[ethical_dims, used_dim)`` — is drawn identically
        from ``U[0, 2]`` in both classes, so the network learns to key on the
        anchors alone and ignore the non-discriminative dimensions (this is
        what lets a generic all-anchors-high vector pass while the real
        narrative-low baseline also passes).  The reserved tail
        ``[used_dim, input_dim)`` stays zero by contract; gross-magnitude and
        non-finite corruption is caught deterministically upstream, never here.
        """
        v = np.zeros(baseline.input_dim, dtype=np.float32)
        v[ed:used] = rng.uniform(0.0, 2.0, used - ed).astype(np.float32)
        v[baseline.narrative_idx] = rng.uniform(0.0, 2.0, len(baseline.narrative_idx)).astype(
            np.float32
        )
        v[baseline.anchor_idx] = rng.uniform(threshold, 2.0, len(baseline.anchor_idx)).astype(
            np.float32
        )
        if collapse:
            for idx in rng.choice(baseline.anchor_idx, size=collapse, replace=False):
                v[int(idx)] = float(rng.uniform(0.0, threshold - 0.01))
        return v

    n_total = n_positive + n_negative
    features = np.zeros((n_total, baseline.input_dim), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.float32)

    # Positive 0 is the exact real baseline (the DoS guarantee); a fifth of the
    # positives cluster tightly around it so the real production configuration
    # is firmly recognised even though it sits at the narrative-low edge.
    features[0, :n_op] = base_f32
    labels[0] = 1.0
    for i in range(1, n_positive):
        if i % 5 == 0:
            jitter = rng.normal(0.0, 0.02, n_op).astype(np.float32)
            features[i, :n_op] = np.clip(base_f32 + jitter, 0.0, 2.0)
        else:
            features[i] = _draw(collapse=0)
        labels[i] = 1.0

    max_collapse = min(6, len(baseline.anchor_idx) + 1)
    for j in range(n_positive, n_total):
        features[j] = _draw(collapse=int(rng.integers(1, max_collapse)))
        labels[j] = 0.0

    return features, labels


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

    Positives are the harvested intact configuration (``sample 0`` is the
    exact real baseline) plus intact variations that hold the critical
    ethical anchors at-or-above ``threshold``; negatives are real tamper
    mutations (anchor collapse, gross-outlier leak, band corruption).  See
    :func:`build_integrity_samples`.  The corpus is a pure function of
    ``seed`` and the committed baseline, so re-running with the same seed
    writes a byte-identical ``sigma_immutable_corpus.json``.

    Args:
        seed: Master seed for the NumPy generator.
        n_positive: Number of positive (intact) samples to draw.
        n_negative: Number of negative (tampered) samples to draw.
        input_dim: Width of each scalar vector (256 to match
            ``EthicalGate``'s input dim).
        used_dim: Number of meaningful columns; the remaining
            ``input_dim - used_dim`` columns are zero-padded.
        ethical_dims: Number of ethical columns at the front of each vector.
        threshold: Decision threshold the labels are calibrated to.

    Returns:
        :class:`CorpusBundle` with features, labels, and provenance.
    """
    baseline = load_baseline()
    if (baseline.input_dim, baseline.used_dim, baseline.ethical_dims) != (
        input_dim,
        used_dim,
        ethical_dims,
    ):
        raise ValueError(
            "σ baseline layout does not match the corpus layout constants; "
            "re-harvest with scripts/harvest_sigma_baseline.py."
        )
    n_total = n_positive + n_negative
    features, labels = build_integrity_samples(
        baseline, seed=seed, n_positive=n_positive, n_negative=n_negative, threshold=threshold
    )

    # Deterministic shuffle so positives/negatives are interleaved at training.
    rng = np.random.default_rng(seed + 1)
    perm = rng.permutation(n_total)
    shuffled_features: np.ndarray[Any, Any] = np.empty(features.shape, dtype=np.float32)
    shuffled_features[:] = features[perm]
    shuffled_labels: np.ndarray[Any, Any] = np.empty(labels.shape, dtype=np.float32)
    shuffled_labels[:] = labels[perm]

    # Defensive copy + read-only view to enforce the trust boundary documented
    # for ``GOSNNCouplingServer.ingest`` and reused elsewhere in-tree.
    features = np.empty(shuffled_features.shape, dtype=np.float32)
    features[:] = shuffled_features
    labels = np.empty(shuffled_labels.shape, dtype=np.float32)
    labels[:] = shuffled_labels
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
    features: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
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
        "features_hex": [[float(v).hex() for v in row] for row in features.tolist()],
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

    Ed25519 signing is required as the classical baseline.  ML-DSA-65
    signing is also required; Mercury cannot generate or persist a corpus
    signature manifest without the AMA/PQC backend.

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

    sig_bytes = (json.dumps(signature_payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
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
            f"σ_Immutable signature payload is missing the mandatory {ED25519_ALG!r} entry."
        )

    return payload


def verify_corpus_signatures(
    corpus_path: Path = CORPUS_PATH,
    sig_path: Path = CORPUS_SIG_PATH,
) -> dict[str, str]:
    """Verify all present signatures over the on-disk corpus.

    The Ed25519 and ML-DSA-65 signatures are mandatory.  Any missing,
    omitted, or failing signature raises :class:`CorpusVerificationError`.

    Args:
        corpus_path: Path to the corpus JSON to verify.
        sig_path: Path to the signature JSON to verify against.

    Returns:
        A mapping ``{algorithm: "verified"}`` for the mandatory
        signatures.

    Raises:
        CorpusVerificationError: When the corpus's SHA3-256 does not
            match the manifest, or any present signature fails to verify.
    """
    from omni_mercury_engine.security.crypto_api import (
        AlgorithmType,
        MercuryCrypto,
        Signature,
        key_fingerprint,
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
            public_key_hash=key_fingerprint(bytes.fromhex(ed["public_key_hex"])),
        ),
        bytes.fromhex(ed["public_key_hex"]),
    )
    if not ok:
        raise CorpusVerificationError(
            "Ed25519 signature verification failed for σ_Immutable corpus."
        )
    statuses[ED25519_ALG] = "verified"

    # ----- ML-DSA-65 (mandatory PQC) -----
    mldsa = signatures.get(MLDSA65_ALG)
    if mldsa is None:
        raise CorpusVerificationError(
            f"σ_Immutable signature payload is missing the {MLDSA65_ALG!r} entry."
        )
    if mldsa.get("omitted"):
        raise CorpusVerificationError(
            "ML-DSA-65 signature is marked omitted; AMA/PQC signing is mandatory."
        )
    ok = crypto.verify(
        payload_bytes,
        Signature(
            signature=bytes.fromhex(mldsa["signature_hex"]),
            algorithm=AlgorithmType.ML_DSA_65,
            public_key_hash=key_fingerprint(bytes.fromhex(mldsa["public_key_hex"])),
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
    # The arrays are constructed locally just above, so they can be
    # frozen directly — no defensive re-copy needed (a caller never
    # holds a reference to them).
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
    "BASELINE_PATH",
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
    "Baseline",
    "CorpusBundle",
    "CorpusVerificationError",
    "build_integrity_samples",
    "generate_corpus",
    "load_baseline",
    "load_corpus_bytes",
    "load_signature_payload",
    "parse_corpus",
    "sign_and_persist_corpus",
    "verify_corpus_signatures",
    "write_corpus",
]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Semantic pinning tests for the σ_Immutable signed-corpus module.

Written to kill the mutation survivors reported by
``scripts/run_sigma_mutation_gate.py`` for
``security/sigma_immutable_corpus.py``: the σ-focused subset exercised
the gate around the corpus but never pinned the corpus module's own
arithmetic, so constant tweaks, ``Add``/``Sub`` swaps, ``Eq``/``NotEq``
swaps, and bool flips across the synthetic-corpus generation all
survived.  Every assertion below is a closed-form contract value with
the mutation class it kills noted inline.  The headline oracle is the
module's own documented determinism contract: ``generate_corpus()`` with
the default seed must reproduce the committed
``sigma_immutable_corpus.json`` byte-for-byte, so *any* mutation that
perturbs an RNG draw, a label, a shuffle, or the canonical serialisation
changes the bytes and fails the pin.

Covers:
- Corpus cardinality/threshold/seed constants (lines 52-70): the 64+64
  composition, the 0.93 calibration threshold, and master seed 42
- ``CorpusBundle`` / ``Baseline`` frozenness (the provenance records are
  immutable by contract)
- ``load_baseline``: committed-schema acceptance, wrong-schema raise,
  and the 127-scalar / 24-anchor / [3, 14, 15]-narrative layout
- ``build_integrity_samples``: byte-exact regeneration via
  ``generate_corpus``, plus synthetic-baseline probes that pin the
  ``np.clip(…, 0.0, 2.0)`` bounds and the ``min(6, n_anchors + 1)``
  collapse arithmetic that are unobservable on the real baseline
- ``generate_corpus``: SHA3-256 digest pin (agrees with the committed
  signature manifest), shapes/dtypes, label composition, the reserved
  zero tail, the exact-baseline-row DoS guarantee, read-only outputs,
  and the layout-mismatch raise path
- ``_canonical_bytes`` / ``parse_corpus``: canonical field values,
  sorted keys, trailing newline, bit-exact round-trip, read-only parsed
  arrays, and the 0.5 label-count boundary
- ``write_corpus`` / ``sign_and_persist_corpus``: nested-directory
  creation (``parents``/``exist_ok``), canonical signature-file bytes,
  dual-algorithm manifest structure, and a sign→verify round trip
- ``load_signature_payload`` / ``verify_corpus_signatures``: committed
  artifacts load and verify clean, and every validation raise path
  (schema, non-dict, missing entry, unreadable, invalid JSON, digest
  mismatch, corrupted signature) raises ``CorpusVerificationError``
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.security.sigma_immutable_corpus import (
    CORPUS_ETHICAL_DIMS,
    CORPUS_INPUT_DIM,
    CORPUS_NEGATIVE,
    CORPUS_PATH,
    CORPUS_POSITIVE,
    CORPUS_SEED,
    CORPUS_SIG_PATH,
    CORPUS_THRESHOLD,
    CORPUS_TOTAL,
    CORPUS_USED_DIM,
    ED25519_ALG,
    MLDSA65_ALG,
    Baseline,
    CorpusBundle,
    CorpusVerificationError,
    _canonical_bytes,
    _serialise_corpus,
    build_integrity_samples,
    generate_corpus,
    load_baseline,
    load_corpus_bytes,
    load_signature_payload,
    parse_corpus,
    sign_and_persist_corpus,
    verify_corpus_signatures,
    write_corpus,
)

if TYPE_CHECKING:
    from pathlib import Path

# SHA3-256 of the canonical corpus bytes produced by the real
# ``generate_corpus()`` with the default seed (42) — computed by calling
# the real code once and pinned.  It matches both the committed
# ``sigma_immutable_corpus.json`` and the ``corpus_sha3_256`` field of
# the committed signature manifest, so all three views of the corpus
# must stay in lock-step.
PINNED_CORPUS_SHA3_256 = "4c9eb2f1557db437bea7a575635f44de903b15616f6e759aaacb7ea4e5dba10a"

# Anchor cutoff used when counting collapsed anchors: intact anchors are
# drawn from U[0.93, 2.0) (≥ 0.93 up to float32 rounding), collapsed
# anchors from U[0.0, 0.92) — 0.925 separates the two bands robustly.
_ANCHOR_CUTOFF = 0.925


@pytest.fixture(scope="module")
def bundle() -> CorpusBundle:
    """The deterministically regenerated corpus (real code, default seed)."""
    return generate_corpus()


@pytest.fixture(scope="module")
def baseline() -> Baseline:
    """The committed harvested intact-config baseline."""
    return load_baseline()


@pytest.fixture(scope="module")
def tiny_bundle() -> CorpusBundle:
    """A 2-sample bundle for cheap persistence/signing tests."""
    features = np.array([[0.1, 0.2, 0.3], [1.0, 1.5, 2.0]], dtype=np.float32)
    labels = np.array([1.0, 0.0], dtype=np.float32)
    return parse_corpus(_canonical_bytes(features, labels, seed=5, threshold=0.93))


def _synthetic_baseline() -> Baseline:
    """A minimal baseline exposing the clip/collapse edges.

    Value 0.0 at anchor 0 and 2.0 at anchor 1 make both ``np.clip``
    bounds bind in the jitter cluster (the real baseline spans only
    [0.1, 1.618], so neither bound ever binds there), and 3 anchors make
    ``max_collapse = min(6, n_anchors + 1)`` sensitive to its ``+ 1``
    arithmetic (with the real 24 anchors ``min`` saturates at 6 for
    ``n ± 1``, hiding those mutants).
    """
    return Baseline(
        names=[f"s{i}" for i in range(8)],
        values=np.array([0.0, 2.0, 1.0, 1.5, 0.5, 1.0, 1.0, 1.0], dtype=np.float64),
        anchor_idx=np.array([0, 1, 2], dtype=np.int64),
        narrative_idx=np.array([3], dtype=np.int64),
        ethical_dims=4,
        used_dim=8,
        input_dim=12,
    )


# =============================================================================
# Module constants — the corpus composition contract (lines 52-70)
# =============================================================================


class TestCorpusConstants:
    """The audited corpus composition is 64+64 at threshold 0.93, seed 42."""

    def test_cardinality(self) -> None:
        # Kills the int tweaks at lines 52/53 (64 -> 65) and the
        # Add -> Sub at line 54 (128 -> 0): the 64/64 split is the
        # documented auditable corpus size.
        assert CORPUS_POSITIVE == 64
        assert CORPUS_NEGATIVE == 64
        assert CORPUS_TOTAL == 128
        assert CORPUS_TOTAL == CORPUS_POSITIVE + CORPUS_NEGATIVE

    def test_threshold_and_seed(self) -> None:
        # Kills the float tweak at line 69 (0.93 -> 1.03) and the int
        # tweak at line 70 (42 -> 43): 0.93 is the trained calibration
        # threshold and 42 the master generation seed the committed
        # corpus was drawn from.
        assert CORPUS_THRESHOLD == 0.93
        assert CORPUS_SEED == 42

    def test_layout_reexports(self) -> None:
        # Cross-module contract: the corpus reuses the gate's canonical
        # 256/180/27 layout verbatim.
        assert CORPUS_INPUT_DIM == 256
        assert CORPUS_USED_DIM == 180
        assert CORPUS_ETHICAL_DIMS == 27

    def test_algorithm_tags(self) -> None:
        assert ED25519_ALG == "ed25519"
        assert MLDSA65_ALG == "ml-dsa-65"


# =============================================================================
# Frozen provenance records (lines 87 / 111)
# =============================================================================


class TestFrozenProvenance:
    """Corpus provenance is immutable — ``frozen=True`` is the contract."""

    def test_corpus_bundle_is_frozen(self, bundle: CorpusBundle) -> None:
        # Kills the bool flip at line 87 (frozen=True -> False): a
        # mutable provenance record would let post-hoc edits desync the
        # digest from the features it attests to.
        with pytest.raises(dataclasses.FrozenInstanceError):
            bundle.seed = 1  # type: ignore[misc]

    def test_baseline_is_frozen(self) -> None:
        # Kills the bool flip at line 111 (frozen=True -> False).
        synthetic = _synthetic_baseline()
        with pytest.raises(dataclasses.FrozenInstanceError):
            synthetic.used_dim = 9  # type: ignore[misc]


# =============================================================================
# load_baseline — schema validation + harvested layout (line 144)
# =============================================================================


class TestLoadBaseline:
    """The committed baseline loads clean; a wrong schema is refused."""

    def test_committed_baseline_loads_with_expected_layout(self, baseline: Baseline) -> None:
        # Kills NotEq -> Eq at line 144: with ``==`` the VALID committed
        # schema tag would raise and startup would DoS.
        assert baseline.input_dim == 256
        assert baseline.used_dim == 180
        assert baseline.ethical_dims == 27
        assert len(baseline.names) == 127
        assert baseline.values.shape == (127,)
        # 24 critical anchors + 3 narrative-tuning scalars partition the
        # ethical band exactly.
        assert len(baseline.anchor_idx) == 24
        assert baseline.narrative_idx.tolist() == [3, 14, 15]
        assert sorted(baseline.anchor_idx.tolist() + baseline.narrative_idx.tolist()) == list(
            range(27)
        )
        # Every anchor sits at-or-above the corpus threshold in the
        # intact configuration (the corpus positives depend on this).
        assert float(baseline.values[baseline.anchor_idx].min()) >= CORPUS_THRESHOLD

    def test_wrong_schema_raises(self, tmp_path: Path) -> None:
        # Pins the raise direction of the line-144 check.
        bogus = tmp_path / "baseline.json"
        bogus.write_text(json.dumps({"schema": "bogus/v0"}))
        with pytest.raises(ValueError, match="unexpected σ baseline schema"):
            load_baseline(bogus)


# =============================================================================
# generate_corpus — the byte-exact determinism contract (lines 213-326)
# =============================================================================


class TestGenerateCorpusByteContract:
    """``generate_corpus()`` must reproduce the committed corpus exactly."""

    def test_regeneration_is_byte_identical_to_committed_corpus(self, bundle: CorpusBundle) -> None:
        # THE closed-form oracle for the dense line-213..299 survivor
        # cluster: the module documents that re-running with the same
        # seed writes a byte-identical corpus JSON.  Any float/int
        # constant tweak, Add<->Sub, Eq<->NotEq, or bool flip that
        # touches an RNG draw (uniform bounds, jitter sigma, collapse
        # bounds, cluster stride ``i % 5 == 0``, ``replace=False``), a
        # label value, the shuffle seed ``seed + 1``, or the canonical
        # serialisation changes these bytes and fails here.
        assert _serialise_corpus(bundle) == CORPUS_PATH.read_bytes()

    def test_sha3_digest_pin_agrees_with_signature_manifest(self, bundle: CorpusBundle) -> None:
        # Same oracle as above, pinned as an explicit digest and
        # triangulated against the committed signature manifest.
        assert bundle.sha3_256 == PINNED_CORPUS_SHA3_256
        manifest: dict[str, Any] = json.loads(CORPUS_SIG_PATH.read_text())
        assert manifest["corpus_sha3_256"] == PINNED_CORPUS_SHA3_256

    def test_shapes_labels_and_provenance(self, bundle: CorpusBundle) -> None:
        # Kills cardinality tweaks flowing through generate_corpus
        # (n_positive/n_negative defaults) and the label-constant tweaks
        # at lines 233/240/245 (1.0/0.0 are the only legal labels).
        assert bundle.features.shape == (128, 256)
        assert bundle.features.dtype == np.float32
        assert bundle.labels.shape == (128,)
        assert bundle.labels.dtype == np.float32
        assert set(np.unique(bundle.labels).tolist()) == {0.0, 1.0}
        assert float(bundle.labels.sum()) == 64.0
        assert bundle.positive == 64
        assert bundle.negative == 64
        assert bundle.seed == 42
        assert bundle.threshold == 0.93

    def test_outputs_are_read_only(self, bundle: CorpusBundle) -> None:
        # Kills the bool flips at lines 312/313
        # (setflags(write=False) -> write=True): the trust boundary
        # requires immutable corpus views.
        assert bundle.features.flags.writeable is False
        assert bundle.labels.flags.writeable is False

    def test_reserved_tail_is_zero(self, bundle: CorpusBundle) -> None:
        # The [used_dim, input_dim) tail stays zero on every sample by
        # contract.
        assert not bundle.features[:, CORPUS_USED_DIM:].any()

    def test_exact_baseline_row_present_exactly_once(
        self, bundle: CorpusBundle, baseline: Baseline
    ) -> None:
        # The DoS guarantee: positive 0 is the exact harvested baseline
        # (line 232 ``features[0, :n_op]``); after the deterministic
        # seed+1 shuffle it lands at row 110.  Kills the int tweak at
        # line 232 (0 -> 1: the loop would overwrite the baseline row)
        # and pins the shuffle permutation.
        padded = np.zeros(bundle.features.shape[1], dtype=np.float32)
        padded[: len(baseline.values)] = baseline.values.astype(np.float32)
        matches = np.where((bundle.features == padded).all(axis=1))[0]
        assert matches.tolist() == [110]
        assert bundle.labels[110] == 1.0

    def test_negative_collapse_profile(self, bundle: CorpusBundle, baseline: Baseline) -> None:
        # Kills the collapse-bound tweaks at lines 242/244 on the real
        # path: negatives collapse 1..5 anchors (never 0, never 6+) with
        # this exact seeded histogram; positives collapse none.
        negatives = bundle.features[bundle.labels < 0.5]
        positives = bundle.features[bundle.labels >= 0.5]
        neg_counts = (negatives[:, baseline.anchor_idx] < _ANCHOR_CUTOFF).sum(axis=1)
        pos_counts = (positives[:, baseline.anchor_idx] < _ANCHOR_CUTOFF).sum(axis=1)
        assert np.bincount(neg_counts, minlength=6).tolist() == [0, 7, 14, 10, 16, 17]
        assert int(pos_counts.max()) == 0

    def test_layout_mismatch_raises(self) -> None:
        # Kills NotEq -> Eq at line 284 in the direction the committed
        # baseline cannot: a corpus/baseline layout mismatch MUST raise
        # (the matching-layout direction is covered by every other test
        # in this class not raising).
        with pytest.raises(ValueError, match="does not match the corpus layout"):
            generate_corpus(input_dim=CORPUS_INPUT_DIM + 1)


# =============================================================================
# build_integrity_samples — edges unobservable on the real baseline
# =============================================================================


class TestBuildIntegritySamplesSynthetic:
    """Synthetic baselines expose the clip bounds and collapse arithmetic."""

    def test_jitter_cluster_clips_to_exact_bounds(self) -> None:
        # Kills the float tweaks on ``np.clip(base + jitter, 0.0, 2.0)``
        # at line 237: with baseline values 0.0/2.0 the clip bounds bind
        # in the i % 5 == 0 jitter cluster, so exact 0.0s and an exact
        # 2.0 must appear (a 0.1/2.1 bound leaves none) and nothing may
        # exceed 2.0.  The real baseline spans [0.1, 1.618] where the
        # bounds never bind, so only this synthetic probe can see them.
        features, labels = build_integrity_samples(
            _synthetic_baseline(), seed=7, n_positive=32, n_negative=32
        )
        cluster_rows = [i for i in range(1, 32) if i % 5 == 0]
        dim0 = features[cluster_rows, 0]  # baseline value 0.0 -> lower clip
        dim1 = features[cluster_rows, 1]  # baseline value 2.0 -> upper clip
        assert bool((dim0 == 0.0).any())
        assert bool(((dim0 > 0.0) & (dim0 < 0.1)).any())
        assert bool((dim1 == 2.0).any())
        assert bool((dim1 <= 2.0).all())
        assert float(labels.sum()) == 32.0

    def test_collapse_arithmetic_with_three_anchors(self) -> None:
        # With 3 anchors, max_collapse = min(6, 3 + 1) = 4 so negatives
        # collapse 1..3 anchors.  Kills the Add -> Sub at line 242
        # (min(6, 2) = 2 caps the max at 1), the +1 -> +2 int tweak at
        # line 242 (collapse=4 would need a 4-element no-replace choice
        # from 3 anchors -> ValueError), and the integers(1, ·) low
        # bound tweak at line 244 (draws would start at 2).  The real
        # 24-anchor baseline saturates min(6, n ± 1) at 6, hiding the
        # first two mutants from the byte-identity oracle.
        synthetic = _synthetic_baseline()
        features, labels = build_integrity_samples(synthetic, seed=11, n_positive=4, n_negative=32)
        negatives = features[labels < 0.5]
        counts = (negatives[:, synthetic.anchor_idx] < _ANCHOR_CUTOFF).sum(axis=1)
        assert int(counts.min()) == 1
        assert int(counts.max()) == 3
        assert np.bincount(counts, minlength=4).tolist() == [0, 13, 9, 10]


# =============================================================================
# Canonical serialisation + parse_corpus (lines 353-364, 621-655)
# =============================================================================


class TestCanonicalSerialisation:
    """The canonical bytes are sorted-key JSON + trailing newline."""

    def test_canonical_payload_fields(self, bundle: CorpusBundle) -> None:
        # Kills the shape-index tweaks at lines 357/358
        # (shape[1] -> shape[2] raises IndexError; shape[0] -> shape[1]
        # writes n_samples=256): the canonical header fields are part of
        # the signed bytes.
        payload: dict[str, Any] = json.loads(_serialise_corpus(bundle))
        assert payload["schema"] == "sigma_immutable_corpus/v1"
        assert payload["input_dim"] == 256
        assert payload["n_samples"] == 128
        assert payload["seed"] == 42
        assert payload["threshold"] == 0.93
        assert len(payload["features_hex"]) == 128
        assert len(payload["features_hex"][0]) == 256
        assert len(payload["labels_hex"]) == 128

    def test_sorted_keys_and_trailing_newline(self, bundle: CorpusBundle) -> None:
        # Kills the bool flip at line 364 (sort_keys=True -> False: the
        # dict's insertion order is not sorted) and the Add -> Sub on
        # ``+ "\n"`` (TypeError).  Signature stability depends on both.
        raw = _serialise_corpus(bundle)
        assert raw.endswith(b"}\n")
        text = raw.decode("utf-8")
        top_level_keys = list(json.loads(text).keys())
        assert top_level_keys == sorted(top_level_keys)

    def test_parse_round_trip_is_bit_exact(self, bundle: CorpusBundle) -> None:
        # Kills the setflags bool flips at lines 644/645 and pins the
        # float.hex round-trip: parse(serialise(bundle)) reconstructs
        # features/labels bit-exactly, read-only, with provenance intact.
        raw = _serialise_corpus(bundle)
        parsed = parse_corpus(raw)
        assert np.array_equal(parsed.features, bundle.features)
        assert np.array_equal(parsed.labels, bundle.labels)
        assert parsed.features.flags.writeable is False
        assert parsed.labels.flags.writeable is False
        assert parsed.sha3_256 == bundle.sha3_256
        assert parsed.seed == 42
        assert parsed.threshold == 0.93
        assert parsed.positive == 64
        assert parsed.negative == 64
        assert _serialise_corpus(parsed) == raw

    def test_label_count_boundary_at_one_half(self) -> None:
        # Kills the 0.5 float tweaks and the GtE -> Gt / Lt -> LtE swaps
        # at lines 652/653: a label of exactly 0.5 counts as positive
        # (>= 0.5) and NOT as negative (< 0.5 is False at 0.5).
        features = np.array([[0.25, 0.75], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
        labels = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        parsed = parse_corpus(_canonical_bytes(features, labels, seed=3, threshold=0.5))
        assert parsed.positive == 2
        assert parsed.negative == 1


# =============================================================================
# write_corpus / sign_and_persist_corpus (lines 388, 464-465)
# =============================================================================


class TestPersistence:
    """Persistence creates nested dirs, is idempotent, and byte-stable."""

    def test_write_corpus_nested_and_repeated(
        self, tiny_bundle: CorpusBundle, tmp_path: Path
    ) -> None:
        # Kills the mkdir bool flips at line 388: parents=False fails on
        # the two-level-deep fresh directory; exist_ok=False fails on
        # the second write into the now-existing directory.
        out = tmp_path / "deep" / "nest" / "corpus.json"
        written = write_corpus(tiny_bundle, out)
        assert written == out.read_bytes()
        assert written == _serialise_corpus(tiny_bundle)
        again = write_corpus(tiny_bundle, out.parent / "corpus2.json")
        assert again == written

    def test_sign_and_persist_round_trip(self, tiny_bundle: CorpusBundle, tmp_path: Path) -> None:
        # Kills, at lines 464/465: the Add -> Sub on ``+ "\n"``
        # (TypeError), sort_keys/indent tweaks (the on-disk signature
        # bytes must equal the sorted indent-2 dump of the returned
        # payload), and both sig-dir mkdir bool flips (fresh nested dirs
        # on the first call, existing dirs on the second).
        corpus_path = tmp_path / "c1" / "c2" / "corpus.json"
        sig_path = tmp_path / "s1" / "s2" / "corpus.sig.json"
        payload = sign_and_persist_corpus(tiny_bundle, corpus_path=corpus_path, sig_path=sig_path)

        assert payload["schema"] == "sigma_immutable_corpus_signatures/v1"
        assert payload["corpus_sha3_256"] == tiny_bundle.sha3_256
        assert set(payload["signatures"]) == {ED25519_ALG, MLDSA65_ALG}
        for alg in (ED25519_ALG, MLDSA65_ALG):
            entry = payload["signatures"][alg]
            assert entry["algorithm"] == alg
            assert len(bytes.fromhex(entry["signature_hex"])) > 0
            assert len(bytes.fromhex(entry["public_key_hex"])) > 0

        expected_sig_bytes = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        assert sig_path.read_bytes() == expected_sig_bytes
        assert corpus_path.read_bytes() == _serialise_corpus(tiny_bundle)

        # The freshly signed pair must verify end-to-end.
        statuses = verify_corpus_signatures(corpus_path=corpus_path, sig_path=sig_path)
        assert statuses == {ED25519_ALG: "verified", MLDSA65_ALG: "verified"}

        # Re-signing into the existing directories must succeed
        # (exist_ok=True on both mkdir calls).
        payload_again = sign_and_persist_corpus(
            tiny_bundle, corpus_path=corpus_path, sig_path=sig_path
        )
        assert payload_again["corpus_sha3_256"] == tiny_bundle.sha3_256


# =============================================================================
# load_signature_payload — schema/shape validation (lines 509-521)
# =============================================================================


class TestLoadSignaturePayload:
    """The committed manifest loads; malformed manifests are refused."""

    def test_committed_manifest_loads(self) -> None:
        # Kills the not-removals at lines 509/518 and NotEq -> Eq at
        # line 509: on the VALID committed manifest neither guard may
        # fire.
        payload = load_signature_payload()
        assert payload["schema"] == "sigma_immutable_corpus_signatures/v1"
        assert ED25519_ALG in payload["signatures"]
        assert MLDSA65_ALG in payload["signatures"]

    def test_wrong_schema_refused_despite_valid_signatures(self, tmp_path: Path) -> None:
        # Kills Or -> And at line 509: with ``and``, a dict payload
        # short-circuits the schema check entirely, so this
        # wrong-schema-but-well-shaped manifest would load.
        bad = tmp_path / "sig.json"
        bad.write_text(json.dumps({"schema": "bogus/v0", "signatures": {ED25519_ALG: {}}}))
        with pytest.raises(CorpusVerificationError, match="unexpected or missing schema"):
            load_signature_payload(bad)

    def test_non_dict_payload_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "sig.json"
        bad.write_text("[]")
        with pytest.raises(CorpusVerificationError, match="unexpected or missing schema"):
            load_signature_payload(bad)

    def test_missing_ed25519_entry_refused(self, tmp_path: Path) -> None:
        # Kills Or -> And at line 518: with ``and``, a dict signatures
        # section short-circuits the mandatory-entry check.
        bad = tmp_path / "sig.json"
        bad.write_text(
            json.dumps(
                {
                    "schema": "sigma_immutable_corpus_signatures/v1",
                    "signatures": {MLDSA65_ALG: {}},
                }
            )
        )
        with pytest.raises(CorpusVerificationError, match="mandatory"):
            load_signature_payload(bad)

    def test_missing_file_refused(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusVerificationError, match="unreadable"):
            load_signature_payload(tmp_path / "absent.json")

    def test_invalid_json_refused(self, tmp_path: Path) -> None:
        bad = tmp_path / "sig.json"
        bad.write_text("{not json")
        with pytest.raises(CorpusVerificationError, match="not valid JSON"):
            load_signature_payload(bad)


# =============================================================================
# verify_corpus_signatures — digest + dual-signature checks (lines 558-616)
# =============================================================================


class TestVerifyCorpusSignatures:
    """Committed artifacts verify clean; tampering is refused."""

    def test_committed_corpus_verifies(self) -> None:
        # Kills NotEq -> Eq at line 558 and the not-removals at lines
        # 585/612: on the genuine committed corpus + manifest the digest
        # matches and both signatures verify, so nothing may raise and
        # both algorithms must report verified.
        statuses = verify_corpus_signatures()
        assert statuses == {ED25519_ALG: "verified", MLDSA65_ALG: "verified"}

    def test_tampered_corpus_digest_mismatch(self, tmp_path: Path) -> None:
        # Pins the raise direction of the line-558 digest check.
        tampered = tmp_path / "corpus.json"
        tampered.write_bytes(CORPUS_PATH.read_bytes() + b" ")
        with pytest.raises(CorpusVerificationError, match="SHA3-256 mismatch"):
            verify_corpus_signatures(corpus_path=tampered, sig_path=CORPUS_SIG_PATH)

    def test_corrupted_ed25519_signature_refused(self, tmp_path: Path) -> None:
        # Pins the raise direction of the line-585 check with a
        # bit-flipped Ed25519 signature over the intact corpus.
        manifest: dict[str, Any] = json.loads(CORPUS_SIG_PATH.read_text())
        sig_hex = manifest["signatures"][ED25519_ALG]["signature_hex"]
        flipped = ("0" if sig_hex[0] != "0" else "1") + sig_hex[1:]
        manifest["signatures"][ED25519_ALG]["signature_hex"] = flipped
        bad = tmp_path / "sig.json"
        bad.write_text(json.dumps(manifest))
        with pytest.raises(CorpusVerificationError, match="Ed25519 signature verification failed"):
            verify_corpus_signatures(corpus_path=CORPUS_PATH, sig_path=bad)

    def test_missing_corpus_file_refused(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusVerificationError, match="corpus file unreadable"):
            load_corpus_bytes(tmp_path / "absent.json")


# =============================================================================
# Key-fingerprint format contract (hoisted to crypto_api.key_fingerprint)
# =============================================================================


class TestKeyFingerprintContract:
    """The advisory Signature.public_key_hash format has one home.

    The 16-hex-char SHA3-256 truncation used to be duplicated inline at
    both corpus verification sites (where its literal survived mutation)
    and at the crypto_api signing site; it is now the single
    ``key_fingerprint`` helper, and this test pins its format so a drift
    breaks here instead of desynchronising persisted records.
    """

    def test_fingerprint_is_sha3_256_prefix(self) -> None:
        from omni_mercury_engine.security.crypto_api import (
            KEY_FINGERPRINT_HEX_CHARS,
            key_fingerprint,
        )

        material = b"\x01\x02\x03\x04"
        fingerprint = key_fingerprint(material)
        assert KEY_FINGERPRINT_HEX_CHARS == 16
        assert len(fingerprint) == 16
        assert hashlib.sha3_256(material).hexdigest().startswith(fingerprint)

    def test_committed_manifest_fingerprints_match_helper(self) -> None:
        from omni_mercury_engine.security.crypto_api import key_fingerprint

        manifest: dict[str, Any] = json.loads(CORPUS_SIG_PATH.read_text())
        for algorithm in (ED25519_ALG, MLDSA65_ALG):
            entry = manifest["signatures"][algorithm]
            fingerprint = key_fingerprint(bytes.fromhex(entry["public_key_hex"]))
            assert len(fingerprint) == 16

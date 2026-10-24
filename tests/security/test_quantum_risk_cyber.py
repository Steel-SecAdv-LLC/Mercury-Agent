# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioural tests for the quantum-risk engine.

Each class pins one contract. The classification tests include the exact
failures of the module this replaced: a substring match read ``ML-DSA`` as
``DSA``, a hash token made ``ECDSA-SHA256`` look quantum-safe, and mixed-case
names never matched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omni_mercury_engine.security.quantum_risk_cyber import (
    CNSA2_DEADLINES,
    MERCURY_CRYPTO_USES,
    OID_NAMES,
    CRQCAssumption,
    CryptoAsset,
    QuantumStatus,
    assess_asset,
    audit_mercury_posture,
    classify_algorithm,
    prioritize,
    read_cbom,
)

R, V, B, U = (
    QuantumStatus.RESISTANT,
    QuantumStatus.VULNERABLE,
    QuantumStatus.BROKEN,
    QuantumStatus.UNKNOWN,
)
SRC = Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"


class TestClassification:
    @pytest.mark.parametrize(
        ("name", "canonical", "status", "category"),
        [
            ("ML-DSA-65", "ML-DSA-65", R, 3),
            ("ml_dsa_65", "ML-DSA-65", R, 3),
            ("MLDSA65", "ML-DSA-65", R, 3),
            ("ML-KEM-1024", "ML-KEM-1024", R, 5),
            ("SLH-DSA-SHAKE-256f", "SLH-DSA-SHAKE-256f", R, 5),
            ("DSA-2048", "DSA-2048", V, None),
            ("Ed25519", "Ed25519", V, None),
            ("RSA-1024", "RSA-1024", B, None),
            ("SIKEp434", "SIKE", B, None),
            ("Rainbow-III", "Rainbow", B, None),
            ("SHA-1", "SHA-1", B, None),
            ("3DES", "3DES", B, None),
            ("AES-128-GCM", "AES-128", R, 1),
            ("AES-256-GCM", "AES-256", R, 5),
            ("HS384", "HMAC-SHA-384", R, None),
        ],
    )
    def test_exact_names_and_aliases(
        self, name: str, canonical: str, status: QuantumStatus, category: int | None
    ) -> None:
        profile = classify_algorithm(name)
        assert (profile.canonical, profile.status, profile.nist_category) == (
            canonical,
            status,
            category,
        )

    def test_ml_dsa_is_never_read_as_dsa(self) -> None:
        assert classify_algorithm("ML-DSA-87").status is R
        assert classify_algorithm("DSA").status is V

    @pytest.mark.parametrize(
        "name",
        ["ecdsa-with-SHA256", "SHA256withECDSA", "ECDSA-P-256-SHA384", "sha512WithRSAEncryption"],
    )
    def test_a_hash_never_upgrades_a_vulnerable_signature(self, name: str) -> None:
        profile = classify_algorithm(name)
        assert profile.status is V
        assert profile.kind == "signature"
        assert profile.cnsa2 is False

    def test_unrecognised_names_are_unknown_not_guessed(self) -> None:
        for name in ("FooCrypt", "ML-DSA-66", "RSA-SHA999", "ECDSA-P-999"):
            profile = classify_algorithm(name)
            assert profile.status is U, name
            assert profile.kind == "unknown"

    def test_prestandard_names_map_to_their_standard_but_are_flagged(self) -> None:
        kyber = classify_algorithm("Kyber768")
        assert (kyber.status, kyber.nist_category, kyber.prestandard) == (R, 3, True)
        assert "ML-KEM-768" in kyber.standard
        sphincs = classify_algorithm("SPHINCS+-SHA2-256f-simple")
        assert sphincs.prestandard and sphincs.nist_category == 5

    def test_cnsa2_approves_only_the_level_5_suite(self) -> None:
        approved = {"ML-KEM-1024", "ML-DSA-87", "AES-256", "SHA-384", "SHA-512", "LMS", "XMSS"}
        rejected = {"ML-KEM-768", "ML-DSA-65", "AES-128", "SHA-256", "SLH-DSA-SHA2-256f"}
        assert all(classify_algorithm(n).cnsa2 for n in approved)
        assert not any(classify_algorithm(n).cnsa2 for n in rejected)

    def test_hybrid_key_establishment_is_as_strong_as_its_post_quantum_part(self) -> None:
        hybrid = classify_algorithm("X25519MLKEM768")
        assert hybrid.status is R and hybrid.nist_category == 3
        assert {c.canonical for c in hybrid.components} == {"X25519", "ML-KEM-768"}
        draft = classify_algorithm("X25519Kyber768Draft00")
        assert draft.status is R and draft.prestandard

    def test_composite_signature_takes_its_strongest_public_key_part(self) -> None:
        composite = classify_algorithm("MLDSA65-ECDSA-P256")
        assert composite.status is R
        broken_pair = classify_algorithm("SIKEp434-X25519")
        assert broken_pair.status is V  # X25519 still stands, classically

    def test_ecb_mode_is_flagged(self) -> None:
        assert any("ECB" in n for n in classify_algorithm("AES-256-ECB").notes)
        assert not any("ECB" in n for n in classify_algorithm("AES-256-GCM").notes)

    @pytest.mark.parametrize(
        ("oid", "canonical"),
        [
            ("2.16.840.1.101.3.4.3.18", "ML-DSA-65"),
            ("2.16.840.1.101.3.4.4.3", "ML-KEM-1024"),
            ("2.16.840.1.101.3.4.3.25", "SLH-DSA-SHA2-256f"),
            ("2.16.840.1.101.3.4.3.31", "SLH-DSA-SHAKE-256f"),
            ("1.3.101.112", "Ed25519"),
        ],
    )
    def test_oids_resolve_to_their_standard_names(self, oid: str, canonical: str) -> None:
        assert OID_NAMES[oid] == canonical
        assert classify_algorithm(oid).canonical == canonical

    def test_an_rsa_signature_oid_is_a_vulnerable_signature(self) -> None:
        profile = classify_algorithm("1.2.840.113549.1.1.11")
        assert profile.status is V and profile.kind == "signature"
        assert classify_algorithm("1.2.3.4").status is U


class TestCRQCAssumption:
    def test_exactly_one_form_is_required(self) -> None:
        with pytest.raises(ValueError):
            CRQCAssumption()
        with pytest.raises(ValueError):
            CRQCAssumption(years=10, cumulative=((10, 0.5),))

    @pytest.mark.parametrize(
        "curve",
        [((10, 0.5), (5, 0.6)), ((5, 0.6), (10, 0.5)), ((5, 1.2),), ((5, 0.2), (5, 0.3))],
    )
    def test_malformed_curves_are_rejected(self, curve: tuple[tuple[float, float], ...]) -> None:
        with pytest.raises(ValueError):
            CRQCAssumption(cumulative=curve)

    def test_interpolation_and_lower_bound(self) -> None:
        curve = CRQCAssumption(cumulative=((5, 0.05), (10, 0.2), (15, 0.5), (30, 0.9)))
        assert curve.probability_by(15) == (0.5, False)
        for horizon, expected in ((12.5, 0.35), (2.5, 0.025)):
            probability, lower_bound = curve.probability_by(horizon)
            assert probability == pytest.approx(expected) and not lower_bound
        assert curve.probability_by(30) == (0.9, False)  # on the curve: exact
        assert curve.probability_by(40) == (0.9, True)


def _asset(**kwargs: object) -> CryptoAsset:
    base: dict[str, object] = {
        "name": "archive",
        "algorithms": ("RSA-2048", "AES-256-GCM"),
        "protects": "confidentiality",
        "shelf_life_years": 10.0,
        "migration_years": 5.0,
    }
    base.update(kwargs)
    return CryptoAsset(**base)  # type: ignore[arg-type]


class TestMosca:
    def test_without_an_assumption_the_horizon_is_stated(self) -> None:
        a = assess_asset(_asset(), current_year=2026)
        assert a.exposure_horizon_years == 15.0
        assert a.crqc_exposed is None and a.exposure_probability is None
        assert any("within 15 years" in f for f in a.findings)

    @pytest.mark.parametrize(("years", "exposed"), [(12, True), (15, False), (20, False)])
    def test_point_assumption(self, years: float, exposed: bool) -> None:
        a = assess_asset(_asset(), current_year=2026, crqc=CRQCAssumption(years=years))
        assert a.crqc_exposed is exposed

    def test_curve_assumption_gives_a_probability(self) -> None:
        crqc = CRQCAssumption(cumulative=((10, 0.2), (20, 0.6)))
        a = assess_asset(_asset(), current_year=2026, crqc=crqc)
        assert a.exposure_probability == pytest.approx(0.4)

    def test_quantum_resistant_protection_is_not_exposed(self) -> None:
        a = assess_asset(
            _asset(algorithms=("ML-KEM-768", "AES-256-GCM")),
            current_year=2026,
            crqc=CRQCAssumption(years=1),
        )
        assert a.exposure_horizon_years is None and a.crqc_exposed is None

    def test_unrecordable_traffic_is_not_harvested(self) -> None:
        a = assess_asset(_asset(algorithms=("ECDH-P-256",), recordable=False), current_year=2026)
        assert a.exposure_horizon_years is None

    def test_signatures_are_exposed_only_while_they_must_be_trusted(self) -> None:
        signed = _asset(algorithms=("ECDSA-P-256",), protects="authenticity", shelf_life_years=20)
        a = assess_asset(signed, current_year=2026, crqc=CRQCAssumption(years=15))
        assert a.exposure_horizon_years == 25.0 and a.crqc_exposed is True

    def test_a_vulnerable_signature_does_not_expose_confidentiality(self) -> None:
        a = assess_asset(
            _asset(algorithms=("ECDSA-P-256", "ML-KEM-768", "AES-256-GCM")), current_year=2026
        )
        assert a.exposure_horizon_years is None
        assert a.weakest is R


class TestPolicyDeadlines:
    @pytest.mark.parametrize(
        ("algorithm", "deadline"),
        [("RSA-2048", 2030), ("RSA", 2030), ("RSA-3072", 2035), ("ECDH-P-384", 2035)],
    )
    def test_nist_ir_8547_dates_follow_the_strength_band(
        self, algorithm: str, deadline: int
    ) -> None:
        a = assess_asset(_asset(algorithms=(algorithm,)), current_year=2026)
        assert a.nist_deadline == deadline

    @pytest.mark.parametrize(("migration", "late"), [(3, False), (4, False), (4.5, True)])
    def test_lateness_compares_migration_finish_with_the_deadline(
        self, migration: float, late: bool
    ) -> None:
        a = assess_asset(
            _asset(algorithms=("RSA-2048",), migration_years=migration), current_year=2026
        )
        assert a.nist_late is late

    def test_cnsa2_category_sets_deadline_and_names_noncompliant_algorithms(self) -> None:
        a = assess_asset(
            _asset(
                algorithms=("ML-KEM-768", "AES-256-GCM"),
                cnsa2_category="software_firmware_signing",
                migration_years=5,
            ),
            current_year=2026,
        )
        assert a.cnsa2_deadline == CNSA2_DEADLINES["software_firmware_signing"][1] == 2030
        assert a.cnsa2_noncompliant == ("ML-KEM-768",)
        assert a.cnsa2_late is True
        assert any("ML-KEM-1024" in r for r in a.recommendations)

    def test_password_hashing_is_outside_cnsa2_scope(self) -> None:
        a = assess_asset(
            _asset(
                algorithms=("ML-KEM-1024", "AES-256-GCM", "Argon2id", "scrypt"),
                cnsa2_category="custom_legacy",
            ),
            current_year=2026,
        )
        assert a.cnsa2_noncompliant == ()
        assert a.cnsa2_late is False

    def test_invalid_assets_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            _asset(protects="integrity")
        with pytest.raises(ValueError):
            _asset(migration_years=-1)
        with pytest.raises(ValueError):
            _asset(cnsa2_category="mainframes")


class TestPrioritisation:
    def test_broken_then_exposed_then_late_then_horizon(self) -> None:
        crqc = CRQCAssumption(years=12)
        assets = [
            _asset(name="safe", algorithms=("ML-KEM-1024", "AES-256-GCM")),
            _asset(name="late", algorithms=("RSA-2048",), shelf_life_years=1, migration_years=6),
            _asset(name="exposed", algorithms=("RSA-3072",), shelf_life_years=20),
            _asset(name="broken", algorithms=("3DES",), shelf_life_years=0),
        ]
        order = [
            a.asset
            for a in prioritize(assess_asset(x, current_year=2026, crqc=crqc) for x in assets)
        ]
        assert order == ["broken", "exposed", "late", "safe"]


def _cbom(*components: dict[str, object]) -> dict[str, object]:
    return {"bomFormat": "CycloneDX", "specVersion": "1.6", "components": list(components)}


def _algorithm(ref: str, name: str, **props: object) -> dict[str, object]:
    algo = {k: v for k, v in props.items() if k != "oid"}
    crypto: dict[str, object] = {"assetType": "algorithm", "algorithmProperties": algo}
    if "oid" in props:
        crypto["oid"] = props["oid"]
    return {"type": "cryptographic-asset", "bom-ref": ref, "name": name, "cryptoProperties": crypto}


class TestCBOM:
    def test_classifies_by_oid_name_and_parameter_set(self) -> None:
        entries = read_cbom(
            _cbom(
                _algorithm("a", "whatever", oid="2.16.840.1.101.3.4.3.19"),
                _algorithm("b", "ML-KEM", parameterSetIdentifier="768"),
                {"type": "library", "name": "openssl"},
                {
                    "type": "application",
                    "name": "app",
                    "components": [_algorithm("c", "Ed25519")],
                },
            )
        )
        assert [(e.bom_ref, e.profile.canonical) for e in entries] == [
            ("a", "ML-DSA-87"),
            ("b", "ML-KEM-768"),
            ("c", "Ed25519"),
        ]

    @pytest.mark.parametrize(
        ("name", "claimed", "finding"),
        [
            ("RSA-2048", 3, "claims quantum security level 3"),
            ("ML-DSA-65", 0, "understates"),
            ("ML-DSA-65", 5, "the standard assigns category 3"),
        ],
    )
    def test_inventory_claims_are_checked_against_the_standard(
        self, name: str, claimed: int, finding: str
    ) -> None:
        (entry,) = read_cbom(_cbom(_algorithm("x", name, nistQuantumSecurityLevel=claimed)))
        assert entry.claimed_level == claimed
        assert any(finding in f for f in entry.findings)

    def test_a_correct_claim_raises_nothing(self) -> None:
        (entry,) = read_cbom(_cbom(_algorithm("x", "ML-DSA-65", nistQuantumSecurityLevel=3)))
        assert entry.findings == ()

    def test_non_cyclonedx_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="CycloneDX"):
            read_cbom({"bomFormat": "SPDX"})


class TestMercurySelfAudit:
    @pytest.mark.parametrize("use", MERCURY_CRYPTO_USES, ids=lambda u: u.source)
    def test_each_declaration_matches_its_source_file(self, use: object) -> None:
        source = (SRC / use.source).read_text(encoding="utf-8")  # type: ignore[attr-defined]
        for algorithm in use.algorithms:  # type: ignore[attr-defined]
            assert algorithm in source, f"{algorithm} not found in {use.source}"  # type: ignore[attr-defined]
            assert classify_algorithm(algorithm).status is not U

    def test_audit_names_the_quantum_vulnerable_emergency_tier(self) -> None:
        audit = {a.asset.split(" [")[0]: a for a in audit_mercury_posture(current_year=2026)}
        assert audit["sovereign emergency tier"].weakest is V
        assert audit["sovereign emergency tier"].nist_deadline == 2035
        assert audit["hybrid public-key encryption (KEM/DEM)"].weakest is R
        assert audit["FINDΩYOU™ evidence packets"].weakest is R

    def test_cnsa2_audit_lists_what_mercury_would_have_to_change(self) -> None:
        audit = {
            a.asset.split(" [")[0]: a
            for a in audit_mercury_posture(current_year=2026, cnsa2_category="custom_legacy")
        }
        assert audit["hybrid public-key encryption (KEM/DEM)"].cnsa2_noncompliant == ()
        assert "ML-DSA-65" in audit["FINDΩYOU™ evidence packets"].cnsa2_noncompliant
        assert "Ed25519" in audit["sovereign emergency tier"].cnsa2_noncompliant


def test_released_classes_are_not_restored() -> None:
    """DEPRECATION §6.15: the released classes flagged ML-DSA and passed SIKE/Rainbow."""
    import omni_mercury_engine.security.quantum_risk_cyber as module

    for name in (
        "QuantumRiskCyber",
        "PostQuantumMigrationPlanner",
        "QuantumThreat",
        "ThreatLevel",
        "CryptoSystem",
    ):
        assert not hasattr(module, name), name

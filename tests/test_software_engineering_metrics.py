"""
Tests for the extended SOFTWARE_ENGINEERING scalar group.

Guards the four metric families added to GOSNN:
- ISO/IEC 25010:2011 product-quality sub-characteristics
- Halstead complexity measures
- McCabe cyclomatic + cognitive (SonarQube) complexity
- NIST SAMATE software assurance metrics
"""

from __future__ import annotations

from typing import Any

import pytest

from omni_mercury_engine.core.global_omni_scalar_network import (
    GlobalOmniScalarNetwork,
    ScalarGroup,
    reset_global_network,
)


@pytest.fixture(autouse=True)
def _reset_gosnn() -> Any:
    reset_global_network()
    yield
    reset_global_network()


@pytest.fixture
def se_scalars() -> dict[str, float]:
    gosnn = GlobalOmniScalarNetwork()
    return gosnn.scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING]


class TestISO25010:
    """ISO/IEC 25010:2011 must expose all 8 top-level characteristics."""

    EXPECTED_TOP_LEVEL = {
        "func": ["completeness", "correctness", "appropriateness"],
        "perf": ["time_behavior", "resource_util", "capacity"],
        "compat": ["coexistence", "interoperability"],
        "usab": [
            "appropriateness_recog",
            "learnability",
            "operability",
            "user_error_protect",
            "ui_aesthetics",
            "accessibility",
        ],
        "rel": ["maturity", "availability", "fault_tolerance", "recoverability"],
        "sec": [
            "confidentiality",
            "integrity",
            "non_repudiation",
            "accountability",
            "authenticity",
        ],
        "maint": [
            "modularity",
            "reusability",
            "analyzability",
            "modifiability",
            "testability",
        ],
        "port": ["adaptability", "installability", "replaceability"],
    }

    def test_all_eight_characteristics_present(self, se_scalars: dict[str, float]) -> None:
        for prefix, subs in self.EXPECTED_TOP_LEVEL.items():
            for sub in subs:
                key = f"omni_iso25010_{prefix}_{sub}"
                assert key in se_scalars, f"missing ISO 25010 scalar: {key}"

    def test_iso25010_subcharacteristic_count(self, se_scalars: dict[str, float]) -> None:
        # 3+3+2+6+4+5+5+3 = 31 sub-characteristics
        iso_keys = [k for k in se_scalars if k.startswith("omni_iso25010_")]
        assert len(iso_keys) == 31

    def test_iso25010_security_is_high_weight(self, se_scalars: dict[str, float]) -> None:
        # Confidentiality and integrity should be among the highest-weighted
        # ISO 25010 sub-characteristics (security-critical).
        assert se_scalars["omni_iso25010_sec_confidentiality"] >= 1.25
        assert se_scalars["omni_iso25010_sec_integrity"] >= 1.25


class TestHalstead:
    """Halstead 1977 measures: 7 derived quantities."""

    EXPECTED_KEYS = [
        "omni_halstead_vocabulary",
        "omni_halstead_length",
        "omni_halstead_volume",
        "omni_halstead_difficulty",
        "omni_halstead_effort",
        "omni_halstead_time_to_program",
        "omni_halstead_delivered_bugs",
    ]

    def test_all_seven_halstead_measures_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing Halstead measure: {key}"

    def test_halstead_measures_are_penalty_scalars(self, se_scalars: dict[str, float]) -> None:
        # All Halstead measures are penalty-direction (less is better).
        for key in self.EXPECTED_KEYS:
            assert (
                se_scalars[key] < 1.0
            ), f"Halstead measure {key} should be a penalty scalar (weight < 1.0)"

    def test_delivered_bugs_is_strongest_penalty(self, se_scalars: dict[str, float]) -> None:
        # Delivered bugs (B = V / 3000) is the most safety-critical Halstead
        # derivation; weight should be at least as strong as effort.
        assert se_scalars["omni_halstead_delivered_bugs"] <= se_scalars["omni_halstead_effort"]


class TestMcCabe:
    """McCabe + cognitive complexity (SonarQube definition)."""

    def test_cyclomatic_and_cognitive_present(self, se_scalars: dict[str, float]) -> None:
        assert "omni_mccabe_cyclomatic_complexity" in se_scalars
        assert "omni_cognitive_complexity_sonar" in se_scalars

    def test_essential_design_and_npath_present(self, se_scalars: dict[str, float]) -> None:
        assert "omni_mccabe_essential_complexity" in se_scalars
        assert "omni_mccabe_design_complexity" in se_scalars
        assert "omni_npath_complexity" in se_scalars

    def test_all_complexity_metrics_are_penalty(self, se_scalars: dict[str, float]) -> None:
        for key in [
            "omni_mccabe_cyclomatic_complexity",
            "omni_mccabe_essential_complexity",
            "omni_mccabe_design_complexity",
            "omni_cognitive_complexity_sonar",
            "omni_npath_complexity",
        ]:
            assert se_scalars[key] < 1.0


class TestMaintainabilityIndex:
    """SEI and Microsoft VS variants of the Maintainability Index."""

    def test_sei_and_vs_variants_present(self, se_scalars: dict[str, float]) -> None:
        assert "omni_maintainability_index_sei" in se_scalars
        assert "omni_maintainability_index_vs" in se_scalars
        assert "omni_maintainability_index_delta" in se_scalars

    def test_aggregate_still_present(self, se_scalars: dict[str, float]) -> None:
        # Pre-existing aggregate scalar must remain for backward compatibility.
        assert "omni_maintainability_index" in se_scalars


class TestNISTSAMATE:
    """NIST SAMATE software assurance metrics."""

    EXPECTED_KEYS = [
        "omni_samate_cwe_coverage",
        "omni_samate_sard_conformance",
        "omni_samate_weakness_density",
        "omni_samate_assurance_case_strength",
        "omni_samate_tool_effectiveness",
        "omni_samate_false_discovery_rate",
        "omni_samate_residual_risk",
        "omni_samate_evidence_completeness",
        "omni_samate_independent_verification",
        "omni_samate_supply_chain_assurance",
    ]

    def test_all_samate_metrics_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing SAMATE metric: {key}"

    def test_samate_penalty_direction_consistent(self, se_scalars: dict[str, float]) -> None:
        # Weakness density, false discovery rate, residual risk -> penalty.
        for key in [
            "omni_samate_weakness_density",
            "omni_samate_false_discovery_rate",
            "omni_samate_residual_risk",
        ]:
            assert se_scalars[key] < 1.0

    def test_samate_positive_direction_consistent(self, se_scalars: dict[str, float]) -> None:
        # All other SAMATE metrics are positive-direction.
        for key in [
            "omni_samate_cwe_coverage",
            "omni_samate_sard_conformance",
            "omni_samate_assurance_case_strength",
            "omni_samate_tool_effectiveness",
            "omni_samate_evidence_completeness",
            "omni_samate_independent_verification",
            "omni_samate_supply_chain_assurance",
        ]:
            assert se_scalars[key] > 1.0


class TestSigmaImmutableLayoutBudget:
    """The σ_Immutable trained gate has a fixed 256-D input layout where
    the active band ends at SIGMA_USED_BAND_END=180.  Pure-measurement
    scalars (ISO 25010, Halstead, McCabe, NIST SAMATE, MI variants)
    must NOT enter ``_collect_all_scalars`` or the trained network sees
    non-zero values in its reserved zero-padded tail and rejects the
    vector as poisoned.
    """

    def test_operational_scalar_count_within_band_budget(self) -> None:
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SIGMA_USED_BAND_END,
        )

        gosnn = GlobalOmniScalarNetwork()
        operational = gosnn._collect_all_scalars()
        assert len(operational) <= SIGMA_USED_BAND_END, (
            f"_collect_all_scalars returned {len(operational)} scalars; "
            f"σ_Immutable trained network requires <= {SIGMA_USED_BAND_END}"
        )

    def test_metric_only_scalars_excluded(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        operational = gosnn._collect_all_scalars()
        for key in operational:
            assert not key.startswith("omni_iso25010_"), key
            assert not key.startswith("omni_halstead_"), key
            assert not key.startswith("omni_mccabe_"), key
            assert not key.startswith("omni_samate_"), key
            assert key != "omni_cognitive_complexity_sonar"
            assert key != "omni_npath_complexity"
            assert key != "omni_maintainability_index_sei"
            assert key != "omni_maintainability_index_vs"
            assert key != "omni_maintainability_index_delta"

    def test_metric_only_scalars_still_discoverable(self) -> None:
        # They must remain accessible via scalar_groups[SOFTWARE_ENGINEERING]
        # for inspection/registration/reporting -- only the σ vector excludes them.
        gosnn = GlobalOmniScalarNetwork()
        se = gosnn.scalar_groups[ScalarGroup.SOFTWARE_ENGINEERING]
        for key in (
            "omni_iso25010_sec_integrity",
            "omni_halstead_delivered_bugs",
            "omni_mccabe_cyclomatic_complexity",
            "omni_cognitive_complexity_sonar",
            "omni_samate_supply_chain_assurance",
            "omni_maintainability_index_sei",
        ):
            assert key in se, f"{key} should remain in scalar_groups[SE]"


class TestGroupInvariants:
    """Cross-cutting invariants on the extended SOFTWARE_ENGINEERING group."""

    def test_no_duplicate_keys(self, se_scalars: dict[str, float]) -> None:
        # Dict literals deduplicate silently; verify total is the expected sum.
        assert len(se_scalars) == 101

    def test_all_weights_in_sane_range(self, se_scalars: dict[str, float]) -> None:
        for key, value in se_scalars.items():
            assert 0.5 < value < 2.0, f"weight for {key} = {value} is outside sane range (0.5, 2.0)"

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the extended SOFTWARE_ENGINEERING scalar group.

Guards the nine diagnostic measurement families and one operational
family registered in GOSNN's SOFTWARE_ENGINEERING group:

Operational (drive the σ_Immutable gate):
- Legacy code-quality / optimization / 3R synergy

Diagnostic measurement (registered, but excluded from the gate):
- ISO/IEC 25010:2011 product-quality sub-characteristics
- Halstead complexity measures
- McCabe + cognitive (SonarQube) complexity
- Maintainability Index variants (SEI / VS / delta)
- NIST SAMATE software assurance metrics
- DORA / DevOps Research and Assessment delivery metrics
- SLSA Supply-chain Levels for Software Artifacts v1.0
- Supply-chain / repository-integrity checks (Mercury-native)
- ISO/IEC 5055 / CISQ automated source-code quality measures
- NIST SP 800-218 SSDF practice groups
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


_METRIC_ONLY_PREFIXES = (
    "omni_iso25010_",
    "omni_halstead_",
    "omni_mccabe_",
    "omni_samate_",
    "omni_dora_",
    "omni_slsa_",
    "omni_ossf_",
    "omni_iso5055_",
    "omni_ssdf_",
)

_METRIC_ONLY_KEYS = (
    "omni_cognitive_complexity_sonar",
    "omni_npath_complexity",
    "omni_maintainability_index_sei",
    "omni_maintainability_index_vs",
    "omni_maintainability_index_delta",
)


class TestDORADelivery:
    """DORA / DevOps Research and Assessment - 4 delivery metrics."""

    EXPECTED_KEYS = [
        "omni_dora_deployment_frequency",
        "omni_dora_lead_time_for_changes",
        "omni_dora_mean_time_to_restore",
        "omni_dora_change_failure_rate",
    ]

    def test_all_four_dora_metrics_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing DORA metric: {key}"

    def test_throughput_positive_stability_penalty(self, se_scalars: dict[str, float]) -> None:
        # Deployment frequency: positive direction; the other three are
        # penalty-direction (shorter / lower is better).
        assert se_scalars["omni_dora_deployment_frequency"] > 1.0
        for key in (
            "omni_dora_lead_time_for_changes",
            "omni_dora_mean_time_to_restore",
            "omni_dora_change_failure_rate",
        ):
            assert se_scalars[key] < 1.0


class TestSLSASupplyChain:
    """SLSA Supply-chain Levels for Software Artifacts v1.0 - 4 scalars."""

    EXPECTED_KEYS = [
        "omni_slsa_level",
        "omni_slsa_source_integrity",
        "omni_slsa_build_provenance",
        "omni_slsa_dependency_attestation",
    ]

    def test_all_four_slsa_metrics_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing SLSA metric: {key}"

    def test_slsa_all_positive_direction(self, se_scalars: dict[str, float]) -> None:
        # All four SLSA properties are positive-direction.
        for key in self.EXPECTED_KEYS:
            assert se_scalars[key] > 1.0


class TestSupplyChainIntegrity:
    """Supply-chain & repository-integrity checks - 10 scalars (Mercury-native)."""

    EXPECTED_KEYS = [
        "omni_ossf_branch_protection",
        "omni_ossf_code_review_required",
        "omni_ossf_ci_tests_required",
        "omni_ossf_dependency_update_tool",
        "omni_ossf_dangerous_workflow",
        "omni_ossf_pinned_dependencies",
        "omni_ossf_sast_enabled",
        "omni_ossf_token_permissions",
        "omni_ossf_signed_releases",
        "omni_ossf_vulnerabilities",
    ]

    def test_all_ten_ossf_metrics_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing supply-chain integrity metric: {key}"

    def test_ossf_penalty_direction(self, se_scalars: dict[str, float]) -> None:
        # Dangerous workflows and open vulnerabilities are penalty-direction.
        assert se_scalars["omni_ossf_dangerous_workflow"] < 1.0
        assert se_scalars["omni_ossf_vulnerabilities"] < 1.0

    def test_ossf_positive_direction(self, se_scalars: dict[str, float]) -> None:
        for key in (
            "omni_ossf_branch_protection",
            "omni_ossf_code_review_required",
            "omni_ossf_ci_tests_required",
            "omni_ossf_dependency_update_tool",
            "omni_ossf_pinned_dependencies",
            "omni_ossf_sast_enabled",
            "omni_ossf_token_permissions",
            "omni_ossf_signed_releases",
        ):
            assert se_scalars[key] > 1.0


class TestISO5055CISQ:
    """ISO/IEC 5055 / CISQ automated source-code quality measures - 4 scalars."""

    EXPECTED_KEYS = [
        "omni_iso5055_reliability",
        "omni_iso5055_performance_efficiency",
        "omni_iso5055_security",
        "omni_iso5055_maintainability",
    ]

    def test_all_four_iso5055_measures_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing ISO/IEC 5055 measure: {key}"

    def test_security_is_strongest(self, se_scalars: dict[str, float]) -> None:
        # ISO 5055 Security is the most safety-critical of the four CISQ
        # measures; weight should be at least as strong as the others.
        sec = se_scalars["omni_iso5055_security"]
        for key in (
            "omni_iso5055_reliability",
            "omni_iso5055_performance_efficiency",
            "omni_iso5055_maintainability",
        ):
            assert sec >= se_scalars[key]


class TestNISTSSDF:
    """NIST SP 800-218 Secure Software Development Framework - 4 practice groups."""

    EXPECTED_KEYS = [
        "omni_ssdf_prepare_organization",
        "omni_ssdf_protect_software",
        "omni_ssdf_produce_well_secured_software",
        "omni_ssdf_respond_to_vulnerabilities",
    ]

    def test_all_four_ssdf_practices_present(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert key in se_scalars, f"missing NIST SSDF practice: {key}"

    def test_ssdf_all_positive_direction(self, se_scalars: dict[str, float]) -> None:
        for key in self.EXPECTED_KEYS:
            assert se_scalars[key] > 1.0


class TestSigmaImmutableLayoutBudget:
    """The σ_Immutable trained gate has a fixed 256-D input layout where
    the active band ends at SIGMA_USED_BAND_END=180.  Diagnostic
    measurement scalars (ISO 25010, Halstead, McCabe, MI variants, NIST
    SAMATE, DORA, SLSA, supply-chain integrity, ISO 5055, NIST SSDF) must NOT enter
    ``_collect_all_scalars`` or the trained network sees non-zero values
    in its reserved zero-padded tail and rejects the vector as poisoned.
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
            for prefix in _METRIC_ONLY_PREFIXES:
                assert not key.startswith(prefix), (
                    f"diagnostic measurement scalar {key} leaked into operational vector "
                    f"(prefix {prefix})"
                )
            assert (
                key not in _METRIC_ONLY_KEYS
            ), f"diagnostic measurement scalar {key} leaked into operational vector"

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
            "omni_dora_change_failure_rate",
            "omni_slsa_build_provenance",
            "omni_ossf_signed_releases",
            "omni_iso5055_security",
            "omni_ssdf_protect_software",
        ):
            assert key in se, f"{key} should remain in scalar_groups[SE]"

    def test_filter_helpers_partition_each_group(self) -> None:
        """``_operational_scalars_for`` + ``_metric_only_scalars_for`` exactly
        partition each group's registered scalars (no overlap, no loss).
        """
        gosnn = GlobalOmniScalarNetwork()
        for group in ScalarGroup:
            registered = gosnn.scalar_groups[group]
            operational = gosnn._operational_scalars_for(group)
            metric_only = gosnn._metric_only_scalars_for(group)
            assert set(operational).isdisjoint(metric_only), group
            assert set(operational) | set(metric_only) == set(registered), group
            assert len(operational) + len(metric_only) == len(registered), group


class TestOperationalAggregationConsistency:
    """Every operational aggregation path must honour the same filter.

    Regression guards against the silent semantic drift where new
    diagnostic measurement families would leak into the hierarchical
    accountability bucket or the dimensional-state fusion vectors.
    """

    def test_hierarchical_accountability_excludes_metric_only(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        result = gosnn.compute_hierarchical_score()
        # Accountability = operational SOFTWARE_ENGINEERING (45) + ADVANCED_REASONING (16) = 61.
        assert result["category_sizes"]["accountability"] == 45 + 16

    def test_dimensional_states_use_operational_only(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        states = gosnn._prepare_dimensional_states(base_scalars={}, context={})
        # First state is the (empty) base vector; the rest are per-group
        # operational vectors.  The SOFTWARE_ENGINEERING per-group state
        # should have exactly 45 entries (operational subset), not 127.
        # We identify the SE state by its length and content.
        se_op_values = list(
            gosnn._operational_scalars_for(ScalarGroup.SOFTWARE_ENGINEERING).values()
        )
        se_state_candidates = [s for s in states if len(s) == len(se_op_values)]
        assert se_state_candidates, "no dimensional state matches the operational SE size"

    def test_get_scalar_statistics_splits_operational_and_metric_only(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        stats = gosnn.get_scalar_statistics()
        # SOFTWARE_ENGINEERING must report both counts so operators can
        # see the filter at work.
        se_stats = stats["groups"]["software_engineering"]
        assert se_stats["count"] == 45  # operational
        assert se_stats["count_metric_only"] == 82  # diagnostic
        assert se_stats["count_registered"] == 127
        # Totals should reconcile.
        assert stats["total_scalars"] == stats["total_registered"] - stats["total_metric_only"]


class TestGroupInvariants:
    """Cross-cutting invariants on the extended SOFTWARE_ENGINEERING group."""

    def test_no_duplicate_keys(self, se_scalars: dict[str, float]) -> None:
        # Dict literals deduplicate silently; verify total is the expected sum.
        # 45 operational + 82 diagnostic = 127.
        assert len(se_scalars) == 127

    def test_all_weights_in_sane_range(self, se_scalars: dict[str, float]) -> None:
        for key, value in se_scalars.items():
            assert 0.5 < value < 2.0, f"weight for {key} = {value} is outside sane range (0.5, 2.0)"

    def test_advanced_reasoning_extended(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        ar = gosnn.scalar_groups[ScalarGroup.ADVANCED_REASONING]
        assert len(ar) == 16
        # ``omni_metasymbolic_grounding`` is the 16th AR scalar — it is
        # an operational signal, not a diagnostic measurement, so it
        # MUST appear in ``_collect_all_scalars``.
        assert "omni_metasymbolic_grounding" in ar
        assert "omni_metasymbolic_grounding" in gosnn._collect_all_scalars()

    def test_total_operational_count(self) -> None:
        gosnn = GlobalOmniScalarNetwork()
        operational = gosnn._collect_all_scalars()
        # 27 ETHICAL + 7 COSMIC + 7 QC + 9 HUMANITARIAN + 6 SECURITY
        # + 45 SE-op + 10 MEDICAL + 16 AR = 127.
        assert len(operational) == 127

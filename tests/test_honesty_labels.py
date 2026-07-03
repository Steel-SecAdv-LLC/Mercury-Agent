# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests pinning the honesty/scope labels (issue #8 cleanup).

These make the "heuristic, not measured" status *executable* so future edits
that re-introduce overstated claims are caught.
"""

from __future__ import annotations


class TestBainPowerHeuristicLabelled:
    def test_coefficients_are_named_constants(self) -> None:
        from omni_mercury_engine.scaling import bain_ai_scaling as b

        # Hoisted so the heuristic coefficients are visible/tunable.
        assert b.BASE_POWER_W == 100.0
        assert hasattr(b, "PER_GIGAPARAM_W")
        assert hasattr(b, "PER_BATCH_ITEM_W")
        assert hasattr(b, "PER_HUNDRED_TOKENS_W")

    def test_power_estimate_uses_constants(self) -> None:
        from omni_mercury_engine.scaling.bain_ai_scaling import (
            BASE_POWER_W,
            BainAIScaling,
        )

        scaler = BainAIScaling()
        # Zero model/batch/sequence -> base power exactly.
        assert scaler.estimate_power_consumption(0, 0, 0) == BASE_POWER_W

    def test_docstring_no_longer_claims_hyperscaler_grounding(self) -> None:
        from omni_mercury_engine.scaling.bain_ai_scaling import BainAIScaling

        doc = BainAIScaling.estimate_power_consumption.__doc__ or ""
        assert "NOT CALIBRATED" in doc
        assert "hyperscaler deployments" not in doc


class TestHierarchicalPlannerScoped:
    def test_planner_type_members_reserved(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import PlannerType

        # Members still exist (public API) but are documented as cosmetic.
        assert {p.name for p in PlannerType} >= {"OPTIONS", "MAXQ", "FEUDAL", "HAM"}
        assert "RESERVED" in (PlannerType.__doc__ or "") or "cosmetic" in (
            PlannerType.__doc__ or ""
        )

    def test_generic_subgoals_are_a_fixed_template(self) -> None:
        from omni_mercury_engine.cognitive.hierarchical_planning import (
            AbstractionLevel,
            Goal,
            GoalDecomposer,
        )

        dec = GoalDecomposer()
        goal = Goal(
            goal_id="g1",
            description="some unknown objective",
            level=AbstractionLevel.TACTICAL,
        )
        subs = dec._generate_generic_subgoals(goal)
        # The honest, documented behaviour: fixed initialize/process/complete.
        assert any("initialize" in s for s in subs)
        assert any("process" in s for s in subs)
        assert any("complete" in s for s in subs)


class TestAbductionIsLexical:
    def test_likelihood_is_jaccard_not_inference(self) -> None:
        from omni_mercury_engine.cognitive.multi_hop_reasoner import (
            MultiHopReasoner,
            Proposition,
        )

        r = MultiHopReasoner()
        obs = Proposition(prop_id="o", content="server cpu spike and memory leak", truth_value=1.0)
        overlapping = Proposition(prop_id="h1", content="memory leak in server", truth_value=1.0)
        disjoint = Proposition(prop_id="h2", content="weather is sunny today", truth_value=1.0)
        lk_overlap = r._compute_likelihood(obs, overlapping)
        lk_disjoint = r._compute_likelihood(obs, disjoint)
        # Floor 0.3 for zero overlap; overlap ranks strictly higher.
        assert abs(lk_disjoint - 0.3) < 1e-9
        assert lk_overlap > lk_disjoint


class TestOmniScalarMetricOnlyInert:
    def test_metric_only_scalars_excluded_from_operational_path(self) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        net = GlobalOmniScalarNetwork()
        operational = net._collect_all_scalars()
        # Not one diagnostic metric-only scalar leaks into the operational vector.
        assert all(not net._is_metric_only_scalar(k) for k in operational)

    def test_metric_only_prefixes_declared(self) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
        )

        assert len(GlobalOmniScalarNetwork._METRIC_ONLY_PREFIXES) > 0

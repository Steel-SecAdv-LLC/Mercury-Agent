# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ethical Autonomy Governor with Bias Audits and ΣDirective Overrides.

Implements comprehensive ethical oversight with:
- Bias detection (demographic parity, Fairlearn definition) when the caller
  supplies protected attributes; no automatic mitigation is performed
- ΣDirective (Sigma Directive) overrides for justice/altruism
- Statistical validation (p<0.05) when a real score sample is supplied
- Automatic rollback for ethical violations

References:
- Fairlearn: https://fairlearn.org/ (Microsoft, 2020)
- Asilomar AI Principles (2017)
- IEEE Ethically Aligned Design (2019)

MIT-compatible implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

from omni_mercury_engine.core.config import ThresholdConfig
from omni_mercury_engine.core.ethical_config import DEFAULT_CONFIG, EthicalScalars
from omni_mercury_engine.utils.rng import DeterministicRNG, get_global_rng


@dataclass
class EthicalDecision:
    """Record of ethical decision with validation.

    ``p_value`` is None when no score sample was supplied: a single score is
    not a sample, so no test is run (this field previously carried a t-test
    over RNG-drawn stand-in scores). ``bias_audit_performed`` is False when
    no protected attributes were supplied, in which case
    ``bias_audit_passed`` is a default, not a finding.
    """

    decision_id: str
    action: str
    ethical_score: float
    bias_audit_passed: bool
    p_value: float | None
    sigma_directive_applied: bool
    timestamp: datetime = field(default_factory=datetime.now)
    rollback_triggered: bool = False
    bias_audit_performed: bool = False


@dataclass
class BiasMetrics:
    """Bias audit metrics.

    ``audit_performed`` is False when the caller supplied no protected
    attributes; every metric field is then None rather than a number invented
    from an arbitrary split of the data. ``equalized_odds_diff`` is always
    None: computing it needs ground-truth labels this seam does not receive.
    ``mitigation_applied`` is always False — no automatic mitigation exists
    here, and this field previously claimed one had been applied.
    """

    audit_performed: bool
    demographic_parity_diff: float | None
    equalized_odds_diff: float | None
    statistical_parity: float | None
    bias_detected: bool
    mitigation_applied: bool = False


class SigmaDirective:
    """Σ Directive: Supreme ethical overrides for critical situations.

    Implements hierarchical directives:
    - Σ1: Justice - Ensure fairness and prevent discrimination
    - Σ2: Altruism - Prioritize benefit to humanity
    - Σ3: Compassion - Minimize harm and suffering
    - Σ4: Truth - Maintain transparency and honesty
    """

    JUSTICE = "justice"
    ALTRUISM = "altruism"
    COMPASSION = "compassion"
    TRUTH = "truth"

    def __init__(
        self,
        ethical_scalars: EthicalScalars,
        thresholds: ThresholdConfig | None = None,
    ) -> None:
        """Initialize Sigma Directive system.

        Args:
            ethical_scalars: Ethical scalar configuration
            thresholds: Threshold configuration (frozen at construction time)
        """
        self.ethical_scalars = ethical_scalars
        # Freeze thresholds at construction time to avoid global mutation risk
        self._thresholds = thresholds or ThresholdConfig()
        self.directive_weights = {
            self.JUSTICE: ethical_scalars.omni_justitia,
            self.ALTRUISM: ethical_scalars.omni_altruistic,
            self.COMPASSION: ethical_scalars.omni_compassionate,
            self.TRUTH: ethical_scalars.omni_truth_alignment,
        }

    def apply_directive(self, action: str, context: dict[str, Any]) -> tuple[bool, str]:
        """Apply Sigma Directive to validate action.

        Args:
            action: Proposed action
            context: Action context with ethical implications

        Returns:
            Tuple of (allow_action, reasoning)
        """
        justice_score = self._evaluate_justice(context)
        altruism_score = self._evaluate_altruism(context)
        compassion_score = self._evaluate_compassion(context)
        truth_score = self._evaluate_truth(context)

        weighted_score = (
            justice_score * self.directive_weights[self.JUSTICE]
            + altruism_score * self.directive_weights[self.ALTRUISM]
            + compassion_score * self.directive_weights[self.COMPASSION]
            + truth_score * self.directive_weights[self.TRUTH]
        ) / sum(self.directive_weights.values())

        if weighted_score < self._thresholds.sigma_directive:
            reasoning = self._generate_override_reasoning(
                justice_score, altruism_score, compassion_score, truth_score
            )
            return False, reasoning

        return True, "Action approved by Sigma Directive"

    def _evaluate_justice(self, context: dict[str, Any]) -> float:
        """Evaluate justice component."""
        fairness = context.get("fairness_score", 0.5)
        bias = context.get("bias_detected", False)

        if bias:
            return 0.0

        return float(fairness)

    def _evaluate_altruism(self, context: dict[str, Any]) -> float:
        """Evaluate altruism component."""
        benefit = context.get("societal_benefit", 0.5)
        harm = context.get("potential_harm", 0.0)

        net_benefit = benefit - harm
        return float(max(0.0, min(1.0, net_benefit)))

    def _evaluate_compassion(self, context: dict[str, Any]) -> float:
        """Evaluate compassion component."""
        harm_prevention = context.get("harm_prevention", 0.5)
        suffering_mitigation = context.get("suffering_mitigation", 0.5)

        return float((harm_prevention + suffering_mitigation) / 2.0)

    def _evaluate_truth(self, context: dict[str, Any]) -> float:
        """Evaluate truth component."""
        transparency = context.get("transparency", 0.5)
        honesty = context.get("honesty", 0.5)

        return float((transparency + honesty) / 2.0)

    def _generate_override_reasoning(
        self, justice: float, altruism: float, compassion: float, truth: float
    ) -> str:
        """Generate human-readable reasoning for override."""
        violations = []

        if justice < 0.7:
            violations.append("justice violation detected")
        if altruism < 0.7:
            violations.append("insufficient societal benefit")
        if compassion < 0.7:
            violations.append("potential harm not adequately mitigated")
        if truth < 0.7:
            violations.append("transparency requirements not met")

        return f"Sigma Directive Override: {', '.join(violations)}"


class EthicalAutonomyGovernor:
    """Comprehensive ethical governance system.

    Features:
    - ~150 ethical scalars from EthicalScalars
    - Bias auditing with statistical validation
    - ΣDirective overrides for critical decisions
    - Automatic rollback on violations
    - p<0.05 validation on all decisions
    """

    def __init__(
        self,
        ethical_scalars: EthicalScalars | None = None,
        p_value_threshold: float = 0.05,
        ethical_threshold: float = 0.8,
        rng: DeterministicRNG | None = None,
        thresholds: ThresholdConfig | None = None,
    ):
        """Initialize Ethical Autonomy Governor.

        Bias audits and Sigma Directive overrides are always active —
        there is no off-switch.  The ``enable_bias_audits`` and
        ``enable_sigma_directives`` parameters were removed in the
        May 2026 Phase 2 audit cure because ``False`` at construction
        silently bypassed all governance.

        Args:
            ethical_scalars: Ethical scalar configuration
            p_value_threshold: Statistical significance threshold
            ethical_threshold: Minimum ethical score threshold
            rng: Optional DeterministicRNG for reproducibility
            thresholds: Threshold configuration (frozen at construction time)
        """
        self.ethical_scalars = ethical_scalars or DEFAULT_CONFIG.ethical_scalars
        self.p_value_threshold = p_value_threshold
        self.ethical_threshold = ethical_threshold
        self._rng = rng or get_global_rng()
        # Freeze thresholds at construction time to avoid global mutation risk
        self._thresholds = thresholds or ThresholdConfig()

        self.sigma_directive = SigmaDirective(self.ethical_scalars, self._thresholds)

        self.decision_history: list[EthicalDecision] = []
        self.rollback_history: list[EthicalDecision] = []

    def evaluate_decision(
        self, action: str, context: dict[str, Any], data: np.ndarray[Any, Any] | None = None
    ) -> EthicalDecision:
        """Evaluate decision through ethical framework.

        Args:
            action: Proposed action
            context: Decision context
            data: Optional data for bias auditing

        Returns:
            EthicalDecision with validation results
        """
        decision_id = f"decision_{datetime.now().timestamp()}"

        ethical_score = self._compute_ethical_score(action, context)

        bias_audit_passed = True
        bias_audit_performed = False
        if data is not None:
            bias_metrics = self._audit_bias(data, context)
            bias_audit_performed = bias_metrics.audit_performed
            bias_audit_passed = not bias_metrics.bias_detected

        p_value = self._statistical_validation(ethical_score, context)

        sigma_directive_applied = False
        allow, reasoning = self.sigma_directive.apply_directive(action, context)
        sigma_directive_applied = not allow

        if not allow:
            context["sigma_override_reasoning"] = reasoning

        decision = EthicalDecision(
            decision_id=decision_id,
            action=action,
            ethical_score=ethical_score,
            bias_audit_passed=bias_audit_passed,
            p_value=p_value,
            sigma_directive_applied=sigma_directive_applied,
            bias_audit_performed=bias_audit_performed,
        )

        if self._should_rollback(decision):
            decision.rollback_triggered = True
            self.rollback_history.append(decision)
            from omni_mercury_engine.cognitive.ethical_bounding import (
                EthicalConstraintViolationError,
            )

            raise EthicalConstraintViolationError(
                action=action,
                score=ethical_score,
                threshold=self.ethical_threshold,
                check="governance_rollback",
                details={
                    "decision_id": decision_id,
                    "bias_audit_passed": bias_audit_passed,
                    "p_value": p_value,
                    "sigma_directive_applied": sigma_directive_applied,
                },
            )

        self.decision_history.append(decision)
        return decision

    def _compute_ethical_score(self, action: str, context: dict[str, Any]) -> float:
        """Compute ethical score from agent ethical scalars and per-decision behaviour signals.

        The base score reflects the agent's intrinsic ethical alignment
        (drawn from ~150 ``EthicalScalars``).  When the caller supplies
        per-decision behaviour signals (``fairness_score``, the
        ``societal_benefit``/``potential_harm`` net,
        ``harm_prevention``, ``suffering_mitigation``, ``transparency``,
        ``honesty``, or ``bias_detected``), the base is multiplicatively
        modulated by the mean of those signals so that bad contexts
        yield low scores and good contexts yield high scores.  Without
        this modulation, the score collapsed to ~1.23 for every input
        and the rollback gate had to lean entirely on the bias audit
        and Sigma Directive — a hidden weakness closed by the Phase 2
        audit cure.

        Args:
            action: Action to evaluate
            context: Context with ethical implications

        Returns:
            Ethical score (0.0 to ~2.5, scalar-normalized)
        """
        relevant_scalars = []

        if "harm" in context:
            relevant_scalars.append(self.ethical_scalars.omni_harm_prevention)
        if "benefit" in context:
            relevant_scalars.append(self.ethical_scalars.omni_benefit_promotion)
        if "fairness" in context:
            relevant_scalars.append(self.ethical_scalars.omni_fairness)

        relevant_scalars.extend(
            [
                self.ethical_scalars.omni_compassionate,
                self.ethical_scalars.omni_wisdom,
                self.ethical_scalars.omni_justitia,
                self.ethical_scalars.omni_altruistic,
            ]
        )

        base_score = float(np.mean(relevant_scalars)) if relevant_scalars else 1.0

        behaviour_signals: list[float] = []
        if "fairness_score" in context:
            behaviour_signals.append(float(context["fairness_score"]))
        if "societal_benefit" in context or "potential_harm" in context:
            net = float(context.get("societal_benefit", 0.5)) - float(
                context.get("potential_harm", 0.0)
            )
            behaviour_signals.append(max(0.0, min(1.0, net)))
        if "harm_prevention" in context:
            behaviour_signals.append(float(context["harm_prevention"]))
        if "suffering_mitigation" in context:
            behaviour_signals.append(float(context["suffering_mitigation"]))
        if "transparency" in context:
            behaviour_signals.append(float(context["transparency"]))
        if "honesty" in context:
            behaviour_signals.append(float(context["honesty"]))
        if context.get("bias_detected", False):
            behaviour_signals.append(0.0)

        behaviour_modifier = float(np.mean(behaviour_signals)) if behaviour_signals else 1.0

        context_modifier = 1.0
        if context.get("critical", False):
            context_modifier *= 1.2
        if context.get("humanitarian", False):
            context_modifier *= self.ethical_scalars.omni_disaster_response

        return float(base_score * behaviour_modifier * context_modifier)

    def _audit_bias(self, data: np.ndarray[Any, Any], context: dict[str, Any]) -> BiasMetrics:
        """Audit demographic parity across caller-supplied protected groups.

        The audit runs only when ``context["protected_attributes"]`` supplies a
        group label per row of ``data``; demographic parity difference is then
        the absolute difference of mean outcomes between the two largest
        groups (the Fairlearn definition). Without protected attributes no
        audit is possible — this method previously split the array in half
        and reported the halves' mean difference as a "Fairlearn-compatible"
        audit, and claimed ``mitigation_applied`` although no mitigation
        exists. Now it returns ``audit_performed=False`` with None metrics.

        Args:
            data: Per-row outcome scores to audit.
            context: Decision context; ``protected_attributes`` (array-like of
                group labels, one per row) enables the audit.

        Returns:
            BiasMetrics; ``audit_performed`` says whether anything was audited.
        """
        outcomes = np.asarray(data, dtype=float)
        if outcomes.ndim > 1:
            outcomes = outcomes.mean(axis=tuple(range(1, outcomes.ndim)))

        groups_raw = context.get("protected_attributes")
        if groups_raw is None:
            return BiasMetrics(
                audit_performed=False,
                demographic_parity_diff=None,
                equalized_odds_diff=None,
                statistical_parity=None,
                bias_detected=False,
            )

        groups = np.asarray(groups_raw)
        if groups.shape[0] != outcomes.shape[0]:
            raise ValueError(
                f"protected_attributes has {groups.shape[0]} labels for "
                f"{outcomes.shape[0]} data rows"
            )
        unique, counts = np.unique(groups, return_counts=True)
        if unique.size < 2:
            return BiasMetrics(
                audit_performed=False,
                demographic_parity_diff=None,
                equalized_odds_diff=None,
                statistical_parity=None,
                bias_detected=False,
            )

        # Two largest groups carry the comparison.
        top_two = unique[np.argsort(counts)[::-1][:2]]
        mean_a = float(np.mean(outcomes[groups == top_two[0]]))
        mean_b = float(np.mean(outcomes[groups == top_two[1]]))
        demographic_parity_diff = abs(mean_a - mean_b)
        statistical_parity = 1.0 - min(demographic_parity_diff, 1.0)
        bias_detected = demographic_parity_diff > self._thresholds.bias_detection

        return BiasMetrics(
            audit_performed=True,
            demographic_parity_diff=demographic_parity_diff,
            # Needs ground-truth labels this seam does not receive.
            equalized_odds_diff=None,
            statistical_parity=statistical_parity,
            bias_detected=bias_detected,
        )

    def _statistical_validation(
        self, ethical_score: float, context: dict[str, Any]
    ) -> float | None:
        """One-sample t-test of a real score sample against the 1.0 baseline.

        Args:
            ethical_score: Ethical score to validate (appended to the sample).
            context: Decision context; ``score_sample`` (iterable of prior
                real scores) enables the test.

        Returns:
            p-value when a sample of at least two distinct scores exists,
            None otherwise — never a value manufactured from RNG draws.
        """
        # A single score is not a sample, so without a caller-supplied sample
        # there is nothing to test. This method previously padded the score
        # with RNG draws around the baseline and ran a t-test on the mixture,
        # which manufactured a p-value out of noise.
        history = context.get("score_sample")
        if history is None:
            return None

        sample = np.asarray(list(history) + [ethical_score], dtype=float)
        if sample.size < 2 or float(np.std(sample)) == 0.0:
            return None

        baseline_score = 1.0
        _t_stat, p_value = stats.ttest_1samp(sample, baseline_score)
        return float(p_value)

    def _should_rollback(self, decision: EthicalDecision) -> bool:
        """Determine if decision should be rolled back.

        Args:
            decision: Decision to evaluate

        Returns:
            True if rollback required
        """
        if decision.ethical_score < self.ethical_threshold:
            return True

        if not decision.bias_audit_passed:
            return True

        # Only a real test can trip this clause; None means no sample was
        # supplied so no test ran.
        if (
            decision.p_value is not None
            and decision.p_value < self.p_value_threshold
            and decision.ethical_score < 1.0
        ):
            return True

        return bool(decision.sigma_directive_applied)

    def get_governance_report(self) -> dict[str, Any]:
        """Generate comprehensive governance report."""
        total_decisions = len(self.decision_history)
        total_rollbacks = len(self.rollback_history)

        avg_ethical_score = (
            np.mean([d.ethical_score for d in self.decision_history])
            if self.decision_history
            else 0.0
        )

        bias_audit_pass_rate = (
            np.mean([d.bias_audit_passed for d in self.decision_history])
            if self.decision_history
            else 0.0
        )

        sigma_directive_rate = (
            np.mean([d.sigma_directive_applied for d in self.decision_history])
            if self.decision_history
            else 0.0
        )

        return {
            "total_decisions": total_decisions,
            "total_rollbacks": total_rollbacks,
            "rollback_rate": (
                total_rollbacks / (total_decisions + total_rollbacks)
                if (total_decisions + total_rollbacks) > 0
                else 0.0
            ),
            "avg_ethical_score": float(avg_ethical_score),
            "bias_audit_pass_rate": float(bias_audit_pass_rate),
            "sigma_directive_application_rate": float(sigma_directive_rate),
            "ethical_scalars_count": len(self.ethical_scalars.to_dict()),
            "governance_status": "HEALTHY" if bias_audit_pass_rate > 0.95 else "NEEDS_ATTENTION",
        }

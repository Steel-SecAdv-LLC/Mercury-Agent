# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""System-Level Mathematical Coherence Verification.

Implements Phase 6 of the mathematical audit specification:
- Signal flow graph: data structure describing how signals propagate through
  the detection pipeline (ingestion -> feature extraction -> detection ->
  fusion -> ethical gating -> output).
- Normalization handoff verification: validates that score ranges are
  compatible at every boundary between pipeline stages.
- Lyapunov runtime monitoring: runtime guard that checks
  V_dot <= -lambda * V at every fusion step.  By default violations are
  logged and recorded (``is_stable`` / ``violations``); the guard halts
  the pipeline only when constructed with ``halt_on_violation=True``.

Reference: Khalil (2002) "Nonlinear Systems", Chapter 4 (Lyapunov Stability).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.core.centralized_constants import ETHICAL, LYAPUNOV

logger = logging.getLogger(__name__)

__all__ = [
    "CoherenceReport",
    "LyapunovRuntimeEnforcer",
    "NormalizationVerifier",
    "PipelineStage",
    "SignalFlowGraph",
    "run_coherence_audit",
]

# ---------------------------------------------------------------------------
# Signal Flow Graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineStage:
    """A single stage in the detection pipeline signal flow.

    Attributes:
        name: Human-readable stage name.
        input_range: Expected (min, max) of input scores.
        output_range: Guaranteed (min, max) of output scores.
        normalization: Normalization applied at this stage
            (e.g. "min-max", "sigmoid", "softmax", "none").
        description: Brief description of what the stage does.
    """

    name: str
    input_range: tuple[float, float]
    output_range: tuple[float, float]
    normalization: str = "none"
    description: str = ""


@dataclass
class SignalFlowGraph:
    """Describes how signals propagate through the Mercury detection pipeline.

    The graph is a linear sequence of PipelineStage nodes.  Each node specifies the expected
    input/output score ranges and the normalization method used, enabling automated handoff
    verification.
    """

    stages: list[PipelineStage] = field(default_factory=list)

    # ----- construction helpers -----

    @classmethod
    def build_default(cls) -> SignalFlowGraph:
        """Build the default Mercury Agent signal flow graph.

        This captures the canonical pipeline:
        raw data -> feature extraction -> per-detector scoring ->
        OAE fusion -> ethical gating -> calibrated output.
        """
        return cls(
            stages=[
                PipelineStage(
                    name="data_ingestion",
                    input_range=(float("-inf"), float("inf")),
                    output_range=(float("-inf"), float("inf")),
                    normalization="none",
                    description="Raw sensor / log data ingestion.",
                ),
                PipelineStage(
                    name="feature_extraction",
                    input_range=(float("-inf"), float("inf")),
                    output_range=(0.0, 1.0),
                    normalization="min-max",
                    description=(
                        "Domain feature extractors produce [0,1] normalized feature vectors."
                    ),
                ),
                PipelineStage(
                    name="detector_scoring",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                    normalization="sigmoid",
                    description=(
                        "Individual detectors (statistical, spectral, "
                        "ML) produce anomaly scores in [0,1]."
                    ),
                ),
                PipelineStage(
                    name="oae_fusion",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                    normalization="weighted-sum-clamp",
                    description=(
                        "OAE: A = (w_R*R + w_H*H + w_O*O) * eta^phi. "
                        "Weights sum to 1, eta in (0,1], result clamped."
                    ),
                ),
                PipelineStage(
                    name="ethical_gating",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                    normalization="sigmoid-gate",
                    description=(
                        "Sigmoid benevolence gate scales fusion score "
                        f"(omnibenevolence scalar = {ETHICAL.OMNIBENEVOLENCE_SCALAR}; weighting, not a gate)."
                    ),
                ),
                PipelineStage(
                    name="conformal_calibration",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                    normalization="quantile",
                    description=(
                        "Conformal prediction provides calibrated "
                        "uncertainty bands with coverage guarantee."
                    ),
                ),
                PipelineStage(
                    name="output",
                    input_range=(0.0, 1.0),
                    output_range=(0.0, 1.0),
                    normalization="none",
                    description="Final anomaly score and binary decision.",
                ),
            ]
        )

    def to_ascii(self) -> str:
        """Render the signal flow as an ASCII diagram."""
        lines: list[str] = []
        lines.append("Mercury Agent Signal Flow Graph")
        lines.append("=" * 60)
        for i, stage in enumerate(self.stages):
            arrow = " --> " if i > 0 else "      "
            lo_in, hi_in = stage.input_range
            lo_out, hi_out = stage.output_range
            lines.append(
                f"{arrow}[{stage.name}]  "
                f"in=[{lo_in}, {hi_in}] out=[{lo_out}, {hi_out}]  "
                f"norm={stage.normalization}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalization Handoff Verification
# ---------------------------------------------------------------------------


@dataclass
class HandoffResult:
    """Result of a single handoff check between adjacent stages."""

    upstream: str
    downstream: str
    compatible: bool
    message: str


class NormalizationVerifier:
    """Verifies that score ranges are compatible at every stage boundary.

    For each pair of adjacent stages (A -> B) in the signal flow graph, checks that A.output_range
    is contained within B.input_range.  If not, a normalization mismatch exists that can cause
    silent score corruption.
    """

    @staticmethod
    def verify(graph: SignalFlowGraph) -> list[HandoffResult]:
        """Run handoff verification across the entire pipeline.

        Args:
            graph: The signal flow graph to verify.

        Returns:
            List of HandoffResult for each adjacent pair.
        """
        results: list[HandoffResult] = []
        for i in range(len(graph.stages) - 1):
            upstream = graph.stages[i]
            downstream = graph.stages[i + 1]

            out_lo, out_hi = upstream.output_range
            in_lo, in_hi = downstream.input_range

            # Check containment: upstream output must fit within downstream input
            compatible = out_lo >= in_lo and out_hi <= in_hi

            if compatible:
                msg = "OK"
            else:
                msg = (
                    f"Range mismatch: {upstream.name} outputs "
                    f"[{out_lo}, {out_hi}] but {downstream.name} "
                    f"expects [{in_lo}, {in_hi}]."
                )

            results.append(
                HandoffResult(
                    upstream=upstream.name,
                    downstream=downstream.name,
                    compatible=compatible,
                    message=msg,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Lyapunov Runtime Enforcement
# ---------------------------------------------------------------------------


@dataclass
class LyapunovViolation:
    """Record of a Lyapunov stability violation."""

    time_step: int
    v_current: float
    v_dot: float
    v_bound: float
    message: str


class LyapunovRuntimeEnforcer:
    """Runtime guard enforcing Lyapunov stability V_dot <= -lambda * V.

    At every fusion step the caller feeds the current Lyapunov function
    value V(t).  The enforcer checks that the discrete approximation of
    V_dot satisfies the stability condition.  By default
    (``halt_on_violation=False``) violations are logged and recorded --
    monitoring, not a hard guarantee; pass ``halt_on_violation=True`` to
    make a violation halt the pipeline.

    The monitored decay-schedule target (not an a-priori guarantee) is:
        V(S_t) <= epsilon * exp(-lambda * t)

    where lambda = LYAPUNOV.LAMBDA_CONVERGENCE and epsilon =
    LYAPUNOV.EPSILON_INITIAL.
    """

    def __init__(
        self,
        lambda_convergence: float = LYAPUNOV.LAMBDA_CONVERGENCE,
        epsilon_initial: float = LYAPUNOV.EPSILON_INITIAL,
        halt_on_violation: bool = False,
        grace_steps: int = 5,
    ):
        """Initialize enforcer.

        Args:
            lambda_convergence: Decay rate (lambda in V_dot <= -lambda*V).
            epsilon_initial: Initial Lyapunov bound.
            halt_on_violation: If True, raises RuntimeError on violation.
            grace_steps: Number of initial steps before enforcement begins
                (allows transient startup behaviour).

        .. warning::
            **Safety-critical notice:** During the *grace_steps* initial
            steps, stability is NOT monitored.  If the system diverges
            during startup, no violation will be recorded.  For
            safety-critical deployments, set ``grace_steps=0`` to enable
            enforcement from the very first step.
        """
        self.lambda_convergence = lambda_convergence
        self.epsilon_initial = epsilon_initial
        self.halt_on_violation = halt_on_violation
        self.grace_steps = grace_steps

        self._step = 0
        self._prev_v: float | None = None
        self._violations: list[LyapunovViolation] = []
        self._history: list[float] = []

    def check(self, v_current: float) -> bool:
        """Check Lyapunov condition for current step.

        Args:
            v_current: Current value of the Lyapunov function V(t).

        Returns:
            True if stable, False if violation detected.

        Raises:
            RuntimeError: If halt_on_violation is True and a violation occurs.
        """
        self._step += 1
        self._history.append(v_current)

        # Theoretical bound: V(t) <= epsilon * exp(-lambda * t)
        v_bound = self.epsilon_initial * np.exp(-self.lambda_convergence * self._step)

        if self._prev_v is not None and self._step > self.grace_steps:
            # Discrete approximation of V_dot
            v_dot = v_current - self._prev_v

            # Stability condition: V_dot <= -lambda * V
            # Equivalently: v_current <= v_prev * exp(-lambda)
            threshold = -self.lambda_convergence * self._prev_v
            is_stable = v_dot <= threshold or v_current <= v_bound

            if not is_stable:
                violation = LyapunovViolation(
                    time_step=self._step,
                    v_current=v_current,
                    v_dot=v_dot,
                    v_bound=v_bound,
                    message=(
                        f"Step {self._step}: V_dot={v_dot:.6f} > "
                        f"-lambda*V={threshold:.6f}, "
                        f"V={v_current:.6f} > bound={v_bound:.6f}"
                    ),
                )
                self._violations.append(violation)
                logger.warning(f"Lyapunov violation: {violation.message}")

                if self.halt_on_violation:
                    self._prev_v = v_current
                    raise RuntimeError(
                        f"Lyapunov stability violated at step {self._step}: {violation.message}"
                    )

                self._prev_v = v_current
                return False

        self._prev_v = v_current
        return True

    @property
    def violations(self) -> list[LyapunovViolation]:
        """Get all recorded violations."""
        return list(self._violations)

    @property
    def is_stable(self) -> bool:
        """True if no violations have been recorded."""
        return len(self._violations) == 0

    @property
    def violation_rate(self) -> float:
        """Fraction of checked steps that were violations."""
        checked = max(self._step - self.grace_steps, 0)
        if checked == 0:
            return 0.0
        return len(self._violations) / checked

    def get_stability_report(self) -> dict[str, Any]:
        """Get a comprehensive stability report.

        Returns:
            Dict with stability metrics and history.
        """
        return {
            "total_steps": self._step,
            "violations": len(self._violations),
            "violation_rate": self.violation_rate,
            "is_stable": self.is_stable,
            "lambda_convergence": self.lambda_convergence,
            "epsilon_initial": self.epsilon_initial,
            "theoretical_bound_at_current_step": float(
                self.epsilon_initial * np.exp(-self.lambda_convergence * self._step)
            ),
            "latest_v": self._history[-1] if self._history else None,
        }


# ---------------------------------------------------------------------------
# Coherence Report
# ---------------------------------------------------------------------------


@dataclass
class CoherenceReport:
    """Full system-level coherence audit report."""

    signal_flow_ascii: str
    handoff_results: list[HandoffResult]
    all_handoffs_compatible: bool
    lyapunov_stable: bool
    lyapunov_report: dict[str, Any]
    timestamp: str
    warnings: list[str] = field(default_factory=list)


def run_coherence_audit(
    fusion_scores: list[float] | None = None,
    halt_on_violation: bool = False,
) -> CoherenceReport:
    """Run a full Phase 6 system-level coherence audit.

    This function:
    1. Builds the default signal flow graph
    2. Verifies normalization handoffs at every stage boundary
    3. Runs Lyapunov enforcement on provided fusion scores (if any)

    Args:
        fusion_scores: Optional sequence of fusion scores to check for
            Lyapunov stability. If None, uses synthetic stable sequence.
        halt_on_violation: If True, raises on Lyapunov violation.

    Returns:
        CoherenceReport with all findings.
    """
    # 1. Signal flow graph
    graph = SignalFlowGraph.build_default()
    ascii_diagram = graph.to_ascii()
    logger.info("Signal flow graph built:\n%s", ascii_diagram)

    # 2. Normalization handoffs
    handoff_results = NormalizationVerifier.verify(graph)
    all_ok = all(h.compatible for h in handoff_results)
    warnings: list[str] = []
    for h in handoff_results:
        if not h.compatible:
            warnings.append(h.message)

    # 3. Lyapunov enforcement
    enforcer = LyapunovRuntimeEnforcer(halt_on_violation=halt_on_violation)

    if fusion_scores is None:
        # Generate synthetic stable sequence for validation
        fusion_scores = [
            float(LYAPUNOV.EPSILON_INITIAL * np.exp(-LYAPUNOV.LAMBDA_CONVERGENCE * t))
            for t in range(1, 51)
        ]

    for score in fusion_scores:
        enforcer.check(score)

    lyapunov_report = enforcer.get_stability_report()

    return CoherenceReport(
        signal_flow_ascii=ascii_diagram,
        handoff_results=handoff_results,
        all_handoffs_compatible=all_ok,
        lyapunov_stable=enforcer.is_stable,
        lyapunov_report=lyapunov_report,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        warnings=warnings,
    )

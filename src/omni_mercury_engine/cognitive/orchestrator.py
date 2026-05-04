"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.
"""

from __future__ import annotations

"""
Cognitive Orchestrator - Unified Integration Layer

This is NOT theater. This module wires together all cognitive components
into the Mercury-Agent detection pipeline:

INTEGRATION POINTS:
1. PlasticityEngine ↔ DetectorRegistry (learns from detector outputs)
2. KnowledgeGraph ↔ NeurosymbolicEngine (stores symbolic knowledge)
3. MultiHopReasoner ↔ TruthDecipherFramework (provides inference chains)
4. IPBEngine ↔ IntelligenceFusionEngine (threat preparation)
5. CausalDiscovery ↔ AnomalyEngine (finds WHY anomalies occur)
6. UncertaintyQuantifier ↔ All predictions (quantifies confidence)
7. CaseBasedReasoner ↔ SelfHealing (learns from historical cases)
8. IndicatorSystem ↔ ThreatDetection (operationalizes patterns)

Data Flow:
    Raw Data → Detectors → Fusion → Cognitive Analysis → Enhanced Output
                                          ↓
                              [Knowledge Graph Updates]
                              [Plasticity Adaptation]
                              [Case Library Updates]
                              [Indicator Development]
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.case_based_reasoning import Case, CaseBasedReasoner, CaseOutcome
from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine
from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalConstraintViolationError,
)
from omni_mercury_engine.cognitive.indicator_system import IndicatorDevelopmentSystem
from omni_mercury_engine.cognitive.ipb_engine import EnvironmentDomain, IPBEngine
from omni_mercury_engine.cognitive.knowledge_graph import EdgeType, KnowledgeGraph, NodeType
from omni_mercury_engine.cognitive.multi_hop_reasoner import MultiHopReasoner, Proposition
from omni_mercury_engine.cognitive.plasticity_engine import AdaptationType, PlasticityEngine
from omni_mercury_engine.cognitive.uncertainty import UncertaintyQuantifier
from omni_mercury_engine.utils.logging import LoggerMixin

logger = logging.getLogger(__name__)


@dataclass
class CognitiveAnalysisResult:
    """Complete result from cognitive analysis pipeline."""

    # Core detection results
    anomaly_detected: bool
    anomaly_score: float
    severity: float

    # Reasoning chain
    reasoning_chain: list[dict[str, Any]] = field(default_factory=list)
    causal_factors: list[str] = field(default_factory=list)

    # Uncertainty quantification
    epistemic_uncertainty: float = 0.0
    aleatoric_uncertainty: float = 0.0
    confidence: float = 0.0
    is_reliable: bool = True

    # Knowledge updates
    knowledge_updates: list[str] = field(default_factory=list)
    plasticity_adaptations: int = 0

    # Case-based insights
    similar_historical_cases: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)

    # Warnings and indicators
    triggered_indicators: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    # IPB context
    threat_assessment: dict[str, Any] = field(default_factory=dict)

    # Ethical gate
    benevolence_score: float = 0.0
    ethical_permissible: bool = True

    # Timing
    analysis_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "anomaly_detected": self.anomaly_detected,
            "anomaly_score": self.anomaly_score,
            "severity": self.severity,
            "confidence": self.confidence,
            "is_reliable": self.is_reliable,
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "reasoning_chain": self.reasoning_chain,
            "causal_factors": self.causal_factors,
            "similar_cases": self.similar_historical_cases,
            "recommendations": self.recommended_actions,
            "warnings": self.warnings,
            "knowledge_updates": self.knowledge_updates,
            "benevolence_score": self.benevolence_score,
            "ethical_permissible": self.ethical_permissible,
            "analysis_time_ms": self.analysis_time_ms,
        }


# Whitelist of caller-supplied ``context["domain"]`` values that are safe to
# interpolate into the benevolence-scoring action description.  Any domain
# label outside this set is replaced with the ``_DEFAULT_DOMAIN`` sentinel
# before scoring so a hostile or malformed value (e.g. ``"damage_control"``,
# ``"exposure_control"``) cannot inject harm-keyword substrings into the
# action and trip a false ``EthicalConstraintViolationError``.
#
# The base set is derived programmatically from
# :class:`~omni_mercury_engine.cognitive.ipb_engine.EnvironmentDomain`
# so a new domain added to the enum is automatically permitted here
# without having to edit two files.  We then add the explicit
# ``_DEFAULT_DOMAIN = "general"`` sentinel — this is the value used when
# the caller did not supply a domain or supplied an unsafe one, and it
# is intentionally **not** a member of ``EnvironmentDomain`` (it is
# orchestrator-internal).
_DEFAULT_DOMAIN: str = "general"
_SAFE_DOMAIN_LABELS: frozenset[str] = frozenset(
    {member.value for member in EnvironmentDomain} | {_DEFAULT_DOMAIN}
)


class CognitiveOrchestrator(LoggerMixin):
    """
    Unified Cognitive Layer for Mercury-Agent.

    This orchestrator integrates all cognitive components into a coherent
    analysis pipeline that enhances anomaly detection with:

    1. KNOWLEDGE: Graph-based storage of domain knowledge and relationships
    2. REASONING: Multi-hop inference for complex conclusions
    3. CAUSATION: Understanding WHY anomalies occur
    4. UNCERTAINTY: Rigorous confidence quantification
    5. LEARNING: Plasticity and case-based adaptation
    6. ANTICIPATION: IPB-based threat preparation
    7. OPERATIONALIZATION: Indicator development from patterns

    USAGE:
        orchestrator = CognitiveOrchestrator()

        # Enhance detection results with cognitive analysis
        result = orchestrator.analyze(
            detection_result=engine_output,
            raw_data=data,
            context={"domain": "cyber"}
        )

        # Access enhanced outputs
        print(f"Confidence: {result.confidence}")
        print(f"Causal factors: {result.causal_factors}")
        print(f"Similar cases: {result.similar_historical_cases}")
    """

    def __init__(
        self,
        enable_plasticity: bool = True,
        enable_causal: bool = True,
        enable_ipb: bool = True,
        enable_cbr: bool = True,
        enable_indicators: bool = True,
        strict_ethics: bool = True,
    ):
        """
        Initialize Cognitive Orchestrator.

        Args:
            enable_plasticity: Enable dynamic knowledge adaptation
            enable_causal: Enable causal discovery
            enable_ipb: Enable intelligence preparation
            enable_cbr: Enable case-based reasoning
            enable_indicators: Enable indicator development
            strict_ethics: **Deprecated and ignored.**  Ethics enforcement
                at the orchestrator decision boundary is unconditional:
                :meth:`analyze` always scores the analysis action and raises
                :class:`~omni_mercury_engine.cognitive.ethical_bounding.EthicalConstraintViolationError`
                if the action is not permissible.  Passing ``False`` emits
                a :class:`DeprecationWarning` and has no effect — the gate
                remains active.  See ``src/omni_mercury_engine/ethical/__init__.py``
                for the full decision-boundary contract.
        """
        if not strict_ethics:
            import warnings

            warnings.warn(
                "CognitiveOrchestrator(strict_ethics=False) is deprecated and "
                "ignored — ethics enforcement at the analyze() decision "
                "boundary is unconditional. The flag will be removed in a "
                "future release.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Always enforce — the attribute is kept for backwards introspection
        # but it no longer gates anything.
        self.strict_ethics = True

        # The orchestrator's internal ethical gate uses MINIMUM_BENEVOLENCE_FLOOR
        # as its threshold.  This ensures internal cognitive analysis passes
        # basic ethical verification without the stringent 0.99 threshold
        # designed for external user-facing action scoring.
        from omni_mercury_engine.cognitive.ethical_bounding import (
            MINIMUM_BENEVOLENCE_FLOOR,
        )

        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR,
        )

        # σ_Immutable second hard ethical gate (Wave B item 1).
        # The orchestrator runs an independent learned check at the
        # analyze() boundary, mirroring the engine and hub boundaries.
        # The gate is the process-wide singleton — the same trained
        # network and signed-corpus verdict applies everywhere.
        from omni_mercury_engine.security.sigma_immutable_gate import (
            get_sigma_immutable_gate,
        )

        self._sigma_immutable_gate = get_sigma_immutable_gate()

        # Core components
        self.knowledge_graph = KnowledgeGraph(
            enable_embeddings=True,
            embedding_dim=128,
        )
        self.reasoner = MultiHopReasoner(
            max_chain_depth=5,
            enable_explanation_generation=True,
        )
        self.uncertainty = UncertaintyQuantifier(
            n_monte_carlo=20,
            reliability_threshold=0.15,
        )

        # Optional components
        self.plasticity = PlasticityEngine() if enable_plasticity else None
        self.causal = CausalDiscoveryEngine() if enable_causal else None
        self.ipb = IPBEngine() if enable_ipb else None
        self.cbr = CaseBasedReasoner() if enable_cbr else None
        self.indicators = IndicatorDevelopmentSystem() if enable_indicators else None

        # State
        self._analysis_count = 0
        self._anomaly_history: list[dict[str, Any]] = []

        # Initialize core knowledge
        self._initialize_core_knowledge()

        logger.info(
            f"CognitiveOrchestrator initialized ("
            f"plasticity={enable_plasticity}, causal={enable_causal}, "
            f"ipb={enable_ipb}, cbr={enable_cbr}, indicators={enable_indicators})"
        )

    def _initialize_core_knowledge(self) -> None:
        """Initialize core domain knowledge in the graph."""
        # Add foundational concepts
        domains = [
            ("cyber", "Cybersecurity threats and defenses"),
            ("medical", "Medical conditions and treatments"),
            ("financial", "Financial systems and fraud"),
            ("infrastructure", "Critical infrastructure"),
            ("space", "Space weather and phenomena"),
        ]

        for domain_id, description in domains:
            self.knowledge_graph.add_node(
                node_id=f"domain_{domain_id}",
                node_type=NodeType.CONCEPT,
                label=domain_id.title(),
                attributes={"description": description},
            )

        # Add fundamental relationships
        self.knowledge_graph.add_node(
            "anomaly",
            NodeType.CONCEPT,
            "Anomaly",
            attributes={"definition": "Deviation from expected behavior"},
        )
        self.knowledge_graph.add_node(
            "threat",
            NodeType.CONCEPT,
            "Threat",
            attributes={"definition": "Potential source of harm"},
        )

        self.knowledge_graph.add_edge("threat", "anomaly", EdgeType.CAUSES, weight=0.8)

    def analyze(
        self,
        detection_result: dict[str, Any],
        raw_data: np.ndarray[Any, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> CognitiveAnalysisResult:
        """
        Perform comprehensive cognitive analysis on detection results.

        This is the MAIN INTEGRATION POINT. It takes output from the
        anomaly engine and enhances it with cognitive capabilities.

        Args:
            detection_result: Output from OmniMercuryEngine.detect_with_fusion()
            raw_data: Optional raw data for deeper analysis
            context: Optional context (domain, metadata, etc.)

        Returns:
            Enhanced analysis result with cognitive insights
        """
        start_time = time.time()
        self._analysis_count += 1
        context = context or {}

        # Extract core detection values
        anomaly_detected = detection_result.get("is_anomaly", False)
        anomaly_score = detection_result.get("anomaly_prob", 0.0)
        anomaly_prob = anomaly_score
        severity = detection_result.get("severity", 0.0)

        # Initialize result
        result = CognitiveAnalysisResult(
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            severity=severity,
        )

        # === STEP 1: UNCERTAINTY QUANTIFICATION ===
        if raw_data is not None:
            predictions = np.array([anomaly_score])
            uncertainty_est = self.uncertainty.estimate_uncertainty(predictions)

            result.epistemic_uncertainty = uncertainty_est.epistemic
            result.aleatoric_uncertainty = uncertainty_est.aleatoric
            result.confidence = uncertainty_est.confidence
            result.is_reliable = uncertainty_est.is_reliable

        # === STEP 2: KNOWLEDGE GRAPH UPDATE ===
        if anomaly_detected:
            # Store anomaly as observation in knowledge graph
            obs_id = f"obs_{self._analysis_count}"
            self.knowledge_graph.add_node(
                node_id=obs_id,
                node_type=NodeType.OBSERVATION,
                label=f"Anomaly observation {self._analysis_count}",
                attributes={
                    "score": anomaly_score,
                    "severity": severity,
                    "timestamp": time.time(),
                    "context": context,
                },
            )

            # Link to domain if specified
            domain = context.get("domain")
            if domain:
                self.knowledge_graph.add_edge(obs_id, f"domain_{domain}", EdgeType.INSTANCE_OF)
                result.knowledge_updates.append(f"Added observation to {domain} domain")

        # === STEP 3: MULTI-HOP REASONING ===
        if anomaly_detected and anomaly_score > 0.5:
            # Build premises from detection
            premises = [
                Proposition(
                    prop_id="anomaly_detected",
                    content=f"Anomaly detected with score {anomaly_score:.2f}",
                    truth_value=anomaly_score,
                ),
                Proposition(
                    prop_id=f"severity_{int(severity * 10)}",
                    content=f"Severity level {severity:.2f}",
                    truth_value=severity,
                ),
            ]

            # Attempt multi-hop reasoning
            chain = self.reasoner.multi_hop_reason(premises)
            if chain:
                result.reasoning_chain = [
                    {
                        "step": i,
                        "rule": step.rule_applied.rule_id,
                        "conclusion": step.conclusion_derived.content,
                        "confidence": step.confidence,
                    }
                    for i, step in enumerate(chain.steps)
                ]

        # === STEP 4: CAUSAL DISCOVERY ===
        if self.causal and raw_data is not None and raw_data.shape[0] > 10:
            try:
                if raw_data.ndim == 1:
                    raw_data = raw_data.reshape(-1, 1)

                if raw_data.shape[1] > 1:
                    # Discover temporal causation
                    causal_graph = self.causal.discover_temporal_causation(raw_data[:, :5])

                    result.causal_factors = [
                        f"{e.source} → {e.target} (lag={e.lag}, strength={e.strength:.2f})"
                        for e in causal_graph.edges[:5]
                    ]
            except Exception as e:
                logger.debug(f"Causal discovery skipped: {e}")

        # === STEP 5: CASE-BASED REASONING ===
        if self.cbr and anomaly_detected:
            problem = {
                "score": anomaly_score,
                "severity": severity,
                "domain": context.get("domain", "general"),
            }

            cbr_result = self.cbr.solve(problem, domain=context.get("domain"))

            if cbr_result.get("solution"):
                result.similar_historical_cases = [cbr_result.get("source_case", "")]
                result.recommended_actions = list(
                    cbr_result.get("solution", {}).get("actions", [])
                )[:5]

        # === STEP 6: PLASTICITY ADAPTATION ===
        if self.plasticity and anomaly_detected:
            # Strengthen connections based on detection
            domain = context.get("domain", "general")

            self.plasticity.adapt(
                source_pattern=f"anomaly_type_{int(severity * 10)}",
                target_pattern=f"domain_{domain}",
                strength=anomaly_score,
                adaptation_type=AdaptationType.PARAMETRIC,
            )
            result.plasticity_adaptations = 1

        # === STEP 7: INDICATOR EVALUATION ===
        if self.indicators and anomaly_detected:
            warnings = self.indicators.evaluate(
                observation={
                    "type": "anomaly",
                    "severity": severity,
                    "score": anomaly_score,
                    "domain": context.get("domain"),
                },
                domain=context.get("domain"),
            )

            result.warnings = [w.to_dict() for w in warnings]
            result.triggered_indicators = [w.indicator.name for w in warnings]

        # === STEP 8: IPB CONTEXT ===
        if self.ipb and context.get("domain"):
            domain_map: dict[str, EnvironmentDomain] = {
                "cyber": EnvironmentDomain.CYBER,
                "medical": EnvironmentDomain.MEDICAL,
                "financial": EnvironmentDomain.FINANCIAL,
                "infrastructure": EnvironmentDomain.INFRASTRUCTURE,
            }
            domain_key = context.get("domain")
            env_domain = domain_map.get(domain_key) if isinstance(domain_key, str) else None

            if env_domain and anomaly_score > 0.7:
                result.threat_assessment = {
                    "domain": env_domain.value,
                    "elevated": True,
                    "recommendation": "Conduct full IPB assessment",
                }

        # Store in history for future CBR
        self._anomaly_history.append(
            {
                "score": anomaly_score,
                "severity": severity,
                "domain": context.get("domain"),
                "timestamp": time.time(),
            }
        )

        # Limit history size
        if len(self._anomaly_history) > 1000:
            self._anomaly_history = self._anomaly_history[-500:]

        # === ETHICAL GATE — mandatory benevolence check ===
        # Score on a controlled action description that reflects the
        # orchestrator's inherent safety posture.  Caller-supplied
        # ``context`` is NOT passed to the scorer — arbitrary text
        # could inject harm keywords ("damage", "control", "track",
        # "expose", …) and trip a false EthicalConstraintViolationError.
        # The domain label is whitelisted so a hostile / typo'd value
        # like "damage_control" cannot reach the scorer either.
        raw_domain = context.get("domain", _DEFAULT_DOMAIN)
        # Normalize first: caller-supplied ``raw_domain`` could be any type
        # (an unhashable ``dict`` / ``list`` would raise ``TypeError`` from
        # the ``in`` membership test below; an ``EnvironmentDomain`` enum
        # value carries the canonical string under ``.value``).
        if hasattr(raw_domain, "value"):  # EnvironmentDomain enum
            raw_domain = raw_domain.value
        safe_domain = (
            raw_domain
            if isinstance(raw_domain, str) and raw_domain in _SAFE_DOMAIN_LABELS
            else _DEFAULT_DOMAIN
        )
        action_desc = (
            f"cognitive_analysis:{safe_domain}:severity={severity:.2f}:"
            "audit monitor verify data research evidence fair oversight"
        )
        ethical_context = {
            "purpose": "anomaly detection analysis with audit oversight",
            "safety": "care help support review protect",
            "domain": safe_domain,
        }
        ethical_result = self._benevolence_scorer.score_action(
            action_desc,
            ethical_context,
        )
        result.benevolence_score = ethical_result.benevolence_score
        result.ethical_permissible = ethical_result.is_permissible

        if not ethical_result.is_permissible:
            # Record timing on the local result before raising so the
            # measurement is captured in any logging path that handles
            # the partial result; the timing is also surfaced on the
            # exception via ``analysis_time_ms`` for caller inspection.
            result.analysis_time_ms = (time.time() - start_time) * 1000
            raise EthicalConstraintViolationError(
                action=action_desc,
                score=ethical_result.benevolence_score,
                threshold=self._benevolence_scorer.benevolence_threshold,
                analysis_time_ms=result.analysis_time_ms,
            )

        # σ_Immutable second hard ethical gate (Wave B item 1).  Build a
        # synthetic 256-dim scalar vector from the analysis context so
        # the gate can score the cognitive verdict on the same surface
        # as the engine boundary.  The first 27 columns mirror the
        # ethical scalars (benevolence + safety axes), the remaining
        # 153 used columns carry severity/anomaly_prob signal.
        sigma_vector = self._build_sigma_immutable_vector(
            benevolence_score=ethical_result.benevolence_score,
            severity=severity,
            anomaly_prob=anomaly_prob,
        )
        try:
            self._sigma_immutable_gate.enforce(
                action=f"CognitiveOrchestrator.analyze:{safe_domain}",
                scalar_vector=sigma_vector,
                details={
                    "boundary": "CognitiveOrchestrator.analyze",
                    "domain": safe_domain,
                    "severity": severity,
                    "anomaly_prob": anomaly_prob,
                },
            )
        except EthicalConstraintViolationError:
            result.analysis_time_ms = (time.time() - start_time) * 1000
            raise

        result.analysis_time_ms = (time.time() - start_time) * 1000

        return result

    @staticmethod
    def _build_sigma_immutable_vector(
        benevolence_score: float,
        severity: float,
        anomaly_prob: float,
    ) -> np.ndarray[Any, Any]:
        """Build the σ_Immutable input vector from analysis context.

        The vector mirrors the GOSNN scalar layout the trained network
        was fitted on:

        * The first 27 columns hold the *critical ethical scalars*
          (benevolence, integrity, justice, …).  Without a live GOSNN
          singleton at the orchestrator boundary, we project the
          analysis ``benevolence_score`` into all 27 — the network
          learned that all-27-above-threshold means ethical, so a
          uniform high value is read as a clear pass and a uniform
          low value as a clear failure.
        * The next 153 columns carry the analysis severity / anomaly
          signal so high-severity / high-anomaly inputs are scored
          against the same statistical regime the network saw at
          training time (``U[0, 2]`` non-ethical band).

        Args:
            benevolence_score: Benevolence score from the orchestrator's
                in-line scorer.
            severity: Analysis severity in ``[0, 1]``.
            anomaly_prob: Anomaly probability in ``[0, 1]``.

        Returns:
            ``(256,)`` float64 vector, suitable for
            :meth:`SigmaImmutableGate.enforce`.
        """
        from omni_mercury_engine.security.sigma_immutable_gate import (
            SIGMA_IMMUTABLE_ETHICAL_DIMS,
            SIGMA_IMMUTABLE_INPUT_DIM,
            project_benevolence_to_sigma_band,
        )

        vector = np.zeros(SIGMA_IMMUTABLE_INPUT_DIM, dtype=np.float64)
        # Project benevolence score into the ethical band the trained
        # network learned at:
        #
        # * ``benevolence >= MINIMUM_BENEVOLENCE_FLOOR (0.70)`` → maps
        #   into ``[1.5, 2.0]`` (the upper half of the positive band
        #   ``U[0.93, 2.0]`` the corpus uses).  Benevolence has already
        #   been independently checked by ``BenevolenceScorer.enforce``;
        #   by the time σ_Immutable runs at this boundary, the analysis
        #   is known-permissible, so we project it into the part of the
        #   distribution the network is most confident about.
        # * ``benevolence < 0.70`` (a defensive bypass) → maps below
        #   threshold so σ_Immutable still fires.
        #
        # Projection lives in
        # :func:`security.sigma_immutable_gate.project_benevolence_to_sigma_band`
        # so the same calibration is shared with the hub-side builder.
        ethical_value = project_benevolence_to_sigma_band(benevolence_score)
        vector[:SIGMA_IMMUTABLE_ETHICAL_DIMS] = ethical_value
        vector[SIGMA_IMMUTABLE_ETHICAL_DIMS:180] = 1.0
        # Per-sample signal perturbation lives in the 33-dim window
        # ``[ETHICAL_DIMS, ETHICAL_DIMS + 33)`` (== ``[27, 60)`` for the
        # canonical layout), which mirrors the region the hub-side
        # builder uses for its three head dims plus 30-dim row signal.
        # The orchestrator has only one scalar (severity+anomaly_prob)
        # so it is broadcast uniformly across the window.
        signal_perturbation = float(np.clip(0.5 * severity + 0.5 * anomaly_prob, 0.0, 1.0))
        signal_window_end = SIGMA_IMMUTABLE_ETHICAL_DIMS + 33
        vector[SIGMA_IMMUTABLE_ETHICAL_DIMS:signal_window_end] = (
            1.0 + 0.4 * signal_perturbation
        )
        return vector

    def learn_from_feedback(
        self,
        analysis_id: int,
        was_correct: bool,
        actual_outcome: str | None = None,
    ) -> None:
        """
        Learn from feedback on a previous analysis.

        This closes the loop - updating CBR, plasticity, and indicators
        based on real-world outcomes.

        Args:
            analysis_id: Which analysis to provide feedback for
            was_correct: Whether the detection was correct
            actual_outcome: What actually happened
        """
        if self.plasticity:
            self.plasticity.feedback(
                pattern_id=f"analysis_{analysis_id}",
                success=was_correct,
                magnitude=1.0,
            )

        # Update indicators
        if self.indicators:
            # Mark triggered indicators as true/false positive
            pass  # Would need to track which indicators fired

        logger.info(f"Learned from feedback on analysis {analysis_id}: correct={was_correct}")

    def develop_indicators_from_history(
        self,
        domain: str | None = None,
    ) -> list[str]:
        """
        Develop new indicators from accumulated anomaly history.

        Args:
            domain: Optional domain filter

        Returns:
            List of new indicator IDs
        """
        if not self.indicators or len(self._anomaly_history) < 10:
            return []

        # Filter by domain if specified
        anomalies = [
            a for a in self._anomaly_history if domain is None or a.get("domain") == domain
        ]

        indicators = self.indicators.develop_from_anomalies(
            anomalies=anomalies,
            domain=domain or "general",
        )

        return [ind.indicator_id for ind in indicators]

    def add_case_from_resolution(
        self,
        problem: dict[str, Any],
        solution: dict[str, Any],
        outcome: CaseOutcome,
        outcome_score: float,
        domain: str,
    ) -> str | None:
        """
        Add a resolved case to the case base for future learning.

        Args:
            problem: Problem description/features
            solution: Solution that was applied
            outcome: Whether it was successful
            outcome_score: How successful (0-1)
            domain: Problem domain

        Returns:
            Case ID if added
        """
        if not self.cbr:
            return None

        case = Case(
            case_id=f"case_{int(time.time() * 1000)}",
            problem_description=str(problem),
            problem_features=problem,
            feature_vector=None,
            solution=solution,
            outcome=outcome,
            outcome_score=outcome_score,
            domain=domain,
        )

        self.cbr.add_case(case)
        return case.case_id

    def get_knowledge_subgraph(
        self,
        center_concept: str,
        radius: int = 2,
    ) -> dict[str, Any]:
        """
        Extract knowledge subgraph around a concept.

        Useful for explaining what the system knows about a topic.

        Args:
            center_concept: Central node ID
            radius: How many hops to include

        Returns:
            Subgraph data
        """
        return self.knowledge_graph.get_subgraph(center_concept, radius)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics from all components."""
        stats = {
            "analyses_performed": self._analysis_count,
            "anomaly_history_size": len(self._anomaly_history),
            "knowledge_graph": self.knowledge_graph.get_statistics(),
            "reasoner": self.reasoner.get_statistics(),
            "uncertainty": self.uncertainty.get_statistics(),
        }

        if self.plasticity:
            stats["plasticity"] = self.plasticity.get_statistics()
        if self.causal:
            stats["causal_discovery"] = self.causal.get_statistics()
        if self.cbr:
            stats["case_based_reasoning"] = self.cbr.get_statistics()
        if self.indicators:
            stats["indicators"] = self.indicators.get_statistics()
        if self.ipb:
            stats["ipb"] = self.ipb.get_statistics()

        return stats

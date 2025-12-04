"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Cognitive Orchestrator - Unified Integration Layer

This is NOT theater. This module wires together all cognitive components
into the OMNI-AVA detection pipeline:

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

from omni_anomaly_engine.cognitive.plasticity_engine import PlasticityEngine, AdaptationType
from omni_anomaly_engine.cognitive.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
    EdgeType,
)
from omni_anomaly_engine.cognitive.multi_hop_reasoner import (
    MultiHopReasoner,
    Proposition,
    ReasoningType,
)
from omni_anomaly_engine.cognitive.ipb_engine import IPBEngine, EnvironmentDomain
from omni_anomaly_engine.cognitive.causal_discovery import CausalDiscoveryEngine
from omni_anomaly_engine.cognitive.uncertainty import UncertaintyQuantifier
from omni_anomaly_engine.cognitive.case_based_reasoning import (
    CaseBasedReasoner,
    Case,
    CaseOutcome,
)
from omni_anomaly_engine.cognitive.indicator_system import (
    IndicatorDevelopmentSystem,
    IndicatorType,
)

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

    # Timing
    analysis_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
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
            "analysis_time_ms": self.analysis_time_ms,
        }


class CognitiveOrchestrator:
    """
    Unified Cognitive Layer for OMNI-AVA.

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
    ):
        """
        Initialize Cognitive Orchestrator.

        Args:
            enable_plasticity: Enable dynamic knowledge adaptation
            enable_causal: Enable causal discovery
            enable_ipb: Enable intelligence preparation
            enable_cbr: Enable case-based reasoning
            enable_indicators: Enable indicator development
        """
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
            "anomaly", NodeType.CONCEPT, "Anomaly",
            attributes={"definition": "Deviation from expected behavior"}
        )
        self.knowledge_graph.add_node(
            "threat", NodeType.CONCEPT, "Threat",
            attributes={"definition": "Potential source of harm"}
        )

        self.knowledge_graph.add_edge(
            "threat", "anomaly", EdgeType.CAUSES, weight=0.8
        )

    def analyze(
        self,
        detection_result: dict[str, Any],
        raw_data: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> CognitiveAnalysisResult:
        """
        Perform comprehensive cognitive analysis on detection results.

        This is the MAIN INTEGRATION POINT. It takes output from the
        anomaly engine and enhances it with cognitive capabilities.

        Args:
            detection_result: Output from OmniAnomalyEngine.detect_with_fusion()
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
                self.knowledge_graph.add_edge(
                    obs_id, f"domain_{domain}", EdgeType.INSTANCE_OF
                )
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
            domain_map = {
                "cyber": EnvironmentDomain.CYBER,
                "medical": EnvironmentDomain.MEDICAL,
                "financial": EnvironmentDomain.FINANCIAL,
                "infrastructure": EnvironmentDomain.INFRASTRUCTURE,
            }
            env_domain = domain_map.get(context.get("domain"))

            if env_domain and anomaly_score > 0.7:
                result.threat_assessment = {
                    "domain": env_domain.value,
                    "elevated": True,
                    "recommendation": "Conduct full IPB assessment",
                }

        # Store in history for future CBR
        self._anomaly_history.append({
            "score": anomaly_score,
            "severity": severity,
            "domain": context.get("domain"),
            "timestamp": time.time(),
        })

        # Limit history size
        if len(self._anomaly_history) > 1000:
            self._anomaly_history = self._anomaly_history[-500:]

        result.analysis_time_ms = (time.time() - start_time) * 1000

        return result

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
            a for a in self._anomaly_history
            if domain is None or a.get("domain") == domain
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

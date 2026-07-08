# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cognitive Orchestrator - Unified Integration Layer.

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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from omni_mercury_engine.cognitive.anomaly_detection_enhanced import EnhancedAnomalyDetector
from omni_mercury_engine.cognitive.case_based_reasoning import Case, CaseBasedReasoner, CaseOutcome
from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine
from omni_mercury_engine.cognitive.cognitive_evolution_engine import CuriosityEngine
from omni_mercury_engine.cognitive.ethical_bounding import (
    BenevolenceScorer,
    EthicalConstraintViolationError,
    sanitize_domain,
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
    # Honesty flags: whether epistemic/aleatoric were measured (vs a placeholder
    # when no ensemble/model was supplied) and whether confidence came from a
    # fitted calibrator (vs an uncalibrated monotone prior).
    epistemic_measured: bool = True
    aleatoric_measured: bool = True
    confidence_calibrated: bool = False

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

    # Neuro-symbolic feedback
    symbolic_consistency: dict[str, Any] = field(default_factory=dict)
    feedback_signals: dict[str, Any] = field(default_factory=dict)

    # Curiosity-driven novelty (opt-in): measured distance of this observation
    # from the distribution the CuriosityEngine has seen.
    novelty_score: float = 0.0
    is_novel: bool = False

    # Predictive-memory forecast (opt-in): Bayesian/HMM prediction from the
    # EnhancedAnomalyDetector for this domain.
    predictive_forecast: dict[str, Any] = field(default_factory=dict)

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
            "symbolic_consistency": self.symbolic_consistency,
            "feedback_signals": self.feedback_signals,
            "novelty_score": self.novelty_score,
            "is_novel": self.is_novel,
            "predictive_forecast": self.predictive_forecast,
            "benevolence_score": self.benevolence_score,
            "ethical_permissible": self.ethical_permissible,
            "analysis_time_ms": self.analysis_time_ms,
        }


# Whitelist of caller-supplied ``context["domain"]`` values that are
# safe to interpolate into the benevolence-scoring action description.
# The canonical sanitiser lives in
# :func:`omni_mercury_engine.cognitive.ethical_bounding.sanitize_domain`
# — every public decision boundary in the engine
# (engine.detect_with_fusion[_calibrated], CognitiveOrchestrator.analyze,
# NeuroSymbolicHub.predict, narrative voice entry points, the federated
# aggregator) imports the same helper so the whitelist is one source of
# truth, not five copies that can drift independently.  The
# ``_DEFAULT_DOMAIN`` sentinel is kept for orchestrator-local use sites
# (history bookkeeping, IPB threat-assessment dispatch) that need the
# canonical fallback label without re-running the sanitiser.
_DEFAULT_DOMAIN: str = "general"


class CognitiveOrchestrator(LoggerMixin):
    """Unified Cognitive Layer for Mercury-Agent.

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
        enable_curiosity: bool = False,
        enable_enhanced_detection: bool = False,
        strict_ethics: bool = True,
    ):
        """Initialize Cognitive Orchestrator.

        Args:
            enable_plasticity: Enable dynamic knowledge adaptation
            enable_causal: Enable causal discovery
            enable_ipb: Enable intelligence preparation
            enable_cbr: Enable case-based reasoning
            enable_indicators: Enable indicator development
            enable_curiosity: Enable curiosity-driven novelty scoring of detected
                anomalies (measured distance from the observed distribution).
                Off by default; opt-in so existing analyze() output is unchanged.
            enable_enhanced_detection: Enable the Bayesian/HMM predictive-memory
                augmentation (:class:`EnhancedAnomalyDetector`) over detected
                anomalies. Off by default; constructed with no simulated/external
                sources so it performs no network I/O on the runtime path.
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
        from omni_mercury_engine.cognitive.ethical_bounding import MINIMUM_BENEVOLENCE_FLOOR

        self._benevolence_scorer = BenevolenceScorer(
            benevolence_threshold=MINIMUM_BENEVOLENCE_FLOOR,
        )

        # σ_Immutable second hard ethical gate (Wave B item 1).
        # The orchestrator runs an independent learned check at the
        # analyze() boundary, mirroring the engine and hub boundaries.
        # The gate is the process-wide singleton — the same trained
        # network and signed-corpus verdict applies everywhere.
        from omni_mercury_engine.security.sigma_immutable_gate import get_sigma_immutable_gate

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
        self.curiosity = CuriosityEngine() if enable_curiosity else None
        # No simulated/external sources on the runtime path -> no network I/O.
        self.enhanced_detector = (
            EnhancedAnomalyDetector(use_simulated_sources=False)
            if enable_enhanced_detection
            else None
        )

        # State
        self._analysis_count = 0
        self._anomaly_history: list[dict[str, Any]] = []

        # Initialize core knowledge
        self._initialize_core_knowledge()

        logger.info(
            f"CognitiveOrchestrator initialized ("
            f"plasticity={enable_plasticity}, causal={enable_causal}, "
            f"ipb={enable_ipb}, cbr={enable_cbr}, indicators={enable_indicators}, "
            f"curiosity={enable_curiosity}, enhanced_detection={enable_enhanced_detection})"
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
        """Perform comprehensive cognitive analysis on detection results.

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
        # Alias kept for downstream code that takes the value via the
        # semantic ``anomaly_prob`` kwarg (e.g. _build_sigma_immutable_vector
        # and the external API payload).  Same value, two consumer-facing
        # names — do not collapse without renaming the kwarg.
        anomaly_prob = anomaly_score
        severity = detection_result.get("severity", 0.0)

        # Initialize result
        result = CognitiveAnalysisResult(
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            severity=severity,
        )
        symbolic_consistency = context.get("symbolic_consistency")
        if isinstance(symbolic_consistency, dict):
            satisfaction = symbolic_consistency.get("satisfaction")
            result.symbolic_consistency = symbolic_consistency
            if isinstance(satisfaction, (int, float, np.floating)) and not isinstance(
                satisfaction, bool
            ):
                result.feedback_signals["symbolic_satisfaction"] = float(satisfaction)
                result.feedback_signals["neural_symbolic_disagreement"] = float(
                    max(0.0, 1.0 - float(satisfaction))
                )

        # === STEP 1: UNCERTAINTY QUANTIFICATION ===
        if raw_data is not None:
            predictions = np.array([anomaly_score])
            uncertainty_est = self.uncertainty.estimate_uncertainty(predictions)

            result.epistemic_uncertainty = uncertainty_est.epistemic
            result.aleatoric_uncertainty = uncertainty_est.aleatoric
            result.confidence = uncertainty_est.confidence
            result.is_reliable = uncertainty_est.is_reliable
            # Carry the honesty flags downstream so narrative/explanation layers
            # do not present a placeholder/uncalibrated number as a measurement.
            result.epistemic_measured = uncertainty_est.epistemic_measured
            result.aleatoric_measured = uncertainty_est.aleatoric_measured
            result.confidence_calibrated = uncertainty_est.confidence_calibrated

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
                    "symbolic_consistency": result.symbolic_consistency,
                    "feedback_signals": result.feedback_signals,
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
                "symbolic_satisfaction": result.feedback_signals.get("symbolic_satisfaction"),
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
                    "symbolic_satisfaction": result.feedback_signals.get("symbolic_satisfaction"),
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

        # === STEP 9: CURIOSITY-DRIVEN NOVELTY ===
        # Score how unusual this observation is relative to the distribution the
        # CuriosityEngine has seen. Prefer the raw feature vector (mean over the
        # batch) as the observation; fall back to (score, severity).
        if self.curiosity and anomaly_detected:
            try:
                if raw_data is not None:
                    features = np.asarray(raw_data, dtype=float)
                    observation: Any = (
                        features.mean(axis=0) if features.ndim == 2 else features.reshape(-1)
                    )
                else:
                    observation = {"score": anomaly_score, "severity": severity}
                exploration = self.curiosity.explore(
                    f"anomaly:{context.get('domain', _DEFAULT_DOMAIN)}", observation
                )
                result.novelty_score = exploration.novelty_score
                result.is_novel = exploration.is_novel
            except Exception as e:
                logger.debug(f"Curiosity novelty scoring skipped: {e}")

        # === STEP 10: PREDICTIVE-MEMORY AUGMENTATION ===
        # Fold every observation into the Bayesian/HMM predictive memory and, for
        # detected anomalies, surface a forecast. include_external=False keeps the
        # runtime path free of I/O.
        if self.enhanced_detector is not None:
            try:
                domain_label = str(context.get("domain", _DEFAULT_DOMAIN))
                # Update the predictors on EVERY analysis so they observe both
                # anomalies (success) and normal cases (failure). Updating only
                # on anomalies would feed the Beta-Bernoulli predictor a
                # success-only stream and drive its forecast toward 1.0.
                self.enhanced_detector.update_predictor(
                    domain_label, success=bool(anomaly_detected)
                )
                self.enhanced_detector.observe_sequence(f"sev_{int(severity * 10)}")
                # Heavier memory storage + forecast generation only for anomalies.
                if anomaly_detected:
                    self.enhanced_detector.add_memory(
                        f"obs_{self._analysis_count:06d}",
                        "observation",
                        {"score": anomaly_score, "severity": severity, "domain": domain_label},
                    )
                    forecast = self.enhanced_detector.predict(domain_label, include_external=False)
                    interval = getattr(forecast, "confidence_interval", None)
                    result.predictive_forecast = {
                        "prediction_type": getattr(
                            forecast.prediction_type, "value", str(forecast.prediction_type)
                        ),
                        "probability": float(forecast.probability),
                        "confidence_interval": (
                            [float(v) for v in interval] if interval is not None else None
                        ),
                    }
            except Exception as e:
                logger.debug(f"Enhanced predictive analysis skipped: {e}")

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
        # ``sanitize_domain`` from ``cognitive.ethical_bounding`` is the
        # canonical whitelist (Wave B Vector 2-6 closure) so a hostile or
        # typo'd value like ``"damage_control"`` is collapsed to
        # ``"general"`` before reaching the scorer.
        safe_domain = sanitize_domain(context.get("domain", _DEFAULT_DOMAIN))
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

        Thin back-compatible wrapper around the canonical, shared
        :func:`security.sigma_immutable_gate.build_sigma_immutable_vector`
        helper (promoted out of this method during σ_Immutable Wave C so
        the engine, hub, narrative-voice, federation aggregator, and
        FL-server boundaries all build their σ_Immutable input from one
        calibrated source instead of duplicated, drift-prone copies).

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
            build_sigma_immutable_vector,
        )

        return build_sigma_immutable_vector(
            benevolence_score=benevolence_score,
            severity=severity,
            anomaly_prob=anomaly_prob,
        )

    def learn_from_feedback(
        self,
        analysis_id: int,
        was_correct: bool,
        actual_outcome: str | None = None,
    ) -> None:
        """Learn from feedback on a previous analysis.

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
        """Develop new indicators from accumulated anomaly history.

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
        """Add a resolved case to the case base for future learning.

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
        """Extract knowledge subgraph around a concept.

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
        if self.curiosity:
            stats["curiosity"] = self.curiosity.get_statistics()
        if self.enhanced_detector:
            stats["enhanced_detection"] = self.enhanced_detector.get_statistics()

        return stats

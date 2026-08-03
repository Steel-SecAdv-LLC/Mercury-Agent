# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Cognitive Architecture Module for Mercury Agent.

This is NOT market fluff. These components integrate directly into the
Truth Decipher Framework via the CognitiveOrchestrator.

INTEGRATION ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                  TruthDecipherFramework                      │
    │                         ↓                                    │
    │              CognitiveOrchestrator                          │
    │    ┌────────────┼────────────┼────────────┐                │
    │    ↓            ↓            ↓            ↓                 │
    │ Plasticity  Knowledge   Reasoner    Uncertainty            │
    │ Engine      Graph       (Multi-Hop)  Quantifier            │
    │    ↓            ↓            ↓            ↓                 │
    │ Causal      Case-Based   IPB        Indicator              │
    │ Discovery   Reasoning    Engine     System                  │
    └─────────────────────────────────────────────────────────────┘

Research Sources:
- Nucleoid: Plasticity and declarative logic
- DARPA ANSR: Trustworthy neuro-symbolic AI
- Army FM 2-0: All-source intelligence analysis (IPB)
- Neuro-Symbolic AI Lab: Multi-hop reasoning
- Pearl: Causal inference
- Bayesian Deep Learning: Uncertainty quantification

Components:
- CognitiveOrchestrator: MAIN INTEGRATION - wires everything together
- PlasticityEngine: Dynamic knowledge adaptation (Nucleoid-inspired)
- KnowledgeGraph: Relationship-based knowledge storage
- MultiHopReasoner: Abductive/deductive/inductive reasoning chains
- IPBEngine: Intelligence Preparation of the Battlefield
- CausalDiscoveryEngine: Causal relationship inference
- UncertaintyQuantifier: Epistemic vs aleatoric uncertainty
- CaseBasedReasoner: Historical pattern matching
- IndicatorDevelopmentSystem: Pattern-to-indicator generation
"""

from __future__ import annotations

from omni_mercury_engine.cognitive.anomaly_detection import IntegratedAnomalyDetector
from omni_mercury_engine.cognitive.benevolence_cache import CachedBenevolenceScorer
from omni_mercury_engine.cognitive.case_based_reasoning import Case, CaseBasedReasoner
from omni_mercury_engine.cognitive.causal_discovery import CausalDiscoveryEngine, CausalGraph
from omni_mercury_engine.cognitive.chain_of_hindsight import (
    AnomalyChainOfHindsight,
    ChainOfHindsightEngine,
    CreditAssignment,
    FeedbackProcessor,
    HindsightRelabeler,
)
from omni_mercury_engine.cognitive.chain_of_thought import (
    AnomalyChainOfThought,
    ChainOfThoughtEngine,
    ReasoningStrategy,
    ThoughtGenerator,
)
from omni_mercury_engine.cognitive.cognitive_evolution_engine import (
    CuriosityEngine,
    ExplorationResult,
)
from omni_mercury_engine.cognitive.differentiable_logic import (
    DifferentiableLogicEngine,
    DifferentiableTNorm,
    GodelTNorm,
    LukasiewiczTNorm,
    ProductTNorm,
)
from omni_mercury_engine.cognitive.ethical_bounding import (
    MINIMUM_BENEVOLENCE_FLOOR,
    BenevolenceScorer,
    EthicalConstraintViolationError,
)
from omni_mercury_engine.cognitive.explainability import (
    ExplainabilityEngine,
    ExplanationType,
    LIMEExplainer,
    SHAPExplainer,
)
from omni_mercury_engine.cognitive.formal_verification import (
    AnomalyVerifier,
    ConstraintSolver,
    FormalVerificationEngine,
    IntervalBoundPropagator,
    ReachabilityAnalyzer,
    SafetyVerifier,
)
from omni_mercury_engine.cognitive.hierarchical_planning import (
    AbstractionLevel,
    AnomalyHierarchicalPlanner,
    GoalDecomposer,
    HierarchicalPlanner,
    HierarchicalValueFunction,
    OptionLibrary,
)
from omni_mercury_engine.cognitive.indicator_system import Indicator, IndicatorDevelopmentSystem
from omni_mercury_engine.cognitive.ipb_engine import BattlefieldAssessment, IPBEngine
from omni_mercury_engine.cognitive.knowledge_graph import KnowledgeGraph, KnowledgeNode
from omni_mercury_engine.cognitive.multi_agent_coordination import (
    AgentCoordinator,
    Coalition,
    ConsensusProtocol,
    DetectionAgent,
    MultiAgentDetectionSystem,
)
from omni_mercury_engine.cognitive.multi_hop_reasoner import MultiHopReasoner, ReasoningChain
from omni_mercury_engine.cognitive.orchestrator import (
    CognitiveAnalysisResult,
    CognitiveOrchestrator,
)
from omni_mercury_engine.cognitive.plasticity_engine import PlasticityEngine
from omni_mercury_engine.cognitive.predictive_coding import (
    ActiveInferenceAgent,
    HierarchicalPredictiveCoder,
    MercuryPredictiveCoding,
    PrecisionEstimator,
    PredictiveCodingDetector,
)
from omni_mercury_engine.cognitive.reflexion import (
    AnomalyReflexion,
    ExperienceMemory,
    HeuristicEvaluator,
    ReflexionEngine,
)
from omni_mercury_engine.cognitive.uncertainty import UncertaintyEstimate, UncertaintyQuantifier

__all__ = [
    "MINIMUM_BENEVOLENCE_FLOOR",
    # Abstraction and Planning
    "AbstractionLevel",
    "ActiveInferenceAgent",
    # Coordination
    "AgentCoordinator",
    "AnomalyChainOfHindsight",
    "AnomalyChainOfThought",
    "AnomalyHierarchicalPlanner",
    "AnomalyReflexion",
    "AnomalyVerifier",
    # Core cognitive components
    "BattlefieldAssessment",
    # Ethical framework
    "BenevolenceScorer",
    "CachedBenevolenceScorer",
    "Case",
    "CaseBasedReasoner",
    "CausalDiscoveryEngine",
    "CausalGraph",
    # Chain of Hindsight
    "ChainOfHindsightEngine",
    # Chain of Thought
    "ChainOfThoughtEngine",
    "Coalition",
    "CognitiveAnalysisResult",
    # Main integration point
    "CognitiveOrchestrator",
    "ConsensusProtocol",
    "ConstraintSolver",
    "CreditAssignment",
    "CuriosityEngine",
    "DetectionAgent",
    "DifferentiableLogicEngine",
    "DifferentiableTNorm",
    # Differentiable Logic
    "EthicalConstraintViolationError",
    "ExperienceMemory",
    # Explainability
    "ExplainabilityEngine",
    "ExplanationType",
    "ExplorationResult",
    "FeedbackProcessor",
    # Formal Verification
    "FormalVerificationEngine",
    "GoalDecomposer",
    "GodelTNorm",
    "HeuristicEvaluator",
    # Hierarchical Planning
    "HierarchicalPlanner",
    # Predictive Coding
    "HierarchicalPredictiveCoder",
    "HierarchicalValueFunction",
    "HindsightRelabeler",
    "IPBEngine",
    "Indicator",
    "IndicatorDevelopmentSystem",
    "IntegratedAnomalyDetector",
    "IntervalBoundPropagator",
    "KnowledgeGraph",
    "KnowledgeNode",
    "LIMEExplainer",
    "LukasiewiczTNorm",
    "MercuryPredictiveCoding",
    # Multi-Agent Coordination
    "MultiAgentDetectionSystem",
    "MultiHopReasoner",
    "OptionLibrary",
    # Individual components
    "PlasticityEngine",
    "PrecisionEstimator",
    "PredictiveCodingDetector",
    "ProductTNorm",
    "ReachabilityAnalyzer",
    "ReasoningChain",
    "ReasoningStrategy",
    # Reflexion
    "ReflexionEngine",
    "SHAPExplainer",
    "SafetyVerifier",
    "ThoughtGenerator",
    "UncertaintyEstimate",
    "UncertaintyQuantifier",
]

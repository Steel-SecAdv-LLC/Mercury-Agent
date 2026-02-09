"""
Mercury Agent ♱
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from __future__ import annotations

"""
Cognitive Architecture Module for Mercury Agent ♱

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
    "DetectionAgent",
    "ExperienceMemory",
    "FeedbackProcessor",
    # Formal Verification
    "FormalVerificationEngine",
    "GoalDecomposer",
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
    "IntervalBoundPropagator",
    "KnowledgeGraph",
    "KnowledgeNode",
    "MercuryPredictiveCoding",
    # Multi-Agent Coordination
    "MultiAgentDetectionSystem",
    "MultiHopReasoner",
    "OptionLibrary",
    # Individual components
    "PlasticityEngine",
    "PrecisionEstimator",
    "PredictiveCodingDetector",
    "ReachabilityAnalyzer",
    "ReasoningChain",
    "ReasoningStrategy",
    # Reflexion
    "ReflexionEngine",
    "SafetyVerifier",
    "ThoughtGenerator",
    "UncertaintyEstimate",
    "UncertaintyQuantifier",
]

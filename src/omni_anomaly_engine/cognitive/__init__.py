"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

"""
Cognitive Architecture Module for OMNI ♱ AVA

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

from omni_anomaly_engine.cognitive.case_based_reasoning import Case, CaseBasedReasoner
from omni_anomaly_engine.cognitive.causal_discovery import CausalDiscoveryEngine, CausalGraph
from omni_anomaly_engine.cognitive.indicator_system import Indicator, IndicatorDevelopmentSystem
from omni_anomaly_engine.cognitive.ipb_engine import BattlefieldAssessment, IPBEngine
from omni_anomaly_engine.cognitive.knowledge_graph import KnowledgeGraph, KnowledgeNode
from omni_anomaly_engine.cognitive.multi_hop_reasoner import MultiHopReasoner, ReasoningChain
from omni_anomaly_engine.cognitive.orchestrator import (
    CognitiveAnalysisResult,
    CognitiveOrchestrator,
)
from omni_anomaly_engine.cognitive.plasticity_engine import PlasticityEngine
from omni_anomaly_engine.cognitive.uncertainty import UncertaintyEstimate, UncertaintyQuantifier

__all__ = [
    # Main integration point
    "CognitiveOrchestrator",
    "CognitiveAnalysisResult",
    # Individual components
    "PlasticityEngine",
    "KnowledgeGraph",
    "KnowledgeNode",
    "MultiHopReasoner",
    "ReasoningChain",
    "IPBEngine",
    "BattlefieldAssessment",
    "CausalDiscoveryEngine",
    "CausalGraph",
    "UncertaintyQuantifier",
    "UncertaintyEstimate",
    "CaseBasedReasoner",
    "Case",
    "IndicatorDevelopmentSystem",
    "Indicator",
]

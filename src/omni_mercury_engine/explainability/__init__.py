"""
Explainability Module for Mercury Agent.

Provides comprehensive explanation capabilities for anomaly detection models,
including SHAP values, counterfactual explanations, and GDPR Article 22
compliance support.

Key Components:
- MercuryExplainer: Unified high-level interface for all explanation methods
- SHAP explainers: Feature attribution using Shapley values
- Counterfactual generators: Actionable explanations showing what changes
  would alter predictions
- GDPR compliance: Article 22 compliant reports for automated decisions

References:
- Lundberg & Lee (2017): A Unified Approach to Interpreting Model Predictions
- Wachter et al. (2017): Counterfactual Explanations without Opening the Black Box
- Mothilal et al. (2020): DiCE: Diverse Counterfactual Explanations
- GDPR Article 22: Automated individual decision-making
"""

from omni_mercury_engine.explainability.shap import (
    ExplainerType,
    ShapExplanation,
    GlobalExplanation,
    ShapExplainer,
    ExactShapExplainer,
    KernelShapExplainer,
    SamplingShapExplainer,
    TreeShapExplainer,
    LinearShapExplainer,
    create_shap_explainer,
)
from omni_mercury_engine.explainability.counterfactuals import (
    DistanceMetric,
    CounterfactualMethod,
    FeatureConstraint,
    Counterfactual,
    CounterfactualSet,
    CounterfactualGenerator,
    WachterCounterfactual,
    DiCECounterfactual,
    GrowingSpheresCounterfactual,
    PrototypeCounterfactual,
    create_counterfactual_generator,
)
from omni_mercury_engine.explainability.gdpr_compliance import (
    DecisionCategory,
    ExplanationLevel,
    DataSubjectInfo,
    DecisionInfo,
    ExplanationReport,
    ComplianceAuditRecord,
    GDPRExplainer,
)
from omni_mercury_engine.explainability.explainer import (
    AnomalyExplanation,
    GlobalAnomalyExplanation,
    MercuryExplainer,
)

__all__ = [
    # Main interface
    "MercuryExplainer",
    "AnomalyExplanation",
    "GlobalAnomalyExplanation",
    # SHAP
    "ExplainerType",
    "ShapExplanation",
    "GlobalExplanation",
    "ShapExplainer",
    "ExactShapExplainer",
    "KernelShapExplainer",
    "SamplingShapExplainer",
    "TreeShapExplainer",
    "LinearShapExplainer",
    "create_shap_explainer",
    # Counterfactuals
    "DistanceMetric",
    "CounterfactualMethod",
    "FeatureConstraint",
    "Counterfactual",
    "CounterfactualSet",
    "CounterfactualGenerator",
    "WachterCounterfactual",
    "DiCECounterfactual",
    "GrowingSpheresCounterfactual",
    "PrototypeCounterfactual",
    "create_counterfactual_generator",
    # GDPR Compliance
    "DecisionCategory",
    "ExplanationLevel",
    "DataSubjectInfo",
    "DecisionInfo",
    "ExplanationReport",
    "ComplianceAuditRecord",
    "GDPRExplainer",
]

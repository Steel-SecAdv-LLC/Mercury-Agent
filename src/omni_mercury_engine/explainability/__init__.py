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

from omni_mercury_engine.explainability.counterfactuals import (
    Counterfactual,
    CounterfactualGenerator,
    CounterfactualMethod,
    CounterfactualSet,
    DiCECounterfactual,
    DistanceMetric,
    FeatureConstraint,
    GrowingSpheresCounterfactual,
    PrototypeCounterfactual,
    WachterCounterfactual,
    create_counterfactual_generator,
)
from omni_mercury_engine.explainability.explainer import (
    AnomalyExplanation,
    GlobalAnomalyExplanation,
    MercuryExplainer,
)
from omni_mercury_engine.explainability.gdpr_compliance import (
    ComplianceAuditRecord,
    DataSubjectInfo,
    DecisionCategory,
    DecisionInfo,
    ExplanationLevel,
    ExplanationReport,
    GDPRExplainer,
)
from omni_mercury_engine.explainability.shap import (
    ExactShapExplainer,
    ExplainerType,
    GlobalExplanation,
    KernelShapExplainer,
    LinearShapExplainer,
    SamplingShapExplainer,
    ShapExplainer,
    ShapExplanation,
    TreeShapExplainer,
    create_shap_explainer,
)


__all__ = [
    "AnomalyExplanation",
    "ComplianceAuditRecord",
    "Counterfactual",
    "CounterfactualGenerator",
    "CounterfactualMethod",
    "CounterfactualSet",
    "DataSubjectInfo",
    # GDPR Compliance
    "DecisionCategory",
    "DecisionInfo",
    "DiCECounterfactual",
    # Counterfactuals
    "DistanceMetric",
    "ExactShapExplainer",
    # SHAP
    "ExplainerType",
    "ExplanationLevel",
    "ExplanationReport",
    "FeatureConstraint",
    "GDPRExplainer",
    "GlobalAnomalyExplanation",
    "GlobalExplanation",
    "GrowingSpheresCounterfactual",
    "KernelShapExplainer",
    "LinearShapExplainer",
    # Main interface
    "MercuryExplainer",
    "PrototypeCounterfactual",
    "SamplingShapExplainer",
    "ShapExplainer",
    "ShapExplanation",
    "TreeShapExplainer",
    "WachterCounterfactual",
    "create_counterfactual_generator",
    "create_shap_explainer",
]

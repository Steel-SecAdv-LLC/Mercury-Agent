"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Machine Learning module for Mercury Agent

Provides attention mechanisms, feature encoders, fusion networks, training, and inference.

Note: This module requires PyTorch. Imports are lazy to allow core package
to function without torch installed. Access ML components only when torch is available.
"""

from typing import TYPE_CHECKING, Any

# Centralized availability check (fast, no side-effects)
from omni_mercury_engine._compat import HAS_TORCH

__all__ = [
    "HAS_PYTORCH_LIGHTNING",
    "HAS_TORCH",
    "MAML",
    "ActiveLearner",
    "AffectiveEncoder",
    "AnomalyExplainer",
    # Meta-Learning (from arxiv 2508.11957v1)
    "AnomalyMetaLearner",
    "AstrophysicalEncoder",
    "AuxiliaryMaxVariance",
    "BiometricEncoder",
    "CheckpointCallback",
    "ConceptDriftEvaluator",
    "ConvergenceMonitor",
    # Cortical Network (Neuroscience-inspired)
    "CorticalColumn",
    "CorticalConfig",
    "CorticalLaminatedNetwork",
    "CorticalLoss",
    "CrossDomainTransferLearner",
    "CrossModalAttention",
    "DifferenceTargetPropagation",
    "FairnessAuditor",
    "FewShotLearner",
    "FusionInference",
    "FusionTrainer",
    # Golgi/Nissl/Weigert Analyzers (Brain Stain-inspired)
    "GolgiAnalyzer",
    "HebbianLearningRule",
    "IsolationScorer",
    "LateralInhibition",
    "LightweightAutoencoder",
    "LightweightMLP",
    "LyapunovAnomalyLoss",
    "MLPConfig",
    "MemoryEfficientCache",
    "MetaLearningAdapter",
    "MetaLearningAlgorithm",
    "MultiEnvPPOTrainer",
    "MultiHeadDetectorAttention",
    "NisslAnalyzer",
    "OmniFusionModel",
    "OnlineLearningPipeline",
    "PPOConfig",
    "PPOTrainer",
    "ParallelExecutor",
    "PrototypicalNetworks",
    "QuantumEncoder",
    "Reptile",
    "SparseCoding",
    "SpatialAttention",
    "SpikeTimingDependentPlasticity",
    "StatisticalEncoder",
    "SyntheticGradientModule",
    "SyntheticGradientPredictor",
    "TemporalAttention",
    "TemporalEncoder",
    "ThalamocorticalGate",
    "ThreeRAnomalyTrainer",
    "ThreeRAnomalyTransformer",
    "ThreeRAttentionBlock",
    "TrainingStats",
    "WeigertAnalyzer",
    "apply_all_optimizations",
    "compute_fairness_score",
    "create_active_learner",
    "create_concept_drift_evaluator",
    "create_cross_domain_learner",
    "create_drift_detector",
    "create_explainer",
    "create_few_shot_learner",
    "create_meta_learner",
    "create_online_pipeline",
    "quick_anomaly_score",
]

# Default value for HAS_PYTORCH_LIGHTNING when torch is not available
HAS_PYTORCH_LIGHTNING = False

# =============================================================================
# Lightweight Primitives (no torch required - pure NumPy)
# =============================================================================
from omni_mercury_engine.ml.lightweight_primitives import (
    IsolationScorer,
    LightweightAutoencoder,
    LightweightMLP,
    MLPConfig,
    quick_anomaly_score,
)

# Lazy imports - only load when torch is available OR during type checking
if HAS_TORCH or TYPE_CHECKING:
    from omni_mercury_engine.ml.advanced_optimizers import (
        AuxiliaryMaxVariance,
        DifferenceTargetPropagation,
        SyntheticGradientModule,
        SyntheticGradientPredictor,
    )
    from omni_mercury_engine.ml.attention import (
        CrossModalAttention,
        MultiHeadDetectorAttention,
        SpatialAttention,
        TemporalAttention,
    )
    from omni_mercury_engine.ml.cortical_network import (
        CorticalColumn,
        CorticalConfig,
        CorticalLaminatedNetwork,
        CorticalLoss,
        GolgiAnalyzer,
        HebbianLearningRule,
        LateralInhibition,
        NisslAnalyzer,
        SparseCoding,
        SpikeTimingDependentPlasticity,
        ThalamocorticalGate,
        WeigertAnalyzer,
    )
    from omni_mercury_engine.ml.encoders import (
        AffectiveEncoder,
        AstrophysicalEncoder,
        BiometricEncoder,
        QuantumEncoder,
        StatisticalEncoder,
        TemporalEncoder,
    )
    from omni_mercury_engine.ml.fusion_network import OmniFusionModel
    from omni_mercury_engine.ml.inference import FusionInference
    from omni_mercury_engine.ml.ppo_trainer import (
        CheckpointCallback,
        ConvergenceMonitor,
        MultiEnvPPOTrainer,
        PPOConfig,
        PPOTrainer,
        TrainingStats,
    )
    from omni_mercury_engine.ml.three_r_attention import (
        ThreeRAnomalyTransformer,
        ThreeRAttentionBlock,
    )
    from omni_mercury_engine.ml.training import (
        HAS_PYTORCH_LIGHTNING,
        FusionTrainer,
        LyapunovAnomalyLoss,
        ThreeRAnomalyTrainer,
    )


def _require_torch() -> None:
    """Raise ImportError if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for ML components. " "Install with: pip install torch torchvision"
        )


# Lazy imports for drift detection (requires scipy)
def create_drift_detector(*args: Any, **kwargs: Any) -> Any:
    """Create a drift detector. Lazy import to avoid scipy dependency at module load."""
    from omni_mercury_engine.ml.drift import create_drift_detector as _create

    return _create(*args, **kwargs)


# Lazy imports for fairness auditing (requires numpy only)
def compute_fairness_score(*args: Any, **kwargs: Any) -> Any:
    """Compute fairness score. Lazy import to avoid loading full module."""
    from omni_mercury_engine.ml.fairness import compute_fairness_score as _compute

    return _compute(*args, **kwargs)


class FairnessAuditor:
    """Lazy-loaded FairnessAuditor wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.fairness import FairnessAuditor as _FairnessAuditor

        return _FairnessAuditor(*args, **kwargs)


# Lazy imports for optimization utilities
def apply_all_optimizations(*args: Any, **kwargs: Any) -> Any:
    """Apply all optimizations. Lazy import to avoid psutil/joblib dependency."""
    from omni_mercury_engine.ml.optimization import apply_all_optimizations as _apply

    return _apply(*args, **kwargs)


class MemoryEfficientCache:
    """Lazy-loaded MemoryEfficientCache wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.optimization import MemoryEfficientCache as _Cache

        return _Cache(*args, **kwargs)


class ParallelExecutor:
    """Lazy-loaded ParallelExecutor wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.optimization import ParallelExecutor as _Executor

        return _Executor(*args, **kwargs)


# =============================================================================
# Advanced ML Capabilities (v1.4) - Lazy Imports
# =============================================================================


def create_concept_drift_evaluator(*args: Any, **kwargs: Any) -> Any:
    """Create concept drift evaluator. Lazy import."""
    from omni_mercury_engine.ml.concept_drift_evaluation import (
        create_concept_drift_evaluator as _create,
    )

    return _create(*args, **kwargs)


def create_few_shot_learner(*args: Any, **kwargs: Any) -> Any:
    """Create few-shot learner. Lazy import."""
    from omni_mercury_engine.ml.few_shot_learning import create_few_shot_learner as _create

    return _create(*args, **kwargs)


def create_cross_domain_learner(*args: Any, **kwargs: Any) -> Any:
    """Create cross-domain transfer learner. Lazy import."""
    from omni_mercury_engine.ml.cross_domain_transfer import (
        create_cross_domain_learner as _create,
    )

    return _create(*args, **kwargs)


def create_explainer(*args: Any, **kwargs: Any) -> Any:
    """Create SHAP explainer. Lazy import."""
    from omni_mercury_engine.ml.explainability import create_explainer as _create

    return _create(*args, **kwargs)


def create_active_learner(*args: Any, **kwargs: Any) -> Any:
    """Create active learner. Lazy import."""
    from omni_mercury_engine.ml.active_learning import create_active_learner as _create

    return _create(*args, **kwargs)


def create_online_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Create online learning pipeline. Lazy import."""
    from omni_mercury_engine.ml.online_learning import create_online_pipeline as _create

    return _create(*args, **kwargs)


class ConceptDriftEvaluator:
    """Lazy-loaded ConceptDriftEvaluator wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.concept_drift_evaluation import (
            ConceptDriftEvaluator as _Evaluator,
        )

        return _Evaluator(*args, **kwargs)


class FewShotLearner:
    """Lazy-loaded FewShotLearner wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.few_shot_learning import FewShotLearner as _Learner

        return _Learner(*args, **kwargs)


class CrossDomainTransferLearner:
    """Lazy-loaded CrossDomainTransferLearner wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.cross_domain_transfer import (
            CrossDomainTransferLearner as _Learner,
        )

        return _Learner(*args, **kwargs)


class AnomalyExplainer:
    """Lazy-loaded AnomalyExplainer wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.explainability import AnomalyExplainer as _Explainer

        return _Explainer(*args, **kwargs)


class ActiveLearner:
    """Lazy-loaded ActiveLearner wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.active_learning import ActiveLearner as _Learner

        return _Learner(*args, **kwargs)


class OnlineLearningPipeline:
    """Lazy-loaded OnlineLearningPipeline wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.online_learning import (
            OnlineLearningPipeline as _Pipeline,
        )

        return _Pipeline(*args, **kwargs)


# =============================================================================
# Meta-Learning Capabilities (arxiv 2508.11957v1 - AI Agents Survey)
# =============================================================================


def create_meta_learner(*args: Any, **kwargs: Any) -> Any:
    """Create meta-learner for few-shot adaptation. Lazy import."""
    from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner as _Learner

    return _Learner(*args, **kwargs)


class MetaLearningAdapter:
    """Lazy-loaded MetaLearningAdapter wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import MetaLearningAdapter as _Adapter

        return _Adapter(*args, **kwargs)


class MetaLearningAlgorithm:
    """Lazy-loaded MetaLearningAlgorithm enum wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import MetaLearningAlgorithm as _Algo

        return _Algo(*args, **kwargs)


class MAML:
    """Lazy-loaded MAML wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import MAML as _MAML

        return _MAML(*args, **kwargs)


class PrototypicalNetworks:
    """Lazy-loaded PrototypicalNetworks wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import PrototypicalNetworks as _Proto

        return _Proto(*args, **kwargs)


class Reptile:
    """Lazy-loaded Reptile wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import Reptile as _Reptile

        return _Reptile(*args, **kwargs)


class AnomalyMetaLearner:
    """Lazy-loaded AnomalyMetaLearner wrapper."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from omni_mercury_engine.ml.meta_learning import AnomalyMetaLearner as _Learner

        return _Learner(*args, **kwargs)

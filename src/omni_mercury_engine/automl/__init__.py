"""
AutoML Module for Mercury Agent.

Provides automatic hyperparameter tuning and model selection using
Bayesian optimization, Hyperband, and ASHA algorithms.

Key Components:
- MercuryAutoML: High-level AutoML interface
- BayesianOptimizer: TPE-based Bayesian optimization
- HyperbandScheduler: Hyperband for efficient early stopping
- ASHAScheduler: Asynchronous Successive Halving Algorithm

References:
- Bergstra et al. (2011): Algorithms for Hyper-Parameter Optimization
- Li et al. (2018): Hyperband: A Novel Bandit-Based Approach
- Li et al. (2020): A System for Massively Parallel Hyperparameter Tuning
"""

from omni_mercury_engine.automl.search_space import (
    SearchSpace,
    HyperParameter,
    UniformParameter,
    LogUniformParameter,
    IntUniformParameter,
    CategoricalParameter,
    ConditionalParameter,
)
from omni_mercury_engine.automl.schedulers import (
    HyperbandScheduler,
    HyperbandBracket,
    ASHAScheduler,
    MedianStoppingScheduler,
    TrialScheduler,
    SchedulerDecision,
    TrialInfo,
)
from omni_mercury_engine.automl.optimizer import (
    MercuryAutoML,
    BayesianOptimizer,
    TrialResult,
    OptimizationResult,
    TrialStatus,
    Sampler,
    TPESampler,
    GaussianProcessSampler,
    RandomSampler,
    SimpleAnomalyModel,
    SimpleClassifier,
    SimpleRegressor,
)

__all__ = [
    # Main interface
    "MercuryAutoML",
    "OptimizationResult",
    "TrialResult",
    "TrialStatus",
    # Optimizer and samplers
    "BayesianOptimizer",
    "Sampler",
    "TPESampler",
    "GaussianProcessSampler",
    "RandomSampler",
    # Schedulers
    "HyperbandScheduler",
    "HyperbandBracket",
    "ASHAScheduler",
    "MedianStoppingScheduler",
    "TrialScheduler",
    "SchedulerDecision",
    "TrialInfo",
    # Search space
    "SearchSpace",
    "HyperParameter",
    "UniformParameter",
    "LogUniformParameter",
    "IntUniformParameter",
    "CategoricalParameter",
    "ConditionalParameter",
    # Default models
    "SimpleAnomalyModel",
    "SimpleClassifier",
    "SimpleRegressor",
]

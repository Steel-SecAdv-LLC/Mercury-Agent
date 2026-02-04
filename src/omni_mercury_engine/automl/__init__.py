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

from omni_mercury_engine.automl.optimizer import (
    BayesianOptimizer,
    GaussianProcessSampler,
    MercuryAutoML,
    OptimizationResult,
    RandomSampler,
    Sampler,
    SimpleAnomalyModel,
    SimpleClassifier,
    SimpleRegressor,
    TPESampler,
    TrialResult,
    TrialStatus,
)
from omni_mercury_engine.automl.schedulers import (
    ASHAScheduler,
    HyperbandBracket,
    HyperbandScheduler,
    MedianStoppingScheduler,
    SchedulerDecision,
    TrialInfo,
    TrialScheduler,
)
from omni_mercury_engine.automl.search_space import (
    CategoricalParameter,
    ConditionalParameter,
    HyperParameter,
    IntUniformParameter,
    LogUniformParameter,
    SearchSpace,
    UniformParameter,
)


__all__ = [
    "ASHAScheduler",
    # Optimizer and samplers
    "BayesianOptimizer",
    "CategoricalParameter",
    "ConditionalParameter",
    "GaussianProcessSampler",
    "HyperParameter",
    "HyperbandBracket",
    # Schedulers
    "HyperbandScheduler",
    "IntUniformParameter",
    "LogUniformParameter",
    "MedianStoppingScheduler",
    # Main interface
    "MercuryAutoML",
    "OptimizationResult",
    "RandomSampler",
    "Sampler",
    "SchedulerDecision",
    # Search space
    "SearchSpace",
    # Default models
    "SimpleAnomalyModel",
    "SimpleClassifier",
    "SimpleRegressor",
    "TPESampler",
    "TrialInfo",
    "TrialResult",
    "TrialScheduler",
    "TrialStatus",
    "UniformParameter",
]

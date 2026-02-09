"""
Bayesian Optimization for AutoML.

Implements Tree-structured Parzen Estimator (TPE) for efficient
hyperparameter optimization with support for early stopping.

References:
- Bergstra et al. (2011): Algorithms for Hyper-Parameter Optimization
- Bergstra et al. (2013): Making a Science of Model Search
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from omni_mercury_engine.automl.schedulers import (
    ASHAScheduler,
    HyperbandScheduler,
    MedianStoppingScheduler,
    SchedulerDecision,
    TrialScheduler,
)
from omni_mercury_engine.automl.search_space import (
    CategoricalParameter,
    HyperParameter,
    IntUniformParameter,
    LogUniformParameter,
    SearchSpace,
    UniformParameter,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class TrialStatus(Enum):
    """Status of a trial."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    PRUNED = auto()
    FAILED = auto()


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial_id: str
    config: dict[str, Any]
    metric: float
    metrics_history: list[dict[str, float]] = field(default_factory=list)
    status: TrialStatus = TrialStatus.COMPLETED
    duration: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of the full optimization process."""

    best_config: dict[str, Any]
    best_metric: float
    best_trial_id: str
    all_trials: list[TrialResult]
    total_duration: float
    n_trials: int
    n_pruned: int
    convergence_history: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class Sampler(ABC):
    """Base class for hyperparameter samplers."""

    @abstractmethod
    def sample(
        self,
        search_space: SearchSpace,
        n_samples: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample configurations from the search space."""
        pass

    @abstractmethod
    def tell(
        self,
        config: dict[str, Any],
        metric: float,
    ) -> None:
        """Update the sampler with a new observation."""
        pass


class RandomSampler(Sampler):
    """Random sampling from the search space."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random sampler."""
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        search_space: SearchSpace,
        n_samples: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample random configurations."""
        return [search_space.sample() for _ in range(n_samples)]

    def tell(
        self,
        config: dict[str, Any],
        metric: float,
    ) -> None:
        """No-op for random sampler."""
        pass


class TPESampler(Sampler):
    """
    Tree-structured Parzen Estimator (TPE) sampler.

    Models P(x|y) using two distributions: one for good configurations
    and one for bad configurations, then samples from the ratio.
    """

    def __init__(
        self,
        gamma: float = 0.25,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int | None = None,
    ) -> None:
        """
        Initialize TPE sampler.

        Args:
            gamma: Fraction of trials to consider as "good"
            n_startup_trials: Number of random trials before TPE
            n_ei_candidates: Number of candidates for EI optimization
            seed: Random seed
        """
        self._gamma = gamma
        self._n_startup = n_startup_trials
        self._n_ei_candidates = n_ei_candidates
        self._rng = np.random.default_rng(seed)

        self._observations: list[tuple[dict[str, Any], float]] = []
        self._param_observations: dict[str, list[tuple[Any, float]]] = {}

    def sample(
        self,
        search_space: SearchSpace,
        n_samples: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample configurations using TPE."""
        if len(self._observations) < self._n_startup:
            return [search_space.sample() for _ in range(n_samples)]

        configs = []
        for _ in range(n_samples):
            config = self._sample_tpe(search_space)
            configs.append(config)

        return configs

    def _sample_tpe(self, search_space: SearchSpace) -> dict[str, Any]:
        """Sample a single configuration using TPE."""
        config = {}

        n_below = int(np.ceil(self._gamma * len(self._observations)))

        sorted_obs = sorted(self._observations, key=lambda x: x[1])
        below_configs = [c for c, _ in sorted_obs[:n_below]]
        above_configs = [c for c, _ in sorted_obs[n_below:]]

        for name, param in search_space.parameters.items():
            below_values = [c.get(name) for c in below_configs if name in c]
            above_values = [c.get(name) for c in above_configs if name in c]

            if not below_values or not above_values:
                config[name] = param.sample()
                continue

            config[name] = self._sample_param_tpe(param, below_values, above_values)

        return config

    def _sample_param_tpe(
        self,
        param: HyperParameter,
        below_values: list[Any],
        above_values: list[Any],
    ) -> Any:
        """Sample a single parameter using TPE."""
        candidates = []
        ei_values = []

        for _ in range(self._n_ei_candidates):
            candidate = param.sample()
            candidates.append(candidate)

            l_below = self._parzen_estimator(candidate, below_values, param)
            l_above = self._parzen_estimator(candidate, above_values, param)

            ei = l_below / (l_above + 1e-12)
            ei_values.append(ei)

        best_idx = np.argmax(ei_values)
        return candidates[best_idx]

    def _parzen_estimator(
        self,
        value: Any,
        observations: list[Any],
        param: HyperParameter,
    ) -> float:
        """Compute Parzen estimator density."""
        if isinstance(param, CategoricalParameter):
            counts = {}
            for obs in observations:
                counts[obs] = counts.get(obs, 0) + 1

            n_total = len(observations)
            n_choices = len(param.choices)

            prob = (counts.get(value, 0) + 1) / (n_total + n_choices)
            return prob

        elif isinstance(param, IntUniformParameter):
            obs_array = np.array(observations, dtype=float)
            value_float = float(value)

            sigma = max(1.0, np.std(obs_array) + 1e-6)
            weights = stats.norm.pdf(value_float, loc=obs_array, scale=sigma)
            return np.mean(weights) + 1e-12

        elif isinstance(param, LogUniformParameter):
            obs_array = np.log(np.array(observations) + 1e-12)
            value_log = np.log(value + 1e-12)

            sigma = max(0.1, np.std(obs_array) + 1e-6)
            weights = stats.norm.pdf(value_log, loc=obs_array, scale=sigma)
            return np.mean(weights) + 1e-12

        else:
            obs_array = np.array(observations, dtype=float)
            value_float = float(value)

            sigma = max(0.01, np.std(obs_array) + 1e-6)
            weights = stats.norm.pdf(value_float, loc=obs_array, scale=sigma)
            return np.mean(weights) + 1e-12

    def tell(
        self,
        config: dict[str, Any],
        metric: float,
    ) -> None:
        """Update TPE with a new observation."""
        self._observations.append((config, metric))

        for name, value in config.items():
            if name not in self._param_observations:
                self._param_observations[name] = []
            self._param_observations[name].append((value, metric))


class GaussianProcessSampler(Sampler):
    """
    Gaussian Process-based Bayesian optimization sampler.

    Uses Expected Improvement (EI) acquisition function.
    """

    def __init__(
        self,
        n_startup_trials: int = 5,
        n_candidates: int = 1000,
        seed: int | None = None,
    ) -> None:
        """Initialize GP sampler."""
        self._n_startup = n_startup_trials
        self._n_candidates = n_candidates
        self._rng = np.random.default_rng(seed)

        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._search_space: SearchSpace | None = None
        self._param_names: list[str] = []

    def sample(
        self,
        search_space: SearchSpace,
        n_samples: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample configurations using GP."""
        self._search_space = search_space
        self._param_names = list(search_space.parameters.keys())

        if len(self._X) < self._n_startup:
            return [search_space.sample() for _ in range(n_samples)]

        configs = []
        for _ in range(n_samples):
            config = self._sample_gp(search_space)
            configs.append(config)

        return configs

    def _sample_gp(self, search_space: SearchSpace) -> dict[str, Any]:
        """Sample using GP with EI acquisition."""
        X = np.array(self._X)
        y = np.array(self._y)

        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-6
        y_normalized = (y - y_mean) / y_std

        candidates = []
        for _ in range(self._n_candidates):
            config = search_space.sample()
            x = self._config_to_vector(config)
            candidates.append((config, x))

        best_config = None
        best_ei = -np.inf

        y_best = np.min(y_normalized)

        for config, x in candidates:
            mu, sigma = self._predict(x, X, y_normalized)
            ei = self._expected_improvement(mu, sigma, y_best)

            if ei > best_ei:
                best_ei = ei
                best_config = config

        return best_config if best_config else search_space.sample()

    def _config_to_vector(self, config: dict[str, Any]) -> np.ndarray:
        """Convert config to numerical vector."""
        vector = []
        for name in self._param_names:
            param = self._search_space.parameters[name]
            value = config[name]

            if isinstance(param, CategoricalParameter):
                idx = param.choices.index(value) / max(1, len(param.choices) - 1)
                vector.append(idx)
            elif isinstance(param, LogUniformParameter):
                log_val = np.log(value)
                bounds = param.get_bounds()
                log_low = np.log(bounds[0])
                log_high = np.log(bounds[1])
                normalized = (log_val - log_low) / (log_high - log_low + 1e-12)
                vector.append(normalized)
            elif isinstance(param, (UniformParameter, IntUniformParameter)):
                bounds = param.get_bounds()
                normalized = (value - bounds[0]) / (bounds[1] - bounds[0] + 1e-12)
                vector.append(normalized)
            else:
                vector.append(float(value))

        return np.array(vector)

    def _predict(
        self,
        x: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        """Predict mean and variance using GP."""
        length_scale = 0.5
        noise = 1e-4

        K = self._rbf_kernel(X, X, length_scale) + noise * np.eye(len(X))
        k_star = self._rbf_kernel(X, x.reshape(1, -1), length_scale).flatten()

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu = np.dot(k_star, alpha)

            v = np.linalg.solve(L, k_star)
            k_star_star = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1), length_scale)[0, 0]
            sigma = np.sqrt(max(1e-8, k_star_star - np.dot(v, v)))
        except np.linalg.LinAlgError:
            mu = np.mean(y)
            sigma = np.std(y) + 1e-6

        return mu, sigma

    def _rbf_kernel(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        length_scale: float,
    ) -> np.ndarray:
        """RBF (Gaussian) kernel."""
        sq_dist = (
            np.sum(X1**2, axis=1).reshape(-1, 1)
            + np.sum(X2**2, axis=1).reshape(1, -1)
            - 2 * np.dot(X1, X2.T)
        )
        return np.exp(-0.5 * sq_dist / (length_scale**2))

    def _expected_improvement(
        self,
        mu: float,
        sigma: float,
        y_best: float,
        xi: float = 0.01,
    ) -> float:
        """Compute Expected Improvement."""
        if sigma <= 0:
            return 0.0

        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
        return ei

    def tell(
        self,
        config: dict[str, Any],
        metric: float,
    ) -> None:
        """Update GP with new observation."""
        if self._search_space is None:
            return

        x = self._config_to_vector(config)
        self._X.append(x)
        self._y.append(metric)


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.

    Combines a sampler (TPE or GP) with optional early stopping
    via trial schedulers (Hyperband, ASHA).
    """

    def __init__(
        self,
        search_space: SearchSpace,
        objective: Callable[[dict[str, Any]], float],
        sampler: str = "tpe",
        scheduler: str | None = None,
        direction: str = "minimize",
        n_trials: int = 100,
        n_jobs: int = 1,
        seed: int | None = None,
    ) -> None:
        """
        Initialize Bayesian optimizer.

        Args:
            search_space: Search space definition
            objective: Objective function to optimize
            sampler: "tpe", "gp", or "random"
            scheduler: "hyperband", "asha", "median", or None
            direction: "minimize" or "maximize"
            n_trials: Maximum number of trials
            n_jobs: Number of parallel jobs (currently sequential only)
            seed: Random seed
        """
        self._search_space = search_space
        self._objective = objective
        self._direction = direction
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._seed = seed

        self._sampler = self._create_sampler(sampler, seed)
        self._scheduler = self._create_scheduler(scheduler)

        self._trials: list[TrialResult] = []
        self._best_trial: TrialResult | None = None
        self._trial_counter = 0

    def _create_sampler(self, sampler: str, seed: int | None) -> Sampler:
        """Create the specified sampler."""
        samplers = {
            "tpe": lambda: TPESampler(seed=seed),
            "gp": lambda: GaussianProcessSampler(seed=seed),
            "random": lambda: RandomSampler(seed=seed),
        }
        return samplers.get(sampler, samplers["tpe"])()

    def _create_scheduler(self, scheduler: str | None) -> TrialScheduler | None:
        """Create the specified scheduler."""
        if scheduler is None:
            return None

        schedulers = {
            "hyperband": HyperbandScheduler,
            "asha": ASHAScheduler,
            "median": MedianStoppingScheduler,
        }
        scheduler_cls = schedulers.get(scheduler)
        return scheduler_cls() if scheduler_cls else None

    def optimize(self) -> OptimizationResult:
        """Run the optimization."""
        start_time = time.time()
        convergence_history = []
        n_pruned = 0

        for trial_idx in range(self._n_trials):
            config = self._sampler.sample(self._search_space, n_samples=1)[0]
            trial_id = self._generate_trial_id(config, trial_idx)

            logger.info(f"Trial {trial_idx + 1}/{self._n_trials}: {trial_id}")

            trial_start = time.time()
            try:
                metric = self._objective(config)
                status = TrialStatus.COMPLETED
            except Exception as e:
                logger.warning(f"Trial {trial_id} failed: {e}")
                metric = float("inf") if self._direction == "minimize" else float("-inf")
                status = TrialStatus.FAILED

            trial_duration = time.time() - trial_start

            if self._scheduler is not None:
                result = {"metric": metric, "iteration": trial_idx}
                decision = self._scheduler.on_trial_result(trial_id, result)
                if decision == SchedulerDecision.STOP:
                    status = TrialStatus.PRUNED
                    n_pruned += 1
                    self._scheduler.on_trial_complete(trial_id)

            trial_result = TrialResult(
                trial_id=trial_id,
                config=config,
                metric=metric,
                status=status,
                duration=trial_duration,
                iteration=trial_idx,
            )
            self._trials.append(trial_result)

            if status == TrialStatus.COMPLETED:
                self._sampler.tell(config, metric)
                self._update_best(trial_result)

            if self._best_trial:
                convergence_history.append(self._best_trial.metric)
            else:
                convergence_history.append(metric)

        total_duration = time.time() - start_time

        return OptimizationResult(
            best_config=self._best_trial.config if self._best_trial else {},
            best_metric=self._best_trial.metric if self._best_trial else float("inf"),
            best_trial_id=self._best_trial.trial_id if self._best_trial else "",
            all_trials=self._trials,
            total_duration=total_duration,
            n_trials=len(self._trials),
            n_pruned=n_pruned,
            convergence_history=convergence_history,
        )

    def _generate_trial_id(self, config: dict[str, Any], idx: int) -> str:
        """Generate unique trial ID."""
        config_str = str(sorted(config.items()))
        hash_str = hashlib.sha256(config_str.encode()).hexdigest()[:8]
        return f"trial_{idx:04d}_{hash_str}"

    def _update_best(self, trial: TrialResult) -> None:
        """Update best trial if this one is better."""
        if self._best_trial is None:
            self._best_trial = trial
            return

        if self._direction == "minimize":
            if trial.metric < self._best_trial.metric:
                self._best_trial = trial
        elif trial.metric > self._best_trial.metric:
            self._best_trial = trial


class MercuryAutoML:
    """
    High-level AutoML interface for Mercury Agent.

    Provides automated hyperparameter optimization, model selection,
    and feature engineering for anomaly detection models.

    Example:
        automl = MercuryAutoML(
            task="anomaly_detection",
            metric="f1",
            time_budget=3600,
            n_trials=100,
        )

        # Define search space
        automl.add_parameter("learning_rate", "log_uniform", 1e-5, 1e-1)
        automl.add_parameter("n_estimators", "int_uniform", 10, 500)
        automl.add_parameter("max_depth", "int_uniform", 3, 20)

        # Run optimization
        result = automl.fit(X_train, y_train, X_val, y_val)

        # Get best model
        best_model = automl.get_best_model()
        predictions = best_model.predict(X_test)
    """

    SUPPORTED_TASKS = ["anomaly_detection", "classification", "regression"]
    SUPPORTED_METRICS = {
        "anomaly_detection": ["f1", "precision", "recall", "auc", "average_precision"],
        "classification": ["accuracy", "f1", "precision", "recall", "auc"],
        "regression": ["mse", "rmse", "mae", "r2"],
    }

    def __init__(
        self,
        task: str = "anomaly_detection",
        metric: str = "f1",
        direction: str | None = None,
        time_budget: float | None = None,
        n_trials: int = 100,
        sampler: str = "tpe",
        scheduler: str | None = "asha",
        seed: int | None = None,
    ) -> None:
        """
        Initialize MercuryAutoML.

        Args:
            task: Task type (anomaly_detection, classification, regression)
            metric: Optimization metric
            direction: minimize or maximize (auto-detected if None)
            time_budget: Maximum time in seconds (optional)
            n_trials: Maximum number of trials
            sampler: Sampler type (tpe, gp, random)
            scheduler: Early stopping scheduler (hyperband, asha, median, None)
            seed: Random seed
        """
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Task must be one of {self.SUPPORTED_TASKS}")

        if metric not in self.SUPPORTED_METRICS.get(task, []):
            raise ValueError(
                f"Metric must be one of {self.SUPPORTED_METRICS[task]} for task '{task}'"
            )

        self._task = task
        self._metric = metric
        self._time_budget = time_budget
        self._n_trials = n_trials
        self._sampler = sampler
        self._scheduler = scheduler
        self._seed = seed

        if direction is None:
            minimize_metrics = ["mse", "rmse", "mae", "loss"]
            self._direction = "minimize" if metric in minimize_metrics else "maximize"
        else:
            self._direction = direction

        self._search_space = SearchSpace()
        self._model_factory: Callable[[dict[str, Any]], Any] | None = None
        self._best_model: Any = None
        self._best_config: dict[str, Any] = {}
        self._result: OptimizationResult | None = None

    def add_parameter(
        self,
        name: str,
        param_type: str,
        *args: Any,
        **kwargs: Any,
    ) -> MercuryAutoML:
        """
        Add a hyperparameter to the search space.

        Args:
            name: Parameter name
            param_type: uniform, log_uniform, int_uniform, categorical
            *args: Parameter-specific arguments
            **kwargs: Parameter-specific keyword arguments

        Returns:
            self for method chaining
        """
        param_classes = {
            "uniform": UniformParameter,
            "log_uniform": LogUniformParameter,
            "int_uniform": IntUniformParameter,
            "categorical": CategoricalParameter,
        }

        param_class = param_classes.get(param_type)
        if param_class is None:
            raise ValueError(f"Unknown parameter type: {param_type}")

        param = param_class(name, *args, **kwargs)
        self._search_space.add(param)

        return self

    def set_model_factory(
        self,
        factory: Callable[[dict[str, Any]], Any],
    ) -> MercuryAutoML:
        """
        Set a custom model factory function.

        Args:
            factory: Function that takes config dict and returns a model

        Returns:
            self for method chaining
        """
        self._model_factory = factory
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        eval_func: Callable[[Any, np.ndarray, np.ndarray | None], float] | None = None,
    ) -> OptimizationResult:
        """
        Run hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training labels (optional for unsupervised)
            X_val: Validation features
            y_val: Validation labels
            eval_func: Custom evaluation function

        Returns:
            OptimizationResult with best configuration and metrics
        """
        if X_val is None:
            n_val = int(0.2 * len(X_train))
            X_val = X_train[-n_val:]
            X_train = X_train[:-n_val]
            if y_train is not None:
                y_val = y_train[-n_val:]
                y_train = y_train[:-n_val]

        def objective(config: dict[str, Any]) -> float:
            if self._model_factory is not None:
                model = self._model_factory(config)
            else:
                model = self._create_default_model(config)

            if hasattr(model, "fit"):
                if y_train is not None:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train)

            if eval_func is not None:
                score = eval_func(model, X_val, y_val)
            else:
                score = self._evaluate_model(model, X_val, y_val)

            if self._direction == "minimize":
                return score
            else:
                return -score

        optimizer = BayesianOptimizer(
            search_space=self._search_space,
            objective=objective,
            sampler=self._sampler,
            scheduler=self._scheduler,
            direction="minimize",
            n_trials=self._n_trials,
            seed=self._seed,
        )

        self._result = optimizer.optimize()

        if self._direction == "maximize":
            self._result.best_metric = -self._result.best_metric
            for trial in self._result.all_trials:
                trial.metric = -trial.metric
            self._result.convergence_history = [-m for m in self._result.convergence_history]

        self._best_config = self._result.best_config

        if self._model_factory is not None:
            self._best_model = self._model_factory(self._best_config)
        else:
            self._best_model = self._create_default_model(self._best_config)

        if hasattr(self._best_model, "fit"):
            full_X = np.vstack([X_train, X_val]) if X_val is not None else X_train
            if y_train is not None and y_val is not None:
                full_y = np.concatenate([y_train, y_val])
                self._best_model.fit(full_X, full_y)
            else:
                self._best_model.fit(full_X)

        return self._result

    def _create_default_model(self, config: dict[str, Any]) -> Any:
        """Create a default model based on task type."""
        if self._task == "anomaly_detection":
            return SimpleAnomalyModel(**config)
        elif self._task == "classification":
            return SimpleClassifier(**config)
        else:
            return SimpleRegressor(**config)

    def _evaluate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray | None,
    ) -> float:
        """Evaluate model using the specified metric."""
        if self._task == "anomaly_detection":
            if hasattr(model, "predict"):
                predictions = model.predict(X_val)
            elif hasattr(model, "decision_function"):
                predictions = model.decision_function(X_val)
            else:
                predictions = np.zeros(len(X_val))

            if y_val is None:
                return float(np.mean(predictions))

            return self._compute_metric(y_val, predictions)

        else:
            predictions = model.predict(X_val)
            if y_val is None:
                return 0.0
            return self._compute_metric(y_val, predictions)

    def _compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute the specified metric."""
        y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.dtype == float else y_pred

        if self._metric == "accuracy":
            return np.mean(y_true == y_pred_binary)

        elif self._metric == "precision":
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true == 0))
            return tp / (tp + fp + 1e-10)

        elif self._metric == "recall":
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            fn = np.sum((y_pred_binary == 0) & (y_true == 1))
            return tp / (tp + fn + 1e-10)

        elif self._metric == "f1":
            precision = self._compute_metric(y_true, y_pred_binary)
            self._metric = "recall"
            recall = self._compute_metric(y_true, y_pred_binary)
            self._metric = "f1"
            return 2 * precision * recall / (precision + recall + 1e-10)

        elif self._metric in ["mse", "loss"]:
            return np.mean((y_true - y_pred) ** 2)

        elif self._metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

        elif self._metric == "mae":
            return np.mean(np.abs(y_true - y_pred))

        elif self._metric == "r2":
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / (ss_tot + 1e-10)

        elif self._metric == "auc":
            return self._compute_auc(y_true, y_pred)

        elif self._metric == "average_precision":
            return self._compute_average_precision(y_true, y_pred)

        return 0.0

    def _compute_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5

        tpr_values = []
        fpr_values = []
        tp = 0
        fp = 0

        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_values.append(tp / n_pos)
            fpr_values.append(fp / n_neg)

        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i - 1]) * (tpr_values[i] + tpr_values[i - 1]) / 2

        return auc

    def _compute_average_precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute Average Precision."""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]

        precisions = []
        recalls = []
        tp = 0
        n_pos = np.sum(y_true == 1)

        if n_pos == 0:
            return 0.0

        for i, label in enumerate(y_true_sorted):
            if label == 1:
                tp += 1
                precision = tp / (i + 1)
                recall = tp / n_pos
                precisions.append(precision)
                recalls.append(recall)

        if not precisions:
            return 0.0

        return np.mean(precisions)

    def get_best_model(self) -> Any:
        """Get the best model found during optimization."""
        return self._best_model

    def get_best_config(self) -> dict[str, Any]:
        """Get the best configuration found."""
        return self._best_config

    def get_results(self) -> OptimizationResult | None:
        """Get full optimization results."""
        return self._result

    def get_feature_importance(self) -> dict[str, float]:
        """
        Get hyperparameter importance based on trial history.

        Uses functional ANOVA to estimate parameter importance.
        """
        if self._result is None or not self._result.all_trials:
            return {}

        importance = {}
        all_metrics = [
            t.metric for t in self._result.all_trials if t.status == TrialStatus.COMPLETED
        ]

        if not all_metrics:
            return {}

        total_var = np.var(all_metrics)
        if total_var < 1e-10:
            return dict.fromkeys(self._search_space.parameters, 0.0)

        for param_name in self._search_space.parameters:
            param_values: dict[Any, list[float]] = {}

            for trial in self._result.all_trials:
                if trial.status != TrialStatus.COMPLETED:
                    continue
                value = trial.config.get(param_name)
                if value not in param_values:
                    param_values[value] = []
                param_values[value].append(trial.metric)

            if len(param_values) <= 1:
                importance[param_name] = 0.0
                continue

            group_means = [np.mean(metrics) for metrics in param_values.values()]
            between_var = np.var(group_means) * len(group_means)

            importance[param_name] = between_var / total_var

        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}

        return importance


class SimpleAnomalyModel:
    """Simple anomaly detection model for default AutoML usage."""

    def __init__(
        self,
        contamination: float = 0.1,
        threshold_percentile: float = 95,
        **kwargs: Any,
    ) -> None:
        """Initialize simple anomaly model."""
        self._contamination = contamination
        self._threshold_percentile = threshold_percentile
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._threshold: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SimpleAnomalyModel:
        """Fit the model."""
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0) + 1e-10

        scores = self.decision_function(X)
        self._threshold = np.percentile(scores, self._threshold_percentile)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if self._mean is None:
            return np.zeros(len(X))

        z_scores = np.abs((X - self._mean) / self._std)
        return np.mean(z_scores, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        scores = self.decision_function(X)
        return (scores > self._threshold).astype(int)


class SimpleClassifier:
    """Simple classifier for default AutoML usage."""

    def __init__(
        self,
        n_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize simple classifier."""
        self._n_neighbors = n_neighbors
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleClassifier:
        """Fit the classifier."""
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        if self._X_train is None:
            return np.zeros(len(X))

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self._X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[: self._n_neighbors]
            nearest_labels = self._y_train[nearest_indices]
            prediction = int(np.round(np.mean(nearest_labels)))
            predictions.append(prediction)

        return np.array(predictions)


class SimpleRegressor:
    """Simple regressor for default AutoML usage."""

    def __init__(
        self,
        n_neighbors: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize simple regressor."""
        self._n_neighbors = n_neighbors
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SimpleRegressor:
        """Fit the regressor."""
        self._X_train = X
        self._y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values."""
        if self._X_train is None:
            return np.zeros(len(X))

        predictions = []
        for x in X:
            distances = np.sqrt(np.sum((self._X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[: self._n_neighbors]
            nearest_values = self._y_train[nearest_indices]
            prediction = np.mean(nearest_values)
            predictions.append(prediction)

        return np.array(predictions)

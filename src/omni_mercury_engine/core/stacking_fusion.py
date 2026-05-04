"""
DEPRECATED: This module uses sklearn for anomaly detection.

Mercury's production detector is MercuryAnomalyDetector in detectors/statistical.py. This module is
retained for reference only and will be removed in a future release.

Do not import this module in production or benchmark code paths.

Original: Stacking and Bayesian Model Averaging Fusion. Copyright (C) 2025 Steel Security Advisors
LLC License: GPL-3.0-or-later
"""

from __future__ import annotations

import warnings

warnings.warn(
    f"{__name__} is deprecated. Use MercuryAnomalyDetector.",
    DeprecationWarning,
    stacklevel=2,
)

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# Golden ratio for ethical scaling
PHI = 1.618033988749895

# Default ethical threshold
SIGMA_IMMUTABLE_DEFAULT = 0.96


class BaseDetector(Protocol):
    """Protocol for base detectors in ensemble."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the detector to ``(X, y)``."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard-label predictions for ``X``."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities for ``X``."""


@dataclass
class FusionResult:
    """Result of ensemble fusion."""

    predictions: np.ndarray
    probabilities: np.ndarray
    detector_weights: dict[str, float]
    fusion_method: str
    ethical_gate_passed: bool
    ethical_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianWeights:
    """Bayesian model weights with uncertainty."""

    weights: np.ndarray
    posterior_probs: np.ndarray
    weight_uncertainty: np.ndarray  # Standard deviation of weights
    bic_scores: np.ndarray  # BIC for each model


class StackingFusion:
    """
    Stacking (Stacked Generalization) for detector fusion.

    Uses cross-validated predictions from base detectors as features
    for a meta-learner. More principled than simple voting/averaging.

    Reference: Wolpert (1992) "Stacked Generalization"
    """

    def __init__(
        self,
        meta_learner: Any = None,
        cv_folds: int = 5,
        use_proba: bool = True,
        passthrough: bool = False,
        seed: int = 42,
    ):
        """
        Initialize stacking fusion.

        Args:
            meta_learner: Meta-learner model (default: LogisticRegression)
            cv_folds: Cross-validation folds for generating meta-features
            use_proba: Use probability predictions (vs binary)
            passthrough: Include original features in meta-learner
            seed: Random seed
        """
        if meta_learner is None:
            try:
                from omni_mercury_engine.ml.mercury_ml import LogisticRegression
            except ImportError as e:
                raise ImportError(
                    "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
                ) from e
            meta_learner = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=seed,
            )
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.use_proba = use_proba
        self.passthrough = passthrough
        self.seed = seed

        self.detectors: dict[str, Any] = {}
        self.detector_names: list[str] = []
        self._fitted = False

    def add_detector(
        self, name: str, detector: Any, ethical_score: float | None = None
    ) -> StackingFusion:
        """
        Add a base detector to the ensemble.

        Args:
            name: Unique name for detector
            detector: Fitted or unfitted detector
            ethical_score: Optional ethical score (ignored for StackingFusion, included for API compatibility)

        Returns:
            Self for method chaining
        """
        # ethical_score is accepted but not used in StackingFusion (for API compatibility)
        _ = ethical_score
        self.detectors[name] = detector
        self.detector_names.append(name)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> StackingFusion:
        """
        Fit stacking ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        if not self.detectors:
            raise ValueError("Must add detectors before fitting")

        np.random.seed(self.seed)

        # Import sklearn functions needed for cross-validation
        try:
            from omni_mercury_engine.ml.mercury_ml import cross_val_predict
        except ImportError as e:
            raise ImportError(
                "This feature requires scikit-learn. Install with: pip install mercury-agent[ml]"
            ) from e

        # Generate out-of-fold predictions for each detector
        meta_features = []

        for name, detector in self.detectors.items():
            logger.debug(f"Generating meta-features for {name}")

            try:
                # Fit detector
                try:
                    detector.fit(X, y)
                except TypeError:
                    detector.fit(X)

                # Cross-validated predictions
                if self.use_proba:
                    try:
                        oof_pred = cross_val_predict(
                            detector,
                            X,
                            y,
                            cv=self.cv_folds,
                            method="predict_proba",
                        )
                        if oof_pred.ndim == 2:
                            oof_pred = oof_pred[:, 1]
                    except (AttributeError, ValueError, TypeError):
                        oof_pred = cross_val_predict(
                            detector,
                            X,
                            y,
                            cv=self.cv_folds,
                        ).astype(float)
                else:
                    oof_pred = cross_val_predict(
                        detector,
                        X,
                        y,
                        cv=self.cv_folds,
                    ).astype(float)

                meta_features.append(oof_pred)

            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
                # Use zeros as fallback
                meta_features.append(np.zeros(len(X)))

        # Stack meta-features
        meta_X = np.column_stack(meta_features)

        if self.passthrough:
            meta_X = np.hstack([meta_X, X])

        # Fit meta-learner
        self.meta_learner.fit(meta_X, y)
        self._fitted = True

        logger.info(
            f"StackingFusion fitted: {len(self.detectors)} detectors, "
            f"meta_features shape={meta_X.shape}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using stacking ensemble.

        Args:
            X: Test features

        Returns:
            Binary predictions
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        meta_X = self._get_meta_features(X)
        return np.asarray(self.meta_learner.predict(meta_X))  # type: ignore[no-any-return, unused-ignore]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using stacking ensemble.

        Args:
            X: Test features

        Returns:
            Probability predictions
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict_proba()")

        meta_X = self._get_meta_features(X)
        try:
            return np.asarray(self.meta_learner.predict_proba(meta_X))  # type: ignore[no-any-return, unused-ignore]
        except AttributeError:
            # Meta-learner doesn't support proba
            return np.asarray(self.meta_learner.predict(meta_X).astype(float))  # type: ignore[no-any-return, unused-ignore]

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base detectors."""
        meta_features = []

        for name, detector in self.detectors.items():
            if self.use_proba:
                try:
                    pred = detector.predict_proba(X)
                    if pred.ndim == 2:
                        pred = pred[:, 1]
                except (AttributeError, ValueError, TypeError):
                    pred = detector.predict(X).astype(float)
            else:
                pred = detector.predict(X).astype(float)

            meta_features.append(pred)

        meta_X = np.column_stack(meta_features)

        if self.passthrough:
            meta_X = np.hstack([meta_X, X])

        return meta_X

    def get_detector_importance(self) -> dict[str, float]:
        """
        Get importance of each detector from meta-learner.

        Returns:
            Dictionary mapping detector name to importance weight
        """
        if not self._fitted:
            return {}

        try:
            # For linear meta-learners
            coefs = self.meta_learner.coef_[0]
            if self.passthrough:
                coefs = coefs[: len(self.detector_names)]

            # Normalize to sum to 1
            weights = np.abs(coefs) / (np.sum(np.abs(coefs)) + 1e-10)

            return dict(zip(self.detector_names, weights))
        except AttributeError:
            # Non-linear meta-learner
            return {name: 1.0 / len(self.detectors) for name in self.detector_names}


class BayesianModelAveraging:
    """
    Bayesian Model Averaging (BMA) for detector fusion.

    Weights detectors by their posterior model probability,
    accounting for model uncertainty. More robust than
    simple averaging when detectors have varying quality.

    Reference: Hoeting et al. (1999) "Bayesian Model Averaging"
    """

    def __init__(
        self,
        prior_type: str = "uniform",
        use_bic: bool = True,
        min_weight: float = 0.01,
    ):
        """
        Initialize BMA fusion.

        Args:
            prior_type: Prior over models ("uniform", "complexity_penalized")
            use_bic: Use BIC for model evidence approximation
            min_weight: Minimum weight for any detector
        """
        self.prior_type = prior_type
        self.use_bic = use_bic
        self.min_weight = min_weight

        self.detectors: dict[str, Any] = {}
        self.weights: BayesianWeights | None = None
        self._fitted = False

    def add_detector(
        self, name: str, detector: Any, ethical_score: float | None = None
    ) -> BayesianModelAveraging:
        """
        Add a detector to the ensemble.

        Args:
            name: Unique name for detector
            detector: Detector instance
            ethical_score: Optional ethical score (ignored for BMA, included for API compatibility)

        Returns:
            Self for method chaining
        """
        # ethical_score is accepted but not used in BMA (for API compatibility)
        _ = ethical_score
        self.detectors[name] = detector
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> BayesianModelAveraging:
        """
        Fit BMA ensemble and compute posterior weights.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        n_detectors = len(self.detectors)
        if n_detectors == 0:
            raise ValueError("Must add detectors before fitting")

        log_marginal_likelihoods = []
        bic_scores = []

        for name, detector in self.detectors.items():
            # Fit detector
            try:
                detector.fit(X, y)
            except TypeError:
                detector.fit(X)

            # Compute log marginal likelihood approximation via BIC
            # BIC = -2 * log_likelihood + k * log(n)
            # Log marginal likelihood ≈ -BIC/2

            try:
                proba = detector.predict_proba(X)
                if proba.ndim == 2:
                    proba = proba[:, 1]

                # Validate probabilities for NaN/Inf before computing log likelihood
                if not np.all(np.isfinite(proba)):
                    logger.warning(f"Non-finite probabilities from {name}, replacing with 0.5")
                    proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)

                proba = np.clip(proba, 1e-10, 1 - 1e-10)

                # Log likelihood
                ll = np.sum(y * np.log(proba) + (1 - y) * np.log(1 - proba))

                # Validate log likelihood
                if not np.isfinite(ll):
                    logger.warning(f"Non-finite log-likelihood from {name}, using fallback")
                    ll = -1000.0  # Large negative value as fallback

                # Number of parameters (approximate)
                try:
                    k = np.sum(
                        [
                            np.prod(p.shape)
                            for p in detector.get_params().values()
                            if hasattr(p, "shape")
                        ]
                    )
                except (AttributeError, ValueError, TypeError):
                    k = 10  # Default estimate

                # BIC
                n = len(y)
                bic = -2 * ll + k * np.log(n)
                bic_scores.append(bic)

                # Log marginal likelihood approximation
                log_ml = -bic / 2
                log_marginal_likelihoods.append(log_ml)

            except Exception as e:
                logger.warning(f"Failed to compute likelihood for {name}: {e}")
                log_marginal_likelihoods.append(-np.inf)
                bic_scores.append(np.inf)

        # Compute posterior model probabilities
        log_ml = np.array(log_marginal_likelihoods)
        log_ml = log_ml - np.max(log_ml)  # Numerical stability

        # Prior probabilities
        if self.prior_type == "uniform":
            log_prior = np.zeros(n_detectors)
        else:
            # Penalize complexity (use BIC as proxy)
            log_prior = -np.array(bic_scores) / (2 * np.max(np.abs(bic_scores)) + 1e-10)

        # Posterior
        log_posterior = log_ml + log_prior
        posterior = np.exp(log_posterior - np.logaddexp.reduce(log_posterior))

        # Apply minimum weight constraint
        posterior = np.maximum(posterior, self.min_weight)
        posterior = posterior / np.sum(posterior)

        # Estimate uncertainty (using Laplace approximation)
        # This is a simplified estimate
        weight_uncertainty = np.sqrt(posterior * (1 - posterior) / len(y))

        self.weights = BayesianWeights(
            weights=posterior,
            posterior_probs=posterior,
            weight_uncertainty=weight_uncertainty,
            bic_scores=np.array(bic_scores),
        )

        self._fitted = True

        logger.info(
            f"BMA fitted: {n_detectors} detectors, "
            f"weights={dict(zip(self.detectors.keys(), posterior))}"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using BMA weighted average."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using BMA weighted average.

        Args:
            X: Test features

        Returns:
            Weighted average of detector predictions
        """
        if not self._fitted or self.weights is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        weighted_sum = np.zeros(len(X))

        for i, (name, detector) in enumerate(self.detectors.items()):
            weight = self.weights.weights[i]

            try:
                proba = detector.predict_proba(X)
                if proba.ndim == 2:
                    proba = proba[:, 1]
            except (AttributeError, ValueError, TypeError):
                proba = detector.predict(X).astype(float)

            weighted_sum += weight * proba

        return weighted_sum

    def get_weights_with_uncertainty(self) -> dict[str, tuple[float, float]]:
        """
        Get weights with uncertainty estimates.

        Returns:
            Dictionary mapping detector name to (weight, std_dev)
        """
        if not self._fitted or self.weights is None:
            return {}

        return {
            name: (self.weights.weights[i], self.weights.weight_uncertainty[i])
            for i, name in enumerate(self.detectors.keys())
        }


class EthicallyConstrainedFusion:
    """
    Fusion with ethical constraints integrated from GOSNN.

    Learns optimal detector weights while ensuring ethical compliance through sigma_Immutable
    threshold gating and benevolence weighting.
    """

    def __init__(
        self,
        sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
        benevolence_weight: float = 0.1,
        use_golden_ratio: bool = True,
    ):
        """
        Initialize ethically constrained fusion.

        Args:
            sigma_immutable: Ethical threshold (0.93-0.96)
            benevolence_weight: Weight for benevolence term in loss
            use_golden_ratio: Apply phi-based harmonic weighting
        """
        self.sigma_immutable = sigma_immutable
        self.benevolence_weight = benevolence_weight
        self.use_golden_ratio = use_golden_ratio

        self.detectors: dict[str, Any] = {}
        self.weights: np.ndarray | None = None
        self.ethical_scores: dict[str, float] = {}
        self._fitted = False

    def add_detector(
        self,
        name: str,
        detector: Any,
        ethical_score: float = 1.0,
    ) -> EthicallyConstrainedFusion:
        """
        Add detector with ethical score.

        Args:
            name: Detector name
            detector: Detector instance
            ethical_score: Ethical compliance score (0-1)
        """
        self.detectors[name] = detector
        self.ethical_scores[name] = ethical_score
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> EthicallyConstrainedFusion:
        """
        Fit fusion with ethical constraints.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        n_detectors = len(self.detectors)
        if n_detectors == 0:
            raise ValueError("Must add detectors before fitting")

        # Fit all detectors and collect predictions
        detector_preds = []

        for name, detector in self.detectors.items():
            try:
                detector.fit(X, y)
            except TypeError:
                detector.fit(X)

            try:
                proba = detector.predict_proba(X)
                if proba.ndim == 2:
                    proba = proba[:, 1]
            except (AttributeError, ValueError, TypeError):
                proba = detector.predict(X).astype(float)

            # Validate predictions for NaN/Inf before adding to ensemble
            if not np.all(np.isfinite(proba)):
                logger.warning(f"Non-finite predictions from {name}, replacing with 0.5")
                proba = np.nan_to_num(proba, nan=0.5, posinf=1.0, neginf=0.0)

            detector_preds.append(proba)

        detector_preds = np.array(  # type: ignore[assignment, unused-ignore]
            detector_preds
        ).T  # (n_samples, n_detectors)
        ethical_vec = np.array([self.ethical_scores[name] for name in self.detectors])

        # Optimize weights with ethical constraints
        def objective(w: np.ndarray) -> float:
            # Normalize weights
            w = np.abs(w)
            w = w / (np.sum(w) + 1e-10)

            # Weighted prediction
            pred = detector_preds @ w

            # Binary cross-entropy loss
            pred = np.clip(pred, 1e-10, 1 - 1e-10)
            bce = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

            # Ethical penalty: penalize low-ethical detectors
            ethical_penalty = self.benevolence_weight * np.sum(w * (1 - ethical_vec))

            # Sigma_immutable constraint: average ethical score must exceed threshold
            avg_ethical = np.sum(w * ethical_vec)
            constraint_penalty = 10.0 * max(0, self.sigma_immutable - avg_ethical)

            return float(bce + ethical_penalty + constraint_penalty)

        # Initial weights (optionally phi-weighted)
        if self.use_golden_ratio and n_detectors >= 3:
            phi_weights = np.array([PHI, 1.0, 1.0 / PHI])
            phi_weights = np.tile(phi_weights, n_detectors // 3 + 1)[:n_detectors]
            w0 = phi_weights / np.sum(phi_weights)
        else:
            w0 = np.ones(n_detectors) / n_detectors

        # Optimize
        result = minimize(
            objective,
            w0,
            method="L-BFGS-B",
            bounds=[(0.01, 1.0)] * n_detectors,
        )

        # Normalize final weights
        self.weights = np.abs(result.x)
        self.weights = self.weights / np.sum(self.weights)

        self._fitted = True

        # Log results
        avg_ethical = np.sum(self.weights * ethical_vec)
        logger.info(
            f"EthicallyConstrainedFusion fitted: "
            f"weights={dict(zip(self.detectors.keys(), self.weights))}, "
            f"avg_ethical={avg_ethical:.3f}, threshold={self.sigma_immutable}"
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict with ethically-weighted fusion."""
        if not self._fitted or self.weights is None:
            raise RuntimeError("Must call fit() before predict_proba()")

        weighted_sum = np.zeros(len(X))

        for i, (name, detector) in enumerate(self.detectors.items()):
            weight = self.weights[i]

            try:
                proba = detector.predict_proba(X)
                if proba.ndim == 2:
                    proba = proba[:, 1]
            except (AttributeError, ValueError, TypeError):
                proba = detector.predict(X).astype(float)

            # Apply ethical gating: scale by detector's ethical score
            ethical_factor = self.ethical_scores[name] ** (1 / PHI)
            weighted_sum += weight * proba * ethical_factor

        return weighted_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_ethical_compliance(self) -> dict[str, Any]:
        """Get ethical compliance metrics."""
        if not self._fitted or self.weights is None:
            return {}

        ethical_vec = np.array([self.ethical_scores[name] for name in self.detectors])
        avg_ethical = np.sum(self.weights * ethical_vec)

        return {
            "average_ethical_score": avg_ethical,
            "sigma_immutable_threshold": self.sigma_immutable,
            "passes_threshold": avg_ethical >= self.sigma_immutable,
            "detector_weights": dict(zip(self.detectors.keys(), self.weights)),
            "detector_ethical_scores": self.ethical_scores.copy(),
        }


def create_fusion_ensemble(
    detectors: dict[str, Any],
    method: str = "fibring",
    ethical_scores: dict[str, float] | None = None,
    **kwargs: Any,
) -> StackingFusion | BayesianModelAveraging | EthicallyConstrainedFusion:
    """
    Factory function to create fusion ensemble.

    Args:
        detectors: Dictionary of detector name to detector
        method: Fusion method. Recognised values:

            - ``"fibring"`` *(default)*: Returns an
              :class:`EthicallyConstrainedFusion` constructed with
              ``use_golden_ratio=True``.  This is the named composition
              that pairs with the hub-level :data:`FusionMode.FIBRING`:
              golden-ratio-aware base + correlation-aware decorrelation
              + per-detector ethical weighting.  Phi-weighted base
              initialisation is applied by ``EthicallyConstrainedFusion.fit``
              when there are at least three detectors; for ensembles with
              fewer than three detectors the base falls back to uniform
              weights, with the ethical-weighting and decorrelation
              layers still active.
            - ``"ethical"``: Alias for the fibring path retained for
              backwards compatibility with existing callers.
            - ``"stacking"``: Stacked-generalisation meta-learner.
            - ``"bma"``: Bayesian model averaging.

        ethical_scores: Ethical scores for each detector (used by ``"ethical"``
            and ``"fibring"`` methods). Defaults to 1.0 when absent.
        **kwargs: Additional arguments for specific methods. For ``"fibring"``,
            ``use_golden_ratio`` defaults to True.

    Returns:
        Configured fusion ensemble.
    """
    ensemble: StackingFusion | BayesianModelAveraging | EthicallyConstrainedFusion
    if method == "stacking":
        ensemble = StackingFusion(**kwargs)
    elif method == "bma":
        ensemble = BayesianModelAveraging(**kwargs)
    elif method in ("ethical", "fibring"):
        # Fibring at the ensemble level is the EthicallyConstrainedFusion with
        # phi-weighted base. The hub-level FibringComposer (core/fibring_fusion.py)
        # provides the streaming decorrelator + domain-affinity bias on top.
        kwargs.setdefault("use_golden_ratio", True)
        ensemble = EthicallyConstrainedFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    for name, detector in detectors.items():
        if method in ("ethical", "fibring") and ethical_scores:
            ensemble.add_detector(name, detector, ethical_scores.get(name, 1.0))
        else:
            ensemble.add_detector(name, detector)

    return ensemble

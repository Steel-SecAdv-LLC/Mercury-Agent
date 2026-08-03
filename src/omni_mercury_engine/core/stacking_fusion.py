# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DEPRECATED: legacy stacking / Bayesian-averaging fusion ensemble.

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

# Golden ratio for the reliability-weight exponent
PHI = 1.618033988749895

# Default weighted-average reliability floor
SIGMA_IMMUTABLE_DEFAULT = 0.96


class BaseDetector(Protocol):
    """Protocol for base detectors in ensemble."""

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """Fit the detector to ``(X, y)``."""

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return hard-label predictions for ``X``."""

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return calibrated class probabilities for ``X``."""


@dataclass
class FusionResult:
    """Result of ensemble fusion."""

    predictions: np.ndarray[Any, Any]
    probabilities: np.ndarray[Any, Any]
    detector_weights: dict[str, float]
    fusion_method: str
    reliability_floor_met: bool
    reliability_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianWeights:
    """Bayesian model weights with uncertainty."""

    weights: np.ndarray[Any, Any]
    posterior_probs: np.ndarray[Any, Any]
    weight_uncertainty: np.ndarray[Any, Any]  # Standard deviation of weights
    bic_scores: np.ndarray[Any, Any]  # BIC for each model


class StackingFusion:
    """Stacking (Stacked Generalization) for detector fusion.

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
        """Initialize stacking fusion.

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
                    "Default meta-learner needs omni_mercury_engine.ml.mercury_ml "
                    "(numpy/scipy); the package install is incomplete."
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
        self, name: str, detector: Any, reliability_score: float | None = None
    ) -> StackingFusion:
        """Add a base detector to the ensemble.

        Args:
            name: Unique name for detector
            detector: Fitted or unfitted detector
            reliability_score: Optional reliability weight (ignored here; present for API parity)

        Returns:
            Self for method chaining
        """
        # reliability_score is accepted but not used in StackingFusion (for API compatibility)
        _ = reliability_score
        self.detectors[name] = detector
        self.detector_names.append(name)
        return self

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> StackingFusion:
        """Fit stacking ensemble.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self for method chaining
        """
        if not self.detectors:
            raise ValueError("Must add detectors before fitting")

        # Intentional global-state seeding: the downstream ``cross_val_predict``
        # in ``mercury_ml`` uses a ``KFold`` splitter whose internal shuffling
        # consumes the legacy global ``np.random`` state when ``random_state``
        # is not threaded through.  Until ``cross_val_predict`` accepts an
        # explicit ``random_state``/``Generator`` parameter, calling
        # ``np.random.seed(self.seed)`` here is the only way to keep the CV
        # fold indices reproducible from ``self.seed``.  Tracked debt:
        # surface ``random_state`` in ``mercury_ml.cross_val_predict`` so
        # this site can graduate to ``np.random.default_rng(self.seed)``
        # plumbing.
        np.random.seed(self.seed)

        # Mercury's cross-validation helper (omni_mercury_engine.ml.mercury_ml)
        try:
            from omni_mercury_engine.ml.mercury_ml import cross_val_predict
        except ImportError as e:
            raise ImportError(
                "Cross-validated stacking needs omni_mercury_engine.ml.mercury_ml "
                "(numpy/scipy); the package install is incomplete."
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

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict using stacking ensemble.

        Args:
            X: Test features

        Returns:
            Binary predictions
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict()")

        meta_X = self._get_meta_features(X)
        return np.asarray(self.meta_learner.predict(meta_X))  # type: ignore[no-any-return, unused-ignore]

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict probabilities using stacking ensemble.

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

    def _get_meta_features(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
        """Get importance of each detector from meta-learner.

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
    """Bayesian Model Averaging (BMA) for detector fusion.

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
        """Initialize BMA fusion.

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
        self, name: str, detector: Any, reliability_score: float | None = None
    ) -> BayesianModelAveraging:
        """Add a detector to the ensemble.

        Args:
            name: Unique name for detector
            detector: Detector instance
            reliability_score: Optional reliability weight (ignored here; present for API parity)

        Returns:
            Self for method chaining
        """
        # reliability_score is accepted but not used in BMA (for API compatibility)
        _ = reliability_score
        self.detectors[name] = detector
        return self

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> BayesianModelAveraging:
        """Fit BMA ensemble and compute posterior weights.

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

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict using BMA weighted average."""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict probabilities using BMA weighted average.

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
        """Get weights with uncertainty estimates.

        Returns:
            Dictionary mapping detector name to (weight, std_dev)
        """
        if not self._fitted or self.weights is None:
            return {}

        return {
            name: (self.weights.weights[i], self.weights.weight_uncertainty[i])
            for i, name in enumerate(self.detectors.keys())
        }


class ReliabilityWeightedFusion:
    """Stacking fusion that weights each detector by a per-detector reliability score.

    Renamed from ``EthicallyConstrainedFusion``. Nothing about it was ethical: the
    per-detector float it weights by is supplied by the caller
    (:meth:`add_detector`, default ``1.0``), and no ethics signal is computed,
    consulted or enforced anywhere in the class. Weighting an ensemble by
    per-member trust is an ordinary and useful technique; calling that trust
    "ethical compliance" implied a safety control that does not exist here, and
    left the real control harder to find.

    Ethics enforcement is the fail-closed harm-uplift gate at the public decision
    surfaces (``cognitive/decision_gate.py``). This class cannot refuse anything.
    """

    def __init__(
        self,
        sigma_immutable: float = SIGMA_IMMUTABLE_DEFAULT,
        reliability_penalty_weight: float = 0.1,
        use_golden_ratio: bool = True,
    ):
        """Initialize reliability-weighted fusion.

        Args:
            sigma_immutable: Minimum weighted-average reliability the optimiser is
                pushed toward. A soft penalty in the objective, not a gate, and
                unrelated to the σ_Immutable configuration-integrity gate in
                ``security/sigma_immutable_gate.py``.
            reliability_penalty_weight: Coefficient on the low-reliability penalty
                term in the weight-optimisation loss.
            use_golden_ratio: Apply phi-based harmonic weighting
        """
        self.sigma_immutable = sigma_immutable
        self.reliability_penalty_weight = reliability_penalty_weight
        self.use_golden_ratio = use_golden_ratio

        self.detectors: dict[str, Any] = {}
        self.weights: np.ndarray[Any, Any] | None = None
        self.reliability_scores: dict[str, float] = {}
        self._fitted = False

    def add_detector(
        self,
        name: str,
        detector: Any,
        reliability_score: float = 1.0,
    ) -> ReliabilityWeightedFusion:
        """Add a detector with a caller-supplied reliability weight.

        Args:
            name: Detector name
            detector: Detector instance
            reliability_score: Per-detector trust weight in [0, 1]. Defaults to
                ``1.0``, which makes the weighting a no-op.
        """
        self.detectors[name] = detector
        self.reliability_scores[name] = reliability_score
        return self

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> ReliabilityWeightedFusion:
        """Fit fusion under the reliability penalty.

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
        reliability_vec = np.array([self.reliability_scores[name] for name in self.detectors])

        # Optimise weights under the reliability penalty
        def objective(w: np.ndarray[Any, Any]) -> float:
            # Normalize weights
            w = np.abs(w)
            w = w / (np.sum(w) + 1e-10)

            # Weighted prediction
            pred = detector_preds @ w

            # Binary cross-entropy loss
            pred = np.clip(pred, 1e-10, 1 - 1e-10)
            bce = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

            # Penalise weight placed on low-reliability detectors
            reliability_penalty = self.reliability_penalty_weight * np.sum(
                w * (1 - reliability_vec)
            )

            # Soft floor on the weighted-average reliability (a penalty, not a gate)
            avg_reliability = np.sum(w * reliability_vec)
            constraint_penalty = 10.0 * max(0, self.sigma_immutable - avg_reliability)

            return float(bce + reliability_penalty + constraint_penalty)

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
        avg_reliability = np.sum(self.weights * reliability_vec)
        logger.info(
            f"ReliabilityWeightedFusion fitted: "
            f"weights={dict(zip(self.detectors.keys(), self.weights))}, "
            f"avg_reliability={avg_reliability:.3f}, threshold={self.sigma_immutable}"
        )

        return self

    def predict_proba(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict with reliability-weighted fusion."""
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

            # Scale each detector's probability by its reliability weight
            reliability_factor = self.reliability_scores[name] ** (1 / PHI)
            weighted_sum += weight * proba * reliability_factor

        return weighted_sum

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Predict binary labels."""
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_reliability_report(self) -> dict[str, Any]:
        """Report the weighted-average reliability and per-detector weights.

        Renamed from ``get_ethical_compliance``: it reports caller-supplied trust
        weights, not ethical compliance, and ``floor_met`` is descriptive -- no
        code path refuses anything when it is False.
        """
        if not self._fitted or self.weights is None:
            return {}

        reliability_vec = np.array([self.reliability_scores[name] for name in self.detectors])
        avg_reliability = np.sum(self.weights * reliability_vec)

        return {
            # Cast out of ``np.float64`` / ``np.bool_`` so the report is plain
            # JSON-serialisable Python, and so ``floor_met is True`` holds for
            # callers that identity-check it.
            "average_reliability": float(avg_reliability),
            "reliability_floor": float(self.sigma_immutable),
            "floor_met": bool(avg_reliability >= self.sigma_immutable),
            "detector_weights": dict(zip(self.detectors.keys(), self.weights)),
            "detector_reliability_scores": self.reliability_scores.copy(),
        }


def create_fusion_ensemble(
    detectors: dict[str, Any],
    method: str = "fibring",
    reliability_scores: dict[str, float] | None = None,
    **kwargs: Any,
) -> StackingFusion | BayesianModelAveraging | ReliabilityWeightedFusion:
    """Factory function to create fusion ensemble.

    Args:
        detectors: Dictionary of detector name to detector
        method: Fusion method. Recognised values:

            - ``"fibring"`` *(default)*: Returns an
              :class:`ReliabilityWeightedFusion` constructed with
              ``use_golden_ratio=True``.  This is the named composition
              that pairs with the hub-level :data:`FusionMode.FIBRING`:
              golden-ratio-aware base + correlation-aware decorrelation
              + per-detector reliability weighting.  Phi-weighted base
              initialisation is applied by ``ReliabilityWeightedFusion.fit``
              when there are at least three detectors; for ensembles with
              fewer than three detectors the base falls back to uniform
              weights, with the reliability-weighting and decorrelation
              layers still active.
            - ``"ethical"``: Legacy alias for the fibring path. The name is
              inaccurate -- the path applies reliability weights, not an ethics
              control -- and is retained only so existing callers keep working.
            - ``"stacking"``: Stacked-generalisation meta-learner.
            - ``"bma"``: Bayesian model averaging.

        reliability_scores: Per-detector trust weight in [0, 1], used by the
            ``"fibring"``/``"ethical"`` paths. Defaults to 1.0 when absent, which
            makes the weighting a no-op.
        **kwargs: Additional arguments for specific methods. For ``"fibring"``,
            ``use_golden_ratio`` defaults to True.

    Returns:
        Configured fusion ensemble.
    """
    ensemble: StackingFusion | BayesianModelAveraging | ReliabilityWeightedFusion
    if method == "stacking":
        ensemble = StackingFusion(**kwargs)
    elif method == "bma":
        ensemble = BayesianModelAveraging(**kwargs)
    elif method in ("ethical", "fibring"):
        # Fibring at the ensemble level is the ReliabilityWeightedFusion with
        # phi-weighted base. The hub-level FibringComposer (core/fibring_fusion.py)
        # provides the streaming decorrelator + domain-affinity bias on top.
        kwargs.setdefault("use_golden_ratio", True)
        ensemble = ReliabilityWeightedFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    for name, detector in detectors.items():
        if method in ("ethical", "fibring") and reliability_scores:
            ensemble.add_detector(name, detector, reliability_scores.get(name, 1.0))
        else:
            ensemble.add_detector(name, detector)

    return ensemble

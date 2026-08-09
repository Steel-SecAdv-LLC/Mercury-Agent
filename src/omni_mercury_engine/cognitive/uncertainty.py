# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Uncertainty Quantification Module - Production Implementation.

Provides rigorous uncertainty estimation for neuro-symbolic AI:
- Monte Carlo Dropout: Epistemic uncertainty via stochastic forward passes
- Deep Ensembles: Model disagreement as uncertainty proxy
- Heteroscedastic Networks: Learned aleatoric uncertainty
- Temperature Scaling: Post-hoc calibration with LBFGS optimization
- Adaptive Conformal Inference: Distribution-free coverage with online updates

Research Sources:
- Gal & Ghahramani (2016): Dropout as Bayesian Approximation
- Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty
- Kendall & Gal (2017): What Uncertainties Do We Need in Bayesian Deep Learning?
- Guo et al. (2017): On Calibration of Modern Neural Networks
- Gibbs & Candes (2021): Adaptive Conformal Inference
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Optional PyTorch support
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]


class UncertaintyType(Enum):
    """Types of uncertainty."""

    EPISTEMIC = "epistemic"  # Model uncertainty - reducible with more data
    ALEATORIC = "aleatoric"  # Data uncertainty - irreducible
    TOTAL = "total"  # Combined uncertainty


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""

    VERY_LOW = 0.2
    LOW = 0.4
    MODERATE = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class UncertaintyEstimate:
    """Complete uncertainty estimate for a prediction."""

    prediction: float
    epistemic: float  # Model uncertainty (MC Dropout variance)
    aleatoric: float  # Data uncertainty (heteroscedastic or residual)
    total: float  # Combined: sqrt(epistemic^2 + aleatoric^2)
    confidence: float  # Calibrated confidence [0, 1]
    confidence_interval: tuple[float, float]  # (lower, upper)
    calibration_error: float  # Current ECE estimate
    is_reliable: bool  # Meets reliability criteria
    explanation: str
    # Additional diagnostics
    mutual_information: float = 0.0  # BALD acquisition function
    predictive_entropy: float = 0.0  # Total predictive uncertainty
    mc_samples: int = 0  # Number of MC samples used
    is_overconfident: bool = False  # High confidence despite poor calibration
    # Transparency flags: whether epistemic/aleatoric were *measured* (ensemble / MC
    # variance, heteroscedastic residual) versus a default placeholder used when
    # no ensemble/model/features were supplied. ``confidence_calibrated`` is True
    # only when a fitted calibrator (not the uncalibrated monotone prior) drove
    # the confidence number.
    epistemic_measured: bool = True
    aleatoric_measured: bool = True
    confidence_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "prediction": self.prediction,
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "total": self.total,
            "confidence": self.confidence,
            "ci": self.confidence_interval,
            "calibration_error": self.calibration_error,
            "reliable": self.is_reliable,
            "explanation": self.explanation,
            "mutual_information": self.mutual_information,
            "predictive_entropy": self.predictive_entropy,
            "mc_samples": self.mc_samples,
            "overconfident": self.is_overconfident,
            "epistemic_measured": self.epistemic_measured,
            "aleatoric_measured": self.aleatoric_measured,
            "confidence_calibrated": self.confidence_calibrated,
        }


@dataclass
class CalibrationResult:
    """Result of calibration assessment."""

    expected_confidence: list[float]
    observed_accuracy: list[float]
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Adaptive Calibration Error
    is_calibrated: bool
    temperature: float  # Optimal temperature for scaling
    reliability_diagram: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return {
            "ece": self.ece,
            "mce": self.mce,
            "ace": self.ace,
            "calibrated": self.is_calibrated,
            "temperature": self.temperature,
            "diagram": self.reliability_diagram,
        }


class MCDropoutWrapper:
    """Monte Carlo Dropout wrapper for PyTorch models.

    Enables dropout at inference time for epistemic uncertainty estimation. Based on Gal &
    Ghahramani (2016).
    """

    def __init__(self, model: Any, dropout_rate: float = 0.1, seed: int | None = None) -> None:
        """Args:.

            model: PyTorch model with dropout layers
            dropout_rate: Dropout probability (if not already in model)
            seed: Optional seed for reproducible MC-Dropout sampling.  When
                provided, ``predict_with_uncertainty`` re-seeds PyTorch's
                global RNG (``torch.manual_seed`` + the CUDA generators if
                available) before each call so the dropout draws are
                reproducible across runs.  ``None`` (default) leaves the
                global RNG untouched — same effective behavior as the
                original implementation.  This is the correct seeding path
                for MC-Dropout because the dropout stochasticity lives in
                ``torch.nn.Dropout`` (PyTorch RNG), not NumPy.

        Note:
            The MC-input-perturbation site
            (``UncertaintyQuantifier._monte_carlo_sampling``) is a separate
            randomness source and owns its own ``np.random.Generator``;
            pass ``seed=`` to ``UncertaintyQuantifier`` for that path.
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self._seed = seed
        self._original_training_state = None
        # NumPy ``Generator`` retained for any future NumPy-side perturbation
        # the wrapper might add; current implementation is torch-only.
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def enable_dropout(self) -> None:
        """Enable dropout layers for MC sampling."""
        if not TORCH_AVAILABLE:
            return

        self._original_training_state = self.model.training

        def apply_dropout(m: nn.Module) -> None:
            if isinstance(m, nn.Dropout):
                m.train()

        self.model.apply(apply_dropout)

    def disable_dropout(self) -> None:
        """Restore original dropout state."""
        if not TORCH_AVAILABLE or self._original_training_state is None:
            return

        if not self._original_training_state:
            self.model.eval()

    def predict_with_uncertainty(
        self,
        x: Any,
        n_samples: int = 30,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Generate predictions with MC Dropout uncertainty.

        Args:
            x: Input tensor
            n_samples: Number of MC forward passes

        Returns:
            (mean_prediction, epistemic_std, all_samples)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for MC Dropout")

        # Re-seed PyTorch's global RNG when the wrapper was constructed
        # with an explicit seed.  This is the correct reproducibility path
        # for MC-Dropout because ``torch.nn.Dropout`` consumes the torch
        # RNG, not NumPy's.  Done once per call (not per sample) so each
        # ``predict_with_uncertainty`` invocation is reproducible without
        # collapsing the n_samples draws to identical outputs.
        if self._seed is not None:
            torch.manual_seed(self._seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._seed)

        self.enable_dropout()

        samples_list: list[np.ndarray[Any, Any]] = []
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                if isinstance(output, torch.Tensor):
                    samples_list.append(output.cpu().numpy())
                else:
                    samples_list.append(np.array(output))

        self.disable_dropout()

        samples_arr = np.array(samples_list)  # (n_samples, batch, outputs)
        mean = samples_arr.mean(axis=0)
        std = samples_arr.std(axis=0)

        return mean, std, samples_arr


class TemperatureScaler:
    """Temperature scaling for neural network calibration.

    Learns a single temperature parameter to scale logits, optimized via NLL on a validation set.
    Based on Guo et al. (2017).
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        """Initialize the instance."""
        self.temperature = init_temperature
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray[Any, Any],
        labels: np.ndarray[Any, Any],
        max_iter: int = 100,
    ) -> float:
        """Fit temperature using LBFGS optimization.

        Args:
            logits: Pre-softmax outputs (n_samples, n_classes)
            labels: True labels (n_samples,)
            max_iter: Maximum optimization iterations

        Returns:
            Optimal temperature
        """

        def nll_loss(T: np.ndarray[Any, Any]) -> float:
            """Negative log-likelihood with temperature scaling."""
            temp = float(max(T[0], 0.01))  # Ensure positive temperature
            scaled_logits = logits / temp

            # Softmax with numerical stability
            max_logits = np.max(scaled_logits, axis=1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # Cross-entropy loss
            n = len(labels)
            correct_probs = probs[np.arange(n), labels.astype(int)]
            loss = float(-np.mean(np.log(correct_probs + 1e-10)))
            return loss

        # Optimize temperature
        result = optimize.minimize(
            nll_loss,
            x0=[self.temperature],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0)],
            options={"maxiter": max_iter},
        )

        self.temperature = result.x[0]
        self._fitted = True

        logger.info(f"Temperature scaling: T={self.temperature:.4f}")
        return self.temperature

    def scale(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def calibrated_probs(self, logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Get calibrated probabilities."""
        scaled = self.scale(logits)
        # Softmax
        max_logits = np.max(scaled, axis=1, keepdims=True)
        exp_logits = np.exp(scaled - max_logits)
        result: np.ndarray[Any, Any] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return result


class AdaptiveConformalInference:
    """Adaptive Conformal Inference for online uncertainty quantification.

    Provides distribution-free prediction intervals with finite-sample coverage guarantees that
    adapt to distribution shift. Based on Gibbs & Candes (2021).
    """

    def __init__(
        self,
        target_coverage: float = 0.9,
        gamma: float = 0.01,
        window_size: int = 500,
    ):
        """Initialize the adaptive conformal inference estimator.

        Args:
            target_coverage: Target coverage level (1 - alpha)
            gamma: Learning rate for alpha adjustment
            window_size: Size of calibration window
        """
        self.target_coverage = target_coverage
        self.alpha = 1 - target_coverage
        self.gamma = gamma
        self.window_size = window_size

        # Calibration scores (nonconformity scores)
        self.scores: deque[Any] = deque(maxlen=window_size)

        # Tracking
        self.coverage_history: list[float] = []
        self.alpha_history: list[float] = [self.alpha]

    def update(self, score: float, covered: bool) -> None:
        """Update adaptive alpha based on coverage.

        Args:
            score: Nonconformity score for new point
            covered: Whether prediction set covered true value
        """
        self.scores.append(score)

        # Adaptive alpha update (Gibbs & Candes, Eq. 3)
        # If covered: decrease alpha (widen intervals)
        # If not covered: increase alpha (tighten intervals)
        err = 1 - int(covered)
        self.alpha = self.alpha + self.gamma * (err - self.alpha)
        self.alpha = np.clip(self.alpha, 0.001, 0.5)

        self.alpha_history.append(self.alpha)

        # Track running coverage
        if len(self.coverage_history) > 0:
            n = len(self.coverage_history)
            running_cov = (n * self.coverage_history[-1] + int(covered)) / (n + 1)
            self.coverage_history.append(running_cov)
        else:
            self.coverage_history.append(float(covered))

    def get_quantile(self) -> float:
        """Get current conformal quantile threshold."""
        if len(self.scores) == 0:
            return float("inf")

        n = len(self.scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile_level = min(1.0, quantile_level)

        return float(np.quantile(list(self.scores), quantile_level))

    def predict_interval(
        self,
        point_prediction: float,
        score_function: Callable[[float], float] | None = None,
        residual_std: float | None = None,
    ) -> tuple[float, float]:
        """Compute prediction interval.

        Args:
            point_prediction: Point estimate
            score_function: Function to compute nonconformity scores
            residual_std: Standard deviation for simple intervals

        Returns:
            (lower, upper) prediction interval
        """
        q = self.get_quantile()

        if residual_std is not None:
            # Simple scaled interval
            half_width = q * residual_std if q != float("inf") else 1.96 * residual_std
        else:
            # Use quantile directly as half-width
            half_width = q if q != float("inf") else 1.0

        return (point_prediction - half_width, point_prediction + half_width)

    def get_diagnostics(self) -> dict[str, Any]:
        """Get ACI diagnostics."""
        return {
            "current_alpha": self.alpha,
            "target_coverage": self.target_coverage,
            "empirical_coverage": self.coverage_history[-1] if self.coverage_history else None,
            "calibration_size": len(self.scores),
            "quantile_threshold": self.get_quantile(),
        }


class HeteroscedasticEstimator:
    """Estimates input-dependent (heteroscedastic) aleatoric uncertainty.

    Uses local variance estimation or learns a variance prediction head. Based on Kendall & Gal
    (2017).
    """

    def __init__(self, window_size: int = 50, min_samples: int = 10) -> None:
        """Initialize the instance."""
        self.window_size = window_size
        self.min_samples = min_samples
        self._residuals: deque[float] = deque(maxlen=1000)
        self._features: deque[np.ndarray[Any, Any]] = deque(maxlen=1000)

    def update(
        self, prediction: float, true_value: float, features: np.ndarray[Any, Any] | None = None
    ) -> None:
        """Store residual for variance estimation."""
        residual = true_value - prediction
        self._residuals.append(residual)
        if features is not None:
            self._features.append(features.flatten()[:10])  # Store first 10 features

    def estimate_variance(self, features: np.ndarray[Any, Any] | None = None) -> float:
        """Estimate aleatoric variance, optionally conditioned on features.

        Args:
            features: Input features for heteroscedastic estimation

        Returns:
            Estimated variance
        """
        if len(self._residuals) < self.min_samples:
            return 0.1  # Default variance

        residuals = np.array(self._residuals)

        if features is None or len(self._features) < self.min_samples:
            # Homoscedastic: global variance
            return float(np.var(residuals))

        # Heteroscedastic: local variance based on feature similarity
        features_flat = features.flatten()[:10]
        stored_features = np.array(list(self._features))

        # Compute distances to stored points
        distances = np.linalg.norm(stored_features - features_flat, axis=1)

        # Kernel-weighted local variance (Nadaraya-Watson style)
        bandwidth = np.median(distances) + 1e-6
        weights = np.exp(-(distances**2) / (2 * bandwidth**2))
        weights /= weights.sum() + 1e-10

        # Weighted variance
        weighted_mean = np.sum(weights * residuals)
        weighted_var = np.sum(weights * (residuals - weighted_mean) ** 2)

        return max(float(weighted_var), 1e-6)


class UncertaintyQuantifier:
    """Production Uncertainty Quantification Engine.

    Implements uncertainty estimation following published methods:

    1. Monte Carlo Dropout (Gal & Ghahramani 2016)
       - Enables dropout at test time for epistemic uncertainty
       - Multiple stochastic forward passes
       - Variance across passes = epistemic uncertainty

    2. Deep Ensembles (Lakshminarayanan et al. 2017)
       - Aggregates predictions from multiple models
       - Model disagreement as uncertainty proxy
       - Both epistemic and aleatoric decomposition

    3. Temperature Scaling (Guo et al. 2017)
       - Post-hoc calibration via learned temperature
       - LBFGS optimization on validation NLL
       - Single parameter, model-agnostic

    4. Adaptive Conformal Inference (Gibbs & Candes 2021)
       - Distribution-free prediction intervals
       - Online adaptation to distribution shift
       - Finite-sample coverage guarantees

    5. Heteroscedastic Uncertainty (Kendall & Gal 2017)
       - Input-dependent aleatoric uncertainty
       - Local variance estimation
       - Separates data noise from model uncertainty

    6. Bayesian Confidence Calibration (optional)
       - Beta-Bernoulli conjugate prior per (domain, goal_type) context
       - Continuously improving confidence from outcome feedback
       - Blends raw UQ confidence with domain-specific historical accuracy
    """

    def __init__(
        self,
        n_monte_carlo: int = 30,
        calibration_bins: int = 15,
        reliability_threshold: float = 0.05,
        enable_aci: bool = True,
        aci_coverage: float = 0.9,
        bayesian_calibrator: Any | None = None,
        seed: int | None = None,
    ):
        """Initialize Uncertainty Quantifier.

        Args:
            n_monte_carlo: Number of MC samples for epistemic estimation
            calibration_bins: Number of bins for ECE calculation
            reliability_threshold: Max ECE for reliable predictions
            enable_aci: Enable Adaptive Conformal Inference
            aci_coverage: Target coverage for ACI
            bayesian_calibrator: Optional BayesianConfidenceCalibrator for
                domain-specific confidence adjustment via Beta-Bernoulli model
            seed: Optional seed for the per-instance ``Generator`` driving
                Monte-Carlo perturbation of the input. ``None`` (default)
                uses an OS-seeded ``Generator`` — same effective behavior
                as before.
        """
        self.n_monte_carlo = n_monte_carlo
        self.calibration_bins = calibration_bins
        self.reliability_threshold = reliability_threshold
        # Keep the constructor seed (not just the derived Generator) so the
        # confidence calibrator can be fit reproducibly: re-fitting the same
        # instance must use the same CV split / bootstrap seed, or the
        # accept/reject verdict would flap run-to-run.
        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Components
        self.temperature_scaler = TemperatureScaler()
        self.aci = AdaptiveConformalInference(target_coverage=aci_coverage) if enable_aci else None
        self.heteroscedastic = HeteroscedasticEstimator()
        self.bayesian_calibrator = bayesian_calibrator
        # The single confidence-calibration routing point. Fitted on the
        # accumulated (confidence, correct?) history via fit_confidence_calibrator;
        # until then confidence is the uncalibrated monotone prior (see
        # _compute_confidence) and UncertaintyEstimate.confidence_calibrated=False.
        self._confidence_calibrator: Any = None

        # Calibration history
        self._predictions: deque[float] = deque(maxlen=5000)
        self._confidences: deque[float] = deque(maxlen=5000)
        self._outcomes: deque[bool] = deque(maxlen=5000)
        self._logits_history: list[np.ndarray[Any, Any]] = []
        self._labels_history: list[int] = []

        # Statistics
        self._stats = {
            "estimates_computed": 0,
            "calibrations_performed": 0,
            "mc_samples_total": 0,
            "avg_epistemic": 0.0,
            "avg_aleatoric": 0.0,
            "temperature": 1.0,
        }

        logger.info(
            f"UncertaintyQuantifier initialized (MC={n_monte_carlo}, "
            f"ACI={'enabled' if enable_aci else 'disabled'}, "
            f"Bayesian={'enabled' if bayesian_calibrator else 'disabled'})"
        )

    def estimate_uncertainty(
        self,
        predictions: np.ndarray[Any, Any],
        prediction_function: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] | None = None,
        input_data: np.ndarray[Any, Any] | None = None,
        model: Any = None,
        return_samples: bool = False,
        ensemble_predictions: np.ndarray[Any, Any] | None = None,
    ) -> UncertaintyEstimate:
        """Estimate uncertainty for a prediction using MC Dropout / an ensemble.

        Args:
            predictions: Initial model predictions
            prediction_function: Function to generate predictions (for MC sampling)
            input_data: Input data for prediction function
            model: PyTorch model (for MC Dropout wrapper)
            return_samples: Whether to include MC samples in result
            ensemble_predictions: Optional stack of predictions from independent
                ensemble members / detectors, shape ``(n_members,)`` or
                ``(n_members, ...)``. When supplied (and no MC path ran), epistemic
                uncertainty is the **measured** variance across members rather than
                a placeholder.

        Returns:
            Complete uncertainty estimate with epistemic/aleatoric decomposition.
            ``epistemic_measured`` / ``aleatoric_measured`` flag whether each
            component was actually measured (ensemble/MC variance, heteroscedastic
            residuals) versus a default placeholder used when no such information
            was supplied -- so a caller passing only a single scalar gets an
            *transparent* "unmeasured" signal instead of a fabricated constant.
        """
        self._stats["estimates_computed"] += 1

        if predictions.ndim == 0:
            predictions = np.array([predictions])
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # === MC DROPOUT / ENSEMBLE SAMPLING ===
        mc_predictions = None

        if model is not None and TORCH_AVAILABLE:
            # Use MC Dropout wrapper for PyTorch models, threading our
            # constructor seed through so torch.manual_seed is set inside
            # ``predict_with_uncertainty`` for reproducible dropout draws.
            wrapper_seed: int | None = None
            if self._rng is not None:
                # Derive a deterministic torch seed from our NumPy RNG.
                # ``self._rng`` was built from the user-supplied seed in
                # ``__init__`` — calling ``.integers(...)`` here advances
                # it once and yields a stable value for repeated calls
                # against the same instance only AFTER reset; for
                # cross-call reproducibility callers should pass an
                # explicit ``seed`` and reuse the same ``UncertaintyQuantifier``
                # before any other RNG-consuming work.
                wrapper_seed = int(self._rng.integers(0, 2**31 - 1))
            wrapper = MCDropoutWrapper(model, seed=wrapper_seed)
            try:
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.FloatTensor(input_data)
                elif input_data is not None:
                    input_tensor = input_data
                else:
                    input_tensor = torch.FloatTensor(predictions)
                _mean_pred, _epistemic_std, mc_predictions = wrapper.predict_with_uncertainty(
                    input_tensor, n_samples=self.n_monte_carlo
                )
                self._stats["mc_samples_total"] += self.n_monte_carlo
            except Exception as e:
                logger.warning(f"MC Dropout failed: {e}, falling back to standard estimation")
                mc_predictions = None

        if mc_predictions is None and prediction_function is not None and input_data is not None:
            # Standard MC sampling via prediction function
            mc_predictions = self._monte_carlo_sampling(
                prediction_function, input_data, self.n_monte_carlo
            )
            self._stats["mc_samples_total"] += self.n_monte_carlo

        if mc_predictions is None and ensemble_predictions is not None:
            # Explicit ensemble: treat each member as a sample so epistemic
            # uncertainty is the measured cross-member variance.
            ens = np.asarray(ensemble_predictions, dtype=float)
            if ens.ndim == 1:
                ens = ens.reshape(-1, 1)
            if ens.shape[0] > 1:
                mc_predictions = ens

        if mc_predictions is None:
            mc_predictions = predictions

        # === EPISTEMIC UNCERTAINTY ===
        # Variance across MC samples (reducible with more data/better model)
        if mc_predictions.shape[0] > 1:
            epistemic = float(np.var(mc_predictions, axis=0).mean())
            epistemic_measured = True
            # Compute BALD: mutual information for active learning
            predictive_entropy = self._compute_predictive_entropy(mc_predictions)
            expected_entropy = self._compute_expected_entropy(mc_predictions)
            mutual_information = max(0, predictive_entropy - expected_entropy)
        else:
            # No ensemble/MC samples were supplied -- epistemic uncertainty is
            # NOT measurable from a single point. Use a neutral placeholder and
            # flag it as unmeasured rather than presenting 0.1 as a measurement.
            epistemic = 0.1
            epistemic_measured = False
            predictive_entropy = 0.0
            mutual_information = 0.0

        # === ALEATORIC UNCERTAINTY ===
        # Input-dependent data uncertainty (heteroscedastic)
        if input_data is not None:
            aleatoric = self.heteroscedastic.estimate_variance(input_data)
            # estimate_variance returns a default until enough residuals are seen.
            aleatoric_measured = (
                len(self.heteroscedastic._residuals) >= self.heteroscedastic.min_samples
            )
        else:
            aleatoric = self._estimate_aleatoric_from_predictions(mc_predictions)
            aleatoric_measured = mc_predictions.shape[0] > 1

        # === TOTAL UNCERTAINTY ===
        # Proper propagation: sqrt(epistemic^2 + aleatoric^2)
        total = np.sqrt(epistemic**2 + aleatoric**2)

        # === POINT ESTIMATE AND INTERVALS ===
        prediction = float(mc_predictions.mean())

        # Confidence interval via ACI or standard normal
        if self.aci is not None and len(self.aci.scores) >= 10:
            ci_low, ci_high = self.aci.predict_interval(prediction, residual_std=np.sqrt(total))
        else:
            # Standard 95% CI
            ci_low = prediction - 1.96 * np.sqrt(total)
            ci_high = prediction + 1.96 * np.sqrt(total)

        # === CONFIDENCE ===
        # Route through the single calibration point when one has been fitted on
        # observed outcomes (see fit_confidence_calibrator); otherwise emit the
        # uncalibrated, parameter-free monotone prior (no golden-ratio constant)
        # and flag it as uncalibrated below.
        confidence = self._compute_confidence(epistemic, aleatoric)
        confidence_calibrated = bool(
            self._confidence_calibrator is not None
            and getattr(self._confidence_calibrator, "is_calibrated", False)
        )

        calibration_error = self._compute_ece()

        # === RELIABILITY ASSESSMENT ===
        is_reliable = (
            calibration_error < self.reliability_threshold
            and epistemic < 0.5
            and confidence > 0.3
            and (
                self.aci is None
                or len(self.aci.coverage_history) < 10
                or self.aci.coverage_history[-1] > 0.8
            )
        )

        # === OVERCONFIDENCE DETECTION ===
        # Ref: Kaddour et al. (2026) "Agentic Uncertainty Reveals Agentic Overconfidence"
        # High confidence paired with poor calibration signals systematic overconfidence.
        is_overconfident = confidence > 0.8 and calibration_error > self.reliability_threshold

        # === EXPLANATION ===
        explanation = self._generate_explanation(
            epistemic,
            aleatoric,
            confidence,
            is_reliable,
            mutual_information,
            predictive_entropy,
            is_overconfident,
        )
        if not (epistemic_measured and aleatoric_measured):
            unmeasured = []
            if not epistemic_measured:
                unmeasured.append("epistemic (no ensemble/MC samples)")
            if not aleatoric_measured:
                unmeasured.append("aleatoric (insufficient residual history)")
            explanation = (
                f"NOTE: {', '.join(unmeasured)} not measured this call -- placeholder used; "
                "supply an ensemble/model or accumulate outcomes for a measured estimate. "
            ) + explanation
        if not confidence_calibrated:
            explanation += (
                " Confidence is an uncalibrated monotone prior (no calibrator fitted yet); "
                "call fit_confidence_calibrator once outcomes accumulate."
            )

        # Update running statistics
        alpha = 0.05  # EMA smoothing
        self._stats["avg_epistemic"] = (1 - alpha) * self._stats[
            "avg_epistemic"
        ] + alpha * epistemic
        self._stats["avg_aleatoric"] = (1 - alpha) * self._stats[
            "avg_aleatoric"
        ] + alpha * aleatoric
        self._stats["temperature"] = self.temperature_scaler.temperature

        return UncertaintyEstimate(
            prediction=prediction,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence,
            confidence_interval=(ci_low, ci_high),
            calibration_error=calibration_error,
            is_reliable=is_reliable,
            explanation=explanation,
            mutual_information=mutual_information,
            predictive_entropy=predictive_entropy,
            mc_samples=self.n_monte_carlo if mc_predictions.shape[0] > 1 else 0,
            is_overconfident=is_overconfident,
            epistemic_measured=epistemic_measured,
            aleatoric_measured=aleatoric_measured,
            confidence_calibrated=confidence_calibrated,
        )

    def estimate_epistemic(
        self,
        prediction_function: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
        input_data: np.ndarray[Any, Any],
        n_samples: int | None = None,
    ) -> float:
        """Estimate epistemic uncertainty via MC sampling.

        Args:
            prediction_function: Stochastic prediction function
            input_data: Input data
            n_samples: Override default MC samples

        Returns:
            Epistemic uncertainty (variance across samples)
        """
        n = n_samples or self.n_monte_carlo
        mc_predictions = self._monte_carlo_sampling(prediction_function, input_data, n)
        return float(np.var(mc_predictions, axis=0).mean())

    def estimate_aleatoric(
        self,
        data: np.ndarray[Any, Any],
        features: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """Estimate aleatoric uncertainty (heteroscedastic if features provided).

        Args:
            data: Observed data
            features: Input features for heteroscedastic estimation

        Returns:
            Aleatoric uncertainty estimate
        """
        if features is not None:
            return self.heteroscedastic.estimate_variance(features)
        return float(np.var(data))

    def calibrate(
        self,
        predictions: np.ndarray[Any, Any],
        confidences: np.ndarray[Any, Any],
        outcomes: np.ndarray[Any, Any],
        logits: np.ndarray[Any, Any] | None = None,
    ) -> CalibrationResult:
        """Assess calibration and optionally fit temperature scaling.

        Args:
            predictions: Model predictions
            confidences: Confidence scores
            outcomes: True outcomes (binary)
            logits: Pre-softmax outputs (for temperature scaling)

        Returns:
            Calibration assessment with ECE, MCE, and temperature
        """
        self._stats["calibrations_performed"] += 1

        # Fit temperature scaling if logits provided
        if logits is not None and len(np.unique(outcomes)) > 1:
            # Ensure labels are proper format
            labels = outcomes.astype(int)
            if logits.ndim == 1:
                logits = np.column_stack([1 - logits, logits])
            self.temperature_scaler.fit(logits, labels)

        # === BINNED CALIBRATION METRICS (Vectorized O(n) implementation) ===
        bin_boundaries = np.linspace(0, 1, self.calibration_bins + 1)

        # Vectorized binning using digitize (O(n) instead of O(n*bins))
        bin_indices = np.digitize(confidences, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.calibration_bins - 1)

        # Vectorized computation of bin statistics using bincount
        bin_counts_arr = np.bincount(bin_indices, minlength=self.calibration_bins)
        bin_conf_sums = np.bincount(
            bin_indices, weights=confidences, minlength=self.calibration_bins
        )
        bin_acc_sums = np.bincount(
            bin_indices, weights=outcomes.astype(float), minlength=self.calibration_bins
        )

        # Compute per-bin averages and handle empty bins
        non_empty = bin_counts_arr > 0
        expected_conf_arr = np.zeros(self.calibration_bins)
        observed_acc_arr = np.zeros(self.calibration_bins)

        # For non-empty bins: compute actual averages
        expected_conf_arr[non_empty] = bin_conf_sums[non_empty] / bin_counts_arr[non_empty]
        observed_acc_arr[non_empty] = bin_acc_sums[non_empty] / bin_counts_arr[non_empty]

        # For empty bins: use bin center as fallback
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        expected_conf_arr[~non_empty] = bin_centers[~non_empty]
        observed_acc_arr[~non_empty] = bin_centers[~non_empty]

        # Convert to lists for compatibility
        expected_conf = expected_conf_arr.tolist()
        observed_acc = observed_acc_arr.tolist()
        bin_counts = bin_counts_arr.tolist()

        # Vectorized calibration errors
        calibration_gaps = np.abs(expected_conf_arr - observed_acc_arr)
        adaptive_errors_arr = bin_counts_arr * calibration_gaps

        total_samples = int(bin_counts_arr.sum())

        # Expected Calibration Error (weighted average)
        ece = float(adaptive_errors_arr.sum() / max(total_samples, 1))

        # Maximum Calibration Error
        mce = float(calibration_gaps.max()) if len(calibration_gaps) > 0 else 0.0

        # Adaptive Calibration Error (handles varying bin sizes better)
        # Uses sqrt weighting to reduce sensitivity to large bins
        sqrt_weights = np.sqrt(bin_counts_arr)
        ace = float((sqrt_weights * calibration_gaps).sum() / (sqrt_weights.sum() + 1e-10))

        # Store history
        self._predictions.extend(predictions.tolist())
        self._confidences.extend(confidences.tolist())
        self._outcomes.extend(outcomes.tolist())

        # Update ACI with new data
        if self.aci is not None:
            for pred, conf, out in zip(predictions, confidences, outcomes, strict=False):
                score = abs(pred - out) if isinstance(out, (int, float)) else 0
                covered = conf > 0.5 if out else conf <= 0.5
                self.aci.update(score, covered)

        return CalibrationResult(
            expected_confidence=expected_conf,
            observed_accuracy=observed_acc,
            ece=float(ece),
            mce=float(mce),
            ace=float(ace),
            is_calibrated=ece < self.reliability_threshold,
            temperature=self.temperature_scaler.temperature,
            reliability_diagram={
                "expected": expected_conf,
                "observed": observed_acc,
                "counts": [float(c) for c in bin_counts],
            },
        )

    def uncertainty_aware_decision(
        self,
        uncertainty: UncertaintyEstimate,
        action_threshold: float = 0.5,
        epistemic_threshold: float = 0.3,
        aleatoric_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Make uncertainty-aware decisions with epistemic/aleatoric decomposition.

        Args:
            uncertainty: Uncertainty estimate
            action_threshold: Prediction threshold for action
            epistemic_threshold: Max epistemic uncertainty for action
            aleatoric_threshold: Max aleatoric uncertainty for action

        Returns:
            Decision with reasoning
        """
        decision = {
            "should_act": False,
            "should_defer": False,
            "should_collect_more_data": False,
            "action": "wait",
            "reason": "",
            "epistemic_concern": False,
            "aleatoric_concern": False,
        }

        # Check epistemic uncertainty (model doesn't know)
        if uncertainty.epistemic > epistemic_threshold:
            decision["epistemic_concern"] = True
            decision["should_collect_more_data"] = True
            decision["action"] = "collect_more_data"
            decision["reason"] = (
                f"High model uncertainty ({uncertainty.epistemic:.3f} > {epistemic_threshold}). "
                "Need more training data or model improvement."
            )
            return decision

        # Check aleatoric uncertainty (inherent noise)
        if uncertainty.aleatoric > aleatoric_threshold:
            decision["aleatoric_concern"] = True
            # Can't reduce aleatoric, but can acknowledge it
            decision["reason"] = (
                f"High data uncertainty ({uncertainty.aleatoric:.3f}). "
                "This is inherent variability that cannot be reduced."
            )

        # Check reliability
        if not uncertainty.is_reliable:
            decision["should_defer"] = True
            decision["action"] = "defer_to_human"
            decision["reason"] = (
                f"Prediction unreliable (ECE={uncertainty.calibration_error:.3f}). "
                "Human review recommended."
            )
            return decision

        # Check overconfidence (Kaddour et al. 2026)
        # High confidence with poor calibration is worse than low confidence —
        # the system doesn't know that it doesn't know.
        if uncertainty.is_overconfident:
            decision["should_defer"] = True
            decision["action"] = "defer_to_human"
            decision["reason"] = (
                f"Overconfidence detected: confidence={uncertainty.confidence:.1%} "
                f"but calibration_error={uncertainty.calibration_error:.3f}. "
                "Confidence exceeds calibration quality. Human review recommended."
            )
            return decision

        # Confident action
        if uncertainty.prediction > action_threshold and uncertainty.confidence > 0.7:
            decision["should_act"] = True
            decision["action"] = "take_action"
            decision["reason"] = (
                f"Confident prediction ({uncertainty.confidence:.1%}) "
                f"above threshold ({action_threshold}). "
                f"CI: [{uncertainty.confidence_interval[0]:.3f}, "
                f"{uncertainty.confidence_interval[1]:.3f}]"
            )
        else:
            decision["action"] = "monitor"
            decision["reason"] = (
                f"Prediction ({uncertainty.prediction:.3f}) with "
                f"confidence {uncertainty.confidence:.1%}. Monitoring."
            )

        return decision

    def decompose_uncertainty(
        self,
        predictions_ensemble: np.ndarray[Any, Any],
    ) -> dict[str, float]:
        """Decompose total uncertainty into epistemic and aleatoric.

        Uses law of total variance:
        - Epistemic = Var[E[Y|X, theta]]  (variance of means)
        - Aleatoric = E[Var[Y|X, theta]]  (mean of variances)

        Args:
            predictions_ensemble: Shape (n_models, n_samples, n_outputs)

        Returns:
            Decomposed uncertainties
        """
        if predictions_ensemble.ndim < 2:
            total = float(np.std(predictions_ensemble))
            return {
                "epistemic": 0.0,
                "aleatoric": total,
                "total": total,
                "ensemble_disagreement": 0.0,
            }

        if predictions_ensemble.ndim == 2:
            # (n_models, n_outputs) - single sample
            model_means = predictions_ensemble.mean(axis=1)
            epistemic = float(np.var(model_means))
            # No aleatoric from single sample
            aleatoric = 0.0
        else:
            # (n_models, n_samples, n_outputs)
            # Epistemic: variance of per-model means
            model_means = predictions_ensemble.mean(axis=1)  # (n_models, n_outputs)
            epistemic = float(np.var(model_means, axis=0).mean())

            # Aleatoric: mean of per-model variances
            model_vars = predictions_ensemble.var(axis=1)  # (n_models, n_outputs)
            aleatoric = float(np.mean(model_vars))

        total = np.sqrt(epistemic + aleatoric)
        epistemic_ratio = epistemic / (epistemic + aleatoric + 1e-10)

        # Ensemble disagreement: normalized spread of per-model predictions.
        # High disagreement signals low consensus among ensemble members,
        # which is a direct indicator of epistemic uncertainty.
        ensemble_disagreement = float(np.std(model_means))

        return {
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "total": float(total),
            "epistemic_ratio": float(epistemic_ratio),
            "ensemble_disagreement": ensemble_disagreement,
        }

    def conformal_prediction(
        self,
        calibration_scores: np.ndarray[Any, Any],
        test_score: float,
        alpha: float = 0.1,
    ) -> dict[str, Any]:
        """Standard conformal prediction interval.

        Args:
            calibration_scores: Nonconformity scores from calibration set
            test_score: Score for test point
            alpha: Significance level

        Returns:
            Conformal prediction result
        """
        n = len(calibration_scores)
        if n == 0:
            return {
                "in_prediction_set": True,
                "threshold": float("inf"),
                "test_score": test_score,
                "coverage_guarantee": 1 - alpha,
                "calibration_size": 0,
            }

        # Finite-sample valid quantile
        quantile_level = np.ceil((n + 1) * (1 - alpha)) / n
        quantile_level = min(1.0, quantile_level)

        threshold = float(np.quantile(calibration_scores, quantile_level))
        is_in_set = test_score <= threshold

        return {
            "in_prediction_set": is_in_set,
            "threshold": threshold,
            "test_score": test_score,
            "coverage_guarantee": 1 - alpha,
            "calibration_size": n,
            "effective_quantile": quantile_level,
        }

    def conformal_fused_scores(
        self,
        fused_scores: np.ndarray[Any, Any],
        residual_std: float | None = None,
    ) -> dict[str, Any]:
        """Wrap fused detection scores with conformal prediction intervals.

        Uses the Adaptive Conformal Inference (ACI) component to provide
        distribution-free intervals on post-fusion anomaly scores. This bridges
        the gap between conformal_prediction.py and the fusion layer — fused
        scores go in, intervals with coverage guarantees come out.

        Args:
            fused_scores: Array of fused anomaly scores from the fusion layer
            residual_std: Optional residual standard deviation for interval width.
                If None, uses the running average epistemic uncertainty.

        Returns:
            Dictionary with 'predictions', 'lower_bounds', 'upper_bounds',
            'coverage_level', and 'has_intervals' flag
        """
        result: dict[str, Any] = {
            "predictions": fused_scores,
            "has_intervals": False,
            "lower_bounds": fused_scores,
            "upper_bounds": fused_scores,
            "coverage_level": 0.0,
        }

        if self.aci is None or len(self.aci.scores) == 0:
            return result

        if residual_std is None:
            residual_std = max(self._stats.get("avg_epistemic", 0.1), 0.01)

        lowers = []
        uppers = []
        for score in fused_scores:
            lo, hi = self.aci.predict_interval(float(score), residual_std=residual_std)
            lowers.append(lo)
            uppers.append(hi)

        result["lower_bounds"] = np.array(lowers)
        result["upper_bounds"] = np.array(uppers)
        result["has_intervals"] = True
        result["coverage_level"] = self.aci.target_coverage

        return result

    def update_with_outcome(
        self,
        prediction: float,
        confidence: float,
        true_value: float | bool,
        features: np.ndarray[Any, Any] | None = None,
    ) -> None:
        """Update calibration and heteroscedastic estimates with observed outcome.

        Args:
            prediction: Model prediction
            confidence: Confidence score
            true_value: Observed true value
            features: Input features (for heteroscedastic update)
        """
        # Update heteroscedastic estimator
        true_value_float = float(true_value) if isinstance(true_value, bool) else true_value
        self.heteroscedastic.update(prediction, true_value_float, features)

        # Update ACI
        if self.aci is not None:
            score = abs(prediction - true_value_float)
            # Determine coverage (prediction within CI)
            covered = score < (1.96 * self._stats.get("avg_aleatoric", 0.1) + 0.1)
            self.aci.update(score, covered)

        # Store for calibration
        self._predictions.append(prediction)
        self._confidences.append(confidence)
        outcome = (prediction > 0.5) == (true_value_float > 0.5)
        self._outcomes.append(outcome)

    def calibrate_with_bayesian(
        self,
        raw_confidence: float,
        domain: str,
        goal: str = "detect",
    ) -> float:
        """Blend raw UQ confidence with Bayesian domain-specific calibration.

        When a BayesianConfidenceCalibrator is attached, this interpolates
        between the raw confidence from MC Dropout / temperature scaling
        and the Bayesian posterior mean for this (domain, goal_type) context.
        The interpolation weight is the Bayesian familiarity factor — novel
        contexts trust the raw UQ score, familiar contexts trust the posterior.

        Args:
            raw_confidence: Confidence from estimate_uncertainty()
            domain: Detection domain (e.g., "medical", "security", "network")
            goal: Goal description for context classification

        Returns:
            Calibrated confidence blending raw UQ with Bayesian prior
        """
        if self.bayesian_calibrator is None:
            return raw_confidence

        bayesian_confidence = self.bayesian_calibrator.get_confidence(domain, goal)

        # Weighted average: give equal weight to both sources.
        # The Bayesian calibrator internally handles familiarity weighting
        # (novel contexts → prior ≈ 0.76, familiar contexts → posterior).
        blended = 0.5 * raw_confidence + 0.5 * bayesian_confidence
        return float(np.clip(blended, 0.01, 0.99))

    def update_bayesian(
        self,
        domain: str,
        goal: str,
        success: bool,
        timestamp: float = 0.0,
    ) -> None:
        """Update the Bayesian calibrator with an observed outcome.

        Args:
            domain: Detection domain
            goal: Goal description
            success: Whether the detection was correct
            timestamp: Observation timestamp
        """
        if self.bayesian_calibrator is not None:
            self.bayesian_calibrator.update(domain, goal, success, timestamp)

    def _monte_carlo_sampling(
        self,
        prediction_function: Callable[..., Any],
        input_data: np.ndarray[Any, Any],
        n_samples: int,
    ) -> np.ndarray[Any, Any]:
        """Generate Monte Carlo samples."""
        samples = []
        for i in range(n_samples):
            try:
                # Add small noise to simulate stochastic prediction
                # (In real MC Dropout, this comes from dropout layers)
                noisy_input = input_data + self._rng.standard_normal(input_data.shape) * 0.01
                pred = prediction_function(noisy_input)
                if isinstance(pred, np.ndarray):
                    samples.append(pred.flatten())
                else:
                    samples.append(np.array([pred]).flatten())
            except Exception as e:
                if i == 0:
                    logger.warning(f"MC sampling error: {e}")
                continue

        if not samples:
            # Fallback
            return np.atleast_2d(input_data.mean())

        return np.array(samples)

    def _estimate_aleatoric_from_predictions(self, predictions: np.ndarray[Any, Any]) -> float:
        """Estimate aleatoric from prediction spread."""
        if predictions.ndim == 1 or predictions.shape[0] == 1:
            return 0.1  # Default

        # Use interquartile range for robustness
        q75, q25 = np.percentile(predictions, [75, 25])
        iqr = q75 - q25

        # Scale to variance-like quantity
        return float((iqr / 1.35) ** 2)

    def _compute_predictive_entropy(self, predictions: np.ndarray[Any, Any]) -> float:
        """Compute predictive entropy H[y|x, D]."""
        # Average prediction
        mean_pred = predictions.mean(axis=0)

        # For binary: H = -p*log(p) - (1-p)*log(1-p)
        mean_pred = np.clip(mean_pred, 1e-10, 1 - 1e-10)
        entropy = -mean_pred * np.log(mean_pred) - (1 - mean_pred) * np.log(1 - mean_pred)

        return float(np.mean(entropy))

    def _compute_expected_entropy(self, predictions: np.ndarray[Any, Any]) -> float:
        """Compute expected entropy E[H[y|x, theta]]."""
        # Entropy of each MC sample
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        entropies = -predictions * np.log(predictions) - (1 - predictions) * np.log(1 - predictions)

        return float(np.mean(entropies))

    def _raw_confidence(self, total: float) -> float:
        """Uncalibrated, parameter-free monotone prior: confidence from total UQ.

        ``1 / (1 + total)`` is strictly decreasing in total uncertainty, maps
        ``total=0 -> 1`` and saturates toward 0, and -- unlike the previous
        golden-ratio ``exp(-total * PHI)`` -- carries **no magic constant**. It is only a
        *prior*: it is not calibrated against accuracy until a calibrator is
        fitted (see :meth:`fit_confidence_calibrator`), which is why any estimate
        produced from it alone is flagged ``confidence_calibrated=False``.
        """
        return float(np.clip(1.0 / (1.0 + max(0.0, total)), 0.01, 0.99))

    def _compute_confidence(self, epistemic: float, aleatoric: float) -> float:
        """Confidence from uncertainties, routed through the calibrator if fit.

        Falls back to the uncalibrated monotone prior (:meth:`_raw_confidence`)
        until :meth:`fit_confidence_calibrator` has accepted a calibrator.
        """
        total = float(np.sqrt(epistemic**2 + aleatoric**2))
        raw = self._raw_confidence(total)
        cal = self._confidence_calibrator
        if cal is not None and getattr(cal, "is_calibrated", False):
            return float(np.clip(cal.transform_one(raw), 0.01, 0.99))
        return raw

    def fit_confidence_calibrator(self, min_samples: int = 50, method: str = "auto") -> Any:
        """Fit the confidence routing point on accumulated (confidence, correct?).

        Maps the raw monotone confidence onto measured P(correct) using the
        history recorded by :meth:`update_with_outcome`. Returns the
        :class:`~omni_mercury_engine.core.confidence.ConfidenceReport` (with
        held-out ECE/Brier) or ``None`` if there is not yet enough history.
        Subsequent ``estimate_uncertainty`` calls then emit calibrated
        confidence (``confidence_calibrated=True``).
        """
        from omni_mercury_engine.core.confidence import CalibratedConfidence

        if len(self._confidences) < min_samples or len(self._outcomes) < min_samples:
            return None
        scores = np.asarray(list(self._confidences), dtype=float)
        labels = np.asarray([1 if o else 0 for o in self._outcomes], dtype=float)
        # Use the instance's constructor seed (falling back to
        # CalibratedConfidence's fixed default when None) rather than drawing a
        # fresh seed from self._rng -- so re-fitting the same instance is
        # reproducible and the bootstrap accept/reject verdict does not flap.
        calibrator = CalibratedConfidence(method=method, seed=self._seed)
        report = calibrator.fit(scores, labels)
        self._confidence_calibrator = calibrator
        self._stats["calibrations_performed"] += 1
        return report

    def _compute_ece(self) -> float:
        """Compute current Expected Calibration Error using vectorized operations.

        Vectorized implementation for O(n) performance instead of O(n²) with loops. Uses numpy
        histogram and binned statistics for efficient binning.
        """
        if len(self._outcomes) < 20:
            return 0.1  # Default for limited data

        confidences = np.array(list(self._confidences)[-500:])
        outcomes = np.array(list(self._outcomes)[-500:])

        # Adaptive binning based on sample size
        n_bins = min(10, max(3, len(confidences) // 50))

        # Vectorized binning using digitize (O(n) operation)
        bin_indices = np.digitize(confidences, np.linspace(0, 1, n_bins + 1)) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Vectorized computation of bin statistics using bincount
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        bin_conf_sums = np.bincount(bin_indices, weights=confidences, minlength=n_bins)
        bin_acc_sums = np.bincount(bin_indices, weights=outcomes.astype(float), minlength=n_bins)

        # Compute per-bin averages (avoiding division by zero)
        non_empty = bin_counts > 0
        avg_confs = np.zeros(n_bins)
        avg_accs = np.zeros(n_bins)
        avg_confs[non_empty] = bin_conf_sums[non_empty] / bin_counts[non_empty]
        avg_accs[non_empty] = bin_acc_sums[non_empty] / bin_counts[non_empty]

        # Vectorized ECE: weighted sum of calibration errors
        calibration_errors = np.abs(avg_confs - avg_accs) * bin_counts
        ece = calibration_errors.sum() / len(confidences)

        return float(ece)

    def _generate_explanation(
        self,
        epistemic: float,
        aleatoric: float,
        confidence: float,
        is_reliable: bool,
        mutual_info: float,
        predictive_entropy: float,
        is_overconfident: bool = False,
    ) -> str:
        """Generate human-readable uncertainty explanation."""
        parts = []

        # Overconfidence warning (most critical — surface first)
        if is_overconfident:
            parts.append(
                f"WARNING: Overconfidence detected (confidence: {confidence:.0%}). "
                "Calibration history indicates confidence exceeds actual accuracy. "
                "Treat this prediction with skepticism."
            )

        # Overall reliability
        if is_reliable:
            parts.append(f"Prediction is reliable (confidence: {confidence:.0%}).")
        else:
            parts.append(f"Prediction has LIMITED reliability (confidence: {confidence:.0%}).")

        # Uncertainty breakdown with actionable insight
        total = np.sqrt(epistemic**2 + aleatoric**2)
        epistemic_pct = epistemic / (total + 1e-10) * 100

        if epistemic > aleatoric:
            parts.append(
                f"Model uncertainty dominates ({epistemic_pct:.0f}% of total). "
                f"Epistemic={epistemic:.3f}, Aleatoric={aleatoric:.3f}. "
                "ACTIONABLE: More training data or model improvements would help."
            )
        else:
            parts.append(
                f"Data uncertainty dominates ({100 - epistemic_pct:.0f}% of total). "
                f"Epistemic={epistemic:.3f}, Aleatoric={aleatoric:.3f}. "
                "This variability is inherent and cannot be reduced with more data."
            )

        # BALD for active learning
        if mutual_info > 0.1:
            parts.append(
                f"High information gain potential (MI={mutual_info:.3f}). "
                "This sample would be valuable for active learning."
            )

        return " ".join(parts)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive quantifier statistics."""
        stats: dict[str, Any] = {
            **self._stats,
            "prediction_history_size": len(self._predictions),
            "current_ece": self._compute_ece(),
            "temperature": self.temperature_scaler.temperature,
            "temperature_fitted": self.temperature_scaler._fitted,
        }

        if self.aci is not None:
            stats["aci"] = self.aci.get_diagnostics()

        return stats

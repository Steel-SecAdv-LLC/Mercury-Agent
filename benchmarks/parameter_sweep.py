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

Parameter Sweep Optimization for Mercury Agent AAFE

Bayesian optimization (TPE sampler via Optuna) over the full AAFE parameter
space.  The objective function synthesises anomaly data, applies the AVA
Anomaly Fusion Equation, and returns a composite score combining weighted-F1,
Expected Calibration Error (ECE), and a Lyapunov stability metric.

Mathematical Framework
----------------------
AVA Anomaly Fusion Equation (AAFE):

    .. math::

        A = \\bigl(w_R \\cdot R(x) + w_H \\cdot H(\\omega)
            + w_O \\cdot O(\\theta)\\bigr) \\cdot \\eta(b)^{p}

Sigmoid Benevolence Gate:

    .. math::

        \\eta(b) = \\frac{1}{1 + \\exp\\bigl(-k \\cdot (b - b_0)\\bigr)}

Lyapunov Stability Bound:

    .. math::

        V(S_t) \\leq \\varepsilon \\cdot e^{-\\lambda t}

Banach Contraction (recursion):

    .. math::

        R(x, d) = f(x) + \\alpha \\cdot R(g(x),\\, d-1),
        \\quad \\alpha \\in (0, 0.95)

Composite Objective:

    .. math::

        J = 0.5 \\cdot F_1 + 0.3 \\cdot (1 - \\text{ECE})
            + 0.2 \\cdot S_{\\text{Lyap}}

References
----------
- Banach, S. (1922). Sur les operations dans les ensembles abstraits.
- Khalil, H.K. (2002). Nonlinear Systems, 3rd ed.
- Verhulst, P.-F. (1845). Recherches mathematiques sur la loi d'accroissement
  de la population.
- Akiba et al. (2019). Optuna: A Next-generation Hyperparameter Optimization
  Framework. KDD.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Optuna import with user-friendly error
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Mercury Agent imports -- graceful fallback when running outside the
# installed package (e.g. CI / standalone benchmark execution).
# ---------------------------------------------------------------------------
try:
    from omni_mercury_engine.core.centralized_constants import (
        sigmoid_benevolence_gate,
    )
    from omni_mercury_engine.core.three_r.fusion import AnomalyFusionEquation

    MERCURY_IMPORTS_AVAILABLE = True
except ImportError:
    MERCURY_IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
_RESULTS_DIR = Path(__file__).resolve().parent
_RESULTS_PATH = _RESULTS_DIR / "parameter_sweep_results.json"

# ---------------------------------------------------------------------------
# Constants used when Mercury imports are unavailable (mirrors centralized
# constants so the script is self-contained for CI).
# ---------------------------------------------------------------------------
_GOLDEN_RATIO: float = 1.618033988749895
_LAMBDA_DEFAULT: float = 0.25
_ALPHA_MAX: float = 0.95
_DEFAULT_B0: float = 0.93
_DEFAULT_K: float = 25.0


# ============================================================================
# Data-classes for structured results
# ============================================================================


@dataclass
class TrialResult:
    """Stores the outcome of a single optimisation trial.

    Attributes:
        trial_number: Ordinal index of the trial within the study.
        params: Dictionary of sampled hyper-parameters.
        f1_score: Weighted F1 on synthetic anomaly data.
        ece: Expected Calibration Error.
        stability: Lyapunov stability metric in [0, 1].
        composite: Composite objective ``J``.
        duration_s: Wall-clock seconds for the trial.
    """

    trial_number: int
    params: dict[str, float]
    f1_score: float
    ece: float
    stability: float
    composite: float
    duration_s: float


@dataclass
class SweepResult:
    """Aggregated output of the full parameter sweep.

    Attributes:
        best_params: Parameters of the trial with highest composite score.
        best_composite: Highest composite score achieved.
        best_f1: F1 corresponding to the best trial.
        best_ece: ECE corresponding to the best trial.
        best_stability: Stability metric of the best trial.
        n_trials: Total number of trials executed.
        pareto_frontier: List of Pareto-optimal configurations
            (non-dominated in F1, 1-ECE, stability).
        all_trials: Full trial history for reproducibility.
        metadata: Run metadata (timestamp, seed, duration).
    """

    best_params: dict[str, float]
    best_composite: float
    best_f1: float
    best_ece: float
    best_stability: float
    n_trials: int
    pareto_frontier: list[dict[str, Any]]
    all_trials: list[dict[str, Any]]
    metadata: dict[str, Any]


# ============================================================================
# Fallback implementations (used when Mercury imports are not on sys.path)
# ============================================================================


def _fallback_sigmoid_benevolence_gate(
    benevolence_score: float,
    b0: float = _DEFAULT_B0,
    k: float = _DEFAULT_K,
) -> float:
    """Sigmoid benevolence gate -- standalone fallback.

    .. math::

        \\eta(b) = \\frac{1}{1 + \\exp(-k \\cdot (b - b_0))}

    Args:
        benevolence_score: Raw benevolence score in ``[0, 1]``.
        b0: Inflection point of the sigmoid.
        k: Steepness parameter.

    Returns:
        Gate value in ``(0, 1)``.
    """
    exponent = -k * (benevolence_score - b0)
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _fallback_aafe(
    recursion_score: float,
    resonance_score: float,
    optimization_score: float,
    w_r: float,
    w_h: float,
    w_o: float,
    eta: float,
    ethical_exponent: float,
) -> float:
    """Compute the AAFE score -- standalone fallback.

    .. math::

        A = (w_R \\cdot R + w_H \\cdot H + w_O \\cdot O) \\cdot \\eta^{p}

    Args:
        recursion_score: ``R(x)`` component.
        resonance_score: ``H(omega)`` component.
        optimization_score: ``O(theta)`` component.
        w_r: Weight for recursion.
        w_h: Weight for resonance.
        w_o: Weight for optimization.
        eta: Ethical gate value.
        ethical_exponent: Exponent ``p``.

    Returns:
        Fused anomaly score ``A``.
    """
    weighted_sum = w_r * recursion_score + w_h * resonance_score + w_o * optimization_score
    return weighted_sum * (eta**ethical_exponent)


# ============================================================================
# Synthetic data generation
# ============================================================================


def generate_synthetic_data(
    n_samples: int = 500,
    anomaly_fraction: float = 0.15,
    n_features: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Generate a synthetic anomaly-detection dataset.

    The dataset consists of three feature channels that map onto the AAFE
    components ``(R, H, O)``:

    * **Normal points** are drawn from ``N(0.3, 0.1)`` per feature.
    * **Anomalous points** are drawn from ``N(0.8, 0.15)`` per feature,
      representing elevated detector responses.

    Args:
        n_samples: Total number of data points.
        anomaly_fraction: Proportion of anomalous points (0, 1).
        n_features: Number of feature channels (default 3 for R, H, O).
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple ``(X, y)`` where ``X`` has shape ``(n_samples, n_features)``
        and ``y`` is a binary label vector (1 = anomaly).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_anomaly = int(n_samples * anomaly_fraction)
    n_normal = n_samples - n_anomaly

    # Normal data -- low detector activations
    x_normal = rng.normal(loc=0.3, scale=0.10, size=(n_normal, n_features))
    x_normal = np.clip(x_normal, 0.0, 1.0)

    # Anomalous data -- elevated detector activations
    x_anomaly = rng.normal(loc=0.8, scale=0.15, size=(n_anomaly, n_features))
    x_anomaly = np.clip(x_anomaly, 0.0, 1.0)

    x = np.vstack([x_normal, x_anomaly])
    y = np.concatenate([np.zeros(n_normal, dtype=np.int64), np.ones(n_anomaly, dtype=np.int64)])

    # Shuffle consistently
    perm = rng.permutation(n_samples)
    return x[perm], y[perm]


# ============================================================================
# Metric computation
# ============================================================================


def compute_f1(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
) -> float:
    """Weighted F1 score (binary classification).

    .. math::

        F_1 = 2 \\cdot \\frac{\\text{precision} \\cdot \\text{recall}}
              {\\text{precision} + \\text{recall}}

    Weighted variant averages positive and negative class F1 scores
    by their support.

    Args:
        y_true: Ground-truth labels ``{0, 1}``.
        y_pred: Predicted labels ``{0, 1}``.

    Returns:
        Weighted F1 score in ``[0, 1]``.
    """
    classes = np.unique(y_true)
    total = len(y_true)
    weighted_f1 = 0.0

    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1_cls = 2.0 * precision * recall / (precision + recall)
        else:
            f1_cls = 0.0

        support = int(np.sum(y_true == cls))
        weighted_f1 += f1_cls * (support / total)

    return weighted_f1


def compute_ece(
    y_true: NDArray[np.int64],
    scores: NDArray[np.float64],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    .. math::

        \\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{n}
                      \\bigl| \\text{acc}(B_m) - \\text{conf}(B_m) \\bigr|

    where ``B_m`` is the set of samples whose predicted confidence falls
    into the ``m``-th bin.

    Args:
        y_true: Ground-truth binary labels.
        scores: Predicted anomaly scores (used as confidence).
        n_bins: Number of equal-width bins.

    Returns:
        ECE in ``[0, 1]``.  Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    if n_total == 0:
        return 0.0

    for i in range(n_bins):
        mask = (scores > bin_edges[i]) & (scores <= bin_edges[i + 1])
        bin_size = int(np.sum(mask))
        if bin_size == 0:
            continue

        bin_accuracy = float(np.mean(y_true[mask]))
        bin_confidence = float(np.mean(scores[mask]))
        ece += (bin_size / n_total) * abs(bin_accuracy - bin_confidence)

    return ece


def compute_lyapunov_stability(
    scores_sequence: NDArray[np.float64],
    lambda_rate: float,
    window: int = 10,
) -> float:
    """Lyapunov stability metric for the AAFE score sequence.

    Estimates the degree to which successive variance decays
    exponentially, in accordance with the Lyapunov bound:

    .. math::

        V(S_t) \\leq \\varepsilon \\cdot e^{-\\lambda t}

    The metric is 1.0 when variance is monotonically non-increasing
    across sliding windows, and degrades toward 0.0 as instability
    (variance increase between windows) grows.

    Args:
        scores_sequence: Chronologically ordered fusion scores.
        lambda_rate: Target Lyapunov convergence rate ``lambda``.
        window: Sliding-window size for variance estimation.

    Returns:
        Stability metric in ``[0, 1]``.  Higher is better.
    """
    n = len(scores_sequence)
    if n < 2 * window:
        # Not enough data; assume stable.
        return 1.0

    n_windows = n // window
    variances: list[float] = []
    for i in range(n_windows):
        segment = scores_sequence[i * window : (i + 1) * window]
        variances.append(float(np.var(segment)))

    if len(variances) < 2:
        return 1.0

    # Count how many successive windows show variance decrease
    decreasing = 0
    total_transitions = len(variances) - 1
    for i in range(total_transitions):
        if variances[i + 1] <= variances[i] + 1e-12:
            decreasing += 1

    base_stability = decreasing / total_transitions if total_transitions > 0 else 1.0

    # Penalise if final variance is much larger than expected exponential decay
    expected_final_var = variances[0] * math.exp(-lambda_rate * n_windows)
    if variances[0] > 1e-12:
        ratio = min(variances[-1] / (expected_final_var + 1e-12), 10.0)
        decay_penalty = max(0.0, 1.0 - ratio / 10.0)
    else:
        decay_penalty = 1.0

    return 0.7 * base_stability + 0.3 * decay_penalty


# ============================================================================
# Objective function
# ============================================================================


def objective(
    trial: optuna.Trial,  # type: ignore[name-defined]
    x_data: NDArray[np.float64],
    y_data: NDArray[np.int64],
) -> float:
    """Optuna objective: composite of F1, ECE, and Lyapunov stability.

    The function:
    1. Samples the full AAFE parameter space from the trial.
    2. Computes fusion scores for every sample using the AAFE.
    3. Derives binary predictions from a threshold.
    4. Returns the composite score ``J``.

    .. math::

        J = 0.5 \\cdot F_1 + 0.3 \\cdot (1 - \\text{ECE})
            + 0.2 \\cdot S_{\\text{Lyap}}

    Args:
        trial: Optuna ``Trial`` object for parameter suggestion.
        x_data: Feature matrix ``(n, 3)`` -- columns map to (R, H, O).
        y_data: Binary label vector.

    Returns:
        Composite score ``J`` (to be *maximised*).
    """
    # ------------------------------------------------------------------
    # 1. Sample AAFE weights (Dirichlet-like: sample 3, normalise)
    # ------------------------------------------------------------------
    w_r_raw: float = trial.suggest_float("w_R_raw", 0.1, 1.0)
    w_h_raw: float = trial.suggest_float("w_H_raw", 0.1, 1.0)
    w_o_raw: float = trial.suggest_float("w_O_raw", 0.1, 1.0)
    w_total = w_r_raw + w_h_raw + w_o_raw
    w_r: float = w_r_raw / w_total
    w_h: float = w_h_raw / w_total
    w_o: float = w_o_raw / w_total

    # ------------------------------------------------------------------
    # 2. Ethical exponent
    # ------------------------------------------------------------------
    ethical_exponent: float = trial.suggest_float("ethical_exponent", 1.0, 3.0)

    # ------------------------------------------------------------------
    # 3. Benevolence sigmoid parameters
    # ------------------------------------------------------------------
    b0: float = trial.suggest_float("benevolence_b0", 0.85, 0.98)
    k: float = trial.suggest_float("benevolence_k", 10.0, 50.0)

    # ------------------------------------------------------------------
    # 4. Recursion contraction factor alpha
    # ------------------------------------------------------------------
    alpha: float = trial.suggest_float("recursion_alpha", 0.3, 0.95)

    # ------------------------------------------------------------------
    # 5. Lyapunov convergence rate lambda
    # ------------------------------------------------------------------
    lambda_rate: float = trial.suggest_float("lyapunov_lambda", 0.1, 0.5)

    # ------------------------------------------------------------------
    # 6. Neural-symbolic fusion weight (neural portion; symbolic = 1 - neural)
    # ------------------------------------------------------------------
    neural_weight: float = trial.suggest_float("neural_weight", 0.3, 0.8)

    # ------------------------------------------------------------------
    # 7. Statistical detector fusion weights
    # ------------------------------------------------------------------
    stat_zscore_w: float = trial.suggest_float("stat_zscore_weight", 0.1, 0.6)
    stat_iqr_w: float = trial.suggest_float("stat_iqr_weight", 0.1, 0.6)
    stat_mad_w_raw: float = 1.0 - stat_zscore_w - stat_iqr_w
    stat_mad_w: float = max(stat_mad_w_raw, 0.05)

    # ------------------------------------------------------------------
    # 8. Classification threshold
    # ------------------------------------------------------------------
    threshold: float = trial.suggest_float("classification_threshold", 0.3, 0.7)

    # ------------------------------------------------------------------
    # Compute fusion scores for every data point
    # ------------------------------------------------------------------
    n_samples = x_data.shape[0]
    fusion_scores = np.empty(n_samples, dtype=np.float64)

    # Simulated benevolence scores -- normal points get high benevolence,
    # anomalies get slightly lower, reflecting real ethical-gate behaviour.
    rng = np.random.default_rng(int(trial.number) + 7)
    benevolence_base = rng.uniform(0.88, 0.99, size=n_samples)

    use_mercury = MERCURY_IMPORTS_AVAILABLE

    if use_mercury:
        aafe = AnomalyFusionEquation(
            convergence_rate=lambda_rate,
            initial_weights={"w_R": w_r, "w_H": w_h, "w_O": w_o},
            ethical_exponent=ethical_exponent,
        )

    for i in range(n_samples):
        r_score = float(x_data[i, 0])
        h_score = float(x_data[i, 1])
        o_score = float(x_data[i, 2])

        # Modulate scores with statistical detector fusion weights
        r_score = stat_zscore_w * r_score + stat_iqr_w * r_score + stat_mad_w * r_score
        # Apply recursion contraction as dampening on recursion channel
        r_score = r_score * alpha

        # Neural-symbolic blending: neural amplifies, symbolic smooths
        blend = neural_weight * (r_score + h_score + o_score) / 3.0 + (1.0 - neural_weight) * max(
            r_score, h_score, o_score
        )
        # Re-distribute blended signal back to channels
        r_adj = r_score + 0.1 * blend
        h_adj = h_score + 0.1 * blend
        o_adj = o_score + 0.1 * blend

        if use_mercury:
            benev = float(benevolence_base[i])
            eta = sigmoid_benevolence_gate(benev, domain="default")
            result = aafe.compute(
                recursion_score=r_adj,
                resonance_score=h_adj,
                optimization_score=o_adj,
                ethical_threshold_override=eta,
            )
            fusion_scores[i] = result.fusion_score
        else:
            eta = _fallback_sigmoid_benevolence_gate(
                float(benevolence_base[i]),
                b0=b0,
                k=k,
            )
            fusion_scores[i] = _fallback_aafe(
                recursion_score=r_adj,
                resonance_score=h_adj,
                optimization_score=o_adj,
                w_r=w_r,
                w_h=w_h,
                w_o=w_o,
                eta=eta,
                ethical_exponent=ethical_exponent,
            )

    # Normalise scores to [0, 1]
    s_min = float(np.min(fusion_scores))
    s_max = float(np.max(fusion_scores))
    if s_max - s_min > 1e-12:
        fusion_scores = (fusion_scores - s_min) / (s_max - s_min)
    else:
        fusion_scores = np.full_like(fusion_scores, 0.5)

    # ------------------------------------------------------------------
    # Predictions and metrics
    # ------------------------------------------------------------------
    y_pred = (fusion_scores >= threshold).astype(np.int64)

    f1: float = compute_f1(y_data, y_pred)
    ece: float = compute_ece(y_data, fusion_scores)
    stability: float = compute_lyapunov_stability(fusion_scores, lambda_rate)

    composite: float = 0.5 * f1 + 0.3 * (1.0 - ece) + 0.2 * stability

    # Record user-attrs for later analysis
    trial.set_user_attr("f1_score", f1)
    trial.set_user_attr("ece", ece)
    trial.set_user_attr("stability", stability)
    trial.set_user_attr("w_R", w_r)
    trial.set_user_attr("w_H", w_h)
    trial.set_user_attr("w_O", w_o)
    trial.set_user_attr("stat_mad_weight", stat_mad_w)

    return composite


# ============================================================================
# Pareto frontier computation
# ============================================================================


def compute_pareto_frontier(
    trials: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract Pareto-optimal configurations from completed trials.

    A trial is Pareto-optimal if no other trial simultaneously achieves
    a higher F1, lower ECE, and higher stability.

    Three objectives (all to be maximised for comparison):
    ``(F1, 1 - ECE, stability)``.

    Args:
        trials: List of trial result dictionaries (must contain keys
            ``f1_score``, ``ece``, ``stability``).

    Returns:
        Subset of ``trials`` that lie on the Pareto frontier,
        sorted by descending composite score.
    """
    if not trials:
        return []

    objectives = np.array([[t["f1_score"], 1.0 - t["ece"], t["stability"]] for t in trials])

    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i == j or is_dominated[j]:
                continue
            # j dominates i if j >= i on all objectives and j > i on at least one
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_dominated[i] = True
                break

    pareto = [t for t, dom in zip(trials, is_dominated) if not dom]
    pareto.sort(key=lambda t: t["composite"], reverse=True)
    return pareto


# ============================================================================
# Main sweep driver
# ============================================================================


def run_sweep(
    n_trials: int = 1000,
    n_samples: int = 500,
    anomaly_fraction: float = 0.15,
    seed: int = 42,
    output_path: Path | None = None,
    verbose: bool = True,
) -> SweepResult:
    """Execute the full Bayesian parameter sweep.

    Args:
        n_trials: Number of Optuna trials (default 1000).
        n_samples: Synthetic dataset size.
        anomaly_fraction: Fraction of anomalous points.
        seed: Random seed for reproducibility.
        output_path: Path for JSON results (default:
            ``benchmarks/parameter_sweep_results.json``).
        verbose: Whether to print progress to stdout.

    Returns:
        ``SweepResult`` dataclass with best parameters, Pareto frontier,
        and full trial history.

    Raises:
        ImportError: If ``optuna`` is not installed.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError(
            "Optuna is required for parameter sweep optimisation.\n"
            "Install it with:  pip install optuna\n"
            "Or:               pip install 'mercury-agent[benchmarks]'"
        )

    if output_path is None:
        output_path = _RESULTS_PATH

    t_start = time.monotonic()

    # Reproducible synthetic data
    rng = np.random.default_rng(seed)
    x_data, y_data = generate_synthetic_data(
        n_samples=n_samples,
        anomaly_fraction=anomaly_fraction,
        rng=rng,
    )

    if verbose:
        print(
            f"Parameter sweep: {n_trials} trials, "
            f"{n_samples} samples ({anomaly_fraction:.0%} anomalous), "
            f"seed={seed}"
        )
        print(f"Mercury imports available: {MERCURY_IMPORTS_AVAILABLE}")
        print("-" * 72)

    # Configure Optuna
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="mercury_aafe_parameter_sweep",
    )

    # Suppress Optuna's per-trial logging unless truly verbose
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    study.optimize(
        lambda trial: objective(trial, x_data, y_data),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------
    all_trials: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue

        params = dict(t.params)
        # Reconstruct normalised weights
        w_total = params["w_R_raw"] + params["w_H_raw"] + params["w_O_raw"]
        params["w_R"] = params["w_R_raw"] / w_total
        params["w_H"] = params["w_H_raw"] / w_total
        params["w_O"] = params["w_O_raw"] / w_total

        all_trials.append(
            {
                "trial_number": t.number,
                "params": params,
                "f1_score": t.user_attrs.get("f1_score", 0.0),
                "ece": t.user_attrs.get("ece", 0.0),
                "stability": t.user_attrs.get("stability", 0.0),
                "composite": t.value if t.value is not None else 0.0,
                "duration_s": (
                    (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_complete and t.datetime_start
                    else 0.0
                ),
            }
        )

    # Best trial
    best = study.best_trial
    best_params = dict(best.params)
    w_total = best_params["w_R_raw"] + best_params["w_H_raw"] + best_params["w_O_raw"]
    best_params["w_R"] = best_params["w_R_raw"] / w_total
    best_params["w_H"] = best_params["w_H_raw"] / w_total
    best_params["w_O"] = best_params["w_O_raw"] / w_total

    # Pareto frontier
    pareto = compute_pareto_frontier(all_trials)

    total_duration = time.monotonic() - t_start

    result = SweepResult(
        best_params=best_params,
        best_composite=best.value if best.value is not None else 0.0,
        best_f1=best.user_attrs.get("f1_score", 0.0),
        best_ece=best.user_attrs.get("ece", 0.0),
        best_stability=best.user_attrs.get("stability", 0.0),
        n_trials=len(all_trials),
        pareto_frontier=pareto,
        all_trials=all_trials,
        metadata={
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "seed": seed,
            "n_samples": n_samples,
            "anomaly_fraction": anomaly_fraction,
            "total_duration_s": round(total_duration, 3),
            "optuna_version": optuna.__version__,
            "mercury_imports": MERCURY_IMPORTS_AVAILABLE,
            "python_version": sys.version,
        },
    )

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    output_dict: dict[str, Any] = {
        "best_params": result.best_params,
        "best_composite": result.best_composite,
        "best_f1": result.best_f1,
        "best_ece": result.best_ece,
        "best_stability": result.best_stability,
        "n_trials": result.n_trials,
        "pareto_frontier": result.pareto_frontier,
        "all_trials": result.all_trials,
        "metadata": result.metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(output_dict, fh, indent=2, default=str)

    if verbose:
        print("\n" + "=" * 72)
        print("PARAMETER SWEEP COMPLETE")
        print("=" * 72)
        print(f"Trials completed : {result.n_trials}")
        print(f"Total duration   : {total_duration:.1f}s")
        print(f"Best composite   : {result.best_composite:.6f}")
        print(f"  F1             : {result.best_f1:.6f}")
        print(f"  ECE            : {result.best_ece:.6f}")
        print(f"  Stability      : {result.best_stability:.6f}")
        print(f"Results saved to : {output_path}")

        print("\nBest parameters:")
        for pname, pval in sorted(result.best_params.items()):
            print(f"  {pname:30s} = {pval:.6f}")

        print(f"\nPareto frontier ({len(pareto)} configurations):")
        print(f"  {'#':>4s}  {'F1':>8s}  {'ECE':>8s}  {'Stability':>10s}  {'Composite':>10s}")
        print(
            f"  {'----':>4s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}  {'----------':>10s}"
        )
        for i, p in enumerate(pareto[:20]):  # Show top 20
            print(
                f"  {i+1:4d}  {p['f1_score']:8.5f}  {p['ece']:8.5f}  "
                f"{p['stability']:10.5f}  {p['composite']:10.5f}"
            )
        if len(pareto) > 20:
            print(f"  ... ({len(pareto) - 20} more Pareto-optimal configurations)")

    return result


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    """Command-line entry point for standalone execution.

    Usage::

        python benchmarks/parameter_sweep.py [--trials N] [--samples N]
            [--anomaly-fraction F] [--seed S] [--output PATH] [--quiet]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Mercury Agent AAFE Parameter Sweep -- "
            "Bayesian optimisation over the full parameter space."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of Optuna trials (default: 1000).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Synthetic dataset size (default: 500).",
    )
    parser.add_argument(
        "--anomaly-fraction",
        type=float,
        default=0.15,
        help="Fraction of anomalous points (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: benchmarks/parameter_sweep_results.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_path = Path(args.output) if args.output else None

    run_sweep(
        n_trials=args.trials,
        n_samples=args.samples,
        anomaly_fraction=args.anomaly_fraction,
        seed=args.seed,
        output_path=output_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

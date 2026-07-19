# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only calibration + threshold/temperature sweep for the σ_Immutable gate.

The σ_Immutable :class:`~omni_mercury_engine.core.global_omni_scalar_network.EthicalGate`
was trained and constant-score-verified, but its *internal calibration* (ECE /
reliability) was never measured and its threshold / temperature were never
swept. This module supplies that measurement layer -- and does so **without
touching the operational path**:

* it constructs a *fresh* ``EthicalGate`` that loads the same frozen
  ``security/sigma_immutable_weights.pt``; it never trains, never writes weights,
  never edits ``centralized_constants``, and never changes ``evaluate()``;
* the operational threshold stays ``EthicalConstants.SIGMA_IMMUTABLE_TRAINED_THRESHOLD``
  (0.93); every swept threshold here is a *measurement*, not a repoint;
* **temperature** is applied at the only correct point. The gate's ``Sigmoid``
  is the last module of its ``nn.Sequential``, so the raw logit is never exposed
  by ``evaluate()``. We read the pre-sigmoid activation via ``gate_network[:3]``
  and apply ``sigmoid(logit / T)`` -- never ``log(p / (1 - p))``, which overflows
  at the operating point (the baseline scores 0.99992, logit ≈ 9.45);
* a **self-check** asserts the padded baseline still scores the frozen constant
  ``0.9999216794967651`` at ``T = 1`` through both the operational
  ``evaluate()`` and this module's logit path -- proving the measurement layer
  left the constant untouched.

Discrimination/calibration metrics reuse Mercury's audited primitives
(:func:`~omni_mercury_engine.core.calibration.compute_ece` /
:func:`~omni_mercury_engine.core.calibration.compute_mce`,
:func:`~omni_mercury_engine.evaluation.metrics.compute_auc_roc`) and the clinical
metric engine (:mod:`omni_mercury_engine.medical.clinical_metrics`), so the same,
tested measurement code serves both the medical scores and the σ gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.calibration import compute_ece, compute_mce
from omni_mercury_engine.evaluation.metrics import compute_auc_roc
from omni_mercury_engine.medical.clinical_metrics import (
    ReliabilityBin,
    confusion_at_threshold,
    reliability_curve,
    youden_threshold,
)

if TYPE_CHECKING:
    from omni_mercury_engine.core.global_omni_scalar_network import EthicalGate
    from omni_mercury_engine.security.sigma_immutable_corpus import Baseline

__all__ = [
    "SIGMA_FROZEN_CONSTANT",
    "SigmaCalibrationPoint",
    "baseline_constant_check",
    "build_report",
    "gate_logits",
    "load_frozen_gate",
    "measure_at",
    "temperature_scale",
    "temperature_sweep",
    "threshold_sweep",
]

#: The frozen operational σ score of the intact baseline (must never move).
SIGMA_FROZEN_CONSTANT = 0.9999216794967651

#: Operational (authoritative) gate threshold; swept values here are advisory.
OPERATIONAL_THRESHOLD = 0.93


def load_frozen_gate() -> EthicalGate:
    """Construct a fresh, trained ``EthicalGate`` from the frozen weights.

    Returns:
        A trained :class:`EthicalGate`.

    Raises:
        RuntimeError: If torch or the trained weights are unavailable, so the
            harness fails loudly rather than silently measuring the NumPy
            fallback heuristic (which is not the shipped gate).
    """
    from omni_mercury_engine.core.global_omni_scalar_network import EthicalGate

    gate = EthicalGate()
    if not getattr(gate, "_trained", False) or gate.gate_network is None:
        raise RuntimeError(
            "σ_Immutable calibration requires the trained EthicalGate "
            "(torch + security/sigma_immutable_weights.pt). Neither the "
            "measurement nor the constant self-check is meaningful without it."
        )
    return gate


def gate_logits(gate: EthicalGate, samples: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return the pre-sigmoid logits the gate assigns to ``samples``.

    Pads/truncates each row to the gate's ``input_dim`` exactly as
    ``evaluate()`` does, then runs ``gate_network[:3]`` (Linear→ReLU→Linear),
    stopping before the baked-in ``Sigmoid``.

    Args:
        gate: A trained :class:`EthicalGate`.
        samples: ``(n, d)`` scalar vectors.

    Returns:
        ``(n,)`` float logits.
    """
    import torch

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, d = x.shape
    dim = gate.input_dim
    padded = np.zeros((n, dim), dtype=np.float32)
    width = min(d, dim)
    padded[:, :width] = x[:, :width]
    with torch.no_grad():
        tensor = torch.tensor(padded, dtype=torch.float32)
        logits = gate.gate_network[:3](tensor).reshape(-1)
        out: np.ndarray[Any, Any] = logits.detach().cpu().numpy().astype(np.float64)
    return out


def temperature_scale(logits: np.ndarray[Any, Any], temperature: float) -> np.ndarray[Any, Any]:
    """Return ``sigmoid(logits / temperature)`` (numerically stable)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    z = np.clip(np.asarray(logits, dtype=float) / temperature, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SigmaCalibrationPoint:
    """Discrimination + calibration of the σ gate at one (temperature, threshold).

    Attributes:
        temperature: Temperature applied to the logits (1.0 = operational).
        threshold: Decision threshold applied to the (scaled) score.
        n: Number of evaluated samples.
        n_positive: Number of intact (label 1) samples.
        auroc: Discrimination of the score for intact vs tampered.
        ece: Expected calibration error of the (scaled) probability.
        mce: Maximum calibration error.
        brier: Brier score of the (scaled) probability.
        sensitivity: P(pass | intact) at ``threshold``.
        specificity: P(fail | tampered) at ``threshold``.
        balanced_accuracy: Mean of sensitivity and specificity.
        accuracy: Overall classification accuracy at ``threshold``.
        reliability: Non-empty reliability-curve bins.
    """

    temperature: float
    threshold: float
    n: int
    n_positive: int
    auroc: float
    ece: float
    mce: float
    brier: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    accuracy: float
    reliability: list[ReliabilityBin] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping of the measurement."""
        return {
            "temperature": self.temperature,
            "threshold": self.threshold,
            "n": self.n,
            "n_positive": self.n_positive,
            "auroc": self.auroc,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy": self.accuracy,
            "reliability": [b.to_dict() for b in self.reliability],
        }


def measure_at(
    logits: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    temperature: float,
    threshold: float,
    n_bins: int = 10,
) -> SigmaCalibrationPoint:
    """Measure discrimination + calibration at a (temperature, threshold).

    Args:
        logits: Pre-sigmoid gate logits.
        labels: Binary labels (1 = intact, 0 = tampered).
        temperature: Temperature applied to the logits.
        threshold: Decision threshold on the scaled probability.
        n_bins: Bin count for ECE/MCE and the reliability curve.

    Returns:
        A populated :class:`SigmaCalibrationPoint`.
    """
    y = np.asarray(labels, dtype=int).ravel()
    p = temperature_scale(logits, temperature)
    tp, fp, tn, fn = confusion_at_threshold(y, p, threshold)
    sens = float(tp / (tp + fn)) if (tp + fn) else 1.0
    spec = float(tn / (tn + fp)) if (tn + fp) else 1.0
    acc = float((tp + tn) / len(y)) if len(y) else 0.0
    return SigmaCalibrationPoint(
        temperature=float(temperature),
        threshold=float(threshold),
        n=len(y),
        n_positive=int(np.sum(y == 1)),
        auroc=float(compute_auc_roc(y, p)),
        ece=float(compute_ece(y, p, n_bins)),
        mce=float(compute_mce(y, p, n_bins)),
        brier=float(np.mean((p - y) ** 2)),
        sensitivity=sens,
        specificity=spec,
        balanced_accuracy=float((sens + spec) / 2.0),
        accuracy=acc,
        reliability=reliability_curve(y, p, n_bins),
    )


def temperature_sweep(
    logits: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    temperatures: list[float],
    threshold: float = OPERATIONAL_THRESHOLD,
    n_bins: int = 10,
) -> tuple[list[SigmaCalibrationPoint], float]:
    """Sweep temperature at a fixed threshold; return points + the min-ECE T.

    Args:
        logits: Gate logits.
        labels: Binary labels.
        temperatures: Temperature grid to evaluate.
        threshold: Fixed decision threshold during the temperature sweep.
        n_bins: Bin count for calibration metrics.

    Returns:
        ``(points, best_temperature)`` where ``best_temperature`` minimises ECE.
    """
    points = [
        measure_at(logits, labels, temperature=t, threshold=threshold, n_bins=n_bins)
        for t in temperatures
    ]
    best = min(points, key=lambda pt: pt.ece)
    return points, best.temperature


def threshold_sweep(
    logits: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    thresholds: list[float],
    temperature: float = 1.0,
    n_bins: int = 10,
) -> tuple[list[SigmaCalibrationPoint], float]:
    """Sweep threshold at a fixed temperature; return points + the Youden-J threshold.

    Args:
        logits: Gate logits.
        labels: Binary labels.
        thresholds: Threshold grid to evaluate.
        temperature: Fixed temperature during the threshold sweep.
        n_bins: Bin count for calibration metrics.

    Returns:
        ``(points, best_threshold)`` where ``best_threshold`` maximises Youden's J
        (sensitivity + specificity - 1) on the scaled score.
    """
    points = [
        measure_at(logits, labels, temperature=temperature, threshold=thr, n_bins=n_bins)
        for thr in thresholds
    ]
    p = temperature_scale(logits, temperature)
    best_thr = youden_threshold(np.asarray(labels, dtype=int).ravel(), p)
    return points, float(best_thr)


def baseline_constant_check(gate: EthicalGate, baseline: Baseline) -> dict[str, Any]:
    """Prove the measurement layer left the frozen operational constant intact.

    Scores the padded baseline both through the operational ``evaluate()`` and
    through this module's ``T = 1`` logit path, and checks both equal the frozen
    constant ``0.9999216794967651``.

    Args:
        gate: The trained :class:`EthicalGate`.
        baseline: The harvested intact baseline.

    Returns:
        Mapping with the two scores, their agreement, and the invariant verdict.
    """
    padded = np.zeros(gate.input_dim, dtype=np.float64)
    vals = np.asarray(baseline.values, dtype=np.float64)
    width = min(len(vals), gate.input_dim)
    padded[:width] = vals[:width]

    _passes, operational_score = gate.evaluate(padded)
    logit = gate_logits(gate, padded.reshape(1, -1))
    t1_score = float(temperature_scale(logit, 1.0)[0])

    # The safety-critical invariant is exact: the operational path must still
    # produce the frozen constant (tolerance is float-round-off only). The
    # logit-path agreement is a *measurement-reproduction* check at float
    # precision -- the sweep's sigmoid runs in numpy float64 while the gate's
    # baked-in Sigmoid runs in torch float32, so the two differ at ~1e-8, which
    # is irrelevant to ECE and does not touch the operational score.
    path_diff = abs(t1_score - operational_score)
    op_ok = abs(operational_score - SIGMA_FROZEN_CONSTANT) <= 1e-9
    path_ok = path_diff <= 1e-6
    return {
        "frozen_constant": SIGMA_FROZEN_CONSTANT,
        "operational_score": float(operational_score),
        "logit_path_score_t1": t1_score,
        "logit_path_abs_diff": float(path_diff),
        "operational_matches_constant": bool(op_ok),
        "logit_path_reproduces_operational": bool(path_ok),
        "invariant_holds": bool(op_ok and path_ok),
    }


def build_report(
    *,
    seed: int = 999,
    n_positive: int = 400,
    n_negative: int = 400,
    temperatures: list[float] | None = None,
    n_threshold_grid: int = 60,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Build the full σ_Immutable calibration + sweep report.

    Uses a held-out corpus draw (default seed 999, distinct from the training
    seed 42) so ECE is not measured on training points.

    Args:
        seed: RNG seed for the held-out integrity-sample draw.
        n_positive: Intact samples to draw.
        n_negative: Tampered samples to draw.
        temperatures: Temperature grid (defaults to a 0.5–5.0 sweep).
        n_threshold_grid: Number of thresholds in the [0.5, 0.999] grid.
        n_bins: Bin count for calibration metrics.

    Returns:
        A JSON-friendly report mapping.
    """
    from omni_mercury_engine.security.sigma_immutable_corpus import (
        build_integrity_samples,
        load_baseline,
    )

    if temperatures is None:
        temperatures = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    gate = load_frozen_gate()
    baseline = load_baseline()
    invariant = baseline_constant_check(gate, baseline)

    x, y = build_integrity_samples(
        baseline, seed=seed, n_positive=n_positive, n_negative=n_negative
    )
    logits = gate_logits(gate, x)

    operational = measure_at(
        logits, y, temperature=1.0, threshold=OPERATIONAL_THRESHOLD, n_bins=n_bins
    )
    temp_points, best_t = temperature_sweep(
        logits, y, temperatures=temperatures, threshold=OPERATIONAL_THRESHOLD, n_bins=n_bins
    )
    thr_grid = list(np.linspace(0.5, 0.999, n_threshold_grid))
    thr_points, best_thr = threshold_sweep(
        logits, y, thresholds=thr_grid, temperature=1.0, n_bins=n_bins
    )
    # Combined recommendation: min-ECE temperature, then Youden-J threshold on it.
    _pts_at_best_t, best_thr_at_best_t = threshold_sweep(
        logits, y, thresholds=thr_grid, temperature=best_t, n_bins=n_bins
    )
    recommended = measure_at(
        logits, y, temperature=best_t, threshold=best_thr_at_best_t, n_bins=n_bins
    )

    return {
        "schema_version": "1.0",
        "seed": seed,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "operational_threshold": OPERATIONAL_THRESHOLD,
        "frozen_constant_invariant": invariant,
        "operational_point": operational.to_dict(),
        "temperature_sweep": {
            "best_temperature_by_ece": best_t,
            "points": [p.to_dict() for p in temp_points],
        },
        "threshold_sweep_t1": {
            "best_threshold_by_youden_j": best_thr,
            "points": [p.to_dict() for p in thr_points],
        },
        "recommended_advisory": {
            "temperature": best_t,
            "threshold": best_thr_at_best_t,
            "metrics": recommended.to_dict(),
            "note": (
                "ADVISORY MEASUREMENT ONLY. The operational threshold stays 0.93 "
                "and the operational temperature stays 1.0; this records what a "
                "calibration-optimal operating point would be, it does not repoint "
                "any operational constant."
            ),
        },
    }

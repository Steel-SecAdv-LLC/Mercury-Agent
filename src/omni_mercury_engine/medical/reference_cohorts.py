# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Seeded reference cohorts for clinical-score measurement (documented DGPs).

Measuring AUROC / sensitivity / calibration for a clinical score needs *labelled*
data: ``(score, outcome)`` pairs. Real clinical validation requires governed
datasets (MIMIC-III, institutional EHR) that cannot ship in a repository, so the
harness ships two **reproducible synthetic cohorts** with fully documented
data-generating processes (DGPs). They exist to:

1. validate the measurement + calibration wiring end-to-end, deterministically;
2. characterise a score's *internal* discrimination/calibration under a stated
   generative model.

They do **not** establish real-world clinical accuracy. The DGP for the
Framingham cohort is deliberately *not* the Framingham point model itself
(that would be circular): outcomes come from an independent logistic model whose
coefficients only share the literature-established *direction* of each risk
factor, plus Gaussian noise and an age x blood-pressure interaction the coarse
point bins do not capture -- so the instrument's AUROC lands realistically below
1.0. Every cohort is seeded, so the same seed yields identical arrays.

The cohorts are the honest stand-in for real data; the harness itself
(:mod:`omni_mercury_engine.medical.clinical_metrics` /
:mod:`~omni_mercury_engine.medical.clinical_calibration`) is dataset-agnostic --
point it at real ``(score, label)`` arrays and it produces the same metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ReferenceCohort",
    "framingham_cvd_cohort",
    "split_cohort",
    "synthetic_calibrated_cohort",
]


@dataclass
class ReferenceCohort:
    """A labelled reference cohort: aligned score + outcome arrays.

    Attributes:
        name: Short cohort identifier.
        description: One-line human description.
        dgp_doc: The exact data-generating process (for audit/reproducibility).
        scores: ``(n,)`` clinical scores / probabilities.
        labels: ``(n,)`` binary outcomes aligned with ``scores``.
        seed: The seed that produced the cohort.
        meta: Optional extra provenance (coefficients, prevalence, etc.).
    """

    name: str
    description: str
    dgp_doc: str
    scores: np.ndarray[Any, Any]
    labels: np.ndarray[Any, Any]
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate score/label alignment and label binarity."""
        self.scores = np.asarray(self.scores, dtype=float).ravel()
        self.labels = np.asarray(self.labels, dtype=float).ravel()
        if self.scores.shape != self.labels.shape:
            raise ValueError("scores and labels must have equal length")
        uniq = set(np.unique(self.labels).tolist())
        if not uniq <= {0.0, 1.0}:
            raise ValueError(f"labels must be binary, got {sorted(uniq)}")


def _sigmoid(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Numerically-stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def synthetic_calibrated_cohort(
    n: int = 2000, *, seed: int = 0, miscalibration: float = 1.0
) -> ReferenceCohort:
    """Cohort with a known score-to-outcome relationship (harness validation).

    A latent probability ``q`` is drawn ``~ Beta(2, 2)`` and the outcome is
    ``Bernoulli(q)``, so ``q`` is a *perfectly calibrated* score. The reported
    score is ``sigmoid(miscalibration * logit(q))``: ``miscalibration == 1``
    reproduces ``q`` (well calibrated), ``> 1`` makes it over-confident and
    ``< 1`` under-confident. This lets a test assert that ECE detects the
    injected miscalibration and that a calibrator removes it.

    Args:
        n: Number of cases.
        seed: RNG seed.
        miscalibration: Logit-scale distortion applied to the true probability.

    Returns:
        A :class:`ReferenceCohort`.
    """
    rng = np.random.RandomState(seed)
    q = rng.beta(2.0, 2.0, size=n)
    labels = (rng.uniform(size=n) < q).astype(float)
    logit = np.log(q / (1.0 - q))
    scores = _sigmoid(miscalibration * logit)
    return ReferenceCohort(
        name="synthetic_calibrated",
        description="Known-probability cohort for harness self-validation",
        dgp_doc=(
            "q ~ Beta(2,2); outcome ~ Bernoulli(q); "
            f"score = sigmoid({miscalibration} * logit(q))"
        ),
        scores=scores,
        labels=labels,
        seed=seed,
        meta={"miscalibration": miscalibration, "prevalence": float(np.mean(labels))},
    )


def framingham_cvd_cohort(n: int = 1500, *, seed: int = 0) -> ReferenceCohort:
    """Framingham 10-year CVD-risk cohort scored by the real instrument.

    Risk factors are sampled from documented adult ranges; the Framingham
    10-year CVD risk fraction is computed by the *real*
    :class:`~omni_mercury_engine.medical.cardiology.cardiology_predictor.FraminghamRiskCalculator`.
    The ground-truth 10-year CVD event is drawn from an **independent** logistic
    DGP (see ``dgp_doc``) sharing only each factor's literature direction, so the
    instrument's discrimination is measured, not tautologically recovered.

    Args:
        n: Number of synthetic patients.
        seed: RNG seed.

    Returns:
        A :class:`ReferenceCohort` whose ``scores`` are Framingham 10-year CVD
        risk fractions in ``[0, 1]`` and whose ``labels`` are synthetic events.

    Raises:
        ImportError: If the cardiology module (and its backend) cannot import.
    """
    from omni_mercury_engine.medical.cardiology.cardiology_predictor import (
        FraminghamRiskCalculator,
    )

    rng = np.random.RandomState(seed)
    calc = FraminghamRiskCalculator()

    age = rng.randint(30, 80, size=n)
    is_male = rng.uniform(size=n) < 0.5
    total_chol = np.clip(rng.normal(200, 35, size=n), 120, 340)
    hdl_chol = np.clip(rng.normal(52, 14, size=n), 20, 100)
    sbp = np.clip(rng.normal(128, 18, size=n), 90, 200)
    smoker = rng.uniform(size=n) < 0.22
    diabetes = rng.uniform(size=n) < 0.12

    # Independent outcome DGP: standardised risk factors, literature-direction
    # coefficients, an age x SBP interaction the point bins miss, plus noise.
    z_age = (age - 55.0) / 10.0
    z_sbp = (sbp - 130.0) / 20.0
    z_tc = (total_chol - 200.0) / 40.0
    z_hdl = (hdl_chol - 50.0) / 15.0
    logit = (
        -2.3
        + 0.62 * z_age
        + 0.45 * z_sbp
        + 0.30 * z_tc
        - 0.42 * z_hdl
        + 0.55 * smoker.astype(float)
        + 0.60 * diabetes.astype(float)
        + 0.35 * is_male.astype(float)
        + 0.18 * z_age * z_sbp
        + rng.normal(0.0, 0.55, size=n)
    )
    p_event = _sigmoid(logit)
    labels = (rng.uniform(size=n) < p_event).astype(float)

    scores = np.empty(n, dtype=float)
    for i in range(n):
        demographics = {
            "age": int(age[i]),
            "gender": "male" if is_male[i] else "female",
            "total_cholesterol_mg_dl": float(total_chol[i]),
            "hdl_cholesterol_mg_dl": float(hdl_chol[i]),
            "systolic_bp_mmhg": float(sbp[i]),
            "smoker": bool(smoker[i]),
            "diabetes": bool(diabetes[i]),
        }
        result = calc.calculate_risk(demographics)
        scores[i] = float(result["10_year_cvd_risk_percent"]) / 100.0

    return ReferenceCohort(
        name="framingham_cvd",
        description="Framingham 10-year CVD risk vs an independent logistic outcome DGP",
        dgp_doc=(
            "factors ~ documented adult ranges; "
            "outcome ~ Bernoulli(sigmoid(-2.3 + 0.62*z_age + 0.45*z_sbp + 0.30*z_tc "
            "- 0.42*z_hdl + 0.55*smoker + 0.60*diabetes + 0.35*male "
            "+ 0.18*z_age*z_sbp + N(0,0.55))); "
            "score = FraminghamRiskCalculator 10-year CVD risk fraction"
        ),
        scores=scores,
        labels=labels,
        seed=seed,
        meta={
            "prevalence": float(np.mean(labels)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        },
    )


def split_cohort(
    cohort: ReferenceCohort, *, calibration_fraction: float = 0.5, seed: int = 0
) -> tuple[ReferenceCohort, ReferenceCohort]:
    """Deterministically split a cohort into calibration and test halves.

    Args:
        cohort: The cohort to split.
        calibration_fraction: Fraction assigned to the calibration split.
        seed: Seed for the permutation (independent of the cohort's own seed).

    Returns:
        ``(calibration_cohort, test_cohort)`` -- disjoint, same DGP provenance.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    n = len(cohort.scores)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_cal = round(n * calibration_fraction)
    cal_idx = np.sort(perm[:n_cal])
    test_idx = np.sort(perm[n_cal:])

    def _subset(name_suffix: str, idx: np.ndarray[Any, Any]) -> ReferenceCohort:
        return ReferenceCohort(
            name=f"{cohort.name}_{name_suffix}",
            description=cohort.description,
            dgp_doc=cohort.dgp_doc,
            scores=cohort.scores[idx],
            labels=cohort.labels[idx],
            seed=cohort.seed,
            meta={**cohort.meta, "split": name_suffix, "n": len(idx)},
        )

    return _subset("cal", cal_idx), _subset("test", test_idx)

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit energy-based-model (EBM) anomaly detector.

An energy-based model defines an unnormalised density ``p(x) ∝ exp(-E(x))`` via
an *energy* function that is low on the data manifold and high off it (LeCun et
al., *A Tutorial on Energy-Based Learning*, 2006). Anomalies are exactly the
high-energy points. The practical obstacle — the intractable partition function
— is sidestepped by **score matching** (Hyvärinen, 2005), which fits the energy
from the data score (gradient of log-density) without ever normalising.

This detector delay-embeds the signal and fits an explicit quadratic
(Gaussian-family) energy ``E(x) = 0.5 (x-μ)ᵀ Λ (x-μ)`` whose score-matching
solution is the embedding-space precision ``Λ = Σ⁻¹``. Each point is scored by
its calibrated free energy, squashed into ``[0, 1]``. Operating in the raw
delay-embedding space (rather than a periodic random-feature lift) keeps the
energy monotone in amplitude, so off-manifold shifts and bursts are reliably
high-energy. Pure NumPy (always importable); registered as an opt-in BASE
detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite,
    finite_features,
    finite_scores,
    squash_scale,
)

if TYPE_CHECKING:
    import torch

__all__ = ["EnergyBasedDetector"]

_LN2 = float(np.log(2.0))


class EnergyBasedDetector(BaseDetector):
    """Gaussian-family quadratic energy-based-model detector.

    A causal delay embedding of dimension ``embed_dim`` is standardised with the
    training statistics; :meth:`fit` estimates the embedding-space mean and
    precision (the score-matching solution for a quadratic energy). :meth:`detect`
    scores each point by its free energy ``0.5 (x-μ)ᵀ Λ (x-μ)`` squashed into
    ``[0, 1]``.
    """

    def __init__(
        self,
        embed_dim: int = 8,
        ridge: float = 1e-3,
        shrinkage: float = 0.1,
        calibration_quantile: float = 0.98,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the energy-based detector.

        Args:
            embed_dim: Delay-embedding dimension (window length). Must be >= 1.
            ridge: Tikhonov regularisation added before inverting the embedding
                covariance (keeps the precision well conditioned). Must be > 0.
            shrinkage: Convex shrinkage toward the diagonal covariance in
                ``[0, 1)`` for a less overfit precision.
            calibration_quantile: Training free-energy quantile at the 0.5
                boundary; ``1 - calibration_quantile`` is the normal-regime FPR.
                Must be in ``(0, 1)``.
            seed: RNG seed (reserved for API symmetry; the estimator is
                deterministic).
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError(f"shrinkage must be in [0, 1), got {shrinkage}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.embed_dim = int(embed_dim)
        self.ridge = float(ridge)
        self.shrinkage = float(shrinkage)
        self.calibration_quantile = float(calibration_quantile)
        self.seed = int(seed)
        self._mu: np.ndarray[Any, Any] | None = None
        self._precision: np.ndarray[Any, Any] | None = None
        self._data_mean: float = 0.0
        self._data_std: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has fitted the energy/scale."""
        return self._is_fitted

    @staticmethod
    def _to_1d_f64(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return bound_finite(np.asarray(data, dtype=np.float64)).ravel()

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _embed(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Causal, standardised delay embedding: row ``t`` = ``[y_{t-m+1}..y_t]``."""
        m = self.embed_dim
        n = series.size
        emb = np.zeros((n, m), dtype=np.float64)
        if n == 0:
            return emb
        u = (series - self._data_mean) / self._data_std
        for i in range(m):
            # Fill the lag-``i`` column only where a real lagged sample exists
            # (rows ``i..n-1``). When ``i >= n`` (a series shorter than the embed
            # dimension) that target slice is empty while ``u[:n-i]`` would be a
            # non-empty negative slice, so guard the assignment; the warm-up
            # rows are then edge-padded with the first sample below.
            if i < n:
                emb[i:, m - 1 - i] = u[: n - i]
            emb[:i, m - 1 - i] = u[0]  # edge-pad the warm-up region
        return emb

    def _energy(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Free energy of each embedded point under the quadratic EBM."""
        if self._mu is None or self._precision is None:
            return np.zeros(series.size, dtype=np.float64)
        d = self._embed(series) - self._mu
        # 0.5 * d^T Λ d, computed row-wise.
        return 0.5 * np.einsum("ij,jk,ik->i", d, self._precision, d)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> EnergyBasedDetector:
        """Fit the embedding-space mean/precision and the calibration scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        self._data_mean = float(np.mean(series)) if series.size else 0.0
        self._data_std = float(np.std(series)) + 1e-9

        emb = self._embed(series)
        if emb.shape[0] == 0:
            # No samples to fit on: leave the detector inert (``_energy`` returns
            # zeros when ``_mu``/``_precision`` are ``None``) rather than deriving
            # a NaN mean/covariance from an empty array.
            self._mu = None
            self._precision = None
            self._scale = 1.0
            self._is_fitted = True
            return self
        self._mu = np.mean(emb, axis=0)
        # Held-out calibration: fit the precision on the first 70% of embedded
        # points and calibrate the squash scale on the remaining 30% so the
        # score threshold is not optimistically biased by in-sample energies.
        n = emb.shape[0]
        split = max(1, int(0.7 * n)) if n >= 4 else n
        cov = np.atleast_2d(np.cov(emb[:split], rowvar=False))
        # A single calibration row makes ``np.cov`` (ddof=1) divide by zero and
        # return NaN; fall back to a zero covariance so the ridge term below
        # yields a finite isotropic precision instead of a NaN one.
        cov = np.nan_to_num(cov)
        # Shrink toward the diagonal for a well-conditioned, less overfit precision.
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
        # Score-matching solution for a Gaussian-family energy: precision = Σ⁻¹.
        self._precision = np.linalg.inv(cov + self.ridge * np.eye(self.embed_dim))

        cal = self._energy(series)
        raw = cal[split:] if split < n else cal
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the free energy.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 energies.
        """
        raw = self._energy(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from EBM free energy.

        Energies are squashed monotonically via ``1 - exp(-E / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._energy(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale)).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }

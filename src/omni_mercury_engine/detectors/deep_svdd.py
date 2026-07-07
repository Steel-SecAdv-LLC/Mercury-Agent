# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""One-class SVDD hypersphere anomaly detector on a random-feature embedding.

Support Vector Data Description (Tax & Duin, *Support Vector Data Description*,
Machine Learning 2004) — the one-class principle behind Deep SVDD (Ruff et al.,
ICML 2018) — learns the smallest hypersphere enclosing the normal data in a
feature space; points far outside the sphere are anomalies. Deep SVDD replaces
the kernel with a trained deep embedding, but the detection principle is the
distance from an embedding to the sphere centre.

This detector realises that principle without a trainable deep network (and
therefore without PyTorch): a delay embedding is lifted through a fixed, seeded
random ``tanh`` projection (a saturating nonlinear embedding), the sphere centre
is the feature-space mean of the training data, and the anomaly signal is the
distance from that centre. Each point's distance is squashed into ``[0, 1]``.
Pure NumPy (always importable); registered as an opt-in BASE detector.
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

__all__ = ["DeepSVDDDetector"]

_LN2 = float(np.log(2.0))


class DeepSVDDDetector(BaseDetector):
    """One-class SVDD hypersphere detector on a random-feature embedding.

    A delay embedding is mapped through a fixed random ``tanh`` projection;
    :meth:`fit` sets the hypersphere centre to the feature-space mean of the
    training data and calibrates the radius. :meth:`detect` scores each point by
    its distance to the centre, squashed into ``[0, 1]``.
    """

    def __init__(
        self,
        embed_dim: int = 8,
        n_features: int = 32,
        bandwidth: float = 1.0,
        ridge: float = 1e-3,
        shrinkage: float = 0.1,
        calibration_quantile: float = 0.98,
        seed: int = 0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SVDD detector.

        Args:
            embed_dim: Delay-embedding dimension (window length). Must be >= 1.
            n_features: Random projection dimension. Must be >= 2.
            bandwidth: Inverse projection scale (larger ``bandwidth`` ⇒ gentler
                ``tanh`` saturation). Must be > 0.
            ridge: Tikhonov regularisation added before inverting the feature
                covariance (keeps the metric well conditioned). Must be > 0.
            shrinkage: Convex shrinkage toward the diagonal covariance in
                ``[0, 1)`` for a less overfit metric.
            calibration_quantile: Training-distance quantile at the 0.5 boundary;
                ``1 - calibration_quantile`` is the normal-regime FPR. In ``(0, 1)``.
            seed: RNG seed for the fixed random features (determinism).
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if n_features < 2:
            raise ValueError(f"n_features must be >= 2, got {n_features}")
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be > 0")
        if ridge <= 0.0:
            raise ValueError(f"ridge must be > 0, got {ridge}")
        if not 0.0 <= shrinkage < 1.0:
            raise ValueError(f"shrinkage must be in [0, 1), got {shrinkage}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.embed_dim = int(embed_dim)
        self.n_features = int(n_features)
        self.bandwidth = float(bandwidth)
        self.ridge = float(ridge)
        self.shrinkage = float(shrinkage)
        self.calibration_quantile = float(calibration_quantile)
        self.seed = int(seed)
        self._omega: np.ndarray[Any, Any] | None = None
        self._phase: np.ndarray[Any, Any] | None = None
        self._center: np.ndarray[Any, Any] | None = None
        self._precision: np.ndarray[Any, Any] | None = None
        self._data_mean: float = 0.0
        self._data_std: float = 1.0
        self._radius: float = 0.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has set the centre/radius/scale."""
        return self._is_fitted

    def _to_1d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        return bound_finite(np.asarray(data, dtype=np.float64), detector=self.name).ravel()

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _embed(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Causal delay embedding: row ``t`` = ``[y_{t-m+1}, ..., y_t]``."""
        m = self.embed_dim
        n = series.size
        emb = np.zeros((n, m), dtype=np.float64)
        if n == 0:
            return emb
        u = (series - self._data_mean) / self._data_std
        for i in range(m):
            # Guard the lag-``i`` fill for series shorter than the embed
            # dimension (``i >= n`` leaves an empty target slice); warm-up rows
            # are edge-padded with the first sample below.
            if i < n:
                emb[i:, m - 1 - i] = u[: n - i]
            emb[:i, m - 1 - i] = u[0]
        return emb

    def _features(self, emb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Bounded random ``tanh`` projection (a fixed nonlinear embedding).

        Unlike a periodic random-Fourier map, ``tanh`` is monotone and
        saturating, so large-magnitude or shifted inputs map to a *stable* far
        region of the embedding rather than wrapping back toward the training
        cluster — the property Deep SVDD needs to keep off-manifold points far
        from the sphere centre.
        """
        assert self._omega is not None and self._phase is not None
        proj = emb @ self._omega + self._phase
        return np.sqrt(1.0 / self.n_features) * np.tanh(proj)

    def _distances(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Mahalanobis distance of each embedded point from the sphere centre.

        Using the feature-covariance metric (an ellipsoidal SVDD boundary)
        rather than a raw Euclidean radius counteracts the concentration of
        measure that makes high-dimensional Euclidean distances nearly constant,
        restoring a usable margin between normal and off-manifold points.
        """
        if self._center is None or self._precision is None:
            return np.zeros(series.size, dtype=np.float64)
        d = self._features(self._embed(series)) - self._center
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self._precision, d), 0.0))

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DeepSVDDDetector:
        """Fit the random features, hypersphere centre/radius, and squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        self._data_mean = float(np.mean(series)) if series.size else 0.0
        self._data_std = float(np.std(series)) + 1e-9
        rng = np.random.default_rng(self.seed)
        self._omega = rng.normal(0.0, 1.0 / self.bandwidth, (self.embed_dim, self.n_features))
        self._phase = rng.normal(0.0, 1.0, self.n_features)

        emb = self._embed(series)
        if emb.shape[0] == 0:
            # No samples to fit on: leave the detector inert (``_distances``
            # returns zeros when ``_center``/``_precision`` are ``None``).
            self._center = None
            self._precision = None
            self._radius = 0.0
            self._scale = 1.0
            self._is_fitted = True
            return self
        phi = self._features(emb)
        self._center = np.mean(phi, axis=0)
        # Fit the ellipsoidal metric on the first 70% of points and calibrate the
        # radius/scale on the held-out 30% so the threshold is unbiased.
        n = phi.shape[0]
        split = max(1, int(0.7 * n)) if n >= 4 else n
        cov = np.atleast_2d(np.cov(phi[:split], rowvar=False))
        # A single calibration row makes ``np.cov`` (ddof=1) return NaN; fall back
        # to a zero covariance so the ridge term yields a finite precision.
        cov = np.nan_to_num(cov)
        diag = np.diag(np.diag(cov))
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * diag
        self._precision = np.linalg.inv(cov + self.ridge * np.eye(self.n_features))
        raw = self._distances(series)
        cal = raw[split:] if split < n else raw
        self._radius = float(np.quantile(cal, self.calibration_quantile)) if cal.size else 0.0
        self._scale = self._squash_scale(cal)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: distance to the hypersphere centre.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 distances.
        """
        raw = self._distances(self._to_1d_f64(data))
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from centre distance.

        Distances are squashed monotonically via ``1 - exp(-d / scale)`` using
        the training scale from :meth:`fit`.
        """
        raw = self._distances(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale)).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {"radius": self._radius},
        }

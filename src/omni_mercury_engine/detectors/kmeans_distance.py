# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""K-means-distance fusion detector reviving the dormant cognitive clusterer.

Distance to the nearest learned cluster centroid is a classic unsupervised
anomaly signal: a point far from *every* centroid is poorly explained by the
learned structure and is therefore anomalous. The clustering machinery is the
``KMeansClusterer`` from :mod:`omni_mercury_engine.cognitive.neural_memory_layer`,
where it was orphaned (the module is never used in any live path).

``benchmarks/dormant_module_revival.py`` measures this signal on genuinely
labelled ADBench data and finds it real (mean held-out ROC-AUC ~0.8-0.98 across
breastw/thyroid/WBC/cardio), in contrast to the other tabular-scoring orphans
(predictive-coding free-energy, case-based retrieval) which score at chance.
That measurement -- not the module's interface -- is what justifies promoting
the clusterer to a first-class detector here. Whether it *adds* to the live
fusion ensemble (which already carries a distance/density detector) is settled
separately by a fusion-marginal ablation; this class only makes the proven
signal available as a standard detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector

if TYPE_CHECKING:
    import torch

__all__ = ["KMeansDistanceDetector"]


class KMeansDistanceDetector(BaseDetector):
    """Unsupervised detector emitting per-centroid distances as fusion features.

    Implements the :class:`~omni_mercury_engine.core.base.BaseDetector`
    contract consumed by :meth:`OmniMercuryEngine._extract_fusion_features` --
    ``fit``, ``is_fitted`` and ``extract_features`` returning a per-sample
    ``(n_samples, n_clusters + 1)`` feature block (distance to every centroid
    plus the nearest-centroid distance). Features are standardised internally so
    Euclidean distance is scale-invariant, matching how the standalone benchmark
    measured the signal. Registered in ``DETECTOR_MANIFEST`` (BASE,
    feature_dim ``n_clusters + 1``) so it is reachable through the engine via
    ``enable_detector("kmeans_distance")``; it is opt-in, not a default base
    detector, so the calibrated fusion ensemble is unchanged until enabled.
    """

    def __init__(self, n_clusters: int = 8, config: dict[str, Any] | None = None) -> None:
        """Initialize the detector with the target cluster count.

        Args:
            n_clusters: Number of k-means centroids to learn (>= 1).
            config: Optional ``BaseDetector`` config (``threshold``,
                ``auto_calibrate``, ...).

        Raises:
            ValueError: If ``n_clusters`` is less than 1.
        """
        super().__init__(config)
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters = int(n_clusters)
        self._clusterer: Any = None
        self._mean: np.ndarray[Any, Any] | None = None
        self._std: np.ndarray[Any, Any] | None = None
        # Robust scale (median training nearest-distance) used to squash the
        # nearest-centroid distance into a [0, 1] score for ``detect``.
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned the cluster centroids."""
        return self._clusterer is not None

    @staticmethod
    def _to_2d_f32(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy or torch input to a 2-D float32 array (1-D -> one sample).

        Accepts ``torch.Tensor`` (the BaseDetector contract) without a hard torch
        import: a tensor is detached to CPU numpy via duck-typing.
        """
        detach = getattr(data, "detach", None)  # torch.Tensor, without importing torch
        if callable(detach):
            data = detach().cpu().numpy()
        return np.atleast_2d(np.nan_to_num(np.asarray(data, dtype=np.float32)))

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> KMeansDistanceDetector:
        """Fit the k-means clusterer on standardised features.

        Args:
            data: Training features ``(n_samples, n_features)``; a 1-D array is
                treated as a single sample (``(1, n_features)``).

        Returns:
            ``self``, with centroids and the ``[0, 1]`` score scale learned.
        """
        from omni_mercury_engine.cognitive.neural_memory_layer import KMeansClusterer

        arr = self._to_2d_f32(data)
        self._mean = arr.mean(axis=0)
        self._std = arr.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        k = max(1, min(self.n_clusters, len(arr)))
        self._clusterer = KMeansClusterer(n_clusters=k).fit((arr - self._mean) / self._std)
        nearest = self._cluster_distances(arr).min(axis=1)
        self._scale = float(np.median(nearest)) + 1e-9
        self._is_fitted = True
        return self

    def _cluster_distances(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample distance to every centroid, ``(n_samples, n_clusters)``."""
        if self._clusterer is None or self._mean is None or self._std is None:
            raise RuntimeError("KMeansDistanceDetector must be fit before use")
        arr = self._to_2d_f32(data)
        scaled = (arr - self._mean) / self._std
        dists = np.asarray(self._clusterer.get_cluster_distances(scaled), dtype=np.float32)
        if dists.ndim != 2 or dists.shape[0] != len(arr):
            dists = dists.reshape(len(arr), -1)
        return dists

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion features: distance to each centroid plus the nearest.

        Args:
            data: Features ``(n_samples, n_features)``.

        Returns:
            ``(n_samples, n_clusters + 1)`` float32 feature block.
        """
        dists = self._cluster_distances(data)
        nearest = dists.min(axis=1, keepdims=True)
        return np.concatenate([dists, nearest], axis=1).astype(np.float32)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` (the live-inference contract).

        ``OmniMercuryEngine.detect``/``detect_with_fusion`` call every detector's
        ``detect`` and read ``result["scores"]``; without this the revived
        detector would be silently skipped at inference. The nearest-centroid
        distance is squashed monotonically into ``[0, 1]`` via
        ``1 - exp(-d / scale)`` (``scale`` = median training nearest-distance),
        so points near a centroid score ~0 and points far from every centroid
        approach 1.
        """
        nearest = self._cluster_distances(data).min(axis=1)
        scores = 1.0 - np.exp(-nearest / self._scale)
        scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
        return {
            # Scalar summary for the BaseDetector result contract; the per-sample
            # ``scores`` array remains the signal the fusion/consensus path reads.
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > 0.5,
            "confidence": scores,
        }

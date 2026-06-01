"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

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

from typing import Any

import numpy as np

__all__ = ["KMeansDistanceDetector"]


class KMeansDistanceDetector:
    """Unsupervised detector emitting per-centroid distances as fusion features.

    Implements the base-detector contract consumed by
    :meth:`OmniMercuryEngine._extract_fusion_features` -- ``fit``,
    ``is_fitted`` and ``extract_features`` returning a per-sample
    ``(n_samples, n_clusters + 1)`` feature block (distance to every centroid
    plus the nearest-centroid distance). Features are standardised internally so
    Euclidean distance is scale-invariant, matching how the standalone benchmark
    measured the signal.
    """

    def __init__(self, n_clusters: int = 8) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters = int(n_clusters)
        self._clusterer: Any = None
        self._mean: np.ndarray[Any, Any] | None = None
        self._std: np.ndarray[Any, Any] | None = None

    def is_fitted(self) -> bool:
        return self._clusterer is not None

    def fit(self, X: np.ndarray[Any, Any]) -> KMeansDistanceDetector:
        from omni_mercury_engine.cognitive.neural_memory_layer import KMeansClusterer

        arr = np.nan_to_num(np.asarray(X, dtype=float))
        if arr.ndim != 2:
            arr = arr.reshape(len(arr), -1)
        self._mean = arr.mean(axis=0)
        self._std = arr.std(axis=0)
        self._std[self._std < 1e-8] = 1.0
        k = max(1, min(self.n_clusters, len(arr)))
        self._clusterer = KMeansClusterer(n_clusters=k).fit((arr - self._mean) / self._std)
        return self

    def extract_features(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self._clusterer is None or self._mean is None or self._std is None:
            raise RuntimeError("KMeansDistanceDetector must be fit before extract_features")
        arr = np.nan_to_num(np.asarray(X, dtype=float))
        if arr.ndim != 2:
            arr = arr.reshape(len(arr), -1)
        scaled = (arr - self._mean) / self._std
        dists = np.asarray(self._clusterer.get_cluster_distances(scaled), dtype=np.float32)
        if dists.ndim != 2 or dists.shape[0] != len(arr):
            dists = dists.reshape(len(arr), -1)
        nearest = dists.min(axis=1, keepdims=True)
        return np.concatenate([dists, nearest], axis=1).astype(np.float32)

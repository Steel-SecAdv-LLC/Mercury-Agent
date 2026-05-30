"""
Federated node -- runs MercuryAnomalyDetector locally, exports fitted state.

Each node:
1. Receives local data (never leaves the node)
2. Fits MercuryAnomalyDetector locally
3. Reads the detector's internal fitted attributes directly
4. Applies differential privacy noise (optional but recommended)
5. Transmits ONLY the noised statistics to the aggregator

The raw data never leaves the node.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from omni_mercury_engine.federation.statistics import FittedStatistics

if TYPE_CHECKING:
    import numpy as np


class FederatedNode:
    """A federated Mercury node that trains locally and exports statistics.

    Usage:
        node = FederatedNode(node_id="hospital_A")
        node.fit(local_data)
        stats = node.export_statistics(epsilon=1.0)
        # Send stats to aggregator via any transport
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._detector: Any = None
        self._fitted = False
        self._n_samples = 0
        self._n_features = 0
        self._data_hash = ""

    def fit(self, X: np.ndarray[Any, Any]) -> None:
        """
        Fit MercuryAnomalyDetector on local data.

        Args:
            X: Local training data, shape (n_samples, n_features).
               This data NEVER leaves the node.
        """
        from omni_mercury_engine.detectors.statistical import (
            MercuryAnomalyDetector,
        )

        self._detector = MercuryAnomalyDetector()
        self._detector.fit(X)
        self._n_samples = X.shape[0]
        self._n_features = X.shape[1] if X.ndim > 1 else 1
        self._data_hash = hashlib.sha256(X.tobytes()).hexdigest()[:16]
        self._fitted = True

    def export_statistics(
        self,
        epsilon: float | None = None,
        delta: float = 1e-5,
    ) -> FittedStatistics:
        """Extract fitted statistics from detector's internal state.

        IMPORTANT: This reads the detector's actual attributes, not
        re-derived approximations. The attributes are set during fit()
        and are exactly what detect() uses at inference time.

        Args:
            epsilon: Differential privacy budget. None = no noise.
            delta: DP delta parameter.

        Returns:
            FittedStatistics that can be safely transmitted.
        """
        if not self._fitted or self._detector is None:
            raise RuntimeError("Must call fit() before export_statistics()")

        det = self._detector

        # Read directly from fitted detector -- no re-computation
        stats = FittedStatistics(
            node_id=self.node_id,
            timestamp=time.time(),
            n_samples=self._n_samples,
            n_features=self._n_features,
            # Basic statistics
            mean=det.mean.copy(),
            std=det.std.copy(),
            q1=det.q1.copy(),
            q3=det.q3.copy(),
            # ResonanceScore
            res_h_train=det._res_h_train.copy(),
            res_noise_ratio=det._res_noise_ratio.copy(),
            # KinematicScore
            kin_jerk_mean=det._kin_jerk_mean.copy(),
            kin_jerk_std=det._kin_jerk_std.copy(),
            kin_accel_mean=det._kin_accel_mean.copy(),
            kin_accel_std=det._kin_accel_std.copy(),
            # InfoGeometryScore
            ig_mean=det._ig_mean.copy(),
            ig_cov_inv=det._ig_cov_inv.copy(),
            ig_log_det=det._ig_log_det,
            # Provenance
            data_hash=self._data_hash,
        )

        if epsilon is not None:
            from omni_mercury_engine.federation.privacy import (
                DifferentialPrivacy,
            )

            dp = DifferentialPrivacy(epsilon=epsilon, delta=delta)
            stats = dp.apply(stats)

        return stats

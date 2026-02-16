"""Fitted statistics container -- the ONLY data that leaves a federated node.

These fields map 1:1 to MercuryAnomalyDetector's internal state after fit().
The detector stores 13 attributes; this container carries all 13 plus metadata.

IMPORTANT: If MercuryAnomalyDetector.fit() changes what it stores,
this class MUST be updated to match. They are coupled by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FittedStatistics:
    """Container for a fitted MercuryAnomalyDetector's complete state.

    These statistics are the ONLY information that leaves a federated node.
    Combined with differential privacy, they provide formal privacy guarantees.

    The fields below map directly to MercuryAnomalyDetector attributes:

    Basic statistics (used by z-score and IQR scoring):
        mean, std, q1, q3 -- shape (n_features,) each

    ResonanceScore state:
        res_h_train     -- harmonic energy ratio per feature, shape (n_features,)
        res_noise_ratio -- noise ratio per feature, shape (n_features,)

    KinematicScore state:
        kin_jerk_mean, kin_jerk_std   -- shape (n_features,) each
        kin_accel_mean, kin_accel_std -- shape (n_features,) each

    InfoGeometryScore state:
        ig_mean    -- Gaussian manifold center, shape (n_features,)
        ig_cov_inv -- precision matrix (regularized inverse covariance),
                     shape (n_features, n_features)
        ig_log_det -- log-determinant of regularized covariance, scalar
    """

    # --- Metadata ---
    node_id: str
    timestamp: float
    n_samples: int
    n_features: int

    # --- Basic statistics ---
    mean: np.ndarray              # (n_features,)
    std: np.ndarray               # (n_features,)
    q1: np.ndarray                # (n_features,)
    q3: np.ndarray                # (n_features,)

    # --- ResonanceScore ---
    res_h_train: np.ndarray       # (n_features,)
    res_noise_ratio: np.ndarray   # (n_features,)

    # --- KinematicScore ---
    kin_jerk_mean: np.ndarray     # (n_features,)
    kin_jerk_std: np.ndarray      # (n_features,)
    kin_accel_mean: np.ndarray    # (n_features,)
    kin_accel_std: np.ndarray     # (n_features,)

    # --- InfoGeometryScore ---
    ig_mean: np.ndarray           # (n_features,)
    ig_cov_inv: np.ndarray        # (n_features, n_features)
    ig_log_det: float = 0.0

    # --- Privacy metadata ---
    epsilon: float | None = None
    delta: float | None = None

    # --- Provenance ---
    data_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dictionary."""
        d: dict[str, Any] = {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "ig_log_det": self.ig_log_det,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "data_hash": self.data_hash,
        }
        for key in [
            "mean", "std", "q1", "q3",
            "res_h_train", "res_noise_ratio",
            "kin_jerk_mean", "kin_jerk_std",
            "kin_accel_mean", "kin_accel_std",
            "ig_mean", "ig_cov_inv",
        ]:
            d[key] = getattr(self, key).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FittedStatistics:
        """Deserialize from dictionary."""
        arrays = {}
        for key in [
            "mean", "std", "q1", "q3",
            "res_h_train", "res_noise_ratio",
            "kin_jerk_mean", "kin_jerk_std",
            "kin_accel_mean", "kin_accel_std",
            "ig_mean", "ig_cov_inv",
        ]:
            arrays[key] = np.array(d[key])
        return cls(
            node_id=d["node_id"],
            timestamp=d["timestamp"],
            n_samples=d["n_samples"],
            n_features=d["n_features"],
            ig_log_det=d.get("ig_log_det", 0.0),
            epsilon=d.get("epsilon"),
            delta=d.get("delta"),
            data_hash=d.get("data_hash", ""),
            **arrays,
        )

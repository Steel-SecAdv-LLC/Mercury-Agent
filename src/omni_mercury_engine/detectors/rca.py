# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Root-cause localisation detector over a causal / service graph.

When many correlated signals (service metrics, sensor channels) go anomalous at
once, the operational question is not *whether* there is an anomaly but *where it
originated*. Graph-based root-cause analysis propagates per-node anomaly evidence
over the causal / service dependency graph and ranks nodes by how much they
explain the observed pattern — the pattern behind tools like MonitorRank and
CloudRanger (Lin et al., 2018) and the random-walk RCA family.

This detector consumes a multivariate signal (one column per graph node), learns
per-node normal baselines in :meth:`fit`, converts each observation into per-node
standardised residuals, and runs a reverse personalised-random-walk over the
supplied adjacency to produce ranked root-cause attributions. The scalar anomaly
score is the calibrated peak node residual; the ranked causes and attribution
weights are returned in ``metadata``. Pure NumPy (always importable); registered
as an opt-in BASE detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors.detection_config import DetectionConfig, apply_nan_policy

if TYPE_CHECKING:
    import torch

__all__ = ["RootCauseGraphDetector"]

_LN2 = float(np.log(2.0))


class RootCauseGraphDetector(BaseDetector):
    """Random-walk root-cause localiser over a causal / service graph.

    The adjacency ``A`` (``A[i, j] > 0`` means ``i`` causally influences ``j``)
    is supplied at construction. :meth:`fit` learns per-node baselines from
    training rows; :meth:`detect` builds per-node standardised residuals, scores
    the row by the peak residual (squashed to ``[0, 1]``), and ranks root causes
    by a reverse personalised random walk seeded from the residual vector.
    """

    def __init__(
        self,
        adjacency: np.ndarray[Any, Any] | None = None,
        damping: float = 0.85,
        walk_iters: int = 50,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the RCA graph detector.

        Args:
            adjacency: ``(n_nodes, n_nodes)`` non-negative causal adjacency
                (``A[i, j] > 0`` ⇒ ``i`` influences ``j``). If ``None`` the graph
                is inferred at :meth:`fit` from training correlations.
            damping: Random-walk damping / teleport factor in ``(0, 1)``.
            walk_iters: Power-iteration steps for the random walk. Must be >= 1.
            calibration_quantile: Training peak-residual quantile at the 0.5
                boundary; ``1 - calibration_quantile`` is the normal-regime FPR.
                Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If ``adjacency`` is not square/non-negative, or a
                parameter is out of range.
        """
        super().__init__(config)
        self._detection_config = DetectionConfig.resolve(self._config)
        if adjacency is not None:
            adj = np.asarray(adjacency, dtype=np.float64)
            if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
                raise ValueError("adjacency must be a square 2-D matrix")
            if np.any(adj < 0.0):
                raise ValueError("adjacency must be non-negative")
            self._adjacency: np.ndarray[Any, Any] | None = adj
        else:
            self._adjacency = None
        if not 0.0 < damping < 1.0:
            raise ValueError(f"damping must be in (0, 1), got {damping}")
        if walk_iters < 1:
            raise ValueError("walk_iters must be >= 1")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.damping = float(damping)
        self.walk_iters = int(walk_iters)
        self.calibration_quantile = float(calibration_quantile)
        self._node_mean: np.ndarray[Any, Any] | None = None
        self._node_std: np.ndarray[Any, Any] | None = None
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned node baselines/scale."""
        return self._is_fitted

    def _to_2d_f64(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 2-D ``(rows, nodes)`` array (NaN policy applied)."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = np.asarray(data, dtype=np.float64)
        sanitized, _ = apply_nan_policy(
            arr,
            policy=self._detection_config.nan_policy,
            detector=self.name,
            field="input",
            max_magnitude=self._detection_config.max_magnitude,
        )
        arr = sanitized
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        q = float(np.quantile(raw, self.calibration_quantile))
        if q < 1e-9:
            q = float(np.mean(raw)) + 1e-9
        return max(q / _LN2, 1e-9)

    def _residuals(self, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Per-node standardised residuals ``|x - μ| / σ`` for each row."""
        assert self._node_mean is not None and self._node_std is not None
        return np.abs(rows - self._node_mean) / self._node_std

    def _walk(self, residual_row: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reverse personalised random walk seeded by a residual vector.

        Anomaly evidence flows *backwards* along causal edges (child → parent) so
        that upstream causes accumulate the mass of their affected descendants.
        """
        assert self._adjacency is not None
        n = residual_row.size
        seed = residual_row / (residual_row.sum() + 1e-12)
        # Column-normalise A: each child splits its mass across its parents, so
        # ``transition @ r`` moves residual mass child→parent (columns→rows) --
        # column-stochastic and mass-conserving, with no explicit transpose.
        col_sums = self._adjacency.sum(axis=0, keepdims=True)
        transition = self._adjacency / np.where(col_sums > 0.0, col_sums, 1.0)
        r = seed.copy()
        for _ in range(self.walk_iters):
            r = (1.0 - self.damping) * seed + self.damping * (transition @ r)
        total = r.sum()
        return r / total if total > 0.0 else np.full(n, 1.0 / n)

    def _peak_residuals(self, rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Peak per-node standardised residual for each row."""
        resid = self._residuals(rows)
        return resid.max(axis=1)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> RootCauseGraphDetector:
        """Learn per-node baselines (and, if needed, infer the graph) + scale.

        Args:
            data: Training rows ``(n_rows, n_nodes)`` of normal behaviour.

        Returns:
            ``self``.
        """
        rows = self._to_2d_f64(data)
        n_nodes = rows.shape[1]
        self._node_mean = rows.mean(axis=0)
        self._node_std = rows.std(axis=0) + 1e-9
        if self._adjacency is None:
            # Infer a causal graph from absolute training correlations
            # (thresholded, self-loops removed) when none was supplied.
            if rows.shape[0] >= 2 and n_nodes > 1:
                # A zero-variance (constant) node makes corrcoef divide by zero;
                # nan_to_num rescues the resulting NaN/Inf to 0 (no edge), so
                # silence the expected warning rather than let it leak.
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.abs(np.corrcoef(rows, rowvar=False))
                corr = np.nan_to_num(corr)
                np.fill_diagonal(corr, 0.0)
                corr[corr < 0.3] = 0.0
                self._adjacency = corr
            else:
                self._adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        elif self._adjacency.shape[0] != n_nodes:
            raise ValueError(f"adjacency size {self._adjacency.shape[0]} != n_nodes {n_nodes}")
        raw = self._peak_residuals(rows)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-row fusion feature: the peak node residual.

        Args:
            data: Input rows ``(n_rows, n_nodes)``.

        Returns:
            ``(n_rows, 1)`` float32 peak residuals.
        """
        raw = self._peak_residuals(self._to_2d_f64(data))
        return raw.astype(np.float32).reshape(-1, 1)

    def rank_root_causes(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> list[tuple[int, float]]:
        """Rank nodes as root causes for the final row's residual pattern.

        Args:
            data: Input rows ``(n_rows, n_nodes)``; the last row is localised.

        Returns:
            ``(node_index, attribution)`` pairs sorted by descending attribution.
        """
        rows = self._to_2d_f64(data)
        resid = self._residuals(rows)[-1]
        attribution = self._walk(resid)
        order = np.argsort(attribution)[::-1]
        return [(int(i), float(attribution[i])) for i in order]

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-row anomaly scores in ``[0, 1]`` plus ranked root causes.

        The peak node residual per row is squashed via ``1 - exp(-r / scale)``;
        ``metadata['ranked_causes']`` holds the random-walk attribution for the
        last row.
        """
        rows = self._to_2d_f64(data)
        raw = self._peak_residuals(rows)
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = np.clip(1.0 - np.exp(-raw / scale), 0.0, 1.0).astype(np.float32)
        ranked = self.rank_root_causes(rows) if self._is_fitted else []
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {"ranked_causes": ranked},
        }

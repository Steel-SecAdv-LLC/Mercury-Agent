# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Interacting-Multiple-Model (IMM) switching state-space residual detector.

The IMM estimator (Blom & Bar-Shalom, *The interacting multiple model algorithm
for systems with Markovian switching coefficients*, IEEE TAC 1988) runs a bank
of Kalman filters whose dynamics differ, and blends them each step through a
Markov mode-transition matrix. It is the classical answer to *tracking a system
whose regime switches* — a quiet constant-velocity model captures the normal
regime, while a high-process-noise "manoeuvring" model absorbs abrupt changes.

For anomaly detection the useful signal is the **combined one-step-ahead
predictive innovation**: when the observed value is well explained by the
current mode mixture the normalised innovation is small, but a change the whole
filter bank fails to predict produces a large innovation. This module wraps the
IMM in the :class:`~omni_mercury_engine.core.base.BaseDetector` contract; it is
pure NumPy (always importable) and registered as an opt-in BASE detector.
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

__all__ = ["IMMDetector"]

# ``1 - exp(-s / scale) = 0.5`` at ``s = scale * ln 2``; anchoring ``scale`` to a
# high training quantile places the 0.5 boundary in the tail for controlled FPR.
_LN2 = float(np.log(2.0))


class IMMDetector(BaseDetector):
    """Interacting-Multiple-Model switching state-space residual detector.

    Two constant-velocity Kalman filters share the state ``[level, slope]`` and
    the scalar measurement model ``y = level``; they differ only in process
    noise — a *quiet* mode (small ``q``) that tracks the normal regime and a
    *manoeuvring* mode (large ``q``) that accommodates regime switches. The IMM
    mixes them through a Markov transition matrix and scores each observation by
    the mode-averaged, normalised predictive innovation squashed into ``[0, 1]``.
    """

    def __init__(
        self,
        quiet_process_var: float = 1e-4,
        maneuver_process_var: float = 1.0,
        switch_prob: float = 0.05,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the IMM detector.

        Args:
            quiet_process_var: Process-noise variance of the quiet (normal-regime)
                mode. Must be > 0.
            maneuver_process_var: Process-noise variance of the manoeuvring mode;
                must be > ``quiet_process_var`` so the second mode is the one that
                absorbs abrupt changes.
            switch_prob: Off-diagonal Markov mode-transition probability (chance of
                switching mode each step). Must be in ``(0, 0.5)``.
            calibration_quantile: Training-innovation quantile placed at the 0.5
                anomaly boundary; ``1 - calibration_quantile`` is the resulting
                normal-regime false-positive rate. Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any variance is non-positive, the mode ordering is
                violated, or a probability/quantile is out of range.
        """
        super().__init__(config)
        if quiet_process_var <= 0.0 or maneuver_process_var <= 0.0:
            raise ValueError("process variances must be > 0")
        if maneuver_process_var <= quiet_process_var:
            raise ValueError("maneuver_process_var must exceed quiet_process_var")
        if not 0.0 < switch_prob < 0.5:
            raise ValueError(f"switch_prob must be in (0, 0.5), got {switch_prob}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        self.quiet_process_var = float(quiet_process_var)
        self.maneuver_process_var = float(maneuver_process_var)
        self.switch_prob = float(switch_prob)
        self.calibration_quantile = float(calibration_quantile)
        self._obs_var: float = 1.0
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned the noise/scale."""
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

    def _innovations(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Mode-averaged normalised one-step innovations for each observation.

        Runs the two-mode IMM: mode mixing → per-mode Kalman predict/update with
        Gaussian likelihood → mode-probability update → moment-matched combined
        estimate. The returned per-sample residual is the absolute combined
        predictive innovation divided by its predictive standard deviation.
        """
        n = series.size
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out

        # State-transition (constant velocity) and measurement matrices.
        f_mat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        h_vec = np.array([1.0, 0.0], dtype=np.float64)
        r_obs = max(self._obs_var, 1e-9)
        q_modes = (self.quiet_process_var, self.maneuver_process_var)
        n_modes = 2

        # Markov mode-transition matrix (rows sum to 1).
        p_trans = np.array(
            [
                [1.0 - self.switch_prob, self.switch_prob],
                [self.switch_prob, 1.0 - self.switch_prob],
            ],
            dtype=np.float64,
        )

        # Per-mode state means / covariances and mode probabilities.
        means = [np.array([series[0], 0.0], dtype=np.float64) for _ in range(n_modes)]
        covs = [np.eye(2, dtype=np.float64) * (r_obs + q_modes[j]) for j in range(n_modes)]
        mode_prob = np.full(n_modes, 1.0 / n_modes, dtype=np.float64)

        for t in range(n):
            # --- Interaction / mixing -------------------------------------
            # c_j = normaliser for mode j's mixed prior.
            c_j = p_trans.T @ mode_prob
            c_j = np.where(c_j <= 0.0, 1e-12, c_j)
            mix_w = (p_trans * mode_prob[:, None]) / c_j[None, :]  # w[i, j]

            mixed_means = []
            mixed_covs = []
            for j in range(n_modes):
                m_mix = sum(mix_w[i, j] * means[i] for i in range(n_modes))
                p_mix = np.zeros((2, 2), dtype=np.float64)
                for i in range(n_modes):
                    d = means[i] - m_mix
                    p_mix += mix_w[i, j] * (covs[i] + np.outer(d, d))
                mixed_means.append(m_mix)
                mixed_covs.append(p_mix)

            # --- Mode-matched Kalman predict + update ---------------------
            likelihood = np.zeros(n_modes, dtype=np.float64)
            comb_pred_mean = 0.0
            comb_pred_var = 0.0
            for j in range(n_modes):
                q_mat = np.array([[q_modes[j], 0.0], [0.0, q_modes[j]]], dtype=np.float64)
                x_pred = f_mat @ mixed_means[j]
                p_pred = f_mat @ mixed_covs[j] @ f_mat.T + q_mat
                y_pred = float(h_vec @ x_pred)
                s_inn = float(h_vec @ p_pred @ h_vec + r_obs)
                s_inn = max(s_inn, 1e-12)
                resid = float(series[t]) - y_pred
                # Gaussian innovation likelihood for the mode-probability update.
                likelihood[j] = np.exp(-0.5 * resid * resid / s_inn) / np.sqrt(2.0 * np.pi * s_inn)
                comb_pred_mean += mode_prob[j] * y_pred
                comb_pred_var += mode_prob[j] * (s_inn + y_pred * y_pred)
                # Kalman gain update on the observation.
                k_gain = (p_pred @ h_vec) / s_inn
                means[j] = x_pred + k_gain * resid
                covs[j] = p_pred - np.outer(k_gain, h_vec @ p_pred)

            # Combined predictive moments (mixture) → normalised innovation.
            comb_pred_var -= comb_pred_mean * comb_pred_mean
            comb_pred_std = np.sqrt(max(comb_pred_var, 1e-12))
            out[t] = abs(float(series[t]) - comb_pred_mean) / comb_pred_std

            # --- Mode-probability update ----------------------------------
            new_prob = c_j * likelihood
            total = float(np.sum(new_prob))
            mode_prob = (
                new_prob / total
                if total > 0.0 and np.isfinite(total)
                else np.full(n_modes, 1.0 / n_modes)
            )
        return out

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> IMMDetector:
        """Estimate observation noise and the innovation squash scale.

        Args:
            data: Training series (flattened to 1-D), predominantly normal.

        Returns:
            ``self``.
        """
        series = self._to_1d_f64(data)
        if series.size >= 2:
            diffs = np.diff(series)
            med = float(np.median(diffs))
            mad = float(np.median(np.abs(diffs - med)))
            self._obs_var = max((1.4826 * mad) ** 2, 1e-6)
        else:
            self._obs_var = 1.0
        raw = self._innovations(series)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the combined normalised innovation.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 innovations.
        """
        raw = self._innovations(self._to_1d_f64(data))
        return finite_features(raw, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` from IMM innovations.

        Innovations are squashed monotonically via ``1 - exp(-r / scale)`` using
        the training scale from :meth:`fit` (or the input's own robust scale when
        unfitted), so tracked observations score near 0 and un-predicted regime
        changes approach 1.
        """
        raw = self._innovations(self._to_1d_f64(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale), detector=self.name).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }

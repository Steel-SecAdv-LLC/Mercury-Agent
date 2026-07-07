# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral-Residual (SR) saliency detector for streaming time series.

The Spectral Residual transform (Ren et al., *Time-Series Anomaly Detection
Service at Microsoft*, KDD 2019) is a fast, training-free saliency method: the
log-amplitude spectrum of a signal is smoothed and subtracted from itself, and
the residual spectrum is inverted back to the time domain to produce a saliency
map whose peaks localise the points that are *poorly explained* by the signal's
own periodic structure. Anomalies show up as high-saliency spikes.

This module wraps the transform in the
:class:`~omni_mercury_engine.core.base.BaseDetector` contract consumed by the
Mercury fusion engine. ``fit`` learns a robust saliency scale from training data
so ``detect`` can emit per-sample scores squashed into ``[0, 1]``; the detector
is otherwise stateless, which is what makes it cheap enough for the streaming
path. It depends only on NumPy's FFT, so it is always available (no PyTorch or
other optional stack required) and is registered as an opt-in BASE detector in
``DETECTOR_MANIFEST``.
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

__all__ = ["SpectralResidualDetector"]

# ``1 - exp(-s / scale) = 0.5`` at ``s = scale * ln 2``; anchoring ``scale`` to a
# high training quantile places the 0.5 anomaly boundary at that quantile for a
# controlled ``1 - calibration_quantile`` false-positive rate.
_LN2 = float(np.log(2.0))


class SpectralResidualDetector(BaseDetector):
    """Streaming saliency detector built on the Spectral-Residual transform.

    The detector flattens its input to a 1-D series, computes the SR saliency
    map, and scores each point by the local saliency z-score. ``fit`` records a
    robust (median-absolute-deviation) saliency scale on training data so that
    :meth:`detect` can squash raw saliency into a monotone ``[0, 1]`` anomaly
    score via ``1 - exp(-s / scale)``. No model weights are learned, matching
    the training-free nature of the transform.
    """

    def __init__(
        self,
        window_amp: int = 3,
        window_local: int = 21,
        estimation_points: int = 5,
        calibration_quantile: float = 0.98,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the SR detector.

        Args:
            window_amp: Odd averaging-filter width for smoothing the
                log-amplitude spectrum (``q`` in the paper). Must be >= 1.
            window_local: Odd width of the local averaging window used to turn
                the saliency map into a local z-score. Must be >= 1.
            estimation_points: Number of trailing points averaged to extrapolate
                the series before the FFT, which sharpens saliency on the most
                recent (streaming) sample. Must be >= 1.
            calibration_quantile: Training-saliency quantile placed at the 0.5
                anomaly boundary; ``1 - calibration_quantile`` is the resulting
                normal-regime false-positive rate. Must be in ``(0, 1)``.
            config: Optional ``BaseDetector`` config (``threshold``,
                ``auto_calibrate``, ...).

        Raises:
            ValueError: If any window/estimation parameter is < 1 or
                ``calibration_quantile`` is out of ``(0, 1)``.
        """
        super().__init__(config)
        if window_amp < 1 or window_local < 1 or estimation_points < 1:
            raise ValueError("window_amp, window_local and estimation_points must all be >= 1")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        # Averaging filters are symmetric; force odd widths so the window is
        # centred on the point being scored.
        self.window_amp = int(window_amp) | 1
        self.window_local = int(window_local) | 1
        self.estimation_points = int(estimation_points)
        self.calibration_quantile = float(calibration_quantile)
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has learned the saliency scale."""
        return self._is_fitted

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    @staticmethod
    def _to_1d_f64(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D float64 series.

        A torch tensor is detached to CPU numpy by duck-typing so importing this
        module never requires PyTorch. Multi-dimensional input is flattened in C
        order into a single series.
        """
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = bound_finite(np.asarray(data, dtype=np.float64)).ravel()
        return arr

    @staticmethod
    def _moving_average(values: np.ndarray[Any, Any], window: int) -> np.ndarray[Any, Any]:
        """Centred moving average with edge padding, preserving length."""
        if window <= 1 or values.size == 0:
            return values.astype(np.float64, copy=True)
        window = min(window, values.size)
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window // 2
        padded = np.pad(values, pad, mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")
        return smoothed[pad : pad + values.size]

    def _extend_series(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Extrapolate one point so the newest sample sits inside the window.

        The paper extends the series by predicting the next value from the mean
        slope of the last ``estimation_points`` gradients; this keeps the
        saliency of the final (streaming) point from being suppressed by the
        FFT's periodic boundary.
        """
        n = series.size
        if n < 2:
            return series
        k = min(self.estimation_points, n - 1)
        recent = series[-k - 1 :]
        slopes = recent[1:] - recent[:-1]
        pred = series[-1] + float(np.mean(slopes))
        return np.append(series, pred)

    def _saliency_map(self, series: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute the Spectral-Residual saliency map for a 1-D series."""
        n = series.size
        if n < 2:
            return np.zeros(n, dtype=np.float64)
        extended = self._extend_series(series)
        spectrum = np.fft.fft(extended)
        amplitude = np.abs(spectrum)
        # Guard the logarithm against exact-zero amplitudes (constant segments).
        log_amp = np.log(np.maximum(amplitude, 1e-8))
        averaged = self._moving_average(log_amp, self.window_amp)
        spectral_residual = log_amp - averaged
        phase = np.angle(spectrum)
        reconstructed = np.fft.ifft(np.exp(spectral_residual + 1j * phase))
        saliency = np.abs(reconstructed)
        # Drop the extrapolated tail point to restore the original length.
        return saliency[:n]

    def _scores(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-point local saliency magnitude (pre-squash), length ``n``."""
        series = self._to_1d_f64(data)
        saliency = self._saliency_map(series)
        local = self._moving_average(saliency, self.window_local)
        # Local deviation above the smoothed saliency baseline is the anomaly
        # signal; relative form keeps it scale-invariant across series.
        deviation = saliency - local
        return np.maximum(deviation, 0.0)

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> SpectralResidualDetector:
        """Learn the robust saliency scale used to squash scores into ``[0, 1]``.

        Args:
            data: Training time series (any shape; flattened to 1-D). Should be
                predominantly normal so the learned scale reflects background
                saliency rather than anomalies.

        Returns:
            ``self``.
        """
        raw = self._scores(data)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-sample fusion feature: the raw local saliency deviation.

        Args:
            data: Input series (flattened to 1-D).

        Returns:
            ``(n_samples, 1)`` float32 array of saliency deviations.
        """
        raw = self._scores(data)
        return finite_features(raw).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-sample anomaly scores in ``[0, 1]`` for the live-inference path.

        Saliency deviations are squashed monotonically via ``1 - exp(-s /
        scale)`` using the training scale from :meth:`fit` (or the input's own
        robust scale when called unfitted), so background points score near 0
        and salient spikes approach 1.
        """
        raw = self._scores(data)
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = 1.0 - np.exp(-raw / scale)
        scores = finite_scores(scores).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
        }

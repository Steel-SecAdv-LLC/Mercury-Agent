# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TimeGPT Adapter for Mercury-Agent.

Integrates Nixtla's TimeGPT foundation model for zero-shot
time-series forecasting and anomaly detection.

TimeGPT is trained on 100B+ data points across diverse domains:
- Retail
- Electricity
- Finance
- IoT

Reference:
    Nixtla TimeGPT: https://github.com/Nixtla/nixtla
    TimeGEN-1 in Azure: Microsoft Build 2024
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from omni_mercury_engine.core.exceptions import ModelException
from omni_mercury_engine.detectors.provenance import DecisionSource
from omni_mercury_engine.models.foundation.base_foundation import (
    BaseFoundationModel,
    FoundationModelConfig,
)

logger = logging.getLogger(__name__)


def _api_level(level: float) -> int | float:
    """A confidence level as Nixtla takes it: integral levels as int.

    Nixtla names interval columns after the level it was given
    (``TimeGPT-lo-90``); an integral level is sent as an int so the name is
    the documented one.
    """
    return int(level) if float(level).is_integer() else float(level)


@dataclass
class TimeGPTConfig(FoundationModelConfig):
    """Configuration for TimeGPT adapter.

    Attributes:
        api_key: Nixtla API key
        model: Model variant ('timegpt-1', 'timegpt-1-long-horizon')
        freq: Time series frequency (e.g., 'H', 'D', 'M')
        fh: Forecast horizon (alias for prediction_length)
        finetune_steps: Steps for fine-tuning (0 = no fine-tuning)
        interval_level: Confidence level, in percent (0-100), of the
            forecast's prediction interval -- Nixtla's ``level`` (the
            0-1 ``quantiles`` used to be passed there).
        anomaly_level: Confidence level, in percent, of the interval
            outside which ``detect_anomalies`` flags a point (Nixtla's
            default, 99).
    """

    api_key: str | None = None
    model: str = "timegpt-1"
    freq: str = "H"
    fh: int = 24  # Forecast horizon
    finetune_steps: int = 0
    model_name: str = "timegpt-1"
    interval_level: float = 90.0
    anomaly_level: float = 99.0

    def __post_init__(self) -> None:
        """Sync fh with prediction_length; check the interval levels."""
        if self.fh != 24:
            self.prediction_length = self.fh
        for name in ("interval_level", "anomaly_level"):
            level = getattr(self, name)
            if not 0.0 < level < 100.0:
                raise ValueError(f"{name} is a percentage in (0, 100), got {level}")


class TimeGPTAdapter(BaseFoundationModel):
    """TimeGPT adapter for time-series anomaly detection.

    Provides zero-shot forecasting and anomaly detection using
    Nixtla's pre-trained TimeGPT foundation model.

    Features:
        - Zero-shot prediction without training
        - Anomaly detection via prediction confidence intervals
        - Optional fine-tuning on domain data
        - Multiple model variants for different horizons

    Example:
        >>> adapter = TimeGPTAdapter(api_key="your_api_key")
        >>> results = adapter.detect_anomalies(time_series)
        >>> print(f"Anomalies at: {np.where(results['is_anomaly'])[0]}")
    """

    decision_source = DecisionSource(
        kind="published",
        description=(
            "Anomaly verdicts are served from Nixtla's own "
            "detect_anomalies API: a point is anomalous when it falls "
            "outside TimeGPT's level-`anomaly_level` (default 99%) "
            "prediction interval, and the score is its distance beyond the "
            "interval over the interval width. The decision rule is the "
            "external vendor's, not a local threshold."
        ),
        citation=(
            "Nixtla TimeGPT anomaly-detection API, level-99 prediction "
            "intervals (github.com/Nixtla/nixtla; TimeGEN-1, Microsoft "
            "Build 2024), as referenced in this module"
        ),
    )

    def __init__(self, config: TimeGPTConfig | dict[str, Any] | None = None) -> None:
        """Initialize TimeGPT adapter.

        Args:
            config: Adapter configuration including API key
        """
        if config is None:
            config = TimeGPTConfig()
        elif isinstance(config, dict):
            config = TimeGPTConfig(**config)

        # Store typed config in a separate attribute (don't override base class config)
        self.timegpt_config: TimeGPTConfig = config
        super().__init__(config)

        self._client: Any = None

    @property
    def config(self) -> TimeGPTConfig:
        """Return the typed TimeGPT configuration."""
        return self.timegpt_config

    @config.setter
    def config(self, value: dict[str, Any] | TimeGPTConfig) -> None:
        """Store the underlying config object (required for base class compatibility).

        The base class sets self.config to a dict during __init__. We intercept this and store it,
        but always return the typed config.
        """
        if isinstance(value, TimeGPTConfig):
            self.timegpt_config = value
        # If dict, it's from base class init - we already have typed config set

    def _initialize_model(self) -> None:
        """Initialize Nixtla client."""
        try:
            from nixtla import NixtlaClient

            api_key = self.timegpt_config.api_key
            if api_key is None:
                # Try environment variable
                import os

                api_key = os.environ.get("NIXTLA_API_KEY")

            if api_key is None:
                raise NotImplementedError(
                    "No Nixtla API key provided. Set NIXTLA_API_KEY or pass "
                    "api_key in TimeGPTConfig. Silent mock degradation is not "
                    "permitted (Phase 2 audit cure)."
                )
            else:
                self._client = NixtlaClient(api_key=api_key)
                logger.info("TimeGPT client initialized successfully")

        except ImportError as exc:
            raise NotImplementedError(
                "nixtla package not installed. Install with: pip install nixtla. "
                "Silent mock degradation is not permitted (Phase 2 audit cure)."
            ) from exc

    def _to_dataframe(
        self,
        series: np.ndarray[Any, Any],
        start_time: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Convert numpy array to pandas DataFrame for TimeGPT.

        Args:
            series: Time series array [T]
            start_time: Optional start timestamp

        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        if start_time is None:
            # Whole seconds: the verdict is aligned back to the input by
            # timestamp, and sub-second stamps need not survive the API.
            start_time = pd.Timestamp.now().floor("s")

        freq_map = {
            "H": "h",
            "D": "D",
            "M": "ME",
            "W": "W",
            "T": "min",
            "S": "s",
        }
        freq = freq_map.get(self.timegpt_config.freq, self.timegpt_config.freq)

        timestamps = pd.date_range(
            start=start_time,
            periods=len(series),
            freq=freq,
        )

        return pd.DataFrame(
            {
                "ds": timestamps,
                "y": series,
            }
        )

    def forecast(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Generate forecasts using TimeGPT.

        Args:
            series: Input time series [T] or [B, T]
            horizon: Forecast horizon (default from config)

        Returns:
            Dict with forecast, lower, upper bounds
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        horizon = horizon or self.foundation_config.prediction_length
        batch_size = series.shape[0]

        forecasts = []
        lowers = []
        uppers = []

        level = _api_level(self.timegpt_config.interval_level)
        lo_col, hi_col = f"TimeGPT-lo-{level}", f"TimeGPT-hi-{level}"
        for i in range(batch_size):
            df = self._to_dataframe(series[i])
            try:
                result = self._client.forecast(
                    df=df,
                    h=horizon,
                    model=self.timegpt_config.model,
                    freq=self.timegpt_config.freq,
                    finetune_steps=self.timegpt_config.finetune_steps,
                    level=[level],
                )
            except Exception as e:
                # A naive last-value forecast with +-10 % "bounds" used to be
                # returned here, indistinguishable from a TimeGPT forecast.
                raise ModelException(f"TimeGPT forecast failed for series {i}: {e}") from e
            missing = [c for c in ("TimeGPT", lo_col, hi_col) if c not in result.columns]
            if missing:
                # The point forecast used to stand in for an absent bound.
                raise ModelException(f"TimeGPT forecast response lacks the columns {missing}")
            forecasts.append(result["TimeGPT"].to_numpy(dtype=np.float64))
            lowers.append(result[lo_col].to_numpy(dtype=np.float64))
            uppers.append(result[hi_col].to_numpy(dtype=np.float64))

        # For single series input (1D), return just the forecast array for API simplicity
        # For batch input (2D), return the full dict with all bounds
        if batch_size == 1:
            return {"forecast": forecasts[0], "lower": lowers[0], "upper": uppers[0]}

        return {
            "forecast": np.stack(forecasts),
            "lower": np.stack(lowers),
            "upper": np.stack(uppers),
        }

    def detect_anomalies(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies using TimeGPT.

        Uses prediction intervals to identify anomalies:
        - Points outside prediction intervals are anomalous
        - Score = distance from interval / interval width

        Args:
            series: Input time series [T] or [B, T]

        Returns:
            Dict with scores, is_anomaly flags, threshold
        """
        self._ensure_initialized()

        if isinstance(series, torch.Tensor):
            series = series.cpu().numpy()

        if series.ndim == 1:
            series = series.reshape(1, -1)

        batch_size, seq_len = series.shape
        all_scores = []
        all_anomalies = []

        level = _api_level(self.timegpt_config.anomaly_level)
        lo_col, hi_col = f"TimeGPT-lo-{level}", f"TimeGPT-hi-{level}"
        for i in range(batch_size):
            df = self._to_dataframe(series[i])
            try:
                result = self._client.detect_anomalies(
                    df=df, freq=self.timegpt_config.freq, level=level
                )
            except Exception as e:
                # A rolling z-score used to be returned here, indistinguishable
                # from TimeGPT's own verdict.
                raise ModelException(f"TimeGPT anomaly detection failed for series {i}: {e}") from e
            scores, is_anomaly = self._align_anomalies(df, result, lo_col, hi_col)
            all_scores.append(scores)
            all_anomalies.append(is_anomaly)

        return {
            "scores": np.stack(all_scores) if batch_size > 1 else all_scores[0],
            "is_anomaly": np.stack(all_anomalies) if batch_size > 1 else all_anomalies[0],
            "threshold": self.foundation_config.anomaly_threshold,
        }

    @staticmethod
    def _align_anomalies(
        df: pd.DataFrame, result: pd.DataFrame, lo_col: str, hi_col: str
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Per-input-point score and flag from TimeGPT's in-sample verdict.

        TimeGPT grades the points it has context for; the rest are NaN
        (unassessed) with no flag.  The score is the distance outside the
        prediction interval over its width (0 inside), as documented above;
        without interval columns it is TimeGPT's own 0/1 flag.
        """
        if "anomaly" not in result.columns or "ds" not in result.columns:
            raise ModelException(
                "TimeGPT anomaly response lacks the 'ds' / 'anomaly' columns; got "
                f"{list(result.columns)}"
            )
        index = pd.Index(pd.to_datetime(df["ds"]))
        rows = index.get_indexer(pd.to_datetime(result["ds"]))
        if (rows < 0).any():
            raise ModelException("TimeGPT graded timestamps that are not in the input")
        flags = np.zeros(len(df), dtype=bool)
        flags[rows] = result["anomaly"].to_numpy(dtype=bool)
        scores = np.full(len(df), np.nan)
        if lo_col in result.columns and hi_col in result.columns:
            y = df["y"].to_numpy(dtype=np.float64)[rows]
            lo = result[lo_col].to_numpy(dtype=np.float64)
            hi = result[hi_col].to_numpy(dtype=np.float64)
            width = np.maximum(hi - lo, 1e-12)
            scores[rows] = np.maximum.reduce([np.zeros_like(y), y - hi, lo - y]) / width
        else:
            scores[rows] = flags[rows].astype(np.float64)
        return scores, flags

    def detect(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
    ) -> dict[str, Any]:
        """Detect anomalies in time series data.

        This is the primary detection interface that wraps detect_anomalies
        for a consistent API across all foundation model adapters.

        Args:
            series: Input time series [T] or [B, T]

        Returns:
            Dict with scores, is_anomaly flags, and threshold
        """
        return self.detect_anomalies(series)

    def fine_tune(
        self,
        series: np.ndarray[Any, Any] | torch.Tensor,
        steps: int = 100,
    ) -> TimeGPTAdapter:
        """Fine-tune TimeGPT on each later forecast call.

        TimeGPT fine-tunes server-side on the series of each forecast request
        (Nixtla's ``finetune_steps``); this sets the number of steps.  The
        method used to log "Fine-tuned TimeGPT" without sending anything.

        Args:
            series: Accepted for the released signature; TimeGPT fine-tunes
                on the series each forecast call sends, not on this one.
            steps: Fine-tuning steps per forecast call (>= 0).

        Returns:
            Self for method chaining
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError(f"steps must be a non-negative int, got {steps!r}")
        self._ensure_initialized()
        self.timegpt_config.finetune_steps = steps
        logger.info("TimeGPT will fine-tune %d steps on each forecast call", steps)
        return self

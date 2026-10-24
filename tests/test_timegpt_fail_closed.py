# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TimeGPT serves Nixtla's verdict or raises; it never fabricates one.

On an API error the adapter used to return a last-value forecast with
+-10 % "bounds" or a rolling z-score, indistinguishable from TimeGPT's own
output; it passed the 0-1 ``quantiles`` as Nixtla's ``level`` (a percentage,
0-100: https://github.com/Nixtla/nixtla, ``NixtlaClient.forecast``); and an
absent interval column silently became the point forecast.  A fake client
stands in for the Nixtla API, which CI cannot reach.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.core.exceptions import ModelException
from omni_mercury_engine.models.foundation.timegpt_adapter import TimeGPTAdapter, TimeGPTConfig


class _FakeNixtla:
    """Records calls and answers in the documented response shape."""

    def __init__(self, *, fail: bool = False, drop: tuple[str, ...] = ()) -> None:
        self.fail = fail
        self.drop = drop
        self.calls: list[dict[str, Any]] = []

    def _maybe_fail(self) -> None:
        if self.fail:
            raise ConnectionError("503 Service Unavailable")

    def forecast(self, *, df: pd.DataFrame, h: int, level: list[Any], **kw: Any) -> pd.DataFrame:
        self.calls.append({"method": "forecast", "level": level, **kw})
        self._maybe_fail()
        lv = level[0]
        point = np.full(h, float(df["y"].iloc[-1]))
        out = pd.DataFrame(
            {
                "TimeGPT": point,
                f"TimeGPT-lo-{lv}": point - 2.0,
                f"TimeGPT-hi-{lv}": point + 2.0,
            }
        )
        return out.drop(columns=list(self.drop))

    def detect_anomalies(self, *, df: pd.DataFrame, level: Any, **kw: Any) -> pd.DataFrame:
        self.calls.append({"method": "detect_anomalies", "level": level, **kw})
        self._maybe_fail()
        graded = df.iloc[3:].reset_index(drop=True)  # TimeGPT needs context first
        y = graded["y"].to_numpy()
        out = pd.DataFrame(
            {
                "ds": graded["ds"],
                "y": y,
                "TimeGPT": np.zeros(len(y)),
                f"TimeGPT-lo-{level}": np.full(len(y), -1.0),
                f"TimeGPT-hi-{level}": np.full(len(y), 1.0),
                "anomaly": np.abs(y) > 1.0,
            }
        )
        return out.drop(columns=list(self.drop))


def _adapter(client: _FakeNixtla, **config: Any) -> TimeGPTAdapter:
    adapter = TimeGPTAdapter(TimeGPTConfig(**config))
    adapter._client = client
    adapter._is_initialized = True
    return adapter


SERIES = np.array([0.1, -0.2, 0.0, 0.3, 5.0, 0.2, -3.0, 0.1])


def test_forecast_requests_a_percentage_level_and_serves_its_interval() -> None:
    client = _FakeNixtla()
    out = _adapter(client).forecast(SERIES, horizon=4)
    assert client.calls[0]["level"] == [90]  # a percentage, sent as int
    np.testing.assert_allclose(out["forecast"], 0.1)
    np.testing.assert_allclose(out["lower"], 0.1 - 2.0)
    np.testing.assert_allclose(out["upper"], 0.1 + 2.0)


def test_forecast_api_error_raises_instead_of_a_naive_forecast() -> None:
    with pytest.raises(ModelException, match="503"):
        _adapter(_FakeNixtla(fail=True)).forecast(SERIES, horizon=4)


def test_missing_interval_column_raises_instead_of_the_point_forecast() -> None:
    with pytest.raises(ModelException, match="TimeGPT-lo-90"):
        _adapter(_FakeNixtla(drop=("TimeGPT-lo-90",))).forecast(SERIES, horizon=4)


def test_anomalies_are_timegpts_aligned_to_the_input() -> None:
    client = _FakeNixtla()
    out = _adapter(client).detect_anomalies(SERIES)
    assert client.calls[0]["level"] == 99
    assert out["scores"].shape == SERIES.shape
    assert np.isnan(out["scores"][:3]).all()  # no context: unassessed, not "normal"
    assert out["is_anomaly"].tolist() == [False, False, False, False, True, False, True, False]
    # Distance outside the [-1, 1] interval over its width.
    assert out["scores"][4] == pytest.approx((5.0 - 1.0) / 2.0)
    assert out["scores"][6] == pytest.approx((-1.0 - -3.0) / 2.0)
    assert out["scores"][5] == 0.0


def test_anomaly_api_error_raises_instead_of_a_z_score() -> None:
    with pytest.raises(ModelException, match="503"):
        _adapter(_FakeNixtla(fail=True)).detect_anomalies(SERIES)


def test_without_interval_columns_the_score_is_timegpts_flag() -> None:
    client = _FakeNixtla(drop=("TimeGPT-lo-99", "TimeGPT-hi-99"))
    out = _adapter(client).detect_anomalies(SERIES)
    assert out["scores"][3:].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("level", [0.0, -5.0, 100.0, 150.0])
def test_interval_level_is_a_percentage(level: float) -> None:
    with pytest.raises(ValueError, match="percentage"):
        TimeGPTConfig(interval_level=level)


def test_fine_tune_sets_the_per_call_steps() -> None:
    client = _FakeNixtla()
    adapter = _adapter(client).fine_tune(SERIES, steps=5)
    adapter.forecast(SERIES, horizon=2)
    assert client.calls[0]["finetune_steps"] == 5
    with pytest.raises(ValueError, match="non-negative"):
        adapter.fine_tune(SERIES, steps=-1)

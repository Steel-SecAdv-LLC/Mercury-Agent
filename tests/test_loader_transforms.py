# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test loader transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.loaders.transforms import prepare_for_detector


def test_basic_dataframe_to_numpy() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = prepare_for_detector(df, expected_columns=["a", "b"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    assert result.dtype == np.float64


def test_nan_handling() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = prepare_for_detector(df, expected_columns=["a", "b"])
    assert not np.any(np.isnan(result))


def test_empty_dataframe_raises() -> None:
    df = pd.DataFrame({"a": [], "b": []})
    with pytest.raises(ValueError, match="Empty DataFrame"):
        prepare_for_detector(df, expected_columns=["a", "b"])


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_for_detector(df, expected_columns=["a", "b"])

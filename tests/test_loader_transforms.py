"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

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

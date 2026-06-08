# Copyright (C) 2025 Steel Security Advisors LLC
"""Standardized transformation from raw loader output to MercuryAnomalyDetector input."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def prepare_for_detector(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> np.ndarray[Any, Any]:
    """Convert loader DataFrame to detector-ready numpy array.

    Args:
        df: Output of loader.engineer_features() wrapped as DataFrame,
            or a raw DataFrame with the expected columns.
        expected_columns: Loader's FEATURE_COLUMNS list.

    Returns:
        np.ndarray[Any, Any] of shape (n_samples, n_features), dtype float64.

    Raises:
        ValueError: If columns missing, NaN present after cleaning, or empty DataFrame.
    """
    if df.empty:
        raise ValueError("Empty DataFrame — no data to transform")

    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[expected_columns].copy()

    # Enforce numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Handle NaN: drop rows where > 50% features are NaN, impute rest with median
    threshold = len(expected_columns) * 0.5
    X = X.dropna(thresh=int(threshold))
    X = X.fillna(X.median())

    # Handle infinities
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    if X.empty:
        raise ValueError("All rows dropped during cleaning")

    return np.asarray(X.to_numpy(dtype=np.float64))

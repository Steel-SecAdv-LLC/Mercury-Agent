# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ground-truth integrity for NetworkSecurityLoader.

Regression: when the dataset-infrastructure loaders were unavailable and the
fetched dataframe had no ``label`` column, ``get_ground_truth`` fabricated an
all-normal (all-zeros) label vector. That silently corrupts any benchmark
grading against it — a mute detector scores a perfect run, a working one
scores all false positives. It must fail loud instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.network_security_loader import NetworkSecurityLoader


def test_ground_truth_fails_loud_when_labels_underivable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No label column + no dataset-infra labels → raise, never fabricate."""
    loader = NetworkSecurityLoader()

    # Force the dataset-infrastructure path to yield nothing.
    monkeypatch.setattr(loader, "_load_labels_from_dataset", lambda event_id: None)
    # Fallback dataframe has real rows but no 'label' column.
    monkeypatch.setattr(
        loader,
        "fetch_historical",
        lambda event_id: pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6]}),
    )

    with pytest.raises(DataSourceUnavailableError, match="Refusing to fabricate labels"):
        loader.get_ground_truth("nsl_kdd")


def test_ground_truth_uses_label_column_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The transparent path still works: a 'label' column is returned verbatim."""
    loader = NetworkSecurityLoader()

    monkeypatch.setattr(loader, "_load_labels_from_dataset", lambda event_id: None)
    monkeypatch.setattr(
        loader,
        "fetch_historical",
        lambda event_id: pd.DataFrame({"feature_a": [1, 2, 3], "label": [0, 1, 0]}),
    )

    labels = loader.get_ground_truth("nsl_kdd")
    assert np.array_equal(labels, np.array([0, 1, 0], dtype=np.int64))


def test_ground_truth_empty_dataframe_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty fetch is a distinct, non-fabricating case (empty labels)."""
    loader = NetworkSecurityLoader()

    monkeypatch.setattr(loader, "_load_labels_from_dataset", lambda event_id: None)
    monkeypatch.setattr(loader, "fetch_historical", lambda event_id: pd.DataFrame())

    labels = loader.get_ground_truth("nsl_kdd")
    assert labels.shape == (0,)

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for two loader defects.

1. FEMA event filters that name a fiscal year must actually constrain to it.
   ``hurricane_2024`` and ``fire_2023`` previously filtered by incident type
   only, so they returned declarations from every year despite their names.
2. ``marine_loader._assign_grid_cells`` must not raise on a non-empty frame that
   lacks the coordinate columns (the missing-column default was length-0, so the
   ``grid_cell`` assignment length-mismatched the frame index).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.fema_loader import _EVENT_CATALOG
from omni_mercury_engine.loaders.marine_loader import _assign_grid_cells

_YEAR_SUFFIX = re.compile(r"_(\d{4})$")


def test_fema_year_named_events_constrain_to_that_year() -> None:
    """Every event id ending in ``_YYYY`` must filter on that fiscal year."""
    offenders = []
    for event_id, meta in _EVENT_CATALOG.items():
        match = _YEAR_SUFFIX.search(event_id)
        if not match:
            continue
        year = match.group(1)
        filter_str = meta.get("filter", "")
        if f"fyDeclared eq {year}" not in filter_str:
            offenders.append((event_id, filter_str))
    assert offenders == [], f"year-named events with no year filter: {offenders}"


def test_assign_grid_cells_handles_frame_missing_coordinates() -> None:
    """A non-empty frame without lat/lon columns yields 'unknown' cells, no error."""
    frame = pd.DataFrame({"scientificName": ["a", "b", "c"]})

    out = _assign_grid_cells(frame, 1.0)

    assert list(out["grid_cell"]) == ["unknown", "unknown", "unknown"]


def test_assign_grid_cells_bins_present_coordinates() -> None:
    """With coordinates present, cells are floored to the resolution grid."""
    frame = pd.DataFrame({"decimalLatitude": [40.2, np.nan], "decimalLongitude": [-105.1, 30.0]})

    out = _assign_grid_cells(frame, 1.0)

    assert out["grid_cell"].iloc[0] == "40.0_-106.0"
    assert out["grid_cell"].iloc[1] == "unknown"  # NaN latitude -> unknown

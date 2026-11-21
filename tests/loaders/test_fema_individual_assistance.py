# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FEMA's individual-assistance flag moved columns; the label rule did not follow.

``iaProgramDeclared`` is populated on declarations up to about 2007. Later
declarations record individual assistance as ``ihProgramDeclared`` (the
Individuals and Households Program) and leave ``iaProgramDeclared`` false.
Measured against the live OpenFEMA API on 2026-11-16: across the 1,738
declarations of the ``all_2023`` and ``hurricane_2024`` events not one has
``iaProgramDeclared`` set (Hurricane Helene included), 293 have
``ihProgramDeclared``, and the loader labelled all 1,738 normal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from omni_mercury_engine.loaders.fema_loader import FEMALoader

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _declarations() -> pd.DataFrame:
    rows = [
        # type, ia, ih, pa
        ("DR", True, False, True),  # 1994: IA era
        ("DR", False, True, True),  # 2024: the same assistance, recorded as IH
        ("DR", False, False, True),  # public assistance only
        ("EM", False, True, True),  # an emergency declaration, not a major disaster
        ("DR", False, True, False),  # no public assistance
    ]
    return pd.DataFrame(
        {
            "disasterNumber": range(4000, 4000 + len(rows)),
            "state": ["NC"] * len(rows),
            "declarationType": [row[0] for row in rows],
            "declarationDate": pd.date_range("2024-09-28", periods=len(rows), freq="D", tz="UTC"),
            "incidentType": ["Hurricane"] * len(rows),
            "iaProgramDeclared": [row[1] for row in rows],
            "ihProgramDeclared": [row[2] for row in rows],
            "paProgramDeclared": [row[3] for row in rows],
            "hmProgramDeclared": [True] * len(rows),
        }
    )


def test_a_modern_major_disaster_with_individual_assistance_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``[1, 0, 0, 0, 0]`` -- every declaration since about 2007 was normal."""
    loader = FEMALoader(cache_dir=tmp_path)
    frame = _declarations()
    monkeypatch.setattr(loader, "fetch_historical", lambda _event_id: frame)

    labels = loader.get_ground_truth("hurricane_2024")

    assert labels.tolist() == [1, 1, 0, 0, 0]


def test_the_feature_reads_the_same_flag_the_label_does(tmp_path: Path) -> None:
    """``ia_program`` was 0 on every modern row while the IH column said otherwise."""
    loader = FEMALoader(cache_dir=tmp_path)

    features = loader.engineer_features(_declarations())

    column = FEMALoader.FEATURE_COLUMNS.index("ia_program")
    assert features[:, column].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0]
    # Still excluded from the scored matrix: it is half of the label.
    assert "ia_program" in FEMALoader.LABEL_DERIVED_FEATURES


def test_a_frame_without_the_newer_column_still_reads_the_older_one() -> None:
    frame = _declarations().drop(columns=["ihProgramDeclared"])

    assert np.array_equal(
        FEMALoader._individual_assistance(frame), np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    )

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Surges in FEMA's declaration stream, labelled by NOAA's catalogue.

``FEMALoader`` labels a declaration by its own programme flags. With those
flags kept out of the scored matrix its five events average AUROC 0.4832
(measured 2026-11-20 with feature rows aligned to their labels; 0.4930 before
that alignment), and no column independent of the label tracks it in one
direction (designated areas: 0.71 on one event, 0.23 on another, as measured
2026-11-16). The label is an administrative category, not an anomaly.

``FEMAActivityLoader`` asks what a monitor of that stream asks, whether
declaration activity is surging, and takes its labels from outside FEMA's
record: NOAA NCEI's billion-dollar tropical cyclones, each from its first day
to thirty days after its last (the period a governor has to request a major
disaster declaration, 44 CFR 206.36(a)). Measured live on 2026-11-16 and again
on 2026-11-20, fitted and scored in place: 0.70 to 0.95 per year, mean 0.8583
over 2017 to 2024 both times.

Offline: the transport serves files recorded from OpenFEMA and NOAA
(``tests/fixtures/fema``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.loaders.fema_activity_loader import (
    NOAA_WINDOW_COLUMN,
    REQUEST_WINDOW_DAYS,
    FEMAActivityLoader,
)
from omni_mercury_engine.loaders.label_provenance import LABEL_PROVENANCE_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fema"
_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"


def _sources(seen: list[str] | None = None) -> Callable[..., bytes]:
    def _transport(url: str, **_kwargs: Any) -> bytes:
        if seen is not None:
            seen.append(url)
        if "ncei.noaa.gov" in url:
            return (_FIXTURES / "noaa_billion_dollar_events_1980_2024.csv").read_bytes()
        if "DisasterDeclarationsSummaries" in url and "2023-01-01" in url:
            return (_FIXTURES / "openfema_tropical_declarations_2023.json").read_bytes()
        raise ConnectionError(f"no recorded file for {url}")

    return _transport


def _loader(tmp_path: Path) -> FEMAActivityLoader:
    return FEMAActivityLoader(cache_dir=tmp_path, max_retries=0)


def test_a_year_is_one_row_per_day_counted_from_the_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(_GET_BYTES, _sources(seen))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tropical_declarations_2023")

    assert len(frame) == 365
    assert frame["timestamp"].iloc[0] == pd.Timestamp("2023-01-01", tz="UTC")
    assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(days=1)).all()
    # All 229 recorded designations are counted, on the 12 days they were declared.
    assert int(frame["designations"].sum()) == 229
    assert int((frame["designations"] > 0).sum()) == 12
    # The filter names the three tropical incident types and the calendar year.
    request = next(url for url in seen if "DisasterDeclarationsSummaries" in url)
    assert "Typhoon" in request and "2024-01-01" in request
    assert not loader.is_synthetic(frame)


def test_the_labels_are_noaas_storms_and_the_request_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2023 has two billion-dollar tropical cyclones: Mawar (24-25 May), Idalia (29-31 Aug)."""
    monkeypatch.setattr(_GET_BYTES, _sources())
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("tropical_declarations_2023")
    labels = loader.get_ground_truth("tropical_declarations_2023")

    assert labels.tolist() == frame[NOAA_WINDOW_COLUMN].tolist()
    labelled = frame["timestamp"][labels == 1]
    assert REQUEST_WINDOW_DAYS == 30
    assert labelled.iloc[0] == pd.Timestamp("2023-05-24", tz="UTC")
    assert labelled.iloc[-1] == pd.Timestamp("2023-09-30", tz="UTC")
    # 24 May to 24 June, and 29 August to 30 September.
    assert int(labels.sum()) == 32 + 33
    # Tropical Storm Bolaven was declared on 8 October (six designations) and
    # is not in NOAA's list: the labels are NOAA's, not a restatement of
    # FEMA's activity. And Typhoon Mawar's emergency declaration of 22 May
    # preceded the storm's first day in the catalogue.
    bolaven = frame["timestamp"] == pd.Timestamp("2023-10-08", tz="UTC")
    assert frame.loc[bolaven, "designations"].iloc[0] == 6
    assert labels[bolaven.to_numpy()][0] == 0
    early = frame["timestamp"] == pd.Timestamp("2023-05-22", tz="UTC")
    assert frame.loc[early, "designations"].iloc[0] == 6
    assert labels[early.to_numpy()][0] == 0


def test_features_are_the_counts_and_their_trailing_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, _sources())
    loader = _loader(tmp_path)
    frame = loader.fetch_historical("tropical_declarations_2023")

    features = loader.engineer_features(frame)
    scored = loader.engineer_scored_features(frame)

    assert features.shape == (365, len(FEMAActivityLoader.FEATURE_COLUMNS))
    assert np.array_equal(features, scored)  # nothing is label-derived
    designations = frame["designations"].to_numpy(dtype=float)
    assert np.array_equal(features[:, 0], designations)
    week = FEMAActivityLoader.FEATURE_COLUMNS.index("designations_7d")
    assert features[10, week] == designations[4:11].sum()
    assert features[2, week] == designations[:3].sum()
    assert NOAA_WINDOW_COLUMN not in FEMAActivityLoader.FEATURE_COLUMNS


def test_the_labels_are_declared_external() -> None:
    assert FEMAActivityLoader.LABEL_SOURCE == "expert_annotated"
    source, _why = LABEL_PROVENANCE_REGISTRY["fema_activity_loader.FEMAActivityLoader"]
    assert source == "expert_annotated"
    assert [event["event_id"] for event in FEMAActivityLoader().list_events()] == [
        f"tropical_declarations_{year}" for year in range(2017, 2025)
    ]


def test_there_is_no_stand_in_when_a_source_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with the synthetic opt-in: an external label cannot be invented."""
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")

    def _unreachable(*_args: Any, **_kwargs: Any) -> bytes:
        raise ConnectionError("simulated: network unreachable")

    monkeypatch.setattr(_GET_BYTES, _unreachable)
    loader = _loader(tmp_path)

    with pytest.raises(DataSourceUnavailableError, match="NOAA billion-dollar catalogue"):
        loader.fetch_historical("tropical_declarations_2023")
    assert list(tmp_path.glob("*.json")) == []
    with pytest.raises(ValueError, match="Unknown event_id"):
        loader.fetch_historical("tropical_declarations_1999")


def test_the_catalogue_parser_refuses_another_file_and_skips_bad_rows() -> None:
    text = (
        "Weather and Climate Billion-Dollar Disasters\n"
        "Cost values are in millions of dollars\n"
        "Name,Disaster,Begin Date,End Date,CPI-Adjusted Cost,Unadjusted Cost,Deaths\n"
        '"Hurricane A (August 2023)",Tropical Cyclone,20230829,20230831,3559.7,3456,5\n'
        '"Bad dates",Flooding,2023,20230831,1,1,0\n'
        '"Ends before it begins",Flooding,20230902,20230901,1,1,0\n'
    )

    events = FEMAActivityLoader._parse_billion_dollar_events(text)

    assert events["name"].tolist() == ["Hurricane A (August 2023)"]
    assert events["ended"].iloc[0] == pd.Timestamp("2023-08-31", tz="UTC")
    with pytest.raises(ValueError, match="not the NOAA billion-dollar event list"):
        FEMAActivityLoader._parse_billion_dollar_events("<html>moved</html>")

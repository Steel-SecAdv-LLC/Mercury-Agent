# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``TornadoLoader.fetch_realtime`` reads only the tornado section of the SPC daily CSV.

The SPC daily file (``today.csv``) concatenates three sections -- tornado
(``Time,F_Scale,...``), wind (``Time,Speed,...``) and hail (``Time,Size,...``)
-- each introduced by its own header line. ``fetch_realtime`` handed the whole
body to ``pd.read_csv``, so on a day with no tornadoes it returned two
"tornado reports" that were really the wind and hail header lines, and on any
other day every wind and hail report was returned as a tornado.
``engineer_features`` then raised ``KeyError: 'len'``, because preliminary
daily reports carry no path length, width or casualty columns.

Fixtures: two small three-section bodies in the live ``today.csv`` layout
(inline below), plus the recorded real SPC file for 2012-06-29
(``tests/fixtures/meteorological/spc_storm_reports_20120629.csv``, provenance
in that directory's ``PROVENANCE.json``). All HTTP is patched to serve these
bytes; nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from omni_mercury_engine.loaders.tornado_loader import TornadoLoader

_RECORDED_DAILY_REPORTS = (
    Path(__file__).parents[1] / "fixtures" / "meteorological" / "spc_storm_reports_20120629.csv"
)

#: A day with tornado reports. The second tornado row carries an unquoted
#: comma inside the trailing Comments field, as SPC rows occasionally do.
_TORNADO_DAY = b"""Time,F_Scale,Location,County,State,Lat,Lon,Comments
2105,UNK,3 W ELKHORN,DOUGLAS,NE,41.28,-96.3,TORNADO CROSSED HIGHWAY 6. (OAX)
2132,UNK,BLAIR,WASHINGTON,NE,41.54,-96.13,ROOF DAMAGE TO SEVERAL HOMES, TREES DOWN. (OAX)
Time,Speed,Location,County,State,Lat,Lon,Comments
1850,65,2 N LINCOLN,LANCASTER,NE,40.84,-96.68,MEASURED AT AIRPORT. (OAX)
1915,UNK,WAHOO,SAUNDERS,NE,41.21,-96.62,LARGE TREE LIMBS DOWN. (OAX)
Time,Size,Location,County,State,Lat,Lon,Comments
1840,175,CRETE,SALINE,NE,40.62,-96.96,GOLF BALL SIZE HAIL. (OAX)
"""

#: A day with wind and hail but no tornado: the tornado section is its header only.
_QUIET_DAY = b"""Time,F_Scale,Location,County,State,Lat,Lon,Comments
Time,Speed,Location,County,State,Lat,Lon,Comments
1850,65,2 N LINCOLN,LANCASTER,NE,40.84,-96.68,MEASURED AT AIRPORT. (OAX)
Time,Size,Location,County,State,Lat,Lon,Comments
1840,175,CRETE,SALINE,NE,40.62,-96.96,GOLF BALL SIZE HAIL. (OAX)
"""


def test_only_the_tornado_section_is_returned(tmp_path: Path) -> None:
    """Was: 7 rows -- both tornadoes plus the wind/hail headers and their reports."""
    loader = TornadoLoader(cache_dir=tmp_path)
    with patch.object(TornadoLoader, "_fetch_url", return_value=_TORNADO_DAY):
        df = loader.fetch_realtime()

    assert len(df) == 2
    assert {"time", "mag", "location", "slat", "slon", "comments"} <= set(df.columns)
    assert df["location"].tolist() == ["3 W ELKHORN", "BLAIR"]
    assert df["slat"].tolist() == pytest.approx([41.28, 41.54])
    assert df["slon"].tolist() == pytest.approx([-96.3, -96.13])
    # The unquoted comma stayed inside Comments instead of shifting columns.
    assert df["comments"].iloc[1] == "ROOF DAMAGE TO SEVERAL HOMES, TREES DOWN. (OAX)"
    # Preliminary reports are unrated: "UNK" is an unknown rating, not EF0.
    assert df["mag"].isna().all()


def test_a_day_without_tornadoes_yields_no_reports(tmp_path: Path) -> None:
    """Was: 2 rows -- the wind and hail header lines presented as tornado reports."""
    loader = TornadoLoader(cache_dir=tmp_path)
    with patch.object(TornadoLoader, "_fetch_url", return_value=_QUIET_DAY):
        df = loader.fetch_realtime()

    assert len(df) == 0
    assert {"time", "mag", "slat", "slon"} <= set(df.columns)
    assert loader.engineer_features(df).shape == (0, len(TornadoLoader.FEATURE_COLUMNS))


def test_realtime_reports_engineer_without_archive_only_columns(tmp_path: Path) -> None:
    """Was: ``KeyError: 'len'`` -- daily reports have no path or casualty columns."""
    loader = TornadoLoader(cache_dir=tmp_path)
    with patch.object(TornadoLoader, "_fetch_url", return_value=_TORNADO_DAY):
        df = loader.fetch_realtime()

    features = loader.engineer_features(df)

    assert features.shape == (2, len(TornadoLoader.FEATURE_COLUMNS))
    assert np.isfinite(features).all()
    columns = TornadoLoader.FEATURE_COLUMNS
    assert features[:, columns.index("slat")].tolist() == pytest.approx([41.28, 41.54])
    # Hour comes from the report's HHMM time, not from a missing date column.
    assert features[:, columns.index("hour")].tolist() == [21.0, 21.0]
    for absent in ("path_length", "path_width", "fatalities", "injuries"):
        assert (features[:, columns.index(absent)] == 0.0).all()


def test_recorded_spc_file_yields_its_two_tornado_reports(tmp_path: Path) -> None:
    """The recorded 2012-06-29 file: 2 tornado rows among 675 wind and 49 hail rows.

    Its tornado header spells the rating column ``F-Scale`` (hyphen) where
    the live ``today.csv`` spells it ``F_Scale``; both must map to ``mag``.
    """
    loader = TornadoLoader(cache_dir=tmp_path)
    with patch.object(
        TornadoLoader, "_fetch_url", return_value=_RECORDED_DAILY_REPORTS.read_bytes()
    ):
        df = loader.fetch_realtime()

    assert len(df) == 2
    assert df["location"].tolist() == ["NEWCOMERSTOWN", "NEWCOMERSTOWN"]
    assert "mag" in df.columns
    assert loader.engineer_features(df).shape == (2, len(TornadoLoader.FEATURE_COLUMNS))


def test_crlf_line_endings_do_not_leak_into_values(tmp_path: Path) -> None:
    """A CRLF-terminated body parses to the same frame as an LF-terminated one."""
    loader = TornadoLoader(cache_dir=tmp_path)
    crlf = _TORNADO_DAY.replace(b"\n", b"\r\n")
    with patch.object(TornadoLoader, "_fetch_url", return_value=crlf):
        df = loader.fetch_realtime()

    assert len(df) == 2
    assert "comments" in df.columns
    assert df["comments"].iloc[0] == "TORNADO CROSSED HIGHWAY 6. (OAX)"


def test_missing_tornado_section_fails_loud(tmp_path: Path) -> None:
    """A feed without a tornado header is format drift, not an empty day."""
    hail_only = b"Time,Size,Location,County,State,Lat,Lon,Comments\n1840,175,X,Y,NE,40.6,-96.9,c\n"
    loader = TornadoLoader(cache_dir=tmp_path)
    with (
        patch.object(TornadoLoader, "_fetch_url", return_value=hail_only),
        pytest.raises(ValueError, match="no tornado section"),
    ):
        loader.fetch_realtime()

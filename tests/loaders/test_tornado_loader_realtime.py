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


# ---------------------------------------------------------------------------
# Ratings and the convective day.
# ---------------------------------------------------------------------------

_RATED_DAY = (
    b"Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
    b"2330,EF3,3 W ELKHORN,DOUGLAS,NE,41.28,-96.30,RATED BY SURVEY. (OAX)\n"
    b"0010,F2,BLAIR,WASHINGTON,NE,41.54,-96.13,RATED BY SURVEY. (OAX)\n"
    b"0020,UNK,ARLINGTON,WASHINGTON,NE,41.45,-96.35,PRELIMINARY. (OAX)\n"
)


def test_a_rated_report_keeps_its_rating() -> None:
    """Was: "EF3" coerced to NaN, then featurised as ef_scale 0 and labelled normal."""
    df = TornadoLoader._parse_daily_tornado_section(_RATED_DAY)

    assert df["mag"].iloc[:2].tolist() == [3.0, 2.0]
    assert np.isnan(df["mag"].iloc[2])


def test_reports_either_side_of_midnight_are_forty_minutes_apart() -> None:
    """Was: 2330 and 0010 both dated today, 23 h 20 min apart and out of order."""
    import pandas as pd

    from omni_mercury_engine.loaders.base import spc_convective_day_timestamps

    # Read at 03:00 UTC on the 15th: the file is the convective day that
    # began at 12:00 UTC on the 14th.
    now = pd.Timestamp("2026-11-15 03:00", tz="UTC")
    stamps = spc_convective_day_timestamps(["2330", "0010", 1200, "0259"], now=now)

    assert stamps == [
        pd.Timestamp("2026-11-14 23:30"),
        pd.Timestamp("2026-11-15 00:10"),
        pd.Timestamp("2026-11-14 12:00"),
        pd.Timestamp("2026-11-15 02:59"),
    ]
    assert stamps[1] - stamps[0] == pd.Timedelta(minutes=40)
    # Nothing that has been reported is dated after the moment it is read.
    assert stamps[0] <= now.tz_localize(None)

    # Read at 15:00 UTC the convective day began at 12:00 the same date.
    afternoon = pd.Timestamp("2026-11-15 15:00", tz="UTC")
    assert spc_convective_day_timestamps(["1330"], now=afternoon) == [
        pd.Timestamp("2026-11-15 13:30")
    ]


def test_a_value_that_is_not_a_time_falls_back_to_the_epoch() -> None:
    import pandas as pd

    from omni_mercury_engine.loaders.base import spc_convective_day_timestamps

    assert spc_convective_day_timestamps(["2460", "abc", None]) == [pd.Timestamp("1970-01-01")] * 3


def test_reports_across_midnight_cluster_in_the_tornado_and_hail_loaders() -> None:
    """Both loaders date the daily feed through the same convective-day rule."""
    import pandas as pd

    from omni_mercury_engine.loaders.hail_loader import HailLoader

    frame = pd.DataFrame({"time": ["2350", "0005"]})
    for loader in (TornadoLoader, HailLoader):
        first, second = loader._parse_timestamps(frame)
        assert second - first == pd.Timedelta(minutes=15)


def test_a_file_not_yet_rolled_over_is_dated_on_the_day_it_covers() -> None:
    """Just after 1200 UTC the clock says a new convective day; the rows say otherwise."""
    import pandas as pd

    from omni_mercury_engine.loaders.base import spc_convective_day_timestamps

    # 12:05 UTC on the 15th. A row at 2330 or 0010 cannot have happened yet in
    # the day that began five minutes ago, so the file is the previous day's.
    just_after = pd.Timestamp("2026-11-15 12:05", tz="UTC")
    assert spc_convective_day_timestamps(["2330", "0010", "1215"], now=just_after) == [
        pd.Timestamp("2026-11-14 23:30"),
        pd.Timestamp("2026-11-15 00:10"),
        pd.Timestamp("2026-11-14 12:15"),
    ]
    # A file that has rolled over holds only rows up to now: dated today.
    assert spc_convective_day_timestamps(["1201", "1203"], now=just_after) == [
        pd.Timestamp("2026-11-15 12:01"),
        pd.Timestamp("2026-11-15 12:03"),
    ]
    # A few minutes of clock skew is not a different day.
    assert spc_convective_day_timestamps(["1210"], now=just_after) == [
        pd.Timestamp("2026-11-15 12:10")
    ]
    # The epoch fallback for a value that is not a time is left alone.
    assert spc_convective_day_timestamps(["abc", "2330"], now=just_after) == [
        pd.Timestamp("1970-01-01"),
        pd.Timestamp("2026-11-14 23:30"),
    ]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The landslide loader reads the real catalog: its host, its fields, its words.

Read against the live catalog on 2026-11-17, the loader could not fetch at all
from a network without IPv6 (NASA's host publishes no IPv4 address), and what
it expected of a row did not match a real one:

* it read the kind of slide from ``event_category``; the catalog's field is
  ``landslide_category``, so that feature was zero on every real row;
* the catalog's largest size, ``catastrophic``, and its spelling of rockfall,
  ``rock_fall`` (942 rows), had no code, so both were encoded "unknown" and a
  catastrophic slide nobody died in was labelled normal;
* a date before September 2001 was read as seconds instead of milliseconds;
* "the most recent 2000" were the first 2000 by object id, 1974 to 2025.

Offline. Every row here is a recorded answer from the public ArcGIS Online
copy of the catalog (``tests/fixtures/landslide``), not a made-up one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.loaders import landslide_loader
from omni_mercury_engine.loaders.landslide_loader import (
    SOURCE_MIRROR,
    SOURCE_NASA,
    LandslideLoader,
)
from omni_mercury_engine.security.safe_http import UnsafeURLError

if TYPE_CHECKING:
    from collections.abc import Callable

_GET_BYTES = "omni_mercury_engine.loaders.base.SafeHTTPClient.get_bytes"
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "landslide"
_OSO = (_FIXTURES / "glc_mirror_oso_2014.json").read_bytes()
_SELECTED = (_FIXTURES / "glc_mirror_selected_rows.json").read_bytes()
_VOCABULARY = json.loads((_FIXTURES / "glc_vocabulary.json").read_text())["values"]

NASA_HOST = "maps.nccs.nasa.gov"
MIRROR_HOST = "services1.arcgis.com"


class Transport:
    """Stands in for the HTTP client: NASA's host unreachable, the mirror answering."""

    def __init__(self, mirror: bytes, nasa: bytes | None = None) -> None:
        self.mirror = mirror
        self.nasa = nasa
        self.asked: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, **kwargs: Any) -> bytes:
        self.asked.append((url, dict(kwargs.get("params") or {})))
        if NASA_HOST in url:
            if self.nasa is None:
                # What the client raises here, on a network without IPv6.
                raise UnsafeURLError(f"SafeHTTPClient: host '{NASA_HOST}' did not resolve")
            return self.nasa
        assert MIRROR_HOST in url, url
        return self.mirror

    def hosts(self) -> list[str]:
        return [NASA_HOST if NASA_HOST in url else MIRROR_HOST for url, _ in self.asked]


def _loader(tmp_path: Path) -> LandslideLoader:
    return LandslideLoader(cache_dir=tmp_path, max_retries=0)


def _with_full_names(payload: bytes) -> bytes:
    """The recorded rows under the catalog's full field names, as NASA's service has them."""
    full = {
        "landslide_": "landslide_category",
        "landslide1": "landslide_trigger",
        "landslid_1": "landslide_size",
        "fatality_c": "fatality_count",
        "injury_cou": "injury_count",
        "country_na": "country_name",
        "admin_divi": "admin_division_name",
    }
    body = json.loads(payload)
    for feature in body["features"]:
        feature["attributes"] = {full.get(k, k): v for k, v in feature["attributes"].items()}
    return json.dumps(body).encode()


def test_a_network_that_cannot_reach_nasa_reads_the_public_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: ``UnsafeURLError`` from the first request, and no data at all."""
    transport = Transport(mirror=_OSO)
    monkeypatch.setattr(_GET_BYTES, transport)

    frame = _loader(tmp_path).fetch_historical("oso_2014")

    assert transport.hosts() == [NASA_HOST, MIRROR_HOST]
    assert len(frame) == 24
    assert set(frame["catalog_source"]) == {SOURCE_MIRROR}


def test_nasa_is_read_when_it_answers_and_the_copy_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = Transport(mirror=b"{}", nasa=_with_full_names(_OSO))
    monkeypatch.setattr(_GET_BYTES, transport)

    frame = _loader(tmp_path).fetch_historical("oso_2014")

    assert transport.hosts() == [NASA_HOST]
    assert set(frame["catalog_source"]) == {SOURCE_NASA}
    assert "event_category" in frame.columns
    assert "landslide_category" not in frame.columns


def test_both_sources_give_the_same_columns_and_the_same_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_OSO))
    from_mirror = _loader(tmp_path / "a").fetch_historical("oso_2014")
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=b"{}", nasa=_with_full_names(_OSO)))
    from_nasa = _loader(tmp_path / "b").fetch_historical("oso_2014")

    wanted = {
        "event_category",
        "landslide_trigger",
        "landslide_size",
        "fatality_count",
        "injury_count",
        "country_name",
        "admin_division_name",
    }
    assert wanted <= set(from_mirror.columns)
    assert wanted <= set(from_nasa.columns)
    loader = _loader(tmp_path / "c")
    np.testing.assert_array_equal(
        loader.engineer_features(from_mirror), loader.engineer_features(from_nasa)
    )


def test_the_kind_of_slide_reaches_the_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: a column of zeros, because the catalog has no ``event_category`` field."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_OSO))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("oso_2014")
    features = loader.engineer_features(frame)

    assert sorted(set(frame["event_category"])) == ["landslide", "mudslide", "snow_avalanche"]
    assert set(features[:, 0]) == {1.0, 2.0, 8.0}
    assert (features[:, 3] > 0).any()  # the trigger
    assert (features[:, 7] > 0).all()  # the country


def test_the_oso_slide_is_in_its_own_window_and_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2014-03-22 slide that killed 43 is the one fatal row of the 24."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_OSO))
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("oso_2014")
    labels = loader.get_ground_truth("oso_2014")

    fatal = frame[frame["fatality_count"] > 0]
    assert list(fatal["event_date"].str[:10]) == ["2014-03-22"]
    assert list(fatal["fatality_count"]) == [43]
    assert labels[fatal.index[0]] == 1
    assert list(frame["event_date"]) == sorted(frame["event_date"])


def test_a_catastrophic_slide_is_the_largest_size_and_is_anomalous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: size code 0 ("unknown"), and labelled normal when nobody died."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_SELECTED))
    monkeypatch.setitem(
        landslide_loader._EVENT_CATALOG,
        "recorded_rows",
        {**landslide_loader._EVENT_CATALOG["oso_2014"], "name": "recorded rows"},
    )
    loader = _loader(tmp_path)

    frame = loader.fetch_historical("recorded_rows")
    features = loader.engineer_features(frame)
    labels = loader.get_ground_truth("recorded_rows")

    catastrophic = frame.index[frame["landslide_size"] == "catastrophic"]
    assert len(catastrophic) == 3
    assert set(features[catastrophic, 6]) == {5.0}
    assert features[:, 6].max() == 5.0
    nobody_died = [i for i in catastrophic if frame.loc[i, "fatality_count"] == 0]
    assert len(nobody_died) == 1
    assert labels[nobody_died[0]] == 1


def test_the_catalogs_own_spellings_are_understood(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rock_fall`` is rockfall; ``Riverbank collapse`` is ``riverbank_collapse``."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_SELECTED))
    loader = _loader(tmp_path)

    frame = loader.fetch_realtime()
    category = dict(zip(frame["event_category"], loader.engineer_features(frame)[:, 0]))

    assert category["rock_fall"] == landslide_loader._CATEGORY_ENCODING["rockfall"] == 3
    assert category["Riverbank collapse"] == 9
    assert category["earth_flow"] == 6


@pytest.mark.parametrize(
    ("field", "encoding"),
    [
        ("landslide_category", landslide_loader._CATEGORY_ENCODING),
        ("landslide_trigger", landslide_loader._TRIGGER_ENCODING),
        ("landslide_size", landslide_loader._SIZE_ENCODING),
    ],
)
def test_every_value_in_the_catalog_has_a_code(field: str, encoding: dict[str, int]) -> None:
    """Was: 1,384 categories (9.4%), 176 triggers and 54 sizes fell to "unknown".

    The values and their counts are the catalog's own, read 2026-11-17. Only
    what the catalog itself calls unknown, or leaves blank, may encode as 0.
    """
    import pandas as pd

    counts: dict[str, int] = _VOCABULARY[field]
    terms = landslide_loader._terms(pd.Series(list(counts)))
    unknown = {
        raw: n for raw, term, n in zip(counts, terms, counts.values()) if encoding.get(term, 0) == 0
    }
    assert sum(counts.values()) == 14750
    assert {raw.strip().lower() for raw in unknown} <= {"unknown", "unkown", ""}
    assert not [term for term in terms if term not in encoding]


def test_a_date_before_2001_is_read_as_the_day_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: milliseconds read as seconds -- 1979-08-08 became March of the year 11569."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_SELECTED))
    loader = _loader(tmp_path)

    frame = loader.fetch_realtime()
    features = loader.engineer_features(frame)
    row = frame.index[frame["country_name"] == "New Zealand"][0]

    assert frame.loc[row, "event_date"] == "1979-08-08T00:00:00"
    assert features[row, 8] == 8.0  # month
    assert features[row, 9] == 220.0  # day of year
    assert landslide_loader._parse_date(302918400000) == landslide_loader._parse_date("1979-08-08")


def test_a_row_with_no_date_stays_without_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: the text ``"None"``."""
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_SELECTED))

    frame = _loader(tmp_path).fetch_realtime()

    assert frame.loc[frame["country_name"] == "Greece", "event_date"].isna().all()


def test_the_most_recent_reports_are_asked_for_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Was: no order at all, so the service chose -- its first 2000 rows by object id."""
    transport = Transport(mirror=_SELECTED)
    monkeypatch.setattr(_GET_BYTES, transport)

    _loader(tmp_path).fetch_realtime()

    assert [params["orderByFields"] for _, params in transport.asked] == ["event_date DESC"] * 2
    assert {params["resultRecordCount"] for _, params in transport.asked} == {"2000"}


def test_an_error_answer_is_not_an_empty_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ArcGIS reports a bad query as HTTP 200 with an ``error`` body and no features."""
    refusal = json.dumps({"error": {"code": 400, "message": "Invalid query"}}).encode()
    transport = Transport(mirror=_OSO, nasa=refusal)
    monkeypatch.setattr(_GET_BYTES, transport)

    frame = _loader(tmp_path).fetch_historical("oso_2014")

    assert transport.hosts() == [NASA_HOST, MIRROR_HOST]
    assert len(frame) == 24


def test_when_no_source_answers_it_is_a_connection_error_that_names_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(url: str, **_kwargs: Any) -> bytes:
        raise ConnectionError(url)

    monkeypatch.setattr(_GET_BYTES, down)

    with pytest.raises(ConnectionError, match=f"{SOURCE_NASA}.*{SOURCE_MIRROR}"):
        _loader(tmp_path).fetch_realtime()
    assert not list(tmp_path.glob("*.json"))


def test_the_frame_is_cached_and_read_back_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GET_BYTES, Transport(mirror=_OSO))
    loader = _loader(tmp_path)
    first = loader.fetch_historical("oso_2014")

    refuse: Callable[..., bytes] = Transport(mirror=b"", nasa=b"")
    monkeypatch.setattr(_GET_BYTES, refuse)
    second = loader.fetch_historical("oso_2014")

    assert refuse.asked == []  # type: ignore[attr-defined]
    assert list(second.columns) == list(first.columns)
    np.testing.assert_array_equal(loader.engineer_features(first), loader.engineer_features(second))

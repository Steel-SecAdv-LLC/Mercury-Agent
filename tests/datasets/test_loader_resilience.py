# Copyright (C) 2025 Steel Security Advisors LLC
"""Mercury Agent — Loader resilience / failover tests.

Covers the corrective sweep that hardens the external-API loaders
(NSL-KDD, CICIDS, MITRE, FEMA, EPA, NOAA Storm Events / GSOD / ERDDAP).
All HTTP traffic is mocked: these tests assert behaviour at the
loader-orchestration layer (mirror failover, pagination, year fallback,
filename discovery), not real network availability.
"""

from __future__ import annotations

import gzip
import io
import json
import zipfile
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import requests

from omni_mercury_engine.datasets.base import DatasetConfig, http_get_with_retry
from omni_mercury_engine.datasets.disaster import FEMADisasterLoader
from omni_mercury_engine.datasets.epa_air import EPAAirQualityLoader
from omni_mercury_engine.datasets.noaa_erddap import NOAAERDDAPLoader
from omni_mercury_engine.datasets.noaa_gsod import NOAAGSODLoader
from omni_mercury_engine.datasets.noaa_storm import NOAAStormEventsLoader
from omni_mercury_engine.datasets.security import (
    CICIDSLoader,
    NSLKDDLoader,
    ThreatIntelLoader,
)


def _build_response(status: int, body: bytes = b"") -> requests.Response:
    """Construct a populated requests.Response for SafeHTTPClient mocks."""
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.url = "https://www.fema.gov/x"
    response.reason = "OK" if 200 <= status < 300 else "Error"
    return response


def _build_http_error(status: int) -> requests.HTTPError:
    """Construct a requests.HTTPError analogous to urllib's HTTPError."""
    return requests.HTTPError(response=_build_response(status))


# ---------------------------------------------------------------------------
# http_get_with_retry — shared helper
# ---------------------------------------------------------------------------


class TestHttpGetWithRetry:
    """Resilience tests for ``http_get_with_retry``.

    After the SafeHTTPClient migration the helper no longer touches
    ``urllib.request`` directly -- every retry goes through
    ``SafeHTTPClient.get_bytes`` which is built on ``requests``.  So
    the mocks patch ``SafeHTTPClient.get_bytes`` instead of
    ``urllib.request.urlopen``; the loader-orchestration semantics
    being verified (retry count, 4xx vs 5xx classification, scheme
    rejection) are unchanged.
    """

    def test_returns_body_on_first_success(self) -> None:
        payload = b"hello"

        # http_get_with_retry imports SafeHTTPClient at call time
        # (intentional, keeps the import surface minimal for the
        # core CLI), so we patch the canonical class location.
        with patch(
            "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
            return_value=payload,
        ) as get_bytes:
            body = http_get_with_retry("https://www.fema.gov/api/open/v2/x", retries=3)
        assert body == payload
        assert get_bytes.call_count == 1

    def test_retries_on_5xx_then_succeeds(self) -> None:
        payload = b"ok"
        err = _build_http_error(503)
        side_effects = [err, err, payload]

        with (
            patch(
                "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
                side_effect=side_effects,
            ) as get_bytes,
            patch("time.sleep"),
        ):
            body = http_get_with_retry("https://www.fema.gov/x", retries=3, backoff=1.0)
        assert body == payload
        assert get_bytes.call_count == 3

    def test_does_not_retry_on_404(self) -> None:
        err = _build_http_error(404)

        with (
            patch(
                "omni_mercury_engine.security.safe_http.SafeHTTPClient.get_bytes",
                side_effect=err,
            ) as get_bytes,
            pytest.raises(requests.HTTPError),
        ):
            http_get_with_retry("https://www.fema.gov/x", retries=3)
        assert get_bytes.call_count == 1

    def test_rejects_non_http_scheme(self) -> None:
        # SafeHTTPClient raises UnsafeURLError (a subclass of ValueError)
        # for non-http/https schemes; the wrapper preserves the raise.
        with pytest.raises(ValueError, match="scheme"):
            http_get_with_retry("file:///etc/passwd")


# ---------------------------------------------------------------------------
# FEMA disaster — datetime format + pagination
# ---------------------------------------------------------------------------


class TestFEMADisasterPagination:
    def test_paginates_with_skip_and_uses_iso_datetime(self, tmp_path: Any) -> None:
        config = DatasetConfig(
            name="fema_disaster",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            max_samples=2500,
            preprocessing={"year_range": (2020, 2024), "declaration_types": ["DR"]},
        )
        loader = FEMADisasterLoader(config)

        page = [
            {
                "disasterNumber": 4000 + i,
                "fipsStateCode": "06",
                "declarationDate": "2023-06-15T00:00:00.000Z",
                "incidentType": "Fire",
                "declarationType": "DR",
                "designatedArea": "Statewide",
                "ihProgramDeclared": True,
                "paProgramDeclared": True,
                "hmProgramDeclared": False,
            }
            for i in range(1000)
        ]
        last_page = page[:500]
        responses = [
            json.dumps({"DisasterDeclarationsSummaries": page}).encode("utf-8"),
            json.dumps({"DisasterDeclarationsSummaries": page}).encode("utf-8"),
            json.dumps({"DisasterDeclarationsSummaries": last_page}).encode("utf-8"),
        ]
        captured_urls: list[str] = []

        def fake_get(url: str, **kwargs: Any) -> bytes:
            captured_urls.append(url)
            return responses.pop(0)

        # disaster._download_from_fema does `from .base import http_get_with_retry`
        # at call time, so patching the source module is sufficient.
        with patch(
            "omni_mercury_engine.datasets.base.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is True
        # Three pages were fetched: $skip=0, 1000, 2000.
        assert len(captured_urls) == 3
        assert "%24skip=0" in captured_urls[0]
        assert "%24skip=1000" in captured_urls[1]
        assert "%24skip=2000" in captured_urls[2]
        # Datetime format must be ISO with millisecond precision and lowercase z.
        assert "2020-01-01T00%3A00%3A00.000z" in captured_urls[0]
        assert "2024-12-31T23%3A59%3A59.999z" in captured_urls[0]


# ---------------------------------------------------------------------------
# EPA — year fallback when requested year is not yet published
# ---------------------------------------------------------------------------


def _build_epa_zip(year: int) -> bytes:
    csv_text = (
        "State Code,County Code,Latitude,Longitude,Date Local,Arithmetic Mean,Observation Count\n"
        f"06,037,34.05,-118.25,{year}-06-15,42.5,24\n"
        f"06,037,34.05,-118.25,{year}-06-16,18.0,24\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"daily_88101_{year}.csv", csv_text)
    return buf.getvalue()


class TestEPAYearFallback:
    def test_falls_back_to_prior_year_on_404(self, tmp_path: Any) -> None:
        config = DatasetConfig(
            name="epa_air_quality",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"year": 2026},
        )
        loader = EPAAirQualityLoader(config)

        zip_bytes = _build_epa_zip(2024)

        def fake_get(url: str, **kwargs: Any) -> bytes:
            if "daily_88101_2024.zip" in url:
                return zip_bytes
            raise _build_http_error(404)

        with patch(
            "omni_mercury_engine.datasets.epa_air.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader.download()

        assert ok is True
        assert loader.year == 2024  # fell back from 2026
        cache = tmp_path / "data" / "epa_air_quality" / "epa_pm25_2024.npz"
        assert cache.exists()


# ---------------------------------------------------------------------------
# NOAA Storm Events — discover the correct compile-date filename
# ---------------------------------------------------------------------------


class TestNOAAStormFilenameDiscovery:
    def test_resolves_latest_compile_per_year(self, tmp_path: Any) -> None:
        config = DatasetConfig(
            name="noaa_storm_events",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"year_start": 2023, "year_end": 2023},
        )
        loader = NOAAStormEventsLoader(config)

        index_html = b"""
        <html><body>
        <a href="StormEvents_details-ftp_v1.0_d2023_c20240301.csv.gz">old</a>
        <a href="StormEvents_details-ftp_v1.0_d2023_c20240620.csv.gz">newer</a>
        <a href="StormEvents_details-ftp_v1.0_d2022_c20231120.csv.gz">2022</a>
        </body></html>
        """

        csv_text = (
            "EVENT_TYPE,STATE_FIPS,BEGIN_YEARMONTH,DAMAGE_PROPERTY,DAMAGE_CROPS,"
            "INJURIES_DIRECT,DEATHS_DIRECT,BEGIN_LAT,BEGIN_LON\n"
            "Tornado,40,202305,1500K,0,2,1,35.5,-97.5\n"
            "Flood,48,202306,500K,0,0,0,29.7,-95.3\n"
        )
        gz_payload = gzip.compress(csv_text.encode())

        def fake_get(url: str, **kwargs: Any) -> bytes:
            if url.endswith("csvfiles/"):
                return index_html
            if url.endswith("StormEvents_details-ftp_v1.0_d2023_c20240620.csv.gz"):
                return gz_payload
            raise AssertionError(f"unexpected URL: {url}")

        with patch(
            "omni_mercury_engine.datasets.noaa_storm.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader.download()

        assert ok is True
        cache = tmp_path / "data" / "noaa_storm_events" / "storm_events.npz"
        assert cache.exists()


# ---------------------------------------------------------------------------
# NSL-KDD — secondary mirror takes over on primary failure
# ---------------------------------------------------------------------------


class TestNSLKDDMirrorFailover:
    def test_secondary_mirror_used_when_primary_fails(self, tmp_path: Any) -> None:
        config = DatasetConfig(
            name="nsl-kdd",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"include_test": False},
        )
        loader = NSLKDDLoader(config)

        # Minimal NSL-KDD record (41 features + label + difficulty).
        nslkdd_row = (
            "0,tcp,http,SF,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,8,8,0.00,0.00,"
            "0.00,0.00,1.00,0.00,0.00,9,9,1.00,0.00,0.11,0.00,0.00,0.00,0.00,0.00,"
            "normal,20\n"
        )
        body = (nslkdd_row * 50).encode()

        primary_err = _build_http_error(429)

        calls: list[str] = []

        def fake_get(url: str, **kwargs: Any) -> bytes:
            calls.append(url)
            if url == NSLKDDLoader.NSLKDD_MIRRORS["train"][0]:
                raise primary_err
            return body

        with patch(
            "omni_mercury_engine.datasets.base.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader.download()

        assert ok is True
        assert loader.is_real_data is True
        assert calls[0] == NSLKDDLoader.NSLKDD_MIRRORS["train"][0]
        assert calls[1] == NSLKDDLoader.NSLKDD_MIRRORS["train"][1]


# ---------------------------------------------------------------------------
# MITRE ATT&CK — failover across STIX mirror list
# ---------------------------------------------------------------------------


class TestMITREMirrorFailover:
    def test_falls_back_to_secondary_stix_mirror(self, tmp_path: Any) -> None:
        config = DatasetConfig(
            name="threat-intel",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
        )
        loader = ThreatIntelLoader(config)

        bundle = {
            "objects": [
                {
                    "type": "attack-pattern",
                    "kill_chain_phases": [
                        {"phase_name": "execution"},
                        {"phase_name": "persistence"},
                    ],
                    "x_mitre_platforms": ["Windows", "Linux", "macOS"],
                    "x_mitre_data_sources": ["Process: Creation"],
                    "x_mitre_is_subtechnique": False,
                }
            ]
        }
        body = json.dumps(bundle).encode()

        calls: list[str] = []

        def fake_get(url: str, **kwargs: Any) -> bytes:
            calls.append(url)
            if url == ThreatIntelLoader.MITRE_STIX_MIRRORS[0]:
                raise _build_http_error(404)
            return body

        with patch(
            "omni_mercury_engine.datasets.base.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader._download_from_mitre()

        assert ok is True
        assert loader.is_real_data is True
        assert calls[0] == ThreatIntelLoader.MITRE_STIX_MIRRORS[0]
        assert calls[1] == ThreatIntelLoader.MITRE_STIX_MIRRORS[1]


# ---------------------------------------------------------------------------
# Static surface checks — keep DATA_SOURCES contract intact
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_cicids_data_sources_keys_unchanged(self) -> None:
        assert set(CICIDSLoader.DATA_SOURCES.keys()) == {
            "huggingface",
            "distrinet",
            "cic_official",
        }

    def test_cicids_huggingface_mirrors_include_primary(self) -> None:
        assert "bvk/CICIDS-2017" in CICIDSLoader.HUGGINGFACE_MIRRORS

    def test_erddap_ssh_offsets_monotonically_increasing(self) -> None:
        offsets = NOAAERDDAPLoader._SSH_DATE_OFFSET_DAYS_FALLBACK
        assert list(offsets) == sorted(offsets)
        assert offsets[0] >= 1

    def test_gsod_year_fallback_range_positive(self) -> None:
        assert NOAAGSODLoader._YEAR_FALLBACK_RANGE >= 1


# ---------------------------------------------------------------------------
# Wave 2: safe_urlretrieve, ADBench, USGS, NASA FIRMS retry / failover
# ---------------------------------------------------------------------------


class TestSafeUrlretrieveRetry:
    def test_safe_urlretrieve_inherits_retry_and_writes_file(self, tmp_path: Any) -> None:
        from omni_mercury_engine.datasets.base import safe_urlretrieve

        payload = b"\x89\x50\x4e\x47fake-binary"  # arbitrary bytes
        target = tmp_path / "out" / "x.bin"

        with patch(
            "omni_mercury_engine.datasets.base.http_get_with_retry",
            return_value=payload,
        ):
            safe_urlretrieve("https://raw.githubusercontent.com/foo/bar.bin", target)

        assert target.read_bytes() == payload


class TestADBenchRetry:
    def test_adbench_uses_retry_helper(self, tmp_path: Any) -> None:
        from omni_mercury_engine.datasets.adbench import ADBenchLoader

        config = DatasetConfig(
            name="adbench",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"dataset": "fraud"},
        )
        loader = ADBenchLoader(config)

        # Build a minimal valid NPZ payload with X and y keys.
        buf = io.BytesIO()
        np.savez(buf, X=np.zeros((4, 3), dtype=np.float32), y=np.array([0, 1, 0, 1]))
        npz_bytes = buf.getvalue()

        called: list[str] = []

        def fake_get(url: str, **kw: Any) -> bytes:
            called.append(url)
            return npz_bytes

        with patch(
            "omni_mercury_engine.datasets.adbench.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader.download()

        assert ok is True
        assert len(called) == 1
        assert called[0].endswith("13_fraud.npz")  # ADBench fraud is index 13


class TestUSGSEarthquakeUTC:
    def test_usgs_uses_utc_now(self, tmp_path: Any) -> None:
        from omni_mercury_engine.datasets.environmental import USGSEarthquakeLoader

        config = DatasetConfig(
            name="earthquake",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"min_magnitude": 4.0, "days_back": 7},
            max_samples=50,
        )
        loader = USGSEarthquakeLoader(config)

        # Minimal valid GeoJSON FeatureCollection with one earthquake.
        payload = {
            "features": [
                {
                    "geometry": {"coordinates": [-118.5, 34.1, 12.5]},
                    "properties": {
                        "mag": 5.4,
                        "gap": 80,
                        "dmin": 0.4,
                        "rms": 0.3,
                        "nst": 22,
                        "horizontalError": 1.5,
                        "depthError": 3.0,
                        "magError": 0.1,
                    },
                }
            ]
        }
        body = json.dumps(payload).encode()

        captured: list[str] = []

        def fake_get(url: str, **kw: Any) -> bytes:
            captured.append(url)
            return body

        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader._download_from_usgs()

        assert ok is True
        assert len(captured) == 1
        # Date strings are deterministic per UTC; just sanity-check the params
        assert "starttime=" in captured[0]
        assert "minmagnitude=4.0" in captured[0]
        assert "limit=50" in captured[0]


class TestNASAFIRMSMirrorFailover:
    def test_firms_falls_back_to_alternate_archive(self, tmp_path: Any) -> None:
        from omni_mercury_engine.datasets.environmental import (
            WildfireDataLoader,
        )

        config = DatasetConfig(
            name="wildfire",
            data_dir=str(tmp_path / "data"),
            cache_dir=str(tmp_path / "cache"),
            preprocessing={"source": "modis_7d"},
            max_samples=50,
        )
        loader = WildfireDataLoader(config)

        firms_csv = (
            b"latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,"
            b"instrument,confidence,version,bright_t31,frp,daynight\n"
            b"34.05,-118.25,310.0,1.2,1.0,2024-06-15,1235,Aqua,MODIS,80,6.1NRT,"
            b"295.5,12.5,D\n"
            b"41.50,-122.30,330.0,1.0,1.0,2024-06-15,2010,Terra,MODIS,90,6.1NRT,"
            b"300.0,18.7,N\n"
        )

        primary = loader.FIRMS_URLS["modis_7d"]
        called: list[str] = []

        def fake_get(url: str, **kw: Any) -> bytes:
            called.append(url)
            if url == primary:
                raise _build_http_error(503)
            return firms_csv

        with patch(
            "omni_mercury_engine.datasets.environmental.http_get_with_retry",
            side_effect=fake_get,
        ):
            ok = loader._download_from_firms()

        assert ok is True
        assert called[0] == primary
        assert called[1] in loader.FIRMS_URLS.values()

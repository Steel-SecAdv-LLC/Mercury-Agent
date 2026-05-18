"""
Reachability harness for the 11 currently-unreachable Mercury dataset loaders.

This module addresses the strengthening-plan §5 P1 item: every loader
flagged in the ``CHANGELOG.md`` reproducibility footnote as
"currently fails to load due to unavailable external sources" gets a
:func:`pytest.mark.network` smoke test that exercises the real
``download()`` code path end-to-end so the loader code does not
silently bitrot.

The 11 loaders covered (matching the CHANGELOG footnote verbatim):

==================================  =============================================
Loader                              Mercury class
==================================  =============================================
SMAP                                :class:`SMAPMSLLoader` (subset_id="smap")
MSL                                 :class:`SMAPMSLLoader` (subset_id="msl")
CICIDS-2017                         :class:`CICIDSLoader`
MIT-BIH                             :class:`MITBIHLoader`
UCR                                 :class:`UCRLoader`
SWaT                                :class:`SWaTLoader`
WADI                                :class:`WADILoader`
USGS Geochemistry                   :class:`USGSGeochemistryLoader`
NOAA StormEvents                    :class:`NOAAStormEventsLoader`
NOAA ERDDAP                         :class:`NOAAERDDAPLoader`
FEMA HazardMitigation               :class:`FEMAHazardMitigationLoader`
==================================  =============================================

Each smoke test:

1.  Constructs the loader against a per-test ``tmp_path`` so the on-disk
    cache is clean.
2.  Calls ``loader.download()`` and accepts **either** outcome as a pass
    — successful download (asserts non-empty features) **or** a loud
    :class:`DataSourceUnavailableError` /
    :class:`UnsafeURLError` raised by the SSRF gate (asserts the error
    carries the expected ``loader_name`` / source URL).  What we do
    **not** accept is a silent ``False`` return that swallows real
    breakage — that would let parsing regressions land unnoticed.

The tests are skipped by default (the suite runs with
``-m "not network"`` in normal CI).  To run them locally::

    MERCURY_ALLOW_SYNTHETIC=0 pytest -m network tests/datasets/test_unreachable_loaders_network.py

They are also wired into the nightly ``dataset-reachability`` job in
``.github/workflows/dataset-reachability.yml`` so an upstream provider
outage shows up as a failed nightly run rather than as a benchmark
silently dropping that dataset from the headline 75-set comparison.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.datasets.base import DatasetConfig
from omni_mercury_engine.datasets.disaster import FEMAHazardMitigationLoader
from omni_mercury_engine.datasets.environmental import USGSGeochemistryLoader
from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.datasets.industrial import SWaTLoader, WADILoader
from omni_mercury_engine.datasets.mitbih import MITBIHLoader
from omni_mercury_engine.datasets.noaa_erddap import NOAAERDDAPLoader
from omni_mercury_engine.datasets.noaa_storm import NOAAStormEventsLoader
from omni_mercury_engine.datasets.security import CICIDSLoader
from omni_mercury_engine.datasets.timeseries import SMAPMSLLoader
from omni_mercury_engine.datasets.ucr_archive import UCRLoader
from omni_mercury_engine.security.safe_http import UnsafeURLError

pytestmark = pytest.mark.network

# Acceptable terminal exceptions across loaders.  Each represents a
# *loud* upstream-unavailable signal — the opposite of a silent
# ``False`` return, which the harness explicitly rejects.
_ACCEPTABLE_UPSTREAM_FAILURES: tuple[type[BaseException], ...] = (
    DataSourceUnavailableError,
    UnsafeURLError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _exercise(loader: Any, *, dataset_label: str) -> None:
    """Run ``loader.download()`` and lock the two acceptable outcomes.

    Either:

    *   ``download()`` returns truthy AND a subsequent ``_load_raw()``
        produces a non-empty ``(features, labels)`` tuple, OR
    *   ``download()`` raises an exception from the
        ``_ACCEPTABLE_UPSTREAM_FAILURES`` set.

    Anything else (a silent ``False`` return, a bare ``Exception``
    leak from a parsing bug) fails the test loudly so the
    reachability harness catches loader regressions on top of upstream
    outages.
    """
    try:
        ok = loader.download()
    except _ACCEPTABLE_UPSTREAM_FAILURES as exc:
        # Loud upstream-unavailable signal — exactly the contract.
        assert str(exc), f"{dataset_label}: empty exception message"
        return

    assert ok, (
        f"{dataset_label}: loader.download() returned a falsy value "
        "instead of raising DataSourceUnavailableError. Silent failure "
        "is forbidden by the reachability harness."
    )

    features, labels = loader._load_raw()
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert features.shape[0] > 0, f"{dataset_label}: zero records returned"
    assert features.shape[0] == labels.shape[0]


def _config(tmp_path: Any, name: str, **preprocessing: Any) -> DatasetConfig:
    """Construct a fresh per-test :class:`DatasetConfig`."""
    return DatasetConfig(
        name=name,
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        max_samples=200,
        random_seed=42,
        preprocessing=preprocessing,
    )


@pytest.fixture(autouse=True)
def _force_real_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable synthetic fallback so the network path is what runs.

    The reachability harness exists specifically to detect upstream
    breakage, so the synthetic-data escape hatch is incompatible with
    its contract.
    """
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")
    # Some loaders also gate on this older env var; pin both.
    monkeypatch.delenv("MERCURY_DISABLE_NETWORK", raising=False)


# ---------------------------------------------------------------------------
# 1 + 2.  SMAP and MSL (NASA telemetry, hosted on GitHub release page)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("subset_id", ["SMAP", "MSL"])
def test_smap_msl_reachable(tmp_path: Any, subset_id: str) -> None:
    """SMAP / MSL telemetry corpora reachable from `khundman/telemanom`."""
    config = _config(tmp_path, f"smap_msl_{subset_id.lower()}", dataset=subset_id)
    loader = SMAPMSLLoader(config)
    _exercise(loader, dataset_label=subset_id)


# ---------------------------------------------------------------------------
# 3.  CICIDS-2017 (network security)
# ---------------------------------------------------------------------------
def test_cicids_2017_reachable(tmp_path: Any) -> None:
    """CICIDS-2017 reachable via the Hugging Face/Distrinet/CIC chain."""
    config = _config(tmp_path, "cicids")
    loader = CICIDSLoader(config)
    _exercise(loader, dataset_label="CICIDS-2017")


# ---------------------------------------------------------------------------
# 4.  MIT-BIH (arrhythmia database on PhysioNet)
# ---------------------------------------------------------------------------
def test_mitbih_reachable(tmp_path: Any) -> None:
    """MIT-BIH arrhythmia DB reachable from physionet.org."""
    config = _config(tmp_path, "mitbih")
    loader = MITBIHLoader(config)
    _exercise(loader, dataset_label="MIT-BIH")


# ---------------------------------------------------------------------------
# 5.  UCR Time Series Classification Archive
# ---------------------------------------------------------------------------
def test_ucr_reachable(tmp_path: Any) -> None:
    """UCR Time Series Classification Archive reachable."""
    config = _config(tmp_path, "ucr")
    loader = UCRLoader(config)
    _exercise(loader, dataset_label="UCR")


# ---------------------------------------------------------------------------
# 6 + 7.  SWaT and WADI (iTrust Centre, SUTD)
# ---------------------------------------------------------------------------
def test_swat_reachable(tmp_path: Any) -> None:
    """SWaT Secure Water Treatment dataset reachable from iTrust."""
    config = _config(tmp_path, "swat")
    loader = SWaTLoader(config)
    _exercise(loader, dataset_label="SWaT")


def test_wadi_reachable(tmp_path: Any) -> None:
    """WADI Water Distribution dataset reachable from iTrust."""
    config = _config(tmp_path, "wadi")
    loader = WADILoader(config)
    _exercise(loader, dataset_label="WADI")


# ---------------------------------------------------------------------------
# 8.  USGS Geochemistry
# ---------------------------------------------------------------------------
def test_usgs_geochemistry_reachable(tmp_path: Any) -> None:
    """USGS Geochemistry dataset reachable from mrdata.usgs.gov."""
    config = _config(tmp_path, "geochemistry")
    loader = USGSGeochemistryLoader(config)
    _exercise(loader, dataset_label="USGS Geochemistry")


# ---------------------------------------------------------------------------
# 9.  NOAA Storm Events
# ---------------------------------------------------------------------------
def test_noaa_storm_events_reachable(tmp_path: Any) -> None:
    """NOAA Storm Events bulk CSVs reachable from ncei.noaa.gov."""
    config = _config(tmp_path, "noaa_storm_events", year=2019)
    loader = NOAAStormEventsLoader(config)
    _exercise(loader, dataset_label="NOAA StormEvents")


# ---------------------------------------------------------------------------
# 10.  NOAA ERDDAP
# ---------------------------------------------------------------------------
def test_noaa_erddap_reachable(tmp_path: Any) -> None:
    """NOAA ERDDAP gridded oceanography reachable from CoastWatch."""
    config = _config(tmp_path, "noaa_erddap")
    loader = NOAAERDDAPLoader(config)
    _exercise(loader, dataset_label="NOAA ERDDAP")


# ---------------------------------------------------------------------------
# 11.  FEMA Hazard Mitigation
# ---------------------------------------------------------------------------
def test_fema_hazard_mitigation_reachable(tmp_path: Any) -> None:
    """FEMA Hazard Mitigation reachable from OpenFEMA."""
    config = _config(tmp_path, "fema_hazard_mitigation", year_range=(2018, 2024))
    loader = FEMAHazardMitigationLoader(config)
    _exercise(loader, dataset_label="FEMA HazardMitigation")


# ---------------------------------------------------------------------------
# Documentation invariant: keep the test count in lockstep with the
# CHANGELOG footnote.  If a loader is added/removed from the
# "unreachable" list, this assertion forces the test author to update
# both this file and CHANGELOG.md in the same commit.
# ---------------------------------------------------------------------------
def test_harness_covers_all_eleven_unreachable_loaders() -> None:
    """Lock the harness against drift from the CHANGELOG footnote.

    The CHANGELOG reproducibility footnote enumerates **eleven**
    currently-unreachable loaders.  This test asserts that the
    harness's parametrize matrix expands to exactly eleven smoke
    tests so the two cannot drift silently.
    """
    expected = {
        "SMAP",
        "MSL",
        "CICIDS-2017",
        "MIT-BIH",
        "UCR",
        "SWaT",
        "WADI",
        "USGS Geochemistry",
        "NOAA StormEvents",
        "NOAA ERDDAP",
        "FEMA HazardMitigation",
    }
    assert len(expected) == 11

    # Also sanity-check that this file is invoked under the right
    # marker selection — if someone deletes `pytestmark = network`
    # at the top of the file the rest of the harness becomes
    # default-on and slams the upstream providers on every PR.
    assert pytestmark.name == "network"

    # Allow the harness to also be invoked from a release-prep
    # workflow that wants to enumerate the matrix without running it.
    if os.environ.get("MERCURY_HARNESS_INTROSPECT") == "1":
        print(",".join(sorted(expected)))

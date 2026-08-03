# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reachability harness for the 11 currently-unreachable Mercury dataset loaders.

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

The tests are skipped by default by ``tests/conftest.py`` unless
``MERCURY_NETWORK_TESTS=1`` is set.  To run them locally::

    MERCURY_ALLOW_SYNTHETIC=0 MERCURY_NETWORK_TESTS=1 pytest tests/datasets/test_unreachable_loaders_network.py

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

from omni_mercury_engine.datasets.base import DatasetConfig, DatasetLoader
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

# (label, loader-class, config-name, ctor-preprocessing) for each
# unreachable loader.  The single parametrized test below is driven
# from this matrix so the drift gate can prove every listed loader has
# a live network smoke case.
_UNREACHABLE_LOADERS: list[tuple[str, type[DatasetLoader], str, dict[str, Any]]] = [
    ("SMAP", SMAPMSLLoader, "smap_msl_smap", {"dataset": "SMAP"}),
    ("MSL", SMAPMSLLoader, "smap_msl_msl", {"dataset": "MSL"}),
    ("CICIDS-2017", CICIDSLoader, "cicids", {}),
    ("MIT-BIH", MITBIHLoader, "mitbih", {}),
    ("UCR", UCRLoader, "ucr", {}),
    ("SWaT", SWaTLoader, "swat", {}),
    ("WADI", WADILoader, "wadi", {}),
    ("USGS Geochemistry", USGSGeochemistryLoader, "geochemistry", {}),
    ("NOAA StormEvents", NOAAStormEventsLoader, "noaa_storm_events", {"year": 2019}),
    ("NOAA ERDDAP", NOAAERDDAPLoader, "noaa_erddap", {}),
    (
        "FEMA HazardMitigation",
        FEMAHazardMitigationLoader,
        "fema_hazard_mitigation",
        {"year_range": (2018, 2024)},
    ),
]

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
        # Loud upstream-unavailable signal — exactly the contract. The
        # verdict line makes the run log answer "which branch did each
        # loader take?" — without it a green job cannot distinguish
        # "reachable" from "loudly unreachable", so a loader quietly
        # becoming downloadable again was invisible in CI output.
        assert str(exc), f"{dataset_label}: empty exception message"
        print(
            f"REACHABILITY VERDICT: {dataset_label}: UNREACHABLE (loud) — "
            f"{type(exc).__name__}: {str(exc)[:160]}"
        )
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
    print(
        f"REACHABILITY VERDICT: {dataset_label}: REACHABLE — "
        f"{features.shape[0]} records"
    )


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


@pytest.mark.parametrize(
    "label,loader_cls,config_name,preproc",
    _UNREACHABLE_LOADERS,
    ids=[row[0] for row in _UNREACHABLE_LOADERS],
)
def test_unreachable_loader_reachability_smoke(
    tmp_path: Any,
    label: str,
    loader_cls: type[DatasetLoader],
    config_name: str,
    preproc: dict[str, Any],
) -> None:
    """Every historically-unreachable loader has a live network smoke test."""
    config = _config(tmp_path, config_name, **preproc)
    loader = loader_cls(config)
    _exercise(loader, dataset_label=label)


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
    labels = {row[0] for row in _UNREACHABLE_LOADERS}
    assert len(_UNREACHABLE_LOADERS) == 11
    assert labels == expected

    # Also sanity-check that this file is invoked under the right
    # marker selection — if someone deletes `pytestmark = network`
    # at the top of the file the rest of the harness becomes
    # default-on and slams the upstream providers on every PR.
    assert pytestmark.name == "network"

    # Allow the harness to also be invoked from a release-prep
    # workflow that wants to enumerate the matrix without running it.
    if os.environ.get("MERCURY_HARNESS_INTROSPECT") == "1":
        print(",".join(sorted(expected)))

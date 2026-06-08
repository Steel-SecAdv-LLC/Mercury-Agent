# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline reachability harness for the 11 unreachable Mercury dataset loaders.

This is the default-CI counterpart to
:mod:`tests.datasets.test_unreachable_loaders_network`.  It does
**not** make any network calls.  Instead it exercises the parts of
each loader's code path that are independent of upstream
availability:

1.  Constructor accepts a valid :class:`DatasetConfig`.
2.  The dataset metadata constants (``DATASET_NAME``, ``DATASET_URL``,
    ``LICENSE``, ``CITATION``) are populated.
3.  Calling ``download()`` with the SafeHTTP layer monkeypatched to
    simulate an upstream outage **raises**
    :class:`DataSourceUnavailableError` — i.e. the loader fails
    loudly (the inverse of the silent-``False``-return regression the
    network harness was built to catch).

These offline tests run in every CI job and give us a stable code-
coverage floor for the 11-loader surface even when the upstream
providers are reachable, throttled, or paywalled.  They complement —
they do not replace — the ``@pytest.mark.network`` reachability lane.

See also:

*   ``tests/datasets/test_unreachable_loaders_network.py`` — opt-in
    network probes for the same 11 loaders.
*   ``CHANGELOG.md`` reproducibility footnote — the authoritative
    enumeration of the 11 loaders.
*   ``docs/DATASOURCES.md`` — operator-facing documentation of
    each loader's upstream provider and reachability story.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

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

# (label, loader-class, ctor-preprocessing) for each unreachable loader.
# Order matches the CHANGELOG footnote.
_UNREACHABLE_LOADERS: list[tuple[str, type[DatasetLoader], dict[str, Any]]] = [
    ("SMAP", SMAPMSLLoader, {"dataset": "SMAP"}),
    ("MSL", SMAPMSLLoader, {"dataset": "MSL"}),
    ("CICIDS-2017", CICIDSLoader, {}),
    ("MIT-BIH", MITBIHLoader, {}),
    ("UCR", UCRLoader, {}),
    ("SWaT", SWaTLoader, {}),
    ("WADI", WADILoader, {}),
    ("USGS Geochemistry", USGSGeochemistryLoader, {}),
    ("NOAA StormEvents", NOAAStormEventsLoader, {"year": 2019}),
    ("NOAA ERDDAP", NOAAERDDAPLoader, {}),
    ("FEMA HazardMitigation", FEMAHazardMitigationLoader, {"year_range": (2018, 2024)}),
]


def _config(tmp_path: Any, name: str, **preprocessing: Any) -> DatasetConfig:
    """Construct a per-test :class:`DatasetConfig`."""
    return DatasetConfig(
        name=name,
        data_dir=str(tmp_path / "data"),
        cache_dir=str(tmp_path / "cache"),
        max_samples=50,
        random_seed=42,
        preprocessing=preprocessing,
    )


@pytest.fixture(autouse=True)
def _force_real_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable synthetic fallback so the failure path is exercised.

    Without this, every loader's ``download()`` would fall through
    to the synthetic generator on outage and the offline harness
    would assert nothing useful about the real-data path.
    """
    monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "0")


@pytest.mark.parametrize(
    "label,loader_cls,preproc", _UNREACHABLE_LOADERS, ids=[t[0] for t in _UNREACHABLE_LOADERS]
)
def test_loader_constructs_cleanly(
    tmp_path: Any, label: str, loader_cls: type[DatasetLoader], preproc: dict[str, Any]
) -> None:
    """Loader constructs against a valid :class:`DatasetConfig`."""
    config = _config(tmp_path, loader_cls.DATASET_NAME, **preproc)
    loader = loader_cls(config)

    # Metadata invariants.
    assert loader.DATASET_NAME
    assert loader.DATASET_URL
    assert loader.LICENSE
    assert loader.CITATION
    # The data/cache paths must be created so ``download()`` can write.
    assert loader.data_path.exists()
    assert loader.cache_path.exists()


@pytest.mark.parametrize(
    "label,loader_cls,preproc", _UNREACHABLE_LOADERS, ids=[t[0] for t in _UNREACHABLE_LOADERS]
)
def test_loader_fails_loudly_on_simulated_outage(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    loader_cls: type[DatasetLoader],
    preproc: dict[str, Any],
) -> None:
    """Simulated upstream outage triggers :class:`DataSourceUnavailableError`.

    Patches the four HTTP entry points the loaders share
    (`requests.get`, `urllib.request.urlretrieve`, `SafeHTTPClient.get`,
    `SafeHTTPClient.get_bytes`) to raise :class:`ConnectionError`, then
    asserts ``download()`` either re-raises a loud exception or surfaces
    it via :class:`DataSourceUnavailableError`.  A silent ``False``
    return is treated as a regression.
    """
    import omni_mercury_engine.security.safe_http as safe_http_mod

    config = _config(tmp_path, loader_cls.DATASET_NAME, **preproc)
    loader = loader_cls(config)

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError(f"simulated upstream outage for {label}")

    # Patch every HTTP surface the unreachable loaders touch.  Some
    # loaders use `requests` directly, some use requests.Session through
    # third-party clients, some use the Mercury SafeHTTPClient wrapper,
    # and some still call `urllib.request`.
    with (
        patch("requests.get", side_effect=_boom),
        patch("requests.Session.request", side_effect=_boom),
        patch("requests.Session.get", side_effect=_boom),
        patch("urllib.request.urlopen", side_effect=_boom),
        patch("urllib.request.urlretrieve", side_effect=_boom),
        patch.object(safe_http_mod.SafeHTTPClient, "get", side_effect=_boom),
        patch.object(safe_http_mod.SafeHTTPClient, "get_bytes", side_effect=_boom),
        patch.object(safe_http_mod.SafeHTTPClient, "get_text", side_effect=_boom),
        patch.object(safe_http_mod.SafeHTTPClient, "get_json", side_effect=_boom),
    ):
        # Acceptable outcomes:
        #   1. ``DataSourceUnavailableError`` (the documented
        #      contract — what the loaders SHOULD raise),
        #   2. another loud exception (ConnectionError / OSError /
        #      RuntimeError) that the loader propagates.
        #
        # Forbidden outcomes:
        #   * Returning ``False`` silently,
        #   * Returning ``True`` without any data.
        try:
            result = loader.download()
        except DataSourceUnavailableError as exc:
            assert label.split(maxsplit=1)[0].lower() in str(exc).lower() or exc.loader_name
            return
        except (ConnectionError, OSError, RuntimeError) as exc:
            assert str(exc), f"{label}: empty exception message"
            return

        # A clean tmp_path cannot contain verified cached data.  Any
        # non-exceptional return means the loader swallowed the outage
        # or invented success, both of which violate the loud-failure
        # contract.
        pytest.fail(
            f"{label}: download() returned {result!r} despite every HTTP "
            "surface being patched to raise ConnectionError. This is the "
            "silent-failure regression the offline harness exists to catch."
        )


def test_harness_covers_eleven_loaders() -> None:
    """Coverage drift gate.

    Locks the offline harness against the same enumeration as the
    network harness and the ``CHANGELOG.md`` footnote.  If a loader
    is moved into or out of the unreachable set, both harnesses and
    the CHANGELOG must be updated in the same commit.
    """
    assert len(_UNREACHABLE_LOADERS) == 11
    labels = {row[0] for row in _UNREACHABLE_LOADERS}
    assert labels == {
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

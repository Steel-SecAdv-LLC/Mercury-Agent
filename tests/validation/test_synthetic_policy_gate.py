"""Regression suite for the synthetic-data policy gate.

Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

The validation-pipeline loaders historically exposed a ``use_synthetic``
boolean argument that, when set to ``True``, returned synthetic data
unconditionally -- bypassing the deployment-level
``MERCURY_ALLOW_SYNTHETIC`` policy enforced by
:func:`omni_mercury_engine.datasets.exceptions.check_synthetic_allowed`.

The fix in v1.7.0 routes every synthetic-data branch through the policy
gate so that a caller cannot opt out of policy:

* When ``MERCURY_ALLOW_SYNTHETIC=1`` is set, ``use_synthetic=True`` is
  honoured (the legacy contract for tests and explicit fallback chains).
* When ``MERCURY_ALLOW_SYNTHETIC`` is unset or ``0``, every loader that
  would otherwise return synthetic data raises
  :class:`~omni_mercury_engine.datasets.exceptions.DataSourceUnavailableError`
  -- including the caller-flag path.

This file locks the closure with a parametrised regression that exercises
every concrete loader on the bypass surface.  Without these tests, a
future refactor could silently reintroduce the bypass and Mercury Agent
would deliver synthetic data to a deployment that has explicitly
forbidden it -- a particularly dangerous regression for the humanitarian
crisis-response and missing-persons workloads Mercury Agent serves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.datasets.exceptions import DataSourceUnavailableError
from omni_mercury_engine.validation.data_loaders import (
    MIMICLoader,
    NOAAHurricaneLoader,
    NOAAOceanLoader,
    NOAASpaceWeatherLoader,
    NSLKDDLoader,
    USGSEarthquakeLoader,
)

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disable_synthetic_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the ``MERCURY_ALLOW_SYNTHETIC`` env var for the test scope.

    ``tests/conftest.py`` opts every test into synthetic data by default
    (``setdefault("MERCURY_ALLOW_SYNTHETIC", "1")``) because the wider
    suite exercises offline synthetic-fallback paths.  These regression
    tests deliberately need the *deny* posture to prove the gate fires,
    so we delete the variable rather than rely on ``setdefault``.
    """
    monkeypatch.delenv("MERCURY_ALLOW_SYNTHETIC", raising=False)


# ---------------------------------------------------------------------------
# Bypass-closure regression: caller flag without policy must raise
# ---------------------------------------------------------------------------


class TestCallerFlagCannotBypassPolicy:
    """Each loader must raise when policy is off, even if caller opts in.

    The closure under test is the addition of
    ``check_synthetic_allowed(...)`` immediately before every
    ``_generate_synthetic`` invocation in
    :mod:`omni_mercury_engine.validation.data_loaders`.  This locks the
    contract that synthetic data is *policy-gated*, not *caller-gated*.
    """

    def test_nsl_kdd_caller_flag_raises_without_policy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = NSLKDDLoader(cache_dir=tmp_path)
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "NSL-KDD" in str(err)
            assert "use_synthetic=True" in str(err) or "Caller" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError(
                "use_synthetic=True must not bypass the MERCURY_ALLOW_SYNTHETIC policy"
            )

    def test_usgs_earthquake_caller_flag_raises_without_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = USGSEarthquakeLoader()
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "USGS" in str(err)
            assert "use_synthetic=True" in str(err) or "Caller" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError(
                "use_synthetic=True must not bypass the MERCURY_ALLOW_SYNTHETIC policy"
            )

    def test_mimic_caller_flag_raises_without_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = MIMICLoader()
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "MIMIC" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError("MIMIC-III is synthetic-only and must always gate through policy")

    def test_noaa_space_weather_caller_flag_raises_without_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = NOAASpaceWeatherLoader()
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "NOAA Space Weather" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError(
                "use_synthetic=True must not bypass the MERCURY_ALLOW_SYNTHETIC policy"
            )

    def test_noaa_hurricane_caller_flag_raises_without_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = NOAAHurricaneLoader()
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "NOAA Hurricane" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError(
                "use_synthetic=True must not bypass the MERCURY_ALLOW_SYNTHETIC policy"
            )

    def test_noaa_ocean_caller_flag_raises_without_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _disable_synthetic_policy(monkeypatch)
        loader = NOAAOceanLoader()
        try:
            loader.load(use_synthetic=True, n_samples=100)
        except DataSourceUnavailableError as err:
            assert "NOAA Ocean" in str(err)
        else:  # pragma: no cover - defensive
            raise AssertionError(
                "use_synthetic=True must not bypass the MERCURY_ALLOW_SYNTHETIC policy"
            )


# ---------------------------------------------------------------------------
# Forward-compatibility: when the policy is ON, the caller flag still works
# ---------------------------------------------------------------------------


class TestCallerFlagHonouredWhenPolicyOn:
    """The closure is additive, not subtractive: the legacy contract holds.

    These tests cover the case where the deployment has explicitly opted
    into synthetic fallback by setting ``MERCURY_ALLOW_SYNTHETIC=1``.  The
    caller flag must still produce synthetic data in that posture.
    """

    def test_nsl_kdd_caller_flag_honoured_with_policy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setenv("MERCURY_ALLOW_SYNTHETIC", "1")
        loader = NSLKDDLoader(cache_dir=tmp_path)
        data, labels, meta = loader.load(use_synthetic=True, n_samples=100)
        assert data.shape[0] == 100
        assert labels.shape[0] == 100
        assert meta.source == "synthetic"
        assert isinstance(data, np.ndarray)
        assert isinstance(labels, np.ndarray)

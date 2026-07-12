# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Keyed-loader live-data smoke test (CI-only, scheduled/dispatch).

For each Mercury loader that authenticates with an API key, perform ONE real
fetch against the live endpoint and assert real rows/records come back. This is
what makes the wired GitHub Actions secrets *effective*: it proves each secret
reaches its consuming loader and that the loader returns non-empty real data.

Honesty rules:

* A loader whose key env var is unset is reported ``SKIP`` (the secret is not
  configured in this environment) -- never a silent pass.
* A keyed-and-configured loader that returns empty or raises is a ``FAIL``; the
  job exits non-zero so live-wiring drift is caught.
* No key material is ever printed -- only the env-var *names* and row counts.

This never runs on the PR gate (network-dependent); the workflow schedules it
weekly and exposes ``workflow_dispatch`` for on-demand verification.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

Results = list[tuple[str, str, str]]
_RESULTS: Results = []


def _record(name: str, status: str, detail: str) -> None:
    """Record and print one loader's smoke result (no key material)."""
    _RESULTS.append((name, status, detail))
    print(f"[{status:4s}] {name}: {detail}", flush=True)


def _key_present(*env_vars: str) -> bool:
    """True when at least one of ``env_vars`` is set and non-empty."""
    return any(os.environ.get(v, "").strip() for v in env_vars)


def _smoke(
    name: str, key_vars: tuple[str, ...], fetch: Callable[[], int], min_rows: int = 1
) -> None:
    """Run one loader smoke: SKIP if unkeyed, else fetch and assert rows."""
    if not _key_present(*key_vars):
        _record(name, "SKIP", f"none of {list(key_vars)} set in this environment")
        return
    try:
        n = fetch()
    except Exception as exc:
        _record(name, "FAIL", f"{type(exc).__name__}: {exc}")
        return
    if n < min_rows:
        _record(name, "FAIL", f"returned {n} rows (< {min_rows}); endpoint reachable but empty")
    else:
        _record(name, "OK", f"returned {n} real rows from the live endpoint")


def _domain_rows(cls: type, **kwargs: Any) -> int:
    """Instantiate a BaseDomainLoader and count fetch_realtime() rows."""
    loader = cls(cache_dir=Path(tempfile.mkdtemp()), **kwargs)
    df = loader.fetch_realtime()
    return len(df)


def _alpha_vantage_ok() -> int:
    """Fetch one real Alpha Vantage quote; return 1 on a valid quote."""
    from omni_mercury_engine.integrations.stubs.financial import (
        FinancialAPIProvider,
        FinancialService,
    )

    svc = FinancialService(provider=FinancialAPIProvider.ALPHA_VANTAGE)
    price = asyncio.run(svc.get_price("MSFT"))
    return 1 if price is not None and float(getattr(price, "price", 0.0)) > 0.0 else 0


def _openweather_ok() -> int:
    """Fetch one real OpenWeatherMap current-conditions record; 1 on success."""
    from omni_mercury_engine.integrations.stubs.weather import (
        WeatherAPIProvider,
        WeatherService,
    )

    svc = WeatherService(provider=WeatherAPIProvider.OPENWEATHERMAP)
    data = asyncio.run(svc.get_current("London,GB"))
    return 1 if data is not None else 0


def _usgs_eros_ok() -> int:
    """Authenticate to USGS EROS M2M and dataset-search Landsat; row count.

    Proves BOTH ``USGS_KEY`` (application token) and ``EROSERS_USERNAME`` reach
    the client and that ``login-token`` + ``dataset-search`` return real records.
    Logs out (invalidating the API key) regardless of outcome; no key material
    is ever printed.
    """
    from omni_mercury_engine.integrations.usgs_eros import USGSErosM2MClient

    client = USGSErosM2MClient()
    try:
        datasets = client.dataset_search("landsat")
    finally:
        client.logout()
    return len(datasets)


def main() -> int:
    """Run every keyed-loader smoke and return a process exit code."""
    from omni_mercury_engine.loaders.energy_loader import EnergyLoader
    from omni_mercury_engine.loaders.financial_loader import FinancialLoader
    from omni_mercury_engine.loaders.meteor_loader import MeteorLoader
    from omni_mercury_engine.loaders.space_weather_loader import SpaceWeatherLoader
    from omni_mercury_engine.loaders.wildfire_loader import WildfireLoader

    _smoke(
        "wildfire (NASA FIRMS area API)",
        ("NASA_FIRMS_MAP_KEY", "FIRMS_MAP_KEY"),
        lambda: _domain_rows(WildfireLoader),
    )
    _smoke("energy (EIA electricity)", ("EIA_API_KEY",), lambda: _domain_rows(EnergyLoader))
    _smoke(
        "financial (FRED economic series)", ("FRED_API_KEY",), lambda: _domain_rows(FinancialLoader)
    )
    _smoke(
        "space_weather (NASA DONKI geomagnetic storms)",
        ("NASA_API_KEY",),
        lambda: _domain_rows(SpaceWeatherLoader),
    )
    _smoke(
        "meteor (NASA NeoWs close approaches)",
        ("NASA_API_KEY",),
        lambda: _domain_rows(MeteorLoader),
    )
    _smoke("financial market (Alpha Vantage quote)", ("ALPHA_VANTAGE_API_KEY",), _alpha_vantage_ok)
    _smoke("weather (OpenWeatherMap current)", ("OPENWEATHERMAP_API_KEY",), _openweather_ok)
    _smoke(
        "usgs_eros (M2M Landsat dataset search)",
        ("USGS_KEY", "EROSERS_USERNAME"),
        _usgs_eros_ok,
    )

    ok = sum(1 for _, s, _ in _RESULTS if s == "OK")
    fail = sum(1 for _, s, _ in _RESULTS if s == "FAIL")
    skip = sum(1 for _, s, _ in _RESULTS if s == "SKIP")
    print(f"\nlive-data smoke: {ok} OK, {fail} FAIL, {skip} SKIP", flush=True)
    if fail:
        print("FAILED keyed loaders:", flush=True)
        for name, status, detail in _RESULTS:
            if status == "FAIL":
                print(f"  - {name}: {detail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

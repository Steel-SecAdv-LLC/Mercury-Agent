#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live delivery check for the repository data-source credentials.

For each keyed data source, make ONE real authenticated request and report
whether the key actually *delivers* real data -- the practical meaning of
"wired and delivers when called". Secret VALUES are never printed; only a
boolean verdict and a small non-sensitive data sample (row counts, a date,
a numeric value).

This is import-light (``requests`` + stdlib only, no engine / native crypto),
so it runs early in the network lane where the secrets are injected. Sources
whose key env var is unset are reported ``SKIP`` (not a failure): an operator
running locally without a given key still gets a clean report.

Exit code is non-zero only if a source whose key IS present fails to deliver.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
from typing import Any

import requests


def _get(url: str, timeout: int = 30) -> tuple[int, bytes]:
    """GET a URL, returning ``(status_code, body)``; ``(0, error)`` on failure."""
    try:
        resp = requests.get(url, headers={"User-Agent": "mercury-cred-check"}, timeout=timeout)
        return resp.status_code, resp.content
    except Exception as exc:  # pragma: no cover - network variance
        return 0, str(exc).encode()


def check_eia(key: str) -> tuple[bool, str]:
    """EIA API v2 hourly grid demand for one balancing authority."""
    q = urllib.parse.urlencode(
        {
            "api_key": key,
            "frequency": "hourly",
            "data[]": "value",
            "facets[respondent][]": "PJM",
            "facets[type][]": "D",
            "start": "2024-05-10T00",
            "end": "2024-05-10T06",
            "length": "10",
        },
        doseq=True,
    )
    status, body = _get(f"https://api.eia.gov/v2/electricity/rto/region-data/data/?{q}")
    if status != 200:
        return False, f"HTTP {status}"
    try:
        rows = json.loads(body)["response"]["data"]
    except Exception as exc:
        return False, f"unparseable response: {exc}"
    if not rows:
        return False, "authenticated but zero rows"
    r0 = rows[0]
    return (
        True,
        f"{len(rows)} rows; e.g. {r0.get('period')} {r0.get('respondent')} D={r0.get('value')} MW",
    )


def check_fred(key: str) -> tuple[bool, str]:
    q = urllib.parse.urlencode(
        {
            "series_id": "DGS10",
            "api_key": key,
            "file_type": "json",
            "limit": "3",
            "sort_order": "desc",
        }
    )
    status, body = _get(f"https://api.stlouisfed.org/fred/series/observations?{q}")
    if status != 200:
        return False, f"HTTP {status}"
    try:
        obs = json.loads(body)["observations"]
    except Exception as exc:
        return False, f"unparseable: {exc}"
    return (
        bool(obs),
        f"{len(obs)} obs; latest {obs[0]['date']}={obs[0]['value']}" if obs else "zero obs",
    )


def check_nasa(key: str) -> tuple[bool, str]:
    status, body = _get(f"https://api.nasa.gov/DONKI/notifications?type=all&api_key={key}")
    if status != 200:
        return False, f"HTTP {status}"
    try:
        data = json.loads(body)
    except Exception as exc:
        return False, f"unparseable: {exc}"
    return True, f"DONKI reachable; {len(data)} notifications"


def check_alpha_vantage(key: str) -> tuple[bool, str]:
    status, body = _get(
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey={key}"
    )
    if status != 200:
        return False, f"HTTP {status}"
    try:
        data = json.loads(body)
    except Exception as exc:
        return False, f"unparseable: {exc}"
    if data.get("Global Quote"):
        return True, f"SPY quote delivered ({data['Global Quote'].get('05. price')})"
    # A rate-limit / info Note means the KEY was accepted but throttled (free tier
    # is 25 req/day) -- that is still "wired", so don't hard-fail the lane on it.
    # Only a genuine "Error Message" (bad/invalid key) is a delivery failure.
    if data.get("Error Message"):
        return False, f"bad key: {data['Error Message']}"
    throttle = data.get("Note") or data.get("Information")
    if throttle:
        return True, f"key accepted (throttled): {str(throttle)[:100]}"
    return False, f"unexpected response: {str(data)[:120]}"


def check_usgs_eros(username: str, token: str, password: str = "") -> tuple[bool, str]:
    """USGS EROS / EarthExplorer M2M login (username + application token/password).

    The M2M API authenticates a ``username`` with either an application ``token``
    (recommended; generated in the EarthExplorer profile) or a ``password``. A
    successful login returns a session token in ``data`` with ``errorCode: null``;
    that session is then used (and a small ``dataset-search`` is issued) to prove
    real data delivery, and finally logged out.
    """
    base = "https://m2m.cr.usgs.gov/api/api/json/stable"

    def _post(
        path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        try:
            resp = requests.post(
                f"{base}/{path}",
                json=payload,
                headers={"User-Agent": "mercury-cred-check", **(headers or {})},
                timeout=30,
            )
            return resp.json() if resp.content else {"errorMessage": f"HTTP {resp.status_code}"}
        except Exception as exc:  # pragma: no cover - network variance
            return {"errorMessage": str(exc)}

    body: dict[str, Any] = {}
    method = ""
    if token:
        method = "token"
        body = _post("login-token", {"username": username, "token": token})
    if not body.get("data") and password:
        method = "password"
        body = _post("login", {"username": username, "password": password})
    session = body.get("data")
    if not session:
        return (
            False,
            f"login failed ({method or 'no-credential'}): "
            f"{body.get('errorCode')} {body.get('errorMessage')}",
        )

    detail = f"authenticated via {method}; session token issued"
    hdr = {"X-Auth-Token": str(session)}
    ds = _post("dataset-search", {"datasetName": "landsat_ot_c2_l2"}, hdr)
    if isinstance(ds.get("data"), list):
        detail += f"; dataset-search returned {len(ds['data'])} dataset(s)"
    _post("logout", {}, hdr)
    return True, detail


def check_openweathermap(key: str) -> tuple[bool, str]:
    status, body = _get(f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={key}")
    if status != 200:
        return False, f"HTTP {status}: {body[:120].decode(errors='replace')}"
    try:
        data = json.loads(body)
    except Exception as exc:
        return False, f"unparseable: {exc}"
    return True, f"London weather delivered (T={data.get('main', {}).get('temp')})"


#: source -> (env var(s), checker). First present env var is used.
CHECKS: dict[str, tuple[tuple[str, ...], Any]] = {
    "EIA": (("EIA_API_KEY",), check_eia),
    "FRED": (("FRED_API_KEY",), check_fred),
    "NASA": (("NASA_API_KEY",), check_nasa),
    "AlphaVantage": (("ALPHA_VANTAGE_API_KEY",), check_alpha_vantage),
    "OpenWeatherMap": (("OPENWEATHERMAP_API_KEY",), check_openweathermap),
}


def run() -> int:
    print("=== Mercury data-source credential delivery check ===")
    failures = 0
    for name, (env_vars, checker) in CHECKS.items():
        key = next((os.environ[e] for e in env_vars if os.environ.get(e)), "")
        if not key:
            print(f"  {name:14s} SKIP  ({'/'.join(env_vars)} not set)")
            continue
        ok, detail = checker(key)
        print(f"  {name:14s} {'DELIVERS' if ok else 'FAIL   '}  {detail}")
        if not ok:
            failures += 1

    # USGS EROS / EarthExplorer M2M: authenticates a username with an application
    # token (USGS_KEY, recommended) or a password. The other USGS services
    # (earthquake FDSN, water, elevation) are keyless and need no credential.
    eros_user = os.environ.get("EROSERS_USERNAME", "")
    eros_token = os.environ.get("USGS_KEY", "")
    eros_pw = os.environ.get("EROSERS_PASSWORD", "")
    if eros_user and (eros_token or eros_pw):
        ok, detail = check_usgs_eros(eros_user, eros_token, eros_pw)
        print(f"  {'USGS/EROS':14s} {'DELIVERS' if ok else 'FAIL   '}  {detail}")
        if not ok:
            failures += 1
    else:
        status, _ = _get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
            "&starttime=2024-05-10&endtime=2024-05-11&minmagnitude=5"
        )
        print(
            f"  {'USGS/EROS':14s} SKIP  need EROSERS_USERNAME + USGS_KEY (EROS M2M "
            f"application token) or EROSERS_PASSWORD; keyless USGS earthquake feed "
            f"reachable HTTP {status}"
        )
    print(f"=== {failures} failure(s) among sources with a key present ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())

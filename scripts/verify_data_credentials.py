#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live delivery check for the repository data-source credentials.

For each keyed data source, make ONE real authenticated request and report
whether the key actually *delivers* real data -- the practical meaning of
"wired and delivers when called". Secret VALUES are never printed; only a
verdict and a small non-sensitive data sample (row counts, a date, a numeric
value).

This is import-light (``requests`` + stdlib only, no engine / native crypto),
so it runs early in the network lane where the secrets are injected. Sources
whose key env var is unset are reported ``SKIP`` (not a failure): an operator
running locally without a given key still gets a clean report.

Each checked source lands on one of three verdicts:

``DELIVERS``
    The credential was accepted and real data came back.
``FAIL``
    The provider rejected the credential, or accepted it and returned nothing --
    an actionable credential problem. **This is the only verdict that fails the
    lane.**
``UNREACH``
    The upstream could not answer at all (transport failure, 5xx, or a 429 rate
    limit). This says nothing about the credential, so it is reported and not
    counted against it.

The exception: if *every* source with a key present is ``UNREACH``, the run
verified nothing, and exiting 0 would be a green light with no evidence behind
it -- so that case fails too.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
from typing import Any

import requests

#: Query-string parameters that carry a credential value. ``requests`` transport
#: errors embed the full request URL, so a raw exception string can leak the key
#: (e.g. ``?api_key=SECRET``) into CI logs. :func:`_redact` strips these values
#: before any diagnostic is returned or printed, honouring this module's
#: "secret VALUES are never printed" contract.
_CREDENTIAL_QUERY_PARAM = re.compile(
    r"(?i)((?:api[_-]?key|app[_-]?id|appid|access[_-]?key|token|password|passwd|secret|key)=)"
    r"[^&\s\"'<>]+"
)

#: Env-var NAMES treated as credential-holding for the value-based pass.
#: Mirrors ``omni_mercury_engine.security.redaction._ENV_SECRET_NAME_RE`` —
#: duplicated deliberately: this script's contract is import-light
#: (``requests`` + stdlib, no engine import, because the engine's PQC gate
#: requires the native AMA build that is absent this early in the lane).
_ENV_SECRET_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|MAP_?KEY"
    r"|CREDENTIALS?|AUTH|BEARER|PRIVATE_?KEY|SIGNATURE)(?:$|_)",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    """Strip credential values out of *text* before it is printed.

    Two passes.  The query-parameter pass is structural (any value length,
    no knowledge of the secret needed).  The env-value pass catches keys
    that ride OUTSIDE a query string — NASA FIRMS embeds the MAP key as a
    URL *path segment*, which no query-shaped regex can see.  Values of
    4-7 characters are replaced only at non-alphanumeric boundaries so a
    short value cannot mangle unrelated words; below 4 the value is
    degenerate and skipped (the structural pass still covers it in query
    position).
    """
    text = _CREDENTIAL_QUERY_PARAM.sub(r"\1***", text)
    for name, raw in os.environ.items():
        value = raw.strip()
        if len(value) < 4 or not _ENV_SECRET_NAME.search(name):
            continue
        if len(value) >= 8:
            text = text.replace(value, f"<{name}:redacted>")
        else:
            text = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
                f"<{name}:redacted>",
                text,
            )
    return text


def _get(url: str, timeout: int = 30) -> tuple[int, bytes]:
    """GET a URL, returning ``(status_code, body)``; ``(0, error)`` on failure."""
    try:
        resp = requests.get(url, headers={"User-Agent": "mercury-cred-check"}, timeout=timeout)
        return resp.status_code, resp.content
    except Exception as exc:  # pragma: no cover - network variance
        return 0, str(exc).encode()


def _fail_detail(status: int, body: bytes) -> str:
    """Render a non-200 ``_get`` result into an actionable failure string.

    ``_get`` reports a transport failure (DNS/TLS/timeout/connection-reset) as
    ``(0, <exception text>)``: the ``0`` is not an HTTP status, and collapsing it
    to a bare ``"HTTP 0"`` throws away the one thing that explains the failure in
    CI. So on ``status == 0`` we surface the transport error text; on a real HTTP
    error we report the status plus a short body preview (which for these APIs is
    typically the provider's own error message, e.g. an invalid-key JSON).

    The text is passed through :func:`_redact` first: a ``requests`` transport
    error embeds the full request URL, whose query string carries the API key, so
    the raw string is scrubbed of credential values before it can reach a log.

    Args:
        status: The status code from :func:`_get` (``0`` means transport error).
        body: The response body, or the transport exception text when ``status``
            is ``0``.

    Returns:
        A single-line, non-sensitive diagnostic suitable for the CI report.
    """
    text = _redact(body.decode(errors="replace").strip())
    if status == 0:
        return f"transport error: {text[:200]}" if text else "transport error (no detail)"
    return f"HTTP {status}" + (f": {text[:160]}" if text else "")


#: Verdicts a checker can return.
#:
#: ``UNREACHABLE`` is the one that earns its keep. This lane answers a single
#: question -- *does this credential deliver?* -- and an upstream that cannot
#: answer at all is not evidence either way. Counting it as a credential failure
#: makes the lane fail for reasons no operator can act on, which is how a gate
#: gets ignored. Measured 2026-08-05 in a dispatched run: every key delivered
#: except NASA, which returned ``HTTP 503: upstream connect error or
#: disconnect/reset before headers`` -- an outage on NASA's side that said
#: nothing about ``NASA_API_KEY``, yet turned the lane red.
#:
#: A rejected credential is still a hard failure: 401/403, a provider's
#: invalid-key JSON, and an authenticated-but-empty response all return ``FAIL``.
DELIVERS = "DELIVERS"
FAIL = "FAIL"
UNREACHABLE = "UNREACH"


def _non_200(status: int, body: bytes) -> tuple[str, str]:
    """Classify a non-200 as a credential failure or an upstream outage.

    Server-side 5xx and transport failures (``status == 0``: DNS, TLS, timeout,
    connection reset) are the upstream's problem, not the key's -- a 503 arrives
    identically for a valid and an invalid credential, so it carries no
    information about the credential.

    **429 belongs with them, not with the failures.** A rate limit is the
    provider saying it *recognised* the credential and is declining to serve it
    right now; that is closer to evidence the key works than evidence it does
    not. The Alpha Vantage checker already reaches the same conclusion for the
    HTTP-200 throttle body it returns instead of a 429, and the two paths should
    not disagree about the same fact.

    Everything else -- 401 and 403 above all -- is the provider telling us
    something about the credential we presented, which is exactly what this lane
    is for.

    Args:
        status: Status from :func:`_get` (``0`` means transport error).
        body: Response body, or transport exception text when ``status`` is 0.

    Returns:
        ``(verdict, detail)`` with the detail already redacted.
    """
    detail = _fail_detail(status, body)
    if status == 0 or status == 429 or 500 <= status < 600:
        return UNREACHABLE, detail
    return FAIL, detail


def check_eia(key: str) -> tuple[str, str]:
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
        return _non_200(status, body)
    try:
        rows = json.loads(body)["response"]["data"]
    except Exception as exc:
        return FAIL, f"unparseable response: {exc}"
    if not rows:
        return FAIL, "authenticated but zero rows"
    r0 = rows[0]
    return (
        DELIVERS,
        f"{len(rows)} rows; e.g. {r0.get('period')} {r0.get('respondent')} D={r0.get('value')} MW",
    )


def check_fred(key: str) -> tuple[str, str]:
    """FRED 10-Year Treasury (DGS10) latest observations."""
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
        return _non_200(status, body)
    try:
        obs = json.loads(body)["observations"]
    except Exception as exc:
        return FAIL, f"unparseable: {exc}"
    if not obs:
        return FAIL, "authenticated but zero observations"
    return DELIVERS, f"{len(obs)} obs; latest {obs[0]['date']}={obs[0]['value']}"


def check_nasa(key: str) -> tuple[str, str]:
    """NASA DONKI space-weather notifications feed."""
    status, body = _get(f"https://api.nasa.gov/DONKI/notifications?type=all&api_key={key}")
    if status != 200:
        return _non_200(status, body)
    try:
        data = json.loads(body)
    except Exception as exc:
        return FAIL, f"unparseable: {exc}"
    return DELIVERS, f"DONKI reachable; {len(data)} notifications"


def check_alpha_vantage(key: str) -> tuple[str, str]:
    """Alpha Vantage GLOBAL_QUOTE for a single symbol (SPY)."""
    status, body = _get(
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey={key}"
    )
    if status != 200:
        return _non_200(status, body)
    try:
        data = json.loads(body)
    except Exception as exc:
        return FAIL, f"unparseable: {exc}"
    if data.get("Global Quote"):
        return DELIVERS, f"SPY quote delivered ({data['Global Quote'].get('05. price')})"
    # A rate-limit / info Note means the KEY was accepted but throttled (free tier
    # is 25 req/day) -- that is still "wired", so don't hard-fail the lane on it.
    # Only a genuine "Error Message" (bad/invalid key) is a delivery failure.
    if data.get("Error Message"):
        return FAIL, f"bad key: {data['Error Message']}"
    throttle = data.get("Note") or data.get("Information")
    if throttle:
        return DELIVERS, f"key accepted (throttled): {str(throttle)[:100]}"
    return FAIL, f"unexpected response: {str(data)[:120]}"


def check_usgs_eros(username: str, token: str) -> tuple[str, str]:
    """USGS EROS / EarthExplorer M2M login (username + application token).

    The M2M API authenticates a ``username`` with an application ``token``
    (generated in the EarthExplorer profile) via ``login-token``; USGS retired
    the password ``login`` endpoint on 2026-02-26. A successful login returns a
    session token in ``data`` with ``errorCode: null``; that session is then
    used (and a small ``dataset-search`` is issued) to prove real data
    delivery, and finally logged out.
    """
    base = "https://m2m.cr.usgs.gov/api/api/json/stable"

    def _post(
        path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
    ) -> tuple[int, dict[str, Any]]:
        """POST one M2M call, returning ``(http_status, body)``.

        The status is returned alongside the body so a login failure can be
        classified the same way :func:`_non_200` classifies the GET-based
        checkers: a 5xx or a transport error is M2M being down, not the token
        being wrong. ``0`` marks a transport failure.
        """
        try:
            resp = requests.post(
                f"{base}/{path}",
                json=payload,
                headers={"User-Agent": "mercury-cred-check", **(headers or {})},
                timeout=30,
            )
            body = resp.json() if resp.content else {"errorMessage": f"HTTP {resp.status_code}"}
            return resp.status_code, body
        except Exception as exc:  # pragma: no cover - network variance
            return 0, {"errorMessage": str(exc)}

    status, body = _post("login-token", {"username": username, "token": token})
    session = body.get("data")
    if not session:
        detail = (
            f"login failed (token): {body.get('errorCode')} "
            f"{_redact(str(body.get('errorMessage')))}"
        )
        # Same rule as the GET checkers: M2M being unable to answer is not a
        # verdict on the credential.
        if status == 0 or status == 429 or 500 <= status < 600:
            return UNREACHABLE, detail
        return FAIL, detail

    detail = "authenticated via token; session token issued"
    hdr = {"X-Auth-Token": str(session)}
    _status, ds = _post("dataset-search", {"datasetName": "landsat_ot_c2_l2"}, hdr)
    if isinstance(ds.get("data"), list):
        detail += f"; dataset-search returned {len(ds['data'])} dataset(s)"
    _post("logout", {}, hdr)
    return DELIVERS, detail


def check_openweathermap(key: str) -> tuple[str, str]:
    """Current weather for one city (London) via the OpenWeatherMap API."""
    status, body = _get(f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={key}")
    if status != 200:
        return _non_200(status, body)
    try:
        data = json.loads(body)
    except Exception as exc:
        return FAIL, f"unparseable: {exc}"
    return DELIVERS, f"London weather delivered (T={data.get('main', {}).get('temp')})"


#: source -> (env var(s), checker). First present env var is used.
CHECKS: dict[str, tuple[tuple[str, ...], Any]] = {
    "EIA": (("EIA_API_KEY",), check_eia),
    "FRED": (("FRED_API_KEY",), check_fred),
    "NASA": (("NASA_API_KEY",), check_nasa),
    "AlphaVantage": (("ALPHA_VANTAGE_API_KEY",), check_alpha_vantage),
    "OpenWeatherMap": (("OPENWEATHERMAP_API_KEY",), check_openweathermap),
}


def run() -> int:
    """Run every keyed data-source delivery check; exit 1 if any present key fails, else 0."""
    print("=== Mercury data-source credential delivery check ===")
    failures = 0
    unreachable = 0
    checked = 0
    for name, (env_vars, checker) in CHECKS.items():
        key = next((os.environ[e] for e in env_vars if os.environ.get(e)), "")
        if not key:
            print(f"  {name:14s} SKIP  ({'/'.join(env_vars)} not set)")
            continue
        checked += 1
        verdict, detail = checker(key)
        print(f"  {name:14s} {verdict:8s}  {detail}")
        if verdict == FAIL:
            failures += 1
        elif verdict == UNREACHABLE:
            unreachable += 1

    # USGS EROS / EarthExplorer M2M: authenticates a username with an application
    # token (USGS_KEY) via login-token -- USGS retired the password login
    # endpoint on 2026-02-26. The other USGS services (earthquake FDSN, water,
    # elevation) are keyless and need no credential.
    eros_user = os.environ.get("EROSERS_USERNAME", "")
    eros_token = os.environ.get("USGS_KEY", "")
    if eros_user and eros_token:
        checked += 1
        verdict, detail = check_usgs_eros(eros_user, eros_token)
        print(f"  {'USGS/EROS':14s} {verdict:8s}  {detail}")
        if verdict == FAIL:
            failures += 1
        elif verdict == UNREACHABLE:
            unreachable += 1
    else:
        status, _ = _get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
            "&starttime=2024-05-10&endtime=2024-05-11&minmagnitude=5"
        )
        print(
            f"  {'USGS/EROS':14s} SKIP  need EROSERS_USERNAME + USGS_KEY (EROS M2M "
            f"application token); keyless USGS earthquake feed "
            f"reachable HTTP {status}"
        )
    print(
        f"=== {failures} credential failure(s), {unreachable} upstream-unreachable "
        f"among {checked} source(s) with a key present ==="
    )
    # An unreachable upstream is reported, never counted as a credential verdict.
    # But "every upstream we tried was down" is its own signal and must not read
    # as a clean run, so it fails: at that point the lane measured nothing, and
    # silently exiting 0 would be a green light with no evidence behind it.
    if checked and unreachable == checked:
        print(
            "=== every source with a key present was unreachable; this run "
            "verified no credential ===",
            file=sys.stderr,
        )
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())

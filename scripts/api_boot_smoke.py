#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Boot the API on the install it is run from and check what actually starts.

Every test lane builds the AMA backend before it imports the package, so no
lane ever saw the server on an install without it. That install is the one a
bare ``pip install "mercury-agent-global[api]"`` produces, and on it the
server started, answered ``/health`` with ``healthy``, served its sign-in pages
and had no account API behind them. This script runs the real program on the
real environment, in the two states a CI job passes through:

``--expect refuse``
    Before AMA is provisioned. Importing ``omni_mercury_engine.api.server``
    and running ``mercury-agent serve`` must both stop with the PQC gate's
    refusal, and ``serve`` must stop before printing its banner. The mode
    requires ``ama_cryptography`` to be absent and exits 2 if it is present,
    so a lane that starts installing AMA earlier cannot turn the check into a
    silent pass.

``--expect serve``
    After AMA is provisioned. A ``uvicorn`` process is started with the
    browser frontend enabled and must answer ``/health``; ``/metrics`` must
    report every router as mounted or disabled and none as unavailable; the
    OpenAPI document must carry the account API; the frontend pages must load;
    and the startup log must hold no traceback.

Usage::

    python scripts/api_boot_smoke.py --expect refuse
    python scripts/api_boot_smoke.py --expect serve

Exit codes:
    0  The install behaved as expected.
    1  It did not; each failed expectation is printed.
    2  The mode's precondition does not hold (see above).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Seconds to wait for ``/health`` after starting the server.
BOOT_TIMEOUT = 180.0

#: Paths the account API must serve; their absence was the original fault.
REQUIRED_PATHS = ("/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/me")

#: Frontend pages that must load when the frontend is enabled.
FRONTEND_PAGES = ("/login", "/dashboard", "/console")

#: Log fragments that mean startup did not finish cleanly.
LOG_FAULTS = ("Traceback (most recent call last)", "startup maintenance sweep failed")

_ROUTER_STATE = re.compile(
    r'^omni_mercury_api_router_state\{router="([a-z0-9_]+)",state="([a-z]+)"\} 1$', re.MULTILINE
)


def _base_env() -> dict[str, str]:
    """Environment for a development-mode server with a throwaway signing key."""
    env = dict(os.environ)
    env["MERCURY_AGENT_ENV"] = "development"
    env["JWT_SECRET_KEY"] = secrets.token_hex(32)
    env["MERCURY_FRONTEND_ENABLED"] = "true"
    env.pop("MERCURY_DECEPTION_ENABLED", None)
    return env


def _run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 # nosec B603 - fixed argv, no shell
        argv, capture_output=True, text=True, env=env, timeout=300, check=False
    )


def check_refuse() -> list[str]:
    """Return what is wrong with an AMA-less install's refusal; empty when correct."""
    env = _base_env()
    failures: list[str] = []

    imported = _run([sys.executable, "-c", "import omni_mercury_engine.api.server"], env)
    if imported.returncode == 0:
        failures.append("importing omni_mercury_engine.api.server succeeded without AMA")
    elif "AMA" not in imported.stderr:
        failures.append(
            "the API import failed without naming AMA: " + imported.stderr.strip()[-300:]
        )

    served = _run([sys.executable, "-m", "omni_mercury_engine.cli", "serve"], env)
    if served.returncode != 1:
        failures.append(f"`serve` exited {served.returncode} without AMA, expected 1")
    if "Mercury Agent API Server" in served.stdout:
        failures.append("`serve` printed its banner before refusing")
    if "the API server cannot start" not in served.stderr or "AMA" not in served.stderr:
        failures.append("`serve` did not print the refusal: " + served.stderr.strip()[-300:])
    return failures


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def _get(url: str) -> tuple[int, str]:
    """GET ``url`` on the loopback server; return ``(status, body)``."""
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - fixed http://127.0.0.1
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:  # nosec B310
            return int(response.status), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return int(error.code), error.read().decode("utf-8", "replace")


def _wait_for_health(base: str, server: subprocess.Popen[bytes]) -> str | None:
    """Poll ``/health`` until it answers; return an error string on failure."""
    deadline = time.monotonic() + BOOT_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return f"the server exited with status {server.returncode} before answering /health"
        try:
            status, body = _get(base + "/health")
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
            continue
        if status != 200:
            return f"/health answered {status}"
        if json.loads(body).get("status") != "healthy":
            return f"/health body was {body[:200]}"
        return None
    return f"/health did not answer within {BOOT_TIMEOUT:.0f} s"


def _check_served(base: str) -> list[str]:
    """Check the running server's router states, account API and frontend pages."""
    failures: list[str] = []

    status, metrics = _get(base + "/metrics")
    if status != 200:
        return [f"/metrics answered {status}"]
    states = dict(_ROUTER_STATE.findall(metrics))
    if not states:
        failures.append("/metrics exports no omni_mercury_api_router_state series")
    unavailable = sorted(name for name, state in states.items() if state == "unavailable")
    if unavailable:
        failures.append(f"routers unavailable: {', '.join(unavailable)}")
    for name in ("accounts", "frontend"):
        if states.get(name) != "mounted":
            failures.append(f"router {name!r} is {states.get(name, 'not recorded')}, not mounted")

    status, document = _get(base + "/openapi.json")
    if status != 200:
        failures.append(f"/openapi.json answered {status}")
    else:
        paths = json.loads(document)["paths"]
        missing = [path for path in REQUIRED_PATHS if path not in paths]
        if missing:
            failures.append(f"OpenAPI document lacks {', '.join(missing)}")
        print(f"  {len(paths)} documented paths; router states: {states}")

    for page in FRONTEND_PAGES:
        status, _ = _get(base + page)
        if status != 200:
            failures.append(f"{page} answered {status}")
    return failures


def check_serve() -> list[str]:
    """Start the server and return what is wrong with it; empty when correct."""
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    argv = [
        sys.executable,
        "-m",
        "uvicorn",
        "omni_mercury_engine.api.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with tempfile.TemporaryDirectory(prefix="api-boot-smoke-") as tmp:
        log_path = Path(tmp) / "server.log"
        with log_path.open("wb") as log:
            server = subprocess.Popen(  # noqa: S603 # nosec B603 - fixed argv, no shell
                argv, stdout=log, stderr=subprocess.STDOUT, env=_base_env()
            )
            try:
                error = _wait_for_health(base, server)
                failures = [error] if error else _check_served(base)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=30)
        text = log_path.read_text(encoding="utf-8", errors="replace")
    failures += [f"startup log contains {fault!r}" for fault in LOG_FAULTS if fault in text]
    if failures:
        print("---- server log (tail) ----")
        print(text[-3000:])
    return failures


def main(argv: list[str] | None = None) -> int:
    """Run one mode and report."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--expect", choices=("refuse", "serve"), required=True)
    args = parser.parse_args(argv)

    ama_present = importlib.util.find_spec("ama_cryptography") is not None
    if args.expect == "refuse" and ama_present:
        print(
            "precondition failed: --expect refuse checks an install without AMA, but "
            "ama_cryptography is importable here. Run it before the AMA build step."
        )
        return 2
    if args.expect == "serve" and not ama_present:
        print(
            "precondition failed: --expect serve checks an install with AMA, but "
            "ama_cryptography is not importable here. Run it after the AMA build step."
        )
        return 2

    failures = check_refuse() if args.expect == "refuse" else check_serve()
    for failure in failures:
        print(f"::error title=API boot smoke ({args.expect})::{failure}")
    if failures:
        return 1
    print(f"OK: the API behaved as expected for --expect {args.expect}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

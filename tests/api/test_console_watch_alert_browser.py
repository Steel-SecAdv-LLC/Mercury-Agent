# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A finding the watch reports reaches an open console, in a real browser.

The watch broadcasts each new finding on the alert bus, and the console's
alert stream was only ever tested as text: nothing opened a console and
watched a finding arrive. This starts the real application
(``console_alert_server.py``), opens ``/console`` in headless Chromium
(``scripts/console_watch_alert_check.js``), and has the watcher report one
finding once the console is connected.

It needs Node with Playwright and a Chromium it can launch. Where those are
absent the test is skipped; ``.github/workflows`` does not install them, so
today this runs on a developer's machine and not in CI.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SERVER = Path(__file__).with_name("console_alert_server.py")
CHECK = REPO / "scripts" / "console_watch_alert_check.js"
NODE = shutil.which("node")


def _node_env() -> dict[str, str]:
    """The environment Node needs to find a globally installed Playwright."""
    env = dict(os.environ)
    npm = shutil.which("npm")
    if npm and "NODE_PATH" not in env:
        root = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, check=False)
        if root.returncode == 0 and root.stdout.strip():
            env["NODE_PATH"] = root.stdout.strip()
    return env


def _playwright_available() -> bool:
    if NODE is None:
        return False
    probe = subprocess.run(
        [NODE, "-e", "require('playwright')"], capture_output=True, env=_node_env(), check=False
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _playwright_available(),
    reason="Node with Playwright is not installed: the console is opened in headless Chromium",
)


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_a_new_finding_appears_in_an_open_consoles_banner_terminal_and_agent_panel(
    tmp_path: Path,
) -> None:
    port, trigger = _free_port(), tmp_path / "report"
    env = {
        **os.environ,
        "MERCURY_FRONTEND_ENABLED": "true",
        "MERCURY_WATCH_ENABLED": "false",  # the helper runs the watcher itself, on a scripted feed
        "MERCURY_WATCH_DIR": str(tmp_path / "watch"),
        "MERCURY_MAIL_OUTBOX": str(tmp_path / "outbox"),
        "OMNI_RATE_LIMIT_BURST": "2000",
    }
    env.pop("MERCURY_SMTP_HOST", None)
    env.pop("MERCURY_WATCH_WEBHOOK_URL", None)
    server = subprocess.Popen(
        [sys.executable, str(SERVER), str(port), str(trigger)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
        cwd=REPO,
    )
    try:
        assert server.stdout is not None
        for line in server.stdout:  # the server prints "ready" once it is listening
            if line.strip() == "ready":
                break
        else:
            pytest.fail("the server exited before it was ready")

        check = subprocess.run(
            [str(NODE), str(CHECK)],
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
            env={**_node_env(), "BASE": f"http://127.0.0.1:{port}", "TRIGGER": str(trigger)},
        )

        report = check.stdout + check.stderr
        assert check.returncode == 0, report
        assert "console watch alert: all checks passed" in report
        assert report.count("\nok ") + report.startswith("ok ") == 11, report
        assert "FAIL" not in report
    finally:
        if server.stdin is not None:
            server.stdin.close()  # the helper serves until its stdin closes
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()

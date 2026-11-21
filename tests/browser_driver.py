# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the headless-browser driver (Node + Playwright) is, for the tests that need one.

Shared by ``tests/api/test_console_watch_alert_browser.py`` and the session
gate in ``tests/conftest.py`` (``MERCURY_REQUIRES_BROWSER=1``), so both ask
the same question the same way.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

__all__ = ["DRIVER_DIR", "node_env", "playwright_available"]

#: The pinned driver: ``npm ci --prefix tests/api/browser --ignore-scripts``.
DRIVER_DIR = Path(__file__).resolve().parent / "api" / "browser"


def node_env() -> dict[str, str]:
    """The environment in which Node finds Playwright.

    The pinned install under :data:`DRIVER_DIR` comes first; a Playwright
    installed globally is used when that is absent, and an explicit
    ``NODE_PATH`` is left alone.
    """
    env = dict(os.environ)
    if "NODE_PATH" in env:
        return env
    pinned = DRIVER_DIR / "node_modules"
    if (pinned / "playwright").is_dir():
        env["NODE_PATH"] = str(pinned)
        return env
    npm = shutil.which("npm")
    if npm:
        root = subprocess.run([npm, "root", "-g"], capture_output=True, text=True, check=False)
        if root.returncode == 0 and root.stdout.strip():
            env["NODE_PATH"] = root.stdout.strip()
    return env


def playwright_available() -> bool:
    """Whether Node can load Playwright here."""
    node = shutil.which("node")
    if node is None:
        return False
    probe = subprocess.run(
        [node, "-e", "require('playwright')"], capture_output=True, env=node_env(), check=False
    )
    return probe.returncode == 0

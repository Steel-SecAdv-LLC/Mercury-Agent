# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console control crawl ships as a script whose contract the docs can rely on.

``scripts/console_control_crawl.js`` is a manual audit tool (CI runs no
browser), so this is the only automated check it gets: it parses as
JavaScript where Node is installed, documents every environment variable it
reads, names every console view the served navigation has, and states its
limits in its header rather than claiming a verdict it cannot give.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "console_control_crawl.js"
NOTES = REPO / "docs" / "design" / "mercury_console" / "README.md"


def test_the_script_parses_where_node_is_installed() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; the crawl is a manual tool")
    proc = subprocess.run(
        [node, "--check", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_every_environment_variable_it_reads_is_documented_in_its_header() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    header = text.split("const { chromium }", 1)[0]
    read = set(re.findall(r"process\.env\.([A-Z_]+)", text))
    assert read >= {"BASE", "OUT", "MAX_PER_VIEW", "VIEWS", "DEPTH"}
    undocumented = sorted(name for name in read if name not in header)
    assert not undocumented, f"env vars read but not in the header: {undocumented}"


def test_it_names_every_navigation_view_the_console_serves() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    nav = re.search(r"const NAV = \[(.*?)\];", text, re.S)
    assert nav
    views = re.findall(r'"([^"]+)"', nav.group(1))
    template = (
        REPO
        / "src"
        / "omni_mercury_engine"
        / "api"
        / "frontend_assets"
        / "static"
        / "console"
        / "logic.js"
    ).read_text(encoding="utf-8")
    for view in (
        "Terminal",
        "Agent",
        "Mass Run",
        "Detectors",
        "Domains",
        "MCP tools",
        "AI vendors",
        "Session",
        "Diagnostics",
    ):
        assert view in views, view
        assert f'"{view}"' in template, f"the console no longer has a {view} view"


def test_its_header_states_the_limits_and_the_notes_document_the_replay_rule() -> None:
    header = SCRIPT.read_text(encoding="utf-8").split("const { chromium }", 1)[0]
    assert "manual audit tool" in header and "lead, not a verdict" in header
    assert "CI image" not in header
    notes = NOTES.read_text(encoding="utf-8")
    assert "## 21. Control audit" in notes and "replayed by hand" in notes

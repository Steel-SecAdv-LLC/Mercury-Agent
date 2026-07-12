# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Live delivery check for the injected data-source secrets (network lane).

These ``@pytest.mark.network`` tests run in the ``network-tests`` workflow, where
the repository secrets are injected onto the env vars the loaders read. They
confirm the keys don't just *exist* but actually *deliver* real data when called.
Each source with an unset key skips (so a local run without that key is clean);
a source whose key is present but fails to deliver is a hard failure.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.network

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_data_credentials.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("verify_data_credentials", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eia_key_delivers_grid_data() -> None:
    key = os.environ.get("EIA_API_KEY", "")
    if not key:
        pytest.skip("EIA_API_KEY not set in this environment")
    mod = _load_module()
    ok, detail = mod.check_eia(key)
    assert ok, f"EIA_API_KEY set but did not deliver: {detail}"


def test_all_present_keys_deliver() -> None:
    mod = _load_module()
    # run() returns non-zero only if a source whose key IS present failed.
    assert mod.run() == 0, "a data source with a configured key failed to deliver"

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for the intelligence-layer tests.

Gate-decision auditing is durable by default (it fsyncs to
``artifacts/audit/gate_decisions.jsonl``); the unit tests neither want that
side effect nor to be slowed by fsync, so it is disabled process-wide here. A
test that specifically asserts an audit happened monkeypatches
``record_gate_decision`` instead of relying on the durable sink.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_gate_audit() -> None:
    """Disable durable gate auditing for the whole intel test session."""
    os.environ["MERCURY_GATE_AUDIT_DISABLED"] = "1"

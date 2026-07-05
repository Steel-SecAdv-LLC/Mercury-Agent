# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared fixtures for the intelligence-layer tests.

Gate-decision auditing is durable by default (it fsyncs to
``artifacts/audit/gate_decisions.jsonl``); the unit tests neither want that
side effect nor to be slowed by fsync, so it is disabled for the intel test
session. A test that specifically asserts an audit happened monkeypatches
``record_gate_decision`` (or re-enables the flag with ``monkeypatch.setenv``)
instead of relying on the durable sink.

The disable is **restored on teardown** rather than leaked: this is a
session-scoped ``os.environ`` mutation, and when the intel suite runs in the
same ``pytest`` process as other suites (a full-repo run, not just the isolated
``ci/*`` intel lanes), leaving ``MERCURY_GATE_AUDIT_DISABLED=1`` set would
silently disable auditing for *their* tests too -- e.g. the capability-contract
fail-closed-is-audited assertion would spuriously fail. Saving and restoring the
prior value keeps the fixture's effect scoped to the intel session.
"""

from __future__ import annotations

import pytest

_AUDIT_DISABLED_ENV = "MERCURY_GATE_AUDIT_DISABLED"


@pytest.fixture(autouse=True)
def _disable_gate_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable durable gate auditing for each intel test (auto-restored).

    Function-scoped via ``monkeypatch`` on purpose: a session-scoped
    ``os.environ`` mutation is only torn down at session *end*, so when the intel
    suite runs in the same ``pytest`` process as other suites (a full-repo run,
    not just the isolated ``ci/*`` intel lanes) the disable would still be in
    effect while *their* tests run -- e.g. the capability-contract
    fail-closed-is-audited assertion would spuriously fail. ``monkeypatch``
    restores the prior value after every intel test, so the effect never escapes
    this package. A test that needs the durable sink re-enables it with its own
    ``monkeypatch.setenv`` (which wins, being applied after this fixture).
    """
    monkeypatch.setenv(_AUDIT_DISABLED_ENV, "1")

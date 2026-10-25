# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Safety policy is the live, strict configuration of the dependency gate.

``.safety-policy.yml`` is read by ``safety check --policy-file`` in both
blocking dependency gates. It must stay in the 2.0 format that command parses
(a 3.0 file aborts it), keep every finding reportable, and hold an exception
only with a reason and an expiry no more than 90 days out, so that an
acceptance cannot outlive its review. Checked against Safety 3.8.1: an
exception with a future expiry removes exactly that advisory, an expired one
stops applying, and a misspelled key makes the gate fail rather than pass.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_POLICY = _REPO / ".safety-policy.yml"
_WORKFLOWS = (_REPO / ".github/workflows/ci.yml", _REPO / ".github/workflows/security.yml")
_MAX_EXCEPTION_DAYS = 90


def _security() -> dict[str, Any]:
    policy = yaml.safe_load(_POLICY.read_text(encoding="utf-8"))
    assert str(policy["version"]) == "2.0", "safety check only parses policy version 2.0"
    security = policy["security"]
    assert isinstance(security, dict)
    return security


def test_every_finding_is_reported_and_fails_the_job() -> None:
    security = _security()
    assert float(security["ignore-cvss-severity-below"]) == 0.0
    assert security["ignore-cvss-unknown-severity"] is False
    assert security["ignore-unpinned-requirements"] is False
    assert security["continue-on-vulnerability-error"] is False


def test_each_exception_has_a_reason_and_a_near_expiry() -> None:
    today = dt.date.today()
    for advisory, entry in (_security()["ignore-vulnerabilities"] or {}).items():
        assert str(advisory).isdigit(), f"{advisory!r} is not a Safety advisory id"
        assert str(entry.get("reason", "")).strip(), f"advisory {advisory} has no reason"
        expires = entry.get("expires")
        assert expires, f"advisory {advisory} never expires"
        day = expires if isinstance(expires, dt.date) else dt.date.fromisoformat(str(expires))
        assert today < day <= today + dt.timedelta(days=_MAX_EXCEPTION_DAYS), (
            f"advisory {advisory} expires {day}; re-review it and set an expiry within "
            f"{_MAX_EXCEPTION_DAYS} days"
        )


@pytest.mark.parametrize("workflow", _WORKFLOWS, ids=lambda p: p.name)
def test_both_gates_read_the_policy(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert "safety check" in text
    assert "--policy-file .safety-policy.yml" in text
    assert "--ignore " not in text and "--ignore=" not in text

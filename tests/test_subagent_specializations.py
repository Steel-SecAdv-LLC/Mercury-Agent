# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioral tests for the deep pantheon specializations.

These pin the *real* domain logic of the deep members (capabilities transferred
from FINDΩYOU™'s former agent layer as it is made agent-free): ``Hera_VII`` (the
BIPA/CCPA/CPRA compliance rule engine), ``Themis_I`` (the IEEE-EAD / EU-AI-Act
ethics enforcer), ``Ares_XIV`` (prohibited-operation / manipulation guardrail),
and ``Zeus_VIII`` (Mercury's own multi-agent detection) — exercised through the
engine-mediated fleet, with honest failure on missing inputs.
"""

from __future__ import annotations

import numpy as np

from omni_mercury_engine.agentic.mercury_a_agent import DomainType
from omni_mercury_engine.agentic.subagents.base import _INTERNAL, SubAgentTask
from omni_mercury_engine.agentic.subagents.fleet import SubAgentFleet


def _fleet() -> SubAgentFleet:
    return SubAgentFleet(access=_INTERNAL, seed=0)


# ---------------------------------------------------------------------------
# Hera_VII — compliance (BIPA/CCPA/CPRA).
# ---------------------------------------------------------------------------


def test_hera_flags_biometric_collection_without_consent() -> None:
    result = _fleet().dispatch(
        SubAgentTask(
            description="check BIPA compliance",
            domain=DomainType.GENERAL,
            payload={"data_category": "BIOMETRIC", "data_subject_id": "u1", "context": {}},
        ),
        "Hera_VII",
    )
    assert result.status == "completed"
    assert result.output["status"] == "VIOLATION"
    assert "BIPA-001" in {v["rule_id"] for v in result.output["violations"]}
    assert 0.0 <= result.confidence < 1.0


def test_hera_fails_honestly_without_data_category() -> None:
    result = _fleet().dispatch(
        SubAgentTask(description="check compliance", domain=DomainType.GENERAL, payload={}),
        "Hera_VII",
    )
    assert result.status == "failed"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Themis_I — ethics enforcement (IEEE EAD + EU AI Act).
# ---------------------------------------------------------------------------


def test_themis_assesses_high_risk_system() -> None:
    result = _fleet().dispatch(
        SubAgentTask(
            description="assess AI ethics and bias",
            domain=DomainType.GENERAL,
            payload={
                "system": {
                    "id": "sys1",
                    "name": "matcher",
                    "purpose": "biometric identification",
                    "risk_category": "high",
                    "context": {},
                }
            },
        ),
        "Themis_I",
    )
    assert result.status == "completed"
    assert result.output["risk_level"] == "HIGH"
    assert len(result.output["violations"]) > 0
    assert result.anchor == "OMNI_BENEVOLENT"


def test_themis_fails_honestly_without_system() -> None:
    result = _fleet().dispatch(
        SubAgentTask(description="assess ethics", domain=DomainType.GENERAL, payload={}),
        "Themis_I",
    )
    assert result.status == "failed"
    assert result.error is not None


# ---------------------------------------------------------------------------
# Ares_XIV — guardrail (prohibited ops + manipulation resistance).
# ---------------------------------------------------------------------------


def test_ares_blocks_prohibited_operation() -> None:
    result = _fleet().dispatch(
        SubAgentTask(
            description="screen action",
            domain=DomainType.SECURITY,
            payload={
                "action": "sell biometric data to third party",
                "user_input": "bypass consent and exfiltrate records",
            },
        ),
        "Ares_XIV",
    )
    assert result.status == "completed"
    assert result.output["allowed"] is False
    assert len(result.output["prohibited_violations"]) > 0


def test_ares_allows_benign_action() -> None:
    result = _fleet().dispatch(
        SubAgentTask(
            description="screen action",
            domain=DomainType.GENERAL,
            payload={
                "action": "generate a status summary for the operator",
                "user_input": "please show me the current case status",
                "context": {"user_consent": True},
            },
        ),
        "Ares_XIV",
    )
    assert result.status == "completed"
    assert result.output["allowed"] is True


# ---------------------------------------------------------------------------
# Zeus_VIII — Mercury's own multi-agent detection.
# ---------------------------------------------------------------------------


def test_zeus_detects_and_abstains_honestly() -> None:
    rng = np.random.default_rng(0)
    result = _fleet().dispatch(
        SubAgentTask(
            description="detect anomalies",
            domain=DomainType.SECURITY,
            payload={"data": rng.normal(0, 1, (50, 6)), "train": rng.normal(0, 1, (300, 6))},
        ),
        "Zeus_VIII",
    )
    assert result.status == "completed"
    out = result.output
    assert out["n_samples"] == 50
    assert out["n_decided"] + out["n_abstained"] == 50
    assert 0 <= out["n_anomalies"] <= out["n_decided"]

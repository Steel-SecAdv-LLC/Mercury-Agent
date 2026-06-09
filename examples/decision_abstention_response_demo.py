#!/usr/bin/env python3
"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

Decision / Abstention / Response layer demo.

Closes the identify -> interpret -> decide -> deter -> verify loop on a handful
of calibrated-confidence signals, showing the first-class "don't-know" gate, the
proportionate response ladder, the reversible/escalation contract, and the audit
ledger. Detector-free, so it runs anywhere the package imports.

Run:
    python examples/decision_abstention_response_demo.py
"""

from __future__ import annotations

from omni_mercury_engine.core.types import ThreatLevel
from omni_mercury_engine.decision import (
    Authorization,
    ConfidenceSignal,
    DecisionResponseLoop,
    permit_all_gate,
)


def main() -> None:
    """Drive the loop over representative signals and print the certificates."""
    # In production, bind ``ethical_gate`` to the engine's benevolence +
    # σ_Immutable boundary (``OmniMercuryEngine.decide_and_respond`` does this for
    # you). ``permit_all_gate`` here keeps the demo self-contained.
    loop = DecisionResponseLoop(ethical_gate=permit_all_gate)
    operator = Authorization(authority="soc-operator-7", reason="confirmed incident")

    # Confidence (the conformal set / point) and severity (impact) are distinct
    # axes: the policy decides *whether* it is an anomaly, the severity decides
    # *how assertively* to respond. We pass severity explicitly here; on a real
    # engine result it comes from the detector's ``severity`` field.
    scenarios: list[tuple[str, ConfidenceSignal, ThreatLevel, Authorization | None]] = [
        (
            "confident normal",
            ConfidenceSignal(0.03, prediction_set=(0,), coverage=0.9),
            ThreatLevel.NONE,
            None,
        ),
        (
            "uncertain (abstain)",
            ConfidenceSignal(0.52, prediction_set=(0, 1), coverage=0.9),
            ThreatLevel.MODERATE,
            None,
        ),
        (
            "novel / atypical",
            ConfidenceSignal(0.50, prediction_set=(), coverage=0.9),
            ThreatLevel.HIGH,
            None,
        ),
        (
            "low-severity anomaly",
            ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9),
            ThreatLevel.LOW,
            None,
        ),
        (
            "high-severity anomaly",
            ConfidenceSignal(0.95, prediction_set=(1,), coverage=0.9),
            ThreatLevel.HIGH,
            None,
        ),
        (
            "critical anomaly (no auth)",
            ConfidenceSignal(0.99, prediction_set=(1,), coverage=0.9),
            ThreatLevel.CRITICAL,
            None,
        ),
        (
            "critical anomaly (authorized)",
            ConfidenceSignal(0.99, prediction_set=(1,), coverage=0.9),
            ThreatLevel.CRITICAL,
            operator,
        ),
    ]

    print("=" * 78)
    print("Mercury Agent — Decision / Abstention / Response loop")
    print("=" * 78)
    for label, signal, severity, auth in scenarios:
        result = loop.step(signal, domain="network_security", severity=severity, authorization=auth)
        decision = result.decision
        response = result.response
        print(f"\n▶ {label}")
        print(f"    verdict   : {decision.verdict.value:9}  (three-state: {decision.state.value})")
        print(f"    reason    : {decision.reason}")
        print(f"    response  : {response.action.name}  [{response.action.tier.value}]")
        print(f"    status    : {response.status.value}  (reversible={response.action.reversible})")

    print("\n" + "=" * 78)
    print("Audit ledger summary (the verifiable record):")
    summary = loop.ledger.summary()
    print(f"    total decisions : {summary['total']}")
    print(f"    by verdict      : {summary['by_verdict']}")
    print(f"    by response     : {summary['by_status']}")
    print(f"    abstention rate : {summary['abstention_rate']:.0%}")
    print("=" * 78)


if __name__ == "__main__":
    main()

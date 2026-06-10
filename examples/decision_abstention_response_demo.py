#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Decision / Abstention / Response layer demo.

Closes Mercury's loop from *interpret* to *deter*: it feeds a handful of
representative detection certificates through the
:class:`DecisionAbstentionResponder` and prints, for each, the three‑state
verdict (grounded / unavailable / undecidable), the operational disposition,
the calibrated confidence, the bounded non‑destructive response, and the
human‑readable explanation.

It runs standalone -- no model training and no network -- because the layer is
a pure, deterministic function of the certificate. The final section shows the
loop closing into Mercury's existing channels (a CAP 1.2 alert and an autonomy
``AgentAction``).

Run::

    python examples/decision_abstention_response_demo.py
"""

from __future__ import annotations

from typing import Any

from omni_mercury_engine.decision import (
    DecisionAbstentionResponder,
    DecisionLedger,
    DecisionLoop,
    to_agent_action,
    to_cap_alert,
)


def _conformal(labels: list[int], coverage: float = 0.9) -> dict[str, Any]:
    """A conformal certificate section as ``detect_with_fusion`` attaches it."""
    return {
        "prediction_set": labels,
        "set_size": len(labels),
        "abstain": len(labels) == 2,
        "coverage": coverage,
    }


# Representative certificates spanning the gate's behaviour. Each dict is the
# shape detect_with_fusion() returns once a conformal calibrator is fit.
SCENARIOS: list[tuple[str, dict[str, Any]]] = [
    (
        "Confident, severe anomaly (calibrated singleton)",
        {
            "anomaly_prob": 0.96,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.93,
            "conformal": _conformal([1]),
        },
    ),
    (
        "Confident normal (calibrated singleton)",
        {
            "anomaly_prob": 0.03,
            "is_anomaly": False,
            "threshold_used": 0.5,
            "severity": 0.0,
            "conformal": _conformal([0]),
        },
    ),
    (
        "Calibrated ambiguity -> resolvable don't-know",
        {
            "anomaly_prob": 0.54,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.6,
            "conformal": _conformal([0, 1]),
        },
    ),
    (
        "Atypical point (empty set) -> fail-closed hold",
        {
            "anomaly_prob": 0.42,
            "is_anomaly": False,
            "threshold_used": 0.5,
            "severity": 0.3,
            "conformal": _conformal([]),
        },
    ),
    (
        "Neuro-symbolic disagreement demotes a grounded call",
        {
            "anomaly_prob": 0.91,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.7,
            "conformal": _conformal([1]),
            "symbolic_consistency": {"satisfaction": 0.18},
        },
    ),
    (
        "Ethical gate refusal forces a fail-closed hold",
        {
            "anomaly_prob": 0.99,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.95,
            "conformal": _conformal([1]),
            "gosnn_metadata": {
                "ethical_gate_passed": False,
                "sigma_immutable_score": 0.10,
                "sigma_immutable_threshold": 0.93,
            },
        },
    ),
    (
        "No certificate, near threshold -> uncalibrated don't-know",
        {
            "anomaly_prob": 0.52,
            "is_anomaly": True,
            "threshold_used": 0.5,
            "severity": 0.4,
        },
    ),
]


def main() -> None:
    """Run every scenario through the responder and narrate the outcome."""
    responder = DecisionAbstentionResponder()

    print("=" * 78)
    print("Mercury Agent — Decision / Abstention / Response layer")
    print("identify → interpret → decide → deter, with an explicit don't-know gate")
    print("=" * 78)

    for title, result in SCENARIOS:
        record = responder.decide(result, domain="security")
        print(f"\n▸ {title}")
        print(
            f"   state={record.state.value:<11} disposition={record.disposition.value:<6} "
            f"label={record.decision_label} "
            f"confidence={record.decision_confidence} calibrated={record.calibrated}"
        )
        print(
            f"   response={record.response.action.value} "
            f"(urgency={record.response.urgency}, "
            f"human={record.response.requires_human}, "
            f"fail_closed={record.response.fail_closed})"
        )
        print(f"   {record.explain()}")

    # The loop closing into existing channels.
    print("\n" + "=" * 78)
    print("Closing the loop into existing channels")
    print("=" * 78)
    severe = responder.decide(SCENARIOS[0][1], domain="security")

    action = to_agent_action(severe)
    print(
        f"\nAutonomy AgentAction: type={action.action_type!r} "
        f"confidence={action.confidence:.2f}"
    )

    cap_xml = to_cap_alert(severe, area_description="Sector 7")
    n_lines = len(cap_xml.splitlines()) if cap_xml else 0
    print(f"CAP 1.2 alert emitted: {bool(cap_xml)} ({n_lines} lines of XML)")

    # A grounded-normal decision is passive: no alert is emitted.
    clear = responder.decide(SCENARIOS[1][1], domain="security")
    print(f"Grounded-normal alert emitted: {to_cap_alert(clear) is not None} (monitor only)")

    # The "verify" step: run the stream through a closed loop with an audit
    # ledger, then show the O(1) summary and a JSON persistence round-trip.
    print("\n" + "=" * 78)
    print("Closing the verify loop (audit ledger)")
    print("=" * 78)
    loop = DecisionLoop(responder, ledger=DecisionLedger(maxlen=1000))
    loop.run((result for _, result in SCENARIOS), domain="security")
    summary = loop.summary()
    print(f"\nRecorded {summary['total']} decisions")
    print(f"   by honesty state : {summary['by_state']}")
    print(f"   by disposition   : {summary['by_disposition']}")
    print(f"   abstention rate  : {summary['abstention_rate']:.0%}")
    print(f"   calibrated rate  : {summary['calibrated_rate']:.0%}")

    # The trail round-trips through JSON (a reloadable audit artifact).
    reloaded = DecisionLedger.from_json(loop.ledger.to_json())
    print(f"JSON persistence round-trip preserves the summary: {reloaded.summary() == summary}")


if __name__ == "__main__":
    main()

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A quantile alert must be able to exceed its own threshold.

``histogram_quantile`` returns the upper bound of the highest finite bucket
for any observation beyond it. ``OmniMercurySlowDetection`` alerts on
``p95 > 10`` seconds, and the detection histogram's highest finite bucket was
10.0, so the quantile could reach 10 and never pass it: the alert could not
fire however slow detection became.

For every alert of the form ``<recorded quantile> > <threshold>`` in
``monitoring/prometheus/rules.yml`` whose histogram this package defines, the
highest finite bucket must lie above the threshold.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("prometheus_client")

from omni_mercury_engine.core import metrics

_RULES = Path(__file__).resolve().parents[2] / "monitoring" / "prometheus" / "rules.yml"
_QUANTILE = re.compile(r"histogram_quantile\([^,]+,\s*sum\(rate\((\w+)_bucket")
_ALERT = re.compile(r"^\s*([\w:]+)\s*>\s*([0-9.]+)\s*$")


def _histograms() -> dict[str, tuple[float, ...]]:
    """Finite upper bounds of every histogram the package defines, by metric name."""
    found: dict[str, tuple[float, ...]] = {}
    for value in vars(metrics).values():
        bounds = getattr(value, "_upper_bounds", None)
        name = getattr(value, "_name", None)
        if bounds and isinstance(name, str):
            found[name] = tuple(b for b in bounds if b != float("inf"))
    return found


def _quantile_alerts() -> list[tuple[str, str, float]]:
    """``(alert, histogram metric, threshold)`` for each recorded-quantile alert."""
    rules = [r for g in yaml.safe_load(_RULES.read_text())["groups"] for r in g["rules"]]
    recorded = {
        rule["record"]: match.group(1)
        for rule in rules
        if "record" in rule and (match := _QUANTILE.search(str(rule["expr"])))
    }
    alerts = []
    for rule in rules:
        match = _ALERT.match(str(rule.get("expr", ""))) if "alert" in rule else None
        if match and match.group(1) in recorded:
            alerts.append((rule["alert"], recorded[match.group(1)], float(match.group(2))))
    return alerts


def test_the_slow_detection_alert_is_among_those_checked() -> None:
    checked = {alert: metric for alert, metric, _ in _quantile_alerts()}
    assert checked.get("OmniMercurySlowDetection") == "omni_detection_duration_seconds"
    assert "omni_detection_duration_seconds" in _histograms()


@pytest.mark.parametrize(("alert", "metric", "threshold"), _quantile_alerts())
def test_the_highest_finite_bucket_lies_above_the_alert_threshold(
    alert: str, metric: str, threshold: float
) -> None:
    buckets = _histograms().get(metric)
    if buckets is None:
        pytest.skip(f"{metric} is not a histogram this package defines")
    assert max(buckets) > threshold, (
        f"{alert} fires above {threshold}, but {metric}'s highest finite bucket is "
        f"{max(buckets)}: histogram_quantile can never exceed it"
    )


def test_one_failing_component_has_its_own_alert() -> None:
    """The fleet-wide ratio hides a low-traffic component that fails every request.

    With 96 requests succeeding elsewhere and 4 failing on one route, the
    global success ratio is 0.96 and ``OmniMercuryDetectionSuccessRateDrop``
    (below 0.95) is silent. ``OmniMercuryDetectorFailing`` takes the ratio per
    ``detector_type``, supplies a zero where a component has no success
    series, and requires a minimum number of requests.
    """
    rules = {
        rule["alert"]: rule
        for group in yaml.safe_load(_RULES.read_text())["groups"]
        for rule in group["rules"]
        if "alert" in rule
    }
    assert 96 / 100 >= 0.95  # the case the global alert cannot see

    expr = " ".join(str(rules["OmniMercuryDetectorFailing"]["expr"]).split())
    assert expr.count("sum by (detector_type)") == 4
    assert "omni_detection_success_total" in expr
    assert "or sum by (detector_type) (rate(omni_detection_requests_total" in expr
    assert "* 0 )" in expr or "* 0)" in expr
    assert "< 0.80 and" in expr
    assert "increase(omni_detection_requests_total" in expr
    assert ">= 5" in expr
    assert (
        "{{ $labels.detector_type }}"
        in rules["OmniMercuryDetectorFailing"]["annotations"]["summary"]
    )

    runbook = rules["OmniMercuryDetectorFailing"]["annotations"]["runbook_url"].rsplit("/", 1)[1]
    text = (_RULES.parents[2] / "docs" / "runbooks" / runbook).read_text(encoding="utf-8")
    assert "OmniMercuryDetectorFailing" in text

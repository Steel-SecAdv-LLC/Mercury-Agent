# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ThreatDetector measured against the OWASP CRS v4.7.0 regression corpus.

Baseline before the tightening (measured in-session, same corpus): recall
0.131 (SQLi) / 0.161 (XSS) / 0.296 (traversal); hit rate on the CRS
false-positive-regression payloads 0.065 (SQLi) / 0.235 (XSS); and the
retired rules flagged prose ("red or blue = fine") as SQL injection and a
bare parameter ("onion=3") as XSS.

This suite pins the corrected operating point: recall floors at the new
measured values (minus nothing -- the corpus is fixed, the detector is
deterministic) and false-positive ceilings below the old rates, plus the
named benign phrases that must never fire again.  See
``tests/fixtures/crs/README.md`` for extraction provenance and the caveat
that CRS ``neg`` payloads bound the false-positive rate from above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.security.threat_detection import ThreatDetector

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "crs" / "crs_v470_payloads.json"
CORPUS = json.loads(FIXTURE.read_text(encoding="utf-8"))

#: (family, detect method name, recall floor, neg-hit ceiling).  Floors are
#: the measured values at the tightened operating point; ceilings sit under
#: the retired rules' measured rates (0.065 SQLi / 0.235 XSS).
OPERATING_POINTS = [
    ("sqli", "detect_sql_injection", 0.18, 0.03),
    ("xss", "detect_xss", 0.18, 0.18),
    ("traversal", "detect_path_traversal", 0.33, 0.0),
]


@pytest.fixture(scope="module")
def detector() -> ThreatDetector:
    return ThreatDetector()


@pytest.mark.parametrize(("family", "method", "recall_floor", "fp_ceiling"), OPERATING_POINTS)
def test_measured_operating_point_holds(
    detector: ThreatDetector, family: str, method: str, recall_floor: float, fp_ceiling: float
) -> None:
    """Recall never regresses below, FP rate never rises above, the record."""
    detect = getattr(detector, method)
    positives: list[str] = CORPUS[family]["pos"]
    negatives: list[str] = CORPUS[family]["neg"]
    recall = sum(1 for p in positives if detect(p)["is_threat"]) / len(positives)
    fp_rate = (
        sum(1 for p in negatives if detect(p)["is_threat"]) / len(negatives) if negatives else 0.0
    )
    assert recall >= recall_floor, f"{family}: recall {recall:.3f} fell below {recall_floor}"
    assert fp_rate <= fp_ceiling + 1e-9, f"{family}: FP rate {fp_rate:.3f} above {fp_ceiling}"


@pytest.mark.parametrize(
    ("payload", "method"),
    [
        ("red or blue = fine", "detect_sql_injection"),
        ("the colour is red or blue = either works", "detect_sql_injection"),
        ("onion=3", "detect_xss"),
        ("season=4", "detect_xss"),
        ("onions and shallots", "detect_xss"),
    ],
)
def test_benign_phrases_never_fire(detector: ThreatDetector, payload: str, method: str) -> None:
    """The two reported false positives (and neighbours) stay closed."""
    assert getattr(detector, method)(payload)["is_threat"] is False


@pytest.mark.parametrize(
    ("payload", "method"),
    [
        ("' OR '1'='1", "detect_sql_injection"),
        ("1 OR 1=1--", "detect_sql_injection"),
        ("admin'--", "detect_sql_injection"),
        ("%27%20OR%20%271%27%3D%271", "detect_sql_injection"),
        ("1; DROP TABLE users", "detect_sql_injection"),
        ("1 AND SLEEP(5)", "detect_sql_injection"),
        ("<img src=x onerror=alert(1)>", "detect_xss"),
        ('" onmouseover=alert(document.cookie)', "detect_xss"),
        ("<svg/onload=alert(1)>", "detect_xss"),
        ("javascript:alert(1)", "detect_xss"),
        ("../../etc/passwd", "detect_path_traversal"),
        ("..%2f..%2fetc%2fpasswd", "detect_path_traversal"),
    ],
)
def test_classic_attacks_still_fire(detector: ThreatDetector, payload: str, method: str) -> None:
    """Tightening must not lose the canonical attack forms."""
    assert getattr(detector, method)(payload)["is_threat"] is True


def test_decision_source_carries_the_measurement(detector: ThreatDetector) -> None:
    calibrated: Any = detector.decision_source[0]
    assert calibrated.kind == "calibrated"
    assert "v4.7.0" in calibrated.description or "v4.7.0" in calibrated.calibration_data

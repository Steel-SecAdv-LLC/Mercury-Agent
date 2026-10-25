# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ThreatDetector`` decides exactly what its regexes decide, in linear time.

The released patterns are ``a.*b`` searches, which ``re.search`` retries from
every start: 20k quotes took 0.23 s for ``'.*--`` and 200k took 21.9 s
(measured), on the engine's ``detect_security_threat`` surface.  The linear matchers must agree
with ``re.search(pattern, payload, re.IGNORECASE)`` on every payload.
"""

from __future__ import annotations

import random
import re
import time

import pytest

from omni_mercury_engine.security import threat_detection
from omni_mercury_engine.security.threat_detection import ThreatDetector

DETECTOR = ThreatDetector()
#: The two semantic rules (boolean tautology, CRS-941160-style event
#: handler) are keyed by descriptive names, not regexes: their behaviour is
#: pinned by tests/security/test_threat_crs_corpus.py, so the
#: regex-equivalence fuzz below covers every *regex* pattern.
_SEMANTIC = {
    threat_detection._SQL_OR_TAUTOLOGY_PATTERN,
    threat_detection._XSS_EVENT_HANDLER_PATTERN,
}
RELEASED = [
    pattern
    for pattern in DETECTOR.sql_patterns + DETECTOR.xss_patterns + DETECTOR.path_traversal_patterns
    if pattern not in _SEMANTIC
]

#: Fragments that exercise every boundary of the patterns: case, word
#: boundaries, line breaks, partial tokens and Unicode case folding.
ALPHABET = [
    "<script",
    "<SCRIPT ",
    "<scrip",
    ">",
    "</script>",
    "</SCRIPT>",
    "\n",
    " ",
    "\t",
    "x",
    "_",
    "1",
    "'",
    "--",
    "-",
    "OR",
    "or",
    "Or ",
    "xor",
    "=",
    ";",
    "DROP",
    "drop",
    "TABLE",
    "tables",
    "UNION",
    "SELECT",
    "EXEC",
    "exec",
    "(",
    "on",
    "ON",
    "onion",
    "button",
    "buttons",
    "javascript:",
    "JavaScript:",
    "<iframe",
    "../",
    "..\\",
    "%2e%2e/",
    "%2E%2E\\",
    "é",
    "ſ",
    "K",
    "unıon",
]


def test_every_released_pattern_has_a_linear_matcher() -> None:
    # Patterns added for the CRS calibration round (sleep/benchmark/...)
    # carry no ``a.*b`` ReDoS shape, so they are searched directly with
    # ``re`` (the fallback) and need no dedicated linear matcher; every
    # pre-existing ``a.*b`` pattern still has one, and the semantic rules
    # dispatch to their own functions.
    assert set(threat_detection._LINEAR_MATCHERS) - _SEMANTIC <= set(RELEASED) | {
        r"(\bOR\b.*=.*)",
        r"('.*--)",
        r"on\w+\s*=",
        r"javascript:",
    }


@pytest.mark.parametrize("seed", range(4))
def test_linear_matchers_agree_with_re_search(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(5000):
        payload = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 16)))
        for pattern in RELEASED:
            expected = re.search(pattern, payload, re.IGNORECASE) is not None
            assert threat_detection._matches(pattern, payload) is expected, (pattern, payload)


@pytest.mark.parametrize(
    "payload",
    [
        "' OR '1'='1",
        "1; DROP TABLE users",
        "x UNION SELECT password FROM users",
        "EXEC xp_cmdshell('dir')",
        "admin'--",
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        "javascript:alert(1)",
        "<iframe src=//x>",
        "../../etc/passwd",
        "..\\..\\windows",
        "%2e%2e/etc",
        "<script\n>\n  <script>  </script>",  # a later tag end on the closing line
        "<script>\n</script>",  # closing tag on another line: no match
        "the quick brown fox",
    ],
)
def test_known_payloads_agree_with_re_search(payload: str) -> None:
    for pattern in RELEASED:
        expected = re.search(pattern, payload, re.IGNORECASE) is not None
        assert threat_detection._matches(pattern, payload) is expected, pattern


@pytest.mark.parametrize(
    "payload",
    [
        "'" * 200_000,
        "OR " * 70_000,
        "<script" * 30_000,
        "<script>" * 25_000 + "\n</script>",
        "on" * 100_000,
        ("w" + " " * 50) * 4_000,
    ],
    ids=["quotes", "ors", "openings", "openings-then-close", "on-run", "spaces"],
)
def test_adversarial_payload_is_linear(payload: str) -> None:
    """200k characters: about 30 ms here; the regexes took seconds to tens of seconds."""
    start = time.perf_counter()
    DETECTOR.detect_all(payload)
    assert time.perf_counter() - start < 2.0


def test_added_pattern_is_still_searched_with_re() -> None:
    detector = ThreatDetector()
    detector.sql_patterns.append(r"\bWAITFOR\b\s+DELAY")
    result = detector.detect_sql_injection("1; WAITFOR DELAY '0:0:5'")
    assert result["is_threat"]
    assert r"\bWAITFOR\b\s+DELAY" in result["matched_patterns"]


def test_results_keep_the_released_shape() -> None:
    result = DETECTOR.detect_sql_injection("' OR '1'='1' --")
    assert result["matched_patterns"] == [
        threat_detection._SQL_OR_TAUTOLOGY_PATTERN,
        r"('\s*--)",
    ]
    assert result["confidence"] == pytest.approx(2 / len(DETECTOR.sql_patterns))

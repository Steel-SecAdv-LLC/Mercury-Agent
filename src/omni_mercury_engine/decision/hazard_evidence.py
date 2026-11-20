# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What has been measured of decisions on each hazard, on real events held out.

A decision record says "anomaly, 90% coverage" in the same words for every
hazard. Whether that holds on the next real event of a hazard is a matter of
measurement, and the measurement differs a great deal from one hazard to
another. ``research/hazard_calibration`` makes it: for each hazard with more
than one real event, each event is judged by a calibration and a conformal
certificate fitted on the hazard's other events, with the decision layer as
it ships. This module serves what was found, so that a decision about a
hazard can be read beside the record of decisions about that hazard.

What was measured is the generic detector's score put through that
calibration, the certificate and the decision layer. It is not a measurement
of a certificate the fusion pipeline attaches from its own calibration data,
and :func:`caveat` says so each time.

The evidence is ``hazard_evidence.json`` beside this file, written by
``research/hazard_calibration/measure.py --write``. If it cannot be read,
nothing is claimed: every function here answers as for a hazard with no
measurement.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["caveat", "for_hazard", "published"]

_PATH = Path(__file__).with_name("hazard_evidence.json")


@functools.lru_cache(maxsize=1)
def published() -> dict[str, Any]:
    """The whole of the published evidence, or ``{}`` if it cannot be read."""
    try:
        evidence = json.loads(_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("the hazard evidence could not be read (%s); none is claimed", exc)
        return {}
    return evidence if isinstance(evidence.get("hazards"), dict) else {}


def for_hazard(hazard: str | None) -> dict[str, Any] | None:
    """What was measured for one hazard.

    Returns:
        The hazard's block of the evidence, with ``measured_on`` and
        ``coverage_target``; ``{"not_fetched": True, ...}`` for a hazard
        whose events could not be read when the measurement was made; and
        ``None`` for a name the measurement knows nothing of.
    """
    evidence = published()
    # The domain reaches here from ``Evidence.from_detection``, which reads
    # ``result["domain"]`` as-is — a library caller (or a fuzzed result)
    # can hand a non-string. Coerce defensively: a non-string domain names
    # no measured hazard, so it is "unknown", never an AttributeError on a
    # decision path.
    name = (hazard if isinstance(hazard, str) else "").strip().lower()
    if not name or not evidence:
        return None
    common = {
        "hazard": name,
        "measured_on": evidence.get("measured_on"),
        "coverage_target": evidence.get("coverage_target"),
    }
    told = evidence["hazards"].get(name)
    if told is not None:
        return {**common, **told}
    if name in evidence.get("not_fetched", ()):
        return {**common, "not_fetched": True}
    return None


def caveat(hazard: str | None) -> str | None:
    """One sentence for a decision record about ``hazard``, or ``None`` if nothing is known.

    For a hazard that was measured it gives how often the certificate held
    the true label on held-out real events and how much was abstained on;
    for one that could not be, it says that no such record exists.
    """
    told = for_hazard(hazard)
    if told is None:
        return None
    name, source = told["hazard"], "research/hazard_calibration"
    if told.get("not_fetched"):
        return (
            f"no held-out record exists for {name} decisions: its real events could not be "
            f"read when the measurement was made ({source}, {told['measured_on']})"
        )
    if "not_measured" in told:
        return (
            f"no held-out record exists for {name} decisions: {told['not_measured']} "
            f"({source}, {told['measured_on']})"
        )
    certificate, decision = told["certificate"], told["decision"]
    return (
        f"held-out record for {name}: on {len(told['events'])} real events, each judged by a "
        f"calibration fitted on the others, sets asked for {told['coverage_target']:.0%} "
        f"coverage held the true label {certificate['coverage']:.1%} of the time and "
        f"{decision['abstained']:.1%} of rows were abstained on ({source}, "
        f"{told['measured_on']}; the generic detector's score, not this result's own "
        "certificate)"
    )

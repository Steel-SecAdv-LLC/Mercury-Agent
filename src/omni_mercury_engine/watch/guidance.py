# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What to do: the protective action for each kind of finding.

A finding that says a river is in major flood and stops there leaves the
person reading it informed and unprepared. Each kind of finding carries the
action the responsible agency publishes for that hazard, attributed to it.
Mercury does not write safety advice of its own: a National Weather Service
warning carries the Service's own instruction, word for word, and the rest
restate the standing public guidance of the agency named beside them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_mercury_engine.watch.model import Finding

__all__ = ["EMERGENCY_LINE", "GUIDANCE", "with_guidance"]

#: Said with every delivered finding.
EMERGENCY_LINE = (
    "Mercury reports what public agencies measure and say; it does not issue warnings. "
    "Follow your local officials, and in an emergency call your emergency number "
    "(911 in the United States)."
)

#: Finding kind to (what to do, whose guidance it is).
GUIDANCE: dict[str, tuple[str, str]] = {
    "earthquake": (
        "If you are in the area and feel shaking: drop, cover and hold on until it stops. "
        "Expect aftershocks. If you are on a coast and the shaking was strong or lasted a "
        "long time, go to high ground or inland at once; do not wait for a warning.",
        "USGS and Ready.gov earthquake guidance",
    ),
    "river_flood": (
        "Never walk or drive through flood water: turn around, don't drown. If water is "
        "rising where you are, move to higher ground. Follow your National Weather Service "
        "forecast office and local emergency officials.",
        "National Weather Service flood safety guidance",
    ),
    "volcano_alert": (
        "Follow the volcano observatory's notices and your local emergency management. If "
        "ash falls: stay indoors, keep ash out of buildings, engines and water supplies, "
        "and avoid driving.",
        "USGS Volcano Hazards Program guidance",
    ),
    "buoy_event_mode": (
        "This is a buoy's detection, not a warning: official tsunami messages are at "
        "tsunami.gov. If you are on a coast and feel strong or long shaking, see the sea "
        "draw back or hear it roar, go to high ground or inland immediately.",
        "NOAA National Weather Service tsunami safety guidance",
    ),
    "geomagnetic_storm": (
        "Most people need do nothing. Strong storms can disturb power grids, satellite "
        "navigation and HF radio; their operators act on NOAA's alerts. Aurora may be "
        "visible further from the poles than usual.",
        "NOAA Space Weather Prediction Center",
    ),
    "declarations": (
        "If you live in a declared area you may be eligible for federal assistance: apply "
        "at DisasterAssistance.gov or call the FEMA Helpline, 1-800-621-3362.",
        "Federal Emergency Management Agency",
    ),
    "nws_alert": (
        "Follow the instructions of the National Weather Service and your local officials.",
        "National Weather Service",
    ),
    # An alert whose authority gave an instruction carries that, word for word.
    # These say only whom to follow, where the source gave none.
    "cap_alert": (
        "Follow the warnings and instructions of the authority that issued this alert and "
        "of your local emergency officials.",
        "no instruction accompanied the alert",
    ),
    "gdacs_event": (
        "GDACS does not issue public warnings. Follow the warnings and instructions of your "
        "national and local emergency authorities.",
        "no public instruction accompanies a GDACS alert",
    ),
}


def with_guidance(finding: Finding) -> Finding:
    """``finding`` with its protective action, unless its feed already gave one."""
    if finding.action:
        return finding
    action, source = GUIDANCE.get(finding.kind, ("", ""))
    return replace(finding, action=action, action_source=source) if action else finding

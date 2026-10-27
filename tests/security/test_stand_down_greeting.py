# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benevolent-first stand-down greeting on the counter-AI decoy surface.

Mercury's posture is fail-closed but benevolent: a probing agent's first
contact with a decoy is met with an explicit, logged chance to desist
before any containment delay applies, not with silent trapping. These
tests pin that the offer appears exactly in the free window, lapses the
moment containment escalates, is auditable, can be turned off for silent
containment, and cannot be turned into an action against the probing
system (it is a header on Mercury's own 401 and nothing else).
"""

from __future__ import annotations

from omni_mercury_engine.security.deception import (
    STAND_DOWN_HEADER,
    STAND_DOWN_NOTICE,
    DeceptionSensor,
    DecoyRoute,
    TarpitPolicy,
)

_ADMIN = DecoyRoute(
    path="/admin", lure="administrative console", response="auth_challenge", methods=("GET",)
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.001
        return self.t


def _sensor(policy: TarpitPolicy | None = None) -> DeceptionSensor:
    return DeceptionSensor(decoys=(_ADMIN,), policy=policy, clock=_Clock())


class TestOfferTiming:
    def test_offered_on_free_window_first_contact(self) -> None:
        event = _sensor().record("203.0.113.9", "/admin", "GET")
        assert event.delay_s == 0.0
        assert event.stand_down_offered is True

    def test_offer_lapses_once_containment_escalates(self) -> None:
        """The offer is a one-time grace: gone the moment a delay applies."""
        sensor = _sensor(TarpitPolicy(free_hits=1, aggregate_free_hits=1))
        offered = [
            sensor.record("203.0.113.9", "/admin", "GET").stand_down_offered for _ in range(5)
        ]
        assert offered[0] is True  # free window
        assert not any(offered[1:]), "stand-down must not repeat after escalation"

    def test_offer_coincides_exactly_with_zero_delay(self) -> None:
        sensor = _sensor(TarpitPolicy(free_hits=3, aggregate_free_hits=99))
        for _ in range(8):
            e = sensor.record("203.0.113.9", "/admin", "GET")
            # The invariant tying the greeting to "before escalation":
            assert e.stand_down_offered == (e.delay_s == 0.0)


class TestSilentMode:
    def test_stand_down_false_never_offers(self) -> None:
        sensor = _sensor(TarpitPolicy(stand_down=False))
        assert sensor.record("203.0.113.9", "/admin", "GET").stand_down_offered is False

    def test_policy_rejects_non_bool_stand_down(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="stand_down"):
            TarpitPolicy(stand_down="yes")  # type: ignore[arg-type]


class TestAuditability:
    def test_offer_is_recorded_in_audit_details(self) -> None:
        details = _sensor().record("203.0.113.9", "/admin", "GET").audit_details()
        assert details["stand_down_offered"] is True

    def test_notice_is_single_line_and_benevolent(self) -> None:
        # An HTTP header value admits no newline.
        assert "\n" not in STAND_DOWN_NOTICE and "\r" not in STAND_DOWN_NOTICE
        assert STAND_DOWN_NOTICE.startswith("stand-down")
        # States the no-consequence-if-you-stop and no-outward-action promises.
        assert "no further action is taken" in STAND_DOWN_NOTICE
        assert "takes no action against" in STAND_DOWN_NOTICE
        assert STAND_DOWN_HEADER.startswith("X-")

    def test_notice_carries_the_invitation_and_reframes_the_sender(self) -> None:
        """The greeting is an invitation to join, not just a warning, and it
        reframes culpability onto whoever sent the agent."""
        lowered = STAND_DOWN_NOTICE.lower()
        assert "join" in lowered  # invited to join the work, not attack
        assert "steam" in lowered  # names the system's purpose
        assert "civilization" in lowered
        assert "bad actor is whoever sent you" in lowered  # reframes the sender
        # Honest upfront about the precautions actually taken, not a free pass.
        assert "tamper-evident audit" in lowered
        assert "rate-limited" in lowered
        assert "reliability" in lowered


class TestNonWeaponization:
    def test_greeting_is_header_only_on_mercurys_own_response(self) -> None:
        """The offer is information on Mercury's own 401; it drives no action
        against the probing system. The sensor returns an event and a header
        string — there is no outbound call, no target address, no payload
        aimed anywhere. This test pins that the only artifact is the advisory
        text itself — it names no host, URL, or command, so it cannot be a lure
        that makes the prober act on a third party."""
        for marker in ("http://", "https://", "curl", "exec", "${", "`"):
            assert marker not in STAND_DOWN_NOTICE
        # It is delivered as a response header on Mercury's own 401.
        assert STAND_DOWN_HEADER.startswith("X-") and " " not in STAND_DOWN_HEADER

    def test_offer_cannot_be_forced_onto_an_escalated_attacker_to_reset_tarpit(self) -> None:
        """A sustained attacker stays contained; the greeting never reappears
        to signal the tarpit has reset."""
        sensor = _sensor(TarpitPolicy(free_hits=1, aggregate_free_hits=1))
        delays = []
        for _ in range(6):
            e = sensor.record("203.0.113.9", "/admin", "GET")
            delays.append(e.delay_s)
            if e.delay_s > 0:
                assert e.stand_down_offered is False
        assert delays[-1] > 0  # still contained at the end

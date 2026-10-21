# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The deception core: bounded, unforgeable keying, honest reliability evidence."""

from __future__ import annotations

import asyncio
import logging
import math

import pytest

from omni_mercury_engine.security import deception
from omni_mercury_engine.security.counterintelligence import (
    MAX_OUTCOME_WEIGHT,
    SourceOutcome,
    source_posterior,
)
from omni_mercury_engine.security.deception import (
    DECEPTION_ENABLED_ENV,
    DEFAULT_DECOYS,
    DeceptionSensor,
    DecoyRoute,
    TarpitGate,
    TarpitPolicy,
    TripwireEvent,
    deception_enabled,
    source_key,
)

ADMIN = "/api/v1/admin"
KEYS = "/api/v1/admin/api-keys"
ENV_FILE = "/.env"


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        """Start at ``start`` seconds."""
        self.now = start

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


class TestDeceptionEnabled:
    """Opt-in only: anything but an explicit truthy value is off."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy(self, value: str) -> None:
        """Explicit truthy values enable."""
        assert deception_enabled({DECEPTION_ENABLED_ENV: value}) is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "enabled", "2"])
    def test_falsy(self, value: str) -> None:
        """Everything else disables."""
        assert deception_enabled({DECEPTION_ENABLED_ENV: value}) is False

    def test_unset_is_off(self) -> None:
        """Absent means off."""
        assert deception_enabled({}) is False

    def test_reads_process_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default mapping is the process environment."""
        monkeypatch.setenv(DECEPTION_ENABLED_ENV, "true")
        assert deception_enabled() is True
        monkeypatch.delenv(DECEPTION_ENABLED_ENV)
        assert deception_enabled() is False


class TestSourceKey:
    """Identity that an attacker cannot reset by rotating addresses."""

    def test_ipv4_is_the_address(self) -> None:
        """IPv4 passes through canonically."""
        assert source_key("203.0.113.9") == "203.0.113.9"

    def test_ipv6_is_its_slash_64(self) -> None:
        """IPv6 keys by /64 network."""
        assert source_key("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == "2001:db8:1:2::/64"

    def test_rotation_inside_a_slash_64_is_one_source(self) -> None:
        """Two hosts in one /64 share a key; the next /64 does not."""
        assert source_key("2001:db8:1:2::1") == source_key("2001:db8:1:2:ffff::9")
        assert source_key("2001:db8:1:2::1") != source_key("2001:db8:1:3::1")

    def test_ipv4_mapped_collapses_to_ipv4(self) -> None:
        """One client never gets two identities."""
        assert source_key("::ffff:203.0.113.9") == "203.0.113.9"

    def test_scoped_ipv6(self) -> None:
        """A zone identifier does not break keying."""
        assert source_key("fe80::1%eth0") == "fe80::/64"

    @pytest.mark.parametrize("value", ["unknown", "testclient", "", "999.1.1.1", "1.2.3.4/24"])
    def test_non_literals_share_unknown(self, value: str) -> None:
        """Anything not an IP literal throttles together, never bypasses."""
        assert source_key(value) == "unknown"


class TestTarpitPolicy:
    """Bounded, overflow-proof exponential schedule."""

    @pytest.mark.parametrize(
        ("hits", "delay"), [(0, 0.0), (-3, 0.0), (1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 8.0)]
    )
    def test_default_schedule(self, hits: int, delay: float) -> None:
        """0.5, 1, 2, 4, 8 seconds."""
        assert TarpitPolicy().delay_for(hits) == delay

    @pytest.mark.parametrize("hits", [6, 64, 10**6, 10**18])
    def test_capped_without_overflow(self, hits: int) -> None:
        """Any hit count saturates at the cap; none can overflow."""
        assert TarpitPolicy().delay_for(hits) == 8.0

    def test_growth_one_is_constant(self) -> None:
        """No growth: every hit waits the base delay."""
        policy = TarpitPolicy(base_delay_s=0.3, growth=1.0, max_delay_s=5.0)
        assert [policy.delay_for(n) for n in (1, 2, 50)] == [0.3, 0.3, 0.3]

    def test_cap_equal_to_base(self) -> None:
        """A cap equal to the base pins every hit at it."""
        policy = TarpitPolicy(base_delay_s=2.0, max_delay_s=2.0)
        assert [policy.delay_for(n) for n in (1, 2, 9)] == [2.0, 2.0, 2.0]

    def test_just_below_the_cap(self) -> None:
        """The last uncapped step is exact; the next one caps."""
        policy = TarpitPolicy(base_delay_s=1.0, growth=3.0, max_delay_s=10.0)
        assert policy.delay_for(3) == 9.0
        assert policy.delay_for(4) == 10.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_delay_s": 0.0},
            {"base_delay_s": -1.0},
            {"growth": 0.99},
            {"max_delay_s": 0.1},
            {"window_s": 0.0},
            {"max_concurrent": -1},
            {"base_delay_s": math.nan},
            {"growth": math.inf},
            {"window_s": math.inf},
        ],
    )
    def test_invalid_refused(self, kwargs: dict[str, float]) -> None:
        """Out-of-range or non-finite parameters are refused."""
        with pytest.raises(ValueError):
            TarpitPolicy(**kwargs)  # type: ignore[arg-type]

    def test_zero_concurrency_is_legal(self) -> None:
        """max_concurrent=0 disables holding without disabling the tripwire."""
        assert TarpitPolicy(max_concurrent=0).max_concurrent == 0


class TestDecoyRoute:
    """Decoys are exact, absolute, and honestly described."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"path": "api/v1/admin", "lure": "x"},
            {"path": "/api/ v1", "lure": "x"},
            {"path": "/api/{item}", "lure": "x"},
            {"path": "/api/item}", "lure": "x"},
            {"path": "/ok", "lure": "  "},
            {"path": "/ok", "lure": "x", "response": "teapot"},
            {"path": "/ok", "lure": "x", "methods": ()},
            {"path": "/ok", "lure": "x", "methods": ("GET", "GET")},
            {"path": "/ok", "lure": "x", "methods": ("TRACE",)},
            {"path": "/ok", "lure": "x", "methods": ("get",)},
        ],
    )
    def test_invalid_refused(self, kwargs: dict[str, object]) -> None:
        """Every malformed decoy is refused at construction."""
        with pytest.raises(ValueError):
            DecoyRoute(**kwargs)  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        """An auth-challenge decoy on GET and POST by default."""
        decoy = DecoyRoute("/ok", "lure")
        assert decoy.response == "auth_challenge"
        assert decoy.methods == ("GET", "POST")

    def test_shipped_decoys(self) -> None:
        """Unique paths; silent tripwires answer every method (no telltale 405)."""
        paths = [d.path for d in DEFAULT_DECOYS]
        assert len(paths) == len(set(paths))
        assert "/api/v1/gate/override" in paths
        silent = [d for d in DEFAULT_DECOYS if d.response == "not_found"]
        assert {d.path for d in silent} == {"/.env", "/.git/config"}
        for decoy in silent:
            assert set(decoy.methods) == {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"}


class TestSensorRecording:
    """Tripwire accounting under a controlled clock."""

    def test_unknown_path_refused(self) -> None:
        """Only registered decoys are recordable."""
        with pytest.raises(ValueError, match="not a registered decoy"):
            DeceptionSensor().record("203.0.113.9", "/api/v1/detect", "GET")

    def test_first_event(self) -> None:
        """First touch: one hit, the base delay, one distinct decoy."""
        event = DeceptionSensor(clock=FakeClock()).record("203.0.113.9", ADMIN, "get")
        assert event == TripwireEvent(
            source_key="203.0.113.9",
            client_ip="203.0.113.9",
            path=ADMIN,
            lure="administrative console",
            method="GET",
            hits_in_window=1,
            distinct_decoys=1,
            first_touch=True,
            delay_s=0.5,
            credential_presented=False,
        )

    def test_escalation_across_decoys(self) -> None:
        """Hits count across all decoys; distinct decoys count breadth."""
        sensor = DeceptionSensor(clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        sensor.record("203.0.113.9", ADMIN, "GET")
        event = sensor.record("203.0.113.9", KEYS, "POST")
        assert event.hits_in_window == 3
        assert event.delay_s == 2.0
        assert event.distinct_decoys == 2
        assert event.first_touch is True

    def test_repeat_is_not_a_first_touch(self) -> None:
        """A second touch of the same decoy is not a first touch."""
        sensor = DeceptionSensor(clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert sensor.record("203.0.113.9", ADMIN, "GET").first_touch is False

    def test_window_expiry_resets_the_tarpit(self) -> None:
        """Hits older than the window stop counting; exactly-window-old still counts."""
        clock = FakeClock()
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=100.0), clock=clock)
        sensor.record("203.0.113.9", ADMIN, "GET")
        clock.now += 100.0
        assert sensor.record("203.0.113.9", ADMIN, "GET").hits_in_window == 2
        clock.now += 100.5
        event = sensor.record("203.0.113.9", ADMIN, "GET")
        assert event.hits_in_window == 1
        assert event.delay_s == 0.5

    def test_sources_are_independent(self) -> None:
        """One prober's escalation never touches another source."""
        sensor = DeceptionSensor(clock=FakeClock())
        for _ in range(4):
            sensor.record("203.0.113.9", ADMIN, "GET")
        assert sensor.record("198.51.100.7", ADMIN, "GET").hits_in_window == 1

    def test_ipv6_rotation_shares_the_tarpit(self) -> None:
        """Rotating addresses inside one /64 does not reset escalation."""
        sensor = DeceptionSensor(clock=FakeClock())
        sensor.record("2001:db8:1:2::1", ADMIN, "GET")
        sensor.record("2001:db8:1:2::2", ADMIN, "GET")
        assert sensor.record("2001:db8:1:2::3", ADMIN, "GET").hits_in_window == 3

    def test_hit_history_is_bounded(self) -> None:
        """The per-source history never exceeds max_hits_tracked."""
        sensor = DeceptionSensor(max_hits_tracked=3, clock=FakeClock())
        for _ in range(10):
            event = sensor.record("203.0.113.9", ADMIN, "GET")
        assert event.hits_in_window == 3

    def test_method_normalised_and_clipped(self) -> None:
        """Method is upper-cased and length-capped."""
        event = DeceptionSensor(clock=FakeClock()).record("203.0.113.9", ADMIN, "p" * 40)
        assert event.method == "P" * 16

    def test_credential_presented_is_recorded_as_a_boolean(self) -> None:
        """The signal survives into the event and its audit details; nothing else does."""
        event = DeceptionSensor(clock=FakeClock()).record(
            "203.0.113.9", ADMIN, "GET", credential_presented=True
        )
        assert event.credential_presented is True
        assert event.audit_details()["credential_presented"] is True

    def test_audit_details_omit_the_address(self) -> None:
        """The client address never lands in free-text details."""
        details = (
            DeceptionSensor(clock=FakeClock()).record("203.0.113.9", ADMIN, "GET").audit_details()
        )
        assert "203.0.113.9" not in str(details)
        assert details == {
            "decoy": ADMIN,
            "lure": "administrative console",
            "method": "GET",
            "hits_in_window": 1,
            "distinct_decoys": 1,
            "first_touch": True,
            "tarpit_delay_s": 0.5,
            "credential_presented": False,
        }


class TestAuditVolume:
    """One record per source per decoy per window: scanners cannot flood the chain."""

    def test_first_touch_audited_repeats_suppressed(self) -> None:
        """Repeats inside the window are not re-audited."""
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(audit_sink=events.append, clock=FakeClock())
        for _ in range(5):
            sensor.record("203.0.113.9", ADMIN, "GET")
        assert len(events) == 1
        assert events[0].first_touch is True

    def test_each_decoy_audited_once(self) -> None:
        """Breadth is recorded: each new decoy gets its own record."""
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(audit_sink=events.append, clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        sensor.record("203.0.113.9", KEYS, "GET")
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert [e.path for e in events] == [ADMIN, KEYS]

    def test_re_audited_after_the_window(self) -> None:
        """Sustained probing is recorded again once per window."""
        events: list[TripwireEvent] = []
        clock = FakeClock()
        sensor = DeceptionSensor(
            policy=TarpitPolicy(window_s=60.0), audit_sink=events.append, clock=clock
        )
        sensor.record("203.0.113.9", ADMIN, "GET")
        clock.now += 60.0
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert len(events) == 1
        clock.now += 0.5
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert len(events) == 2
        assert events[1].first_touch is False

    def test_sink_failure_never_breaks_recording(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing sink is logged; the event is still returned."""

        def broken(event: TripwireEvent) -> None:
            raise RuntimeError("audit store down")

        sensor = DeceptionSensor(audit_sink=broken, clock=FakeClock())
        with caplog.at_level(logging.ERROR, logger=deception.__name__):
            event = sensor.record("203.0.113.9", ADMIN, "GET")
        assert event.hits_in_window == 1
        assert "audit sink failed" in caplog.text


class TestBoundedState:
    """Memory is capped no matter how many sources probe."""

    def test_lru_eviction(self) -> None:
        """Past max_sources the least recently seen source is evicted."""
        sensor = DeceptionSensor(max_sources=2, clock=FakeClock())
        sensor.record("203.0.113.1", ADMIN, "GET")
        sensor.record("203.0.113.2", ADMIN, "GET")
        sensor.record("203.0.113.1", ADMIN, "GET")  # refresh .1
        sensor.record("203.0.113.3", ADMIN, "GET")  # evicts .2
        assert sensor.tracked_sources() == 2
        assert sensor.outcomes("203.0.113.2") == ()
        assert len(sensor.outcomes("203.0.113.1")) == 1
        assert len(sensor.outcomes("203.0.113.3")) == 1

    def test_single_slot(self) -> None:
        """max_sources=1 keeps exactly the newest source."""
        sensor = DeceptionSensor(max_sources=1, clock=FakeClock())
        sensor.record("203.0.113.1", ADMIN, "GET")
        sensor.record("203.0.113.2", ADMIN, "GET")
        assert sensor.tracked_sources() == 1
        assert sensor.outcomes("203.0.113.1") == ()


class TestReliabilityBridge:
    """Tripwires become honest evidence in the counterintelligence model."""

    def test_one_refuted_outcome_per_distinct_decoy(self) -> None:
        """Retries add nothing; breadth adds one refutation per decoy."""
        sensor = DeceptionSensor(clock=FakeClock())
        for _ in range(7):
            sensor.record("203.0.113.9", ADMIN, "GET")
        sensor.record("203.0.113.9", ENV_FILE, "GET")
        assert sensor.outcomes("203.0.113.9") == (
            SourceOutcome(confirmed=False, weight=1.0),
            SourceOutcome(confirmed=False, weight=1.0),
        )

    def test_untracked_source_has_no_outcomes(self) -> None:
        """A source that never touched a decoy has no evidence against it."""
        assert DeceptionSensor().outcomes("198.51.100.7") == ()

    def test_posterior_is_the_counterintelligence_posterior(self) -> None:
        """The bridge applies the module's own model, not a parallel one."""
        sensor = DeceptionSensor(outcome_weight=0.75, clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        sensor.record("203.0.113.9", KEYS, "GET")
        expected = source_posterior(
            "203.0.113.9", "C", (SourceOutcome(False, 0.75), SourceOutcome(False, 0.75))
        )
        assert sensor.posterior("203.0.113.9", "C") == expected

    def test_breadth_lowers_reliability(self) -> None:
        """More distinct decoys means a lower posterior mean."""
        sensor = DeceptionSensor(clock=FakeClock())
        baseline = sensor.posterior("203.0.113.9").mean
        sensor.record("203.0.113.9", ADMIN, "GET")
        one = sensor.posterior("203.0.113.9").mean
        sensor.record("203.0.113.9", KEYS, "GET")
        two = sensor.posterior("203.0.113.9").mean
        assert baseline == 0.5
        assert two < one < baseline

    @pytest.mark.parametrize("weight", [-0.1, MAX_OUTCOME_WEIGHT + 0.01, math.nan, math.inf])
    def test_invalid_weight_refused(self, weight: float) -> None:
        """Weights outside the module's admissible range are refused."""
        with pytest.raises(ValueError, match="outcome_weight"):
            DeceptionSensor(outcome_weight=weight)

    def test_boundary_weights_accepted(self) -> None:
        """0 and MAX_OUTCOME_WEIGHT are admissible."""
        DeceptionSensor(outcome_weight=0.0)
        DeceptionSensor(outcome_weight=MAX_OUTCOME_WEIGHT)


class TestSensorConstruction:
    """Construction refuses incoherent configurations."""

    def test_empty_decoys_refused(self) -> None:
        """At least one decoy."""
        with pytest.raises(ValueError, match="at least one decoy"):
            DeceptionSensor(decoys=())

    def test_duplicate_paths_refused(self) -> None:
        """Paths must be unique."""
        with pytest.raises(ValueError, match="unique"):
            DeceptionSensor(decoys=(DecoyRoute("/a", "x"), DecoyRoute("/a", "y")))

    @pytest.mark.parametrize("kwargs", [{"max_sources": 0}, {"max_hits_tracked": 0}])
    def test_capacities_must_be_positive(self, kwargs: dict[str, int]) -> None:
        """Capacities of zero are refused."""
        with pytest.raises(ValueError, match=">= 1"):
            DeceptionSensor(**kwargs)  # type: ignore[arg-type]

    def test_exposes_configuration(self) -> None:
        """Decoys and policy are readable."""
        policy = TarpitPolicy(base_delay_s=1.0)
        sensor = DeceptionSensor(decoys=DEFAULT_DECOYS[:1], policy=policy)
        assert sensor.decoys == DEFAULT_DECOYS[:1]
        assert sensor.policy is policy


class TestTarpitGate:
    """The tarpit can never consume the server."""

    def test_holds_the_scheduled_delay(self) -> None:
        """A positive delay under capacity is slept and reported."""
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        gate = TarpitGate(2, sleep=record)
        assert asyncio.run(gate.hold(1.5)) == 1.5
        assert slept == [1.5]
        assert gate.active == 0

    @pytest.mark.parametrize("delay", [0.0, -1.0, math.nan])
    def test_non_positive_delays_skip(self, delay: float) -> None:
        """Zero, negative and NaN delays never sleep."""
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        assert asyncio.run(TarpitGate(2, sleep=record).hold(delay)) == 0.0
        assert slept == []

    def test_at_capacity_answers_immediately(self) -> None:
        """Past the cap, responses are not held."""

        async def scenario() -> tuple[float, float, int]:
            release = asyncio.Event()
            started = asyncio.Event()

            async def blocking(seconds: float) -> None:
                started.set()
                await release.wait()

            gate = TarpitGate(1, sleep=blocking)
            first = asyncio.create_task(gate.hold(2.0))
            await started.wait()
            held = gate.active
            second = await gate.hold(3.0)
            release.set()
            return await first, second, held

        first, second, held = asyncio.run(scenario())
        assert (first, second, held) == (2.0, 0.0, 1)

    def test_zero_capacity_never_holds(self) -> None:
        """max_concurrent=0 never sleeps."""
        slept: list[float] = []

        async def record(seconds: float) -> None:
            slept.append(seconds)

        assert asyncio.run(TarpitGate(0, sleep=record).hold(1.0)) == 0.0
        assert slept == []

    def test_slot_released_when_sleep_fails(self) -> None:
        """A failing or cancelled hold always frees its slot."""

        async def failing(seconds: float) -> None:
            raise RuntimeError("cancelled")

        gate = TarpitGate(1, sleep=failing)
        with pytest.raises(RuntimeError):
            asyncio.run(gate.hold(1.0))
        assert gate.active == 0

    def test_negative_capacity_refused(self) -> None:
        """A negative cap is refused."""
        with pytest.raises(ValueError, match=">= 0"):
            TarpitGate(-1)

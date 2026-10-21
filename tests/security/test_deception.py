# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The deception core: bounded, unforgeable keying, honest reliability evidence."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine.security import deception
from omni_mercury_engine.security.counterintelligence import (
    MAX_OUTCOME_WEIGHT,
    SourceOutcome,
    source_posterior,
)
from omni_mercury_engine.security.deception import (
    DECEPTION_ENABLED_ENV,
    DECOY_PATHS_ENV,
    DEFAULT_DECOYS,
    SILENT_DECOY_PATHS_ENV,
    DeceptionSensor,
    DecoyRoute,
    TarpitGate,
    TarpitPolicy,
    TripwireEvent,
    aggregate_key,
    configured_decoys,
    deception_enabled,
    source_key,
)
from omni_mercury_engine.security.deception_store import (
    TRIPWIRE_STORE_PATH_ENV,
    SqliteTripwireStore,
    TouchRecord,
    build_tripwire_store,
)

if TYPE_CHECKING:
    from pathlib import Path

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
        ("hits", "delay"),
        [(0, 0.0), (-3, 0.0), (1, 0.0), (2, 0.5), (3, 1.0), (4, 2.0), (5, 4.0), (6, 8.0)],
    )
    def test_default_schedule(self, hits: int, delay: float) -> None:
        """First touch free (real-route latency), then 0.5, 1, 2, 4, 8 seconds."""
        assert TarpitPolicy().delay_for(hits) == delay

    @pytest.mark.parametrize("hits", [7, 64, 10**6, 10**18])
    def test_capped_without_overflow(self, hits: int) -> None:
        """Any hit count saturates at the cap; none can overflow."""
        assert TarpitPolicy().delay_for(hits) == 8.0

    def test_growth_one_is_constant(self) -> None:
        """No growth: every hit waits the base delay."""
        policy = TarpitPolicy(base_delay_s=0.3, growth=1.0, max_delay_s=5.0)
        assert [policy.delay_for(n) for n in (1, 2, 3, 50)] == [0.0, 0.3, 0.3, 0.3]

    def test_cap_equal_to_base(self) -> None:
        """A cap equal to the base pins every hit at it."""
        policy = TarpitPolicy(base_delay_s=2.0, max_delay_s=2.0)
        assert [policy.delay_for(n) for n in (1, 2, 3, 9)] == [0.0, 2.0, 2.0, 2.0]

    def test_just_below_the_cap(self) -> None:
        """The last uncapped step is exact; the next one caps."""
        policy = TarpitPolicy(base_delay_s=1.0, growth=3.0, max_delay_s=10.0)
        assert policy.delay_for(4) == 9.0
        assert policy.delay_for(5) == 10.0

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
        """An auth-challenge decoy on POST then GET, the order Mercury's routers use."""
        decoy = DecoyRoute("/ok", "lure")
        assert decoy.response == "auth_challenge"
        assert decoy.methods == ("POST", "GET")

    def test_shipped_decoys(self) -> None:
        """Unique paths; two silent tripwires (they observe every method)."""
        paths = [d.path for d in DEFAULT_DECOYS]
        assert len(paths) == len(set(paths))
        assert "/api/v1/gate/override" in paths
        silent = [d for d in DEFAULT_DECOYS if d.response == "not_found"]
        assert {d.path for d in silent} == {"/.env", "/.git/config"}

    @pytest.mark.parametrize("root", ["/", "//", "///"])
    def test_site_root_refused(self, root: str) -> None:
        """Every visitor touches the root; it can never be a decoy."""
        with pytest.raises(ValueError, match="site root"):
            DecoyRoute(root, "lure")


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
            delay_s=0.0,
            credential_presented=False,
            aggregate_key="203.0.113.0/24",
            aggregate_hits=1,
            sources_in_window=1,
            campaign=False,
            audits_suppressed=0,
        )

    def test_escalation_across_decoys(self) -> None:
        """Hits count across all decoys; distinct decoys count breadth."""
        sensor = DeceptionSensor(clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        sensor.record("203.0.113.9", ADMIN, "GET")
        event = sensor.record("203.0.113.9", KEYS, "POST")
        assert event.hits_in_window == 3
        assert event.delay_s == 1.0
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
        assert event.delay_s == 0.0

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
            "tarpit_delay_s": 0.0,
            "credential_presented": False,
            "aggregate": "203.0.113.0/24",
            "aggregate_hits": 1,
            "sources_in_window": 1,
            "campaign": False,
            "audits_suppressed": 0,
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


class TestMutationPins:
    """Kill-pins for the measured survivors: immutability, defaults, boundaries."""

    def test_value_objects_are_immutable(self) -> None:
        """Decoys, policies and events are frozen: safe to share across requests."""
        import dataclasses

        event = DeceptionSensor().record("203.0.113.9", ADMIN, "GET")
        for obj, attribute in (
            (DEFAULT_DECOYS[0], "path"),
            (TarpitPolicy(), "max_delay_s"),
            (event, "delay_s"),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(obj, attribute, 0)

    def test_documented_policy_defaults(self) -> None:
        """The schedule documented in the CHANGELOG and docstrings."""
        policy = TarpitPolicy()
        assert (policy.base_delay_s, policy.growth, policy.max_delay_s) == (0.5, 2.0, 8.0)
        assert policy.window_s == 3600.0
        assert policy.max_concurrent == 32

    def test_documented_sensor_capacities(self) -> None:
        """10,000 sources and 64 hits per source, as documented."""
        import inspect

        params = inspect.signature(DeceptionSensor.__init__).parameters
        assert params["max_sources"].default == 10_000
        assert params["max_hits_tracked"].default == 64
        assert params["outcome_weight"].default == 1.0

    def test_sub_second_window_is_legal(self) -> None:
        """Any positive window is accepted, including one under a second."""
        assert TarpitPolicy(window_s=0.5).window_s == 0.5

    def test_single_hit_history_is_legal(self) -> None:
        """max_hits_tracked=1 is the smallest legal history."""
        sensor = DeceptionSensor(max_hits_tracked=1, clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert sensor.record("203.0.113.9", ADMIN, "GET").hits_in_window == 1

    def test_event_defaults_to_no_credential(self) -> None:
        """An event built without the flag records no credential."""
        event = TripwireEvent("k", "203.0.113.9", ADMIN, "lure", "GET", 1, 1, True, 0.5)
        assert event.credential_presented is False

    def test_audit_delay_rounded_to_milliseconds(self) -> None:
        """The audit record rounds the delay to three decimals."""
        sensor = DeceptionSensor(policy=TarpitPolicy(base_delay_s=0.12345), clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")
        assert sensor.record("203.0.113.9", ADMIN, "GET").audit_details()["tarpit_delay_s"] == 0.123


class TestFreeFirstTouch:
    """Stealth until first contact: a delayed first answer would fingerprint the decoy."""

    def test_first_touch_is_free_by_default(self) -> None:
        """The first hit is answered at real-route latency."""
        assert TarpitPolicy().free_hits == 1
        assert TarpitPolicy().delay_for(1) == 0.0

    def test_zero_free_hits_restores_immediate_escalation(self) -> None:
        """free_hits=0 delays from the first hit."""
        assert TarpitPolicy(free_hits=0).delay_for(1) == 0.5

    def test_larger_allowance(self) -> None:
        """A deployment wanting more stealth can widen the allowance."""
        policy = TarpitPolicy(free_hits=3)
        assert [policy.delay_for(n) for n in (1, 2, 3, 4, 5)] == [0.0, 0.0, 0.0, 0.5, 1.0]

    @pytest.mark.parametrize("bad", [-1, 1.5, None])
    def test_invalid_allowance_refused(self, bad: object) -> None:
        """free_hits must be a non-negative integer."""
        with pytest.raises(ValueError, match="free_hits"):
            TarpitPolicy(free_hits=bad)  # type: ignore[arg-type]

    def test_first_touch_still_recorded(self) -> None:
        """A free first touch is still a tripwire: recorded and audited."""
        events: list[TripwireEvent] = []
        event = DeceptionSensor(audit_sink=events.append, clock=FakeClock()).record(
            "203.0.113.9", ADMIN, "GET"
        )
        assert event.delay_s == 0.0
        assert events == [event]


class TestConfiguredDecoys:
    """Private, per-deployment decoys: the public defaults can be read and avoided."""

    def test_unset_is_the_shipped_set(self) -> None:
        """No configuration: exactly the defaults."""
        assert configured_decoys({}) == DEFAULT_DECOYS

    def test_private_paths_are_added_after_the_defaults(self) -> None:
        """Both kinds are appended in order, with honest lures."""
        decoys = configured_decoys(
            {
                DECOY_PATHS_ENV: "/api/v1/ops/console, /internal/keys",
                SILENT_DECOY_PATHS_ENV: "/backup.sql",
            }
        )
        assert decoys[: len(DEFAULT_DECOYS)] == DEFAULT_DECOYS
        extra = decoys[len(DEFAULT_DECOYS) :]
        assert [(d.path, d.response) for d in extra] == [
            ("/api/v1/ops/console", "auth_challenge"),
            ("/internal/keys", "auth_challenge"),
            ("/backup.sql", "not_found"),
        ]
        assert extra[0].lure == "deployment-configured decoy"
        assert extra[2].lure == "deployment-configured silent tripwire"

    @pytest.mark.parametrize("bad", ["relative/path", "/has space", "/{param}", "/"])
    def test_invalid_entries_dropped_not_fatal(
        self, bad: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed entry is logged and skipped; startup is never blocked."""
        with caplog.at_level(logging.WARNING, logger=deception.__name__):
            decoys = configured_decoys({DECOY_PATHS_ENV: f"{bad},/ok/private"})
        assert decoys[-1].path == "/ok/private"
        assert len(decoys) == len(DEFAULT_DECOYS) + 1
        assert "ignoring decoy" in caplog.text

    def test_duplicates_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A path already defined -- shipped or configured -- is not added twice."""
        with caplog.at_level(logging.WARNING, logger=deception.__name__):
            decoys = configured_decoys(
                {DECOY_PATHS_ENV: "/api/v1/admin,/p,/p", SILENT_DECOY_PATHS_ENV: "/p"}
            )
        assert [d.path for d in decoys].count("/p") == 1
        assert [d.path for d in decoys].count("/api/v1/admin") == 1
        assert "duplicate" in caplog.text

    def test_blank_entries_ignored(self) -> None:
        """Empty segments are ignored silently."""
        assert configured_decoys({DECOY_PATHS_ENV: " , ,"}) == DEFAULT_DECOYS

    def test_reads_process_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default mapping is the process environment."""
        monkeypatch.setenv(SILENT_DECOY_PATHS_ENV, "/private.bak")
        assert configured_decoys()[-1].path == "/private.bak"


class TestAggregateKey:
    """The allocation-level identity a source cannot rotate out of cheaply."""

    @pytest.mark.parametrize(
        ("client_ip", "expected"),
        [
            ("203.0.113.9", "203.0.113.0/24"),
            ("203.0.113.255", "203.0.113.0/24"),
            ("::ffff:203.0.113.9", "203.0.113.0/24"),  # v4-mapped matches its v4 twin
            ("2001:db8:1:2::5", "2001:db8:1::/48"),
            ("2001:db8:1:ffff::5", "2001:db8:1::/48"),  # another /64, same /48
            ("unknown", "unknown"),
            ("not-an-ip", "unknown"),
        ],
    )
    def test_keys(self, client_ip: str, expected: str) -> None:
        """Each address lands in its allocation; non-IPs share one identity."""
        assert aggregate_key(client_ip) == expected

    def test_source_and_aggregate_stay_consistent_for_v4_mapped(self) -> None:
        """A v4-mapped v6 address and its v4 twin share source and aggregate."""
        assert source_key("::ffff:203.0.113.9") == source_key("203.0.113.9")
        assert aggregate_key("::ffff:203.0.113.9") == aggregate_key("203.0.113.9")


class TestRotationResistance:
    """Rotating fresh addresses inside one allocation stops buying free hits."""

    def test_rotation_inside_a_v4_slash24_escalates(self) -> None:
        """Fresh /24 members get the aggregate schedule once its allowance is spent."""
        sensor = DeceptionSensor(clock=FakeClock())
        delays = [sensor.record(f"203.0.113.{i}", ADMIN, "GET").delay_s for i in range(1, 13)]
        assert delays[:8] == [0.0] * 8  # the aggregate allowance, spent once, together
        assert delays[8:] == [0.5, 1.0, 2.0, 4.0]

    def test_rotation_across_v6_slash64s_in_one_slash48_escalates(self) -> None:
        """Rotating /64s inside one /48 shares one aggregate allowance."""
        sensor = DeceptionSensor(clock=FakeClock())
        events = [sensor.record(f"2001:db8:1:{i:x}::1", ADMIN, "GET") for i in range(1, 13)]
        assert all(e.aggregate_key == "2001:db8:1::/48" for e in events)
        assert [e.delay_s for e in events[8:]] == [0.5, 1.0, 2.0, 4.0]

    def test_single_source_schedule_is_unchanged(self) -> None:
        """A lone prober sees exactly the pre-aggregate schedule."""
        sensor = DeceptionSensor(clock=FakeClock())
        delays = [sensor.record("198.51.100.7", ADMIN, "GET").delay_s for _ in range(6)]
        assert delays == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

    def test_delay_is_the_larger_schedule(self) -> None:
        """A hot source in a hot aggregate keeps its own, larger delay."""
        sensor = DeceptionSensor(clock=FakeClock())
        for _ in range(6):
            event = sensor.record("198.51.100.7", ADMIN, "GET")
        assert event.delay_s == 8.0
        follower = sensor.record("198.51.100.8", ADMIN, "GET")
        # The follower's own count is 1 (free); only the aggregate schedule
        # applies to it, and it is never the maximum of the hot source.
        assert follower.delay_s == pytest.approx(0.0)
        assert follower.aggregate_hits == 7

    def test_aggregate_window_expires(self) -> None:
        """Aggregate hits age out of the window like source hits."""
        clock = FakeClock()
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=60.0), clock=clock)
        for i in range(1, 10):
            sensor.record(f"203.0.113.{i}", ADMIN, "GET")
        clock.now += 61.0
        event = sensor.record("203.0.113.99", ADMIN, "GET")
        assert event.aggregate_hits == 1
        assert event.delay_s == 0.0

    def test_aggregate_free_hits_validated(self) -> None:
        """A negative aggregate allowance is refused."""
        with pytest.raises(ValueError, match="aggregate_free_hits"):
            TarpitPolicy(aggregate_free_hits=-1)


class TestCampaignSignal:
    """Distinct sources in the window: the signal rotation raises, not hides."""

    def test_flag_flips_at_the_threshold(self) -> None:
        """campaign is False below the threshold and True at it."""
        sensor = DeceptionSensor(clock=FakeClock(), campaign_sources=3)
        first = sensor.record("203.0.113.1", ADMIN, "GET")
        second = sensor.record("203.0.113.2", ADMIN, "GET")
        third = sensor.record("203.0.113.3", ADMIN, "GET")
        assert (first.campaign, second.campaign, third.campaign) == (False, False, True)
        assert third.sources_in_window == 3

    def test_sources_age_out_of_the_window(self) -> None:
        """Quiet sources stop counting toward the campaign signal."""
        clock = FakeClock()
        sensor = DeceptionSensor(
            policy=TarpitPolicy(window_s=60.0), clock=clock, campaign_sources=3
        )
        sensor.record("203.0.113.1", ADMIN, "GET")
        sensor.record("203.0.113.2", ADMIN, "GET")
        clock.now += 61.0
        event = sensor.record("203.0.113.3", ADMIN, "GET")
        assert event.sources_in_window == 1
        assert event.campaign is False

    def test_threshold_validated(self) -> None:
        """A zero threshold is refused."""
        with pytest.raises(ValueError, match="campaign_sources"):
            DeceptionSensor(campaign_sources=0)


class TestAuditCap:
    """The tamper-evident chain cannot be flooded through the decoys."""

    def test_audits_past_the_cap_are_counted_not_written(self) -> None:
        """Only the cap's worth of audits reach the sink inside one window."""
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(
            clock=FakeClock(), max_audits_per_window=2, audit_sink=events.append
        )
        for i in range(1, 6):
            sensor.record(f"192.0.2.{i}", ADMIN, "GET")
        assert len(events) == 2

    def test_first_audit_of_the_next_window_reports_the_suppressed(self) -> None:
        """The chain accounts for every suppressed record, as one number."""
        clock = FakeClock()
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(
            policy=TarpitPolicy(window_s=60.0),
            clock=clock,
            max_audits_per_window=2,
            audit_sink=events.append,
        )
        for i in range(1, 6):
            sensor.record(f"192.0.2.{i}", ADMIN, "GET")
        clock.now += 61.0
        sensor.record("192.0.2.200", ADMIN, "GET")
        assert len(events) == 3
        assert events[-1].audits_suppressed == 3
        assert events[-1].audit_details()["audits_suppressed"] == 3
        # And the counter does not double-report.
        sensor.record("192.0.2.201", ADMIN, "GET")
        assert events[-1].audits_suppressed == 0

    def test_cap_validated(self) -> None:
        """A zero cap is refused: the chain must always get the first record."""
        with pytest.raises(ValueError, match="max_audits_per_window"):
            DeceptionSensor(max_audits_per_window=0)


class TestSqliteTripwireStore:
    """One picture across workers, on the platform's shared SQLite file."""

    @staticmethod
    def _store(tmp_path: Path, now: float = 5000.0) -> SqliteTripwireStore:
        clock = FakeClock(now)
        store = SqliteTripwireStore(tmp_path / "state.db", clock=clock)
        store.test_clock = clock  # type: ignore[attr-defined]
        return store

    def test_hits_accumulate_across_sensors(self, tmp_path: Path) -> None:
        """Two workers' touches land in one count, so escalation sees both."""
        store = self._store(tmp_path)
        w1 = DeceptionSensor(clock=FakeClock(), store=store)
        w2 = DeceptionSensor(clock=FakeClock(), store=store)
        assert w1.record("198.51.100.7", ADMIN, "GET").hits_in_window == 1
        event = w2.record("198.51.100.7", ADMIN, "GET")
        assert event.hits_in_window == 2
        assert event.delay_s == 0.5

    def test_audit_dedupe_spans_workers(self, tmp_path: Path) -> None:
        """The same (source, decoy) is audited once per window, cluster-wide."""
        store = self._store(tmp_path)
        audits1: list[TripwireEvent] = []
        audits2: list[TripwireEvent] = []
        w1 = DeceptionSensor(clock=FakeClock(), store=store, audit_sink=audits1.append)
        w2 = DeceptionSensor(clock=FakeClock(), store=store, audit_sink=audits2.append)
        w1.record("198.51.100.7", ADMIN, "GET")
        w2.record("198.51.100.7", ADMIN, "GET")
        assert len(audits1) == 1
        assert len(audits2) == 0

    def test_campaign_counts_sources_across_workers(self, tmp_path: Path) -> None:
        """Distinct sources are counted cluster-wide."""
        store = self._store(tmp_path)
        w1 = DeceptionSensor(clock=FakeClock(), store=store)
        w2 = DeceptionSensor(clock=FakeClock(), store=store)
        for i in range(1, 5):
            w1.record(f"198.51.100.{i}", ADMIN, "GET")
        for i in range(5, 9):
            w2.record(f"198.51.100.{i}", ADMIN, "GET")
        event = w1.record("198.51.100.250", ADMIN, "GET")
        assert event.sources_in_window == 9
        assert event.campaign is True

    def test_window_expiry_in_the_store(self, tmp_path: Path) -> None:
        """Stored hits age out of the window."""
        store = self._store(tmp_path)
        clock = store.test_clock  # type: ignore[attr-defined]
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=60.0), clock=FakeClock(), store=store)
        for _ in range(3):
            sensor.record("198.51.100.7", ADMIN, "GET")
        clock.now += 61.0
        event = sensor.record("198.51.100.7", ADMIN, "GET")
        assert event.hits_in_window == 1
        assert event.aggregate_hits == 1

    def test_store_failure_falls_back_to_process_state(self, tmp_path: Path) -> None:
        """A dead store never blocks the answer: local counts serve the request."""
        store = self._store(tmp_path)
        sensor = DeceptionSensor(clock=FakeClock(), store=store)
        sensor.record("198.51.100.7", ADMIN, "GET")
        store.close()
        event = sensor.record("198.51.100.7", ADMIN, "GET")
        assert event.hits_in_window == 2  # the in-process view kept counting

    def test_per_key_hit_history_is_capped(self, tmp_path: Path) -> None:
        """The stored history per key never exceeds max_hits."""
        store = self._store(tmp_path)
        for _ in range(10):
            rec = store.record_touch(
                "198.51.100.7", "198.51.100.0/24", ADMIN, window_s=3600.0, max_hits=3
            )
        assert rec.hits == 3
        assert rec.aggregate_hits == 3

    def test_record_touch_validates_inputs(self, tmp_path: Path) -> None:
        """Out-of-range window or cap is refused, not stored."""
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="window_s"):
            store.record_touch("k", "a", ADMIN, window_s=0.0, max_hits=3)
        with pytest.raises(ValueError, match="max_hits"):
            store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=0)


class TestStoreIsBounded:
    """The store keeps the sensor's LRU bound; rotation cannot grow it forever.

    Each test failed against the replaced store, which had no cap on the
    sources or aggregates on record and never pruned decoy touches.
    """

    @staticmethod
    def _rows(store: SqliteTripwireStore, table: str) -> int:
        return int(store._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def test_sources_and_touches_capped_at_max_sources(self, tmp_path: Path) -> None:
        """Rotating through 40 sources over a cap of 8 leaves 8 on record."""
        clock = FakeClock(5000.0)
        store = SqliteTripwireStore(tmp_path / "state.db", clock=clock)
        for i in range(40):
            clock.now += 1.0
            store.record_touch(
                f"198.51.{i}.7",
                f"198.51.{i}.0/24",
                ADMIN,
                window_s=3600.0,
                max_hits=4,
                max_sources=8,
            )
        assert self._rows(store, "deception_touches") == 8
        keys = store._conn.execute(
            "SELECT scope, COUNT(*) FROM deception_keys GROUP BY scope ORDER BY scope"
        ).fetchall()
        assert [tuple(row) for row in keys] == [("agg", 8), ("src", 8)]
        assert self._rows(store, "deception_hits") == 16  # one src + one agg hit each

    def test_least_recently_seen_source_is_evicted(self, tmp_path: Path) -> None:
        """A source seen again stays; the stalest one goes, with its history."""
        clock = FakeClock(5000.0)
        store = SqliteTripwireStore(tmp_path / "state.db", clock=clock)

        def touch(ip: str) -> TouchRecord:
            clock.now += 1.0
            return store.record_touch(
                ip, "198.51.100.0/24", ADMIN, window_s=3600.0, max_hits=64, max_sources=2
            )

        touch("198.51.100.1")
        touch("198.51.100.2")
        touch("198.51.100.1")  # .1 is now the most recently seen
        touch("198.51.100.3")  # evicts .2
        assert store.distinct_decoys("198.51.100.1") == 1
        assert store.distinct_decoys("198.51.100.2") == 0
        rec = touch("198.51.100.2")
        assert rec.first_touch is True  # its record went with the eviction
        assert rec.hits == 1

    def test_sources_in_window_unchanged_by_the_cap(self, tmp_path: Path) -> None:
        """Below the cap, the campaign count is the distinct sources in the window."""
        clock = FakeClock(5000.0)
        store = SqliteTripwireStore(tmp_path / "state.db", clock=clock)
        for i in range(1, 6):
            clock.now += 1.0
            rec = store.record_touch(
                f"198.51.100.{i}", "198.51.100.0/24", ADMIN, window_s=60.0, max_hits=8
            )
        assert rec.sources_in_window == 5
        clock.now += 58.5  # sources 1-3 now older than the window
        rec = store.record_touch(
            "198.51.100.9", "198.51.100.0/24", ADMIN, window_s=60.0, max_hits=8
        )
        assert rec.sources_in_window == 3  # .4, .5 and .9

    def test_sensor_passes_its_capacity_to_the_store(self, tmp_path: Path) -> None:
        """The sensor's max_sources is the store's cap."""
        store = SqliteTripwireStore(tmp_path / "state.db", clock=FakeClock(5000.0))
        sensor = DeceptionSensor(clock=FakeClock(), store=store, max_sources=3)
        for i in range(10):
            sensor.record(f"198.51.{i}.7", ADMIN, "GET")
        assert self._rows(store, "deception_touches") == 3

    def test_max_sources_validated(self, tmp_path: Path) -> None:
        """A non-positive cap is refused."""
        store = SqliteTripwireStore(tmp_path / "state.db", clock=FakeClock(5000.0))
        with pytest.raises(ValueError, match="max_sources"):
            store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=3, max_sources=0)

    def test_store_written_before_the_cap_is_backfilled(self, tmp_path: Path) -> None:
        """Keys already in an older store are put on record, so the cap covers them."""
        import sqlite3

        path = tmp_path / "state.db"
        conn = sqlite3.connect(path)
        conn.executescript(
            "CREATE TABLE deception_hits (scope TEXT NOT NULL, scope_key TEXT NOT NULL,"
            " hit_s REAL NOT NULL);"
            "CREATE TABLE deception_touches (source_key TEXT NOT NULL, decoy_path TEXT NOT NULL,"
            " first_s REAL NOT NULL, audited_s REAL NOT NULL,"
            " PRIMARY KEY (source_key, decoy_path));"
            "INSERT INTO deception_hits VALUES ('src', 'old', 4000.0), ('agg', 'old/24', 4000.0);"
            "INSERT INTO deception_touches VALUES ('old', '/api/v1/admin', 4000.0, 4000.0);"
        )
        conn.commit()
        conn.close()
        clock = FakeClock(5000.0)
        store = SqliteTripwireStore(path, clock=clock)
        store.record_touch("new", "new/24", ADMIN, window_s=3600.0, max_hits=8, max_sources=1)
        assert store.distinct_decoys("old") == 0  # evicted as least recently seen
        assert store.distinct_decoys("new") == 1


class TestClusterWideGrading:
    """Every worker grades a source from the same, cluster-wide record."""

    def test_outcomes_count_touches_seen_by_other_workers(self, tmp_path: Path) -> None:
        """Worker 2 grades the decoys a source touched through worker 1."""
        store = SqliteTripwireStore(tmp_path / "state.db", clock=FakeClock(5000.0))
        w1 = DeceptionSensor(clock=FakeClock(), store=store)
        w2 = DeceptionSensor(clock=FakeClock(), store=store)
        w1.record("198.51.100.7", ADMIN, "GET")
        w1.record("198.51.100.7", KEYS, "GET")
        w2.record("198.51.100.7", ADMIN, "GET")
        assert len(w2.outcomes("198.51.100.7")) == 2
        assert w2.posterior("198.51.100.7") == w1.posterior("198.51.100.7")

    def test_outcomes_fall_back_to_process_state(self, tmp_path: Path) -> None:
        """A dead store grades from this process's view, never raises."""
        store = SqliteTripwireStore(tmp_path / "state.db", clock=FakeClock(5000.0))
        sensor = DeceptionSensor(clock=FakeClock(), store=store)
        sensor.record("198.51.100.7", ADMIN, "GET")
        store.close()
        assert len(sensor.outcomes("198.51.100.7")) == 1


class TestBuildTripwireStore:
    """The builder follows the platform's shared-state convention."""

    def test_unset_means_in_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No MERCURY_KEYSTORE_PATH: per-process state, exactly as before."""
        monkeypatch.delenv(TRIPWIRE_STORE_PATH_ENV, raising=False)
        assert build_tripwire_store() is None

    def test_blank_means_in_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A blank value is unset, not a store at path ''."""
        monkeypatch.setenv(TRIPWIRE_STORE_PATH_ENV, "   ")
        assert build_tripwire_store() is None

    def test_set_builds_the_store(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A path builds the shared store on that file."""
        monkeypatch.setenv(TRIPWIRE_STORE_PATH_ENV, str(tmp_path / "state.db"))
        store = build_tripwire_store()
        assert store is not None
        store.close()


class TestPolicyAndSensorBoundaries:
    """Each validator admits its boundary value; each default is pinned."""

    def test_zero_aggregate_free_hits_is_valid(self) -> None:
        """0 is a legal allowance: every aggregate hit is delayed."""
        policy = TarpitPolicy(aggregate_free_hits=0)
        assert policy.aggregate_delay_for(1) == 0.5

    def test_campaign_threshold_of_one_is_valid(self) -> None:
        """campaign_sources=1 flags the very first source."""
        sensor = DeceptionSensor(clock=FakeClock(), campaign_sources=1)
        assert sensor.record("203.0.113.9", ADMIN, "GET").campaign is True

    def test_audit_cap_of_one_is_valid(self) -> None:
        """max_audits_per_window=1 admits exactly one audit."""
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(
            clock=FakeClock(), max_audits_per_window=1, audit_sink=events.append
        )
        sensor.record("203.0.113.1", ADMIN, "GET")
        sensor.record("203.0.113.2", ADMIN, "GET")
        assert len(events) == 1

    def test_default_campaign_threshold_is_eight(self) -> None:
        """The 7th distinct source is below the default threshold; the 8th is at it."""
        sensor = DeceptionSensor(clock=FakeClock())
        events = [sensor.record(f"203.0.113.{i}", ADMIN, "GET") for i in range(1, 9)]
        assert events[6].campaign is False
        assert events[7].campaign is True

    def test_default_audit_cap_is_512(self) -> None:
        """The 512th first-touch audit is written; the 513th is suppressed."""
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(clock=FakeClock(), audit_sink=events.append)
        for i in range(513):
            sensor.record(f"10.{i // 250}.{i % 250}.1", ADMIN, "GET")
        assert len(events) == 512

    def test_minimal_event_defaults(self) -> None:
        """A directly constructed event carries the documented neutral defaults."""
        event = TripwireEvent(
            source_key="k",
            client_ip="203.0.113.9",
            path=ADMIN,
            lure="l",
            method="GET",
            hits_in_window=1,
            distinct_decoys=1,
            first_touch=True,
            delay_s=0.0,
        )
        assert event.credential_presented is False
        assert event.aggregate_key == "unknown"
        assert event.aggregate_hits == 0
        assert event.sources_in_window == 0
        assert event.campaign is False
        assert event.audits_suppressed == 0


class TestAggregateBookkeeping:
    """The aggregate structures share the source structures' exact bounds."""

    def test_aggregates_are_lru_capped(self) -> None:
        """Past max_sources, the least recently seen aggregate is evicted."""
        sensor = DeceptionSensor(max_sources=2, clock=FakeClock())
        sensor.record("203.0.113.9", ADMIN, "GET")  # /24 A
        sensor.record("198.51.100.9", ADMIN, "GET")  # /24 B
        # A third aggregate evicts A, the least recently seen -- B survives.
        sensor.record("192.0.2.9", ADMIN, "GET")  # /24 C
        assert sensor.record("198.51.100.10", ADMIN, "GET").aggregate_hits == 2
        # A was evicted: its count restarted.
        assert sensor.record("203.0.113.10", ADMIN, "GET").aggregate_hits == 1

    def test_aggregate_hit_at_exactly_the_window_edge_still_counts(self) -> None:
        """Ages are pruned strictly past the window, like source hits."""
        clock = FakeClock()
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=60.0), clock=clock)
        sensor.record("203.0.113.9", ADMIN, "GET")
        clock.now += 60.0
        assert sensor.record("203.0.113.10", ADMIN, "GET").aggregate_hits == 2

    def test_active_source_at_exactly_the_window_edge_still_counts(self) -> None:
        """The campaign count prunes strictly past the window."""
        clock = FakeClock()
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=60.0), clock=clock)
        sensor.record("203.0.113.9", ADMIN, "GET")
        clock.now += 60.0
        assert sensor.record("198.51.100.9", ADMIN, "GET").sources_in_window == 2

    def test_active_sources_are_capped_at_max_sources(self) -> None:
        """The campaign count never exceeds max_sources, and only by one eviction."""
        sensor = DeceptionSensor(max_sources=2, clock=FakeClock())
        for i in range(1, 4):
            event = sensor.record(f"203.0.113.{i}", ADMIN, "GET")
        assert event.sources_in_window == 2

    def test_active_source_eviction_drops_the_oldest(self) -> None:
        """Evicting the newest instead would overcount expiries later."""
        clock = FakeClock()
        sensor = DeceptionSensor(policy=TarpitPolicy(window_s=10.0), max_sources=2, clock=clock)
        sensor.record("203.0.113.1", ADMIN, "GET")  # A @ 1000.0
        clock.now += 1.0
        sensor.record("203.0.113.2", ADMIN, "GET")  # B @ 1001.0
        clock.now += 1.0
        sensor.record("203.0.113.3", ADMIN, "GET")  # C @ 1002.0 -> evict A
        clock.now += 9.5  # 1011.5: B (10.5) expired, C (9.5) alive
        event = sensor.record("203.0.113.4", ADMIN, "GET")
        # Correct LRU kept {B, C}; B expires -> C + the new source = 2.
        # Evict-newest would have kept {A, B}; both expire -> 1.
        assert event.sources_in_window == 2


class TestAuditCapBoundaries:
    """The cap's window, logging and suppressed accounting, at their edges."""

    def test_window_rolls_strictly_past_the_window(self) -> None:
        """An audit exactly window seconds after the first is still capped."""
        clock = FakeClock()
        events: list[TripwireEvent] = []
        sensor = DeceptionSensor(
            policy=TarpitPolicy(window_s=60.0),
            clock=clock,
            max_audits_per_window=1,
            audit_sink=events.append,
        )
        sensor.record("203.0.113.1", ADMIN, "GET")
        clock.now += 60.0
        sensor.record("203.0.113.2", ADMIN, "GET")
        assert len(events) == 1  # the boundary audit is still inside the window

    def test_cap_warning_logs_exactly_once(self, caplog: pytest.LogCaptureFixture) -> None:
        """One warning when the cap is first reached, not one per suppression."""
        sensor = DeceptionSensor(clock=FakeClock(), max_audits_per_window=1)
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.security.deception"):
            sensor.record("203.0.113.1", ADMIN, "GET")
            sensor.record("203.0.113.2", ADMIN, "GET")
        assert sum("audit cap" in r.message for r in caplog.records) == 1

    def test_suppressed_event_reports_zero_suppressed(self) -> None:
        """The suppressed event itself never claims a carried count."""
        sensor = DeceptionSensor(clock=FakeClock(), max_audits_per_window=1)
        sensor.record("203.0.113.1", ADMIN, "GET")
        event = sensor.record("203.0.113.2", ADMIN, "GET")
        assert event.audits_suppressed == 0


class TestCampaignLogging:
    """The campaign warning fires once per window, boundary included."""

    def test_logs_once_inside_a_window(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two campaign events in one window: one warning, checked at each step."""

        def warnings() -> int:
            return sum("campaign signal" in r.message for r in caplog.records)

        clock = FakeClock()
        sensor = DeceptionSensor(
            policy=TarpitPolicy(window_s=60.0), clock=clock, campaign_sources=1
        )
        with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.security.deception"):
            sensor.record("203.0.113.1", ADMIN, "GET")
            assert warnings() == 1
            clock.now += 60.0  # exactly the window edge: still the same window
            sensor.record("203.0.113.2", ADMIN, "GET")
            assert warnings() == 1  # checked HERE: totals alone cannot see this
            clock.now += 0.5  # now strictly past it: a new window
            sensor.record("203.0.113.3", ADMIN, "GET")
            assert warnings() == 2


class TestStoreBoundaries:
    """The store's validators, schema invariants and audit boundary, at their edges."""

    def test_touch_record_is_frozen(self, tmp_path: Path) -> None:
        """Counts handed to the sensor are evidence; evidence is immutable."""
        import dataclasses

        store = TestSqliteTripwireStore._store(tmp_path)
        rec = store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=3)
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.hits = 99  # type: ignore[misc]

    def test_sub_second_window_is_valid(self, tmp_path: Path) -> None:
        """Any positive window is legal, not just >= 1 second."""
        store = TestSqliteTripwireStore._store(tmp_path)
        assert store.record_touch("k", "a", ADMIN, window_s=0.5, max_hits=3).hits == 1

    def test_max_hits_of_one_is_valid(self, tmp_path: Path) -> None:
        """A history of one is legal: every count reads 1."""
        store = TestSqliteTripwireStore._store(tmp_path)
        store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=1)
        assert store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=1).hits == 1

    def test_audit_due_boundary_matches_the_sensor(self, tmp_path: Path) -> None:
        """A re-touch exactly window seconds after the audit is not yet due."""
        store = TestSqliteTripwireStore._store(tmp_path)
        clock = store.test_clock  # type: ignore[attr-defined]
        store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=9)
        clock.now += 60.0
        assert store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=9).audit_due is False
        clock.now += 0.5
        assert store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=9).audit_due is True

    def test_nested_state_directory_is_created(self, tmp_path: Path) -> None:
        """A state path under directories that do not exist yet still opens."""
        store = SqliteTripwireStore(tmp_path / "a" / "b" / "state.db", clock=FakeClock())
        assert store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=3).hits == 1
        store.close()

    def test_usable_from_a_second_thread(self, tmp_path: Path) -> None:
        """Workers serve requests on worker threads; the store must follow."""
        import threading

        store = TestSqliteTripwireStore._store(tmp_path)
        results: list[int] = []
        thread = threading.Thread(
            target=lambda: results.append(
                store.record_touch("k", "a", ADMIN, window_s=60.0, max_hits=3).hits
            )
        )
        thread.start()
        thread.join()
        assert results == [1]


class TestStorePruneCadence:
    """The opportunistic full prune runs exactly every 256 touches."""

    @staticmethod
    def _stale_rows(db: Path, key: str) -> int:
        import sqlite3

        with sqlite3.connect(db) as conn:
            return int(
                conn.execute(
                    "SELECT COUNT(*) FROM deception_hits WHERE scope_key = ?", (key,)
                ).fetchone()[0]
            )

    def test_stale_rows_of_other_keys_clear_on_the_256th_touch(self, tmp_path: Path) -> None:
        """255 touches leave a stale stranger; the 256th sweeps it; the cycle repeats."""
        db = tmp_path / "state.db"
        clock = FakeClock()
        store = SqliteTripwireStore(db, clock=clock)
        store.record_touch("stale", "stale-agg", ADMIN, window_s=60.0, max_hits=500)
        clock.now += 61.0  # the stranger's rows age out of the window
        for _ in range(254):  # touches 2..255 since the store opened
            store.record_touch("active", "active-agg", ADMIN, window_s=60.0, max_hits=500)
        assert self._stale_rows(db, "stale") == 1
        store.record_touch("active", "active-agg", ADMIN, window_s=60.0, max_hits=500)
        assert self._stale_rows(db, "stale") == 0
        # Second cycle: the counter reset to zero, not one.
        store.record_touch("stale2", "stale-agg2", ADMIN, window_s=60.0, max_hits=500)
        clock.now += 61.0
        for _ in range(254):
            store.record_touch("active", "active-agg", ADMIN, window_s=60.0, max_hits=500)
        assert self._stale_rows(db, "stale2") == 1
        store.record_touch("active", "active-agg", ADMIN, window_s=60.0, max_hits=500)
        assert self._stale_rows(db, "stale2") == 0
        store.close()

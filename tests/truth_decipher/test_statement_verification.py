# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for statement verification in :mod:`omni_mercury_engine.truth_decipher`.

Every expected number is either computed by hand (the arithmetic is in the
comment or the assertion) or taken from an independent oracle: ``scipy.stats``
for Wilson intervals and percentiles, :func:`math.isclose` for tolerances, the
spherical law of cosines for distances, and brute-force enumeration for Allen
relations.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import re
from typing import Any

import pytest
from scipy import stats

from omni_mercury_engine.truth_decipher import (
    DEFAULT_MAX_SPEED_KMH,
    EARTH_MEAN_RADIUS_KM,
    SCIENTIFIC_FRAMING,
    VERIFIABILITY_NO_NORM_NOTE,
    AllenRelation,
    Claim,
    ClaimKey,
    ContradictionKind,
    ContradictionScope,
    CorroborationStatus,
    DeclaredIndependenceClusters,
    Detail,
    EvidenceItem,
    HaversineTravelCheck,
    IndependenceClusterModel,
    LocatedObservation,
    Location,
    Reference,
    SourceReliabilityModel,
    Stance,
    Statement,
    StatementVerifier,
    TableSourceReliability,
    TemporalOrdering,
    TimeInterval,
    TravelAssessment,
    TravelFeasibilityCheck,
    TruthDecipherFramework,
    ValueTolerance,
    VerificationReport,
    VerifierConfig,
    allen_relation,
    assess_verifiability,
    converse,
    converse_relations,
    haversine_km,
    intervals_concurrent,
    log_likelihood_ratio,
    values_compatible,
    wilson_interval,
)

try:
    from hypothesis import (
        given,
        settings,
        strategies as st,
    )
except ImportError:  # pragma: no cover - hypothesis is a test extra
    HAVE_HYPOTHESIS = False
else:
    HAVE_HYPOTHESIS = True

R = EARTH_MEAN_RADIUS_KM
LN = math.log
FORBIDDEN = re.compile(r"\b(lying|liar|liars|lie|lies|deceptive|deception|truthful)\b", re.I)

A_LOC = ClaimKey("A", "location")
A_CAR = ClaimKey("A", "car")


def _verify(
    statements: list[Statement],
    evidence: list[EvidenceItem] | tuple[EvidenceItem, ...] = (),
    **kwargs: Any,
) -> VerificationReport:
    return StatementVerifier(**kwargs).verify(statements, evidence)


def _one_claim(claim: Claim, *, sid: str = "s1", speaker: str = "A") -> Statement:
    return Statement(sid, speaker, 0.0, claims=(claim,))


def _support(eid: str, key: ClaimKey, r: float, **kw: Any) -> EvidenceItem:
    return EvidenceItem(eid, target=key, stance=Stance.SUPPORTS, reliability=r, **kw)


def _refute(eid: str, key: ClaimKey, r: float, **kw: Any) -> EvidenceItem:
    return EvidenceItem(eid, target=key, stance=Stance.REFUTES, reliability=r, **kw)


# --------------------------------------------------------------------------- #
# Wilson intervals.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("confidence", [0.95, 0.9, 0.99])
@pytest.mark.parametrize(("k", "n"), [(0, 1), (1, 1), (3, 10), (7, 7), (12, 40), (1, 3)])
def test_wilson_matches_scipy(k: int, n: int, confidence: float) -> None:
    got = wilson_interval(k, n, confidence)
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=confidence, method="wilson")
    assert got.estimate == pytest.approx(k / n, abs=0, rel=1e-15)
    assert got.lower == pytest.approx(float(ci.low), rel=1e-12, abs=1e-15)
    assert got.upper == pytest.approx(float(ci.high), rel=1e-12, abs=1e-15)
    assert (got.successes, got.trials, got.confidence) == (k, n, confidence)


def test_wilson_closed_form_at_zero_successes() -> None:
    # With k = 0 the Wilson bounds are exactly 0 and z^2 / (n + z^2).
    z = float(stats.norm.ppf(0.975))
    got = wilson_interval(0, 10)
    assert got.lower == 0.0
    assert got.upper == pytest.approx(z * z / (10 + z * z), rel=1e-13)


def test_wilson_abstains_without_trials() -> None:
    got = wilson_interval(0, 0)
    assert (got.estimate, got.lower, got.upper) == (None, None, None)


@pytest.mark.parametrize(("k", "n", "conf"), [(-1, 3, 0.95), (4, 3, 0.95), (1, 3, 1.0)])
def test_wilson_rejects_invalid_input(k: int, n: int, conf: float) -> None:
    with pytest.raises(ValueError):
        wilson_interval(k, n, conf)


# --------------------------------------------------------------------------- #
# Allen's interval algebra.
# --------------------------------------------------------------------------- #
_ALLEN_CASES = [
    ((0, 1), (2, 3), AllenRelation.BEFORE),
    ((0, 1), (1, 2), AllenRelation.MEETS),
    ((0, 2), (1, 3), AllenRelation.OVERLAPS),
    ((0, 1), (0, 2), AllenRelation.STARTS),
    ((1, 2), (0, 3), AllenRelation.DURING),
    ((1, 2), (0, 2), AllenRelation.FINISHES),
    ((0, 1), (0, 1), AllenRelation.EQUALS),
    ((0, 2), (1, 2), AllenRelation.FINISHED_BY),
    ((0, 3), (1, 2), AllenRelation.CONTAINS),
    ((0, 2), (0, 1), AllenRelation.STARTED_BY),
    ((1, 3), (0, 2), AllenRelation.OVERLAPPED_BY),
    ((1, 2), (0, 1), AllenRelation.MET_BY),
    ((2, 3), (0, 1), AllenRelation.AFTER),
    # Degenerate intervals (time points): Vilain's point-interval relations.
    ((0, 0), (0, 2), AllenRelation.STARTS),
    ((2, 2), (0, 2), AllenRelation.FINISHES),
    ((1, 1), (0, 2), AllenRelation.DURING),
    ((0, 2), (0, 0), AllenRelation.STARTED_BY),
    ((0, 2), (2, 2), AllenRelation.FINISHED_BY),
    ((0, 2), (1, 1), AllenRelation.CONTAINS),
    ((1, 1), (1, 1), AllenRelation.EQUALS),
    ((1, 1), (2, 2), AllenRelation.BEFORE),
    ((3, 3), (0, 2), AllenRelation.AFTER),
]


@pytest.mark.parametrize(("a", "b", "expected"), _ALLEN_CASES)
def test_allen_relation_table(
    a: tuple[float, float], b: tuple[float, float], expected: AllenRelation
) -> None:
    ia, ib = TimeInterval(*a), TimeInterval(*b)
    assert allen_relation(ia, ib) is expected
    assert allen_relation(ib, ia) is converse(expected)


def test_allen_relations_are_exhaustive_and_disjoint_on_a_grid() -> None:
    # Brute force over every proper interval with endpoints in {0..4}: each pair
    # gets the relation defined by Allen's endpoint conditions, and every one of
    # the 13 relations occurs.
    def oracle(a: tuple[int, int], b: tuple[int, int]) -> AllenRelation:
        (s1, e1), (s2, e2) = a, b
        table = {
            AllenRelation.BEFORE: e1 < s2,
            AllenRelation.MEETS: e1 == s2,
            AllenRelation.OVERLAPS: s1 < s2 < e1 < e2,
            AllenRelation.STARTS: s1 == s2 and e1 < e2,
            AllenRelation.DURING: s2 < s1 and e1 < e2,
            AllenRelation.FINISHES: e1 == e2 and s2 < s1,
            AllenRelation.EQUALS: s1 == s2 and e1 == e2,
        }
        hits = [r for r, ok in table.items() if ok]
        inverse = {
            converse(r)
            for r, ok in {
                AllenRelation.BEFORE: e2 < s1,
                AllenRelation.MEETS: e2 == s1,
                AllenRelation.OVERLAPS: s2 < s1 < e2 < e1,
                AllenRelation.STARTS: s1 == s2 and e2 < e1,
                AllenRelation.DURING: s1 < s2 and e2 < e1,
                AllenRelation.FINISHES: e1 == e2 and s1 < s2,
            }.items()
            if ok
        }
        found = set(hits) | inverse
        assert len(found) == 1, (a, b, found)
        return found.pop()

    intervals = [(s, e) for s in range(5) for e in range(5) if s < e]
    seen = set()
    for a in intervals:
        for b in intervals:
            rel = allen_relation(TimeInterval(*a), TimeInterval(*b))
            assert rel is oracle(a, b)
            seen.add(rel)
    assert seen == set(AllenRelation)


def test_converse_is_an_involution_and_set_converse() -> None:
    for rel in AllenRelation:
        assert converse(converse(rel)) is rel
    assert converse_relations({AllenRelation.BEFORE, AllenRelation.MEETS}) == {
        AllenRelation.AFTER,
        AllenRelation.MET_BY,
    }


def test_concurrency_excludes_boundary_contact_and_accepts_unknown() -> None:
    assert not intervals_concurrent(TimeInterval(0, 1), TimeInterval(1, 2))
    assert intervals_concurrent(TimeInterval(0, 1.5), TimeInterval(1, 2))
    assert intervals_concurrent(TimeInterval.at(1), TimeInterval(1, 2))
    assert intervals_concurrent(None, TimeInterval(1, 2))
    assert intervals_concurrent(TimeInterval(1, 2), None)


if HAVE_HYPOTHESIS:
    _endpoints = st.integers(min_value=-5, max_value=5)

    @settings(max_examples=300, deadline=None)
    @given(_endpoints, _endpoints, _endpoints, _endpoints)
    def test_property_allen_converse_symmetry(a0: int, a1: int, b0: int, b1: int) -> None:
        a = TimeInterval(min(a0, a1), max(a0, a1))
        b = TimeInterval(min(b0, b1), max(b0, b1))
        assert allen_relation(b, a) is converse(allen_relation(a, b))
        assert intervals_concurrent(a, b) == intervals_concurrent(b, a)


# --------------------------------------------------------------------------- #
# Haversine distance and the travel check.
# --------------------------------------------------------------------------- #
def test_earth_radius_is_the_iugg_mean_radius() -> None:
    # Moritz (2000): R1 = 6 371 008.7714 m; one equatorial degree is then
    # 6371.0088 * pi / 180 = 111.19508 km.
    assert EARTH_MEAN_RADIUS_KM == 6371.0088
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19508, abs=1e-5)
    assert haversine_km(90, 0, -90, 0) == pytest.approx(20015.1144, abs=1e-3)


def test_haversine_reference_distances() -> None:
    # One degree of arc on the equator is R * pi / 180; pole to pole and
    # antipodal points on the equator are pi * R.
    assert haversine_km(0, 0, 0, 1) == pytest.approx(R * math.pi / 180, rel=1e-12)
    assert haversine_km(90, 0, -90, 0) == pytest.approx(math.pi * R, rel=1e-12)
    assert haversine_km(0, 0, 0, 180) == pytest.approx(math.pi * R, rel=1e-12)
    assert haversine_km(12.5, 7.0, 12.5, 7.0) == 0.0


def test_haversine_matches_spherical_law_of_cosines() -> None:
    lat1, lon1, lat2, lon2 = 51.5074, -0.1278, 48.8566, 2.3522
    p1, p2 = math.radians(lat1), math.radians(lat2)
    central = math.acos(
        math.sin(p1) * math.sin(p2)
        + math.cos(p1) * math.cos(p2) * math.cos(math.radians(lon2 - lon1))
    )
    assert haversine_km(lat1, lon1, lat2, lon2) == pytest.approx(R * central, rel=1e-9)
    assert haversine_km(lat2, lon2, lat1, lon1) == pytest.approx(R * central, rel=1e-9)


def _obs(lat: float, lon: float, t0: float, t1: float, radius: float = 0.0) -> LocatedObservation:
    return LocatedObservation(
        "A", Location(lat, lon, radius), TimeInterval(t0, t1), Reference("claim", "s", 0, "A")
    )


def test_travel_speed_hand_computed() -> None:
    # 1 degree on the equator (111.19508 km) covered in one hour.
    check = HaversineTravelCheck()
    got = check.assess(_obs(0, 0, 0, 0), _obs(0, 1, 3600, 3600))
    assert got.gap_seconds == 3600.0
    assert got.required_speed_kmh == pytest.approx(R * math.pi / 180, rel=1e-12)
    assert got.feasible
    # The gap is measured end-to-start whichever observation is passed first.
    back = check.assess(_obs(0, 1, 3600, 4000), _obs(0, 0, 0, 100))
    assert back.gap_seconds == 3500.0
    assert check.max_speed_kmh == DEFAULT_MAX_SPEED_KMH == 1000.0


def test_travel_speed_limit_is_inclusive() -> None:
    first, second = _obs(0, 0, 0, 0), _obs(0, 1, 360, 360)
    speed = HaversineTravelCheck().assess(first, second).required_speed_kmh
    assert HaversineTravelCheck(max_speed_kmh=speed).assess(first, second).feasible
    tighter = HaversineTravelCheck(max_speed_kmh=math.nextafter(speed, 0.0))
    assert not tighter.assess(first, second).feasible


def test_travel_concurrent_or_meeting_far_apart_is_infinite_speed() -> None:
    check = HaversineTravelCheck()
    overlap = check.assess(_obs(0, 0, 0, 100), _obs(0, 1, 50, 150))
    assert overlap.required_speed_kmh == math.inf and not overlap.feasible
    meet = check.assess(_obs(0, 0, 0, 100), _obs(0, 1, 100, 200))
    assert meet.gap_seconds == 0.0 and meet.required_speed_kmh == math.inf


def test_travel_radii_absorb_distance() -> None:
    # 111.195 km apart with radii 60 + 60 km: the circles overlap.
    got = HaversineTravelCheck().assess(_obs(0, 0, 0, 1, 60.0), _obs(0, 1, 0, 1, 60.0))
    assert got.effective_distance_km == 0.0 and got.required_speed_kmh == 0.0 and got.feasible
    partial = HaversineTravelCheck().assess(_obs(0, 0, 0, 0, 10.0), _obs(0, 1, 3600, 3600, 1.0))
    assert partial.effective_distance_km == pytest.approx(R * math.pi / 180 - 11.0, rel=1e-12)
    assert partial.distance_km == pytest.approx(R * math.pi / 180, rel=1e-12)


# --------------------------------------------------------------------------- #
# Value compatibility.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("a", "b", "tol", "expected"),
    [
        (10, 10.5, ValueTolerance(abs_tol=0.5), True),
        (10, 10.5, ValueTolerance(abs_tol=0.49), False),
        (100, 105, ValueTolerance(rel_tol=0.05), True),  # 5 <= 0.05 * 105
        (100, 106, ValueTolerance(rel_tol=0.05), False),  # 6 > 0.05 * 106
        (100, 105, ValueTolerance(rel_tol=0.0476), False),  # 5 > 4.998
        (1.0, 1.0 + 1e-12, ValueTolerance(), True),
        (1.0, 1.001, ValueTolerance(), False),
        (1, 1.0, ValueTolerance(), True),
        (0.0, 1e-300, ValueTolerance(), False),
    ],
)
def test_numeric_tolerance(a: float, b: float, tol: ValueTolerance, expected: bool) -> None:
    assert values_compatible(a, b, tol) is expected
    assert math.isclose(a, b, rel_tol=tol.rel_tol, abs_tol=tol.abs_tol) is expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Red ", "red", True),
        ("dark  blue", "Dark Blue", True),
        ("red", "blue", False),
        (True, True, True),
        (True, False, False),
        (True, 1, False),
        ("true", True, False),
        ("10", 10, False),
    ],
)
def test_categorical_compatibility(a: Any, b: Any, expected: bool) -> None:
    assert values_compatible(a, b, ValueTolerance()) is expected


# --------------------------------------------------------------------------- #
# Likelihood ratios.
# --------------------------------------------------------------------------- #
def test_log_likelihood_ratio_symmetric_model() -> None:
    assert log_likelihood_ratio(0.8, Stance.SUPPORTS) == pytest.approx(LN(4), rel=1e-15)
    assert log_likelihood_ratio(0.8, Stance.REFUTES) == pytest.approx(-LN(4), rel=1e-15)
    assert log_likelihood_ratio(0.9, Stance.SUPPORTS) == pytest.approx(LN(9), rel=1e-15)


def test_log_likelihood_ratio_clips_with_epsilon() -> None:
    assert log_likelihood_ratio(1.0, Stance.SUPPORTS) == pytest.approx(LN(999), rel=1e-12)
    assert log_likelihood_ratio(0.99, Stance.SUPPORTS, epsilon=0.05) == pytest.approx(
        LN(19), rel=1e-12
    )
    assert log_likelihood_ratio(0.0, Stance.SUPPORTS, low_reliability_policy="invert") == (
        pytest.approx(-LN(999), rel=1e-12)
    )


def test_log_likelihood_ratio_low_reliability_policy() -> None:
    assert log_likelihood_ratio(0.5, Stance.SUPPORTS) is None
    assert log_likelihood_ratio(0.5, Stance.SUPPORTS, low_reliability_policy="invert") is None
    assert log_likelihood_ratio(0.3, Stance.SUPPORTS) is None
    assert log_likelihood_ratio(0.3, Stance.SUPPORTS, low_reliability_policy="invert") == (
        pytest.approx(LN(0.3 / 0.7), rel=1e-12)
    )
    assert log_likelihood_ratio(0.3, Stance.REFUTES, low_reliability_policy="invert") == (
        pytest.approx(LN(0.7 / 0.3), rel=1e-12)
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"epsilon": 0.0}, {"epsilon": 0.5}, {"low_reliability_policy": "flip"}],
)
def test_log_likelihood_ratio_rejects_bad_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        log_likelihood_ratio(0.8, Stance.SUPPORTS, **kwargs)
    with pytest.raises(ValueError):
        log_likelihood_ratio(1.2, Stance.SUPPORTS)


# --------------------------------------------------------------------------- #
# Corroboration.
# --------------------------------------------------------------------------- #
def test_independent_clusters_add_log_odds() -> None:
    # Prior 0.5, two independent supporting sources r = 0.8 and r = 0.9:
    # log-odds ln 4 + ln 9 = ln 36, posterior 36 / 37.
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [_support("e1", A_CAR, 0.8, source_id="dmv"), _support("e2", A_CAR, 0.9, source_id="cam")]
    c = _verify([st_], ev).corroboration[0]
    assert c.log_likelihood_ratio == pytest.approx(LN(36), rel=1e-14)
    assert c.posterior == pytest.approx(36 / 37, rel=1e-14)
    assert c.prior_log_odds == 0.0
    assert c.posterior_log_odds == pytest.approx(LN(36), rel=1e-14)
    assert c.status is CorroborationStatus.CORROBORATED
    assert not c.insufficient_evidence
    assert [x.cluster_id for x in c.contributions] == ["source:cam", "source:dmv"]


def test_dependent_items_count_once_strongest_wins() -> None:
    # Same source: only the strongest item (r = 0.9) counts, posterior 0.9.
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [_support("e1", A_CAR, 0.8, source_id="dmv"), _support("e2", A_CAR, 0.9, source_id="dmv")]
    c = _verify([st_], ev).corroboration[0]
    (only,) = c.contributions
    assert (only.evidence_id, only.n_items, only.conflicting) == ("e2", 2, False)
    assert only.log_lr == pytest.approx(LN(9), rel=1e-15)
    assert c.posterior == pytest.approx(0.9, rel=1e-14)
    assert c.status is CorroborationStatus.UNRESOLVED
    assert [m.evidence_id for m in only.members] == ["e2", "e1"]


def test_declared_cluster_overrides_source() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [
        _support("e1", A_CAR, 0.8, source_id="x", independence_cluster="wire"),
        _support("e2", A_CAR, 0.8, source_id="y", independence_cluster="wire"),
    ]
    c = _verify([st_], ev).corroboration[0]
    assert [x.cluster_id for x in c.contributions] == ["cluster:wire"]
    # Equal magnitudes in one direction: the smallest evidence id is chosen.
    assert c.contributions[0].evidence_id == "e1"
    assert c.posterior == pytest.approx(0.8, rel=1e-14)


def test_cluster_tied_in_opposite_directions_contributes_nothing() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [_support("e1", A_CAR, 0.8, source_id="s"), _refute("e2", A_CAR, 0.8, source_id="s")]
    c = _verify([st_], ev).corroboration[0]
    (only,) = c.contributions
    assert (only.evidence_id, only.stance, only.log_lr, only.conflicting) == (None, None, 0.0, True)
    assert c.insufficient_evidence and c.status is CorroborationStatus.UNRESOLVED
    assert c.posterior == 0.5


def test_cluster_with_mixed_directions_is_flagged_but_strongest_counts() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [_support("e1", A_CAR, 0.9, source_id="s"), _refute("e2", A_CAR, 0.8, source_id="s")]
    (only,) = _verify([st_], ev).corroboration[0].contributions
    assert only.evidence_id == "e1" and only.conflicting
    assert only.log_lr == pytest.approx(LN(9), rel=1e-15)


def test_refuting_clusters_contradict() -> None:
    # ln(1/9) twice: posterior 1 / 82.
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [_refute("e1", A_CAR, 0.9), _refute("e2", A_CAR, 0.9)]
    c = _verify([st_], ev).corroboration[0]
    assert c.posterior == pytest.approx(1 / 82, rel=1e-13)
    assert c.status is CorroborationStatus.CONTRADICTED


def test_prior_enters_the_log_odds() -> None:
    # Prior 0.2 (log-odds ln 0.25) plus ln 4 = 0: posterior 0.5.
    st_ = _one_claim(Claim("A", "car", "red"))
    c = _verify(
        [st_], [_support("e1", A_CAR, 0.8)], config=VerifierConfig(prior_probability=0.2)
    ).corroboration[0]
    assert c.prior == 0.2
    assert c.prior_log_odds == pytest.approx(LN(0.25), rel=1e-14)
    assert c.posterior == pytest.approx(0.5, abs=1e-15)


def test_status_needs_evidence_pointing_the_same_way() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    # A high prior with weak refuting evidence stays above 0.95 (0.9557) but is
    # not "corroborated": the evidence points the other way.
    high = _verify(
        [st_], [_refute("e1", A_CAR, 0.6)], config=VerifierConfig(prior_probability=0.97)
    ).corroboration[0]
    assert high.posterior == pytest.approx(1 / (1 + (0.03 / 0.97) * 1.5), rel=1e-13)
    assert high.posterior > 0.95 and high.status is CorroborationStatus.UNRESOLVED
    low = _verify(
        [st_], [_support("e1", A_CAR, 0.6)], config=VerifierConfig(prior_probability=0.03)
    ).corroboration[0]
    assert low.posterior < 0.05 and low.status is CorroborationStatus.UNRESOLVED


def test_a_prior_past_the_threshold_cannot_carry_the_claim_alone() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    # Prior 0.96 plus one item at r = 0.5001 (log LR 0.0004) is above 0.95, but
    # the evidence said almost nothing: it must reach 0.95 on its own (B >= 19).
    weak = _verify(
        [st_], [_support("e1", A_CAR, 0.5001)], config=VerifierConfig(prior_probability=0.96)
    ).corroboration[0]
    assert weak.posterior > 0.95 and weak.status is CorroborationStatus.UNRESOLVED
    strong = _verify(
        [st_], [_support("e1", A_CAR, 0.95)], config=VerifierConfig(prior_probability=0.96)
    ).corroboration[0]
    assert strong.status is CorroborationStatus.CORROBORATED
    weak_low = _verify(
        [st_], [_refute("e1", A_CAR, 0.5001)], config=VerifierConfig(prior_probability=0.04)
    ).corroboration[0]
    assert weak_low.posterior < 0.05 and weak_low.status is CorroborationStatus.UNRESOLVED
    # Below the threshold the prior is the caller's declared base rate and counts.
    based = _verify(
        [st_], [_support("e1", A_CAR, 0.8)], config=VerifierConfig(prior_probability=0.9)
    ).corroboration[0]
    assert based.posterior > 0.95 and based.status is CorroborationStatus.CORROBORATED


def test_a_bayes_factor_of_19_reaches_095_despite_rounding() -> None:
    # One item at r = 0.95 is a Bayes factor of exactly 19; in floating point
    # the posterior is 0.9499999999999998, which must still count as 0.95.
    st_ = _one_claim(Claim("A", "car", "red"))
    c = _verify([st_], [_support("e1", A_CAR, 0.95)]).corroboration[0]
    assert c.posterior == pytest.approx(0.95, abs=1e-15)
    assert c.status is CorroborationStatus.CORROBORATED
    r = _verify([st_], [_refute("e1", A_CAR, 0.95)]).corroboration[0]
    assert r.status is CorroborationStatus.CONTRADICTED


def _shift(p: float, log_odds: float) -> float:
    """``p`` moved by ``log_odds`` on the log-odds scale."""
    return 1.0 / (1.0 + math.exp(-(math.log(p / (1.0 - p)) + log_odds)))


def test_status_thresholds_are_inclusive() -> None:
    # Thresholds compare in log-odds with a 1e-12 tolerance (so a Bayes factor
    # of exactly 19 reaches 0.95); a threshold 1e-9 log-odds past the posterior
    # is beyond it.
    st_ = _one_claim(Claim("A", "car", "red"))
    support = [_support("e1", A_CAR, 0.9)]
    p = _verify([st_], support).corroboration[0].posterior
    at = _verify([st_], support, config=VerifierConfig(corroborated_at=p))
    assert at.corroboration[0].status is CorroborationStatus.CORROBORATED
    above = _verify([st_], support, config=VerifierConfig(corroborated_at=_shift(p, 1e-9)))
    assert above.corroboration[0].status is CorroborationStatus.UNRESOLVED

    refute = [_refute("e1", A_CAR, 0.9)]
    q = _verify([st_], refute).corroboration[0].posterior
    at_lo = _verify([st_], refute, config=VerifierConfig(contradicted_at=q))
    assert at_lo.corroboration[0].status is CorroborationStatus.CONTRADICTED
    below = _verify([st_], refute, config=VerifierConfig(contradicted_at=_shift(q, -1e-9)))
    assert below.corroboration[0].status is CorroborationStatus.UNRESOLVED


def test_missing_and_uninformative_reliability_abstain() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [
        EvidenceItem("e1", target=A_CAR, stance=Stance.SUPPORTS, source_id="unknown"),
        _support("e2", A_CAR, 0.5),
        _support("e3", A_CAR, 0.3),
    ]
    c = _verify([st_], ev).corroboration[0]
    reasons = {x.evidence_id: x.reason for x in c.excluded}
    assert set(reasons) == {"e1", "e2", "e3"}
    assert reasons["e1"].startswith("no reliability available")
    assert reasons["e2"].startswith("uninformative")
    assert "below 0.5" in reasons["e3"]
    assert c.contributions == ()
    assert c.insufficient_evidence and c.posterior == 0.5
    assert c.status is CorroborationStatus.UNRESOLVED


def test_invert_policy_counts_low_reliability_against_the_report() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    c = _verify(
        [st_],
        [_support("e1", A_CAR, 0.1)],
        config=VerifierConfig(low_reliability_policy="invert"),
    ).corroboration[0]
    assert c.posterior == pytest.approx(0.1, rel=1e-13)


def test_reliability_table_and_item_precedence() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    table = TableSourceReliability({"dmv": 0.8})
    ev = [EvidenceItem("e1", target=A_CAR, stance=Stance.SUPPORTS, source_id="dmv")]
    c = _verify([st_], ev, reliability_model=table).corroboration[0]
    assert c.posterior == pytest.approx(0.8, rel=1e-14)
    own = [
        EvidenceItem("e1", target=A_CAR, stance=Stance.SUPPORTS, source_id="dmv", reliability=0.9)
    ]
    c2 = _verify([st_], own, reliability_model=table).corroboration[0]
    assert c2.contributions[0].members[0].reliability == 0.9


def test_reliability_is_clipped_in_corroboration() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    c = _verify([st_], [_support("e1", A_CAR, 1.0)]).corroboration[0]
    member = c.contributions[0].members[0]
    assert (member.reliability, member.reliability_used) == (1.0, 0.999)
    assert c.posterior == pytest.approx(0.999, rel=1e-13)


def test_ambiguous_target_is_excluded_until_disambiguated() -> None:
    s1 = Statement("s1", "A", 0.0, claims=(Claim("A", "location", "home"),))
    s2 = Statement("s2", "A", 1.0, claims=(Claim("A", "location", "office"),))
    loose = [_support("e1", A_LOC, 0.9)]
    rep = _verify([s1, s2], loose)
    for c in rep.corroboration:
        assert [x.evidence_id for x in c.excluded] == ["e1"]
        assert c.excluded[0].reason.startswith("ambiguous target")
        assert c.contributions == ()
    pinned = [_support("e1", A_LOC, 0.9, target_statement_id="s2")]
    by_sid = {c.statement_id: c for c in _verify([s1, s2], pinned).corroboration}
    assert by_sid["s1"].contributions == () and by_sid["s1"].excluded == ()
    assert by_sid["s2"].posterior == pytest.approx(0.9, rel=1e-14)


def test_target_without_disagreement_is_not_ambiguous() -> None:
    s1 = Statement("s1", "A", 0.0, claims=(Claim("A", "location", "home"),))
    s2 = Statement("s2", "B", 1.0, claims=(Claim("A", "location", "Home "),))
    rep = _verify([s1, s2], [_support("e1", A_LOC, 0.9)])
    assert all(c.posterior == pytest.approx(0.9, rel=1e-14) for c in rep.corroboration)


def test_ambiguity_also_covers_disjoint_locations() -> None:
    s1 = Statement("s1", "A", 0.0, claims=(Claim("A", "location", "x", location=Location(0, 0)),))
    s2 = Statement("s2", "A", 1.0, claims=(Claim("A", "location", "x", location=Location(0, 1)),))
    rep = _verify([s1, s2], [_support("e1", A_LOC, 0.9)])
    assert all(c.excluded and c.contributions == () for c in rep.corroboration)


def test_evidence_at_other_times_does_not_bear() -> None:
    st_ = _one_claim(Claim("A", "location", "home", TimeInterval(0, 100)))
    ev = [_support("e1", A_LOC, 0.9, time=TimeInterval(200, 300))]
    c = _verify([st_], ev).corroboration[0]
    assert c.contributions == () and c.excluded == ()
    ev2 = [_support("e1", A_LOC, 0.9, time=TimeInterval(-10, 110))]
    assert _verify([st_], ev2).corroboration[0].contributions != ()


def test_support_must_cover_the_whole_claimed_interval() -> None:
    # Two independent sightings of one instant each inside an 8-hour claim used
    # to corroborate it (posterior 0.988): nothing covered the other 7h59m.
    st_ = _one_claim(Claim("A", "location", "office", TimeInterval(0, 28_800)))
    blips = [
        _support("e1", A_LOC, 0.9, source_id="cam-1", time=TimeInterval(100, 101)),
        _support("e2", A_LOC, 0.9, source_id="cam-2", time=TimeInterval(200, 201)),
    ]
    c = _verify([st_], blips).corroboration[0]
    assert c.contributions == () and c.status is CorroborationStatus.UNRESOLVED
    assert {e.reason for e in c.excluded} == {
        "evidence covers only part of the claimed interval: it can refute the claim, not "
        "support it"
    }
    for covering in (TimeInterval(0, 28_800), TimeInterval(0, 30_000), TimeInterval(-5, 28_800)):
        whole = [_support("e3", A_LOC, 0.9, time=covering)]
        assert _verify([st_], whole).corroboration[0].contributions[0].stance is Stance.SUPPORTS
    # One instant of contrary evidence inside the interval still refutes it.
    contrary = [_refute("e4", A_LOC, 0.9, time=TimeInterval(100, 101))]
    assert _verify([st_], contrary).corroboration[0].contributions[0].stance is Stance.REFUTES


def test_untimed_evidence_does_not_bear_on_a_timed_claim() -> None:
    timed = _one_claim(Claim("A", "location", "office", TimeInterval(50_000, 50_100)))
    badge = [_support("e1", A_LOC, 0.95)]
    c = _verify([timed], badge).corroboration[0]
    assert c.contributions == () and c.status is CorroborationStatus.UNRESOLVED
    assert [e.reason for e in c.excluded] == [
        "evidence has no time, so it cannot bear on a claim about a specific interval"
    ]
    timeless = _one_claim(Claim("A", "location", "office"))
    assert _verify([timeless], badge).corroboration[0].contributions != ()


def test_asserted_evidence_supports_or_refutes_by_value_and_place() -> None:
    claim = Claim("A", "location", "home", TimeInterval(0, 100), Location(0, 0, 1.0))
    st_ = _one_claim(claim)
    same = EvidenceItem(
        "e1", asserted=Claim("A", "location", "HOME", TimeInterval(0, 100)), reliability=0.8
    )
    other = EvidenceItem(
        "e2", asserted=Claim("A", "location", "office", TimeInterval(10, 20)), reliability=0.8
    )
    far = EvidenceItem(
        "e3",
        asserted=Claim("A", "location", "home", TimeInterval(10, 20), Location(0, 1, 1.0)),
        reliability=0.8,
    )
    near = EvidenceItem(
        "e4",
        asserted=Claim("A", "location", "home", TimeInterval(0, 100), Location(0, 0.01, 0.2)),
        reliability=0.8,
    )
    stances = {}
    for item in (same, other, far, near):
        c = _verify([st_], [item]).corroboration[0]
        stances[item.evidence_id] = c.contributions[0].stance
    assert stances == {
        "e1": Stance.SUPPORTS,
        "e2": Stance.REFUTES,
        "e3": Stance.REFUTES,
        "e4": Stance.SUPPORTS,  # 1.112 km apart <= 1.0 + 0.2 km radii
    }


def test_asserted_evidence_uses_its_time_field_without_interval() -> None:
    st_ = _one_claim(Claim("A", "location", "home", TimeInterval(0, 100)))
    item = EvidenceItem(
        "e1",
        asserted=Claim("A", "location", "office"),
        time=TimeInterval(500, 600),
        reliability=0.9,
    )
    rep = _verify([st_], [item])
    assert rep.corroboration[0].contributions == ()
    assert rep.contradictions == ()


def test_custom_protocol_models_plug_in() -> None:
    class Fixed:
        def reliability(self, item: EvidenceItem) -> float | None:
            return 0.75

    class OneCluster:
        def cluster_of(self, item: EvidenceItem) -> str:
            return "all"

    assert isinstance(Fixed(), SourceReliabilityModel)
    assert isinstance(OneCluster(), IndependenceClusterModel)
    assert isinstance(DeclaredIndependenceClusters(), IndependenceClusterModel)
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [EvidenceItem(f"e{i}", target=A_CAR, stance=Stance.SUPPORTS) for i in range(3)]
    c = _verify([st_], ev, reliability_model=Fixed(), cluster_model=OneCluster()).corroboration[0]
    assert [x.cluster_id for x in c.contributions] == ["all"]
    assert c.posterior == pytest.approx(0.75, rel=1e-14)


def test_invalid_reliability_from_a_model_is_rejected() -> None:
    class Broken:
        def reliability(self, item: EvidenceItem) -> float | None:
            return 1.5

    with pytest.raises(ValueError):
        _verify(
            [_one_claim(Claim("A", "car", "red"))],
            [_support("e", A_CAR, 0.9)],
            reliability_model=Broken(),
        )


def test_corroboration_maps_onto_intel_structures() -> None:
    from omni_mercury_engine.intel.provenance import Provenance, ProvenanceOrigin
    from omni_mercury_engine.intel.verifier_loop import ClaimStatus, ClaimVerdict

    assert CorroborationStatus.CORROBORATED.to_claim_status() is ClaimStatus.CONFIRMED
    assert CorroborationStatus.CONTRADICTED.to_claim_status() is ClaimStatus.REFUTED
    assert CorroborationStatus.UNRESOLVED.to_claim_status() is ClaimStatus.UNAVAILABLE

    human = Provenance(ProvenanceOrigin.HUMAN, ("interview-7",), verified=True)
    quoted = Provenance(ProvenanceOrigin.EXTRACTIVE, ("cctv-3",), verified=False)
    st_ = _one_claim(Claim("A", "car", "red"))
    ev = [
        _support("e1", A_CAR, 0.9, source_id="a", provenance=human),
        _support("e2", A_CAR, 0.9, source_id="b", provenance=quoted),
    ]
    c = _verify([st_], ev).corroboration[0]
    merged = c.provenance
    assert merged is not None
    assert merged.origin is ProvenanceOrigin.EXTRACTIVE  # weakest wins
    assert merged.sources == ("interview-7", "cctv-3") and merged.verified is False
    verdict = c.as_claim_verdict()
    assert isinstance(verdict, ClaimVerdict)
    assert verdict.status is ClaimStatus.CONFIRMED
    assert verdict.kind == "evidence_corroboration"
    assert "2 informative" in verdict.reason

    partial = [
        _support("e1", A_CAR, 0.9, source_id="a", provenance=human),
        _support("e2", A_CAR, 0.9, source_id="b"),
    ]
    assert _verify([st_], partial).corroboration[0].provenance is None
    assert _verify([st_], []).corroboration[0].provenance is None


# --------------------------------------------------------------------------- #
# Contradictions.
# --------------------------------------------------------------------------- #
def test_value_contradiction_scopes() -> None:
    s1 = Statement(
        "s1",
        "A",
        0.0,
        claims=(Claim("A", "car", "red"), Claim("A", "car", "blue")),
    )
    s2 = Statement("s2", "A", 1.0, claims=(Claim("A", "car", "green"),))
    s3 = Statement("s3", "B", 2.0, claims=(Claim("A", "car", "red"),))
    rep = _verify([s3, s2, s1])
    found = {
        (c.scope, tuple(r.identifier for r in c.references), c.values) for c in rep.contradictions
    }
    assert found == {
        (ContradictionScope.WITHIN_STATEMENT, ("s1", "s1"), ("red", "blue")),
        (ContradictionScope.SAME_SPEAKER, ("s1", "s2"), ("red", "green")),
        (ContradictionScope.SAME_SPEAKER, ("s1", "s2"), ("blue", "green")),
        (ContradictionScope.BETWEEN_SPEAKERS, ("s1", "s3"), ("blue", "red")),
        (ContradictionScope.BETWEEN_SPEAKERS, ("s2", "s3"), ("green", "red")),
    }
    assert all(c.kind is ContradictionKind.VALUE for c in rep.contradictions)
    assert rep.summary.contradictions_by_scope["within_statement"] == 1
    assert rep.summary.contradictions_by_kind["value"] == 5


def test_value_contradiction_needs_overlapping_time() -> None:
    meets = Statement(
        "s1",
        "A",
        0.0,
        claims=(
            Claim("A", "location", "home", TimeInterval(0, 10)),
            Claim("A", "location", "office", TimeInterval(10, 20)),
        ),
    )
    assert _verify([meets]).contradictions == ()
    overlaps = Statement(
        "s1",
        "A",
        0.0,
        claims=(
            Claim("A", "location", "home", TimeInterval(0, 11)),
            Claim("A", "location", "office", TimeInterval(10, 20)),
        ),
    )
    assert len(_verify([overlaps]).contradictions) == 1


def test_predicate_tolerance_suppresses_numeric_contradiction() -> None:
    st_ = Statement(
        "s1", "A", 0.0, claims=(Claim("A", "height_cm", 180), Claim("A", "height_cm", 182))
    )
    assert len(_verify([st_]).contradictions) == 1
    cfg = VerifierConfig(predicate_tolerances={"height_cm": ValueTolerance(abs_tol=2.0)})
    assert _verify([st_], config=cfg).contradictions == ()
    assert cfg.tolerance_for("height_cm").abs_tol == 2.0
    assert cfg.tolerance_for("weight") == ValueTolerance()


def test_claim_against_asserted_evidence_is_a_value_contradiction() -> None:
    st_ = _one_claim(Claim("A", "location", "home", TimeInterval(0, 100)))
    ev = [
        EvidenceItem("e1", asserted=Claim("A", "location", "office", TimeInterval(50, 60))),
        EvidenceItem("e2", asserted=Claim("A", "location", "home", TimeInterval(50, 60))),
        EvidenceItem("e3", asserted=Claim("A", "location", "office", TimeInterval(150, 160))),
    ]
    (c,) = _verify([st_], ev).contradictions
    assert c.kind is ContradictionKind.VALUE and c.scope is ContradictionScope.CLAIM_VS_EVIDENCE
    assert [r.identifier for r in c.references] == ["s1", "e1"]
    assert c.values == ("home", "office")


def _ordering(
    first: str, second: str, *relations: AllenRelation, subject: str = "A"
) -> TemporalOrdering:
    return TemporalOrdering(
        ClaimKey(subject, first), ClaimKey(subject, second), frozenset(relations)
    )


def test_ordering_checked_against_evidence_times() -> None:
    st_ = Statement("s1", "A", 0.0, orderings=(_ordering("left", "alarm", AllenRelation.BEFORE),))
    left = ClaimKey("A", "left")
    alarm = ClaimKey("A", "alarm")
    consistent = [
        _support("e1", left, 0.9, time=TimeInterval.at(100)),
        _support("e2", alarm, 0.9, time=TimeInterval.at(200)),
        # A refuting item says the event did not happen then, so its time is
        # not an event time: at 300 s it would otherwise contradict "before".
        _refute("e3", left, 0.9, time=TimeInterval.at(300)),
    ]
    assert _verify([st_], consistent).contradictions == ()
    against = [
        _support("e1", left, 0.9, time=TimeInterval.at(300)),
        _support("e2", alarm, 0.9, time=TimeInterval.at(200)),
        _refute("e3", left, 0.9, time=TimeInterval.at(50)),  # refuting items give no time
    ]
    (c,) = _verify([st_], against).contradictions
    assert c.kind is ContradictionKind.TEMPORAL
    assert c.scope is ContradictionScope.CLAIM_VS_EVIDENCE
    assert c.observed_relation is AllenRelation.AFTER
    assert c.claimed_relations == {AllenRelation.BEFORE}
    assert [r.identifier for r in c.references] == ["s1", "e1", "e2"]


def test_ordering_evidence_times_skip_items_reporting_another_value() -> None:
    st_ = Statement(
        "s1",
        "A",
        0.0,
        claims=(Claim("A", "left", "front door"),),
        orderings=(_ordering("left", "alarm", AllenRelation.BEFORE),),
    )
    ev = [
        EvidenceItem("e1", asserted=Claim("A", "left", "back door", TimeInterval.at(300))),
        EvidenceItem("e2", asserted=Claim("A", "alarm", True, TimeInterval.at(200))),
        EvidenceItem("e3", asserted=Claim("A", "left", "front door", TimeInterval.at(100))),
    ]
    temporal = [
        c for c in _verify([st_], ev).contradictions if c.kind is ContradictionKind.TEMPORAL
    ]
    assert temporal == []
    # Pinned to another statement, a target-form time is ignored as well.
    pinned = [
        _support(
            "e1", ClaimKey("A", "left"), 0.9, time=TimeInterval.at(300), target_statement_id="sX"
        ),
        _support("e2", ClaimKey("A", "alarm"), 0.9, time=TimeInterval.at(200)),
    ]
    assert _verify([st_], pinned).contradictions == ()


def test_ordering_against_the_statements_own_times() -> None:
    st_ = Statement(
        "s1",
        "A",
        0.0,
        claims=(
            Claim("A", "left", True, TimeInterval(50, 60)),
            Claim("A", "alarm", True, TimeInterval(10, 20)),
        ),
        orderings=(_ordering("left", "alarm", AllenRelation.BEFORE, AllenRelation.MEETS),),
    )
    (c,) = _verify([st_]).contradictions
    assert c.scope is ContradictionScope.WITHIN_STATEMENT
    assert c.observed_relation is AllenRelation.AFTER
    assert [r.kind for r in c.references] == ["ordering", "claim", "claim"]
    assert "{before, meets}" in c.description


def test_orderings_compared_through_converse() -> None:
    s1 = Statement("s1", "A", 0.0, orderings=(_ordering("x", "y", AllenRelation.BEFORE),))
    agree = Statement("s2", "A", 1.0, orderings=(_ordering("y", "x", AllenRelation.AFTER),))
    assert _verify([s1, agree]).contradictions == ()
    disagree = Statement("s2", "A", 1.0, orderings=(_ordering("y", "x", AllenRelation.BEFORE),))
    (c,) = _verify([s1, disagree]).contradictions
    assert c.scope is ContradictionScope.SAME_SPEAKER and c.kind is ContradictionKind.TEMPORAL
    assert c.claimed_relations == {AllenRelation.BEFORE, AllenRelation.AFTER}
    same_dir = Statement("s3", "B", 1.0, orderings=(_ordering("x", "y", AllenRelation.AFTER),))
    (c2,) = _verify([s1, same_dir]).contradictions
    assert c2.scope is ContradictionScope.BETWEEN_SPEAKERS
    unrelated = Statement("s4", "A", 2.0, orderings=(_ordering("x", "z", AllenRelation.AFTER),))
    assert _verify([s1, unrelated]).contradictions == ()


def test_spatial_impossibility() -> None:
    home = Claim("A", "location", "home", TimeInterval(0, 3600), Location(52.0, 4.0, 1.0))
    st_ = _one_claim(home)
    seen = EvidenceItem(
        "e1",
        asserted=Claim("A", "seen_at", "station", TimeInterval.at(1000), Location(48.0, 2.0)),
    )
    elsewhere = EvidenceItem(
        "e2",
        asserted=Claim("A", "seen_at", "airport", TimeInterval.at(1000), Location(40.0, 2.0)),
    )
    reachable = EvidenceItem(
        "e3",
        asserted=Claim("A", "seen_at", "cafe", TimeInterval.at(7200), Location(52.0, 4.1)),
    )
    other_subject = EvidenceItem(
        "e4", asserted=Claim("B", "seen_at", "x", TimeInterval.at(1000), Location(10.0, 2.0))
    )
    rep = _verify([st_], [seen, elsewhere, reachable, other_subject])
    spatial = [c for c in rep.contradictions if c.kind is ContradictionKind.SPATIAL]
    # Claim vs e1 and claim vs e2; the evidence-evidence pair is not reported.
    assert [tuple(r.identifier for r in c.references) for c in spatial] == [
        ("s1", "e1"),
        ("s1", "e2"),
    ]
    travel = spatial[0].travel
    assert travel is not None
    assert travel.required_speed_kmh == math.inf
    assert travel.distance_km == pytest.approx(haversine_km(52, 4, 48, 2), rel=1e-12)
    assert travel.effective_distance_km == pytest.approx(haversine_km(52, 4, 48, 2) - 1.0)
    assert spatial[0].scope is ContradictionScope.CLAIM_VS_EVIDENCE
    assert spatial[0].keys == (A_LOC, ClaimKey("A", "seen_at"))


def test_spatial_between_claims_and_custom_travel_check() -> None:
    s1 = _one_claim(Claim("A", "location", "x", TimeInterval.at(0), Location(0, 0)), sid="s1")
    s2 = _one_claim(Claim("A", "location", "x", TimeInterval.at(3600), Location(0, 1)), sid="s2")
    assert _verify([s1, s2]).contradictions == ()  # 111 km/h is feasible

    class AlwaysImpossible:
        def assess(self, first: LocatedObservation, second: LocatedObservation) -> TravelAssessment:
            return TravelAssessment(1.0, 1.0, 1.0, 9e9, 1.0, False)

    assert isinstance(AlwaysImpossible(), TravelFeasibilityCheck)
    (c,) = _verify([s1, s2], travel_check=AlwaysImpossible()).contradictions
    assert c.kind is ContradictionKind.SPATIAL and c.scope is ContradictionScope.SAME_SPEAKER
    slow = HaversineTravelCheck(max_speed_kmh=100.0)
    assert len(_verify([s1, s2], travel_check=slow).contradictions) == 1


# --------------------------------------------------------------------------- #
# Consistency, SUE and adaptation.
# --------------------------------------------------------------------------- #
def _sue_case() -> tuple[list[Statement], list[EvidenceItem]]:
    loc = TimeInterval(0, 3600)
    s1 = Statement(
        "s1",
        "A",
        100.0,
        claims=(Claim("A", "location", "home", loc), Claim("A", "car", "red")),
    )
    s2 = Statement(
        "s2",
        "A",
        200.0,
        claims=(Claim("A", "location", "office", loc), Claim("A", "car", "red")),
        disclosed_evidence_ids=frozenset({"e1"}),
    )
    s3 = Statement("s3", "A", 300.0, claims=(Claim("A", "location", "office", loc),))
    ev = [
        EvidenceItem("e1", asserted=Claim("A", "location", "office", TimeInterval.at(10))),
        EvidenceItem("e2", asserted=Claim("A", "car", "blue")),
    ]
    return [s3, s1, s2], ev


def test_sue_inconsistency_rates() -> None:
    statements, ev = _sue_case()
    (sc,) = _verify(statements, ev).consistency
    # s1: e1 refutes location (withheld), e2 refutes car (withheld).
    # s2: e1 supports location (disclosed), e2 refutes car (withheld).
    # s3: e1 supports location (disclosed, cumulative from s2).
    assert (sc.sue.withheld.successes, sc.sue.withheld.trials) == (3, 3)
    assert (sc.sue.disclosed.successes, sc.sue.disclosed.trials) == (0, 2)
    assert sc.sue.withheld == wilson_interval(3, 3)
    assert sc.sue.disclosed == wilson_interval(0, 2)
    assert sc.sue.rate_difference == 1.0
    assert sc.statement_ids == ("s1", "s2", "s3")


def test_sue_rate_difference_abstains_without_disclosed_pairs() -> None:
    st_ = _one_claim(Claim("A", "car", "red"))
    (sc,) = _verify([st_], [EvidenceItem("e2", asserted=Claim("A", "car", "blue"))]).consistency
    assert sc.sue.disclosed.estimate is None and sc.sue.rate_difference is None
    assert sc.sue.withheld.estimate == 1.0


def test_adaptation_to_newly_disclosed_evidence() -> None:
    statements, ev = _sue_case()
    (sc,) = _verify(statements, ev).consistency
    # Opportunities: e1 (new before s2) refutes s1.location -> s2 adapts.
    assert (sc.adaptation.proportion.successes, sc.adaptation.proportion.trials) == (1, 1)
    (event,) = sc.adaptation.events
    assert (event.evidence_id, event.key, event.from_statement, event.to_statement) == (
        "e1",
        A_LOC,
        "s1",
        "s2",
    )
    changes = {(c.kind, c.key, c.from_statement, c.to_statement) for c in sc.changes}
    assert changes == {
        ("changed", A_LOC, "s1", "s2"),
        ("dropped", A_CAR, "s2", "s3"),
    }
    changed = next(c for c in sc.changes if c.kind == "changed")
    assert changed.previous_values == ("home",) and changed.new_values == ("office",)
    assert changed.matches_newly_disclosed == ("e1",)
    assert sc.between_statements and all(
        c.scope is ContradictionScope.SAME_SPEAKER for c in sc.between_statements
    )


def test_no_adaptation_when_the_later_statement_keeps_its_claim() -> None:
    loc = TimeInterval(0, 3600)
    s1 = _one_claim(Claim("A", "location", "home", loc))
    s2 = Statement(
        "s2",
        "A",
        1.0,
        claims=(Claim("A", "location", "home", loc),),
        disclosed_evidence_ids=frozenset({"e1"}),
    )
    s0 = Statement("s0", "A", -1.0, claims=(Claim("A", "phone", "on"),))
    ev = [EvidenceItem("e1", asserted=Claim("A", "location", "office", TimeInterval.at(10)))]
    (sc,) = _verify([s1, s2, s0], ev).consistency
    assert (sc.adaptation.proportion.successes, sc.adaptation.proportion.trials) == (0, 1)
    assert sc.adaptation.events == ()
    kinds = sorted((c.kind, c.key.predicate) for c in sc.changes)
    assert kinds == [("added", "location"), ("dropped", "phone")]


def test_consistency_is_reported_per_speaker() -> None:
    a = Statement("a1", "Ann", 0.0, claims=(Claim("X", "car", "red"), Claim("X", "car", "blue")))
    b = Statement("b1", "Bob", 1.0, claims=(Claim("X", "car", "red"),))
    rep = _verify([b, a])
    assert [s.speaker for s in rep.consistency] == ["Ann", "Bob"]
    ann, bob = rep.consistency
    assert len(ann.within_statement) == 1 and ann.between_statements == ()
    assert bob.within_statement == () and bob.between_statements == ()


# --------------------------------------------------------------------------- #
# Verifiability Approach.
# --------------------------------------------------------------------------- #
def _details(*flags: bool) -> Statement:
    return Statement("s1", "A", 0.0, details=tuple(Detail(f"d{i}", f) for i, f in enumerate(flags)))


def test_verifiability_proportion_without_norms() -> None:
    res = assess_verifiability(_details(True, False, True))
    assert (res.n_details, res.n_verifiable) == (3, 2)
    assert res.proportion == wilson_interval(2, 3)
    assert (res.z_score, res.percentile, res.reference_n) == (None, None, 0)
    assert res.note == VERIFIABILITY_NO_NORM_NOTE


def test_verifiability_against_reference_matches_scipy() -> None:
    reference = [0.2, 0.4, 2 / 3, 0.8, 0.9, 0.5]
    res = assess_verifiability(_details(True, False, True), reference)
    p = 2 / 3
    assert res.percentile == pytest.approx(
        stats.percentileofscore(reference, p, kind="mean"), rel=1e-12
    )
    mean = sum(reference) / len(reference)
    sd = math.sqrt(sum((x - mean) ** 2 for x in reference) / (len(reference) - 1))
    assert res.z_score == pytest.approx((p - mean) / sd, rel=1e-12)
    assert res.reference_n == 6


def test_verifiability_reference_edge_cases() -> None:
    single = assess_verifiability(_details(True, False), [0.5])
    assert single.percentile == 50.0 and single.z_score is None
    flat = assess_verifiability(_details(True, False), [0.3, 0.3])
    assert flat.percentile == 100.0 and flat.z_score is None
    empty = assess_verifiability(_details(True), [])
    assert empty.percentile is None and empty.z_score is None and empty.reference_n == 0
    nothing = assess_verifiability(_details(), [0.5, 0.6])
    assert nothing.proportion.estimate is None and nothing.z_score is None
    assert nothing.percentile is None and "abstained" in nothing.note
    with pytest.raises(ValueError):
        assess_verifiability(_details(True), [1.5])


def test_verifier_passes_reference_through() -> None:
    rep = StatementVerifier().verify([_details(True, True)], verifiability_reference=[0.0, 0.5])
    assert rep.verifiability[0].percentile == 100.0
    assert rep.verifiability[0].z_score == pytest.approx((1.0 - 0.25) / math.sqrt(0.125))


# --------------------------------------------------------------------------- #
# The report: frozen, pure, deterministic, JSON-safe, carefully worded.
# --------------------------------------------------------------------------- #
def _rich_case() -> tuple[list[Statement], list[EvidenceItem]]:
    statements, ev = _sue_case()
    extra = Statement(
        "b1",
        "B",
        50.0,
        claims=(Claim("A", "location", "home", TimeInterval(0, 3600), Location(52.0, 4.0, 1.0)),),
        details=(Detail("met at 9", True), Detail("felt tired", False)),
        orderings=(_ordering("left", "alarm", AllenRelation.BEFORE),),
    )
    more = [
        EvidenceItem(
            "e3",
            asserted=Claim("A", "seen", "x", TimeInterval.at(1000), Location(48.0, 2.0)),
            reliability=0.9,
        ),
        _support("e4", ClaimKey("A", "left"), 0.9, time=TimeInterval.at(500)),
        _support("e5", ClaimKey("A", "alarm"), 0.9, time=TimeInterval.at(400)),
        _support("e6", A_CAR, 0.95, source_id="dmv"),
    ]
    return [*statements, extra], [*ev, *more]


def test_report_is_frozen_and_json_serialisable() -> None:
    statements, ev = _rich_case()
    rep = _verify(statements, ev)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.summary = rep.summary  # type: ignore[misc]
    with pytest.raises(TypeError):
        rep.summary.contradictions_by_kind["value"] = 0  # type: ignore[index]
    data = json.loads(rep.to_json())
    assert data["summary"]["n_claims"] == len(rep.corroboration)
    spatial = [c for c in data["contradictions"] if c["kind"] == "spatial"]
    assert spatial and spatial[0]["travel"]["required_speed_kmh"] == "inf"
    assert data["framing"] == list(SCIENTIFIC_FRAMING)
    assert "National Research Council, 2003" in " ".join(data["framing"])


def test_outputs_never_use_reserved_labels() -> None:
    statements, ev = _rich_case()
    text = _verify(statements, ev).to_json()
    assert not FORBIDDEN.search(text)


def test_inputs_are_not_mutated() -> None:
    statements, ev = _rich_case()
    before = (copy.deepcopy(statements), copy.deepcopy(ev))
    _verify(statements, ev)
    assert (statements, ev) == before


def test_report_is_independent_of_input_order() -> None:
    statements, ev = _rich_case()
    forward = _verify(statements, ev).to_json()
    backward = _verify(list(reversed(statements)), list(reversed(ev))).to_json()
    assert forward == backward


def test_summary_aggregates_only() -> None:
    statements, ev = _rich_case()
    rep = _verify(statements, ev)
    s = rep.summary
    assert s.n_statements == 4 and s.n_evidence == 6
    assert s.n_claims == len(rep.corroboration) == 6
    assert s.n_corroborated + s.n_contradicted + s.n_unresolved == s.n_claims
    statuses = [c.status for c in rep.corroboration]
    assert s.n_corroborated == statuses.count(CorroborationStatus.CORROBORATED)
    assert s.n_contradicted == statuses.count(CorroborationStatus.CONTRADICTED)
    assert s.n_insufficient_evidence == sum(c.insufficient_evidence for c in rep.corroboration)
    assert sum(s.contradictions_by_kind.values()) == len(rep.contradictions)
    assert sum(s.contradictions_by_scope.values()) == len(rep.contradictions)
    assert s.findings[0].startswith(f"{s.n_claims} claim(s) assessed against 6 evidence item(s)")
    assert f"{s.n_corroborated} corroborated" in s.findings[0]
    assert f"{s.n_insufficient_evidence} claim(s) had no informative evidence" in s.findings[1]
    kinds = s.contradictions_by_kind
    assert s.findings[2] == (
        f"{kinds['value']} value, {kinds['temporal']} temporal and {kinds['spatial']} "
        "spatial contradiction(s) found."
    )


def test_alias_and_duplicate_ids() -> None:
    assert TruthDecipherFramework is StatementVerifier
    st_ = _one_claim(Claim("A", "car", "red"))
    with pytest.raises(ValueError, match="duplicate statement_id"):
        _verify([st_, st_])
    with pytest.raises(ValueError, match="duplicate evidence_id"):
        _verify([st_], [_support("e", A_CAR, 0.9), _support("e", A_CAR, 0.8)])


# --------------------------------------------------------------------------- #
# Input validation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build",
    [
        lambda: TimeInterval(2, 1),
        lambda: TimeInterval(0, math.inf),
        lambda: Location(91, 0),
        lambda: Location(0, 181),
        lambda: Location(0, 0, -1),
        lambda: Claim("A", "x", math.nan),
        lambda: Claim("", "x", 1),
        lambda: ClaimKey("A", " "),
        lambda: EvidenceItem("e"),
        lambda: EvidenceItem("e", target=A_CAR),
        lambda: EvidenceItem(
            "e", target=A_CAR, stance=Stance.SUPPORTS, asserted=Claim("A", "car", "r")
        ),
        lambda: EvidenceItem("e", asserted=Claim("A", "car", "r"), stance=Stance.SUPPORTS),
        lambda: EvidenceItem("e", target=A_CAR, stance=Stance.SUPPORTS, reliability=1.01),
        lambda: EvidenceItem(
            "e",
            asserted=Claim("A", "car", "r", TimeInterval(0, 1)),
            time=TimeInterval(5, 6),
        ),
        lambda: Statement("s", "A", 0.0, claims=(Claim("A", "x", 1, statement_id="t"),)),
        lambda: TemporalOrdering(A_CAR, A_CAR, frozenset({AllenRelation.BEFORE})),
        lambda: TemporalOrdering(A_CAR, A_LOC, frozenset()),
        lambda: ValueTolerance(abs_tol=-1),
        lambda: VerifierConfig(prior_probability=1.0),
        lambda: VerifierConfig(corroborated_at=0.05, contradicted_at=0.05),
        lambda: VerifierConfig(reliability_epsilon=0.5),
        lambda: VerifierConfig(low_reliability_policy="flip"),  # type: ignore[arg-type]
        lambda: VerifierConfig(confidence_level=0.0),
        lambda: HaversineTravelCheck(max_speed_kmh=0.0),
        lambda: TableSourceReliability({"x": 2.0}),
    ],
)
def test_invalid_inputs_are_rejected(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: Detail("x", 1),  # type: ignore[arg-type]
        lambda: Claim("A", "x", [1]),  # type: ignore[arg-type]
        lambda: Statement("s", "A", 0.0, disclosed_evidence_ids="e1"),  # type: ignore[arg-type]
        lambda: Statement("s", "A", 0.0, claims=("not a claim",)),  # type: ignore[arg-type]
        lambda: TimeInterval(True, 2),
        lambda: VerifierConfig(predicate_tolerances={"x": 1.0}),  # type: ignore[dict-item]
    ],
)
def test_wrong_types_are_rejected(build: Any) -> None:
    with pytest.raises(TypeError):
        build()


def test_statement_stamps_claims_and_freezes_collections() -> None:
    claims = [Claim("A", "car", "red")]
    disclosed = {"e1"}
    st_ = Statement("s9", "A", 0.0, claims=claims, disclosed_evidence_ids=disclosed)  # type: ignore[arg-type]
    assert st_.claims[0].statement_id == "s9" and claims[0].statement_id is None
    assert isinstance(st_.claims, tuple) and isinstance(st_.disclosed_evidence_ids, frozenset)
    disclosed.add("e2")
    assert st_.disclosed_evidence_ids == {"e1"}
    single = TemporalOrdering(A_CAR, A_LOC, AllenRelation.BEFORE)  # type: ignore[arg-type]
    assert single.relations == {AllenRelation.BEFORE}
    assert str(A_CAR) == "A.car"
    item = EvidenceItem("e", asserted=Claim("A", "car", "r"), time=TimeInterval(1, 2))
    assert item.effective_time == TimeInterval(1, 2) and item.key == A_CAR

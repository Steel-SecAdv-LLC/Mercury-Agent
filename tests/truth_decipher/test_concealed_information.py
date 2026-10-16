# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for the Concealed Information Test in :mod:`omni_mercury_engine.truth_decipher`.

Expected values are computed by hand (arithmetic in the comments) or by an
independent oracle: brute-force enumeration of every rank vector and every
probe relabelling, re-implemented here with plain numpy and itertools.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.truth_decipher import (
    CHANNEL_SIGN,
    CIT_ASSUMPTIONS,
    CIT_CONSENT_SCOPE,
    FINDING_INDICATED,
    FINDING_NOT_INDICATED,
    NOT_EVIDENCE_OF_GUILT,
    Channel,
    CITConfig,
    CITQuestion,
    CITResult,
    ConcealedInformationTest,
    ConsentRecord,
    ConsentRequiredError,
    InsufficientDataError,
    ResponseWindows,
    TimeSeries,
    derive_response_amplitudes,
    exact_rank_sum_p_value,
    heart_rate_change,
    question_from_time_series,
    rank_sum_distribution,
    require_consent,
    respiration_line_length,
    skin_conductance_amplitude,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

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

SC = Channel.SKIN_CONDUCTANCE
HR = Channel.HEART_RATE
RLL = Channel.RESPIRATION_LINE_LENGTH
IRR = ("i1", "i2", "i3", "i4")
FORBIDDEN = re.compile(r"\b(lying|liar|liars|lie|lies|deceptive|deception|truthful)\b", re.I)

CONSENT = ConsentRecord("subj-01", 0.0, frozenset({CIT_CONSENT_SCOPE}), "examiner-7")


def _rows(rows: Any) -> tuple[tuple[float, ...], ...]:
    """A response matrix in the declared type (lists and arrays are also accepted)."""
    return tuple(tuple(float(v) for v in row) for row in rows)


def _q(qid: str, **channels: Sequence[Sequence[float]]) -> CITQuestion:
    responses = {Channel(name): _rows(rows) for name, rows in channels.items()}
    k = len(next(iter(responses.values()))[0])
    return CITQuestion(qid, "probe", tuple(f"i{j}" for j in range(1, k)), responses)


def _run(questions: list[CITQuestion], **cfg: Any) -> CITResult:
    return ConcealedInformationTest(CITConfig(**cfg)).run(
        questions, consent=CONSENT, subject_pseudonym="subj-01", administered_at=10.0
    )


# --------------------------------------------------------------------------- #
# Consent gate.
# --------------------------------------------------------------------------- #
def test_consent_is_required() -> None:
    cit = ConcealedInformationTest()
    q = _q("q1", skin_conductance=[[5, 1, 2, 1, 1]])
    with pytest.raises(ConsentRequiredError, match="required"):
        cit.run([q], consent=None, subject_pseudonym="subj-01", administered_at=10.0)
    with pytest.raises(ConsentRequiredError, match="must be a ConsentRecord"):
        cit.run([q], consent="yes", subject_pseudonym="subj-01", administered_at=10.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("record", "subject", "when", "match"),
    [
        (CONSENT, "subj-02", 10.0, "different subject"),
        (ConsentRecord("subj-01", 0.0, frozenset({"interview"}), "x"), "subj-01", 10.0, "scope"),
        # A bare str scope is accepted and frozen into a one-element set.
        (
            ConsentRecord("subj-01", 11.0, CIT_CONSENT_SCOPE, "x"),  # type: ignore[arg-type]
            "subj-01",
            10.0,
            "after",
        ),
    ],
)
def test_consent_must_cover_this_administration(
    record: ConsentRecord, subject: str, when: float, match: str
) -> None:
    with pytest.raises(ConsentRequiredError, match=match):
        require_consent(record, subject, when)


def test_consent_at_the_administration_instant_is_accepted() -> None:
    record = ConsentRecord("subj-01", 10.0, CIT_CONSENT_SCOPE, "x")  # type: ignore[arg-type]
    assert require_consent(record, "subj-01", 10.0) is record
    assert record.scope == frozenset({CIT_CONSENT_SCOPE})


def test_consent_is_checked_before_any_question_is_read() -> None:
    def questions() -> Iterator[CITQuestion]:
        raise AssertionError("questions were read before consent was checked")
        yield  # pragma: no cover

    with pytest.raises(ConsentRequiredError):
        ConcealedInformationTest().run(
            questions(), consent=None, subject_pseudonym="s", administered_at=0.0
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subject_pseudonym": " ", "obtained_at": 0.0, "scope": "x", "obtained_by": "y"},
        {"subject_pseudonym": "s", "obtained_at": 0.0, "scope": frozenset(), "obtained_by": "y"},
        {"subject_pseudonym": "s", "obtained_at": math.nan, "scope": "x", "obtained_by": "y"},
        {"subject_pseudonym": "s", "obtained_at": 0.0, "scope": "x", "obtained_by": ""},
        {"subject_pseudonym": "s", "obtained_at": 0.0, "scope": ["x", ""], "obtained_by": "y"},
    ],
)
def test_consent_record_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ConsentRecord(**kwargs)


# --------------------------------------------------------------------------- #
# Abstention on publicly disclosed probes.
# --------------------------------------------------------------------------- #
def test_session_level_disclosure_abstains() -> None:
    res = ConcealedInformationTest().run(
        [_q("q1", skin_conductance=[[5, 1, 2, 1, 1]])],
        consent=CONSENT,
        subject_pseudonym="subj-01",
        administered_at=10.0,
        probe_details_publicly_disclosed=True,
    )
    assert res.abstained and res.finding is None
    assert res.abstention_reason is not None and "publicly disclosed" in res.abstention_reason
    assert res.rank_test_p is None and res.permutation_p is None and res.questions == ()
    assert res.not_evidence_of_guilt == NOT_EVIDENCE_OF_GUILT


def test_question_level_disclosure_excludes_that_question() -> None:
    public = CITQuestion(
        "q0", "probe", IRR, {SC: _rows([[9, 0, 0, 0, 0]])}, probe_publicly_disclosed=True
    )
    kept = _q("q1", skin_conductance=[[5, 1, 2, 1, 1]])
    res = _run([public, kept])
    assert [e.question_id for e in res.excluded_questions] == ["q0"]
    assert [q.question_id for q in res.questions] == ["q1"]
    assert res.rank_test_p == pytest.approx(0.2, rel=1e-15)
    only_public = _run([public])
    assert only_public.abstained and only_public.abstention_reason == "no question could be scored"
    assert [e.reason for e in only_public.excluded_questions] == ["probe detail publicly disclosed"]


# --------------------------------------------------------------------------- #
# Standardisation, signs, weights.
# --------------------------------------------------------------------------- #
def test_within_question_standardisation_hand_computed() -> None:
    # [5, 1, 2, 1, 1]: mean 2, population variance 12 / 5, so z = dev / sqrt(2.4).
    res = _run([_q("q1", skin_conductance=[[5, 1, 2, 1, 1]])])
    sd = math.sqrt(2.4)
    (q,) = res.questions
    assert q.item_scores == pytest.approx([3 / sd, -1 / sd, 0.0, -1 / sd, -1 / sd], rel=1e-13)
    assert q.probe_rank == 1
    # Probe minus mean irrelevant: (3 - (-3 / 4)) / sd.
    assert q.standardised_difference == pytest.approx(3.75 / sd, rel=1e-13)
    assert res.effect_size == pytest.approx(3.75 / sd, rel=1e-13)
    assert (q.n_items, q.n_repetitions, q.channels_used) == (5, 1, (SC,))
    assert res.rank_test_p == pytest.approx(0.2, rel=1e-15)
    assert res.permutation_p == pytest.approx(0.2, rel=1e-15)
    assert (res.permutation_method, res.permutations_evaluated) == ("exact", 5)
    assert res.finding == FINDING_NOT_INDICATED and not res.abstained


def test_repetitions_are_standardised_together_then_averaged() -> None:
    # Values 4,0,0 / 2,2,2: mean 10 / 6, all six values enter the z-score.
    rows = [[4, 0, 0], [2, 2, 2]]
    x = np.asarray(rows, dtype=float)
    z = (x - x.mean()) / x.std()
    res = _run([_q("q1", skin_conductance=rows)])
    assert res.questions[0].item_scores == pytest.approx(list(z.mean(axis=0)), rel=1e-13)


def test_channel_signs_make_larger_mean_stronger() -> None:
    assert dict(CHANNEL_SIGN) == {SC: 1, HR: -1, RLL: -1}
    sc = _run([_q("q", skin_conductance=[[9, 1, 2, 1, 1]])])
    hr = _run([_q("q", heart_rate=[[-9, -1, -2, -1, -1]])])  # deceleration
    rll = _run([_q("q", respiration_line_length=[[1, 9, 8, 9, 9]])])  # suppression
    for res in (sc, hr, rll):
        assert res.questions[0].probe_rank == 1
    hr_up = _run([_q("q", heart_rate=[[9, 1, 2, 1, 1]])])  # acceleration
    assert hr_up.questions[0].probe_rank == 5


def test_standardisation_is_invariant_to_affine_channel_scaling() -> None:
    rows = [[5.0, 1.0, 2.0, 1.5, 1.0], [4.0, 2.0, 1.0, 1.0, 0.5]]
    base = _run([_q("q", skin_conductance=rows)])
    scaled = _run([_q("q", skin_conductance=[[7 * v + 3 for v in r] for r in rows])])
    assert scaled.questions[0].item_scores == pytest.approx(
        base.questions[0].item_scores, rel=1e-12, abs=1e-12
    )
    assert scaled.permutation_p == base.permutation_p


def test_channel_weights_combine_signed_z_scores() -> None:
    # SC z = [2, -.5, -.5, -.5, -.5]; HR has the same z but sign -1.
    q = _q("q", skin_conductance=[[4, 0, 0, 0, 0]], heart_rate=[[4, 0, 0, 0, 0]])
    sc_heavy = _run([q], channel_weights={SC: 3.0, HR: 1.0}).questions[0]
    assert sc_heavy.item_scores == pytest.approx([1.0, -0.25, -0.25, -0.25, -0.25], rel=1e-13)
    assert sc_heavy.probe_rank == 1
    hr_heavy = _run([q], channel_weights={SC: 1.0, HR: 3.0}).questions[0]
    assert hr_heavy.item_scores == pytest.approx([-1.0, 0.25, 0.25, 0.25, 0.25], rel=1e-13)
    assert hr_heavy.probe_rank == 5
    # Equal weights cancel to all-zero scores; ties count against the probe.
    tied = _run([q]).questions[0]
    assert tied.item_scores == pytest.approx([0.0] * 5, abs=1e-15)
    assert tied.probe_rank == 5


def test_tie_with_one_irrelevant_ranks_probe_second() -> None:
    res = _run([_q("q", skin_conductance=[[5, 5, 1, 1, 1]])])
    assert res.questions[0].probe_rank == 2


def test_uninformative_channels_are_dropped() -> None:
    q = _q("q", skin_conductance=[[3, 3, 3, 3, 3]], heart_rate=[[-5, 0, 0, 1, 0]])
    (scored,) = _run([q]).questions
    assert scored.channels_used == (HR,)
    assert scored.channels_dropped == ((SC, "zero variance within the question"),)
    no_weight = _run([q], channel_weights={SC: 1.0, HR: 0.0})
    assert no_weight.abstained
    assert [e.reason for e in no_weight.excluded_questions] == ["no channel carried information"]
    # Six equal values have a float standard deviation of about 1.4e-17, not 0:
    # the relative guard must still treat the channel as carrying nothing.
    assert float(np.std([0.1] * 6)) > 0.0
    assert _run([_q("q", heart_rate=[[0.1] * 6])]).abstained


# --------------------------------------------------------------------------- #
# Exact rank test.
# --------------------------------------------------------------------------- #
def _brute_rank_p(rank_sum: int, ks: tuple[int, ...]) -> float:
    vectors = list(itertools.product(*(range(1, k + 1) for k in ks)))
    return sum(1 for v in vectors if sum(v) <= rank_sum) / len(vectors)


def test_rank_sum_p_values_hand_computed() -> None:
    assert exact_rank_sum_p_value(1, [5]) == 0.2
    assert exact_rank_sum_p_value(2, [5, 5]) == 1 / 25
    assert exact_rank_sum_p_value(3, [5, 5]) == 3 / 25
    assert exact_rank_sum_p_value(3, [5, 5, 5]) == 1 / 125
    assert exact_rank_sum_p_value(1, [5, 5]) == 0.0
    assert exact_rank_sum_p_value(10, [5, 5]) == 1.0
    assert rank_sum_distribution([3, 2]) == (1, 2, 2, 1)


@pytest.mark.parametrize("ks", [(3, 4, 5), (2, 2, 2, 2), (6,), (5, 3)])
def test_rank_sum_p_values_match_enumeration(ks: tuple[int, ...]) -> None:
    for t in range(len(ks) - 1, sum(ks) + 2):
        assert exact_rank_sum_p_value(t, ks) == pytest.approx(_brute_rank_p(t, ks), rel=1e-15)


def test_rank_sum_distribution_rejects_degenerate_questions() -> None:
    with pytest.raises(ValueError):
        rank_sum_distribution([])
    with pytest.raises(ValueError):
        rank_sum_distribution([5, 1])


if HAVE_HYPOTHESIS:

    @settings(max_examples=60, deadline=None)
    @given(st.lists(st.integers(min_value=2, max_value=7), min_size=1, max_size=5))
    def test_property_rank_sum_distribution(ks: list[int]) -> None:
        counts = rank_sum_distribution(ks)
        assert sum(counts) == math.prod(ks)
        assert counts == counts[::-1]  # symmetric about the mean
        assert len(counts) == sum(ks) - len(ks) + 1
        assert exact_rank_sum_p_value(sum(ks), ks) == 1.0


def test_two_questions_combined() -> None:
    q1 = _q("q1", skin_conductance=[[10, 0, 0, 0, 0]])
    q2 = _q("q2", skin_conductance=[[1.0, 1.1, 0, 0, 0]])
    res = _run([q1, q2])
    assert [q.probe_rank for q in res.questions] == [1, 2]
    assert res.rank_sum == 3
    assert res.rank_test_p == pytest.approx(3 / 25, rel=1e-15)
    assert (res.lykken_score, res.lykken_max) == (3, 4)
    assert res.mean_probe_rank == 1.5 and res.expected_mean_rank_h0 == 3.0
    # Only relabellings that keep q1's probe and pick item 0 or 1 in q2 reach
    # the observed statistic: 2 of 25.
    assert res.permutation_p == pytest.approx(2 / 25, rel=1e-15)
    d = [q.standardised_difference for q in res.questions]
    assert res.effect_size == pytest.approx((d[0] + d[1]) / 2, rel=1e-15)
    assert d[0] == pytest.approx(2.5, rel=1e-13)


# --------------------------------------------------------------------------- #
# Permutation test against brute force.
# --------------------------------------------------------------------------- #
def _oracle_scores(question: CITQuestion) -> np.ndarray[Any, Any]:
    total = 0.0
    combined = np.zeros((question.n_repetitions, len(question.items)))
    for channel, rows in question.responses.items():
        x = np.asarray(rows, dtype=float)
        combined = combined + CHANNEL_SIGN[channel] * (x - x.mean()) / x.std()
        total += 1.0
    return np.asarray((combined / total).mean(axis=0))


def _brute_permutation_p(questions: list[CITQuestion]) -> float:
    tables = []
    for q in questions:
        s = _oracle_scores(q)
        k = s.size
        tables.append([s[j] - (s.sum() - s[j]) / (k - 1) for j in range(k)])
    observed = sum(t[0] for t in tables)
    combos = list(itertools.product(*tables))
    return sum(1 for c in combos if sum(c) >= observed - 1e-12) / len(combos)


def _random_session(seed: int, ks: tuple[int, ...], reps: int = 2) -> list[CITQuestion]:
    rng = np.random.default_rng(seed)
    out = []
    for n, k in enumerate(ks):
        sc = rng.normal(size=(reps, k))
        sc[:, 0] += 0.8  # a modest probe effect
        hr = rng.normal(size=(reps, k))
        items = tuple(f"i{j}" for j in range(1, k))
        out.append(CITQuestion(f"q{n}", "p", items, {SC: _rows(sc), HR: _rows(hr)}))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_permutation_matches_enumeration(seed: int) -> None:
    session = _random_session(seed, (3, 4, 3))
    res = _run(session)
    assert res.permutation_method == "exact" and res.permutations_evaluated == 36
    assert res.permutation_p == pytest.approx(_brute_permutation_p(session), rel=1e-15)
    for q, oracle in zip(res.questions, session):
        assert q.item_scores == pytest.approx(list(_oracle_scores(oracle)), rel=1e-12)


def test_monte_carlo_permutation_is_seeded_and_never_zero() -> None:
    session = _random_session(7, (5, 5, 5, 5))
    exact = _run(session)
    mc = _run(session, max_exact_permutations=1, monte_carlo_permutations=20_000, seed=11)
    again = _run(session, max_exact_permutations=1, monte_carlo_permutations=20_000, seed=11)
    assert mc.permutation_method == "monte_carlo" and mc.permutations_evaluated == 20_000
    assert mc.permutation_p == again.permutation_p
    assert mc.permutation_p is not None and exact.permutation_p is not None
    count = mc.permutation_p * 20_001
    assert count == pytest.approx(round(count), abs=1e-6) and round(count) >= 1
    se = math.sqrt(exact.permutation_p * (1 - exact.permutation_p) / 20_000)
    assert abs(mc.permutation_p - exact.permutation_p) < 5 * se + 1 / 20_001
    strong = [_q(f"q{i}", skin_conductance=[[50, 0, 0, 0, 0]]) for i in range(8)]
    tiny = _run(strong, max_exact_permutations=1, monte_carlo_permutations=999, seed=0)
    assert tiny.permutation_p == pytest.approx(1 / 1000, rel=1e-12)


def test_exact_enumeration_bound_is_inclusive() -> None:
    session = _random_session(4, (3, 4))
    assert _run(session, max_exact_permutations=12).permutation_method == "exact"
    assert _run(session, max_exact_permutations=11).permutation_method == "monte_carlo"


# --------------------------------------------------------------------------- #
# Finding, alpha and the primary test.
# --------------------------------------------------------------------------- #
def test_finding_is_alpha_controlled_and_inclusive() -> None:
    q = [_q("q1", skin_conductance=[[5, 1, 2, 1, 1]])]  # rank p = 0.2
    assert _run(q, alpha=0.2).finding == FINDING_INDICATED
    assert _run(q, alpha=math.nextafter(0.2, 0.0)).finding == FINDING_NOT_INDICATED
    assert FINDING_INDICATED == "recognition of the probe details indicated"
    assert FINDING_NOT_INDICATED == "recognition of the probe details not indicated"


def test_primary_test_decides_the_finding() -> None:
    session = [
        _q("q1", skin_conductance=[[10, 0, 0, 0, 0]]),
        _q("q2", skin_conductance=[[1.0, 1.1, 0, 0, 0]]),
    ]  # rank p = 0.12, permutation p = 0.08
    rank_first = _run(session, alpha=0.1)
    perm_first = _run(session, alpha=0.1, primary_test="permutation")
    assert rank_first.primary_test == "rank" and rank_first.finding == FINDING_NOT_INDICATED
    assert perm_first.primary_test == "permutation" and perm_first.finding == FINDING_INDICATED


def test_strong_recognition_over_four_questions() -> None:
    session = [_q(f"q{i}", skin_conductance=[[6, 1, 2, 1, 1], [5, 2, 1, 1, 2]]) for i in range(4)]
    res = _run(session)
    assert res.rank_test_p == pytest.approx(1 / 625, rel=1e-15)
    assert res.permutation_p == pytest.approx(1 / 625, rel=1e-15)
    assert res.finding == FINDING_INDICATED


def test_result_text_carries_the_framing_and_no_reserved_labels() -> None:
    res = _run([_q("q1", skin_conductance=[[5, 1, 2, 1, 1]])])
    data = json.loads(json.dumps(res.as_dict()))
    assert data["assumptions"] == list(CIT_ASSUMPTIONS)
    assert "not evidence of guilt" in data["not_evidence_of_guilt"]
    assert data["consent_scope"] == [CIT_CONSENT_SCOPE]
    assert data["subject_pseudonym"] == "subj-01"
    assert "recognition of the probe details" in data["measures"]
    assert not FORBIDDEN.search(json.dumps(data))


def test_inputs_are_not_mutated_and_duplicates_rejected() -> None:
    rows = [[5.0, 1.0, 2.0, 1.0, 1.0]]
    q = _q("q1", skin_conductance=rows)
    before = copy.deepcopy(rows)
    _run([q])
    assert rows == before
    with pytest.raises(ValueError, match="duplicate question_id"):
        _run([q, q])


# --------------------------------------------------------------------------- #
# Question and configuration validation.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build",
    [
        lambda: CITQuestion("q", "p", (), {SC: _rows([[1.0]])}),
        lambda: CITQuestion("q", "p", ("p",), {SC: _rows([[1.0, 2.0]])}),
        lambda: CITQuestion("q", "p", ("a",), {SC: _rows([[1.0, 2.0, 3.0]])}),
        lambda: CITQuestion("q", "p", ("a",), {SC: _rows([[1.0, math.nan]])}),
        lambda: CITQuestion(
            "q", "p", ("a",), {SC: _rows([[1.0, 2.0]]), HR: _rows([[1.0, 2.0], [3.0, 4.0]])}
        ),
        lambda: CITQuestion("q", "p", ("a",), {}),
        lambda: CITQuestion("q", "p", ("a",), {SC: ()}),
        lambda: CITQuestion("q", "p", ("a",), {"pupil": _rows([[1.0, 2.0]])}),  # type: ignore[dict-item]
        lambda: CITConfig(alpha=0.0),
        lambda: CITConfig(primary_test="t"),  # type: ignore[arg-type]
        lambda: CITConfig(channel_weights={SC: -1.0}),
        lambda: CITConfig(channel_weights={SC: 0.0}),
        lambda: CITConfig(max_exact_permutations=0),
        lambda: CITConfig(monte_carlo_permutations=0),
    ],
)
def test_invalid_cit_inputs(build: Any) -> None:
    with pytest.raises(ValueError):
        build()


def test_question_freezes_and_orders_channels() -> None:
    rows = np.array([[1.0, 2.0]])
    # Lists, arrays and channel names are accepted and frozen into the declared types.
    q = CITQuestion("q", "p", ["a"], {"heart_rate": rows, SC: rows})  # type: ignore[arg-type, dict-item]
    rows[0, 0] = 99.0
    assert list(q.responses) == [SC, HR]
    assert q.responses[HR] == ((1.0, 2.0),)
    assert q.items == ("p", "a") and q.irrelevants == ("a",) and q.n_repetitions == 1
    with pytest.raises(TypeError):
        q.responses[SC] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        CITQuestion("q", "p", "a", {SC: _rows([[1.0, 2.0]])})  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Response amplitudes from time series.
# --------------------------------------------------------------------------- #
def test_time_series_copies_and_freezes() -> None:
    t = np.arange(5.0)
    v = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    ts = TimeSeries(t, v)
    t[0] = -100.0
    assert ts.times[0] == 0.0 and t.flags.writeable
    with pytest.raises(ValueError):
        ts.values[0] = 5.0
    for bad in (
        lambda: TimeSeries(np.array([0.0, 0.0]), np.array([1.0, 2.0])),
        lambda: TimeSeries(np.array([0.0, 1.0]), np.array([1.0])),
        lambda: TimeSeries(np.array([0.0, 1.0]), np.array([1.0, math.inf])),
        lambda: TimeSeries(np.array([]), np.array([])),
    ):
        with pytest.raises(ValueError):
            bad()


def test_skin_conductance_amplitude_is_largest_trough_to_peak_rise() -> None:
    t = np.arange(0.0, 8.0, 0.5)
    v = np.full_like(t, 2.0)
    v[t == 1.0] = 2.5  # earlier, higher than the trough
    v[t == 2.0] = 1.9  # trough
    v[t == 3.5] = 2.7  # peak: rise 0.8 from the trough
    v[t == 5.0] = 9.0  # at the window end: excluded (half-open)
    ts = TimeSeries(t, v)
    assert skin_conductance_amplitude(ts, 0.0) == pytest.approx(0.8, rel=1e-12)
    falling = TimeSeries(t, 10.0 - t)
    assert skin_conductance_amplitude(falling, 0.0) == 0.0
    v2 = np.full_like(t, 2.0)
    v2[t == 0.5] = 0.0  # before the window opens at 1 s: excluded
    assert skin_conductance_amplitude(TimeSeries(t, v2), 0.0) == 0.0
    assert skin_conductance_amplitude(ts, 0.0, (1.0, 5.5)) == pytest.approx(7.1, rel=1e-12)
    with pytest.raises(InsufficientDataError):
        skin_conductance_amplitude(TimeSeries(np.array([0.0, 1.0]), np.array([1.0, 2.0])), 0.0)


def test_heart_rate_change_is_post_minus_baseline() -> None:
    t = np.arange(-5.0, 15.0)
    v = np.where(t < 0, 70.0, 64.0)
    v[t == -3] = 76.0  # inside the baseline [-3, 0)
    v[t == -4] = 999.0  # outside it
    v[t == 10] = 999.0  # outside the response window [0, 10)
    ts = TimeSeries(t, v)
    # Baseline mean (76 + 70 + 70) / 3 = 72; response mean 64.
    assert heart_rate_change(ts, 0.0) == pytest.approx(-8.0, rel=1e-13)
    assert heart_rate_change(ts, 0.0, (-2.0, 0.0), (0.0, 5.0)) == pytest.approx(-6.0, rel=1e-13)
    with pytest.raises(InsufficientDataError):
        heart_rate_change(ts, 100.0)


def test_respiration_line_length_is_total_variation() -> None:
    t = np.arange(0.0, 12.0, 2.0)  # 0, 2, ..., 10
    v = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 50.0])
    ts = TimeSeries(t, v)
    assert respiration_line_length(ts, 0.0) == 4.0  # the sample at 10 s is excluded
    assert respiration_line_length(ts, 0.0, (0.0, 10.5)) == 54.0
    with pytest.raises(InsufficientDataError):
        respiration_line_length(ts, 0.0, (0.0, 1.0))


def test_derive_response_amplitudes_per_channel() -> None:
    t = np.arange(-5.0, 30.0)
    hr = TimeSeries(t, np.where((t >= 0) & (t < 10), 64.0, 70.0))
    out = derive_response_amplitudes([0.0, 15.0], heart_rate=hr)
    assert dict(out) == {HR: (-6.0, 0.0)}
    with pytest.raises(ValueError):
        derive_response_amplitudes([0.0])


def _hr_series(probe_onsets: list[float], other_onsets: list[float], end: float) -> TimeSeries:
    t = np.arange(0.0, end)
    v = np.full_like(t, 70.0)
    for onset in probe_onsets:
        v[(t >= onset) & (t < onset + 10)] = 64.0
    for onset in other_onsets:
        v[(t >= onset) & (t < onset + 10)] = 69.0
    return TimeSeries(t, v)


def test_question_from_time_series_assembles_the_matrix() -> None:
    presentations = [
        ("i1", 20.0),
        ("p", 40.0),
        ("i2", 60.0),
        ("p", 80.0),
        ("i2", 100.0),
        ("i1", 120.0),
    ]
    base = _hr_series([40.0, 80.0], [20.0, 60.0, 100.0, 120.0], 140.0)
    values = np.array(base.values)
    values[(base.times >= 80) & (base.times < 90)] = 62.0  # second probe: -8 bpm
    values[(base.times >= 100) & (base.times < 110)] = 68.0  # second i2: -2 bpm
    hr = TimeSeries(base.times, values)
    q = question_from_time_series("q1", "p", ["i1", "i2"], presentations[::-1], heart_rate=hr)
    assert q.items == ("p", "i1", "i2")
    # Repetition r of an item is its r-th presentation in time order.
    assert q.responses[HR] == ((-6.0, -1.0, -1.0), (-8.0, -1.0, -2.0))
    res = _run([q])
    assert res.questions[0].probe_rank == 1
    assert res.rank_test_p == pytest.approx(1 / 3, rel=1e-15)


def test_question_from_time_series_validation() -> None:
    hr = _hr_series([], [], 200.0)
    with pytest.raises(ValueError, match="same number"):
        question_from_time_series(
            "q", "p", ["a"], [("p", 20.0), ("a", 40.0), ("p", 60.0)], heart_rate=hr
        )
    with pytest.raises(ValueError, match="unknown"):
        question_from_time_series("q", "p", ["a"], [("p", 20.0), ("x", 40.0)], heart_rate=hr)
    with pytest.raises(ValueError, match="no presentations"):
        question_from_time_series("q", "p", ["a"], [], heart_rate=hr)
    close = [("p", 20.0), ("a", 25.0)]
    with pytest.raises(ValueError, match="allow_overlap"):
        question_from_time_series("q", "p", ["a"], close, heart_rate=hr)
    ok = question_from_time_series(
        "q", "p", ["a"], close, heart_rate=hr, windows=ResponseWindows(allow_overlap=True)
    )
    assert ok.responses[HR] == ((0.0, 0.0),)
    exactly = [("p", 20.0), ("a", 30.0)]  # 10 s apart equals the 10 s reach: allowed
    assert question_from_time_series("q", "p", ["a"], exactly, heart_rate=hr).n_repetitions == 1


def test_overlap_reach_uses_only_supplied_channels() -> None:
    t = np.arange(0.0, 60.0, 0.5)
    sc = TimeSeries(t, np.full_like(t, 2.0))
    # SCR reach is 5 s, so onsets 6 s apart are fine with skin conductance alone.
    q = question_from_time_series("q", "p", ["a"], [("p", 10.0), ("a", 16.0)], skin_conductance=sc)
    assert q.responses[SC] == ((0.0, 0.0),)
    hr = _hr_series([], [], 60.0)
    with pytest.raises(ValueError, match="allow_overlap"):
        question_from_time_series(
            "q", "p", ["a"], [("p", 10.0), ("a", 16.0)], skin_conductance=sc, heart_rate=hr
        )


def test_response_windows_validation() -> None:
    with pytest.raises(ValueError):
        ResponseWindows(scr=(5.0, 1.0))
    with pytest.raises(ValueError):
        ResponseWindows(hr_baseline=(-3.0, 1.0), hr_response=(0.0, 10.0))
    assert ResponseWindows(hr_baseline=(-3.0, 0.0), hr_response=(0.0, 10.0)).hr_baseline == (
        -3.0,
        0.0,
    )


def test_contactless_heart_rate_session_end_to_end() -> None:
    # Four questions, k = 5, two repetitions each, a 6 bpm deceleration after
    # every probe onset and a 1 bpm one after irrelevant onsets.
    questions = []
    for n in range(4):
        labels = ["p", *IRR]
        order = labels[n:] + labels[:n]
        presentations = [(lab, 20.0 + 20.0 * i) for i, lab in enumerate(order + order[::-1])]
        probes = [t for lab, t in presentations if lab == "p"]
        others = [t for lab, t in presentations if lab != "p"]
        hr = _hr_series(probes, others, 260.0)
        questions.append(question_from_time_series(f"q{n}", "p", IRR, presentations, heart_rate=hr))
    res = _run(questions)
    assert [q.probe_rank for q in res.questions] == [1, 1, 1, 1]
    assert res.rank_test_p == pytest.approx(1 / 625, rel=1e-15)
    assert res.finding == FINDING_INDICATED

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for the counterintelligence screens.

Expected values are computed by hand in the test, or by an independent oracle:
closed-form Beta quantiles, ``math``-only Poisson and binomial tails,
``scipy.stats`` chi-square and FDR control, exact Jaccard, and Monte Carlo
simulation of the Poisson scan statistic. The module this replaces returned a
threat level and casualty figures for empty input. ``TestAssessor`` pins that
empty input now yields no finding at all.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from scipy import stats

from omni_mercury_engine.evaluation.event_coincidence import benjamini_hochberg
from omni_mercury_engine.security.counterintelligence import (
    ADMIRALTY_RELIABILITY_PRIOR_MEAN,
    BENFORD_FIRST_DIGIT_PROBABILITIES,
    DEFAULT_LOG_ODDS_CLIP,
    MAX_OUTCOME_WEIGHT,
    MINHASH_PRIME,
    NIGRINI_NONCONFORMITY,
    AssessmentHistory,
    CounterintelligenceAssessor,
    CounterintelligenceConfig,
    FlagKind,
    GeoFix,
    MinHasher,
    ReportRecord,
    SourceOutcome,
    TrackPoint,
    admiralty_prior,
    analyse_independence,
    benford_first_digit_test,
    benjamini_hochberg_adjusted,
    detect_coordinated_reporting,
    excess_rounding_test,
    find_near_duplicates,
    first_significant_digit,
    fit_outcome_weight_cap,
    impossible_travel,
    jaccard_similarity,
    nigrini_conformity,
    normalize_reports,
    parse_timestamp,
    scan_statistic_pvalue,
    scan_window,
    screen_fabrication,
    source_posterior,
    spectral_signature_scores,
    terminal_digit_test,
    terminal_digits,
    track_points,
    word_shingles,
)
from omni_mercury_engine.utils.geo import EARTH_RADIUS_KM

try:
    from hypothesis import (
        given,
        settings,
        strategies as st,
    )

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - hypothesis is a declared dev dependency
    HAS_HYPOTHESIS = False

T0 = 1_760_000_000.0  # an arbitrary epoch origin for synthetic timestamps
KM_PER_DEGREE = EARTH_RADIUS_KM * math.pi / 180.0


def rec(report_id: str, **fields: Any) -> ReportRecord:
    """Build a normalised record with a default discipline."""
    fields.setdefault("discipline", "open_source")
    return ReportRecord(report_id=report_id, **fields)


def poisson_sf_math(m: int, psi: float) -> float:
    """P(X >= m) for X ~ Poisson(psi), from the series (no scipy)."""
    if m <= 0:
        return 1.0
    return 1.0 - sum(math.exp(-psi) * psi**i / math.factorial(i) for i in range(m))


def binomial_upper_math(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), from exact integer combinatorics."""
    return sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1))


def benford_like_values(counts: list[int]) -> list[float]:
    """Integers with the given first-digit counts and exactly uniform last digits.

    Value ``i`` (first digit ``d``) is ``d * 10**(j + 2) + (i % 100)`` with
    ``j = i % 3``: its first digit is ``d``, its last digit is ``i % 10``, and the
    set spans 100 to 99,999 (about three orders of magnitude).
    """
    values: list[float] = []
    i = 0
    for digit, count in enumerate(counts, start=1):
        for _ in range(count):
            values.append(float(digit * 10 ** (i % 3 + 2) + i % 100))
            i += 1
    return values


# --------------------------------------------------------------------------- #
# 1. Source reliability.
# --------------------------------------------------------------------------- #
class TestAdmiraltyPrior:
    def test_table_is_icd203_band_midpoints(self) -> None:
        bands = {"A": (0.95, 0.99), "B": (0.80, 0.95), "C": (0.55, 0.80)}
        bands |= {"D": (0.20, 0.45), "E": (0.05, 0.20)}
        for grade, (low, high) in bands.items():
            assert ADMIRALTY_RELIABILITY_PRIOR_MEAN[grade] == pytest.approx((low + high) / 2)
        assert ADMIRALTY_RELIABILITY_PRIOR_MEAN["F"] is None

    def test_graded_prior_has_declared_mean_and_strength(self) -> None:
        assert admiralty_prior("A") == pytest.approx((1.94, 0.06))
        assert admiralty_prior(" b ", prior_strength=4.0) == pytest.approx((3.5, 0.5))
        assert admiralty_prior("E", prior_strength=8.0) == pytest.approx((1.0, 7.0))

    @pytest.mark.parametrize("grade", ["F", "f", None])
    def test_cannot_be_judged_is_uniform(self, grade: str | None) -> None:
        assert admiralty_prior(grade, prior_strength=10.0) == (1.0, 1.0)

    def test_unknown_grade_and_bad_strength_raise(self) -> None:
        with pytest.raises(ValueError, match="unknown Admiralty"):
            admiralty_prior("G")
        with pytest.raises(ValueError, match="prior_strength"):
            admiralty_prior("A", prior_strength=0.0)


class TestSourcePosterior:
    def test_uniform_plus_two_confirmations_is_beta_3_1(self) -> None:
        outcomes = [SourceOutcome(True), SourceOutcome(True)]
        post = source_posterior("s", "F", outcomes)
        assert (post.alpha, post.beta) == (3.0, 1.0)
        assert post.mean == 0.75
        # Beta(3, 1) has CDF x**3, so its q-quantile is q**(1/3).
        assert post.credible_interval == pytest.approx((0.025 ** (1 / 3), 0.975 ** (1 / 3)))
        assert post.effective_sample_size == 4.0
        assert post.log_odds_weight == pytest.approx(math.log(3.0))
        assert post.evidence_weight == 2.0
        assert not post.informative_prior

    def test_one_refutation_is_beta_1_2(self) -> None:
        post = source_posterior("s", None, [SourceOutcome(False)])
        # Beta(1, 2) has CDF 1 - (1 - x)**2, so its q-quantile is 1 - sqrt(1 - q).
        expected = (1 - math.sqrt(0.975), 1 - math.sqrt(0.025))
        assert post.credible_interval == pytest.approx(expected)
        assert post.mean == pytest.approx(1 / 3)
        assert post.log_odds_weight == pytest.approx(-math.log(2.0))

    def test_credible_level_is_configurable(self) -> None:
        cfg = CounterintelligenceConfig(credible_level=0.9)
        post = source_posterior("s", "F", [SourceOutcome(True), SourceOutcome(True)], config=cfg)
        assert post.credible_interval == pytest.approx((0.05 ** (1 / 3), 0.95 ** (1 / 3)))
        assert post.credible_level == 0.9

    def test_weighted_outcomes(self) -> None:
        post = source_posterior("s", "F", [SourceOutcome(True, 0.5), SourceOutcome(False, 1.5)])
        assert (post.alpha, post.beta) == (1.5, 2.5)
        assert source_posterior("s", "F", [SourceOutcome(True, 0.0)]).alpha == 1.0

    def test_forgetting_discounts_older_evidence_not_the_prior(self) -> None:
        cfg = CounterintelligenceConfig(forgetting_factor=0.5)
        outcomes = [SourceOutcome(True), SourceOutcome(True), SourceOutcome(False)]
        post = source_posterior("s", "B", outcomes, config=cfg)
        # r: 1 -> 0.5*1 + 1 = 1.5 -> 0.5*1.5 = 0.75; s: 0 -> 0 -> 0 + 1 = 1.
        assert post.confirmed_weight == 0.75
        assert post.refuted_weight == 1.0
        assert (post.alpha, post.beta) == pytest.approx((1.75 + 0.75, 0.25 + 1.0))

    def test_graded_prior_and_informative_counts(self) -> None:
        post = source_posterior("s", "A")
        assert post.log_odds_weight == pytest.approx(math.log(1.94 / 0.06))
        assert post.informative_prior
        graded = source_posterior("s", "b", [SourceOutcome(True)])
        assert graded.grade == "B"
        assert graded.informative_counts == pytest.approx((2.75, 0.25))
        ungraded = source_posterior("s", None, [SourceOutcome(True)])
        assert ungraded.informative_counts == (1.0, 0.0)

    def test_log_odds_weight_is_clipped_both_ways(self) -> None:
        many_true = [SourceOutcome(True)] * 1000
        assert source_posterior("s", "F", many_true).log_odds_weight == DEFAULT_LOG_ODDS_CLIP
        many_false = [SourceOutcome(False)] * 1000
        assert source_posterior("s", "F", many_false).log_odds_weight == -DEFAULT_LOG_ODDS_CLIP
        cfg = CounterintelligenceConfig(log_odds_clip=0.5)
        assert source_posterior("s", "A", config=cfg).log_odds_weight == 0.5
        assert math.log(0.99 / 0.01) == pytest.approx(DEFAULT_LOG_ODDS_CLIP)

    def test_outcome_validation(self) -> None:
        with pytest.raises(TypeError):
            SourceOutcome("yes")  # type: ignore[arg-type]
        for bad in (-1.0, math.nan, math.inf, 1e308, MAX_OUTCOME_WEIGHT + 1e-9):
            with pytest.raises(ValueError, match="weight"):
                SourceOutcome(True, bad)
        assert SourceOutcome(True, MAX_OUTCOME_WEIGHT).weight == MAX_OUTCOME_WEIGHT

    def test_one_huge_adjudication_cannot_set_reliability(self) -> None:
        # Regression: weight 1e308 was accepted, so one record drove the
        # source posterior -- and the discipline's adjusted reliability -- to 1.0.
        history = {
            "outcomes": {
                "A": [{"confirmed": True, "weight": 1e308}],
                "B": [{"confirmed": True, "weight": 1e308}, {"confirmed": True, "weight": 1e308}],
            }
        }
        reports = {
            "signals": {"report_id": "x", "source_id": "A"},
            "human": {"report_id": "y", "source_id": "B"},
        }
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports, history)
        assert result.adjusted_reliability == {"human": 0.85, "signals": 0.9}
        assert sum("weight must be finite" in reason for reason in result.reasons) == 3


class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prior_strength": 0.0},
            {"forgetting_factor": 0.0},
            {"forgetting_factor": 1.01},
            {"credible_level": 0.0},
            {"credible_level": 1.0},
            {"log_odds_clip": 0.0},
            {"discipline_prior_strength": 0.0},
            {"shingle_size": 0},
            {"minhash_permutations": 0},
            {"near_duplicate_threshold": 0.0},
            {"near_duplicate_threshold": 1.01},
            {"coordination_window_hours": 0.0},
            {"coordination_min_sources": 1},
            {"observation_hours": 0.0},
            {"fdr_alpha": 0.0},
            {"fdr_alpha": 1.0},
            {"benford_min_n": 0},
            {"benford_min_orders_of_magnitude": -0.1},
            {"terminal_digit_min_n": 0},
            {"rounding_min_n": 0},
            {"digit_decimals": -1},
            {"terminal_min_units": 0},
            {"fabrication_unit": "batch"},
            {"max_speed_kmh": 0.0},
            {"circular_factor": -0.01},
            {"coordinated_factor": 1.01},
            {"fabrication_factor": 1.5},
            {"impossible_travel_factor": -1.0},
        ],
    )
    def test_out_of_range_parameters_raise(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            CounterintelligenceConfig(**kwargs)

    def test_boundary_values_are_accepted(self) -> None:
        cfg = CounterintelligenceConfig(
            forgetting_factor=1.0,
            near_duplicate_threshold=1.0,
            coordination_min_sources=2,
            benford_min_orders_of_magnitude=0.0,
            digit_decimals=0,
            circular_factor=0.0,
            coordinated_factor=1.0,
            fabrication_factor=0.0,
            impossible_travel_factor=1.0,
            shingle_size=1,
            minhash_permutations=1,
            benford_min_n=1,
            terminal_digit_min_n=1,
            rounding_min_n=1,
            terminal_min_units=1,
        )
        assert cfg.coordination_min_sources == 2

    @pytest.mark.parametrize(
        "table",
        [
            {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.2},
            {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.2, "F": 0.5},
            {"A": 1.0, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.2, "F": None},
            {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.0, "F": None},
            {"A": None, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.2, "F": None},
        ],
    )
    def test_grade_table_validation(self, table: dict[str, float | None]) -> None:
        with pytest.raises(ValueError):
            CounterintelligenceConfig(grade_prior_mean=table)

    def test_custom_grade_table_is_used(self) -> None:
        table = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.4, "E": 0.2, "F": None}
        cfg = CounterintelligenceConfig(grade_prior_mean=table)
        assert source_posterior("s", "D", config=cfg).mean == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# 2a. Shingles and MinHash.
# --------------------------------------------------------------------------- #
class TestShinglesAndMinHash:
    def test_word_shingles(self) -> None:
        assert word_shingles("A b, c d E", 3) == {"a b c", "b c d", "c d e"}
        assert word_shingles("one two", 3) == frozenset()
        assert word_shingles("one two three", 3) == {"one two three"}
        assert word_shingles("x y", 1) == {"x", "y"}
        with pytest.raises(ValueError):
            word_shingles("a b c", 0)

    def test_exact_jaccard(self) -> None:
        a = word_shingles("a b c d e", 3)
        b = word_shingles("a b c d f", 3)
        assert jaccard_similarity(a, b) == 0.5  # {abc, bcd} / {abc, bcd, cde, cdf}
        assert jaccard_similarity([], []) is None
        assert jaccard_similarity(["x"], []) == 0.0

    def test_signature_matches_the_documented_hash_family(self) -> None:
        hasher = MinHasher(8, seed=3)
        shingles = {"alpha beta gamma", "beta gamma delta", "gamma delta eps"}
        rng = np.random.default_rng(3)
        a = [int(v) for v in rng.integers(1, 2**32, size=8, dtype=np.uint64)]
        b = [int(v) for v in rng.integers(0, 2**32, size=8, dtype=np.uint64)]
        import hashlib

        base = [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")
            for s in shingles
        ]
        expected = [min((a[i] * x + b[i]) % MINHASH_PRIME for x in base) for i in range(8)]
        assert hasher.signature(shingles).tolist() == expected
        assert hasher.num_permutations == 8

    def test_identical_and_disjoint_sets(self) -> None:
        hasher = MinHasher(128, seed=1)
        a = {f"a{i}" for i in range(50)}
        b = {f"b{i}" for i in range(50)}
        assert MinHasher.estimate(hasher.signature(a), hasher.signature(set(a))) == 1.0
        # The hash is a bijection on [0, p), so disjoint sets never share a minimum.
        assert MinHasher.estimate(hasher.signature(a), hasher.signature(b)) == 0.0

    def test_seeded_and_deterministic(self) -> None:
        shingles = {f"s{i}" for i in range(40)}
        same = MinHasher(64, seed=9).signature(shingles)
        assert np.array_equal(same, MinHasher(64, seed=9).signature(shingles))
        assert not np.array_equal(same, MinHasher(64, seed=10).signature(shingles))

    def test_estimate_tracks_exact_jaccard(self) -> None:
        hasher = MinHasher(256, seed=0)
        rng = np.random.default_rng(42)
        universe = [f"w{i}" for i in range(400)]
        errors: list[float] = []
        for _ in range(300):
            a = set(rng.choice(universe, size=int(rng.integers(20, 200)), replace=False))
            b = set(rng.choice(universe, size=int(rng.integers(20, 200)), replace=False))
            exact = jaccard_similarity(a, b)
            assert exact is not None
            estimate = MinHasher.estimate(hasher.signature(a), hasher.signature(b))
            standard_error = math.sqrt(max(exact * (1 - exact), 1 / 256) / 256)
            assert abs(estimate - exact) <= 4 * standard_error
            errors.append(estimate - exact)
        # Unbiased within sampling error: 4 standard errors of the mean.
        assert abs(float(np.mean(errors))) <= 4 * float(np.std(errors)) / math.sqrt(300)

    def test_signature_errors(self) -> None:
        hasher = MinHasher(4)
        with pytest.raises(ValueError, match="empty"):
            hasher.signature(set())
        with pytest.raises(ValueError):
            MinHasher.estimate(hasher.signature({"x"}), MinHasher(5).signature({"x"}))
        with pytest.raises(ValueError):
            MinHasher(0)


# --------------------------------------------------------------------------- #
# 2b. Provenance, circularity, shared origins, independence clusters.
# --------------------------------------------------------------------------- #
TEXT = "armoured column of twelve vehicles moving north on route seven at dawn today"


class TestNearDuplicates:
    def test_later_copy_is_directed_derivative(self) -> None:
        records = [
            rec("late", text=TEXT.upper() + "!", reported_at=T0 + 60),
            rec("early", text=TEXT, reported_at=T0),
        ]
        (pair,) = find_near_duplicates(records)
        assert (pair.report_a, pair.report_b, pair.directed) == ("early", "late", True)
        assert pair.jaccard_estimate == 1.0

    def test_equal_or_missing_times_are_undirected(self) -> None:
        for times in ((T0, T0), (None, T0), (None, None)):
            records = [
                rec("b", text=TEXT, reported_at=times[0]),
                rec("a", text=TEXT, reported_at=times[1]),
            ]
            (pair,) = find_near_duplicates(records)
            assert (pair.report_a, pair.report_b, pair.directed) == ("a", "b", False)

    def test_threshold_is_inclusive(self) -> None:
        a = rec("a", text="alpha beta gamma delta epsilon zeta eta theta iota kappa")
        b = rec("b", text="alpha beta gamma delta epsilon zeta lambda mu nu xi")
        estimate = MinHasher.estimate(
            MinHasher(256, 0).signature(word_shingles(a.text or "", 3)),
            MinHasher(256, 0).signature(word_shingles(b.text or "", 3)),
        )
        assert 0.0 < estimate < 1.0
        at = CounterintelligenceConfig(near_duplicate_threshold=estimate)
        above = CounterintelligenceConfig(near_duplicate_threshold=estimate + 1 / 256)
        assert len(find_near_duplicates([a, b], at)) == 1
        assert find_near_duplicates([a, b], above) == ()

    def test_short_or_missing_text_is_not_compared(self) -> None:
        records = [rec("a", text="yes"), rec("b", text="yes"), rec("c"), rec("d", text=TEXT)]
        assert find_near_duplicates(records) == ()


class TestIndependence:
    def test_engine_format_reports_are_independent(self) -> None:
        records = [rec("open_source"), rec("signals", discipline="signals")]
        analysis = analyse_independence(records)
        assert analysis.effective_independent_sources == 2
        assert analysis.distinct_origins == 2
        assert analysis.flags == ()
        assert analysis.origins_of["signals"] == ("report:signals",)

    def test_relay_is_one_cluster_and_a_corroboration_loop(self) -> None:
        records = [
            rec("A", source_id="S1", derived_from=("B",), reported_at=T0 + 10),
            rec("B", source_id="S2", reported_at=T0),
        ]
        analysis = analyse_independence(records)
        (cluster,) = analysis.clusters
        assert cluster.representative == "B"
        assert cluster.report_ids == ("A", "B")
        assert cluster.source_ids == ("S1", "S2")
        assert analysis.origins_of["A"] == ("source:S1", "source:S2")
        assert analysis.distinct_origins == 2
        (flag,) = analysis.flags
        assert flag.kind is FlagKind.CIRCULAR_REPORTING
        assert flag.report_ids == ("A", "B")
        assert flag.evidence["pattern"] == "corroboration_loop"
        assert flag.evidence["path"] == ("A", "B")
        assert flag.evidence["edge_kinds"] == ("citation",)
        assert flag.p_value is None
        assert analysis.edges == (("A", "B", "citation"),)

    def test_two_step_chain_reports_every_loop(self) -> None:
        records = [
            rec("A", derived_from=("B",)),
            rec("B", derived_from=("C",)),
            rec("C"),
        ]
        loops = {
            flag.report_ids: flag.evidence["path"]
            for flag in analyse_independence(records).flags
            if flag.kind is FlagKind.CIRCULAR_REPORTING
        }
        assert loops == {
            ("A", "B"): ("A", "B"),
            ("A", "C"): ("A", "B", "C"),
            ("B", "C"): ("B", "C"),
        }

    def test_citation_cycle(self) -> None:
        records = [
            rec("A", source_id="S1", derived_from=("B",)),
            rec("B", source_id="S2", derived_from=("A",)),
            rec("C", source_id="S3"),
        ]
        analysis = analyse_independence(records)
        assert analysis.cycles == (("A", "B"),)
        (flag,) = analysis.flags
        assert flag.evidence["pattern"] == "citation_cycle"
        assert flag.evidence["cycle"] == ("A", "B", "A")
        assert flag.report_ids == ("A", "B")
        assert "cycle:A" in analysis.origins_of["B"]
        assert analysis.effective_independent_sources == 2

    def test_three_cycle_witness_and_self_citation(self) -> None:
        records = [
            rec("A", derived_from=("B",)),
            rec("B", derived_from=("C",)),
            rec("C", derived_from=("A",)),
        ]
        (flag,) = analyse_independence(records).flags
        assert flag.evidence["cycle"] == ("A", "B", "C", "A")
        selfie = analyse_independence([rec("Z", derived_from=("Z",))])
        assert selfie.cycles == (("Z",),)
        assert selfie.flags[0].evidence["cycle"] == ("Z", "Z")

    def test_cycle_with_external_exit_is_still_a_cycle(self) -> None:
        records = [rec("A", derived_from=("B", "X")), rec("B", derived_from=("A",))]
        analysis = analyse_independence(records)
        assert analysis.cycles == (("A", "B"),)
        assert analysis.origins_of["A"] == ("external:X",)

    def test_shared_external_origin(self) -> None:
        records = [
            rec("A", source_id="S1", derived_from=("X",), reported_at=T0 + 100),
            rec("B", source_id="S2", derived_from=("X",), reported_at=T0 + 50),
            rec("C", source_id="S3", reported_at=T0),
        ]
        analysis = analyse_independence(records)
        assert analysis.effective_independent_sources == 2
        assert analysis.cluster_of["A"] == analysis.cluster_of["B"] == "cluster:B"
        (flag,) = analysis.flags
        assert flag.kind is FlagKind.SHARED_ORIGIN
        assert flag.report_ids == ("A", "B")
        assert flag.evidence["origin"] == "external:X"
        assert flag.evidence["source_keys"] == ("S1", "S2")

    def test_relays_of_one_source_share_its_origin(self) -> None:
        records = [
            rec("A", source_id="S1", derived_from=("B",)),
            rec("C", source_id="S3", derived_from=("D",)),
            rec("B", source_id="S2"),
            rec("D", source_id="S2"),
        ]
        shared = [
            f for f in analyse_independence(records).flags if f.kind is FlagKind.SHARED_ORIGIN
        ]
        assert [(f.report_ids, f.evidence["origin"]) for f in shared] == [(("A", "C"), "source:S2")]

    def test_same_source_is_one_cluster_without_shared_origin_flag(self) -> None:
        records = [
            rec("A", source_id="S1", derived_from=("X",)),
            rec("B", source_id="S1", derived_from=("X",)),
        ]
        analysis = analyse_independence(records)
        assert analysis.effective_independent_sources == 1
        assert analysis.flags == ()
        two_originals = analyse_independence([rec("A", source_id="S"), rec("B", source_id="S")])
        assert two_originals.effective_independent_sources == 1

    def test_claim_groups_separate_corroboration(self) -> None:
        records = [rec("A", claim_id="c1", derived_from=("B",)), rec("B", claim_id="c2")]
        analysis = analyse_independence(records)
        assert analysis.flags == ()
        assert analysis.effective_independent_sources == 1

    def test_near_duplicate_feeds_the_graph(self) -> None:
        records = [
            rec("copy", source_id="S2", text=TEXT, reported_at=T0 + 600),
            rec("orig", source_id="S1", text=TEXT, reported_at=T0),
        ]
        analysis = analyse_independence(records)
        assert analysis.edges == (("copy", "orig", "near_duplicate"),)
        (flag,) = analysis.flags
        assert flag.evidence["edge_kinds"] == ("near_duplicate",)
        assert analysis.clusters[0].representative == "orig"

    def test_undirected_near_duplicates_merge_without_edges(self) -> None:
        records = [
            rec("b", source_id="S2", text=TEXT, reported_at=T0),
            rec("a", source_id="S1", text=TEXT, reported_at=T0),
        ]
        analysis = analyse_independence(records)
        assert analysis.edges == ()
        assert analysis.flags == ()
        assert [c.report_ids for c in analysis.clusters] == [("a", "b")]
        assert analysis.clusters[0].representative == "a"

    def test_representative_prefers_origin_then_time_then_id(self) -> None:
        records = [
            rec("r1", source_id="S", reported_at=T0 + 5, derived_from=("X",)),
            rec("r2", source_id="S", reported_at=T0 + 9),
            rec("r3", source_id="S", reported_at=T0 + 1, derived_from=("X",)),
        ]
        assert analyse_independence(records).clusters[0].representative == "r2"
        relays = [
            rec("r1", source_id="S", reported_at=T0 + 5, derived_from=("X",)),
            rec("r3", source_id="S", reported_at=T0 + 1, derived_from=("X",)),
            rec("r0", source_id="S", derived_from=("X",)),
        ]
        assert analyse_independence(relays).clusters[0].representative == "r3"
        untimed = [rec("q", source_id="S"), rec("p", source_id="S")]
        assert analyse_independence(untimed).clusters[0].representative == "p"


# --------------------------------------------------------------------------- #
# 3. Coordinated reporting.
# --------------------------------------------------------------------------- #
class TestScanStatistic:
    def test_scan_window_counts_distinct_sources(self) -> None:
        events = [(0.0, "r1", "a"), (100.0, "r2", "b"), (200.0, "r3", "a"), (5000.0, "r4", "c")]
        short = scan_window(events, 300.0)
        assert (short.count, short.start, short.end) == (2, 0.0, 300.0)
        assert short.report_ids == ("r1", "r2", "r3")
        assert short.source_keys == ("a", "b")
        assert scan_window(events, 5000.0).count == 3
        assert scan_window([], 10.0).count == 0

    def test_scan_window_is_closed_and_ties_go_earliest(self) -> None:
        assert scan_window([(0.0, "a", "a"), (60.0, "b", "b")], 60.0).count == 2
        assert scan_window([(0.0, "a", "a"), (60.000001, "b", "b")], 60.0).count == 1
        tie = scan_window([(0.0, "a", "a"), (500.0, "b", "b"), (510.0, "c", "c")], 5.0)
        assert (tie.count, tie.start) == (1, 0.0)
        with pytest.raises(ValueError):
            scan_window([(0.0, "a", "a")], 0.0)

    def test_pvalue_equals_the_mecke_expectation(self) -> None:
        # E[M] = lambda T P(X >= k-1) - (k-1) P(X >= k), X ~ Poisson(lambda w).
        for k, rate, window, period in [
            (4, 0.5, 1.0, 24.0),
            (6, 0.1, 1.0, 24.0),
            (3, 2.0, 0.25, 5.0),
        ]:
            psi = rate * window
            expected = rate * period * poisson_sf_math(k - 1, psi) - (k - 1) * poisson_sf_math(
                k, psi
            )
            assert scan_statistic_pvalue(k, rate, window, period) == pytest.approx(
                min(1.0, expected), rel=1e-9
            )

    def test_pvalue_edge_cases(self) -> None:
        assert scan_statistic_pvalue(0, 1.0, 1.0, 10.0) == 1.0
        assert scan_statistic_pvalue(1, 0.05, 1.0, 10.0) == pytest.approx(0.5)  # lambda * T
        assert scan_statistic_pvalue(2, 100.0, 1.0, 10.0) == 1.0  # clipped
        assert scan_statistic_pvalue(4, 0.5, 2.0, 1.0) == scan_statistic_pvalue(4, 0.5, 2.0, 2.0)
        for args in [
            (-1, 1.0, 1.0, 1.0),
            (1, 0.0, 1.0, 1.0),
            (1, 1.0, 0.0, 1.0),
            (1, 1.0, 1.0, 0.0),
        ]:
            with pytest.raises(ValueError):
                scan_statistic_pvalue(*args)

    def test_bound_is_conservative_and_expectation_exact_by_simulation(self) -> None:
        rng = np.random.default_rng(2026)
        rate, window, period, k = 1.0, 1.0, 10.0, 4
        reps = 20000
        hits = 0
        openers = np.empty(reps)
        for r in range(reps):
            times = np.sort(rng.uniform(0.0, period, rng.poisson(rate * period)))
            ends = np.searchsorted(times, times + window, side="right")
            counts = ends - np.arange(times.size)
            hits += bool(times.size) and int(counts.max()) >= k
            openers[r] = np.count_nonzero(counts >= k)
        bound = scan_statistic_pvalue(k, rate, window, period)
        assert hits / reps < bound
        standard_error = float(openers.std()) / math.sqrt(reps)
        assert abs(float(openers.mean()) - bound) < 4 * standard_error


class TestBenjaminiHochberg:
    def test_hand_example(self) -> None:
        q = benjamini_hochberg_adjusted([0.01, 0.04, 0.03, 0.2])
        assert q == pytest.approx((0.04, 0.16 / 3, 0.16 / 3, 0.2))
        assert benjamini_hochberg_adjusted([]) == ()
        assert benjamini_hochberg_adjusted([0.9, 0.95]) == pytest.approx((0.95, 0.95))
        assert benjamini_hochberg_adjusted([0.6]) == (0.6,)

    def test_matches_scipy_and_the_repository_decision(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(50):
            p = np.concatenate([rng.uniform(0, 1, 15), rng.uniform(0, 0.01, 5)])
            q = benjamini_hochberg_adjusted(p.tolist())
            assert q == pytest.approx(tuple(stats.false_discovery_control(p)))
            assert [v <= 0.05 for v in q] == benjamini_hochberg(p.tolist(), 0.05)


def burst(claim: str, n_sources: int, spacing_s: float, start: float = T0) -> list[ReportRecord]:
    """``n_sources`` reports of ``claim`` from distinct sources, ``spacing_s`` apart."""
    return [
        rec(f"{claim}-{i}", claim_id=claim, source_id=f"src{i}", reported_at=start + i * spacing_s)
        for i in range(n_sources)
    ]


class TestCoordinatedReporting:
    CFG = CounterintelligenceConfig(observation_hours=24.0)

    def test_burst_against_low_baseline_is_flagged(self) -> None:
        records = burst("c1", 6, 120.0) + burst("c2", 2, 7200.0)
        tests = detect_coordinated_reporting(records, (), {"c1": 0.1, "c2": 0.1}, None, self.CFG)
        by_claim = {t.claim: t for t in tests}
        c1, c2 = by_claim["c1"], by_claim["c2"]
        assert c1.window is not None
        assert c1.window.count == 6
        assert c1.p_value == pytest.approx(
            0.1 * 24 * poisson_sf_math(5, 0.1) - 5 * poisson_sf_math(6, 0.1), rel=1e-9
        )
        assert c1.flagged
        assert not c2.flagged
        assert (c1.q_value, c2.q_value) == pytest.approx(
            benjamini_hochberg_adjusted([c1.p_value or 0.0, c2.p_value or 0.0])
        )

    def test_high_baseline_is_not_flagged(self) -> None:
        (test,) = detect_coordinated_reporting(
            burst("c1", 6, 120.0), (), {"c1": 50.0}, None, self.CFG
        )
        assert test.applicable
        assert test.p_value == 1.0
        assert not test.flagged

    def test_min_sources_guard(self) -> None:
        records = burst("c1", 2, 1.0)
        rates = {"c1": 1e-9}
        (test,) = detect_coordinated_reporting(records, (), rates, None, self.CFG)
        assert test.q_value is not None
        assert test.q_value < 1e-6
        assert not test.flagged
        relaxed = CounterintelligenceConfig(observation_hours=24.0, coordination_min_sources=2)
        (test,) = detect_coordinated_reporting(records, (), rates, None, relaxed)
        assert test.flagged

    def test_one_source_repeating_is_not_a_burst(self) -> None:
        records = [
            rec(f"r{i}", claim_id="c", source_id="same", reported_at=T0 + i) for i in range(8)
        ]
        (test,) = detect_coordinated_reporting(records, (), {"c": 0.01}, None, self.CFG)
        assert test.window is not None
        assert test.window.count == 1
        assert not test.flagged

    def test_abstains_without_baseline_or_times(self) -> None:
        untimed = [rec("u", claim_id="c3")]
        tests = detect_coordinated_reporting(burst("c1", 3, 1.0) + untimed, (), {"c3": 1.0})
        by_claim = {t.claim: t for t in tests}
        assert not by_claim["c1"].applicable
        assert by_claim["c1"].reason == "no baseline rate for this claim"
        assert by_claim["c3"].reason == "no report carries a usable reported_at"
        assert by_claim["c1"].p_value is None

    def test_default_baseline_and_content_claims(self) -> None:
        records = [
            rec(f"t{i}", source_id=f"s{i}", text=TEXT, reported_at=T0 + 30 * i) for i in range(5)
        ]
        near = find_near_duplicates(records)
        (test,) = detect_coordinated_reporting(records, near, {}, 0.05, self.CFG)
        assert test.claim == "content:t0"
        assert test.flagged
        assert len(test.near_duplicate_pairs) == 10

    def test_observation_period_defaults_to_batch_span(self) -> None:
        records = burst("c1", 4, 60.0) + [rec("late", claim_id="other", reported_at=T0 + 36_000)]
        tests = detect_coordinated_reporting(records, (), {"c1": 0.2})
        test = next(t for t in tests if t.claim == "c1")
        assert test.observation_hours == pytest.approx(10.0)
        assert test.p_value == pytest.approx(scan_statistic_pvalue(4, 0.2, 1.0, 10.0))
        short = detect_coordinated_reporting(burst("c1", 4, 60.0), (), {"c1": 0.2})
        assert short[0].observation_hours == 1.0  # span of 3 minutes, raised to one window


# --------------------------------------------------------------------------- #
# 4. Fabrication screens.
# --------------------------------------------------------------------------- #
BENFORD_COUNTS_1000 = [301, 176, 125, 97, 79, 67, 58, 51, 46]


class TestFirstDigitAndConformity:
    def test_benford_probabilities(self) -> None:
        assert BENFORD_FIRST_DIGIT_PROBABILITIES[0] == pytest.approx(math.log10(2))
        assert BENFORD_FIRST_DIGIT_PROBABILITIES[8] == pytest.approx(math.log10(10 / 9))
        assert sum(BENFORD_FIRST_DIGIT_PROBABILITIES) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("value", "digit"),
        [(0.3, 3), (1000.0, 1), (9.999999, 9), (1e-5, 1), (0.000123, 1), (5e-324, 5), (7, 7)],
    )
    def test_first_significant_digit(self, value: float, digit: int) -> None:
        assert first_significant_digit(value) == digit

    @pytest.mark.parametrize("value", [0.0, -3.0, math.inf, math.nan])
    def test_first_digit_rejects_non_positive(self, value: float) -> None:
        with pytest.raises(ValueError):
            first_significant_digit(value)

    @pytest.mark.parametrize(
        ("mad", "band"),
        [
            (0.0, "close conformity"),
            (0.006, "close conformity"),
            (0.0060001, "acceptable conformity"),
            (0.012, "acceptable conformity"),
            (0.015, "marginally acceptable conformity"),
            (0.0150001, NIGRINI_NONCONFORMITY),
        ],
    )
    def test_nigrini_bands(self, mad: float, band: str) -> None:
        assert nigrini_conformity(mad) == band


class TestBenford:
    def test_statistics_match_hand_computation(self) -> None:
        result = benford_first_digit_test(benford_like_values(BENFORD_COUNTS_1000))
        assert result.applicable
        assert result.observed == tuple(BENFORD_COUNTS_1000)
        probs = [math.log10(1 + 1 / d) for d in range(1, 10)]
        expected = [1000 * p for p in probs]
        chi2 = sum((o - e) ** 2 / e for o, e in zip(BENFORD_COUNTS_1000, expected))
        assert result.statistic == pytest.approx(chi2)
        oracle = stats.chisquare(BENFORD_COUNTS_1000, expected)
        assert result.p_value == pytest.approx(oracle.pvalue)
        mad = sum(abs(o / 1000 - p) for o, p in zip(BENFORD_COUNTS_1000, probs)) / 9
        assert result.mad == pytest.approx(mad)
        assert result.conformity == "close conformity"
        assert result.expected == pytest.approx(tuple(expected))

    def test_degrees_of_freedom_are_eight(self) -> None:
        counts = [340, 170, 115, 90, 75, 65, 55, 50, 40]
        result = benford_first_digit_test(benford_like_values(counts))
        expected = [1000 * math.log10(1 + 1 / d) for d in range(1, 10)]
        chi2 = sum((o - e) ** 2 / e for o, e in zip(counts, expected))
        assert result.statistic == pytest.approx(chi2)
        assert result.p_value == pytest.approx(float(stats.chi2.sf(chi2, 8)), rel=1e-9)
        assert result.p_value == pytest.approx(0.4612, abs=1e-4)

    def test_uniform_digits_are_nonconforming(self) -> None:
        result = benford_first_digit_test(benford_like_values([111] * 9))
        mad = sum(abs(1 / 9 - math.log10(1 + 1 / d)) for d in range(1, 10)) / 9
        assert result.mad == pytest.approx(mad)
        assert result.conformity == NIGRINI_NONCONFORMITY
        assert result.p_value is not None
        assert result.p_value < 1e-50

    def test_minimum_count_boundary(self) -> None:
        values = [float(10 ** (i % 3) * (1 + i % 9)) for i in range(110)]
        assert benford_first_digit_test(values).applicable
        short = benford_first_digit_test(values[:109])
        assert not short.applicable
        assert short.reason == "109 positive values; at least 110 are needed"
        assert benford_first_digit_test(values[:20], min_n=20).applicable

    def test_orders_of_magnitude_boundary(self) -> None:
        exactly_two = [1.0, 100.0] * 60
        assert benford_first_digit_test(exactly_two).applicable
        narrow = benford_first_digit_test([1.0, 99.0] * 60)
        assert not narrow.applicable
        assert narrow.reason == "values span 1.996 orders of magnitude; at least 2 are needed"
        assert not benford_first_digit_test([5.0, 6.0] * 60).applicable

    def test_non_positive_values_are_excluded(self) -> None:
        values = benford_like_values(BENFORD_COUNTS_1000) + [0.0] * 5 + [-4.0] * 3 + [math.nan]
        result = benford_first_digit_test(values)
        assert (result.n, result.excluded) == (1000, 9)
        assert result.observed == tuple(BENFORD_COUNTS_1000)


class TestTerminalDigitsAndRounding:
    def test_uniform_terminal_digits(self) -> None:
        result = terminal_digit_test(list(range(10, 110)))
        assert result.observed == (10,) * 10
        assert (result.statistic, result.p_value) == (0.0, 1.0)

    def test_constant_terminal_digit(self) -> None:
        values = [float(10 * i + 7) for i in range(1, 51)]
        result = terminal_digit_test(values)
        # (50 - 5)**2 / 5 + 9 * (0 - 5)**2 / 5 = 405 + 45.
        assert result.statistic == pytest.approx(450.0)
        assert result.p_value == pytest.approx(stats.chisquare([0] * 7 + [50, 0, 0]).pvalue)

    def test_chi_square_matches_scipy(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.integers(10, 10_000, size=500).astype(float).tolist()
        result = terminal_digit_test(values)
        digits = [int(v) % 10 for v in values]
        observed = np.bincount(digits, minlength=10)
        assert result.observed == tuple(observed.tolist())
        assert result.p_value == pytest.approx(stats.chisquare(observed).pvalue)

    def test_precision_and_exclusions(self) -> None:
        refused = terminal_digit_test([1.5, 22.25] * 40)
        assert not refused.applicable
        assert refused.reason.startswith("values are not all recorded at 0 decimal")
        tenths = [round(1.0 + 0.1 * i, 1) for i in range(100)]
        assert terminal_digit_test(tenths, decimals=1).observed == (10,) * 10
        digits, excluded, refusal = terminal_digits([3.0, 9.0, 10.0, -27.0], min_units=10)
        assert (digits.tolist(), excluded, refusal) == ([0, 7], 2, None)
        too_big = terminal_digits([2.0**53])
        assert too_big[2] == "values too large for an exact float64 terminal digit"
        assert terminal_digits([2.0**53 - 1])[2] is None

    def test_minimum_count_boundary(self) -> None:
        assert terminal_digit_test(list(range(10, 60))).applicable
        short = terminal_digit_test(list(range(10, 59)))
        assert (short.applicable, short.reason) == (
            False,
            "49 usable values; at least 50 are needed",
        )

    def test_excess_rounding_exact_binomial(self) -> None:
        all_rounded = excess_rounding_test([float(10 * i) for i in range(1, 31)])
        assert all_rounded.statistic == 1.0
        assert all_rounded.p_value == pytest.approx(0.2**30)
        others = [float(v) for v in range(11, 40) if v % 5][:20]
        values = [10.0, 15.0, 20.0, 25.0, 30.0, *others]
        result = excess_rounding_test(values)
        assert (result.n, result.observed, result.statistic) == (25, (5, 20), 0.2)
        assert result.expected == pytest.approx((5.0, 20.0))
        assert result.p_value == pytest.approx(binomial_upper_math(5, 25, 0.2))

    def test_rounding_minimum_count(self) -> None:
        assert excess_rounding_test([float(10 * i) for i in range(1, 26)]).applicable
        assert not excess_rounding_test([float(10 * i) for i in range(1, 25)]).applicable

    def test_source_pooling(self) -> None:
        values = tuple(float(v) for v in range(10, 70))
        records = [
            rec("a", source_id="S", numeric_values=values),
            rec("b", source_id="S", numeric_values=values),
        ]
        (pooled,) = screen_fabrication(
            records, CounterintelligenceConfig(fabrication_unit="source")
        )
        assert (pooled.unit, pooled.report_ids, pooled.results[1].n) == (
            "source:S",
            ("a", "b"),
            120,
        )
        per_report = screen_fabrication(records)
        assert [s.unit for s in per_report] == ["report:a", "report:b"]


# --------------------------------------------------------------------------- #
# 5. Impossible travel.
# --------------------------------------------------------------------------- #
class TestImpossibleTravel:
    def test_speed_and_radii(self) -> None:
        distance = 10 * KM_PER_DEGREE
        fast = [
            TrackPoint("a", "E", T0, 0.0, 0.0),
            TrackPoint("b", "E", T0 + 1800, 0.0, 10.0),
        ]
        (violation,) = impossible_travel(fast)
        assert violation.distance_km == pytest.approx(distance)
        assert violation.implied_speed_kmh == pytest.approx(distance / 0.5)
        assert violation.elapsed_hours == 0.5
        slow = [TrackPoint("a", "E", T0, 0.0, 0.0), TrackPoint("b", "E", T0 + 3600, 0.0, 10.0)]
        assert impossible_travel(slow) == ()
        wide = [
            TrackPoint("a", "E", T0, 0.0, 0.0, 300.0),
            TrackPoint("b", "E", T0 + 1800, 0.0, 10.0, 300.0),
        ]
        assert impossible_travel(wide) == ()
        narrow = [
            TrackPoint("a", "E", T0, 0.0, 0.0, 100.0),
            TrackPoint("b", "E", T0 + 1800, 0.0, 10.0, 100.0),
        ]
        (violation,) = impossible_travel(narrow)
        assert violation.effective_distance_km == pytest.approx(distance - 200.0)
        assert violation.implied_speed_kmh == pytest.approx((distance - 200.0) / 0.5)

    def test_bound_is_strict(self) -> None:
        points = [TrackPoint("a", "E", T0, 0.0, 0.0), TrackPoint("b", "E", T0 + 1800, 0.0, 10.0)]
        speed = impossible_travel(points)[0].implied_speed_kmh
        assert impossible_travel(points, max_speed_kmh=speed) == ()
        assert len(impossible_travel(points, max_speed_kmh=speed * (1 - 1e-12))) == 1
        with pytest.raises(ValueError):
            impossible_travel(points, max_speed_kmh=0.0)

    def test_simultaneous_fixes(self) -> None:
        apart = [TrackPoint("a", "E", T0, 0.0, 0.0), TrackPoint("b", "E", T0, 0.0, 1.0)]
        (violation,) = impossible_travel(apart)
        assert violation.implied_speed_kmh == math.inf
        overlapping = [
            TrackPoint("a", "E", T0, 0.0, 0.0, 60.0),
            TrackPoint("b", "E", T0, 0.0, 1.0, 60.0),
        ]
        assert impossible_travel(overlapping) == ()

    def test_fixes_are_ordered_by_time_not_id(self) -> None:
        legal = [
            TrackPoint("a-last", "E", T0 + 7200, 0.0, 0.0),
            TrackPoint("m-middle", "E", T0 + 3600, 0.0, 10.0),
            TrackPoint("z-first", "E", T0, 0.0, 10.0),
        ]
        # In time order the only move is 10 degrees (1,112 km) in 1 h: legal. In id
        # order a-last would precede m-middle, a negative interval.
        assert impossible_travel(legal) == ()
        illegal = [
            TrackPoint("z-first", "E", T0, 0.0, 0.0),
            TrackPoint("a-second", "E", T0 + 1800, 0.0, 10.0),
            TrackPoint("m-third", "E", T0 + 7200, 0.0, 10.0),
        ]
        (violation,) = impossible_travel(illegal)
        assert (violation.report_a, violation.report_b) == ("z-first", "a-second")
        assert violation.elapsed_hours == 0.5

    def test_order_and_entities(self) -> None:
        points = [
            TrackPoint("late", "E", T0 + 1800, 0.0, 10.0),
            TrackPoint("other", "F", T0 + 60, 0.0, 10.0),
            TrackPoint("early", "E", T0, 0.0, 0.0),
            TrackPoint("first-f", "F", T0, 0.0, 0.0),
        ]
        violations = impossible_travel(points)
        assert [(v.entity, v.report_a, v.report_b) for v in violations] == [
            ("E", "early", "late"),
            ("F", "first-f", "other"),
        ]

    def test_track_points_from_records(self) -> None:
        records = [
            rec("a", entity_id="vessel-1", source_id="S", reported_at=T0, location=GeoFix(1, 2, 3)),
            rec("b", source_id="S", reported_at=T0, location=GeoFix(1, 2)),
            rec("c", source_id="S", location=GeoFix(1, 2)),
            rec("d", reported_at=T0, location=GeoFix(1, 2)),
            rec("e", source_id="S", reported_at=T0),
        ]
        assert track_points(records) == (
            TrackPoint("a", "vessel-1", T0, 1.0, 2.0, 3.0),
            TrackPoint("b", "S", T0, 1.0, 2.0, 0.0),
        )


# --------------------------------------------------------------------------- #
# 6. Spectral signatures.
# --------------------------------------------------------------------------- #
def planted_backdoor(shift: float, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1000 N(0, I_32) representations, 5% shifted by ``shift`` along one direction."""
    rng = np.random.default_rng(seed)
    n, d = 1000, 32
    features = rng.standard_normal((n, d))
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    poisoned = np.zeros(n, dtype=bool)
    poisoned[rng.choice(n, 50, replace=False)] = True
    features[poisoned] += shift * direction
    return features, np.zeros(n, dtype=np.int64), poisoned


class TestSpectralSignatures:
    def test_hand_example(self) -> None:
        points = np.array([[-3.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
        result = spectral_signature_scores(points, [0, 0, 0, 0], epsilon=0.25)
        assert result.scores.tolist() == pytest.approx([9.0, 1.0, 1.0, 9.0])
        assert result.flagged.tolist() == [True, False, False, True]  # ceil(1.5 * 0.25 * 4) = 2
        assert result.removal_fraction == 0.375
        (summary,) = result.per_class
        assert (summary.label, summary.n, summary.n_flagged) == (0, 4, 2)
        assert summary.top_singular_value == pytest.approx(math.sqrt(20.0))

    def test_ties_break_by_index(self) -> None:
        points = np.array([[-3.0, 1.0], [-1.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
        result = spectral_signature_scores(points, [0, 0, 0, 0], epsilon=0.1)
        assert result.flagged.tolist() == [True, False, False, False]

    def test_planted_backdoor_recall_and_false_positive_rate(self) -> None:
        features, labels, poisoned = planted_backdoor(shift=8.0)
        result = spectral_signature_scores(features, labels, epsilon=0.05)
        assert int(result.flagged.sum()) == 75  # ceil(1.5 * 0.05 * 1000)
        recall = np.count_nonzero(result.flagged & poisoned) / 50
        false_positive_rate = np.count_nonzero(result.flagged & ~poisoned) / 950
        assert recall == 1.0
        assert false_positive_rate == pytest.approx(25 / 950)
        assert result.scores[poisoned].min() > result.scores[~poisoned].max()

    def test_weak_backdoor_is_not_reliably_found(self) -> None:
        features, labels, poisoned = planted_backdoor(shift=2.0)
        result = spectral_signature_scores(features, labels, epsilon=0.05)
        recall = np.count_nonzero(result.flagged & poisoned) / 50
        assert recall < 0.5  # measured 0.08: near the 0.075 of flagging at random

    def test_clean_class_still_loses_its_quota(self) -> None:
        features, _, poisoned = planted_backdoor(shift=8.0)
        labels = np.where(np.arange(1000) < 500, 0, 1)
        labels[poisoned] = 1
        result = spectral_signature_scores(features, labels, epsilon=0.05)
        flagged_clean_class = np.count_nonzero(result.flagged & (labels == 0))
        n_class0 = int(np.count_nonzero(labels == 0))
        assert flagged_clean_class == math.ceil(0.075 * n_class0 - 1e-9)

    def test_rounding_of_the_quota(self) -> None:
        rng = np.random.default_rng(0)
        result = spectral_signature_scores(rng.standard_normal((200, 4)), [0] * 200, epsilon=0.05)
        assert result.per_class[0].n_flagged == 15  # 1.5 * 0.05 * 200 is 15.000000000000002

    def test_degenerate_classes_abstain(self) -> None:
        features = np.array([[1.0, 2.0], [5.0, 5.0], [5.0, 5.0], [0.1, 0.2], [0.1, 0.2]])
        result = spectral_signature_scores(features, ["a", "b", "b", "c", "c"], epsilon=0.2)
        reasons = {c.label: (c.applicable, c.reason) for c in result.per_class}
        assert reasons == {
            "a": (False, "fewer than 2"),
            "b": (False, "no variance"),
            "c": (False, "no variance"),
        }
        assert not result.flagged.any()

    def test_validation_and_purity(self) -> None:
        features = np.arange(12.0).reshape(6, 2)
        before = features.copy()
        result = spectral_signature_scores(features, [0] * 6, epsilon=0.1)
        assert np.array_equal(features, before)
        assert not result.scores.flags.writeable
        assert not result.flagged.flags.writeable
        assert spectral_signature_scores(features, [0] * 6, epsilon=0.9).removal_fraction == 1.0
        bad_inputs: list[tuple[Any, Any, float, float]] = [
            (features, [0] * 5, 0.1, 1.5),
            (features[0], [0], 0.1, 1.5),
            (features, [[0]] * 6, 0.1, 1.5),
            (np.full((2, 2), np.nan), [0, 0], 0.1, 1.5),
            (features, [0] * 6, 0.0, 1.5),
            (features, [0] * 6, 1.0, 1.5),
            (features, [0] * 6, 0.1, 0.0),
        ]
        for x, y, eps, factor in bad_inputs:
            with pytest.raises(ValueError):
                spectral_signature_scores(x, y, epsilon=eps, removal_factor=factor)


# --------------------------------------------------------------------------- #
# Normalisation.
# --------------------------------------------------------------------------- #
class TestNormalisation:
    def test_engine_format_and_lists(self) -> None:
        records, notes = normalize_reports(
            {"signals": {"threat_score": 0.4}, "open_source": [{"text": "a"}, {"text": "b"}]}
        )
        assert [r.report_id for r in records] == ["open_source[0]", "open_source[1]", "signals"]
        assert notes == ()

    def test_field_parsing_and_notes(self) -> None:
        raw = {
            "report_id": 17,
            "source_id": "S",
            "reported_at": "2026-01-01T00:00:00Z",
            "derived_from": ["x", 3, None],
            "citations": ("y", "x"),
            "information_credibility": 2.0,
            "source_grade": "b",
            "location": {"lat": 10, "lon": -20, "radius_km": 5},
            "numeric_values": [1, 2.5, math.nan, "x", True],
        }
        (record,), notes = normalize_reports({"human": raw})
        assert record.report_id == "17"
        assert record.reported_at == 1767225600.0
        assert record.derived_from == ("x", "3", "y")
        assert record.information_credibility == 2
        assert record.source_grade == "B"
        assert record.location == GeoFix(10.0, -20.0, 5.0)
        assert record.numeric_values == (1.0, 2.5)
        assert notes == (
            "report 17: derived_from/citations ignored (1 non-id entries)",
            "report 17: numeric_values ignored (3 non-finite or non-numeric entries dropped)",
        )

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("reported_at", "2026-01-01T00:00:00"),
            ("reported_at", True),
            ("reported_at", "yesterday"),
            ("information_credibility", 7),
            ("information_credibility", 1.5),
            ("source_grade", "G"),
            ("source_grade", 1),
            ("location", {"lat": 91, "lon": 0}),
            ("location", {"lat": 0, "lon": 181}),
            ("location", {"lat": 0, "lon": 0, "radius_km": -1}),
            ("location", {"lat": 0}),
            ("text", 5),
            ("derived_from", "a-single-string"),
            ("numeric_values", 5),
        ],
    )
    def test_unusable_fields_are_dropped_with_a_note(self, field_name: str, value: object) -> None:
        (record,), notes = normalize_reports({"human": {field_name: value}})
        assert len(notes) == 1
        assert notes[0].startswith(f"report human: {field_name}")
        default = ReportRecord(report_id="human", discipline="human")
        assert dataclasses.replace(record, derived_from=()) == default

    def test_timestamps(self) -> None:
        assert parse_timestamp(1.5) == 1.5
        assert parse_timestamp(np.int64(7)) == 7.0
        assert parse_timestamp("2026-01-01T01:00:00+01:00") == 1767225600.0
        assert parse_timestamp("2026-01-01T00:00:00") is None
        assert parse_timestamp(math.inf) is None
        assert parse_timestamp(None) is None

    def test_duplicate_report_ids_are_one_report_presented_twice(self) -> None:
        reports = {"open_source": {"report_id": "r1"}, "signals": {"report_id": "r1"}}
        records, notes = normalize_reports(reports)
        # Same report twice: linked without a citation edge, so neither copy
        # is presumed to be the original by key order.
        assert [(r.report_id, r.derived_from, r.same_as) for r in records] == [
            ("r1", (), ()),
            ("r1@signals", (), ("r1",)),
        ]
        assert "r1@signals, treated as the same report as r1" in notes[0]
        assert len(analyse_independence(records).clusters) == 1
        three = {"a": [{"report_id": "x"}, {"report_id": "x"}, {"report_id": "x"}]}
        ids = [r.report_id for r in normalize_reports(three)[0]]
        assert ids == ["x", "x@a", "x@a#2"]

    def test_non_mappings(self) -> None:
        with pytest.raises(TypeError):
            normalize_reports([{"x": 1}])  # type: ignore[arg-type]
        records, notes = normalize_reports({"a": 3, "b": [{"text": "ok"}, "bad"]})
        assert [r.report_id for r in records] == ["b[0]"]
        assert notes == (
            "discipline a: value is not a report mapping; ignored",
            "b[1]: not a report mapping; ignored",
        )

    def test_provenance_records_are_read(self) -> None:
        from omni_mercury_engine.intel.provenance import Provenance, ProvenanceOrigin

        attributed = Provenance(ProvenanceOrigin.EXTRACTIVE, sources=("doc-1",))
        synthetic = Provenance(ProvenanceOrigin.MODEL_GENERATED, sources=("doc-2",))
        records, notes = normalize_reports(
            {
                "a": {"provenance": attributed},
                "b": {"provenance": synthetic},
                "c": {"provenance": attributed.as_dict(), "derived_from": ["doc-0"]},
                "d": {"provenance": {"origin": "nonsense", "sources": ["z"]}},
                "e": {"provenance": {"origin": "human", "sources": "z"}},
            }
        )
        by_id = {r.report_id: r for r in records}
        assert (by_id["a"].derived_from, by_id["a"].provenance_adequate) == (("doc-1",), True)
        assert (by_id["b"].derived_from, by_id["b"].provenance_adequate) == (("doc-2",), False)
        assert by_id["c"].derived_from == ("doc-0", "doc-1")
        assert by_id["d"].provenance_adequate is None
        assert by_id["e"].provenance_adequate is None
        assert notes == (
            "report b: provenance fails Provenance.is_adequate() (unattributed origin or no "
            "citation)",
            "report d: provenance ignored (not a Provenance record or its as_dict() mapping)",
            "report e: provenance ignored (not a Provenance record or its as_dict() mapping)",
        )


# --------------------------------------------------------------------------- #
# 7. The orchestrator.
# --------------------------------------------------------------------------- #
PRIORS = {"open_source": 0.75, "human": 0.85, "signals": 0.9, "geospatial": 0.93}


class TestAssessor:
    def test_empty_input_yields_no_finding(self) -> None:
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess({})
        assert result.flags == ()
        assert result.fusion_weight_multipliers == {}
        assert result.adjusted_reliability == {}
        assert result.effective_independent_sources == 0
        assert result.independent_corroboration() == 0
        assert result.reasons == ("no reports to assess",)

    def test_engine_format_degrades_gracefully(self) -> None:
        reports = {
            "open_source": {"confidence": 0.7, "indicators": ["x"], "threat_score": 0.6},
            "signals": {"confidence": 0.8, "threat_score": 0.7},
        }
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports)
        assert result.fusion_weight_multipliers == {"open_source": 1.0, "signals": 1.0}
        assert result.adjusted_reliability == {"open_source": 0.75, "signals": 0.9}
        assert result.independent_corroboration() == 2
        assert result.flags == ()
        assert result.reasons == ()

    def test_adjusted_reliability_pools_prior_and_sources(self) -> None:
        reports = {
            "human": {"report_id": "h", "source_id": "S-E", "source_grade": "E"},
            "signals": {"report_id": "s", "source_id": "S-F", "source_grade": "F"},
            "geospatial": {"report_id": "g", "source_id": "S-F2", "source_grade": "F"},
            "imagery": {"report_id": "i", "source_id": "S-B", "source_grade": "B"},
            "cyber": {"report_id": "c", "source_id": "S-none"},
        }
        history = {"outcomes": {"S-F": [True, True]}}
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports, history)
        assert result.adjusted_reliability["human"] == pytest.approx((1.7 + 0.25) / 4)
        assert result.adjusted_reliability["signals"] == pytest.approx((1.8 + 2) / 4)
        assert result.adjusted_reliability["geospatial"] == 0.93
        assert result.adjusted_reliability["imagery"] == pytest.approx(0.875)
        assert result.adjusted_reliability["cyber"] is None
        assert (
            "discipline cyber: no engine prior and no source information; reliability not adjusted"
            in result.reasons
        )
        strong = CounterintelligenceAssessor(
            config=CounterintelligenceConfig(discipline_prior_strength=6.0),
            discipline_priors=PRIORS,
        ).assess(reports)
        assert strong.adjusted_reliability["human"] == pytest.approx((5.1 + 0.25) / 8)

    def test_two_sources_in_one_discipline_both_count(self) -> None:
        reports = {
            "human": [
                {"report_id": "h1", "source_id": "A1", "source_grade": "A"},
                {"report_id": "h2", "source_id": "E1", "source_grade": "E"},
            ]
        }
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports)
        assert result.adjusted_reliability["human"] == pytest.approx((1.7 + 1.94 + 0.25) / 6)

    def test_conflicting_grades_use_least_reliable(self) -> None:
        reports = {
            "human": {"report_id": "a", "source_id": "S", "source_grade": "B"},
            "signals": {"report_id": "b", "source_id": "S", "source_grade": "D"},
            "cyber": {"report_id": "c", "source_id": "T", "source_grade": "F"},
        }
        result = CounterintelligenceAssessor().assess(reports)
        assert result.source_posteriors["S"].grade == "D"
        assert result.source_posteriors["T"].grade == "F"
        assert (
            "source S: conflicting grades ['B', 'D']; the least reliable judged grade D is used"
            in result.reasons
        )
        mixed = CounterintelligenceAssessor().assess(
            {
                "a": {"report_id": "a", "source_id": "S", "source_grade": "F"},
                "b": {"report_id": "b", "source_id": "S", "source_grade": "C"},
            }
        )
        assert mixed.source_posteriors["S"].grade == "C"

    def test_relay_gets_zero_weight_and_corroboration_counts_clusters(self) -> None:
        reports = {
            "open_source": {"report_id": "o", "source_id": "S1", "reported_at": T0},
            "human": {"report_id": "h", "source_id": "S2", "derived_from": ["o"]},
            "signals": {"report_id": "s", "source_id": "S3"},
        }
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports)
        assert result.report_weight_multipliers == {"h": 0.0, "o": 1.0, "s": 1.0}
        assert result.fusion_weight_multipliers == {
            "human": 0.0,
            "open_source": 1.0,
            "signals": 1.0,
        }
        assert result.independent_corroboration() == 2
        assert result.independent_corroboration(["human", "open_source"]) == 1
        assert result.independent_corroboration(["human"]) == 1
        assert result.discipline_clusters["human"] == ("cluster:o",)
        assert "report h: weight 0, not independent of o, which represents cluster:o" in (
            result.reasons
        )
        assert [f.kind for f in result.flags] == [FlagKind.CIRCULAR_REPORTING]

    def test_cycle_members_get_the_circular_factor(self) -> None:
        reports = {
            "a": {"report_id": "A", "derived_from": ["B"]},
            "b": {"report_id": "B", "derived_from": ["A"]},
        }
        cfg = CounterintelligenceConfig(circular_factor=0.3)
        result = CounterintelligenceAssessor(config=cfg).assess(reports)
        assert result.report_weight_multipliers == {"A": 0.3, "B": 0.0}
        assert result.fusion_weight_multipliers == {"a": 0.3, "b": 0.0}

    def test_fabrication_and_travel_factors_compose(self) -> None:
        rounded = [float(10 * i) for i in range(1, 61)]
        reports = {
            "human": {
                "report_id": "h",
                "source_id": "S",
                "reported_at": T0,
                "location": {"lat": 0.0, "lon": 0.0},
                "numeric_values": rounded,
            },
            "signals": {
                "report_id": "s",
                "source_id": "S2",
                "entity_id": "S",
                "reported_at": T0 + 600,
                "location": {"lat": 0.0, "lon": 20.0},
            },
        }
        cfg = CounterintelligenceConfig(fabrication_factor=0.4, impossible_travel_factor=0.5)
        result = CounterintelligenceAssessor(config=cfg).assess(reports)
        kinds = [f.kind for f in result.flags]
        assert kinds == [FlagKind.FABRICATION, FlagKind.IMPOSSIBLE_TRAVEL]
        assert result.report_weight_multipliers == {"h": pytest.approx(0.2), "s": 0.5}
        fabrication = result.flags_of(FlagKind.FABRICATION)[0]
        fired = [screen["test"] for screen in fabrication.evidence["screens"]]
        assert fired == ["terminal_digit", "excess_rounding"]
        # Terminal digits all 0: chi2 = (60 - 6)**2 / 6 + 9 * 6 = 540 on 9 df, the
        # smaller of the two p-values; its BH q-value is 2p (m = 2, rank 1).
        terminal_p = float(stats.chi2.sf(540.0, 9))
        assert fabrication.p_value == pytest.approx(terminal_p)
        assert fabrication.q_value == pytest.approx(2 * terminal_p)
        rounding = fabrication.evidence["screens"][1]
        assert rounding["p_value"] == pytest.approx(0.2**60)
        assert rounding["q_value"] == pytest.approx(0.2**60)
        travel = result.flags_of(FlagKind.IMPOSSIBLE_TRAVEL)[0]
        assert travel.report_ids == ("h", "s")
        assert travel.evidence["implied_speed_kmh"] == pytest.approx(20 * KM_PER_DEGREE * 6)
        assert "benford_first_digit: report:h not tested (60 positive values; at least 110 " in (
            " ".join(result.reasons)
        )

    def test_significant_benford_without_nonconformity_is_not_flagged(self) -> None:
        counts = [round(100_000 * p) for p in BENFORD_FIRST_DIGIT_PROBABILITIES]
        counts[0] -= 400
        counts[8] += 400
        values = benford_like_values(counts)
        assert len(values) % 10 == 0
        result = CounterintelligenceAssessor().assess({"financial": {"numeric_values": values}})
        benford = result.fabrication_screens[0].results[0]
        assert benford.p_value is not None
        assert benford.p_value < 1e-5
        assert benford.conformity == "close conformity"
        assert result.flags == ()
        nonconforming = benford_like_values([111] * 9)
        flagged = CounterintelligenceAssessor().assess(
            {"financial": {"numeric_values": nonconforming}}
        )
        assert [s["test"] for s in flagged.flags[0].evidence["screens"]] == ["benford_first_digit"]

    def test_coordinated_campaign_flag(self) -> None:
        reports = {
            "open_source": [
                {
                    "report_id": f"p{i}",
                    "source_id": f"acct{i}",
                    "claim_id": "claim-x",
                    "reported_at": T0 + 60 * i,
                }
                for i in range(6)
            ]
        }
        history = {"baseline_rate_per_hour": {"claim-x": 0.1}}
        cfg = CounterintelligenceConfig(observation_hours=24.0, coordinated_factor=0.25)
        result = CounterintelligenceAssessor(config=cfg).assess(reports, history)
        (flag,) = result.flags_of(FlagKind.COORDINATED_REPORTING)
        assert flag.report_ids == tuple(f"p{i}" for i in range(6))
        assert flag.p_value == pytest.approx(scan_statistic_pvalue(6, 0.1, 1.0, 24.0))
        assert flag.evidence["source_keys"] == tuple(f"acct{i}" for i in range(6))
        assert result.report_weight_multipliers == {f"p{i}": 0.25 for i in range(6)}

    def test_credibility_one_without_independent_corroboration(self) -> None:
        alone = {"human": {"report_id": "h", "information_credibility": 1}}
        result = CounterintelligenceAssessor().assess(alone)
        note = "report h: credibility 1 (confirmed by other sources) has no independent "
        assert any(r.startswith(note) for r in result.reasons)
        relayed = dict(alone, signals={"report_id": "s", "derived_from": ["h"]})
        assert any(
            r.startswith(note) for r in CounterintelligenceAssessor().assess(relayed).reasons
        )
        backed = dict(alone, signals={"report_id": "s"})
        assert not any(
            r.startswith(note) for r in CounterintelligenceAssessor().assess(backed).reasons
        )

    def test_citing_a_later_report_is_noted(self) -> None:
        reports = {
            "a": {"report_id": "a", "reported_at": T0, "derived_from": ["b"]},
            "b": {"report_id": "b", "reported_at": T0 + 1},
            "c": {"report_id": "c", "reported_at": T0 + 1, "derived_from": ["b"]},
        }
        reasons = CounterintelligenceAssessor().assess(reports).reasons
        assert "report a cites b, which is timestamped later" in reasons
        assert not any(r.startswith("report c cites") for r in reasons)

    def test_zero_factor_removes_cluster_from_corroboration(self) -> None:
        reports = {
            "a": {"report_id": "a", "derived_from": ["b"]},
            "b": {"report_id": "b", "derived_from": ["a"]},
            "c": {"report_id": "c"},
        }
        cfg = CounterintelligenceConfig(circular_factor=0.0)
        assert (
            CounterintelligenceAssessor(config=cfg).assess(reports).independent_corroboration() == 1
        )

    def test_history_coercion(self) -> None:
        history, notes = AssessmentHistory.coerce(
            {
                "outcomes": {
                    "S": [True, {"confirmed": False, "weight": 1.5}, SourceOutcome(True), "x", {}],
                    "T": 5,
                },
                "baseline_rate_per_hour": {"c": 0.5, "d": -1, "e": "fast", "z": 0},
                "default_baseline_rate_per_hour": 0,
            }
        )
        assert history.outcomes["S"] == (
            SourceOutcome(True),
            SourceOutcome(False, 1.5),
            SourceOutcome(True),
        )
        assert "T" not in history.outcomes
        assert dict(history.baseline_rate_per_hour) == {"c": 0.5}
        assert history.default_baseline_rate_per_hour is None
        assert notes == (
            "history: outcome for source S dropped (str)",
            "history: outcome for source S dropped ('confirmed')",
            "history: outcomes for source T ignored (not a list)",
            "history: baseline rate for claim d dropped (not > 0)",
            "history: baseline rate for claim e dropped (not > 0)",
            "history: baseline rate for claim z dropped (not > 0)",
            "history: default baseline rate dropped (not > 0)",
        )
        same = AssessmentHistory()
        assert AssessmentHistory.coerce(same) == (same, ())
        assert AssessmentHistory.coerce(None) == (AssessmentHistory(), ())
        assert AssessmentHistory.coerce({"outcomes": [1]})[1] == (
            "history: outcomes ignored (not a mapping of source to outcomes)",
        )
        assert AssessmentHistory.coerce({"baseline_rate_per_hour": [1]})[1] == (
            "history: baseline_rate_per_hour ignored (not a mapping)",
        )
        with pytest.raises(TypeError):
            AssessmentHistory.coerce(3)

    def test_priors_validation_and_from_engine(self) -> None:
        for bad in (1.5, -0.1, "high", math.nan):
            with pytest.raises(ValueError):
                CounterintelligenceAssessor(discipline_priors={"x": bad})  # type: ignore[dict-item]

        class StubEngine:
            int_reliability_scores = {"open_source": 0.6}

        assessor = CounterintelligenceAssessor.from_engine(StubEngine())
        assert dict(assessor.discipline_priors) == {"open_source": 0.6}
        assert CounterintelligenceAssessor.from_engine(object()).discipline_priors == {}

    def test_pure_deterministic_and_order_invariant(self) -> None:
        reports: dict[str, Any] = {
            "open_source": [
                {"report_id": "o1", "source_id": "S1", "reported_at": T0, "text": TEXT},
                {"report_id": "o2", "source_id": "S2", "reported_at": T0 + 60, "text": TEXT},
            ],
            "human": {
                "report_id": "h",
                "source_id": "S3",
                "derived_from": ["ext"],
                "numeric_values": list(range(10, 200)),
            },
            "signals": {"report_id": "s", "source_id": "S4", "derived_from": ["ext"]},
        }
        snapshot = copy.deepcopy(reports)
        assessor = CounterintelligenceAssessor(discipline_priors=PRIORS)
        first = assessor.assess(reports, {"default_baseline_rate_per_hour": 0.1})
        assert reports == snapshot
        again = assessor.assess(reports, {"default_baseline_rate_per_hour": 0.1})
        assert first == again
        reordered = dict(reversed(list(reports.items())))
        assert assessor.assess(reordered, {"default_baseline_rate_per_hour": 0.1}) == first
        with pytest.raises(dataclasses.FrozenInstanceError):
            first.reasons = ()  # type: ignore[misc]
        encoded = json.loads(json.dumps(first.as_dict()))
        assert encoded["fusion_weight_multipliers"] == first.fusion_weight_multipliers
        assert encoded["flags"][0]["kind"] == first.flags[0].kind.value
        assert first.flags[0].as_dict()["evidence"] == encoded["flags"][0]["evidence"]

    def test_real_fusion_engine_priors(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.security.intelligence_fusion import IntelligenceFusionEngine

        engine = IntelligenceFusionEngine(enable_neurosymbolic=False, enable_cryptanalysis=False)
        reports = {
            "open_source": {"confidence": 0.7, "threat_score": 0.6},
            "human": {"confidence": 0.8, "threat_score": 0.7},
        }
        result = CounterintelligenceAssessor.from_engine(engine).assess(reports)
        assert result.adjusted_reliability == {
            d: engine.int_reliability_scores[d] for d in ("human", "open_source")
        }
        assert result.fusion_weight_multipliers == {"human": 1.0, "open_source": 1.0}
        assert engine.fuse_intelligence(reports).risk_score == pytest.approx(
            (0.75 * 0.6 + 0.85 * 0.7) / (0.75 + 0.85)
        )


# --------------------------------------------------------------------------- #
# Identity attacks: id hijack, source spoof, duplicate ids.
# --------------------------------------------------------------------------- #
GENUINE = {"confidence": 0.95, "threat_score": 0.95, "reported_at": T0 + 5}


class TestIdentityAttacks:
    """A self-declared id, source or timestamp must not displace a genuine report.

    Regression: default ids were the bare discipline key, so
    ``{"communications": {"report_id": "signals"}}`` (sorting first) made the
    genuine signals report a "derived copy" with weight 0; and cluster
    representatives were chosen by self-reported time, so a spoofed
    ``source_id`` with an earlier ``reported_at`` took the genuine cluster.
    """

    def test_declaring_another_disciplines_default_id_does_not_hijack_it(self) -> None:
        reports = {
            "communications": {"report_id": "signals", "threat_score": 0.0, "reported_at": T0},
            "signals": dict(GENUINE),
        }
        records, notes = normalize_reports(reports)
        by_discipline = {r.discipline: r for r in records}
        assert by_discipline["signals"].report_id == "signals"
        assert by_discipline["signals"].derived_from == by_discipline["signals"].same_as == ()
        assert by_discipline["communications"].report_id == "signals@communications"
        assert by_discipline["communications"].same_as == ()
        assert any("another report's default id" in note for note in notes)
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports)
        assert result.fusion_weight_multipliers == {"communications": 1.0, "signals": 1.0}
        assert result.effective_independent_sources == 2

    def test_list_default_ids_are_reserved_too(self) -> None:
        reports = {"signals": [{"report_id": "signals[1]"}, {}]}
        ids = [r.report_id for r in normalize_reports(reports)[0]]
        assert ids == ["signals[1]@signals", "signals[1]"]

    def test_spoofed_source_with_earlier_time_does_not_take_the_cluster(self) -> None:
        reports = {
            "open_source": {"source_id": "S1", "reported_at": T0, "threat_score": 0.0},
            "signals": {"source_id": "S1", **GENUINE},
        }
        result = CounterintelligenceAssessor(discipline_priors=PRIORS).assess(reports)
        (cluster,) = result.clusters
        # signals (prior 0.9) outranks open_source (0.75) despite reporting later.
        assert cluster.representative == "signals"
        assert result.report_weight_multipliers == {"open_source": 0.0, "signals": 1.0}
        (conflict,) = result.flags_of(FlagKind.CONFLICTING_CLAIMS)
        assert conflict.report_ids == ("open_source", "signals")
        assert conflict.evidence["representative"] == "signals"
        assert dict(conflict.evidence["threat_scores"]) == {"open_source": 0.0, "signals": 0.95}

    def test_duplicate_declared_id_is_resolved_by_reliability_not_key_order(self) -> None:
        reports = {
            "communications": {"report_id": "sig-1", "threat_score": 0.0, "reported_at": T0},
            "signals": {"report_id": "sig-1", **GENUINE},
        }
        priors = {**PRIORS, "communications": 0.88}
        result = CounterintelligenceAssessor(discipline_priors=priors).assess(reports)
        (cluster,) = result.clusters
        assert cluster.representative == "sig-1@signals"
        assert result.fusion_weight_multipliers == {"communications": 0.0, "signals": 1.0}
        assert len(result.flags_of(FlagKind.CONFLICTING_CLAIMS)) == 1

    def test_representative_ranks_reliability_before_time(self) -> None:
        records = [
            rec("early", source_id="S", reported_at=T0),
            rec("late", source_id="S", reported_at=T0 + 9),
        ]
        assert analyse_independence(records).clusters[0].representative == "early"
        ranked = analyse_independence(records, reliability={"early": 0.5, "late": 0.9})
        assert ranked.clusters[0].representative == "late"
        unknown = analyse_independence(records, reliability={"early": None, "late": 0.1})
        assert unknown.clusters[0].representative == "late"

    def test_agreeing_or_unscored_cluster_members_raise_no_conflict(self) -> None:
        same = [
            rec("a", source_id="S", threat_score=0.5),
            rec("b", source_id="S", threat_score=0.5),
            rec("c", source_id="S"),
        ]
        assert analyse_independence(same).flags == ()
        apart = [
            rec("a", source_id="S", threat_score=0.5),
            rec("b", source_id="S", threat_score=0.6),
        ]
        lenient = CounterintelligenceConfig(claim_conflict_tolerance=0.2)
        assert analyse_independence(apart, lenient).flags == ()
        assert len(analyse_independence(apart).flags) == 1

    def test_negative_conflict_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="claim_conflict_tolerance"):
            CounterintelligenceConfig(claim_conflict_tolerance=-0.1)


# --------------------------------------------------------------------------- #
# Property tests.
# --------------------------------------------------------------------------- #
if HAS_HYPOTHESIS:

    @st.composite
    def report_batches(draw: Any) -> dict[str, Any]:
        """Random batches with citations, shared sources, texts and times."""
        n = draw(st.integers(min_value=1, max_value=8))
        ids = [f"r{i}" for i in range(n)]
        words = ["alpha", "beta", "gamma", "delta", "eps", "zeta"]
        batch: dict[str, Any] = {}
        for i, rid in enumerate(ids):
            report: dict[str, Any] = {"report_id": rid}
            if draw(st.booleans()):
                report["source_id"] = draw(st.sampled_from(["S1", "S2", "S3", "S4"]))
            report["derived_from"] = draw(
                st.lists(st.sampled_from([*ids, "ext1", "ext2"]), max_size=3)
            )
            if draw(st.booleans()):
                report["reported_at"] = T0 + draw(st.integers(min_value=0, max_value=7200))
            if draw(st.booleans()):
                report["text"] = " ".join(draw(st.lists(st.sampled_from(words), max_size=8)))
            batch[f"disc{i % 3}-{i}"] = report
        return batch

    class TestProperties:
        @settings(max_examples=150, deadline=None, derandomize=True, database=None)
        @given(report_batches())
        def test_clusters_partition_and_bound_independence(self, batch: dict[str, Any]) -> None:
            result = CounterintelligenceAssessor().assess(batch)
            ids = [r.report_id for r in result.reports]
            members = [rid for c in result.clusters for rid in c.report_ids]
            assert sorted(members) == sorted(ids)
            n_clusters = result.effective_independent_sources
            assert 1 <= n_clusters <= len(ids)
            assert n_clusters <= result.independence.distinct_origins
            for cluster in result.clusters:
                weights = [result.report_weight_multipliers[r] for r in cluster.report_ids]
                assert sum(w > 0 for w in weights) <= 1
                assert all(
                    result.report_weight_multipliers[r] == 0.0
                    for r in cluster.report_ids
                    if r != cluster.representative
                )
            assert all(0.0 <= m <= 1.0 for m in result.fusion_weight_multipliers.values())
            assert result.independent_corroboration() <= n_clusters

        @settings(max_examples=60, deadline=None, derandomize=True, database=None)
        @given(report_batches())
        def test_assessment_is_pure_and_order_invariant(self, batch: dict[str, Any]) -> None:
            snapshot = copy.deepcopy(batch)
            assessor = CounterintelligenceAssessor()
            first = assessor.assess(batch)
            assert batch == snapshot
            assert assessor.assess(dict(reversed(list(batch.items())))) == first

        @settings(max_examples=100, deadline=None, derandomize=True, database=None)
        @given(
            st.lists(st.booleans(), max_size=30),
            st.sampled_from(list("ABCDEF")),
            st.floats(min_value=0.05, max_value=1.0),
        )
        def test_posterior_is_a_valid_beta(
            self, outcomes: list[bool], grade: str, forgetting: float
        ) -> None:
            cfg = CounterintelligenceConfig(forgetting_factor=forgetting)
            post = source_posterior("s", grade, [SourceOutcome(o) for o in outcomes], config=cfg)
            low, high = post.credible_interval
            assert 0.0 <= low <= post.mean <= high <= 1.0 or math.isclose(low, high)
            assert post.alpha >= post.prior_alpha
            assert post.beta >= post.prior_beta
            assert abs(post.log_odds_weight) <= DEFAULT_LOG_ODDS_CLIP
            assert post.effective_sample_size == pytest.approx(post.alpha + post.beta)

        @settings(max_examples=100, deadline=None, derandomize=True, database=None)
        @given(st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=1, max_size=30))
        def test_bh_adjusted_is_monotone_and_bounded(self, p: list[float]) -> None:
            q = benjamini_hochberg_adjusted(p)
            assert all(pi <= qi + 1e-15 <= 1.0 + 1e-15 for pi, qi in zip(p, q))
            ordered = sorted(range(len(p)), key=lambda i: p[i])
            assert all(q[a] <= q[b] + 1e-15 for a, b in itertools.pairwise(ordered))

        @settings(max_examples=60, deadline=None, derandomize=True, database=None)
        @given(
            st.integers(min_value=1, max_value=12),
            st.floats(min_value=0.01, max_value=5.0),
            st.floats(min_value=0.1, max_value=3.0),
            st.floats(min_value=0.1, max_value=48.0),
        )
        def test_scan_pvalue_is_monotone_in_k(
            self, k: int, rate: float, window: float, period: float
        ) -> None:
            p_k = scan_statistic_pvalue(k, rate, window, period)
            assert 0.0 <= scan_statistic_pvalue(k + 1, rate, window, period) <= p_k <= 1.0

else:  # pragma: no cover

    @pytest.mark.skip(reason="hypothesis is not installed")
    def test_properties_need_hypothesis() -> None:
        """Placeholder so the skip is visible in the report."""

    _ = MappingProxyType


class TestOutcomeWeightFit:
    """fit_outcome_weight_cap recovers known weights and backs MAX_OUTCOME_WEIGHT."""

    @staticmethod
    def _sources(
        rng: np.random.Generator, n: int, length: int, regime: int
    ) -> dict[str, list[bool]]:
        """Sources whose reliability is redrawn from Beta(1, 1) every ``regime`` outcomes."""
        out: dict[str, list[bool]] = {}
        for i in range(n):
            outcomes: list[bool] = []
            while len(outcomes) < length:
                p = rng.beta(1.0, 1.0)
                outcomes.extend(bool(x) for x in rng.random(regime) < p)
            out[f"s{i}"] = outcomes[:length]
        return out

    def test_iid_sources_under_the_true_prior_fit_weight_one(self) -> None:
        # Exchangeable Bernoulli outcomes with a Beta(1, 1) reliability: the
        # Beta-Bernoulli predictive is Bayes-optimal at weight 1.
        histories = self._sources(np.random.default_rng(0), 3000, 12, regime=12)
        fit = fit_outcome_weight_cap(
            histories, weights=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0], n_boot=0
        )
        assert fit.optimal_weight == 1.0
        assert fit.break_even_weight > 4.0

    def test_drifting_sources_are_worth_less_than_one_observation(self) -> None:
        histories = self._sources(np.random.default_rng(1), 3000, 12, regime=2)
        fit = fit_outcome_weight_cap(histories, weights=[0.1, 0.25, 0.5, 1.0, 2.0], n_boot=50)
        assert fit.optimal_weight < 1.0
        assert fit.break_even_ci is not None
        low, high = fit.break_even_ci
        assert low <= fit.break_even_weight <= high

    def test_too_few_sources_are_refused(self) -> None:
        with pytest.raises(ValueError, match="two sources"):
            fit_outcome_weight_cap({"a": [True, False], "b": [True]})

    def test_module_cap_matches_the_committed_fit(self) -> None:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / "benchmarks/results/outcome_weight_fit.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["recommended_max_outcome_weight"] == MAX_OUTCOME_WEIGHT
        assert result["fit"]["n_outcomes"] == 8401
        low, _high = result["fit"]["break_even_ci"]
        assert low >= MAX_OUTCOME_WEIGHT


def test_overwatch_nexus_is_not_restored() -> None:
    """DEPRECATION §6.13: the released class scored reports it fabricated itself."""
    import omni_mercury_engine.security.counterintelligence as module

    assert not hasattr(module, "OverwatchNexus")
    assert not hasattr(module, "OverwatchNexusResult")

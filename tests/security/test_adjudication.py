# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Behavioural tests for the adjudicated, non-circular, held-out label ledger.

Expected values are hand-computed or come from independent oracles written
here: kappa from the Fleiss & Cohen (1973) moment identity in exact rational
arithmetic (the module uses the confusion-matrix form), the bootstrap from a
re-implementation of the documented resampling protocol, and the split from
``hashlib`` directly. Forgery tests follow ``tests/test_self_model.py``: each
forgery re-seals every digest a forger without the signing key can recompute,
so exactly one check is left able to catch it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine import self_model
from omni_mercury_engine.security import adjudication as adj
from omni_mercury_engine.security.adjudication import (
    GENESIS_DIGEST,
    AdjudicationLedger,
    AdjudicatorDecision,
    AgreementPolicy,
    Case,
    ExportRefused,
    Outcome,
    Rating,
    assess_agreement,
    bootstrap_kappa_interval,
    load_export,
    normalize_source_id,
    outcome_from_ucdp_row,
    quadratic_weighted_kappa,
    resolve,
    split_for,
    ucdp_fatality_level,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

try:
    from hypothesis import (
        given,
        settings,
        strategies as st,
    )

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover - hypothesis is in the test extra
    HAVE_HYPOTHESIS = False

T0 = datetime(2026, 1, 1, tzinfo=UTC)
DAY = timedelta(days=1)
#: Small-sample policy for protocol tests (the defaults are tested on their own).
SMALL = AgreementPolicy(min_pair_cases=2, n_boot=50)
#: For tests about something other than the gate: any defined kappa passes.
PERMISSIVE = AgreementPolicy(min_pair_cases=2, kappa_threshold=-1.0, n_boot=20)

#: Ratings whose quadratic-weighted kappa is exactly 61/100 (found by exact search).
K61_A = [2, 4, 1, 3, 1, 1, 1, 1, 3, 2, 1, 3]
K61_B = [3, 4, 1, 1, 1, 2, 1, 1, 4, 1, 3, 4]


# ---------------------------------------------------------------------------
# Independent oracles and builders
# ---------------------------------------------------------------------------


def oracle_kappa(a: Sequence[int], b: Sequence[int]) -> Fraction | None:
    """Quadratic kappa via Fleiss & Cohen (1973): 2 s_xy / (s_x^2 + s_y^2 + (mx - my)^2)."""
    n = len(a)
    mx, my = Fraction(sum(a), n), Fraction(sum(b), n)
    vx = sum((x - mx) ** 2 for x in a) / n
    vy = sum((y - my) ** 2 for y in b) / n
    cxy = sum((x - mx) * (y - my) for x, y in zip(a, b)) / n
    denominator = Fraction(vx + vy + (mx - my) ** 2)
    return None if denominator == 0 else Fraction(2 * cxy) / denominator


def bucket(case_id: str) -> int:
    return int(hashlib.sha3_256(case_id.encode("utf-8")).hexdigest(), 16) % 100


def ids_in(split: str, count: int, prefix: str = "case", pct: int = 25) -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < count:
        candidate = f"{prefix}-{i:04d}"
        if (bucket(candidate) < pct) == (split == "heldout"):
            out.append(candidate)
        i += 1
    return out


def reports_for(case_id: str) -> dict[str, Any]:
    """A payload unique to each case id (the merit gate counts duplicates once)."""
    digest = int(hashlib.sha3_256(case_id.encode("utf-8")).hexdigest(), 16)
    return {"open_source": {"threat_score": (digest % 10**6) / 10**6, "confidence": 0.8}}


def make_case(cid: str, /, **overrides: Any) -> Case:
    fields: dict[str, Any] = {
        "case_id": cid,
        "created_at": T0,
        "intel_reports": reports_for(cid),
        "report_source_ids": ("Reuters",),
        "latest_report_at": T0,
    }
    fields.update(overrides)
    return Case(**fields)


def make_outcome(cid: str, /, **overrides: Any) -> Outcome:
    fields: dict[str, Any] = {
        "case_id": cid,
        "observed_at": T0 + DAY,
        "description": f"what happened in {cid}",
        "outcome_source_ids": ("Vanguard",),
    }
    fields.update(overrides)
    return Outcome(**fields)


def rate(rater: str, case_id: str, level: int, at: datetime = T0 + 2 * DAY) -> Rating:
    return Rating(rater, case_id, level, f"{rater} judged {level}", at)


def decide(case_id: str, level: int, who: str = "dana", at: datetime = T0 + 3 * DAY) -> Any:
    return AdjudicatorDecision(who, case_id, level, "resolved on the record", at)


def case_items(
    case_id: str,
    levels: Sequence[int],
    raters: Sequence[str] = ("alice", "bob", "carol"),
    decision: int | None = None,
    **case_overrides: Any,
) -> list[Any]:
    items: list[Any] = [make_case(case_id, **case_overrides), make_outcome(case_id)]
    items += [rate(raters[i], case_id, level) for i, level in enumerate(levels)]
    if decision is not None:
        items.append(decide(case_id, decision))
    return items


def consensus_ledger(
    path: Path, heldout: int, training: int, policy: AgreementPolicy | None = None
) -> tuple[AdjudicationLedger, list[str]]:
    """``heldout`` + ``training`` consensus cases in id order, level 1 + (i mod 5)."""
    wanted = set(ids_in("heldout", heldout)) | set(ids_in("training", training))
    order = sorted(wanted)
    ledger = AdjudicationLedger(path, policy=policy or AgreementPolicy(n_boot=50))
    items: list[Any] = []
    for i, cid in enumerate(order):
        items += case_items(cid, [1 + i % 5, 1 + i % 5])
    ledger.extend(items)
    return ledger, order


def level_of(order: list[str], case_id: str) -> int:
    return 1 + order.index(case_id) % 5


def reseal(record: dict[str, Any], *, entry: bool, chain: bool) -> None:
    """Recompute the digests a forger can recompute without the signing key."""
    if entry:
        body = {k: v for k, v in record["entry"].items() if k != "digest"}
        record["entry"]["digest"] = hashlib.sha3_256(adj._canonical(body)).hexdigest()
    if chain:
        unsigned = {k: v for k, v in record.items() if k not in ("record_digest", "signature")}
        record["record_digest"] = hashlib.sha3_256(adj._canonical(unsigned)).hexdigest()


def rewrite(path: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    """Mutate the records, then re-seal and re-chain all of them (a keyless forger)."""
    records = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(records)
    prev = GENESIS_DIGEST
    for index, record in enumerate(records):
        record["index"] = index
        record["prev_digest"] = prev
        reseal(record, entry=True, chain=True)
        prev = record["record_digest"]
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def make_circular(path: Path, case_id: str) -> None:
    """Re-seal the ledger with ``case_id``'s outcome sourced from its own reporter."""

    def mutate(records: list[dict[str, Any]]) -> None:
        for record in records:
            entry = record["entry"]
            if entry["kind"] == "outcome" and entry["case_id"] == case_id:
                entry["outcome_source_ids"] = ["Reuters"]

    rewrite(path, mutate)


def mldsa() -> Any:
    try:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider
    except (ImportError, RuntimeError):  # pragma: no cover - AMA is required in CI
        pytest.skip("ML-DSA-65 needs the AMA native backend")
    return MLDSAProvider()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class TestRecords:
    def test_timestamps_are_normalised_to_utc(self) -> None:
        plus2 = timezone(timedelta(hours=2))
        case = make_case("c", created_at=datetime(2026, 1, 1, 2, tzinfo=plus2), latest_report_at=T0)
        assert case.created_at == T0 and case.created_at.tzinfo is UTC
        assert case.to_entry("training", 25)["created_at"] == "2026-01-01T00:00:00+00:00"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"created_at": datetime(2026, 1, 1)},  # naive
            {"latest_report_at": datetime(2026, 1, 1)},
            {"created_at": T0 - timedelta(microseconds=1)},  # before its reports
            {"intel_reports": {}},
            {"intel_reports": ["open_source"]},
            {"intel_reports": {"open_source": 0.5}},
            {"intel_reports": {"open_source": {"threat_score": math.nan}}},
            {"intel_reports": {"open_source": {"threat_score": object()}}},
            {"report_source_ids": "Reuters"},  # a bare string is not a list of ids
            {"report_source_ids": ()},
            {"report_source_ids": ("Reuters", "   ")},
            {"case_id": " padded"},
            {"case_id": ""},
        ],
    )
    def test_invalid_cases_are_refused(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            make_case("c", **overrides)

    def test_a_case_may_be_assembled_at_the_moment_of_its_last_report(self) -> None:
        assert make_case("c", created_at=T0, latest_report_at=T0).created_at == T0

    def test_intel_reports_are_copied_not_aliased(self) -> None:
        reports = {"open_source": {"threat_score": 0.4}}
        case = make_case("c", intel_reports=reports)
        reports["open_source"]["threat_score"] = 0.9
        assert case.intel_reports == {"open_source": {"threat_score": 0.4}}

    @pytest.mark.parametrize(
        "overrides",
        [
            {"outcome_level": 3},  # a level without the scale it came from
            {"outcome_scale": "some scale"},
            {"outcome_level": 0, "outcome_scale": "s"},
            {"outcome_level": 6, "outcome_scale": "s"},
            {"outcome_level": True, "outcome_scale": "s"},
            {"outcome_level": 3, "outcome_scale": "  "},
            {"description": " "},
            {"outcome_source_ids": []},
            {"observed_at": datetime(2026, 1, 2)},
        ],
    )
    def test_invalid_outcomes_are_refused(self, overrides: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            make_outcome("c", **overrides)

    @pytest.mark.parametrize("level", [0, 6, True, 3.0, "3", None])
    def test_rating_levels_must_be_threat_levels(self, level: Any) -> None:
        with pytest.raises(ValueError, match="threat level 1-5"):
            Rating("alice", "c", level, "why", T0)

    def test_ratings_and_decisions_need_a_rationale(self) -> None:
        with pytest.raises(ValueError, match="rationale"):
            Rating("alice", "c", 3, "", T0)
        with pytest.raises(ValueError, match="rationale"):
            AdjudicatorDecision("dana", "c", 3, "  ", T0)

    def test_levels_match_the_fusion_threat_levels(self) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.security.intelligence_fusion import ThreatLevel

        assert tuple(t.value for t in ThreatLevel) == adj.LEVELS

    def test_every_record_round_trips_through_its_entry(self) -> None:
        case = make_case("c")
        entry = {**case.to_entry("heldout", 25), "digest": "x"}
        assert Case.from_entry(entry) == (case, "heldout", 25)
        for record, parse in (
            (make_outcome("c", outcome_level=2, outcome_scale="s"), Outcome.from_entry),
            (rate("alice", "c", 4), Rating.from_entry),
            (decide("c", 2), AdjudicatorDecision.from_entry),
        ):
            assert parse({**record.to_entry(), "digest": "x"}) == record

    def test_an_entry_with_an_extra_field_is_malformed(self) -> None:
        entry = {**rate("alice", "c", 4).to_entry(), "digest": "x", "label": 5}
        with pytest.raises(ValueError, match="malformed rating entry"):
            Rating.from_entry(entry)


class TestSourceIds:
    @pytest.mark.parametrize(
        ("raw", "normalised"),
        [
            (" Reuters ", "reuters"),
            ("REUTERS", "reuters"),
            ("Kabul \t Now", "kabul now"),
            ("Ｒｅｕｔｅｒｓ", "reuters"),  # fullwidth, folded by NFKC
            ("Straße", "strasse"),  # casefold, not lower
        ],
    )
    def test_normalisation(self, raw: str, normalised: str) -> None:
        assert normalize_source_id(raw) == normalised


# ---------------------------------------------------------------------------
# Adjudication rules
# ---------------------------------------------------------------------------


class TestResolve:
    @pytest.mark.parametrize(
        ("levels", "decision", "expected"),
        [
            ([3, 3], None, (3, "consensus")),
            ([2, 3], None, (None, "one-level disagreement awaiting")),
            ([3, 2], 2, (2, "adjudicator_decision")),
            ([2, 4], 3, (None, "2-level disagreement awaiting a third rating")),
            ([1, 5], None, (None, "4-level disagreement awaiting a third rating")),
            ([2, 4, 4], None, (4, "majority_of_three")),
            ([2, 4, 2], None, (2, "majority_of_three")),
            ([2, 3, 3], None, (3, "majority_of_three")),
            ([1, 4, 2], None, (None, "three ratings without a majority")),
            ([1, 4, 2], 2, (2, "adjudicator_decision")),
            ([2, 3, 5], None, (None, "three ratings without a majority")),
            ([], None, (None, "0 of the 2 required ratings")),
            ([3], 3, (None, "1 of the 2 required ratings")),
        ],
    )
    def test_rules(
        self, levels: list[int], decision: int | None, expected: tuple[int | None, str]
    ) -> None:
        level, method = resolve(levels, decision)
        assert level == expected[0]
        assert method.startswith(expected[1])


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


class TestKappa:
    def test_hand_computed_value(self) -> None:
        # D_o = 1/5, D_e = 3.4, kappa = 1 - (1/5)/3.4 = 16/17.
        assert quadratic_weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 4]) == Fraction(16, 17)

    def test_exactly_sixty_one_hundredths(self) -> None:
        assert quadratic_weighted_kappa(K61_A, K61_B) == Fraction(61, 100)
        assert oracle_kappa(K61_A, K61_B) == Fraction(61, 100)

    def test_perfect_and_inverted_agreement(self) -> None:
        assert quadratic_weighted_kappa([1, 3, 5, 2], [1, 3, 5, 2]) == 1
        assert quadratic_weighted_kappa([1, 2], [2, 1]) == -1

    def test_undefined_without_expected_disagreement(self) -> None:
        assert quadratic_weighted_kappa([3, 3, 3], [3, 3, 3]) is None

    def test_matches_the_moment_oracle_exactly(self) -> None:
        rng = random.Random(7)
        for _ in range(300):
            n = rng.randint(1, 25)
            a = [rng.randint(1, 5) for _ in range(n)]
            b = [rng.randint(1, 5) for _ in range(n)]
            assert quadratic_weighted_kappa(a, b) == oracle_kappa(a, b)

    def test_only_rating_differences_enter_the_observed_term(self) -> None:
        # Same marginals, one adjacent swap vs one distant swap.
        near = quadratic_weighted_kappa([1, 2, 3, 4, 5], [2, 1, 3, 4, 5])
        far = quadratic_weighted_kappa([1, 2, 3, 4, 5], [5, 2, 3, 4, 1])
        assert near == Fraction(9, 10) and far == Fraction(-3, 5)

    @pytest.mark.parametrize(
        ("a", "b", "match"),
        [
            ([1, 2], [1], "differ in length"),
            ([], [], "at least one"),
            ([0], [1], "threat level"),
            ([1], [6], "threat level"),
            ([True], [1], "threat level"),
        ],
    )
    def test_invalid_inputs(self, a: list[Any], b: list[Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            quadratic_weighted_kappa(a, b)


class TestBootstrap:
    @staticmethod
    def _oracle(
        a: list[int], b: list[int], n_boot: int, confidence: float, seed: int
    ) -> tuple[float | None, float | None, int]:
        idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
        values = []
        for row in idx:
            kappa = oracle_kappa([a[i] for i in row], [b[i] for i in row])
            if kappa is not None:
                values.append(float(kappa))
        if not values:
            return None, None, n_boot
        tail = (1 - confidence) / 2
        low, high = np.quantile(values, [tail, 1 - tail])
        return float(low), float(high), n_boot - len(values)

    @pytest.mark.parametrize(
        ("a", "b", "confidence", "seed"),
        [
            (K61_A, K61_B, 0.95, 0),
            (K61_A, K61_B, 0.80, 11),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], 0.95, 3),
            ([1, 1, 1, 2], [1, 1, 1, 2], 0.95, 5),  # some resamples are undefined
        ],
    )
    def test_matches_the_documented_resampling_protocol(
        self, a: list[int], b: list[int], confidence: float, seed: int
    ) -> None:
        low, high, undefined = bootstrap_kappa_interval(
            a, b, n_boot=300, confidence=confidence, seed=seed
        )
        o_low, o_high, o_undefined = self._oracle(a, b, 300, confidence, seed)
        assert undefined == o_undefined
        assert low == pytest.approx(o_low, abs=1e-12)
        assert high == pytest.approx(o_high, abs=1e-12)

    def test_undefined_resamples_are_counted_and_dropped(self) -> None:
        # Identical raters: every defined resample has kappa 1; a resample is
        # undefined exactly when it never draws the single level-2 case.
        idx = np.random.default_rng(5).integers(0, 4, size=(300, 4))
        expected_undefined = int(np.sum(~(idx == 3).any(axis=1)))
        assert expected_undefined > 0
        result = bootstrap_kappa_interval([1, 1, 1, 2], [1, 1, 1, 2], n_boot=300, seed=5)
        assert result == (1.0, 1.0, expected_undefined)

    def test_every_resample_undefined_gives_no_interval(self) -> None:
        assert bootstrap_kappa_interval([2, 2], [2, 2], n_boot=40) == (None, None, 40)

    def test_seeded_and_deterministic(self) -> None:
        first = bootstrap_kappa_interval(K61_A, K61_B, n_boot=200, seed=1)
        assert first == bootstrap_kappa_interval(K61_A, K61_B, n_boot=200, seed=1)
        assert first != bootstrap_kappa_interval(K61_A, K61_B, n_boot=200, seed=2)

    def test_default_protocol(self) -> None:
        assert bootstrap_kappa_interval(K61_A, K61_B) == self._oracle(K61_A, K61_B, 2000, 0.95, 0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_boot": 0}, "n_boot"),
            ({"confidence": 1.0}, "confidence"),
            ({"confidence": 0.0}, "strictly"),
        ],
    )
    def test_invalid_settings(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            bootstrap_kappa_interval([1, 2], [1, 2], **kwargs)


class TestAgreementGate:
    def test_defaults_and_their_sources(self) -> None:
        from omni_mercury_engine.core.centralized_constants import INTEL_MERIT

        MIN_HELDOUT_CASES = INTEL_MERIT.MIN_HELDOUT_CASES

        policy = AgreementPolicy()
        assert policy.kappa_threshold == adj.DEFAULT_KAPPA_THRESHOLD == 0.61
        assert policy.exact_threshold == Fraction(61, 100)
        assert policy.min_pair_cases == adj.DEFAULT_MIN_PAIR_CASES == MIN_HELDOUT_CASES == 30
        assert (policy.n_boot, policy.confidence, policy.seed) == (2000, 0.95, 0)
        assert policy.gate_on == "point"
        assert adj.DEFAULT_HELDOUT_PCT == 25

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kappa_threshold": 1.01},
            {"kappa_threshold": -1.01},
            {"min_pair_cases": 1},
            {"n_boot": 0},
            {"confidence": 1.0},
            {"confidence": 0.0},
            {"gate_on": "median"},
        ],
    )
    def test_meaningless_policies_are_refused(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            AgreementPolicy(**kwargs)

    @pytest.mark.parametrize("threshold", [-1.0, 1.0])
    def test_threshold_bounds_are_inclusive(self, threshold: float) -> None:
        assert AgreementPolicy(kappa_threshold=threshold).kappa_threshold == threshold

    def test_kappa_exactly_at_the_default_threshold_passes(self) -> None:
        # 36 cases (the 12-case vectors three times over) keep kappa at 61/100.
        report = assess_agreement({("alice", "bob"): (K61_A * 3, K61_B * 3)}, AgreementPolicy())
        pair = report.pair(("bob", "alice"))
        assert (pair.n, pair.kappa, pair.gated, pair.passed) == (36, Fraction(61, 100), True, True)
        assert report.passed
        assert report.reason == "every gated rater pair reaches kappa >= 0.61"

    @pytest.mark.parametrize("threshold", [0.6100000000000001, 0.62])
    def test_kappa_just_below_the_threshold_fails(self, threshold: float) -> None:
        policy = AgreementPolicy(kappa_threshold=threshold)
        report = assess_agreement({("alice", "bob"): (K61_A * 3, K61_B * 3)}, policy)
        assert not report.passed and not report.pair(("alice", "bob")).passed
        assert report.reason == f"kappa below {threshold} or undefined for alice/bob"

    @pytest.mark.parametrize(("n", "gated"), [(29, False), (30, True)])
    def test_a_pair_is_gated_from_min_pair_cases(self, n: int, gated: bool) -> None:
        levels = [1 + i % 5 for i in range(n)]
        report = assess_agreement({("alice", "bob"): (levels, levels)}, AgreementPolicy(n_boot=20))
        assert report.pair(("alice", "bob")).gated is gated
        assert report.passed is gated
        if not gated:
            assert report.reason == "no rater pair has co-rated at least 30 cases"

    def test_one_failing_gated_pair_fails_the_gate(self) -> None:
        good = [1 + i % 5 for i in range(30)]
        bad = list(reversed(good))
        report = assess_agreement(
            {("alice", "bob"): (good, good), ("carol", "alice"): (good, bad)},
            AgreementPolicy(n_boot=20),
        )
        assert [p.raters for p in report.pairs] == [("alice", "bob"), ("alice", "carol")]
        assert [p.passed for p in report.pairs] == [True, False]
        assert not report.passed and report.reason.endswith("undefined for alice/carol")

    def test_an_ungated_pair_does_not_decide_the_gate(self) -> None:
        good = [1 + i % 5 for i in range(30)]
        report = assess_agreement(
            {("alice", "bob"): (good, good), ("alice", "carol"): ([1, 5], [5, 1])},
            AgreementPolicy(n_boot=20),
        )
        carol = report.pair(("alice", "carol"))
        assert (carol.gated, carol.passed, carol.kappa) == (False, False, Fraction(-1))
        assert report.passed

    def test_an_ungated_pair_is_never_reported_as_passing(self) -> None:
        # Perfect agreement on too few cases is unmeasured, not passed.
        report = assess_agreement({("alice", "bob"): ([1, 5], [1, 5])}, AgreementPolicy())
        pair = report.pair(("alice", "bob"))
        assert (pair.kappa, pair.gated, pair.passed) == (Fraction(1), False, False)
        assert pair.to_dict()["passed"] is False and not report.passed

    def test_an_undefined_kappa_fails_closed(self) -> None:
        report = assess_agreement({("alice", "bob"): ([3] * 30, [3] * 30)}, AgreementPolicy())
        pair = report.pair(("alice", "bob"))
        assert pair.kappa is None and pair.gated and not pair.passed
        assert not report.passed

    def test_gating_on_the_interval_lower_bound_is_stricter(self) -> None:
        pairs = {("alice", "bob"): (K61_A * 3, K61_B * 3)}
        low = assess_agreement(pairs, AgreementPolicy(n_boot=200)).pair(("alice", "bob")).ci_low
        assert low is not None and low < 0.61
        strict = assess_agreement(pairs, AgreementPolicy(n_boot=200, gate_on="ci_lower"))
        assert not strict.passed
        assert strict.reason == "kappa CI lower bound below 0.61 or undefined for alice/bob"
        at_low = AgreementPolicy(n_boot=200, gate_on="ci_lower", kappa_threshold=low)
        assert assess_agreement(pairs, at_low).passed
        above = AgreementPolicy(
            n_boot=200, gate_on="ci_lower", kappa_threshold=math.nextafter(low, 1.0)
        )
        assert not assess_agreement(pairs, above).passed

    def test_interval_gate_fails_closed_without_an_interval(self) -> None:
        policy = AgreementPolicy(gate_on="ci_lower", kappa_threshold=-1.0, n_boot=20)
        assert not assess_agreement({("a", "b"): ([3] * 30, [3] * 30)}, policy).passed

    def test_report_serialises_exactly(self) -> None:
        report = assess_agreement({("alice", "bob"): (K61_A * 3, K61_B * 3)}, AgreementPolicy())
        low, high, undefined = bootstrap_kappa_interval(K61_A * 3, K61_B * 3)
        assert report.to_dict() == {
            "passed": True,
            "reason": "every gated rater pair reaches kappa >= 0.61",
            "kappa_threshold": 0.61,
            "gate_on": "point",
            "min_pair_cases": 30,
            "confidence": 0.95,
            "n_boot": 2000,
            "seed": 0,
            "pairs": [
                {
                    "raters": ["alice", "bob"],
                    "n": 36,
                    "kappa": 0.61,
                    "kappa_exact": "61/100",
                    "ci": [low, high],
                    "n_undefined_resamples": undefined,
                    "gated": True,
                    "passed": True,
                }
            ],
        }


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_matches_the_hash_rule(self) -> None:
        for i in range(500):
            cid = f"id-{i}"
            for pct in (1, 25, 50, 99):
                assert split_for(cid, pct) == ("heldout" if bucket(cid) < pct else "training")

    def test_bucket_boundary(self) -> None:
        at_24 = next(f"b-{i}" for i in range(10_000) if bucket(f"b-{i}") == 24)
        at_25 = next(f"b-{i}" for i in range(10_000) if bucket(f"b-{i}") == 25)
        assert split_for(at_24, 25) == "heldout" and split_for(at_24, 24) == "training"
        assert split_for(at_25, 25) == "training" and split_for(at_25, 26) == "heldout"
        assert split_for(at_24) == "heldout" and split_for(at_25) == "training"

    def test_case_ids_hash_as_utf8(self) -> None:
        cid = "affaire-é"
        assert split_for(cid, 50) == ("heldout" if bucket(cid) < 50 else "training")

    @pytest.mark.parametrize("pct", [0, 100, True, 25.0, -5])
    def test_percentage_must_be_1_to_99(self, pct: Any) -> None:
        with pytest.raises(ValueError, match=r"\[1, 99\]"):
            split_for("x", pct)
        with pytest.raises(ValueError, match=r"\[1, 99\]"):
            AdjudicationLedger("unused.jsonl", heldout_pct=pct)

    @pytest.mark.parametrize("pct", [1, 99])
    def test_percentage_bounds_are_inclusive(self, pct: int) -> None:
        assert split_for("x", pct) in adj.SPLITS

    def test_the_split_is_recorded_at_first_write_and_the_rule_is_fixed(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "l.jsonl"
        AdjudicationLedger(path, heldout_pct=40).add_case(make_case("case-a"))
        entry = json.loads(path.read_text())["entry"]
        assert (entry["split"], entry["heldout_pct"]) == (split_for("case-a", 40), 40)
        with pytest.raises(ValueError, match="split rule is heldout_pct 40"):
            AdjudicationLedger(path, heldout_pct=25).add_case(make_case("case-b"))
        assert AdjudicationLedger(path).report().heldout_pct == 40


# ---------------------------------------------------------------------------
# Hash chain
# ---------------------------------------------------------------------------


class TestChain:
    def test_append_chains_records_from_genesis(self, tmp_path: Path) -> None:
        ledger = AdjudicationLedger(tmp_path / "l.jsonl")
        first = ledger.add_case(make_case("c"))
        second = ledger.add_outcome(make_outcome("c"))
        assert first["prev_digest"] == GENESIS_DIGEST and first["index"] == 0
        assert second["prev_digest"] == first["record_digest"] and second["index"] == 1
        assert first["schema"] == adj.LEDGER_SCHEMA
        body = {k: v for k, v in first["entry"].items() if k != "digest"}
        assert first["entry"]["digest"] == hashlib.sha3_256(adj._canonical(body)).hexdigest()
        assert ledger.verify() == (True, "valid")
        assert ledger.records() == [first, second]
        assert ledger.head_digest() == second["record_digest"]

    def test_batch_extend_chains_like_single_appends(self, tmp_path: Path) -> None:
        one = AdjudicationLedger(tmp_path / "one.jsonl")
        case, outcome, first, second = case_items("c", [2, 2])
        one.add_case(case)
        one.add_outcome(outcome)
        one.add_rating(first)
        one.add_rating(second)
        batch = AdjudicationLedger(tmp_path / "batch.jsonl")
        batch.extend(case_items("c", [2, 2]))
        assert one.path.read_text() == batch.path.read_text()

    def test_canonical_form_is_the_self_model_ledgers(self) -> None:
        payload = {"b": [1, 2.5, "é"], "a": {"z": None, "y": True}}
        assert adj._canonical(payload) == self_model._canonical(payload)

    @pytest.mark.parametrize("attack", ["edit", "delete", "reorder"])
    def test_any_tampering_breaks_the_chain(self, tmp_path: Path, attack: str) -> None:
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.extend(case_items("c", [2, 2]))
        lines = path.read_text().splitlines()
        if attack == "edit":
            record = json.loads(lines[2])
            record["entry"]["level"] = 5
            lines[2] = json.dumps(record, sort_keys=True)
        elif attack == "delete":
            del lines[1]
        else:
            lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines) + "\n")
        ok, reason = ledger.verify()
        assert not ok and reason.startswith("record ")

    @pytest.mark.parametrize(
        ("forge", "entry", "chain", "reason"),
        [
            # Each forgery recomputes every digest except the one check it must trip.
            (lambda r: r.update(schema="other/v9"), True, True, "unsupported schema 'other/v9'"),
            (lambda r: r.update(index=5), True, True, "index 5 out of order"),
            (
                lambda r: r.update(prev_digest="ab" * 32),
                True,
                True,
                "does not chain to its predecessor",
            ),
            (
                lambda r: r["entry"].update(description="else"),
                True,
                False,
                "contents altered after recording",
            ),
            (
                lambda r: r["entry"].update(description="else"),
                False,
                True,
                "entry digest mismatch",
            ),
        ],
    )
    def test_each_check_catches_the_forgery_only_it_can_see(
        self, tmp_path: Path, forge: Any, entry: bool, chain: bool, reason: str
    ) -> None:
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.extend([make_case("c"), make_outcome("c")])
        lines = path.read_text().splitlines()
        record = json.loads(lines[1])
        forge(record)
        reseal(record, entry=entry, chain=chain)
        lines[1] = json.dumps(record, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
        assert ledger.verify() == (False, f"record 1: {reason}")

    @pytest.mark.parametrize(
        ("line", "reason"),
        [
            ("{not json", "not valid JSON"),
            ('{"entry": {}, "x": NaN}', "not valid JSON"),
            ("[1, 2]", "not a ledger record"),
            ('{"entry": [1]}', "not a ledger record"),
            ('{"schema": "mercury.adjudication.ledger/v1"}', "not a ledger record"),
        ],
    )
    def test_a_corrupted_line_is_a_broken_record_not_a_crash(
        self, tmp_path: Path, line: str, reason: str
    ) -> None:
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"))
        path.write_text(path.read_text() + line + "\n")
        assert ledger.verify() == (False, f"record 1: {reason}")

    @pytest.mark.parametrize(
        "add",
        [
            lambda led: led.add_case(make_case("d")),
            lambda led: led.add_outcome(make_outcome("c")),
            lambda led: led.add_rating(rate("alice", "c", 1)),
            lambda led: led.add_decision(decide("c", 1)),
            lambda led: led.extend([make_case("d")]),
        ],
    )
    def test_a_broken_chain_is_never_extended(self, tmp_path: Path, add: Any) -> None:
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"))
        record = json.loads(path.read_text())
        record["index"] = 7
        path.write_text(json.dumps(record) + "\n")
        before = path.read_text()
        with pytest.raises(ValueError, match="refusing to extend a broken adjudication ledger"):
            add(ledger)
        assert path.read_text() == before

    def test_missing_ledger_is_empty_and_valid(self, tmp_path: Path) -> None:
        ledger = AdjudicationLedger(tmp_path / "absent.jsonl")
        assert ledger.records() == [] and ledger.verify() == (True, "valid")
        assert ledger.head_digest() == GENESIS_DIGEST
        report = ledger.report()
        assert report.exportable and report.cases == () and report.head_digest == GENESIS_DIGEST

    def test_blank_lines_are_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"))
        path.write_text("\n" + path.read_text() + "  \n")
        assert ledger.verify() == (True, "valid") and len(ledger.records()) == 1

    def test_default_path_honours_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MERCURY_ADJUDICATION_PATH", str(tmp_path / "x.jsonl"))
        assert AdjudicationLedger().path == tmp_path / "x.jsonl"
        monkeypatch.delenv("MERCURY_ADJUDICATION_PATH")
        assert adj.default_ledger_path().name == "adjudication.jsonl"


class TestSignatures:
    def test_signed_ledger_verifies_only_against_the_pinned_key(self, tmp_path: Path) -> None:
        provider = mldsa()
        pair, other = provider.generate_keypair(), provider.generate_keypair()
        ledger = AdjudicationLedger(tmp_path / "l.jsonl")
        ledger.extend(case_items("c", [2]), signing_keypair=(pair.public_key, pair.secret_key))
        assert ledger.verify(pair.public_key.hex()) == (True, "valid")
        assert ledger.verify(other.public_key.hex()) == (
            False,
            "record 0: not signed by the pinned key",
        )

    def test_key_comparison_ignores_case_and_surrounding_whitespace(self, tmp_path: Path) -> None:
        provider = mldsa()
        pair = provider.generate_keypair()
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"), signing_keypair=(pair.public_key, pair.secret_key))
        assert ledger.verify(f"  {pair.public_key.hex().upper()}\n") == (True, "valid")
        record = json.loads(path.read_text())
        record["signature"]["public_key_hex"] = f" {pair.public_key.hex().upper()} "
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        assert ledger.verify(pair.public_key.hex()) == (True, "valid")

    @pytest.mark.parametrize(
        ("damage", "reason"),
        [
            (lambda r: r.pop("signature"), "not signed by the pinned key"),
            (lambda r: r.update(signature="sig"), "not signed by the pinned key"),
            (lambda r: r["signature"].update(value_hex="zz"), "signature could not be checked"),
            (lambda r: r["signature"].pop("value_hex"), "signature could not be checked"),
            (lambda r: r["signature"].update(value_hex=5), "signature could not be checked"),
            (lambda r: r["signature"].update(value_hex="00" * 3309), "signature does not verify"),
        ],
    )
    def test_a_damaged_signature_fails_closed_with_its_reason(
        self, tmp_path: Path, damage: Any, reason: str
    ) -> None:
        provider = mldsa()
        pair = provider.generate_keypair()
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"), signing_keypair=(pair.public_key, pair.secret_key))
        record = json.loads(path.read_text())
        damage(record)
        path.write_text(json.dumps(record, sort_keys=True) + "\n")
        ok, why = ledger.verify(pair.public_key.hex())
        assert not ok and why.startswith(f"record 0: {reason}")
        assert AdjudicationLedger(path).verify() == (True, "valid")  # unpinned: chain only

    def test_a_signature_from_another_record_does_not_verify(self, tmp_path: Path) -> None:
        provider = mldsa()
        pair = provider.generate_keypair()
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        keys = (pair.public_key, pair.secret_key)
        ledger.extend([make_case("c"), make_outcome("c")], signing_keypair=keys)
        first, second = (json.loads(line) for line in path.read_text().splitlines())
        second["signature"] = first["signature"]
        path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        assert ledger.verify(pair.public_key.hex()) == (
            False,
            "record 1: signature does not verify",
        )

    def test_a_malformed_pinned_key_cannot_be_checked(self, tmp_path: Path) -> None:
        provider = mldsa()
        pair = provider.generate_keypair()
        path = tmp_path / "l.jsonl"
        ledger = AdjudicationLedger(path)
        ledger.add_case(make_case("c"), signing_keypair=(pair.public_key, pair.secret_key))
        record = json.loads(path.read_text())
        record["signature"]["public_key_hex"] = "abcd"
        path.write_text(json.dumps(record) + "\n")
        ok, why = ledger.verify("ABCD")
        assert not ok and why.startswith("record 0: signature could not be checked")

    def test_a_verifier_exception_fails_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = mldsa()
        pair = provider.generate_keypair()
        ledger = AdjudicationLedger(tmp_path / "l.jsonl")
        ledger.add_case(make_case("c"), signing_keypair=(pair.public_key, pair.secret_key))
        from omni_mercury_engine.security import crypto_api

        def boom(*_: Any) -> bool:
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(crypto_api.MLDSAProvider, "verify", boom)
        assert ledger.verify(pair.public_key.hex()) == (
            False,
            "record 0: signature could not be checked: backend unavailable",
        )

    def test_a_pinned_ledger_only_accepts_records_signed_by_its_key(self, tmp_path: Path) -> None:
        provider = mldsa()
        pair, other = provider.generate_keypair(), provider.generate_keypair()
        path = tmp_path / "l.jsonl"
        pinned = AdjudicationLedger(path, expected_public_key_hex=pair.public_key.hex().upper())
        with pytest.raises(ValueError, match="pinned to a key"):
            pinned.add_case(make_case("c"))
        with pytest.raises(ValueError, match="pinned to a key"):
            pinned.add_case(make_case("c"), signing_keypair=(other.public_key, other.secret_key))
        assert not path.exists()
        pinned.add_case(make_case("c"), signing_keypair=(pair.public_key, pair.secret_key))
        assert pinned.verify() == (True, "valid")
        AdjudicationLedger(path).add_outcome(make_outcome("c"))  # an unpinned writer
        assert pinned.verify() == (False, "record 1: not signed by the pinned key")
        with pytest.raises(ExportRefused, match="not signed by the pinned key"):
            pinned.training_cases()


# ---------------------------------------------------------------------------
# Protocol (what the API refuses)
# ---------------------------------------------------------------------------


def _ledger(
    tmp_path: Path, items: list[Any], policy: AgreementPolicy = SMALL
) -> AdjudicationLedger:
    ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=policy)
    ledger.extend(items)
    return ledger


def with_fill(items: list[Any]) -> list[Any]:
    """Add two alice/bob consensus cases so the pair is gated and its kappa defined."""
    return items + case_items("fill-1", [1, 1]) + case_items("fill-2", [5, 5])


def label_of(ledger: AdjudicationLedger, case_id: str) -> adj.CaseLabel:
    return next(c for c in ledger.report().cases if c.case_id == case_id)


class TestProtocol:
    @pytest.mark.parametrize(
        ("setup", "item", "match"),
        [
            ([make_case("c")], make_case("c"), "case 'c' is already recorded"),
            ([], make_outcome("c"), "outcome for case 'c', which is not recorded before it"),
            (
                [make_case("c"), make_outcome("c")],
                make_outcome("c"),
                "case 'c' already has an outcome",
            ),
            (
                [make_case("c")],
                rate("alice", "c", 2),
                "rating of case 'c' before its outcome: raters rate outcomes",
            ),
            (case_items("c", [2]), rate("alice", "c", 3), "rater 'alice' already rated case 'c'"),
            (
                case_items("c", []),
                rate("alice", "c", 2, at=T0 + DAY - timedelta(microseconds=1)),
                "dated before its outcome was observed",
            ),
            (
                case_items("c", [2, 2]),
                rate("carol", "c", 2),
                "the first two ratings agree, so a third is not taken",
            ),
            (case_items("c", [1, 3, 5]), rate("erin", "c", 2), "already has three ratings"),
            (
                case_items("c", [2, 3], decision=2),
                rate("carol", "c", 2),
                "rating of case 'c' after its adjudicator decision",
            ),
            (case_items("c", [2, 2]), decide("c", 2), "ratings [2, 2] admit no adjudicator"),
            (case_items("c", [2, 4]), decide("c", 3), "ratings [2, 4] admit no adjudicator"),
            (case_items("c", [2, 4, 4]), decide("c", 4), "ratings [2, 4, 4] admit no"),
            (case_items("c", [2]), decide("c", 2), "ratings [2] admit no adjudicator"),
            (case_items("c", [2, 3]), decide("c", 2, who="bob"), "adjudicator 'bob' rated case"),
            (case_items("c", [2, 3]), decide("c", 4), "decision 4 lies outside the ratings' range"),
            (case_items("c", [2, 3]), decide("c", 1), "decision 1 lies outside the ratings' range"),
            (
                case_items("c", [2, 3]),
                decide("c", 2, at=T0 + 2 * DAY - timedelta(microseconds=1)),
                "decision is dated before a rating it resolves",
            ),
            (
                case_items("c", [2, 3], decision=3),
                decide("c", 3, who="erin"),
                "already has an adjudicator decision",
            ),
        ],
    )
    def test_refusals(self, tmp_path: Path, setup: list[Any], item: Any, match: str) -> None:
        ledger = _ledger(tmp_path, setup)
        before = ledger.path.read_text() if ledger.path.exists() else ""
        with pytest.raises(ValueError, match=match.replace("[", r"\[").replace("]", r"\]")):
            ledger.extend([item])
        after = ledger.path.read_text() if ledger.path.exists() else ""
        assert after == before

    @pytest.mark.parametrize(
        ("levels", "item"),
        [
            ([2, 3], rate("carol", "c", 3)),  # one-level gap: a third rating
            ([2, 3], decide("c", 3)),  # ... or a decision
            ([2, 4], rate("carol", "c", 3)),  # wider gap: a third rating
            ([1, 4, 2], decide("c", 4)),  # no majority: a decision within range
            ([1, 4, 2], decide("c", 1)),
        ],
    )
    def test_allowed_resolutions(self, tmp_path: Path, levels: list[int], item: Any) -> None:
        ledger = _ledger(tmp_path, case_items("c", levels))
        ledger.extend([item])
        assert ledger.verify() == (True, "valid")
        assert ledger.report().protocol_problems == ()

    def test_decisions_and_ratings_may_share_a_timestamp_with_their_predecessor(
        self, tmp_path: Path
    ) -> None:
        items = [make_case("c"), make_outcome("c")]
        items += [rate("alice", "c", 2, at=T0 + DAY), rate("bob", "c", 3, at=T0 + DAY)]
        items.append(decide("c", 3, at=T0 + DAY))
        assert _ledger(tmp_path, items).report().protocol_problems == ()

    def test_a_batch_is_all_or_nothing(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, [make_case("a")])
        before = ledger.path.read_text()
        with pytest.raises(ValueError, match="already recorded"):
            ledger.extend([make_case("b"), make_outcome("b"), make_case("a")])
        assert ledger.path.read_text() == before

    def test_append_returns_the_record_written(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, case_items("c", [2, 3]))
        record = ledger.add_decision(decide("c", 3))
        assert ledger.records()[-1] == record
        assert record["entry"]["kind"] == "decision" and record["index"] == 4


class TestContentForgery:
    """A keyless forger re-seals the whole chain; the protocol replay still refuses it."""

    BASE = ("case-a", "case-b")

    def _base(self, tmp_path: Path) -> AdjudicationLedger:
        items: list[Any] = []
        for cid in self.BASE:
            items += case_items(cid, [2, 2])
        return _ledger(tmp_path, items)

    @pytest.mark.parametrize(
        ("forge", "problem"),
        [
            (
                lambda rs: rs[0]["entry"].update(
                    split="heldout" if split_for("case-a") == "training" else "training"
                ),
                f"record 0: case 'case-a' records split "
                f"'{'heldout' if split_for('case-a') == 'training' else 'training'}', "
                f"but its hash rule gives '{split_for('case-a')}'",
            ),
            (
                lambda rs: rs[4]["entry"].update(heldout_pct=30, split=split_for("case-b", 30)),
                "record 4: case 'case-b' records heldout_pct 30, but this ledger's split rule "
                "is heldout_pct 25",
            ),
            (
                lambda rs: rs[0]["entry"].update(heldout_pct=0),
                "record 0: heldout_pct must be an integer in [1, 99], got 0",
            ),
            (
                lambda rs: rs[0]["entry"].update(created_at="2025-12-31T23:59:59+00:00"),
                "record 0: created_at precedes latest_report_at",
            ),
            (
                lambda rs: rs[2]["entry"].update(label=5),
                "record 2: malformed rating entry",
            ),
            (lambda rs: rs[2]["entry"].update(kind="verdict"), "record 2: unknown entry kind"),
            (
                lambda rs: rs[2]["entry"].update(level=7),
                "record 2: level must be an integer threat level 1-5, got 7",
            ),
            (
                lambda rs: rs[3]["entry"].update(rater_id="alice"),
                "record 3: rater 'alice' already rated case 'case-a'",
            ),
            (
                lambda rs: rs[2]["entry"].update(rated_at="2026-01-01T00:00:00+00:00"),
                "record 2: rating of case 'case-a' is dated before its outcome was observed",
            ),
            (
                lambda rs: rs[2]["entry"].update(rated_at="2026-01-03T00:00:00"),
                "record 2: rated_at must be a timezone-aware datetime",
            ),
            (
                lambda rs: rs[2]["entry"].update(rated_at="yesterday"),
                "record 2: rated_at 'yesterday' is not an ISO 8601 timestamp",
            ),
            (
                lambda rs: rs[2]["entry"].update(rated_at=20260103),
                "record 2: rated_at must be an ISO 8601 timestamp string",
            ),
            (
                lambda rs: rs.insert(1, rs.pop(2)),
                "record 1: rating of case 'case-a' before its outcome: raters rate outcomes",
            ),
            (
                lambda rs: rs.append(json.loads(json.dumps(rs[0]))),
                "record 8: case 'case-a' is already recorded",
            ),
            (
                lambda rs: rs[1]["entry"].update(outcome_level=3),
                "record 1: outcome_level and outcome_scale go together",
            ),
            (
                lambda rs: rs.append(
                    {
                        **json.loads(json.dumps(rs[3])),
                        "entry": {**rs[3]["entry"], "rater_id": "carol"},
                    }
                ),
                "record 8: case 'case-a': the first two ratings agree, so a third is not taken",
            ),
        ],
    )
    def test_only_the_protocol_replay_catches_it(
        self, tmp_path: Path, forge: Callable[[list[dict[str, Any]]], None], problem: str
    ) -> None:
        ledger = self._base(tmp_path)
        rewrite(ledger.path, forge)
        assert ledger.verify() == (True, "valid")  # the chain is consistent
        report = ledger.report()
        assert report.protocol_problems[0].startswith(problem)
        assert not report.exportable
        with pytest.raises(ExportRefused, match="content breaks the protocol: record"):
            ledger.heldout_cases()
        with pytest.raises(ExportRefused, match="content breaks the protocol"):
            ledger.training_cases()
        with pytest.raises(ValueError, match="content breaks the protocol"):
            ledger.add_case(make_case("case-z"))
        stats = ledger.stats()
        assert stats["protocol_problems"][0].startswith(problem)
        assert all(
            "the ledger's content breaks the protocol" in e["reasons"] for e in stats["exclusions"]
        )

    def test_the_untouched_base_is_sound(self, tmp_path: Path) -> None:
        ledger = self._base(tmp_path)
        rewrite(ledger.path, lambda rs: None)
        assert ledger.report().protocol_problems == ()


# ---------------------------------------------------------------------------
# Eligibility: non-circularity, time order, adjudication, agreement
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_consensus_case_is_exported_with_its_method(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, with_fill(case_items("c", [4, 4])))
        label = label_of(ledger, "c")
        assert label.eligible and label.level == 4 and label.method == "consensus"
        assert label.adjudication == adj.Adjudication("c", 4, "consensus", (4, 4))

    @pytest.mark.parametrize("shared", ["Vanguard", " VANGUARD ", "vanguard"])
    def test_circular_sources_are_excluded(self, tmp_path: Path, shared: str) -> None:
        items = case_items("c", [3, 3], report_source_ids=("Reuters", shared))
        ledger = _ledger(tmp_path, with_fill(items))
        assert ledger.exclusions() == [
            {
                "case_id": "c",
                "split": split_for("c"),
                "reasons": ["circular: outcome sources ['vanguard'] also supplied the reports"],
            }
        ]
        labels = ledger.heldout_cases() + ledger.training_cases()
        assert sorted(level for _, level in labels) == [1, 5]  # only the fill cases

    @pytest.mark.parametrize(("offset", "eligible"), [(0, False), (1, True)])
    def test_the_outcome_must_be_strictly_after_the_latest_report(
        self, tmp_path: Path, offset: int, eligible: bool
    ) -> None:
        observed = T0 + DAY
        latest = observed - timedelta(microseconds=offset)
        items = [
            make_case("c", latest_report_at=latest, created_at=latest),
            make_outcome("c", observed_at=observed),
            rate("alice", "c", 3),
            rate("bob", "c", 3),
        ]
        label = label_of(_ledger(tmp_path, with_fill(items)), "c")
        assert label.eligible is eligible
        if not eligible:
            assert label.reasons == (
                "outcome at 2026-01-02T00:00:00+00:00 is not after the latest report at "
                "2026-01-02T00:00:00+00:00",
            )

    def test_every_reason_is_reported(self, tmp_path: Path) -> None:
        items = [
            make_case(
                "c", report_source_ids=("Vanguard",), latest_report_at=T0 + DAY, created_at=T0 + DAY
            ),
            make_outcome("c"),
            rate("alice", "c", 2),
            rate("bob", "c", 3),
        ]
        label = label_of(_ledger(tmp_path, with_fill(items), PERMISSIVE), "c")
        assert label.reasons == (
            "circular: outcome sources ['vanguard'] also supplied the reports",
            "outcome at 2026-01-02T00:00:00+00:00 is not after the latest report at "
            "2026-01-02T00:00:00+00:00",
            "one-level disagreement awaiting an adjudicator decision or a third rating",
        )
        assert label.adjudication is None and label.level is None and label.method is None

    @pytest.mark.parametrize(
        ("items", "reason"),
        [
            ([make_case("c")], "no outcome recorded"),
            (case_items("c", []), "0 of the 2 required ratings recorded"),
            (case_items("c", [2]), "1 of the 2 required ratings recorded"),
            (case_items("c", [1, 3]), "2-level disagreement awaiting a third rating"),
            (case_items("c", [1, 4, 2]), "three ratings without a majority"),
        ],
    )
    def test_unresolved_cases_are_excluded(
        self, tmp_path: Path, items: list[Any], reason: str
    ) -> None:
        ledger = _ledger(tmp_path, with_fill(items), PERMISSIVE)
        exclusions = {e["case_id"]: e["reasons"] for e in ledger.exclusions()}
        assert list(exclusions) == ["c"]
        assert any(r.startswith(reason) for r in exclusions["c"])

    @pytest.mark.parametrize(
        ("levels", "decision", "expected"),
        [
            ([2, 3], 3, (3, "adjudicator_decision")),
            ([2, 4, 4], None, (4, "majority_of_three")),
            ([1, 4, 2], 4, (4, "adjudicator_decision")),
        ],
    )
    def test_resolved_disagreements_carry_their_method(
        self,
        tmp_path: Path,
        levels: list[int],
        decision: int | None,
        expected: tuple[int, str],
    ) -> None:
        ledger = _ledger(
            tmp_path, with_fill(case_items("c", levels, decision=decision)), PERMISSIVE
        )
        label = label_of(ledger, "c")
        assert (label.level, label.method) == expected and label.eligible

    def test_kappa_counts_unresolved_disagreements_and_ignores_third_ratings(
        self, tmp_path: Path
    ) -> None:
        items = case_items("c1", [1, 1]) + case_items("c2", [2, 2])
        items += case_items("c3", [3, 4]) + case_items("c4", [5, 3])
        ledger = _ledger(tmp_path, items)
        pair = ledger.report().agreement.pair(("alice", "bob"))  # type: ignore[union-attr]
        assert (pair.n, pair.kappa) == (4, oracle_kappa([1, 2, 3, 5], [1, 2, 4, 3]))
        ledger.add_rating(rate("carol", "c4", 3))
        agreement = ledger.report().agreement
        assert agreement is not None and [p.raters for p in agreement.pairs] == [("alice", "bob")]
        assert agreement.pair(("alice", "bob")).kappa == pair.kappa

    def test_pairs_are_oriented_by_rater_not_by_submission_order(self, tmp_path: Path) -> None:
        items = case_items("c1", [1, 2]) + case_items("c2", [3, 3])
        items += case_items("c3", [5, 4], raters=("bob", "alice"))
        ledger = _ledger(tmp_path, items)
        pair = ledger.report().agreement.pair(("alice", "bob"))  # type: ignore[union-attr]
        assert pair.kappa == oracle_kappa([1, 3, 4], [2, 3, 5])

    def test_cases_of_an_unmeasured_pair_are_excluded(self, tmp_path: Path) -> None:
        policy = AgreementPolicy(min_pair_cases=3, n_boot=20)
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=policy)
        items: list[Any] = []
        for i, level in enumerate([1, 3, 5]):
            items += case_items(f"ab-{i}", [level, level])
        items += case_items("ac-0", [2, 2], raters=("alice", "carol"))
        ledger.extend(items)
        assert ledger.exclusions() == [
            {
                "case_id": "ac-0",
                "split": split_for("ac-0"),
                "reasons": [
                    "rater pair alice/carol co-rated 1 < 3 cases: their agreement is unmeasured"
                ],
            }
        ]
        report = ledger.report()
        assert report.agreement is not None and report.agreement.passed
        assert sum(c.eligible for c in report.cases) == 3

    def test_a_failed_gate_excludes_every_case(self, tmp_path: Path) -> None:
        items: list[Any] = []
        for i, (a, b) in enumerate(zip([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])):
            items += case_items(f"c{i}", [a, b])
        items += case_items("c-agree", [3, 3])
        ledger = _ledger(tmp_path, items)
        reasons = {e["case_id"]: e["reasons"] for e in ledger.exclusions()}
        assert reasons["c-agree"] == [
            "agreement gate failed: kappa below 0.61 or undefined for alice/bob"
        ]
        assert ledger.heldout_cases() == [] and ledger.training_cases() == []


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_splits_return_exact_labels_in_ledger_order(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        heldout = [c for c in order if bucket(c) < 25]
        training = [c for c in order if bucket(c) >= 25]
        assert (len(heldout), len(training)) == (30, 10)
        expected_h = [(reports_for(c), level_of(order, c)) for c in heldout]
        expected_t = [(reports_for(c), level_of(order, c)) for c in training]
        assert ledger.heldout_cases() == expected_h
        assert ledger.training_cases() == expected_t
        assert ledger.export_heldout_for_merit_gate() == expected_h
        assert ledger.exclusions() == []

    def test_returned_reports_are_copies(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, case_items("c1", [1, 1]) + case_items("c2", [4, 4]))
        split = split_for("c1")
        cases = ledger.heldout_cases() if split == "heldout" else ledger.training_cases()
        cases[0][0]["open_source"]["threat_score"] = 99.0
        again = ledger.heldout_cases() if split == "heldout" else ledger.training_cases()
        assert again[0][0] == reports_for("c1")

    def test_a_reports_labels_do_not_alias_its_cases(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, case_items("c1", [1, 1]) + case_items("c2", [4, 4]))
        report = ledger.report()
        split = split_for("c1")
        report.labels(split)[0][0]["open_source"]["threat_score"] = 99.0
        assert report.labels(split)[0][0] == reports_for("c1")
        assert label_of(ledger, "c1").intel_reports == reports_for("c1")

    def test_the_merit_gate_minimum_is_the_fusion_modules(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        assert len(ledger.export_heldout_for_merit_gate()) == 30
        # Make one held-out case circular: 29 remain, one short of the minimum.
        victim = next(c for c in order if bucket(c) < 25)
        rewrite(
            ledger.path,
            lambda rs: next(
                r for r in rs if r["entry"]["kind"] == "outcome" and r["entry"]["case_id"] == victim
            )["entry"].update(outcome_source_ids=["Reuters"]),
        )
        with pytest.raises(
            ExportRefused,
            match="29 distinct eligible held-out cases; the merit gate needs at least 30",
        ):
            ledger.export_heldout_for_merit_gate()
        assert len(ledger.heldout_cases()) == 29  # listing is not the merit-gate export

    def test_without_torch_the_shared_constant_applies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        from omni_mercury_engine.core.centralized_constants import INTEL_MERIT

        real_import = builtins.__import__

        def no_fusion(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "omni_mercury_engine.security.intelligence_fusion":
                raise ImportError("torch unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_fusion)
        assert adj._min_heldout_cases() == INTEL_MERIT.MIN_HELDOUT_CASES

    def test_the_minimum_is_read_from_intelligence_fusion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pytest.importorskip("torch")
        from omni_mercury_engine.security import intelligence_fusion

        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        monkeypatch.setattr(intelligence_fusion, "MIN_HELDOUT_CASES", 31)
        with pytest.raises(ExportRefused, match="needs at least 31"):
            ledger.export_heldout_for_merit_gate()

    def test_a_failed_gate_refuses_the_merit_export(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(
            tmp_path / "l.jsonl",
            heldout=30,
            training=10,
            policy=AgreementPolicy(n_boot=20, min_pair_cases=41),
        )
        with pytest.raises(
            ExportRefused,
            match="agreement gate failed: no rater pair has " "co-rated at least 41 cases",
        ):
            ledger.export_heldout_for_merit_gate()
        with pytest.raises(ExportRefused, match="agreement gate failed"):
            ledger.export("training", tmp_path / "t.json")

    def test_a_broken_chain_refuses_every_export(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        lines = ledger.path.read_text().splitlines()
        ledger.path.write_text("\n".join(lines[1:]) + "\n")
        for call in (
            ledger.heldout_cases,
            ledger.training_cases,
            ledger.export_heldout_for_merit_gate,
            ledger.pending,
            lambda: ledger.rater_view("alice", "x"),
        ):
            with pytest.raises(ExportRefused, match="fails verification: record 0"):
                call()
        assert ledger.stats() == {
            "ledger": str(ledger.path),
            "valid": False,
            "reason": "record 0: index 1 out of order",
            "records": len(lines) - 1,
            "protocol_problems": [],
            "head_digest": GENESIS_DIGEST,
        }

    def test_an_export_file_holds_only_eligible_cases(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        victim = next(c for c in order if bucket(c) >= 25)
        make_circular(ledger.path, victim)
        document = ledger.export("training", tmp_path / "t.json")
        exported = [c["case_id"] for c in document["cases"]]
        assert exported == [c for c in order if bucket(c) >= 25 and c != victim]
        assert all(c["label"] in adj.LEVELS for c in document["cases"])

    def test_export_file_round_trips(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "out" / "heldout.json"
        document = ledger.export("heldout", out)
        assert json.loads(out.read_text()) == document
        assert document["schema"] == adj.EXPORT_SCHEMA and document["split"] == "heldout"
        assert document["ledger_head_digest"] == ledger.head_digest()
        assert document["agreement"]["passed"] is True
        first = document["cases"][0]
        assert first["method"] == "consensus"
        assert first["case_entry_digest"] == next(
            r["entry"]["digest"]
            for r in ledger.records()
            if r["entry"]["kind"] == "case" and r["entry"]["case_id"] == first["case_id"]
        )
        assert load_export(out) == ledger.heldout_cases()
        training = ledger.export("training", tmp_path / "training.json")
        assert len(training["cases"]) == 10  # no minimum for the training split
        with pytest.raises(ValueError, match="split must be one of"):
            ledger.export("validation", tmp_path / "v.json")

    @pytest.mark.parametrize(
        ("document", "match"),
        [
            ({"schema": "other"}, "not an adjudication export"),
            ([], "not an adjudication export"),
            ({"schema": adj.EXPORT_SCHEMA, "split": "heldout"}, "no case list"),
            ({"schema": adj.EXPORT_SCHEMA, "cases": []}, "'heldout' was expected"),
            ({"schema": adj.EXPORT_SCHEMA, "split": "heldout", "cases": [1]}, "not an object"),
            (
                {
                    "schema": adj.EXPORT_SCHEMA,
                    "split": "heldout",
                    "cases": [{"intel_reports": {"x": {}}, "label": 0}],
                },
                "threat level",
            ),
            (
                {
                    "schema": adj.EXPORT_SCHEMA,
                    "split": "heldout",
                    "cases": [{"intel_reports": {"x": {}}, "label": True}],
                },
                "threat level",
            ),
            (
                {
                    "schema": adj.EXPORT_SCHEMA,
                    "split": "heldout",
                    "cases": [{"intel_reports": [], "label": 3}],
                },
                "intel_reports",
            ),
        ],
    )
    def test_load_export_rejects_malformed_files(
        self, tmp_path: Path, document: Any, match: str
    ) -> None:
        path = tmp_path / "e.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match=match):
            load_export(path)

    def test_the_heldout_export_satisfies_load_trained_weights(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")

        from omni_mercury_engine.security.intelligence_fusion import (
            AllSourceFusionNetwork,
            IntelligenceFusionEngine,
        )

        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        torch.manual_seed(0)
        checkpoint = tmp_path / "candidate.pt"
        torch.save(AllSourceFusionNetwork(input_dim=128).state_dict(), checkpoint)
        engine = IntelligenceFusionEngine()
        heldout = ledger.export_heldout_for_merit_gate()
        try:
            merit = engine.load_trained_weights(checkpoint, heldout)
        except RuntimeError as exc:  # random weights are refused on merit, not on format
            assert "refused" in str(exc) and "n=30" in str(exc)
        else:
            assert merit.n_cases == 30


# ---------------------------------------------------------------------------
# Blind rating
# ---------------------------------------------------------------------------


class TestBlindRating:
    def test_a_rater_sees_the_outcome_but_never_the_reports_or_others_first(
        self, tmp_path: Path
    ) -> None:
        items = [
            make_case("c"),
            make_outcome("c", outcome_level=4, outcome_scale="scale/v1"),
            rate("alice", "c", 2),
            rate("bob", "c", 4),
        ]
        ledger = _ledger(tmp_path, items)
        before = ledger.rater_view("carol", "c")
        assert before == {
            "case_id": "c",
            "outcome": {
                "observed_at": "2026-01-02T00:00:00+00:00",
                "description": "what happened in c",
            },
            "own_rating": None,
            "other_ratings": None,
        }
        dumped = json.dumps(before)
        assert "open_source" not in dumped and "outcome_level" not in dumped
        ledger.add_rating(rate("carol", "c", 3))
        after = ledger.rater_view("carol", "c")
        assert after["own_rating"] == {
            "rater_id": "carol",
            "level": 3,
            "rationale": "carol judged 3",
            "rated_at": "2026-01-03T00:00:00+00:00",
        }
        assert [r["rater_id"] for r in after["other_ratings"]] == ["alice", "bob"]
        assert [r["rater_id"] for r in ledger.rater_view("alice", "c")["other_ratings"]] == [
            "bob",
            "carol",
        ]

    def test_before_an_outcome_there_is_nothing_to_rate(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path, [make_case("c")])
        assert ledger.rater_view("alice", "c")["outcome"] is None
        with pytest.raises(KeyError):
            ledger.rater_view("alice", "missing")

    def test_pending_queues(self, tmp_path: Path) -> None:
        items = [make_case("no-outcome")]
        items += case_items("one-rating", [3])
        items += case_items("gap-1", [2, 3])
        items += case_items("gap-2", [1, 3])
        items += case_items("three-way", [1, 4, 2])
        items += case_items("done", [4, 4])
        items += case_items("decided", [2, 3], decision=2)
        assert _ledger(tmp_path, items).pending() == {
            "awaiting_outcome": ["no-outcome"],
            "awaiting_ratings": ["one-rating"],
            "awaiting_third_rating": ["gap-1", "gap-2"],
            "awaiting_decision": ["gap-1", "three-way"],
        }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class TestStats:
    def test_counts_per_split_and_level_methods_and_exclusions(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        stats = ledger.stats()
        heldout = [c for c in order if bucket(c) < 25]
        per_level = {str(level): 0 for level in adj.LEVELS}
        for cid in heldout:
            per_level[str(level_of(order, cid))] += 1
        assert stats["splits"]["heldout"] == {
            "cases": 30,
            "eligible": 30,
            "excluded": 0,
            "eligible_per_level": per_level,
        }
        assert stats["splits"]["training"]["cases"] == 10
        assert stats["methods"] == {
            "consensus": 40,
            "majority_of_three": 0,
            "adjudicator_decision": 0,
        }
        assert stats["split_rule"]["heldout_pct"] == 25
        assert stats["agreement"]["pairs"][0]["kappa"] == 1.0
        assert stats["min_heldout_cases"] == 30 and stats["heldout_ready"] is True
        assert stats["prospective_eligible"] == 40  # created T0, outcomes from T0 + 1 day
        assert stats["outcome_scale_agreement"] == {"n": 0, "kappa": None}
        assert stats["exclusions"] == [] and stats["valid"] is True
        assert stats["records"] == 160

    def test_per_level_counts_cover_eligible_cases_only(self, tmp_path: Path) -> None:
        ledger, order = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        victim = next(c for c in order if bucket(c) < 25)
        make_circular(ledger.path, victim)
        stats = ledger.stats()
        per_level = {str(level): 0 for level in adj.LEVELS}
        for cid in order:
            if bucket(cid) < 25 and cid != victim:
                per_level[str(level_of(order, cid))] += 1
        assert stats["splits"]["heldout"] == {
            "cases": 30,
            "eligible": 29,
            "excluded": 1,
            "eligible_per_level": per_level,
        }
        assert stats["exclusions"] == [
            {
                "case_id": victim,
                "split": "heldout",
                "reasons": ["circular: outcome sources ['reuters'] also supplied the reports"],
            }
        ]
        assert stats["heldout_ready"] is False

    def test_heldout_readiness_needs_the_minimum(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=29, training=10)
        assert ledger.stats()["heldout_ready"] is False

    def test_retrospective_cases_are_counted_apart(self, tmp_path: Path) -> None:
        items = case_items("pro", [2, 2]) + case_items("retro", [4, 4], created_at=T0 + DAY)
        assert _ledger(tmp_path, items).stats()["prospective_eligible"] == 1

    def test_documented_scale_agreement(self, tmp_path: Path) -> None:
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        for cid, label, scale in (("a", 1, 1), ("b", 3, 2), ("c", 5, 5)):
            ledger.extend(
                [
                    make_case(cid),
                    make_outcome(cid, outcome_level=scale, outcome_scale="s/v1"),
                    rate("alice", cid, label),
                    rate("bob", cid, label),
                ]
            )
        stats = ledger.stats()["outcome_scale_agreement"]
        expected = oracle_kappa([1, 3, 5], [1, 2, 5])
        assert expected is not None
        assert stats == {"n": 3, "kappa": float(expected)}


# ---------------------------------------------------------------------------
# UCDP GED outcomes
# ---------------------------------------------------------------------------

#: Two rows of UCDP GED 26.1 (GEDEvent_v26_1.csv), CC BY 4.0. Source: Uppsala
#: Conflict Data Program, https://ucdp.uu.se/downloads/; Sundberg & Melander
#: (2013), Journal of Peace Research 50(4). Event 1568 predates source_office.
UCDP_CSV = (
    "id,relid,year,active_year,code_status,type_of_violence,conflict_dset_id,conflict_new_id,"
    "conflict_name,dyad_dset_id,dyad_new_id,dyad_name,side_a_dset_id,side_a_new_id,side_a,"
    "side_b_dset_id,side_b_new_id,side_b,number_of_sources,source_article,source_office,"
    "source_date,source_headline,source_original,where_prec,where_coordinates,"
    "where_description,adm_1,adm_2,latitude,longitude,geom_wkt,priogrid_gid,country,"
    "country_id,region,event_clarity,date_prec,date_start,date_end,deaths_a,deaths_b,"
    "deaths_civilians,deaths_unknown,best,high,low,gwnoa,gwnob\n"
    "577966,AFG-2025-3-973-142,2025,true,Clear,3,234,506,IS - Civilians,234,973,"
    'IS - Civilians,234,234,IS,9999,1,Civilians,1,"""Vanguard,2025-04-29,Again, two soldiers '
    'killed in landmine explosion on Borno road""",Vanguard,2025-04-29,"Again, two soldiers '
    'killed in landmine explosion on Borno road","",2,Mulgai village,"Mulgoi area, Damboa '
    'LGA, Borno State",Borno state,Damboa lga,11.221490,13.121480,POINT (13.12148 11.22149),'
    "145827,Nigeria,475,Africa,1,1,2025-04-25 00:00:00.000,2025-04-25 00:00:00.000,0,0,6,0,"
    "6,6,6,,\n"
    "1568,ALG-1992-1-1-6,1992,true,Clear,1,386,386,Algeria: Government,828,828,Government of "
    "Algeria - AIS,109,109,Government of Algeria,537,537,AIS,-1,Reuters 3/19/1992 ALGERIAN "
    'SECURITY WARNS OF KILLING CAMPAIGN.,"","","",,1,Medea town,"Medea town, Medea district, '
    'Medea province",Medea province,Medea commune,36.264169,2.753926,POINT (2.753926 '
    "36.264169),181806,Algeria,615,Africa,1,1,1992-03-17 00:00:00.000,1992-03-17 "
    "00:00:00.000,1,0,1,0,2,2,2,615,\n"
)


def ucdp_rows() -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(UCDP_CSV)))


class TestUcdp:
    @pytest.mark.parametrize(
        ("best", "level"),
        [(0, 1), (1, 2), (4, 2), (5, 3), (24, 3), (25, 4), (99, 4), (100, 5), (10**6, 5)],
    )
    def test_fatality_bins(self, best: int, level: int) -> None:
        assert ucdp_fatality_level(best) == level

    @pytest.mark.parametrize("best", [-1, True, 2.5, "3", None])
    def test_fatality_counts_must_be_non_negative_integers(self, best: Any) -> None:
        with pytest.raises(ValueError, match="non-negative integer"):
            ucdp_fatality_level(best)

    def test_a_real_ged_row(self) -> None:
        outcome = outcome_from_ucdp_row(ucdp_rows()[0])
        assert outcome == Outcome(
            case_id="ucdp-ged-577966",
            # date_start 2025-04-25 begins at 00:00 UTC+14 = 2025-04-24 10:00 UTC.
            observed_at=datetime(2025, 4, 24, 10, 0, tzinfo=UTC),
            description=(
                "UCDP GED event 577966 (IS - Civilians; Nigeria; Mulgoi area, Damboa LGA, Borno "
                "State): best estimate 6 fatalities (low 6, high 6), between 2025-04-25 and "
                "2025-04-25"
            ),
            outcome_source_ids=("UCDP GED", "Vanguard"),
            outcome_level=3,
            outcome_scale=adj.UCDP_FATALITY_SCALE,
        )
        assert outcome_from_ucdp_row(ucdp_rows()[0], case_id="mine").case_id == "mine"

    def test_a_row_without_publishers_is_refused(self) -> None:
        with pytest.raises(ValueError, match="event 1568 names no source_office"):
            outcome_from_ucdp_row(ucdp_rows()[1])

    def _row(self, **overrides: str | None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": "42",
            "date_start": "2024-02-01",
            "date_end": "2024-02-03",
            "best": "130",
            "source_office": "Xinhua News Agency;Xinhua News Agency; AFP ;",
        }
        row.update(overrides)
        return row

    def test_minimal_row_with_plain_dates(self) -> None:
        outcome = outcome_from_ucdp_row(self._row())
        assert outcome.outcome_source_ids == ("UCDP GED", "Xinhua News Agency", "AFP")
        assert outcome.outcome_level == 5
        assert outcome.observed_at == datetime(2024, 1, 31, 10, tzinfo=UTC)
        assert outcome.description == (
            "UCDP GED event 42: best estimate 130 fatalities, between 2024-02-01 and 2024-02-03"
        )

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"id": None}, "no 'id'"),
            ({"id": "x1"}, "id 'x1' is not a non-negative integer"),
            ({"best": "-1"}, "best '-1' is not a non-negative integer"),
            ({"best": " "}, "no 'best'"),
            ({"date_start": "2024-02-01 13:00:00.000"}, "date_start"),
            ({"date_end": "01/02/2024"}, "date_end"),
            ({"date_start": "2024-02-04"}, "is after date_end"),
            ({"source_office": " ; "}, "names no source_office"),
            ({"source_office": None}, "names no source_office"),
        ],
    )
    def test_malformed_rows_are_refused(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            outcome_from_ucdp_row(self._row(**overrides))

    def test_same_day_dates_and_midnight_times_parse(self) -> None:
        row = self._row(date_start="2024-02-01 00:00:00", date_end="2024-02-01 00:00:00.000")
        assert outcome_from_ucdp_row(row).observed_at == datetime(2024, 1, 31, 10, tzinfo=UTC)

    @pytest.mark.parametrize(
        ("report_sources", "latest", "eligible"),
        [
            (("Daily Trust",), datetime(2025, 4, 24, 9, 59, 59, tzinfo=UTC), True),
            (("Daily Trust",), datetime(2025, 4, 24, 10, 0, tzinfo=UTC), False),
            (("vanguard ",), datetime(2025, 4, 1, tzinfo=UTC), False),  # same publisher
            (("ucdp ged",), datetime(2025, 4, 1, tzinfo=UTC), False),  # built from UCDP
        ],
    )
    def test_ucdp_outcomes_enforce_non_circularity(
        self, tmp_path: Path, report_sources: tuple[str, ...], latest: datetime, eligible: bool
    ) -> None:
        outcome = outcome_from_ucdp_row(ucdp_rows()[0])
        case = make_case(
            outcome.case_id,
            report_source_ids=report_sources,
            latest_report_at=latest,
            created_at=latest,
        )
        later = datetime(2026, 1, 1, tzinfo=UTC)
        items = [case, outcome, rate("alice", outcome.case_id, 3, later)]
        items += [rate("bob", outcome.case_id, 3, later)]
        label = label_of(_ledger(tmp_path, with_fill(items)), outcome.case_id)
        assert label.eligible is eligible


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_verify_stats_and_export(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "l.jsonl"
        consensus_ledger(path, heldout=30, training=10)
        base = ["--ledger", str(path), "--n-boot", "20"]
        assert adj.main(["verify", *base]) == 0
        assert json.loads(capsys.readouterr().out) == {
            "ledger": str(path),
            "valid": True,
            "reason": "valid",
        }
        assert adj.main(["stats", *base]) == 0
        stats = json.loads(capsys.readouterr().out)
        assert stats["agreement"]["passed"] and stats["splits"]["heldout"]["eligible"] == 30
        out = tmp_path / "h.json"
        assert adj.main(["export", *base, "--split", "heldout", "--out", str(out)]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["exported"] is True and printed["cases"] == 30
        assert len(load_export(out)) == 30

    def test_cli_policy_options_reach_the_gate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "l.jsonl"
        consensus_ledger(path, heldout=30, training=10)
        out = tmp_path / "t.json"
        args = ["export", "--ledger", str(path), "--split", "training", "--out", str(out)]
        assert adj.main([*args, "--min-pair-cases", "41", "--n-boot", "20"]) == 1
        refused = json.loads(capsys.readouterr().out)
        assert refused == {
            "exported": False,
            "split": "training",
            "reason": "agreement gate failed: no rater pair has co-rated at least 41 cases",
        }
        assert not out.exists()
        assert (
            adj.main(
                [
                    *args,
                    "--kappa-threshold",
                    "1.0",
                    "--gate-on",
                    "ci_lower",
                    "--seed",
                    "3",
                    "--n-boot",
                    "20",
                ]
            )
            == 0
        )
        capsys.readouterr()

    def test_a_tampered_ledger_fails_verify_and_stats(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "l.jsonl"
        consensus_ledger(path, heldout=1, training=1)
        path.write_text(path.read_text().replace('"level": 1', '"level": 2', 1))
        assert adj.main(["verify", "--ledger", str(path)]) == 1
        assert json.loads(capsys.readouterr().out)["valid"] is False
        assert adj.main(["stats", "--ledger", str(path)]) == 1
        capsys.readouterr()

    def test_the_cli_pins_a_key(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "l.jsonl"
        AdjudicationLedger(path).add_case(make_case("c"))
        assert adj.main(["verify", "--ledger", str(path), "--pinned-key", "ab" * 4]) == 1
        assert json.loads(capsys.readouterr().out)["reason"] == (
            "record 0: not signed by the pinned key"
        )


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

if HAVE_HYPOTHESIS:
    pairs_strategy = st.lists(
        st.tuples(st.integers(1, 5), st.integers(1, 5)), min_size=1, max_size=40
    )

    class TestProperties:
        @settings(max_examples=150, deadline=None)
        @given(pairs_strategy)
        def test_kappa_equals_the_moment_identity(self, pairs: list[tuple[int, int]]) -> None:
            a, b = [x for x, _ in pairs], [y for _, y in pairs]
            assert quadratic_weighted_kappa(a, b) == oracle_kappa(a, b)

        @settings(max_examples=150, deadline=None)
        @given(pairs_strategy, st.randoms(use_true_random=False))
        def test_kappa_is_symmetric_bounded_and_order_free(
            self, pairs: list[tuple[int, int]], rnd: random.Random
        ) -> None:
            a, b = [x for x, _ in pairs], [y for _, y in pairs]
            kappa = quadratic_weighted_kappa(a, b)
            assert kappa == quadratic_weighted_kappa(b, a)
            shuffled = pairs[:]
            rnd.shuffle(shuffled)
            assert kappa == quadratic_weighted_kappa(
                [x for x, _ in shuffled], [y for _, y in shuffled]
            )
            assert kappa == quadratic_weighted_kappa(a * 2, b * 2)
            if kappa is not None:
                assert -1 <= kappa <= 1
            if a == b and kappa is not None:
                assert kappa == 1

        @settings(max_examples=200, deadline=None)
        @given(
            st.lists(st.integers(1, 5), min_size=2, max_size=3),
            st.one_of(st.none(), st.integers(1, 5)),
        )
        def test_a_resolved_label_never_leaves_the_ratings_range(
            self, levels: list[int], decision: int | None
        ) -> None:
            if decision is not None and not min(levels) <= decision <= max(levels):
                decision = None  # the protocol refuses such decisions
            level, method = resolve(levels, decision)
            if level is not None:
                assert min(levels) <= level <= max(levels)
                assert method in adj.METHODS
                if method != "adjudicator_decision":
                    assert levels.count(level) >= 2

        @settings(max_examples=200, deadline=None)
        @given(st.text(min_size=1, max_size=30), st.integers(1, 99))
        def test_split_matches_the_formula(self, case_id: str, pct: int) -> None:
            assert split_for(case_id, pct) == ("heldout" if bucket(case_id) < pct else "training")

        @settings(max_examples=200, deadline=None)
        @given(st.text(max_size=30))
        def test_source_normalisation_is_idempotent(self, raw: str) -> None:
            once = normalize_source_id(raw)
            assert normalize_source_id(once) == once


# ---------------------------------------------------------------------------
# Hardening: export provenance, leakage, concurrency and crash recovery
# ---------------------------------------------------------------------------


class TestExportProvenance:
    def test_a_training_export_is_refused_as_heldout(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "training.json"
        ledger.export("training", out)
        with pytest.raises(ValueError, match="'training' export; 'heldout' was expected"):
            load_export(out)
        assert len(load_export(out, split="training")) == 10

    def test_a_verified_export_matches_its_ledger(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        ledger.export("heldout", out)
        assert load_export(out, ledger=ledger) == ledger.heldout_cases()

    def test_an_edited_label_is_refused_against_the_ledger(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        document = ledger.export("heldout", out)
        document["cases"][0]["label"] = 1 + document["cases"][0]["label"] % 5
        out.write_text(json.dumps(document))
        assert load_export(out)  # structurally valid: only the ledger can catch it
        with pytest.raises(ExportRefused, match="does not match the ledger"):
            load_export(out, ledger=ledger)

    def test_an_edited_payload_is_refused_against_the_ledger(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        document = ledger.export("heldout", out)
        document["cases"][-1]["intel_reports"] = {"open_source": {"threat_score": 0.99}}
        out.write_text(json.dumps(document))
        with pytest.raises(ExportRefused, match="does not match the ledger"):
            load_export(out, ledger=ledger)

    def test_records_cut_off_the_ledger_are_detected(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        document = ledger.export("heldout", out)
        head = document["ledger_head_digest"]
        lines = ledger.path.read_text().splitlines(keepends=True)
        ledger.path.write_text("".join(lines[:-3]))
        assert ledger.verify() == (True, "valid")  # a bare chain cannot see it
        ok, reason = ledger.verify(expected_head_digest=head)
        assert not ok and "no longer in the chain" in reason
        with pytest.raises(ExportRefused, match="no longer in the chain"):
            load_export(out, ledger=ledger)

    def test_an_export_stays_valid_after_the_ledger_grows(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        ledger.export("heldout", out)
        before = load_export(out, ledger=ledger)
        extra = ids_in("heldout", 31)[-1]
        ledger.extend(case_items(extra, [3, 3]))
        assert load_export(out, ledger=ledger) == before

    def test_export_replaces_the_file_atomically(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        out.write_text("previous")
        ledger.export("heldout", out)
        assert json.loads(out.read_text())["split"] == "heldout"
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".heldout")] == []


class TestHeldoutMinimumCountsDistinctCases:
    def test_exact_duplicates_count_once_toward_the_minimum(self, tmp_path: Path) -> None:
        held = ids_in("heldout", 30)
        payload = {"open_source": {"threat_score": 0.5, "confidence": 0.8}}
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=AgreementPolicy(n_boot=50))
        items: list[Any] = []
        for i, cid in enumerate(held):
            dup: dict[str, Any] = {"intel_reports": payload} if i < 2 else {}  # two identical
            level = 3 if i < 2 else 1 + i % 5
            items += case_items(cid, [level, level], **dup)
        ledger.extend(items)
        assert len(ledger.heldout_cases()) == 30
        with pytest.raises(
            ExportRefused, match=r"29 distinct eligible held-out cases \(30 with exact"
        ):
            ledger.export_heldout_for_merit_gate()


class TestSplitLeakage:
    def test_the_same_payload_cannot_sit_in_both_splits(self, tmp_path: Path) -> None:
        held, train = ids_in("heldout", 1)[0], ids_in("training", 1)[0]
        payload = {"open_source": {"threat_score": 0.7, "confidence": 0.8}}
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.add_case(make_case(held, intel_reports=payload))
        with pytest.raises(ValueError, match="leaks training data"):
            ledger.add_case(make_case(train, intel_reports=payload))
        assert len(ledger.records()) == 1

    def test_the_same_payload_may_repeat_within_one_split(self, tmp_path: Path) -> None:
        first, second = ids_in("heldout", 2)
        payload = {"open_source": {"threat_score": 0.7, "confidence": 0.8}}
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.extend(
            [make_case(first, intel_reports=payload), make_case(second, intel_reports=payload)]
        )
        assert len(ledger.records()) == 2

    def test_a_forged_cross_split_duplicate_breaks_the_protocol(self, tmp_path: Path) -> None:
        held, train = ids_in("heldout", 1)[0], ids_in("training", 1)[0]
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.extend([make_case(held), make_case(train)])
        copied = ledger.records()[0]["entry"]["intel_reports"]

        def mutate(records: list[dict[str, Any]]) -> None:
            records[1]["entry"]["intel_reports"] = copied

        rewrite(ledger.path, mutate)
        assert ledger.verify() == (True, "valid")
        report = ledger.report()
        assert any("leaks training data" in p for p in report.protocol_problems)


class TestPeopleIds:
    def test_one_rater_under_two_spellings_is_one_rater(self, tmp_path: Path) -> None:
        cid = ids_in("heldout", 1)[0]
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.extend([make_case(cid), make_outcome(cid), rate("alice", cid, 3)])
        with pytest.raises(ValueError, match="already rated"):
            ledger.add_rating(rate(" Alice".strip().upper(), cid, 3))

    def test_a_rater_cannot_adjudicate_under_another_spelling(self, tmp_path: Path) -> None:
        cid = ids_in("heldout", 1)[0]
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.extend(
            [make_case(cid), make_outcome(cid), rate("dana", cid, 2), rate("bob", cid, 3)]
        )
        with pytest.raises(ValueError, match="rated case"):
            ledger.add_decision(decide(cid, 3, who="DANA"))

    def test_rater_view_matches_under_another_spelling(self, tmp_path: Path) -> None:
        cid = ids_in("heldout", 1)[0]
        ledger = AdjudicationLedger(tmp_path / "l.jsonl", policy=SMALL)
        ledger.extend(
            [make_case(cid), make_outcome(cid), rate("alice", cid, 2), rate("bob", cid, 2)]
        )
        view = ledger.rater_view("Alice", cid)
        assert view["own_rating"]["level"] == 2
        assert [r["rater_id"] for r in view["other_ratings"]] == ["bob"]


class TestWritersAndCrashes:
    def test_concurrent_writers_keep_one_chain(self, tmp_path: Path) -> None:
        import threading

        path = tmp_path / "l.jsonl"
        cids = ids_in("heldout", 40)
        errors: list[BaseException] = []

        def write(chunk: list[str]) -> None:
            ledger = AdjudicationLedger(path, policy=SMALL)
            for cid in chunk:
                try:
                    ledger.add_case(make_case(cid))
                except BaseException as exc:  # pragma: no cover - reported below
                    errors.append(exc)

        threads = [threading.Thread(target=write, args=(cids[i::4],)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == []
        ledger = AdjudicationLedger(path, policy=SMALL)
        assert ledger.verify() == (True, "valid")
        assert len(ledger.records()) == 40

    def test_a_torn_tail_is_reported_then_repaired(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        size = ledger.path.stat().st_size
        with ledger.path.open("a", encoding="utf-8") as handle:
            handle.write('{"schema": "mercury.adjudication.ledger/v1", "ind')
        ok, reason = ledger.verify()
        assert not ok and "not valid JSON" in reason
        with pytest.raises(ValueError, match="broken adjudication ledger"):
            ledger.add_case(make_case(ids_in("heldout", 31)[-1]))
        torn = ledger.path.stat().st_size - size
        assert ledger.repair_torn_tail() == torn
        assert ledger.path.stat().st_size == size
        assert ledger.verify() == (True, "valid")
        assert ledger.repair_torn_tail() == 0
        ledger.add_case(make_case(ids_in("heldout", 31)[-1]))
        assert ledger.verify() == (True, "valid")

    def test_repair_never_drops_a_complete_record(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        data = ledger.path.read_bytes()
        ledger.path.write_bytes(data.rstrip(b"\n"))  # last record intact, newline lost
        assert ledger.repair_torn_tail() == 0
        assert ledger.path.read_bytes() == data.rstrip(b"\n")

    def test_repair_refuses_when_earlier_records_are_damaged(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        lines = ledger.path.read_text().splitlines(keepends=True)
        lines[3] = lines[3].replace('"index": 3', '"index": 99')
        ledger.path.write_text("".join(lines) + '{"torn')
        with pytest.raises(ValueError, match="refusing to repair"):
            ledger.repair_torn_tail()

    def test_cli_repair_and_expected_head(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        head = ledger.head_digest()
        with ledger.path.open("a", encoding="utf-8") as handle:
            handle.write("{")
        assert adj.main(["repair", "--ledger", str(ledger.path)]) == 0
        assert json.loads(capsys.readouterr().out)["bytes_removed"] == 1
        assert adj.main(["verify", "--ledger", str(ledger.path), "--expected-head", head]) == 0
        capsys.readouterr()
        lines = ledger.path.read_text().splitlines(keepends=True)
        ledger.path.write_text("".join(lines[:-1]))
        assert adj.main(["verify", "--ledger", str(ledger.path), "--expected-head", head]) == 1


class TestSignedExport:
    """An export signed at write time is checked on its own, away from the ledger."""

    def _signed(self, tmp_path: Path) -> tuple[Path, dict[str, Any], Any]:
        pair = mldsa().generate_keypair()
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        document = ledger.export("heldout", out, signing_keypair=(pair.public_key, pair.secret_key))
        return out, document, pair

    def test_a_signed_export_loads_under_its_pinned_key(self, tmp_path: Path) -> None:
        out, document, pair = self._signed(tmp_path)
        assert document["signature"]["algorithm"] == "ML-DSA-65"
        pairs = load_export(out, expected_public_key_hex=pair.public_key.hex())
        assert len(pairs) == 30

    def test_an_edited_label_breaks_the_signature_without_the_ledger(self, tmp_path: Path) -> None:
        out, document, pair = self._signed(tmp_path)
        document["cases"][0]["label"] = 1 + document["cases"][0]["label"] % 5
        out.write_text(json.dumps(document))
        # The unsigned path could only catch this against the ledger; a
        # signature catches it on a host that holds the file alone, pinned or not.
        with pytest.raises(ExportRefused, match="altered after it was signed"):
            load_export(out)
        with pytest.raises(ExportRefused, match="altered after it was signed"):
            load_export(out, expected_public_key_hex=pair.public_key.hex())

    def test_a_re_signed_forgery_is_refused_under_the_pinned_key(self, tmp_path: Path) -> None:
        out, document, pair = self._signed(tmp_path)
        forger = mldsa().generate_keypair()
        document["cases"][0]["label"] = 1 + document["cases"][0]["label"] % 5
        unsigned = {k: v for k, v in document.items() if k != "signature"}
        document["signature"] = adj._sign_export(unsigned, (forger.public_key, forger.secret_key))
        out.write_text(json.dumps(document))
        assert load_export(out)  # self-consistent under the forger's own key
        with pytest.raises(ExportRefused, match="not signed by the pinned key"):
            load_export(out, expected_public_key_hex=pair.public_key.hex())

    def test_an_unsigned_export_is_refused_when_a_key_is_pinned(self, tmp_path: Path) -> None:
        pair = mldsa().generate_keypair()
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        out = tmp_path / "heldout.json"
        ledger.export("heldout", out)
        assert len(load_export(out)) == 30  # unsigned stays loadable without a pin
        with pytest.raises(ExportRefused, match="not signed"):
            load_export(out, expected_public_key_hex=pair.public_key.hex())

    def test_a_pinned_ledger_pins_its_exports(self, tmp_path: Path) -> None:
        pair = mldsa().generate_keypair()
        keys = (pair.public_key, pair.secret_key)
        path = tmp_path / "l.jsonl"
        wanted = sorted(set(ids_in("heldout", 30)) | set(ids_in("training", 10)))
        items: list[Any] = []
        for i, cid in enumerate(wanted):
            items += case_items(cid, [1 + i % 5, 1 + i % 5])
        pinned = AdjudicationLedger(
            path, policy=AgreementPolicy(n_boot=50), expected_public_key_hex=pair.public_key.hex()
        )
        pinned.extend(items, signing_keypair=keys)
        out = tmp_path / "heldout.json"
        pinned.export("heldout", out)  # written without a signature
        with pytest.raises(ExportRefused, match="not signed"):
            load_export(out, ledger=pinned)
        pinned.export("heldout", out, signing_keypair=keys)
        assert load_export(out, ledger=pinned) == pinned.heldout_cases()

    def test_export_without_a_file_returns_the_document(self, tmp_path: Path) -> None:
        ledger, _ = consensus_ledger(tmp_path / "l.jsonl", heldout=30, training=10)
        document = ledger.export("heldout", None)
        assert len(document["cases"]) == 30
        assert not list(tmp_path.glob("*.json"))

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Discovery artifacts replay: a claim comes out again, or it does not, and says why.

Each discovery instrument reports an artifact: the claim, the procedure, the
numbers the procedure was given, and a digest binding them.  These tests pin
what makes that worth having:

* the procedures are right (a rank p-value is a rank, Holm is Holm, the
  Granger arithmetic re-derives from its sums of squares);
* an artifact replays from the record alone, through JSON and through a
  ``DecisionRecord``, to the verdict it recorded;
* a replay can FAIL: against a reference of the reader's own, against data
  that gives a different answer, or when the artifact was altered;
* what cannot be replayed says so and why, and is never read as holding;
* a hypothesis falls with its premise;
* what leaves the process is only what its reader may see: a shared
  reference is withheld, and the ledger keeps claims and digests, not inputs.

Before this, the record's hypothesis was ``None`` on every served decision
(the reasoner was handed premises none of its rules could match) and its
novelty "challenge" told the reader to call a method on the live engine.
"""

from __future__ import annotations

import copy
import json
import time
import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    given,
    settings,
    strategies as st,
)

from omni_mercury_engine import discovery as D
from omni_mercury_engine.decision.decider import DecisionAbstentionResponder
from omni_mercury_engine.decision.record import DecisionRecord

torch = pytest.importorskip("torch", reason="the orchestrator's uncertainty stack needs [ml]")

from omni_mercury_engine.cognitive.orchestrator import CognitiveOrchestrator

_DETECTION = {"is_anomaly": True, "anomaly_prob": 0.93, "severity": 0.8, "threshold_used": 0.6}


def _wire(value: Any) -> Any:
    """As it travels: through JSON and back."""
    return json.loads(json.dumps(value))


def _planted(rng: np.random.Generator, rows: int = 200, shift: float = 0.0) -> np.ndarray:
    """Four columns of noise in which X0 drives X1 at lag 1."""
    window = rng.normal(size=(rows, 4)) + shift
    window[1:, 1] += 0.6 * window[:-1, 0]
    return window


@pytest.fixture(scope="module")
def analysed() -> dict[str, Any]:
    """A flagged, novel window with a planted driver, after 30 ordinary ones."""
    rng = np.random.default_rng(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orchestrator = CognitiveOrchestrator(enable_curiosity=True)
        for _ in range(30):
            orchestrator.analyze(
                detection_result=_DETECTION,
                raw_data=rng.normal(size=(200, 4)),
                context={"domain": "general"},
            )
        window = _planted(rng, shift=3.0)
        payload = orchestrator.analyze(
            detection_result=_DETECTION, raw_data=window, context={"domain": "general"}
        ).to_dict()
    record = DecisionAbstentionResponder().decide({**_DETECTION, "cognitive_analysis": payload})
    return {"payload": payload, "window": window, "record": _wire(record.to_dict())}


# --------------------------------------------------------------------------- #
# Procedures
# --------------------------------------------------------------------------- #
class TestProcedures:
    def test_rank_p_value_is_the_share_at_least_as_large(self) -> None:
        reference = [1.0, 2.0, 3.0, 4.0]
        assert D.rank_p_value(5.0, reference) == pytest.approx(1 / 5)
        assert D.rank_p_value(3.0, reference) == pytest.approx(3 / 5)  # 3 and 4, plus one
        assert D.rank_p_value(0.0, reference) == pytest.approx(1.0)
        assert D.rank_p_value(9.0, []) == 1.0  # no reference: nothing can be called novel

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(seed=st.integers(min_value=0, max_value=2**16))
    def test_rank_p_values_of_exchangeable_draws_are_uniform_enough(self, seed: int) -> None:
        """At most about alpha of ordinary observations rank in the top alpha."""
        rng = np.random.default_rng(seed)
        reference = rng.standard_exponential(400)
        fresh = rng.standard_exponential(4000)
        share = float(np.mean([D.rank_p_value(v, reference) <= 0.05 for v in fresh]))
        assert share <= 0.10  # 0.05 in expectation; the margin is the reference's own noise

    def test_holm_matches_a_worked_example(self) -> None:
        # Sorted 0.01, 0.02, 0.03, 0.5 with m=4: 0.04, max(0.04, 0.06), 0.06, 0.5.
        adjusted = D.holm_adjust([0.03, 0.5, 0.01, 0.02])
        assert adjusted == pytest.approx([0.06, 0.5, 0.04, 0.06])
        assert D.holm_adjust([]) == []
        assert D.holm_adjust([0.9, 0.8]) == pytest.approx([1.0, 1.0])

    def test_the_pair_test_is_an_f_test_on_its_own_sums_of_squares(self) -> None:
        from scipy import stats

        rng = np.random.default_rng(4)
        x = rng.normal(size=150)
        y = rng.normal(size=150)
        y[1:] += 0.4 * x[:-1]
        found = D.granger_pair(x, y, 5)
        assert found is not None
        f_stat = ((found.rss_restricted - found.rss_unrestricted) / found.df1) / (
            found.rss_unrestricted / found.df2
        )
        assert found.f_statistic == pytest.approx(f_stat)
        assert found.p_raw == pytest.approx(stats.f.sf(f_stat, found.df1, found.df2))
        # Corrected for the lags it searched, and never below the raw value.
        assert found.p_lag_adjusted == pytest.approx(min(1.0, found.lags_tested * found.p_raw))
        assert 0.0 <= found.partial_r2 <= 1.0

    def test_contaminated_rows_are_the_ones_far_from_the_median(self) -> None:
        rng = np.random.default_rng(1)
        window = rng.normal(size=(120, 3))
        window[40:43] += 12.0
        marked = D.contaminated_rows(window)
        assert marked[40:43].all()
        assert marked.sum() <= 5  # the shock, and at most a stray tail value

    def test_a_shifted_window_is_not_stable_and_a_plain_one_is(self) -> None:
        rng = np.random.default_rng(2)
        plain = rng.normal(size=(200, 4))
        shifted = plain.copy()
        shifted[80:130] += 3.0
        assert D.level_stability(plain)["stable"] is True
        unstable = D.level_stability(shifted)
        assert unstable["stable"] is False
        assert "not stable" in unstable["reason"]
        too_short = D.level_stability(plain[:25])
        assert too_short["stable"] is False and "too few" in too_short["reason"]


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #
class TestNoveltyArtifact:
    def test_it_replays_from_the_record_alone(self, analysed: dict[str, Any]) -> None:
        artifact = analysed["record"]["signals"]["cognitive"]["discovery"]["novelty"]
        assert artifact["replayable"] == "from_record"
        report = D.replay(artifact)
        assert report["status"] == D.HOLDS
        assert report["recomputed"]["is_novel"] is artifact["claim"]["is_novel"] is True
        assert report["recomputed"]["p_value"] == pytest.approx(artifact["claim"]["p_value"])

    def test_against_a_reference_it_resembles_it_is_no_longer_novel(
        self, analysed: dict[str, Any]
    ) -> None:
        """A replay can fail: the verdict is about a reference, and the reader may bring one."""
        artifact = analysed["record"]["signals"]["cognitive"]["discovery"]["novelty"]
        rng = np.random.default_rng(9)
        observation = np.asarray(artifact["inputs"]["observation"])
        alike = observation + rng.normal(scale=0.5, size=(80, observation.size))
        report = D.replay(artifact, reference=alike)
        assert report["status"] == D.FAILS
        assert report["recomputed"]["is_novel"] is False
        far = rng.normal(size=(80, observation.size)) - 20.0
        assert D.replay(artifact, reference=far)["status"] == D.HOLDS

    def test_a_reference_of_the_wrong_shape_cannot_be_replayed_against(
        self, analysed: dict[str, Any]
    ) -> None:
        artifact = analysed["record"]["signals"]["cognitive"]["discovery"]["novelty"]
        assert D.replay(artifact, reference=np.zeros((80, 2)))["status"] == D.CANNOT_REPLAY
        assert D.replay(artifact, reference=np.zeros((2, 4)))["status"] == D.CANNOT_REPLAY

    @pytest.mark.parametrize("path", [("claim", "p_value"), ("inputs", "alpha")])
    def test_an_altered_artifact_fails(
        self, analysed: dict[str, Any], path: tuple[str, str]
    ) -> None:
        artifact = copy.deepcopy(analysed["record"]["signals"]["cognitive"]["discovery"]["novelty"])
        artifact[path[0]][path[1]] = 0.5
        report = D.replay(artifact)
        assert report["status"] == D.FAILS
        assert "does not match its own digest" in report["detail"]

    def test_an_undetermined_verdict_yields_no_artifact(self) -> None:
        """Warm-up: there is no claim, so there is nothing to carry or replay."""
        assert (
            D.novelty_artifact(
                observation=[0.1, 0.2],
                reference_mean=None,
                reference_std=None,
                reference_n=1,
                calibration=None,
                alpha=None,
                standardized_distance=0.0,
                p_value=None,
                is_novel=False,
            )
            is None
        )

    def _built(self, features: int, disclose: bool) -> dict[str, Any]:
        rng = np.random.default_rng(3)
        built = D.novelty_artifact(
            observation=rng.normal(size=features).tolist(),
            reference_mean=[0.0] * features,
            reference_std=[1.0] * features,
            reference_n=50,
            calibration=rng.random(40).tolist(),
            alpha=0.05,
            standardized_distance=1.0,
            p_value=0.5,
            is_novel=False,
            disclose=disclose,
        )
        assert built is not None
        return built

    def test_a_shared_reference_is_withheld_and_the_artifact_says_so(self) -> None:
        artifact = self._built(4, disclose=False)
        assert artifact["replayable"] == "no"
        assert "not disclosed" in artifact["not_replayable_because"]
        for vector in ("observation", "reference_mean", "reference_std", "calibration"):
            assert vector not in artifact["inputs"]
        report = D.replay(artifact)
        assert report["status"] == D.CANNOT_REPLAY  # never read as holding

    def test_a_wide_observation_is_not_carried(self) -> None:
        artifact = self._built(D.MAX_FEATURES + 1, disclose=True)
        assert artifact["replayable"] == "no"
        assert "exceed" in artifact["not_replayable_because"]
        assert len(json.dumps(artifact)) < 2000


# --------------------------------------------------------------------------- #
# Causal
# --------------------------------------------------------------------------- #
class TestCausalArtifact:
    def test_the_planted_edge_is_found_and_replays_from_the_record(self) -> None:
        window = _planted(np.random.default_rng(5))
        artifact = _wire(D.causal_artifact(window, max_lag=5, alpha=0.05))
        assert [(e["source"], e["target"], e["lag"]) for e in artifact["claim"]["edges"]] == [
            ("X0", "X1", 1)
        ]
        assert artifact["claim"]["pairs_tested"] == 12
        report = D.replay(artifact)
        assert report["status"] == D.HOLDS
        assert "sums of squares" in report["detail"]

    def test_with_the_data_the_statistics_themselves_are_recomputed(self) -> None:
        window = _planted(np.random.default_rng(5))
        artifact = _wire(D.causal_artifact(window, max_lag=5, alpha=0.05))
        assert D.replay(artifact, data=window)["status"] == D.HOLDS
        # Wider data whose leading columns are the window: the instrument's own view.
        wider = np.column_stack([window, np.zeros((window.shape[0], 3))])
        assert D.replay(artifact, data=wider)["status"] == D.HOLDS

    def test_other_data_is_not_the_window(self) -> None:
        window = _planted(np.random.default_rng(5))
        artifact = D.causal_artifact(window, max_lag=5, alpha=0.05)
        report = D.replay(artifact, data=window + 1e-9)
        assert report["status"] == D.CANNOT_REPLAY
        assert "digest differs" in report["detail"]

    def test_a_statistic_altered_with_its_digest_recomputed_still_fails(self) -> None:
        """Re-sealing a doctored artifact does not make its arithmetic come out."""
        window = _planted(np.random.default_rng(5))
        artifact = D.causal_artifact(window, max_lag=5, alpha=0.05)
        target = next(
            t for t in artifact["inputs"]["tests"] if (t["source"], t["target"]) == ("X0", "X1")
        )
        target["rss_unrestricted"] = target["rss_restricted"]  # X adds nothing
        artifact["digest"] = D._seal(
            artifact["kind"], artifact["procedure"], artifact["claim"], artifact["inputs"]
        )
        report = D.replay(artifact)
        assert report["status"] == D.FAILS
        assert "do not give the recorded edges" in report["detail"]

    def test_noise_under_a_common_shock_claims_no_edge(self) -> None:
        """The prior procedure reported an edge here every time."""
        rng = np.random.default_rng(6)
        window = rng.normal(size=(200, 5))
        window[90:96] += 9.0
        artifact = D.causal_artifact(window, max_lag=5, alpha=0.05)
        assert artifact["claim"]["edges"] == []
        assert artifact["inputs"]["rows_excluded"] >= 6

    def test_a_window_whose_level_moves_is_not_assessed(self) -> None:
        rng = np.random.default_rng(7)
        window = rng.normal(size=(200, 5))
        window[60:110] += 3.0
        artifact = _wire(D.causal_artifact(window, max_lag=5, alpha=0.05))
        assert artifact["claim"]["assessed"] is False
        assert artifact["claim"]["edges"] == [] and artifact["claim"]["pairs_tested"] == 0
        assert "not stable" in artifact["claim"]["not_assessed_because"]
        report = D.replay(artifact)
        assert report["status"] == D.HOLDS and "no edge is claimed" in report["detail"]
        assert D.replay(artifact, data=window)["status"] == D.HOLDS


# --------------------------------------------------------------------------- #
# Hypothesis
# --------------------------------------------------------------------------- #
class TestHypothesisArtifact:
    def test_the_orchestrator_derives_a_conclusion_from_measured_verdicts(
        self, analysed: dict[str, Any]
    ) -> None:
        """Fails on the prior code: the chain was empty and the hypothesis None."""
        payload = analysed["payload"]
        assert [step["rule"] for step in payload["reasoning_chain"]] == [
            "flagged_and_novel",
            "flagged_and_lagged_dependence",
            "new_regime_with_driver",
        ]
        # No confidence was measured for a conjunction, so none is reported.
        assert all(step["confidence"] is None for step in payload["reasoning_chain"])
        artifact = payload["discovery"]["hypothesis"]
        assert artifact["claim"]["conclusion"] == "candidate_driven_regime"
        assert artifact["claim"]["holds"] is True
        assert {p["id"]: p["source"] for p in artifact["inputs"]["premises"]} == {
            "anomaly_flagged": "detection",
            "observation_novel": "novelty",
            "lagged_dependence": "causal",
        }

    def test_the_record_names_the_weakest_premise_and_when_it_fails(
        self, analysed: dict[str, Any]
    ) -> None:
        cognitive = analysed["record"]["signals"]["cognitive"]
        hypothesis = cognitive["hypothesis"]
        assert "X0 leads X1 at lag 1" in hypothesis["conclusion"]
        assert hypothesis["confidence"] is None
        assert hypothesis["chain_depth"] == 3
        falsifier = hypothesis["falsifier"]
        assert falsifier["rule"] == "observation_novel"  # closest to its cut
        assert falsifier["margin"] > 1.0
        assert falsifier["withdrawn_if"] == "observation_novel: statistic > 0.05"
        assert "replay discovery.hypothesis" in falsifier["falsifiable_by"]

    def test_every_artifact_in_the_record_holds_on_replay(self, analysed: dict[str, Any]) -> None:
        reports = D.replay_record(analysed["record"])
        assert [(r["kind"], r["status"]) for r in reports] == [
            ("novelty", D.HOLDS),
            ("causal", D.HOLDS),
            ("hypothesis", D.HOLDS),
        ]
        assert D.replay_record(analysed["record"], data=analysed["window"])[1]["status"] == D.HOLDS

    def test_the_hypothesis_falls_with_its_premise(self, analysed: dict[str, Any]) -> None:
        """Against a reference the observation resembles, novelty fails and takes the conclusion."""
        novelty = analysed["record"]["signals"]["cognitive"]["discovery"]["novelty"]
        observation = np.asarray(novelty["inputs"]["observation"])
        rng = np.random.default_rng(9)
        alike = observation + rng.normal(scale=0.5, size=(80, observation.size))
        reports = {r["kind"]: r for r in D.replay_record(analysed["record"], reference=alike)}
        assert reports["novelty"]["status"] == D.FAILS
        assert reports["causal"]["status"] == D.HOLDS  # untouched by the reference
        assert reports["hypothesis"]["status"] == D.FAILS
        assert "observation_novel" in reports["hypothesis"]["detail"]

    def test_a_premise_that_cannot_be_replayed_leaves_the_conclusion_undecided(
        self, analysed: dict[str, Any]
    ) -> None:
        reports = {
            r["kind"]: r for r in D.replay_record(analysed["record"], data=np.zeros((200, 4)))
        }
        assert reports["causal"]["status"] == D.CANNOT_REPLAY
        assert reports["hypothesis"]["status"] == D.CANNOT_REPLAY  # not HOLDS, not FAILS

    def test_one_premise_alone_derives_nothing(self) -> None:
        """A flag with no novelty and no edge is a flag, not a hypothesis."""
        rng = np.random.default_rng(8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            payload = (
                CognitiveOrchestrator(enable_curiosity=True)
                .analyze(
                    detection_result=_DETECTION,
                    raw_data=rng.normal(size=(200, 4)),
                    context={"domain": "general"},
                )
                .to_dict()
            )
        assert payload["reasoning_chain"] == []
        assert "hypothesis" not in payload["discovery"]


# --------------------------------------------------------------------------- #
# The record as a whole
# --------------------------------------------------------------------------- #
class TestTheRecord:
    def test_the_record_round_trips_with_its_artifacts(self, analysed: dict[str, Any]) -> None:
        wire = analysed["record"]
        assert DecisionRecord.from_dict(wire).to_dict() == wire
        assert sorted(wire["signals"]["cognitive"]["discovery"]) == [
            "causal",
            "hypothesis",
            "novelty",
        ]

    def test_copies_that_differ_fail(self, analysed: dict[str, Any]) -> None:
        """A whole result carries each artifact twice; altering either copy is caught."""
        result = _wire(
            {
                "cognitive_analysis": analysed["payload"],
                "decision": analysed["record"],
            }
        )
        assert {r["status"] for r in D.replay_record(result)} == {D.HOLDS}
        result["decision"]["signals"]["cognitive"]["discovery"]["causal"]["claim"]["edges"][0][
            "partial_r2"
        ] = 0.99
        reports = {r["kind"]: r for r in D.replay_record(result)}
        assert reports["causal"]["status"] == D.FAILS
        assert "differ from each other" in reports["causal"]["detail"]

    def test_an_artifact_damaged_on_the_way_is_not_carried_into_the_record(
        self, analysed: dict[str, Any]
    ) -> None:
        payload = copy.deepcopy(analysed["payload"])
        payload["discovery"]["novelty"]["claim"]["is_novel"] = False  # digest no longer matches
        cognitive = (
            DecisionAbstentionResponder()
            .decide({**_DETECTION, "cognitive_analysis": payload})
            .to_dict()["signals"]["cognitive"]
        )
        assert "novelty" not in cognitive["discovery"]
        assert "causal" in cognitive["discovery"]

    def test_a_record_without_artifacts_replays_to_nothing(self) -> None:
        assert D.replay_record({"signals": {"cognitive": None}}) == []
        assert D.replay_record(None) == []

    def test_the_ledger_keeps_the_claim_and_the_digest_not_the_inputs(
        self, analysed: dict[str, Any]
    ) -> None:
        from omni_mercury_engine.engine import _ledger_record

        wire = analysed["record"]
        kept = _ledger_record(DecisionRecord.from_dict(wire), wire).to_dict()
        for kind, original in wire["signals"]["cognitive"]["discovery"].items():
            slim = kept["signals"]["cognitive"]["discovery"][kind]
            assert "inputs" not in slim
            assert slim["claim"] == original["claim"]
            assert slim["digest"] == original["digest"]  # the full artifact can be matched to it
            assert D.replay(slim)["status"] != D.HOLDS  # and it does not pass for one
        assert len(json.dumps(kept)) < len(json.dumps(wire)) / 2
        # A record with no artifacts is kept as it is.
        bare = DecisionAbstentionResponder().decide(_DETECTION)
        assert _ledger_record(bare, bare.to_dict()) is bare


class TestReferenceScopes:
    def test_each_scope_has_a_reference_of_its_own(self) -> None:
        """What one caller has shown does not decide what is novel for another."""
        rng = np.random.default_rng(10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orchestrator = CognitiveOrchestrator(enable_curiosity=True)

            def analyse(scope: str | None, shift: float) -> dict[str, Any]:
                context: dict[str, Any] = {"domain": "general"}
                if scope is not None:
                    context["reference_scope"] = scope
                return orchestrator.analyze(
                    detection_result=_DETECTION,
                    raw_data=rng.normal(size=(60, 4)) + shift,
                    context=context,
                ).to_dict()

            for _ in range(30):
                analyse("user:a", 0.0)
            # The same far window: novel for the caller with a reference, undetermined for one without.
            assert analyse("user:a", 5.0)["is_novel"] is True
            fresh = analyse("user:b", 5.0)
            assert fresh["novelty_measured"] is False and fresh["is_novel"] is False
            assert orchestrator.curiosity is not None
            assert orchestrator.curiosity.observations_seen == 0  # the unscoped one saw nothing

    def test_scopes_are_bounded(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orchestrator = CognitiveOrchestrator(enable_curiosity=True)
        for index in range(orchestrator.REFERENCE_SCOPE_LIMIT + 20):
            orchestrator._curiosity_for(f"user:{index}")
        assert len(orchestrator._curiosity_by_scope) == orchestrator.REFERENCE_SCOPE_LIMIT
        assert "user:0" not in orchestrator._curiosity_by_scope  # least recently used went


class TestAReplayThatServesOthersIsBounded:
    """A record is the caller's, and so is the work its artifact asks for.

    Each test fails on the prior code, where a replay recomputed whatever
    width and lag the artifact named (64 columns at lag 50 did not finish in
    ten minutes) and rebuilt a reference in time quadratic in its rows.
    """

    def test_a_wider_finding_than_the_instrument_makes_is_not_recomputed(self) -> None:
        rng = np.random.default_rng(3)
        window = rng.normal(size=(60, 12))
        artifact = _wire(D.causal_artifact(window, max_lag=5, alpha=0.05))
        assert artifact["claim"]["pairs_tested"] == 132
        # Offline it is the reader's own time: unbounded, and it holds.
        assert D.replay(artifact, data=window)["status"] == D.HOLDS
        # Served, it is refused before any arithmetic, with the way out.
        for supplied in (window, None):
            report = D.replay(artifact, data=supplied, bounds=D.SERVED_REPLAY_BOUNDS)
            assert report["status"] == D.CANNOT_REPLAY
            assert "at most 5 columns at lags up to 5" in report["detail"]
            assert "declares 12 column(s) at lag 5 with 132 pair(s)" in report["detail"]
            assert "mercury-agent replay" in report["detail"]

    def test_a_longer_lag_than_the_instrument_tries_is_not_recomputed(self) -> None:
        window = _planted(np.random.default_rng(5))
        artifact = _wire(D.causal_artifact(window, max_lag=9, alpha=0.05))
        report = D.replay(artifact, data=window, bounds=D.SERVED_REPLAY_BOUNDS)
        assert report["status"] == D.CANNOT_REPLAY
        assert "declares 4 column(s) at lag 9" in report["detail"]

    def test_what_the_shipped_instrument_makes_is_inside_the_bounds(self) -> None:
        rng = np.random.default_rng(5)
        window = rng.normal(size=(200, 5))
        window[1:, 1] += 0.6 * window[:-1, 0]
        artifact = _wire(D.causal_artifact(window, max_lag=5, alpha=0.05))
        assert artifact["claim"]["pairs_tested"] == D.SERVED_REPLAY_BOUNDS.max_pairs == 20
        for supplied in (window, None):
            report = D.replay(artifact, data=supplied, bounds=D.SERVED_REPLAY_BOUNDS)
            assert report["status"] == D.HOLDS

    def test_a_record_carries_the_bound_to_each_artifact(self, analysed: dict[str, Any]) -> None:
        reports = D.replay_record(analysed["record"], bounds=D.SERVED_REPLAY_BOUNDS)
        assert [r["status"] for r in reports] == [D.HOLDS] * 3  # the orchestrator's own: inside

    def test_a_reference_is_built_the_way_the_instrument_builds_its_own(self) -> None:
        """Row for row the same running statistics as the live novelty engine."""
        from omni_mercury_engine.cognitive.cognitive_evolution_engine import CuriosityEngine

        rng = np.random.default_rng(11)
        rows = rng.normal(loc=3.0, scale=2.0, size=(120, 6))
        engine = CuriosityEngine(calibration_window=256)
        for row in rows:
            engine.explore("row", row)
        mean, std, distances, count = D._reference_from_rows(rows, 256)
        assert count == 120
        np.testing.assert_allclose(mean, rows.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(std, rows.std(axis=0, ddof=1), rtol=1e-12)
        np.testing.assert_allclose(distances, list(engine._calibration), rtol=1e-9)

    def test_building_a_reference_takes_time_in_proportion_to_its_rows(self) -> None:
        """Four times the rows, about four times the time; it was sixteen."""

        def seconds(rows: int) -> float:
            data = np.random.default_rng(0).normal(size=(rows, 8))
            best = float("inf")
            for _ in range(3):
                started = time.perf_counter()
                D._reference_from_rows(data, 256)
                best = min(best, time.perf_counter() - started)
            return best

        small, large = seconds(2_000), seconds(8_000)
        # Linear is 4x; the quadratic version measured 13-19x. 9 separates them
        # with room for a loaded machine on either side.
        assert large / small < 9.0, (small, large)


_junk = st.recursive(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-(10**6), max_value=10**6),
        st.floats(allow_nan=True, allow_infinity=True),
        st.text(max_size=30),
    ),
    lambda children: st.one_of(
        st.lists(children, max_size=4), st.dictionaries(st.text(max_size=10), children, max_size=5)
    ),
    max_leaves=14,
)


class TestReplayIsTotal:
    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(junk=_junk)
    def test_anything_replays_to_a_report_and_never_to_holds(self, junk: Any) -> None:
        report = D.replay(junk)  # must not raise
        assert report["status"] == D.CANNOT_REPLAY
        assert D.replay_record(junk) == []

    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        kind=st.sampled_from(["novelty", "causal", "hypothesis", "other"]),
        claim=_junk,
        inputs=_junk,
        seal=st.booleans(),
    )
    def test_a_malformed_artifact_never_holds(
        self, kind: str, claim: Any, inputs: Any, seal: bool
    ) -> None:
        artifact: dict[str, Any] = {
            "schema": D.SCHEMA,
            "kind": kind,
            "procedure": "x/v1",
            "claim": claim,
            "inputs": inputs,
            "replayable": "from_record",
        }
        if seal and isinstance(claim, dict) and isinstance(inputs, dict):
            try:
                artifact["digest"] = D._seal(kind, "x/v1", claim, inputs)
            except ValueError:  # non-finite numbers cannot be sealed
                return
        report = D.replay(artifact)  # must not raise
        assert report["status"] in (D.FAILS, D.CANNOT_REPLAY)

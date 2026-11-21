# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The served flagship path, end to end: calibrated, sourced, and honest about what it measured.

Run against the real served engine and the shipped checkpoint, through the
real route, with the cognitive critique on (the default).  One flagged window
with a planted driver, after enough flagged windows to give the caller a
reference, must come back as a decision that:

* is **calibrated** and says against what (the conformal certificate and the
  size of the set it was calibrated on);
* is **provenance-complete** (the checkpoint and its digest);
* presents **no unmeasured number as measured**: each measured flag is true
  exactly when the evidence that proves it is in the same record;
* carries discovery artifacts the caller can **replay** -- through
  ``POST /flagship/replay`` here, the same check ``mercury-agent replay``
  runs offline -- and that replay can fail.

And on a shared server, what a caller gets is only theirs: a signed-in
caller has a novelty reference of their own and is handed it; callers
without credentials share one and are not; the server-side ledger keeps
claims and digests, not the callers' numbers.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the API server needs the [api] extra")
pytest.importorskip("torch", reason="the flagship engine needs [ml]")

from click.testing import CliRunner
from fastapi.testclient import TestClient

from omni_mercury_engine import discovery as D
from omni_mercury_engine.api.auth import User
from omni_mercury_engine.api.routes import detection
from omni_mercury_engine.api.server import app
from omni_mercury_engine.cli import main as cli
from omni_mercury_engine.measurement import MeasurementEvidence

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_FLAGSHIP = "/api/v1/detect/flagship"
_REPLAY = "/api/v1/detect/flagship/replay"
_ROWS, _WIDTH = 80, 16  # the shipped checkpoint's feature width
#: Flag pairs: a flag is true exactly when the evidence beside it is present.
_EVIDENCE_OF = {
    "epistemic_measured": "epistemic_evidence",
    "aleatoric_measured": "aleatoric_evidence",
    "confidence_calibrated": "calibration_evidence",
    "novelty_measured": "novelty_evidence",
}


def _window(rng: np.random.Generator, *, far: bool = False) -> np.ndarray:
    """A flagged window: noise with a short burst the detector catches.

    ``far`` moves the whole window away from the others and plants a driver
    (X0 leads X1 at lag 1), which is what the discovery instruments should find.
    """
    window = rng.normal(size=(_ROWS, _WIDTH))
    if far:
        window += 4.0
        window[1:, 1] += 0.7 * window[:-1, 0]
    window[40:44] += 9.0
    return window


@pytest.fixture
def served(
    monkeypatch: pytest.MonkeyPatch, rate_limit_off: None
) -> Iterator[tuple[TestClient, dict[str, User | None]]]:
    """The app with a fresh served engine, and a switch for who is calling."""
    monkeypatch.setattr(detection, "_flagship_engine", None)
    monkeypatch.setattr(detection, "_flagship_ledger", None)
    caller: dict[str, User | None] = {"user": None}
    app.dependency_overrides[detection._get_optional_user] = lambda: caller["user"]
    try:
        yield TestClient(app), caller
    finally:
        app.dependency_overrides.pop(detection._get_optional_user, None)


def _detect(client: TestClient, window: np.ndarray) -> dict[str, Any]:
    response = client.post(_FLAGSHIP, json={"data": window.tolist()})
    assert response.status_code == 200, response.text[:300]
    body: dict[str, Any] = response.json()
    return body


def _discover(client: TestClient, seed: int = 0) -> tuple[dict[str, Any], np.ndarray]:
    """Enough flagged windows for a reference, then the one with something to find."""
    rng = np.random.default_rng(seed)
    for _ in range(24):
        assert _detect(client, _window(rng))["is_anomaly"] is True
    window = _window(rng, far=True)
    return _detect(client, window), window


class TestSignedInCaller:
    def test_the_decision_is_calibrated_sourced_and_claims_no_unmeasured_number(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, caller = served
        caller["user"] = User(id="u-1", username="analyst")
        result, _ = _discover(client)
        decision = result["decision"]

        # Calibrated, and against what.
        assert decision["calibrated"] is True
        assert result["conformal"]["set_size"] in (0, 1, 2)
        provenance = decision["signals"]["model_provenance"]
        assert provenance["conformal"]["n_calibration"] >= 100
        # Sourced: which model, by digest.
        assert len(provenance["checkpoint_sha256"]) == 64
        assert provenance["checkpoint"]

        # No unmeasured number presented as measured, in the payload or the record.
        for carrier in (result["cognitive_analysis"], decision["signals"]["cognitive"]):
            for flag, evidence in _EVIDENCE_OF.items():
                proof = MeasurementEvidence.from_dict(carrier[evidence])
                assert carrier[flag] is (proof is not None), flag
        cognitive = decision["signals"]["cognitive"]
        # One detection point measures no spread, and no calibrator was fitted: said so.
        assert cognitive["epistemic_measured"] is False
        assert cognitive["aleatoric_measured"] is False
        assert cognitive["confidence_calibrated"] is False
        # The novelty verdict IS measured: ranked among this caller's own reference.
        assert cognitive["novelty_measured"] is True
        assert cognitive["novelty_evidence"]["method"] == "conformal_rank"
        assert cognitive["novelty_evidence"]["n"] >= 20

    def test_the_discovery_artifacts_replay_through_the_api_and_can_fail(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, caller = served
        caller["user"] = User(id="u-1", username="analyst")
        result, window = _discover(client)
        decision = result["decision"]
        discovery = decision["signals"]["cognitive"]["discovery"]
        assert discovery["novelty"]["replayable"] == "from_record"
        assert discovery["novelty"]["claim"]["is_novel"] is True
        assert ("X0", "X1") in {
            (e["source"], e["target"]) for e in discovery["causal"]["claim"]["edges"]
        }
        assert decision["signals"]["cognitive"]["hypothesis"]["falsifier"]["withdrawn_if"]

        replayed = client.post(_REPLAY, json={"record": decision})
        assert replayed.status_code == 200, replayed.text[:300]
        body = replayed.json()
        assert body["verdict"] == D.HOLDS and body["artifacts"] == 3
        assert [r["status"] for r in body["replay"]] == [D.HOLDS] * 3

        # The window itself, as sent: the statistics are recomputed from it.
        with_data = client.post(_REPLAY, json={"record": decision, "data": window.tolist()})
        assert with_data.json()["verdict"] == D.HOLDS

        # A reference the observation resembles: no longer novel, and the hypothesis goes with it.
        observation = np.asarray(discovery["novelty"]["inputs"]["observation"])
        alike = observation + np.random.default_rng(1).normal(scale=0.5, size=(60, _WIDTH))
        challenged = client.post(
            _REPLAY, json={"record": decision, "reference": alike.tolist()}
        ).json()
        assert challenged["verdict"] == D.FAILS
        statuses = {r["kind"]: r["status"] for r in challenged["replay"]}
        assert statuses == {"novelty": D.FAILS, "causal": D.HOLDS, "hypothesis": D.FAILS}

        # An altered record.
        altered = copy.deepcopy(decision)
        altered["signals"]["cognitive"]["discovery"]["causal"]["claim"]["edges"] = []
        assert client.post(_REPLAY, json={"record": altered}).json()["verdict"] == D.FAILS

    def test_the_same_record_replays_offline(
        self, served: tuple[TestClient, dict[str, User | None]], tmp_path: Path
    ) -> None:
        client, caller = served
        caller["user"] = User(id="u-1", username="analyst")
        result, window = _discover(client)
        record = tmp_path / "result.json"
        record.write_text(json.dumps(result))
        data = tmp_path / "window.csv"
        np.savetxt(data, window, delimiter=",")

        runner = CliRunner()
        ok = runner.invoke(cli, ["replay", str(record)])
        assert ok.exit_code == 0, ok.output
        assert ok.output.count("HOLDS") == 3
        assert "X0 leads X1 at lag 1" in ok.output

        with_data = runner.invoke(cli, ["replay", str(record), "--data", str(data), "--json"])
        assert with_data.exit_code == 0
        assert json.loads(with_data.output)["exit_status"] == 0

        result["decision"]["signals"]["cognitive"]["discovery"]["novelty"]["claim"][
            "is_novel"
        ] = False
        record.write_text(json.dumps(result))
        failed = runner.invoke(cli, ["replay", str(record)])
        assert failed.exit_code == 1
        assert "FAILS" in failed.output

    def test_the_ledger_keeps_claims_and_digests_not_the_callers_numbers(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, caller = served
        caller["user"] = User(id="u-1", username="analyst")
        result, _ = _discover(client)
        issued = result["decision"]["signals"]["cognitive"]["discovery"]
        ledger = detection._flagship_ledger
        assert ledger is not None
        kept = list(ledger._entries)[-1].to_dict()["signals"]["cognitive"]["discovery"]
        for kind, artifact in issued.items():
            assert "inputs" not in kept[kind]
            assert kept[kind]["digest"] == artifact["digest"]
            assert kept[kind]["claim"] == artifact["claim"]

    def test_one_callers_reference_is_not_anothers(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, caller = served
        caller["user"] = User(id="u-1", username="analyst")
        _discover(client)
        # A second account sends the same kind of far window as its FIRST request.
        caller["user"] = User(id="u-2", username="other")
        fresh = _detect(client, _window(np.random.default_rng(5), far=True))
        cognitive = fresh["decision"]["signals"]["cognitive"]
        assert cognitive["novelty_measured"] is False  # nothing of its own to rank against
        assert "novelty" not in (cognitive["discovery"] or {})


class TestCallerWithoutCredentials:
    def test_the_shared_reference_is_not_handed_over(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, _caller = served  # nobody signed in
        result, _ = _discover(client)
        decision = result["decision"]
        novelty = decision["signals"]["cognitive"]["discovery"]["novelty"]
        # The verdict is measured and says on how many; the reference stays on the server.
        assert decision["signals"]["cognitive"]["novelty_measured"] is True
        assert novelty["replayable"] == "no"
        assert "not disclosed" in novelty["not_replayable_because"]
        assert set(novelty["inputs"]) == {
            "features",
            "reference_n",
            "calibration_n",
            "alpha",
            "observation_digest",
            "reference_digest",
        }
        body = client.post(_REPLAY, json={"record": decision}).json()
        statuses = {r["kind"]: r["status"] for r in body["replay"]}
        assert statuses["novelty"] == D.CANNOT_REPLAY  # never read as holding
        assert statuses["causal"] == D.HOLDS  # the caller's own window: still theirs to replay
        assert body["verdict"] == D.CANNOT_REPLAY


class TestReplayRoute:
    def test_a_record_with_nothing_to_replay_says_so(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, _ = served
        body = client.post(_REPLAY, json={"record": {"state": "grounded"}}).json()
        assert body == {"verdict": "NO_ARTIFACTS", "artifacts": 0, "replay": []}

    def test_a_ragged_matrix_is_a_client_error(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, _ = served
        response = client.post(_REPLAY, json={"record": {}, "data": [[1.0, 2.0], [3.0]]})
        assert response.status_code == 400

    def test_a_crafted_record_cannot_buy_more_work_than_the_instrument_does(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        """Fails on the prior code, which recomputed whatever the record named.

        The record is the caller's. One naming 64 columns at lag 50 did not
        finish in ten minutes when measured; this one is small enough to have
        been recomputed (132 pairs), and is refused all the same, by rule.
        """
        client, _ = served
        window = np.random.default_rng(3).normal(size=(60, 12))
        artifact = D.causal_artifact(window, max_lag=5, alpha=0.05)
        record = {"discovery": {"causal": artifact}}
        for body in ({"record": record, "data": window.tolist()}, {"record": record}):
            response = client.post(_REPLAY, json=body)
            assert response.status_code == 200, response.text[:300]
            answer = response.json()
            assert answer["verdict"] == D.CANNOT_REPLAY
            assert "at most 5 columns at lags up to 5" in answer["replay"][0]["detail"]
            assert "mercury-agent replay" in answer["replay"][0]["detail"]

    def test_a_replay_from_data_takes_a_flagship_slot_and_one_from_the_record_does_not(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        """Fails on the prior code: a recomputation ran outside every admission bound."""
        client, _ = served
        rng = np.random.default_rng(5)
        window = rng.normal(size=(120, 5))
        window[1:, 1] += 0.6 * window[:-1, 0]
        record = {"discovery": {"causal": D.causal_artifact(window, max_lag=5, alpha=0.05)}}
        before = detection._admission_stats.snapshot()

        held = 0
        while detection._flagship_admission.acquire(blocking=False):
            held += 1  # every slot taken, as by detections in flight
        try:
            assert held == detection._FLAGSHIP_MAX_QUEUE
            busy = client.post(_REPLAY, json={"record": record, "data": window.tolist()})
            assert busy.status_code == 429
            assert busy.headers["Retry-After"] == "1"
            # From the record alone it is arithmetic on a few numbers: answered regardless.
            light = client.post(_REPLAY, json={"record": record})
            assert light.status_code == 200 and light.json()["verdict"] == D.HOLDS
        finally:
            for _ in range(held):
                detection._flagship_admission.release()

        # With a slot free the same request is admitted, counted, and released.
        admitted = client.post(_REPLAY, json={"record": record, "data": window.tolist()})
        assert admitted.status_code == 200 and admitted.json()["verdict"] == D.HOLDS
        after = detection._admission_stats.snapshot()
        assert after["rejected_total"] == before["rejected_total"] + 1
        assert after["admitted_total"] == before["admitted_total"] + 1
        assert after["in_flight"] == before["in_flight"]

    def test_the_route_is_in_the_schema(
        self, served: tuple[TestClient, dict[str, User | None]]
    ) -> None:
        client, _ = served
        paths = client.get("/openapi.json").json()["paths"]
        assert "post" in paths[_REPLAY]


class TestReplayCommand:
    def test_exit_status_two_when_there_is_nothing_to_replay(self, tmp_path: Path) -> None:
        record = tmp_path / "bare.json"
        record.write_text(json.dumps({"state": "grounded"}))
        outcome = CliRunner().invoke(cli, ["replay", str(record)])
        assert outcome.exit_code == 2
        assert "nothing to replay" in outcome.output

    def test_a_file_that_is_not_json_is_an_error_not_a_traceback(self, tmp_path: Path) -> None:
        record = tmp_path / "broken.json"
        record.write_text("{not json")
        outcome = CliRunner().invoke(cli, ["replay", str(record)])
        assert outcome.exit_code == 2
        assert "cannot read" in outcome.output

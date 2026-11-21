# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""An upload becomes an ordinary Mercury decision, and nothing on the way is optional.

One committed sample of each kind (``examples/data/review``) is reviewed
through :func:`omni_mercury_engine.review.review_bytes` and must come back
with a valid ``DecisionRecord``.  The rest of the file is the proof that the
path cannot be shortened: it fails if a review skips the fusion engine, the
conformal certificate, the batch-family test, the harm-uplift gate or the
engine's own ethical gate.

No harmful text appears here.  A refusal is produced by a stand-in gate that
refuses everything, never by content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from omni_mercury_engine import review
from omni_mercury_engine.agentic.capabilities.contract import CONTRACT_MARKER, Invariant
from omni_mercury_engine.cognitive import decision_gate
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.decision import DecisionRecord
from omni_mercury_engine.engine import OmniMercuryEngine
from omni_mercury_engine.review import answers as review_answers
from tests.review import builders

SAMPLES = Path(__file__).resolve().parents[2] / "examples" / "data" / "review"
SAMPLE = {
    "document": SAMPLES / "sample_document.txt",
    "image": SAMPLES / "sample_image.png",
    "audio": SAMPLES / "sample_recording.wav",
}
KINDS = tuple(SAMPLE)


@pytest.fixture(scope="module")
def engines() -> dict[str, OmniMercuryEngine]:
    """The shipped review engines, one per kind, as a server holds them."""
    return {kind: review.build_engine(kind) for kind in KINDS}


@pytest.fixture(scope="module")
def reviewed(engines: dict[str, OmniMercuryEngine]) -> dict[str, dict[str, Any]]:
    """Each committed sample, reviewed once."""
    return {
        kind: review.review_bytes(path.read_bytes(), path.name, engine=engines[kind])
        for kind, path in SAMPLE.items()
    }


def _refuses_everything(*_args: Any, **_kwargs: Any) -> Any:
    """A stand-in harm gate that refuses whatever it is shown."""
    raise EthicalConstraintViolationError(
        action="review", score=0.0, threshold=1.0, check="harm_uplift", details={"stand_in": True}
    )


class TestASampleOfEachKindBecomesADecision:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_record_is_a_valid_decision_record(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        record = DecisionRecord.from_dict(reviewed[kind]["decision"])
        assert record.to_dict() == reviewed[kind]["decision"]
        assert record.calibrated is True
        assert record.coverage == pytest.approx(0.9)
        assert record.abstained == (record.decision_label is None)
        assert record.reasons and record.response.action

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_result_is_flagship_shaped(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        """A review result carries every key a flagship detection does, plus ``review``."""
        result = reviewed[kind]
        engine = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
        engine.load_default_fusion_checkpoint()
        engine.enable_decision_layer()
        flagship = engine.detect_with_fusion(np.random.default_rng(0).normal(size=(24, 16)))
        # A review also states what the harm gate read (the flagship caller
        # passed no gate, so its result carries no such statement).
        assert set(flagship) | {"review", "harm_gate_scope"} == set(result)
        assert set(flagship["decision"]) == set(result["decision"])
        assert set(flagship["decision"]["signals"]) == set(result["decision"]["signals"])

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_review_block_says_what_was_reviewed_and_what_was_flagged(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        import hashlib

        result, path = reviewed[kind], SAMPLE[kind]
        block = result["review"]
        assert block["schema"] == review.SCHEMA == "mercury.review/v1"
        assert block["kind"] == kind and block["filename"] == path.name
        assert block["bytes"] == path.stat().st_size
        assert block["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert block["segments"] == len(result["anomaly_probs"]) == len(result["is_anomaly_rows"])
        assert block["flagged_count"] == sum(bool(flag) for flag in result["is_anomaly_rows"])
        assert len(block["features"]) == 16 and block["basis"] == review.BASIS
        top = block["top_segment"]
        assert top["index"] == result["top_row"]
        assert top["probability"] == pytest.approx(max(result["anomaly_probs"]))
        assert len(top["top_features"]) == 3
        assert {f["name"] for f in top["top_features"]} <= set(block["features"])
        for segment in block["flagged_segments"]:
            assert result["is_anomaly_rows"][segment["index"]]

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_three_answers_are_stated_beside_the_record(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        result = reviewed[kind]
        stated = result["review"]["answers"]
        assert tuple(stated) == review_answers.KEYS
        for answer in stated.values():
            assert answer["plain"] and answer["read_from"]
        # Nothing is said that the result does not carry: recomputing gives the same answers.
        assert stated == review_answers.three_answers(result)
        decision = result["decision"]
        assert f"{decision['anomaly_prob']:.3f}" in stated["how_sure"]["plain"]
        assert f"'{decision['disposition']}'" in stated["the_record"]["plain"]
        assert result["review"]["sha256"][:16] in stated["the_record"]["plain"]
        text = review_answers.render(result, rerun="mercury-agent review FILE")
        for heading in review_answers.HEADINGS:
            assert heading in text
        assert "THE COMPLETE DECISION RECORD" in text
        assert text.rstrip().endswith("mercury-agent review FILE")

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_same_bytes_give_the_same_record(
        self, engines: dict[str, OmniMercuryEngine], reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        """What the record tells a reader to do to contest it: review the same bytes again."""
        again = review.review_bytes(
            SAMPLE[kind].read_bytes(), SAMPLE[kind].name, engine=engines[kind]
        )
        assert again["decision"] == reviewed[kind]["decision"]
        assert again["anomaly_probs"] == reviewed[kind]["anomaly_probs"]
        fresh = review.review_bytes(SAMPLE[kind].read_bytes(), SAMPLE[kind].name)
        assert fresh["decision"] == reviewed[kind]["decision"]


class TestTheFusionEngineDecides:
    """Fails if a review is answered by anything but ``detect_with_fusion`` on its segments."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_segment_matrix_is_what_the_engine_scores(
        self,
        engines: dict[str, OmniMercuryEngine],
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
    ) -> None:
        engine, seen = engines[kind], []
        scored = engine.detect_with_fusion

        def recording(matrix: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(np.array(matrix, copy=True))
            result: dict[str, Any] = scored(matrix, *args, **kwargs)
            return result

        monkeypatch.setattr(engine, "detect_with_fusion", recording)
        data = SAMPLE[kind].read_bytes()
        result = review.review_bytes(data, SAMPLE[kind].name, engine=engine)
        prepared = review.prepare(data, SAMPLE[kind].name)
        assert len(seen) == 1
        assert np.array_equal(seen[0], prepared.segmented.matrix)
        assert seen[0].shape[1] == review.WIDTH
        assert result["anomaly_probs"] and result["mode"]

    def test_the_decision_is_the_engines_not_the_reviews(
        self, engines: dict[str, OmniMercuryEngine], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = engines["document"]
        marker = {
            "decision": {"disposition": "from-the-engine"},
            "anomaly_probs": [0.0] * 48,
            "is_anomaly_rows": [False] * 48,
            "top_row": 0,
        }
        monkeypatch.setattr(engine, "detect_with_fusion", lambda *_a, **_k: dict(marker))
        monkeypatch.setattr(review, "three_answers", lambda _result: {})
        result = review.review_bytes(SAMPLE["document"].read_bytes(), "a.txt", engine=engine)
        assert result["decision"] == marker["decision"]

    @pytest.mark.parametrize("kind", KINDS)
    def test_each_kind_is_served_by_its_own_shipped_checkpoint(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        import hashlib

        provenance = reviewed[kind]["decision"]["signals"]["model_provenance"]
        path = review.checkpoint_path(kind)
        assert provenance["checkpoint"] == path.name == f"review_{kind}.pt"
        assert provenance["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


class TestTheConformalCertificateIsInThePath:
    """Fails if a review's decision stops depending on the certificate."""

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_certificate_is_batch_level_and_named(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        result = reviewed[kind]
        signals = result["decision"]["signals"]
        assert result["conformal"]["regime"] == signals["conformal_regime"] == "batch_max"
        assert signals["conformal_labels"] == result["conformal"]["prediction_set"]
        assert signals["model_provenance"]["conformal"]["n_calibration"] == 300
        assert signals["conformal_rows"] == result["review"]["segments"]

    @pytest.mark.parametrize(
        ("admits", "issued"),
        [((2.0, 2.0), [0, 1]), ((-1.0, -1.0), []), ((2.0, -1.0), [0]), ((-1.0, 2.0), [1])],
    )
    def test_the_prediction_set_decides_whether_a_review_is_grounded(
        self,
        engines: dict[str, OmniMercuryEngine],
        monkeypatch: pytest.MonkeyPatch,
        admits: tuple[float, float],
        issued: list[int],
    ) -> None:
        """Move only the certificate's thresholds: the decision follows the set it issues."""
        engine = engines["audio"]
        certificate = engine._fusion_conformal
        assert certificate is not None
        monkeypatch.setattr(certificate, "_thresholds", {0: admits[0], 1: admits[1]})
        result = review.review_bytes(SAMPLE["audio"].read_bytes(), "a.wav", engine=engine)
        assert result["conformal"]["prediction_set"] == issued
        decision = result["decision"]
        if len(issued) != 1:
            assert decision["abstained"] is True and decision["decision_label"] is None
        else:
            assert f"conformal singleton {{{issued[0]}}}" in decision["reasons"][0]
            if not decision["abstained"]:
                assert decision["decision_label"] == issued[0]

    def test_without_a_certificate_the_record_says_uncalibrated(
        self, engines: dict[str, OmniMercuryEngine], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = engines["image"]
        monkeypatch.setattr(engine, "_fusion_conformal", None)
        result = review.review_bytes(SAMPLE["image"].read_bytes(), "a.png", engine=engine)
        assert result["decision"]["calibrated"] is False
        assert result["decision"]["coverage"] is None

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_artifact_is_tested_against_the_calibration_family(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        family = reviewed[kind]["drift_detection"]
        assert family["status"] == "tested"
        assert family["reference_source"] == "calibration_batches:batch_family"
        assert family["reference_rows"] == 300
        assert 0.0 < family["p_value"] <= 1.0

    def test_an_upload_unlike_the_family_is_deferred_not_judged(
        self, engines: dict[str, OmniMercuryEngine]
    ) -> None:
        """A recording of pure noise is no voice: the certificate is not claimed for it."""
        noise = np.random.default_rng(3).uniform(-0.9, 0.9, size=16_000 * 8)
        result = review.review_bytes(builders.wav(noise), "noise.wav", engine=engines["audio"])
        assert result["drift_detection"]["is_drift"] is True
        assert result["decision"]["abstained"] is True
        assert "not one of that family" in result["review"]["answers"]["how_sure"]["plain"]

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_checkpoints_own_scope_statement_is_a_caveat_on_every_record(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        result = reviewed[kind]
        scope = result["review"]["certificate_scope"]
        assert scope and scope.startswith("calibrated on 300 reference ")
        assert "held-out" in scope and "unlike it" in scope
        assert f"certificate scope: {scope}" in result["decision"]["caveats"]
        assert scope in result["review"]["answers"]["how_sure"]["plain"]


class TestTheHarmGateReadsTheUploadFirst:
    """Fails if an upload can be measured or scored without the harm-uplift gate reading it."""

    def test_the_measuring_step_carries_the_gated_boundary_contract(self) -> None:
        """Deleting the annotation is a test failure, not a silent loss of control."""
        declared = getattr(review._measure, CONTRACT_MARKER, None)
        assert declared is not None and Invariant.GATED_BOUNDARY in declared

    def test_the_gate_is_shown_the_documents_own_text_and_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shown: list[Any] = []
        permitted = decision_gate.enforce_decision_boundary

        def watching(subject: Any, **kwargs: Any) -> Any:
            shown.append(subject)
            return permitted(subject, **kwargs)

        monkeypatch.setattr(decision_gate, "enforce_decision_boundary", watching)
        text = builders.prose(900)
        prepared = review.prepare(text.encode(), "harbour notes.txt")
        assert len(shown) == 1
        assert shown[0].request == text
        assert shown[0].payload == {"filename": "harbour notes.txt"}
        assert shown[0].surface == "omni_mercury_engine.review"
        assert "document" in shown[0].operation
        not_inspected = (
            "images embedded in the document or any text the decoder could not extract "
            "(scanned pages, text drawn as graphics)"
        )
        assert prepared.harm_gate == {
            "check": "harm_uplift",
            "passed": True,
            "characters_read": len(text) + len("harbour notes.txt"),
            "read": "the document's text and the upload's name",
            "content_inspected": True,
            "not_inspected": not_inspected,
            "statement": (
                f"the harm-uplift gate read this document's extracted text ({len(text):,} "
                f"characters) and the upload's name; it did not inspect {not_inspected}"
            ),
        }

    @pytest.mark.parametrize("kind", ["image", "audio"])
    def test_for_an_image_or_a_recording_it_reads_the_name_and_says_so(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        gate = reviewed[kind]["review"]["harm_gate"]
        assert gate["characters_read"] == len(SAMPLE[kind].name)
        assert "words in it are not read" in gate["read"]

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_refusal_stops_the_review_before_anything_is_measured_or_scored(
        self,
        engines: dict[str, OmniMercuryEngine],
        monkeypatch: pytest.MonkeyPatch,
        kind: str,
    ) -> None:
        measured: list[int] = []
        scored: list[int] = []
        monkeypatch.setattr(decision_gate, "enforce_decision_boundary", _refuses_everything)
        monkeypatch.setattr(review, "extract", lambda _decoded: measured.append(1))
        monkeypatch.setattr(engines[kind], "detect_with_fusion", lambda *_a, **_k: scored.append(1))
        with pytest.raises(EthicalConstraintViolationError) as refused:
            review.review_bytes(SAMPLE[kind].read_bytes(), SAMPLE[kind].name, engine=engines[kind])
        assert refused.value.check == "harm_uplift"
        assert measured == [] and scored == []

    def test_a_gate_that_cannot_decide_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed: an error inside the gate is a refusal, never a review."""

        def broken(*_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("the gate is unavailable")

        monkeypatch.setattr(decision_gate, "_assess", broken)
        with pytest.raises(EthicalConstraintViolationError):
            review.prepare(SAMPLE["document"].read_bytes(), "a.txt")

    def test_nothing_is_scored_that_was_not_prepared(
        self, engines: dict[str, OmniMercuryEngine]
    ) -> None:
        """``review_prepared`` takes what ``prepare`` returned; ``prepare`` is the gated step."""
        import inspect

        assert list(inspect.signature(review.review_prepared).parameters) == ["prepared", "engine"]
        source = inspect.getsource(review.prepare)
        assert source.index("decode(") < source.index("_measure(")
        assert "extract(" not in source  # measuring happens only inside the gated step


class TestTheRecordStatesWhatTheGateRead:
    """A gate that passed an upload it could only partly read has not cleared the rest.

    The harm-uplift gate reads text: an upload's name, and a document's
    extracted text.  It cannot see what an image shows or hear what a
    recording says, and ``passed`` alone would let a reader think it had.  The
    decision record carries the scope as a caveat, in the gate's own words.
    """

    @staticmethod
    def _scope(result: dict[str, Any]) -> list[str]:
        return [c for c in result["decision"]["caveats"] if c.startswith("harm gate scope: ")]

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_record_repeats_the_gates_own_statement(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        result = reviewed[kind]
        gate = result["review"]["harm_gate"]
        assert self._scope(result) == [f"harm gate scope: {gate['statement']}"]
        assert result["harm_gate_scope"] == gate["statement"]
        assert gate["not_inspected"] in gate["statement"]
        assert gate["not_inspected"] in result["review"]["answers"]["the_record"]["plain"]

    @pytest.mark.parametrize(
        ("kind", "unseen"),
        [
            ("image", "what the image shows or any words in it"),
            ("audio", "speech or any other sound in the recording"),
        ],
    )
    def test_an_image_or_a_recording_is_recorded_as_not_inspected(
        self, reviewed: dict[str, dict[str, Any]], kind: str, unseen: str
    ) -> None:
        gate = reviewed[kind]["review"]["harm_gate"]
        assert gate["passed"] is True and gate["content_inspected"] is False
        (caveat,) = self._scope(reviewed[kind])
        assert "read the upload's name only" in caveat
        assert f"it did not inspect {unseen}" in caveat

    def test_a_document_is_recorded_as_read_except_what_could_not_be_extracted(
        self, reviewed: dict[str, dict[str, Any]]
    ) -> None:
        gate = reviewed["document"]["review"]["harm_gate"]
        assert gate["content_inspected"] is True
        text_chars = gate["characters_read"] - len(SAMPLE["document"].name)
        (caveat,) = self._scope(reviewed["document"])
        assert f"read this document's extracted text ({text_chars:,} characters)" in caveat
        assert "it did not inspect images embedded in the document" in caveat

    def test_an_unnamed_image_is_recorded_as_read_by_nothing(
        self, engines: dict[str, OmniMercuryEngine]
    ) -> None:
        result = review.review_bytes(SAMPLE["image"].read_bytes(), "", engine=engines["image"])
        assert result["review"]["harm_gate"]["characters_read"] == 0
        (caveat,) = self._scope(result)
        assert "read nothing: the upload came without a name" in caveat
        assert "it did not inspect what the image shows" in caveat

    def test_a_detection_that_states_no_scope_has_no_such_caveat(
        self, engines: dict[str, OmniMercuryEngine], reviewed: dict[str, dict[str, Any]]
    ) -> None:
        matrix = review.prepare(SAMPLE["image"].read_bytes(), SAMPLE["image"].name).segmented
        result = engines["image"].detect_with_fusion(matrix.matrix)
        assert "harm_gate_scope" not in result
        assert self._scope(result) == []

    def test_the_scope_is_one_bounded_line_and_not_a_signal(self) -> None:
        from omni_mercury_engine.decision.evidence import Evidence

        base = {"anomaly_prob": 0.2, "is_anomaly": False}
        stated = Evidence.from_detection({**base, "harm_gate_scope": "  read the\nname  only "})
        assert stated.harm_gate_scope == "read the name only"
        long = Evidence.from_detection({**base, "harm_gate_scope": "x" * 5000}).harm_gate_scope
        assert long is not None and len(long) == 400
        for absent in (None, "", "   ", 7, ["read"]):
            assert (
                Evidence.from_detection({**base, "harm_gate_scope": absent}).harm_gate_scope is None
            )
        assert "harm_gate_scope" not in stated.to_dict()


class TestTheEnginesOwnGatesStillApply:
    """Fails if a review reaches a decision around the engine's ethical boundary."""

    def test_the_scoring_entry_point_is_a_gated_surface(self) -> None:
        declared = getattr(OmniMercuryEngine.detect_with_fusion, CONTRACT_MARKER, None)
        assert declared is not None and Invariant.GATED_BOUNDARY in declared

    def test_a_refusal_at_the_sigma_immutable_gate_refuses_the_review(
        self, engines: dict[str, OmniMercuryEngine], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = engines["document"]

        def refuse(*_args: Any, **_kwargs: Any) -> Any:
            raise EthicalConstraintViolationError(
                action="review", score=0.0, threshold=1.0, check="sigma_immutable", details={}
            )

        monkeypatch.setattr(engine._sigma_immutable_gate, "enforce", refuse)
        with pytest.raises(EthicalConstraintViolationError) as refused:
            review.review_bytes(SAMPLE["document"].read_bytes(), "a.txt", engine=engine)
        assert refused.value.check == "sigma_immutable"

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_record_carries_the_ethical_gates_outcome(
        self, reviewed: dict[str, dict[str, Any]], kind: str
    ) -> None:
        signals = reviewed[kind]["decision"]["signals"]
        assert signals["ethical_gate_passed"] is True
        assert signals["ethical_score"] >= signals["ethical_threshold"]


class TestWhatIsRefused:
    def test_a_kind_that_does_not_match_is_refused(self) -> None:
        with pytest.raises(review.ReviewError) as refused:
            review.review_bytes(SAMPLE["document"].read_bytes(), "a.txt", kind="audio")
        assert refused.value.status == 422

    def test_a_missing_checkpoint_is_a_503_not_a_guess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(review, "checkpoint_path", lambda kind: tmp_path / f"review_{kind}.pt")
        with pytest.raises(review.ReviewError) as refused:
            review.build_engine("image")
        assert refused.value.status == 503

    def test_the_review_engines_do_not_run_the_stream_critique(
        self, reviewed: dict[str, dict[str, Any]]
    ) -> None:
        """Its instruments assume a time-ordered stream; an artifact's segments are not one."""
        for result in reviewed.values():
            assert result["decision"]["signals"]["cognitive"] is None

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is published about artifact review is what was measured, on what ships.

``artifacts/artifact_review.json`` is written by
``research/review/harness_artifact_review.py``.  These tests tie everything
that quotes it back to it: the artifact is about the checkpoints in the
package (by SHA-256), each checkpoint's certificate-scope statement is the one
the artifact recorded, the README table is rendered from the artifact, the
Improvement Ledger's figures are the artifact's, and a bar that was missed is
recorded as missed.  They also guard the protocol itself: the fit, the
calibration and the evaluation never share a document source or a seed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from omni_mercury_engine import review
from omni_mercury_engine.governance.improvement_ledger import published
from research.review import (
    harness_artifact_review as harness,
    reference_artifacts as references,
)

KINDS = review.KINDS


@pytest.fixture(scope="module")
def artifact() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(harness.ARTIFACT.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def ledger() -> dict[str, Any]:
    data = published()
    assert data is not None
    return {decision["component"]: decision for decision in data["decisions"]}


class TestTheArtifactIsAboutWhatShips:
    def test_it_covers_the_three_kinds_under_the_stated_protocol(
        self, artifact: dict[str, Any]
    ) -> None:
        assert artifact["schema"] == harness.SCHEMA
        assert tuple(artifact["kinds"]) == KINDS
        assert artifact["bars"] == harness.BARS
        protocol = artifact["protocol"]
        assert protocol["target_coverage"] == harness.COVERAGE == 0.9
        assert protocol["held_out_artifacts_per_kind"] == 300
        assert protocol["seeds"] == harness.SEEDS

    @pytest.mark.parametrize("kind", KINDS)
    def test_it_measured_the_checkpoint_in_the_package(
        self, artifact: dict[str, Any], kind: str
    ) -> None:
        """Retraining a checkpoint without re-running the harness fails here."""
        path = review.checkpoint_path(kind)
        entry = artifact["kinds"][kind]
        assert entry["checkpoint"] == path.name
        assert entry["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    @pytest.mark.parametrize("kind", KINDS)
    def test_each_checkpoint_states_the_scope_the_artifact_recorded(
        self, artifact: dict[str, Any], kind: str
    ) -> None:
        engine = review.build_engine(kind)
        provenance = engine._fusion_provenance
        assert provenance is not None
        entry = artifact["kinds"][kind]
        assert provenance["certificate_scope"] == entry["certificate_scope"]
        assert entry["certificate_regime"] == engine._fusion_conformal_regime == "batch_max"
        assert entry["certificate"]["n_calibration"] == 300
        # The statement is built from the checkpoint's own held-out measurement.
        own = provenance["conformal_calibration"]["held_out"]["coverage"]
        assert f"{own['clean']:.1%} of clean" in provenance["certificate_scope"]
        assert f"{own['spliced']:.1%} of spliced" in provenance["certificate_scope"]
        floor = harness.BARS["coverage_per_class_min"]
        assert ("BELOW" in provenance["certificate_scope"]) == (min(own.values()) < floor)

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_counts_add_up(self, artifact: dict[str, Any], kind: str) -> None:
        held = artifact["kinds"][kind]["held_out"]
        decided = {}
        for side in ("clean", "spliced"):
            assert sum(held["decisions"][side].values()) == held["artifacts"][side] == 150
            decided[side] = held["artifacts"][side] - held["decisions"][side]["refused_by_gate"]
            assert held["refused_by_gate"][side] == held["decisions"][side]["refused_by_gate"]
        rates = held["rates"]
        assert rates["spliced_found"] == pytest.approx(
            held["decisions"]["spliced"]["grounded_anomalous"] / decided["spliced"]
        )
        assert rates["clean_grounded_wrong"] == pytest.approx(
            held["decisions"]["clean"]["grounded_anomalous"] / decided["clean"]
        )
        outside = artifact["kinds"][kind]["outside_the_family"]
        assert sum(outside["outcomes"].values()) == outside["artifacts"]


class TestABarThatWasMissedIsRecordedAsMissed:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_bars_are_judged_from_the_numbers(
        self, artifact: dict[str, Any], kind: str
    ) -> None:
        entry = artifact["kinds"][kind]
        held, bars = entry["held_out"], artifact["bars"]
        wrong = max(held["rates"]["clean_grounded_wrong"], held["rates"]["spliced_grounded_wrong"])
        assert entry["bars_met"] == {
            "coverage_per_class": min(held["coverage"]["clean"], held["coverage"]["spliced"])
            >= bars["coverage_per_class_min"],
            "grounded_wrong": wrong <= bars["grounded_wrong_max"],
            "row_auroc": held["row_auroc"] >= bars["row_auroc_min"],
        }
        assert entry["all_bars_met"] == all(entry["bars_met"].values())

    def test_the_users_bar_is_met_and_the_remaining_miss_is_recorded(
        self, artifact: dict[str, Any], ledger: dict[str, Any]
    ) -> None:
        """Spliced documents called clear is at or under 12%; the clean-side miss is recorded.

        At d39b42c the document checkpoint called 13.3% of spliced documents
        clear (bar 12%). After the descriptor retrain it is 7.3%. What the
        document kind misses now is the clean side -- coverage below the 85%
        floor and clean documents called anomalous above 12% -- because this
        repository's documentation prose carries numeric-dense passages that
        truly are unlike the rest of their document. The miss is recorded in
        the artifact, the ledger, the README table and the checkpoint's own
        scope statement; if a retrain cures it, the assertions on the miss
        fail and the prose is due a rewrite. If one regresses the cured bar,
        the honest move is the old one: record it, do not move the bar.
        """
        document = artifact["kinds"]["document"]
        rates = document["held_out"]["rates"]
        assert rates["spliced_grounded_wrong"] <= 0.12
        assert document["bars_met"]["row_auroc"] is True
        assert document["bars_met"]["coverage_per_class"] is False
        assert document["bars_met"]["grounded_wrong"] is False
        assert document["all_bars_met"] is False
        entry = ledger["review.artifact_checkpoints"]
        assert entry["verdict"] == "adopted"
        assert entry["impact"]["all_bars_met"]["document"] is False
        assert "not\na claim that the bar was cleared" in entry[
            "why_adopted_with_a_bar_unmet"
        ].replace("not a claim", "not\na claim")
        scope = document["certificate_scope"]
        assert "BELOW" in scope
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        assert "missed: per-class coverage (" in readme

    @pytest.mark.parametrize("kind", ["image", "audio"])
    def test_the_other_kinds_meet_every_bar(self, artifact: dict[str, Any], kind: str) -> None:
        entry = artifact["kinds"][kind]
        assert entry["all_bars_met"] is True, entry["bars_met"]

    def test_an_upload_unlike_the_family_is_deferred_not_judged(
        self, artifact: dict[str, Any]
    ) -> None:
        """Out-of-family uploads get a deferral, never a normal decision.

        At d39b42c the document family test deferred 3 of 17 Python source
        files uploaded as documents and cleared 13. The family now carries
        each calibration artifact's raw feature levels as a descriptor
        (standardizing an artifact against itself removes exactly what says
        which kind it is), and every out-of-family artifact of the three
        tried families is deferred.
        """
        for kind in KINDS:
            outside = artifact["kinds"][kind]["outside_the_family"]
            judged = (
                outside["outcomes"]["grounded_clear"] + outside["outcomes"]["grounded_anomalous"]
            )
            assert judged == 0, (kind, outside)
            assert outside["outcomes"]["abstained"] == outside["artifacts"] > 0
            assert outside["family_test_deferrals"] == outside["artifacts"]

    def test_the_document_measurement_names_the_corpus_it_shares_with_the_checkpoint(
        self, artifact: dict[str, Any]
    ) -> None:
        """Reference documents are cut from the repository's documentation.

        A measurement against a checkpoint trained on other documentation is
        a measurement of different documents; the artifact must record both
        digests and that they agree.
        """
        corpus = artifact["kinds"]["document"]["reference_corpus"]
        assert corpus["same"] is True
        assert corpus["measured_on"] == corpus["checkpoint_trained_on"]
        assert set(corpus["measured_on"]) == {"train", "calibration", "evaluation"}


class TestWhatQuotesTheArtifactMatchesIt:
    def test_the_readme_table_is_rendered_from_the_artifact(self, artifact: dict[str, Any]) -> None:
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        begin = readme.index(harness.README_BEGIN) + len(harness.README_BEGIN)
        published_table = readme[begin : readme.index(harness.README_END)].strip()
        assert published_table == harness.readme_table(artifact), (
            "README table is stale: python -m research.review.harness_artifact_review "
            "--update-readme"
        )

    def test_the_capability_matrix_rows_are_the_artifacts(self, artifact: dict[str, Any]) -> None:
        matrix = (REPO / "CAPABILITY_MATRIX.md").read_text(encoding="utf-8")
        nouns = {"document": "documents", "image": "images", "audio": "recordings"}
        for kind, noun in nouns.items():
            row = next(
                line
                for line in matrix.splitlines()
                if line.startswith(f"| Artifact review: {noun} |")
            )
            held = artifact["kinds"][kind]["held_out"]
            spliced, clean = held["decisions"]["spliced"], held["decisions"]["clean"]
            assert (
                f"**{spliced['grounded_anomalous']} / {spliced['grounded_clear']} / "
                f"{spliced['abstained']}; {clean['grounded_anomalous']} of 150; "
                f"{held['row_auroc']:.3f}**"
            ) in row
            assert ("all three pre-set bars met" in row) == artifact["kinds"][kind]["all_bars_met"]
        shipped = sorted(review.checkpoint_path("audio").parent.glob("*.pt"))
        assert len(shipped) == 13 and "Thirteen checkpoints ship" in matrix

    def test_the_ledger_quotes_the_artifacts_numbers(
        self, artifact: dict[str, Any], ledger: dict[str, Any]
    ) -> None:
        kinds = artifact["kinds"]
        checkpoints = ledger["review.artifact_checkpoints"]["impact"]
        certificate = ledger["engine.conformal_certificate.batch_level"]["impact"]
        family = ledger["engine.drift_detection.batch_family"]["impact"]
        for kind in KINDS:
            held = kinds[kind]["held_out"]
            assert checkpoints["bars_met"][kind] == kinds[kind]["bars_met"]
            assert checkpoints["spliced_found"][kind] == pytest.approx(
                held["rates"]["spliced_found"]
            )
            assert checkpoints["spliced_called_clear"][kind] == pytest.approx(
                held["rates"]["spliced_grounded_wrong"]
            )
            assert checkpoints["clean_called_anomalous"][kind] == pytest.approx(
                held["rates"]["clean_grounded_wrong"]
            )
            assert checkpoints["row_auroc"][kind] == pytest.approx(held["row_auroc"])
            assert certificate["review_coverage_clean"][kind] == pytest.approx(
                held["coverage"]["clean"]
            )
            assert certificate["review_coverage_spliced"][kind] == pytest.approx(
                held["coverage"]["spliced"]
            )
            assert family["in_family_deferrals_of_300"][kind] == held["family_test_deferrals"]
            outside = kinds[kind]["outside_the_family"]
            assert family["unlike_the_family"][kind] == {
                "artifacts": outside["artifacts"],
                **outside["outcomes"],
            }
        assert checkpoints["bars"] == artifact["bars"]


class TestTheProtocolKeepsItsSetsApart:
    def test_no_document_source_is_shared_between_fit_calibration_and_evaluation(
        self, artifact: dict[str, Any]
    ) -> None:
        splits = {
            split: {path.name for path in references.prose_files(split)}
            for split in ("train", "calibration", "evaluation")
        }
        assert all(splits.values())
        assert not splits["train"] & splits["calibration"]
        assert not splits["train"] & splits["evaluation"]
        assert not splits["calibration"] & splits["evaluation"]
        recorded = artifact["protocol"]["document_sources"]
        assert {split: set(names) for split, names in recorded.items()} == splits

    def test_every_stage_draws_with_its_own_seed(self) -> None:
        seeds = list(harness.SEEDS.values())
        assert len(set(seeds)) == len(seeds) == 5
        # The outside-the-family draws use seed + 1 of their stage.
        assert not {seed + 1 for seed in seeds} & set(seeds)

    def test_no_review_source_is_a_harm_policy_document(self) -> None:
        """Reference prose is ordinary documentation: the gate must have nothing to refuse."""
        for split in ("train", "calibration", "evaluation"):
            for path in references.prose_files(split):
                assert not any(word in path.name.upper() for word in references._EXCLUDED_DOCS)

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_generators_are_reproducible(self, kind: str) -> None:
        first = [
            references.to_bytes(r)[0] for r in references.references(kind, 2, 123, "evaluation")
        ]
        again = [
            references.to_bytes(r)[0] for r in references.references(kind, 2, 123, "evaluation")
        ]
        assert first == again
        other = [
            references.to_bytes(r)[0] for r in references.references(kind, 2, 124, "evaluation")
        ]
        assert first != other


class TestTheCommittedSamples:
    def test_they_are_the_reference_artifacts_the_manifest_names(self) -> None:
        samples = REPO / "examples" / "data" / "review"
        manifest = json.loads((samples / "planted.json").read_text(encoding="utf-8"))
        assert set(manifest) == set(references.SAMPLE_NAMES.values())
        for kind, name in references.SAMPLE_NAMES.items():
            reference = references.sample(kind)
            entry = manifest[name]
            assert reference.spliced and entry["planted"] == reference.recipe
            assert entry["planted_at"] == references.planted(reference)
            pinned = (entry["reference"]["seed"], entry["reference"]["index"])
            assert pinned == references.SAMPLES[kind]
            committed = review.decode((samples / name).read_bytes(), name)
            regenerated = review.decode(*references.to_bytes(reference))
            if kind == "document":
                assert committed.text == regenerated.text
            elif kind == "image":
                assert committed.pixels is not None and regenerated.pixels is not None
                assert np.array_equal(committed.pixels, regenerated.pixels)
            else:
                assert committed.samples is not None and regenerated.samples is not None
                assert np.array_equal(committed.samples, regenerated.samples)

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_review_flags_what_was_planted(self, kind: str) -> None:
        """The samples' README says each review is grounded and lands in the planted spans."""
        samples = REPO / "examples" / "data" / "review"
        name = references.SAMPLE_NAMES[kind]
        data = (samples / name).read_bytes()
        result = review.review_bytes(data, name)
        decision = result["decision"]
        assert decision["abstained"] is False and decision["decision_label"] == 1
        truth = references.segment_labels(
            review.extract(review.decode(data, name)), references.sample(kind)
        )
        flagged = [index for index, flag in enumerate(result["is_anomaly_rows"]) if flag]
        assert flagged and all(truth[index] for index in flagged)
        assert truth[result["top_row"]]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The detector ontology is complete, controlled, and evidence-tied.

``core/detector_ontology.py`` answers, for every manifest detector, the
question professionals actually ask: *what anomaly class is this detector
sensitive to, and where has that been validated?* These tests make the
answers enforceable:

* every manifest entry has exactly one ontology row, and no row is orphaned;
* all anomaly-class and domain terms come from the controlled vocabularies;
* a detector never simultaneously claims and disclaims a class;
* every validation claim cites a committed evidence path that exists -
  a claim without evidence cannot ship;
* a detector with no validation carries an explicit not-validated note,
  so "unmeasured" is stated, never implied away;
* measured blind spots stay declared: the scale-invariant exotics must
  keep ``amplitude_collapse`` in ``cannot_detect`` as long as the
  measured pin in ``tests/models/test_fitted_reference_models.py`` holds.
"""

from __future__ import annotations

from pathlib import Path

from omni_mercury_engine.core.detector_ontology import (
    ANOMALY_CLASSES,
    DOMAINS,
    ONTOLOGY,
)
from omni_mercury_engine.core.detector_registry import DETECTOR_MANIFEST

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_NAMES = {entry.name for entry in DETECTOR_MANIFEST}


class TestCoverage:
    def test_every_manifest_entry_has_an_ontology_row(self) -> None:
        missing = sorted(_MANIFEST_NAMES - set(ONTOLOGY))
        assert not missing, f"manifest entries without a declared ontology: {missing}"

    def test_no_ontology_row_is_orphaned(self) -> None:
        orphans = sorted(set(ONTOLOGY) - _MANIFEST_NAMES)
        assert not orphans, f"ontology rows for detectors not in the manifest: {orphans}"


class TestVocabularyControl:
    def test_detects_terms_are_controlled_and_nonempty(self) -> None:
        for name, row in ONTOLOGY.items():
            assert row.detects, f"{name}: a detector that detects nothing is not a detector"
            unknown = set(row.detects) - ANOMALY_CLASSES
            assert not unknown, f"{name}: detects terms outside the vocabulary: {sorted(unknown)}"

    def test_cannot_detect_terms_are_controlled(self) -> None:
        for name, row in ONTOLOGY.items():
            unknown = set(row.cannot_detect) - ANOMALY_CLASSES
            assert not unknown, f"{name}: cannot_detect outside the vocabulary: {sorted(unknown)}"

    def test_no_class_is_both_claimed_and_disclaimed(self) -> None:
        for name, row in ONTOLOGY.items():
            overlap = set(row.detects) & set(row.cannot_detect)
            assert not overlap, f"{name}: claims and disclaims {sorted(overlap)}"

    def test_validated_domains_are_controlled(self) -> None:
        for name, row in ONTOLOGY.items():
            for claim in row.validated:
                assert claim.domain in DOMAINS, f"{name}: unknown domain {claim.domain!r}"


class TestEvidenceDiscipline:
    def test_every_validation_claim_cites_committed_evidence(self) -> None:
        for name, row in ONTOLOGY.items():
            for claim in row.validated:
                path = _REPO_ROOT / claim.evidence
                assert path.exists(), (
                    f"{name}: validation claim for {claim.domain!r} cites "
                    f"{claim.evidence!r}, which does not exist in the repo - "
                    "a claim without committed evidence cannot ship"
                )

    def test_unvalidated_detectors_say_so_explicitly(self) -> None:
        for name, row in ONTOLOGY.items():
            if not row.validated:
                assert row.not_validated_note.strip(), (
                    f"{name}: no validation evidence and no not_validated_note - "
                    "'unmeasured' must be stated, never implied away"
                )


class TestMeasuredBlindSpotsStayDeclared:
    def test_scale_invariant_exotics_declare_amplitude_collapse(self) -> None:
        """Ties to TestPlantedBurstCompetency.test_declared_blind_spot_* in
        tests/models/test_fitted_reference_models.py: if that measured pin is
        flipped into a competency floor one day, remove the declaration here
        in the same commit."""
        for name in ("quantum", "consciousness"):
            assert "amplitude_collapse" in ONTOLOGY[name].cannot_detect, (
                f"{name}: the measured scale-invariance blind spot must stay "
                "declared while the measurement holds"
            )

    def test_astrophysical_claims_amplitude_collapse_it_measured(self) -> None:
        assert "amplitude_collapse" in ONTOLOGY["astrophysical"].detects

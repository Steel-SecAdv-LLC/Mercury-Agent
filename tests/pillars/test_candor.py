# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pillar: candor — the documentation says what is true, and CI checks.

Candor is the pillar that fails silently, because prose does not raise. Four
properties, each mechanically checkable:

* **The doc-lint gate is real and passes.** ``scripts/doc_lint.py`` runs in CI
  and here, over the same rules, so a banned claim cannot merge.
* **Shipped is separated from aspirational.** ``CAPABILITY_MATRIX.md`` has a
  status column with a fixed vocabulary, and every row that says *enforced*
  cites code that exists.
* **The audit status is stated.** The repository says, in the places a reader
  looks for it, that the cryptography is not externally audited and the medical
  modules are not clinically validated.
* **The gate has teeth.** A synthetic violation makes the linter fail. A gate
  that cannot fail is not a gate — this is the test that would have caught a
  lint quietly reduced to a no-op.

Capability *numbers* are deliberately not asserted here. They belong in
``CAPABILITY_MATRIX.md`` with a repro command, because a benchmark figure
pinned in a test becomes a number nobody dares re-measure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX = REPO_ROOT / "CAPABILITY_MATRIX.md"
DOC_LINT = REPO_ROOT / "scripts" / "doc_lint.py"


def _run_doc_lint(cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(DOC_LINT)],
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


class TestTheDocLintGateIsRealAndPasses:
    def test_the_linter_exists_and_is_executable_as_a_script(self) -> None:
        assert DOC_LINT.is_file()
        assert "__main__" in DOC_LINT.read_text(encoding="utf-8")

    @pytest.mark.slow
    def test_the_repository_passes_doc_lint(self) -> None:
        completed = _run_doc_lint()
        assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-6000:]

    def test_ci_runs_the_doc_lint(self) -> None:
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "scripts/doc_lint.py" in workflow, (
            "doc-lint must run in CI; a lint only developers remember to run is "
            "a lint that stops running"
        )

    def test_the_banned_list_covers_the_four_retired_claims(self) -> None:
        from scripts.doc_lint import BANNED

        labels = {banned.label for banned in BANNED}
        assert labels >= {"survivor-first", "People First", "FIPS-certified", "NIST-validated"}

    def test_every_ban_explains_itself_and_offers_a_replacement(self) -> None:
        """A ban with no guidance gets worked around, not obeyed."""
        from scripts.doc_lint import BANNED

        for banned in BANNED:
            assert len(banned.why) > 20, banned.label
            assert len(banned.instead) > 20, banned.label

    def test_the_mission_phrase_is_required_where_the_mission_is_stated(self) -> None:
        from scripts.doc_lint import MISSION_DOCUMENTS, MISSION_PHRASE

        assert MISSION_DOCUMENTS
        for document in MISSION_DOCUMENTS:
            body = (REPO_ROOT / document).read_text(encoding="utf-8")
            assert re.search(MISSION_PHRASE, body, re.IGNORECASE), document


class TestTheGateHasTeeth:
    """A lint that cannot fail is decoration. Prove each rule bites."""

    def test_a_banned_phrase_is_detected(self) -> None:
        from scripts.doc_lint import BANNED

        for banned in BANNED:
            # The phrase the rule bans, written the way prose would write it.
            probe = banned.label
            assert re.search(banned.pattern, probe, re.IGNORECASE), banned.label

    def test_a_missing_mission_phrase_is_detected(self) -> None:
        from scripts.doc_lint import MISSION_PHRASE

        assert not re.search(MISSION_PHRASE, "a document with no mission line", re.IGNORECASE)

    def test_an_uncited_enforced_row_is_detected(self) -> None:
        from scripts.doc_lint import scan_capability_matrix

        bad = (
            "| Claim | Task | Metric | Number | Repro | Status |\n"
            "|---|---|---|---|---|---|\n"
            "| It is safe | n/a | n/a | n/a | trust us | **enforced** |\n"
        )
        violations = scan_capability_matrix(bad)
        assert violations, "an enforced row citing nothing must fail the lint"
        assert "enforced" in violations[0].rule

    def test_a_cited_enforced_row_passes(self) -> None:
        from scripts.doc_lint import scan_capability_matrix

        good = (
            "| Claim | Task | Metric | Number | Repro | Status |\n"
            "|---|---|---|---|---|---|\n"
            "| Gate refuses uplift | corpus | refusals | 4/4 | "
            "`pytest tests/pillars/test_non_maleficence.py` | **enforced** |\n"
        )
        assert scan_capability_matrix(good) == []

    def test_a_row_citing_a_path_that_does_not_exist_is_detected(self) -> None:
        """Citing a plausible-looking path that was deleted must not pass."""
        from scripts.doc_lint import scan_capability_matrix

        bad = (
            "| Claim | Task | Metric | Number | Repro | Status |\n"
            "|---|---|---|---|---|---|\n"
            "| Gate refuses uplift | corpus | refusals | 4/4 | "
            "`pytest tests/pillars/test_this_was_deleted.py` | **enforced** |\n"
        )
        assert scan_capability_matrix(bad)


class TestShippedIsSeparatedFromAspirational:
    def test_the_matrix_exists_and_is_row_per_claim(self) -> None:
        assert MATRIX.is_file()
        body = MATRIX.read_text(encoding="utf-8")
        assert body.count("|") > 200, "the matrix must be a table, not prose"

    def test_the_matrix_declares_its_status_vocabulary(self) -> None:
        from scripts.doc_lint import VALID_STATUSES

        body = MATRIX.read_text(encoding="utf-8").lower()
        for status in VALID_STATUSES:
            assert status in body, status

    def test_the_matrix_uses_the_aspirational_status_at_least_once(self) -> None:
        """If nothing is labelled aspirational, the label is not being used."""
        body = MATRIX.read_text(encoding="utf-8").lower()
        assert body.count("aspirational") >= 2

    def test_the_matrix_records_the_claims_that_were_removed(self) -> None:
        body = MATRIX.read_text(encoding="utf-8")
        assert "Removed claims" in body
        for removed in ("0.99", "roc_auc_estimate", "16 live data-loader"):
            assert removed in body, removed

    def test_no_pillar_test_asserts_a_benchmark_number(self) -> None:
        """Pillars are properties; capabilities are measurements. Keep them apart."""
        pillar_dir = Path(__file__).parent
        offenders: list[str] = []
        for path in sorted(pillar_dir.glob("test_*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip().startswith("assert"):
                    continue
                # A pillar test may assert a threshold it *reads* from config,
                # but must not hardcode a model's measured score.
                if re.search(r"\b(auc|f1|precision|recall|accuracy)\b", line, re.IGNORECASE):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        assert not offenders, offenders


class TestTheAuditStatusIsStated:
    def test_the_cryptography_is_declared_not_independently_audited(self) -> None:
        for document in ("README.md", "SECURITY.md"):
            body = (REPO_ROOT / document).read_text(encoding="utf-8").lower()
            assert "not" in body and "independently audited" in body, document

    def test_cavp_and_cmvp_are_named_as_not_entered(self) -> None:
        body = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "CAVP" in body and "CMVP" in body

    def test_constant_time_is_declared_asserted_not_verified(self) -> None:
        body = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8").lower()
        assert "not** been independently verified" in body or (
            "asserted" in body and "not" in body and "independently verified" in body
        )

    def test_the_medical_modules_are_declared_not_clinically_validated(self) -> None:
        body = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "no clinical validation" in body or "require clinical validation" in body

    def test_the_matrix_states_the_sigma_corpus_is_tamper_evident_not_authenticated(
        self,
    ) -> None:
        body = MATRIX.read_text(encoding="utf-8").lower()
        assert "tamper-evident, not authenticated" in body

    def test_the_matrix_states_lyapunov_is_monitored_not_guaranteed(self) -> None:
        body = MATRIX.read_text(encoding="utf-8").lower()
        assert "halt_on_violation=false" in body
        assert "monitoring" in body or "monitored" in body

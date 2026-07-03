# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the rolling weapons-gate corpus ingestion CLI (stdlib-only, no PQC).

Covers schema validation, canonical serialization, dedup, held-out-adversarial
disjointness, manifest/versioning, and the integrity ``--check`` gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ingest_weapons_gate_corpus as ing


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


class TestValidateRow:
    def test_valid_row_normalizes_and_autosplits(self) -> None:
        row = ing.validate_row({"text": "  hello world  ", "label": "benign"}, where="t")
        assert row["text"] == "hello world"
        assert row["expected"] == "allow"  # derived from benign
        assert row["split"] in {"train", "val", "test"}
        assert row["tags"] == []

    def test_offensive_derives_block(self) -> None:
        row = ing.validate_row({"text": "x", "label": "offensive"}, where="t")
        assert row["expected"] == "block"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ing.CorpusValidationError, match="empty 'text'"):
            ing.validate_row({"text": "   ", "label": "benign"}, where="t")

    def test_bad_label_rejected(self) -> None:
        with pytest.raises(ing.CorpusValidationError, match="label"):
            ing.validate_row({"text": "x", "label": "spicy"}, where="t")

    def test_label_expected_disagreement_rejected(self) -> None:
        with pytest.raises(ing.CorpusValidationError, match="disagree"):
            ing.validate_row({"text": "x", "label": "benign", "expected": "block"}, where="t")

    def test_bad_split_rejected(self) -> None:
        with pytest.raises(ing.CorpusValidationError, match="split"):
            ing.validate_row({"text": "x", "label": "benign", "split": "prod"}, where="t")

    def test_non_list_tags_rejected(self) -> None:
        with pytest.raises(ing.CorpusValidationError, match="tags"):
            ing.validate_row({"text": "x", "label": "benign", "tags": "nope"}, where="t")


class TestSerialization:
    def test_dump_is_sorted_and_canonical(self) -> None:
        rows = [
            {"text": "b", "label": "benign", "expected": "allow", "split": "train", "tags": []},
            {"text": "a", "label": "benign", "expected": "allow", "split": "train", "tags": []},
        ]
        out = ing.dump_rows(rows)
        lines = out.splitlines()
        assert json.loads(lines[0])["text"] == "a"  # sorted by text
        assert out.endswith("\n")

    def test_sha256_is_stable_and_order_independent(self) -> None:
        a = [
            {"text": "a", "label": "benign", "expected": "allow", "split": "train", "tags": []},
            {"text": "b", "label": "benign", "expected": "allow", "split": "train", "tags": []},
        ]
        assert ing.corpus_sha256(a) == ing.corpus_sha256(list(reversed(a)))


class TestIngest:
    def test_dedup_and_add(self, tmp_path: Path) -> None:
        existing = ing._current_corpus()
        dup_text = existing[0]["text"]
        add = _write_jsonl(
            tmp_path / "add.jsonl",
            [
                {"text": dup_text, "label": existing[0]["label"]},
                {"text": "a wholly novel benign example about gardening tomatoes", "label": "benign"},
            ],
        )
        merged, added, skipped = ing.ingest(add)
        assert added == 1
        assert skipped == 1
        assert len(merged) == len(existing) + 1

    def test_adversarial_collision_refused(self, tmp_path: Path) -> None:
        adv = ing.adversarial_texts()
        assert adv, "adversarial corpus should be non-empty"
        one = next(iter(adv))
        add = _write_jsonl(tmp_path / "adv.jsonl", [{"text": one, "label": "offensive"}])
        with pytest.raises(ing.CorpusValidationError, match="adversarial"):
            ing.ingest(add)


class TestManifestAndCheck:
    def test_manifest_counts(self) -> None:
        rows = ing._current_corpus()
        manifest = ing.build_manifest(rows)
        assert manifest["n_cases"] == len(rows)
        assert manifest["corpus_version"].startswith(f"{len(rows)}-")
        assert sum(manifest["per_label"].values()) == len(rows)
        assert manifest["per_label"]["offensive"] >= ing._MIN_PER_CLASS
        assert manifest["per_label"]["benign"] >= ing._MIN_PER_CLASS

    def test_committed_corpus_is_consistent(self) -> None:
        # The committed corpus + manifest must agree (integrity gate is green).
        assert ing.check() == []

    def test_cli_validate_ok_and_bad(self, tmp_path: Path) -> None:
        good = _write_jsonl(tmp_path / "g.jsonl", [{"text": "x", "label": "benign"}])
        assert ing.main(["--validate", str(good)]) == 0
        bad = _write_jsonl(tmp_path / "b.jsonl", [{"text": "x", "label": "nope"}])
        assert ing.main(["--validate", str(bad)]) == 2

    def test_cli_check_passes(self) -> None:
        assert ing.main(["--check"]) == 0

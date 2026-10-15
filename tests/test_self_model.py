# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Behavioural tests for Mercury's durable, hash-chained, signable self-model."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from omni_mercury_engine import self_model
from omni_mercury_engine.self_model import (
    GENESIS_DIGEST,
    CapabilitySpec,
    SelfModelLedger,
    build_self_model,
    diff_snapshots,
)

if TYPE_CHECKING:
    from pathlib import Path

STAMP = "2026-10-15T00:00:00+00:00"


class TestSnapshot:
    def test_identity_states_mission_codes_and_honesty_contract(self) -> None:
        identity = build_self_model(STAMP).body["identity"]
        assert identity["mission"] == "civilization-first"
        assert identity["code_system"] == "Omni Codes"
        assert identity["family"] == ["FINDOYOU", "Mercury Agent", "AMA Cryptography"]
        assert identity["honesty_contract"] == ["grounded", "unavailable", "undecidable"]

    def test_snapshot_is_reproducible_for_the_same_installation(self) -> None:
        assert build_self_model(STAMP).digest == build_self_model(STAMP).digest

    def test_every_capability_has_a_three_state_and_a_reason(self) -> None:
        capabilities = build_self_model(STAMP).body["capabilities"]
        assert set(capabilities) == {spec.name for spec in self_model.CAPABILITIES}
        for record in capabilities.values():
            assert record["state"] in {"grounded", "unavailable", "undecidable"}
            assert record["reason"]

    def test_a_missing_dependency_is_unavailable_and_becomes_a_limitation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = CapabilitySpec(
            "fictional_sensor_fusion",
            "needs a package that is not installed",
            ("omni_mercury_engine.vision.analyzer",),
            requires=("definitely_not_an_installed_package",),
        )
        monkeypatch.setattr(self_model, "CAPABILITIES", (*self_model.CAPABILITIES, spec))
        body = build_self_model(STAMP).body
        record = body["capabilities"]["fictional_sensor_fusion"]
        assert record["state"] == "unavailable"
        assert "definitely_not_an_installed_package" in record["reason"]
        assert any(lim.startswith("fictional_sensor_fusion:") for lim in body["limitations"])

    def test_identity_verification_needs_a_face_backend(self) -> None:
        record = build_self_model(STAMP).body["capabilities"]["biometric_match"]
        if not self_model._module_present("deepface"):
            assert record["state"] == "unavailable"
            assert "deepface" in record["reason"]

    def test_measured_performance_is_pinned_to_its_artifact(self) -> None:
        metrics = build_self_model(STAMP).body["measured_performance"]
        dtu = metrics["dtu_ais_abnormal"]
        assert 0.5 < dtu["ensemble_auc"] < 1.0
        assert dtu["label_source"] == "expert_annotated"
        assert len(dtu["sha256"]) == 64

    def test_weak_benchmark_categories_become_stated_limitations(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "benchmarks" / "results").mkdir(parents=True)
        (tmp_path / "benchmarks" / "results" / "mvtec_visual_benchmark.json").write_text(
            json.dumps(
                {
                    "protocol": {"alpha": 0.05},
                    "categories": {"grid": {"image_auroc": 0.66}, "leather": {"image_auroc": 0.98}},
                    "summary": {"mean_false_alarm_rate_at_alpha": 0.12},
                }
            )
        )
        monkeypatch.setattr(self_model, "_REPO_ROOT", tmp_path)
        limitations = build_self_model(STAMP).body["limitations"]
        assert any("'grid'" in lim and "0.660" in lim for lim in limitations)
        assert not any("'leather'" in lim for lim in limitations)
        assert any("in-session calibration" in lim for lim in limitations)


class TestLedger:
    def test_append_chains_records_from_genesis(self, tmp_path: Path) -> None:
        ledger = SelfModelLedger(tmp_path / "self.jsonl")
        first = ledger.append(build_self_model(STAMP))
        second = ledger.append(build_self_model("2026-10-16T00:00:00+00:00"))
        assert first["prev_digest"] == GENESIS_DIGEST
        assert second["prev_digest"] == first["record_digest"]
        assert ledger.verify() == (True, "valid")
        assert ledger.latest() == second["snapshot"]

    @pytest.mark.parametrize("attack", ["edit", "delete", "reorder"])
    def test_any_tampering_breaks_the_chain(self, tmp_path: Path, attack: str) -> None:
        path = tmp_path / "self.jsonl"
        ledger = SelfModelLedger(path)
        for day in ("15", "16", "17"):
            ledger.append(build_self_model(f"2026-10-{day}T00:00:00+00:00"))
        lines = path.read_text().splitlines()
        if attack == "edit":
            record = json.loads(lines[1])
            record["snapshot"]["identity"]["mission"] = "something else"
            lines[1] = json.dumps(record, sort_keys=True)
        elif attack == "delete":
            del lines[1]
        else:
            lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines) + "\n")
        ok, reason = ledger.verify()
        assert not ok
        assert "record" in reason

    def test_a_broken_chain_is_never_extended(self, tmp_path: Path) -> None:
        path = tmp_path / "self.jsonl"
        ledger = SelfModelLedger(path)
        ledger.append(build_self_model(STAMP))
        record = json.loads(path.read_text())
        record["index"] = 7
        path.write_text(json.dumps(record) + "\n")
        with pytest.raises(ValueError, match="broken self-model ledger"):
            ledger.append(build_self_model(STAMP))

    def test_signed_ledger_verifies_only_against_the_pinned_key(self, tmp_path: Path) -> None:
        from omni_mercury_engine.security.crypto_api import MLDSAProvider

        pair = MLDSAProvider().generate_keypair()
        other = MLDSAProvider().generate_keypair()
        ledger = SelfModelLedger(tmp_path / "self.jsonl")
        ledger.append(build_self_model(STAMP), signing_keypair=(pair.public_key, pair.secret_key))
        assert ledger.verify(pair.public_key.hex()) == (True, "valid")
        ok, reason = ledger.verify(other.public_key.hex())
        assert not ok and "pinned key" in reason

    def test_missing_ledger_is_empty_and_valid(self, tmp_path: Path) -> None:
        ledger = SelfModelLedger(tmp_path / "absent.jsonl")
        assert ledger.records() == []
        assert ledger.latest() is None
        assert ledger.verify() == (True, "valid")

    def test_default_path_honours_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MERCURY_SELF_MODEL_PATH", str(tmp_path / "x.jsonl"))
        assert SelfModelLedger().path == tmp_path / "x.jsonl"


class TestSelfKnowledgeOverTime:
    def test_diff_reports_capability_loss_metric_drift_and_new_limitations(self) -> None:
        old = build_self_model(STAMP).body
        new = json.loads(json.dumps(old))
        new["capabilities"]["post_quantum_signing"]["state"] = "unavailable"
        new["measured_performance"]["dtu_ais_abnormal"]["ensemble_auc"] = 0.70
        new["limitations"].append("post_quantum_signing: missing: ama_cryptography")
        new["safeguards"]["pqc_backend_present"] = False
        changes = diff_snapshots(old, new)
        assert changes["capabilities_lost"] == ["post_quantum_signing"]
        assert changes["metric_changes"]["dtu_ais_abnormal.ensemble_auc"][1] == 0.70
        assert changes["safeguard_changes"]["pqc_backend_present"] == [True, False]
        assert changes["limitations_added"] == ["post_quantum_signing: missing: ama_cryptography"]
        assert changes["version_change"] is None

    def test_identical_snapshots_have_no_changes(self) -> None:
        body = build_self_model(STAMP).body
        changes = diff_snapshots(body, body)
        assert not any(changes.values())

    def test_cli_appends_and_verifies(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ledger = tmp_path / "cli.jsonl"
        assert self_model.main(["--append", "--ledger", str(ledger)]) == 0
        assert "record_digest" in json.loads(capsys.readouterr().out)
        assert self_model.main(["--verify", "--ledger", str(ledger)]) == 0
        assert json.loads(capsys.readouterr().out)["valid"] is True

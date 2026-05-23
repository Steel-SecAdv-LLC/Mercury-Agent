"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.

------------------------------------------------------------------------

Tests for the operator tools.

Every tool in ``omni_mercury_engine.tools`` has at least one test
exercising its ``main()`` entry point with a temporary ``--output``
file and asserting the certificate has the expected schema and
``status``. This is the operator-visible contract: a stable schema
identifier and a deterministic exit code mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omni_mercury_engine.tools import (
    algorithm_name_drift_gate,
    api_contract_diff,
    benchmark_diff,
    config_validator,
    convergence_proof_emitter,
    dataset_checksum_manifest,
    helm_values_linter,
    image_surface_auditor,
    killswitch_tester,
    loader_reachability_probe,
    oae_weight_certifier,
    release_manifest_builder,
    sbom_emitter,
    sigma_immutable_verifier,
    synthetic_fallback_auditor,
    workflow_version_drift_gate,
)


def _load_cert(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text())
    assert isinstance(parsed, dict), f"expected dict envelope, got {type(parsed)}"
    return parsed


def test_sigma_immutable_verifier_default_corpus(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = sigma_immutable_verifier.main(["--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.sigma_immutable_verifier/v1"
    # Either fully verified (ok) or Ed25519-verified with ML-DSA omitted (warn).
    assert cert["status"] in {"ok", "warn"}
    assert cert["body"]["signatures"]["ed25519"] == "verified"
    assert rc in {0, 1}


def test_oae_weight_certifier_matches_phi(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = oae_weight_certifier.main(["--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.oae_weight_certifier/v1"
    assert cert["status"] == "ok"
    weights = cert["body"]["expected"]
    assert pytest.approx(weights["w_R"], abs=1e-6) == 0.4472135954999579
    assert pytest.approx(weights["w_H"], abs=1e-6) == 0.276393202250021
    assert pytest.approx(weights["w_O"], abs=1e-6) == 0.276393202250021
    assert pytest.approx(weights["w_R"] + weights["w_H"] + weights["w_O"], abs=1e-12) == 1.0
    assert rc == 0


def test_algorithm_name_drift_gate_ok(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = algorithm_name_drift_gate.main(["--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.algorithm_name_drift_gate/v1"
    assert cert["status"] == "ok", cert["warnings"]
    assert rc == 0


def test_algorithm_name_drift_gate_detects_kyber768(tmp_path: Path) -> None:
    bad = tmp_path / "BAD.md"
    bad.write_text("Mercury uses Kyber768 for KEM.\n")
    out = tmp_path / "cert.json"
    rc = algorithm_name_drift_gate.main(["--docs", str(bad), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert any("Kyber-768" in w for w in cert["warnings"])
    assert rc == 1


def test_algorithm_name_drift_gate_detects_unallowed_dilithium3(tmp_path: Path) -> None:
    bad = tmp_path / "BAD.md"
    bad.write_text("Mercury uses Dilithium-3 signatures directly.\n")
    out = tmp_path / "cert.json"
    rc = algorithm_name_drift_gate.main(["--docs", str(bad), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert any("Dilithium-3" in w for w in cert["warnings"])
    assert rc == 1


def test_algorithm_name_drift_gate_allows_fips204_readme_annotations(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    readme = Path(__file__).resolve().parents[2] / "README.md"
    rc = algorithm_name_drift_gate.main(["--docs", str(readme), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "ok", cert["warnings"]
    assert cert["body"]["per_doc_hits"]["README.md"]["Dilithium-3"] == [1684, 1755]
    assert rc == 0


def test_workflow_version_drift_gate_ok(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = workflow_version_drift_gate.main(["--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.workflow_version_drift_gate/v1"
    assert cert["status"] == "ok", cert["warnings"]
    assert len(cert["body"]["distinct_refs"]) == 1
    assert rc == 0


def test_workflow_version_drift_gate_detects_drift(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        'dependencies = ["ama-cryptography @ git+https://x/AMA.git@v3.2.0"]\n'
    )
    (root / ".github" / "workflows" / "ci.yml").write_text("env:\n  AMA_REF: v2.0\n")
    out = tmp_path / "cert.json"
    rc = workflow_version_drift_gate.main(["--root", str(root), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert rc == 1


def test_config_validator_ok(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "valid.json").write_text(
        json.dumps({"model": {"input_dim": 16, "d_model": 32, "n_heads": 4}})
    )
    out = tmp_path / "cert.json"
    rc = config_validator.main(["--dir", str(cfg_dir), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "ok"
    assert rc == 0


def test_config_validator_detects_missing_lambda(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "broken.json").write_text(
        json.dumps({"model": {"input_dim": 16, "d_model": 32}})  # missing n_heads
    )
    out = tmp_path / "cert.json"
    rc = config_validator.main(["--dir", str(cfg_dir), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert rc == 1


def test_dataset_checksum_manifest_roundtrip(tmp_path: Path) -> None:
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    (data_dir / "a.bin").write_bytes(b"hello mercury")
    (data_dir / "b.bin").write_bytes(b"another file")

    manifest_path = tmp_path / "manifest.json"
    rc = dataset_checksum_manifest.main([str(data_dir), "--output", str(manifest_path)])
    assert rc == 0
    cert = _load_cert(manifest_path)
    assert cert["status"] == "ok"
    assert cert["body"]["file_count"] == 2
    # Save manifest separately for verify path.
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps(cert["body"]))

    # Verify mode: should still be ok.
    out2 = tmp_path / "verify.json"
    rc = dataset_checksum_manifest.main(
        [str(data_dir), "--verify", str(saved), "--output", str(out2)]
    )
    assert rc == 0

    # Mutate a file and re-verify: must fail.
    (data_dir / "a.bin").write_bytes(b"tampered")
    out3 = tmp_path / "verify2.json"
    rc = dataset_checksum_manifest.main(
        [str(data_dir), "--verify", str(saved), "--output", str(out3)]
    )
    cert = _load_cert(out3)
    assert cert["status"] == "fail"
    assert cert["body"]["mismatched"]


def test_benchmark_diff_flags_regression(tmp_path: Path) -> None:
    prev = tmp_path / "prev.json"
    curr = tmp_path / "curr.json"
    prev.write_text(json.dumps({"d": {"smap": {"auc": 0.95}}}))
    curr.write_text(json.dumps({"d": {"smap": {"auc": 0.80}}}))
    out = tmp_path / "cert.json"
    rc = benchmark_diff.main([str(prev), str(curr), "--fail-on-regression", "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert cert["body"]["regression_count"] == 1
    assert rc == 1


def test_benchmark_diff_clean(tmp_path: Path) -> None:
    prev = tmp_path / "prev.json"
    curr = tmp_path / "curr.json"
    prev.write_text(json.dumps({"d": {"smap": {"auc": 0.95}}}))
    curr.write_text(json.dumps({"d": {"smap": {"auc": 0.96}}}))
    out = tmp_path / "cert.json"
    rc = benchmark_diff.main([str(prev), str(curr), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "ok"
    assert rc == 0


def test_killswitch_tester_within_sla(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = killswitch_tester.main(
        ["--warmup", "20", "--max-iterations", "5000", "--output", str(out)]
    )
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.killswitch_tester/v1"
    assert cert["status"] == "ok"
    assert cert["body"]["within_sla"] is True
    assert cert["body"]["trip_latency_ms"] < 1000.0
    assert rc == 0


def test_loader_reachability_probe_matrix_size(tmp_path: Path) -> None:
    # We only check that the matrix has the expected canonical size — running
    # all 11 loaders would hit the network. Restrict to one loader, allow
    # synthetic so the probe doesn't error if synthetic is the only path.
    out = tmp_path / "cert.json"
    rc = loader_reachability_probe.main(
        ["--only", "UCR", "--allow-synthetic", "--timeout", "5", "--output", str(out)]
    )
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.loader_reachability_probe/v1"
    assert cert["body"]["matrix_size"] == 1
    # Any outcome (downloaded / loud-unavailable / ssrf-gate / error) is
    # acceptable for the schema test; the canonical 11-row drift is the
    # important assertion done in the import-time assertion of the tool.
    assert rc in {0, 1}


def test_release_manifest_builder_emits_expected_fields(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = release_manifest_builder.main(["--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"] == "mercury.tools.release_manifest_builder/v1"
    body = cert["body"]
    assert "python_version" in body
    assert "fusion_weights" in body
    assert "ama_cryptography_ref" in body
    assert pytest.approx(body["fusion_weights"]["w_R"], abs=1e-6) == 0.4472135954999579
    assert rc in {0, 1}  # may warn on missing container digest


def test_sbom_emitter_contains_omni_mercury_engine(tmp_path: Path) -> None:
    sbom = tmp_path / "sbom.json"
    out = tmp_path / "cert.json"
    rc = sbom_emitter.main(
        ["--root-name", "mercury-agent", "--sbom-path", str(sbom), "--output", str(out)]
    )
    cert = _load_cert(out)
    assert cert["status"] == "ok"
    assert sbom.exists()
    parsed = json.loads(sbom.read_text())
    assert parsed["bomFormat"] == "CycloneDX"
    assert parsed["specVersion"] == "1.5"
    assert any(c["name"].lower() == "mercury-agent" for c in parsed["components"])
    assert rc == 0


def test_api_contract_diff_snapshot_and_diff(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    out = tmp_path / "cert.json"
    rc = api_contract_diff.main(["--snapshot", str(snap), "--output", str(out)])
    assert rc == 0
    out2 = tmp_path / "cert2.json"
    rc = api_contract_diff.main(["--against", str(snap), "--output", str(out2)])
    cert = _load_cert(out2)
    assert cert["status"] == "ok"
    assert cert["body"]["diff"]["removed"] == []
    assert rc == 0


def test_api_contract_diff_detects_removal(tmp_path: Path) -> None:
    snap = tmp_path / "snap.json"
    snap.write_text(
        json.dumps(
            {
                "entries": {
                    "DefinitelyNotExported_xyz": {
                        "kind": "class",
                        "signature": None,
                        "module": "omni_mercury_engine",
                        "qualname": "X",
                    }
                }
            }
        )
    )
    out = tmp_path / "cert.json"
    rc = api_contract_diff.main(["--against", str(snap), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert "DefinitelyNotExported_xyz" in cert["body"]["diff"]["removed"]
    assert rc == 1


def test_synthetic_fallback_auditor_flags_high_fraction(tmp_path: Path) -> None:
    results = tmp_path / "r.json"
    results.write_text(
        json.dumps(
            {
                "datasets": {
                    "smap": {"synthetic_fraction": 0.95},
                    "msl": {"synthetic_fraction": 0.10},
                }
            }
        )
    )
    out = tmp_path / "cert.json"
    rc = synthetic_fallback_auditor.main([str(results), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] == "fail"
    assert any("smap" in w for w in cert["warnings"])
    assert rc == 1


def test_helm_values_linter_handles_missing_chart(tmp_path: Path) -> None:
    out = tmp_path / "cert.json"
    rc = helm_values_linter.main(["--values", str(tmp_path / "missing.yaml"), "--output", str(out)])
    cert = _load_cert(out)
    assert cert["status"] in {"warn", "fail"}
    assert rc in {0, 1}


def test_image_surface_auditor_dockerfile_mode(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text(
        "FROM debian:bookworm\n"
        "RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*\n"
        "USER 1000\n"
        "ENV LD_LIBRARY_PATH=/opt/ama/lib\n"
        'ENTRYPOINT ["/opt/mercury/bin/run"]\n'
    )
    out = tmp_path / "cert.json"
    rc = image_surface_auditor.main(
        ["--mode", "dockerfile", "--dockerfile", str(df), "--output", str(out)]
    )
    cert = _load_cert(out)
    # gcc was installed without purge → expect a finding.
    assert cert["status"] == "fail"
    assert any("gcc" in w for w in cert["warnings"])
    assert rc == 1


def test_convergence_proof_emitter_check_mode(tmp_path: Path) -> None:
    # --check runs the proof emitter without modifying any doc; we
    # expect status to be either "ok" (block matches existing doc) or
    # "warn"/"fail" (drift), and the body to contain the standard
    # Lyapunov characterisation fields.
    out = tmp_path / "cert.json"
    rc = convergence_proof_emitter.main(["--check", "--output", str(out)])
    cert = _load_cert(out)
    assert cert["schema"].startswith("mercury.tools.convergence_proof_emitter")
    body = cert["body"]
    assert "block_chars" in body
    assert "P_max_eig" in body and "Q_min_eig" in body
    assert rc in {0, 1}

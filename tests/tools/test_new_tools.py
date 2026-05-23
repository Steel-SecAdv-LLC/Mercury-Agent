"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

------------------------------------------------------------------------

Tests for the Part 2 new operator tools.

Each test invokes the tool through its registry entry-point with a
deterministic seed (where applicable), asserts the certificate schema
is the documented ``mercury.tools.<name>/v1``, and pins the
deterministic-output contract for tools whose body is not coupled to
host state.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from omni_mercury_engine.tools import TOOL_REGISTRY

if TYPE_CHECKING:
    from pathlib import Path


def _load_cert(out: Path) -> dict[str, Any]:
    return json.loads(out.read_text())


def _drop_volatile(env: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in env.items() if k != "generated_at"}


# ---------------------------------------------------------------------------
# Ethical / mathematical certifiers


class TestEthicalGateCoverageReport:
    def test_emits_valid_cert(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        TOOL_REGISTRY["ethical_gate_coverage_report"](["--output", str(out)])
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.ethical_gate_coverage_report/v1"
        assert cert["status"] in {"ok", "fail", "warn"}


class TestOaeDimensionalityProbe:
    def test_emits_valid_cert(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        TOOL_REGISTRY["oae_dimensionality_probe"](
            ["--output", str(out), "--trials", "4", "--seed", "0"]
        )
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.oae_dimensionality_probe/v1"
        assert cert["status"] == "ok"
        weights = cert["body"]["weights"]
        assert abs(weights["sum"] - 1.0) < 1e-9

    def test_deterministic(self, tmp_path: Path) -> None:
        out1 = tmp_path / "1.json"
        out2 = tmp_path / "2.json"
        TOOL_REGISTRY["oae_dimensionality_probe"](
            ["--output", str(out1), "--trials", "4", "--seed", "0"]
        )
        TOOL_REGISTRY["oae_dimensionality_probe"](
            ["--output", str(out2), "--trials", "4", "--seed", "0"]
        )
        assert _drop_volatile(_load_cert(out1)) == _drop_volatile(_load_cert(out2))


class TestBenevolenceCalibrationReport:
    def test_emits_valid_cert(self, tmp_path: Path) -> None:
        scores = tmp_path / "scores.npy"
        labels = tmp_path / "labels.npy"
        np.save(scores, np.linspace(0.0, 1.0, 100))
        np.save(labels, (np.linspace(0.0, 1.0, 100) > 0.5).astype(int))
        out = tmp_path / "c.json"
        TOOL_REGISTRY["benevolence_calibration_report"](
            ["--output", str(out), "--scores", str(scores), "--labels", str(labels)]
        )
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.benevolence_calibration_report/v1"
        assert "ece" in cert["body"]


class TestSigmaImmutableDriftMonitor:
    def test_state_persists(self, tmp_path: Path) -> None:
        state = tmp_path / "state.json"
        out = tmp_path / "c.json"
        TOOL_REGISTRY["sigma_immutable_drift_monitor"](
            ["--output", str(out), "--state", str(state), "--current-sigma", "0.5"]
        )
        assert state.exists()


class TestFairnessSubgroupExplorer:
    def test_runs_on_synthetic_data(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        features = rng.integers(0, 3, size=(200, 2))
        scores = rng.random(200)
        labels = (scores > 0.5).astype(int)
        f = tmp_path / "f.npy"
        s = tmp_path / "s.npy"
        ll = tmp_path / "l.npy"
        np.save(f, features)
        np.save(s, scores)
        np.save(ll, labels)
        out = tmp_path / "c.json"
        TOOL_REGISTRY["fairness_subgroup_explorer"](
            [
                "--output",
                str(out),
                "--features",
                str(f),
                "--scores",
                str(s),
                "--labels",
                str(ll),
                "--feature-names",
                "race,sex",
            ]
        )
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.fairness_subgroup_explorer/v1"


class TestOaeEigenMonitor:
    def test_default_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        TOOL_REGISTRY["oae_eigen_monitor"](["--output", str(out)])
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.oae_eigen_monitor/v1"


# ---------------------------------------------------------------------------
# Datasets


class TestDatasetLicenseAuditor:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["dataset_license_auditor"](["--output", str(out)])
        assert rc in (0, 1)
        cert = _load_cert(out)
        assert cert["schema"] == "mercury.tools.dataset_license_auditor/v1"


class TestPiiScrubberProbe:
    def test_clean_text_is_ok(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.txt"
        clean.write_text("hello world, no PII here.\n")
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["pii_scrubber_probe"](["--output", str(out), "--file", str(clean)])
        assert rc == 0
        cert = _load_cert(out)
        assert cert["status"] == "ok"

    def test_dirty_text_fails(self, tmp_path: Path) -> None:
        dirty = tmp_path / "dirty.txt"
        dirty.write_text("contact me at jane.doe@example.com for PII discussion.\n")
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["pii_scrubber_probe"](["--output", str(out), "--file", str(dirty)])
        assert rc == 1
        cert = _load_cert(out)
        assert cert["status"] == "fail"
        assert "email" in cert["body"]["findings"]


class TestSyntheticProvenanceTag:
    def test_emit_then_verify(self, tmp_path: Path) -> None:
        data = tmp_path / "syn.csv"
        data.write_text("a,b\n1,2\n")
        out = tmp_path / "c.json"
        TOOL_REGISTRY["synthetic_provenance_tag"](
            [
                "--output",
                str(out),
                "--emit",
                "--data",
                str(data),
                "--seed",
                "42",
                "--rows",
                "1",
                "--method",
                "uniform",
            ]
        )
        sidecar = data.with_suffix(data.suffix + ".provenance.json")
        assert sidecar.exists()
        # Verify mode
        out2 = tmp_path / "verify.json"
        rc = TOOL_REGISTRY["synthetic_provenance_tag"](
            ["--output", str(out2), "--verify", "--data", str(data)]
        )
        assert rc == 0


class TestLiveDatasetProtectionGate:
    """Live-dataset protection gate.

    Live is the reference distribution.  The gate must:

    1. Pass when a reenactment is statistically indistinguishable from
       the live reference (within KS / KL / AUROC tolerance).
    2. Fail-closed when the reenactment drifts from live — protecting
       the live corpus's primacy.
    3. Fail-closed when the reenactment regresses the discriminative
       AUROC signal that the live distribution provides.
    """

    def _save_npy(self, path: Path, arr: np.ndarray) -> None:
        np.save(path, arr)

    def test_faithful_reenactment_passes(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 1024
        # Reenactment drawn from the same distribution as live — a
        # faithful reenactment of the live reference.
        live = rng.beta(2.0, 5.0, size=(n, 2))
        reenactment = rng.beta(2.0, 5.0, size=(n, 2))
        live_p = tmp_path / "live.npy"
        reenactment_p = tmp_path / "reenactment.npy"
        self._save_npy(live_p, live)
        self._save_npy(reenactment_p, reenactment)
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["live_dataset_protection_gate"](
            [
                "--output",
                str(out),
                "--live-scores",
                str(live_p),
                "--reenactment-scores",
                str(reenactment_p),
                "--column-names",
                "a,b",
            ]
        )
        assert rc == 0
        cert = _load_cert(out)
        assert cert["status"] == "ok"
        assert cert["body"]["policy"] == "live_is_reference"
        for col in cert["body"]["columns"]:
            assert col["ks"] < 0.2

    def test_drifted_reenactment_fails(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 1024
        live = rng.beta(2.0, 5.0, size=(n, 1))
        # Reenactment drifts from live — the mirror Beta(5,2).
        reenactment = rng.beta(5.0, 2.0, size=(n, 1))
        live_p = tmp_path / "live.npy"
        reenactment_p = tmp_path / "reenactment.npy"
        self._save_npy(live_p, live)
        self._save_npy(reenactment_p, reenactment)
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["live_dataset_protection_gate"](
            [
                "--output",
                str(out),
                "--live-scores",
                str(live_p),
                "--reenactment-scores",
                str(reenactment_p),
                "--column-names",
                "score",
            ]
        )
        assert rc == 1, "drifted reenactment must fail-closed against live reference"
        cert = _load_cert(out)
        assert cert["status"] == "fail"
        assert cert["body"]["policy"] == "live_is_reference"
        assert cert["body"]["failures"]
        assert "resolution_guidance" in cert["body"]

    def test_auroc_regression_against_live_fails(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        n = 512
        labels = rng.integers(0, 2, size=n)
        # Live: discriminative — label correlated with score.
        live = (labels + rng.normal(0, 0.1, size=n)).reshape(-1, 1).astype(np.float64)
        # Reenactment: uncorrelated — AUROC collapses to ~0.5.
        reenactment = rng.random(size=n).reshape(-1, 1)
        live_p = tmp_path / "live.npy"
        reenactment_p = tmp_path / "reenactment.npy"
        labels_p = tmp_path / "labels.npy"
        self._save_npy(live_p, live)
        self._save_npy(reenactment_p, reenactment)
        self._save_npy(labels_p, labels)
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["live_dataset_protection_gate"](
            [
                "--output",
                str(out),
                "--live-scores",
                str(live_p),
                "--reenactment-scores",
                str(reenactment_p),
                "--live-labels",
                str(labels_p),
                "--reenactment-labels",
                str(labels_p),
            ]
        )
        cert = _load_cert(out)
        assert rc == 1
        assert any("AUROC" in f for f in cert["body"]["failures"])

    def test_legacy_synthetic_alias_still_accepted(self, tmp_path: Path) -> None:
        """Backward compatibility: ``--synthetic-scores`` is a legacy alias.

        The flag exists so existing release pipelines do not break;
        the semantics are still live-as-reference.
        """
        rng = np.random.default_rng(0)
        n = 256
        live = rng.beta(2.0, 5.0, size=(n, 1))
        reenactment = rng.beta(2.0, 5.0, size=(n, 1))
        live_p = tmp_path / "live.npy"
        reenactment_p = tmp_path / "reenactment.npy"
        self._save_npy(live_p, live)
        self._save_npy(reenactment_p, reenactment)
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["live_dataset_protection_gate"](
            [
                "--output",
                str(out),
                "--live-scores",
                str(live_p),
                "--synthetic-scores",  # legacy alias for --reenactment-scores
                str(reenactment_p),
            ]
        )
        assert rc == 0


class TestNetworkEgressRecorder:
    def test_parses_jsonl(self, tmp_path: Path) -> None:
        trace = tmp_path / "trace.jsonl"
        trace.write_text(
            '{"url": "https://example.com/a", "status": 200, "response_size": 100}\n'
            '{"url": "https://example.com/b", "status": 404, "response_size": 0}\n'
        )
        out = tmp_path / "c.json"
        TOOL_REGISTRY["network_egress_recorder"](["--output", str(out), "--trace", str(trace)])
        cert = _load_cert(out)
        assert cert["body"]["request_count"] == 2


# ---------------------------------------------------------------------------
# Perf


class TestGosnnLatencySlaGate:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["gosnn_latency_sla_gate"](["--output", str(out), "--iterations", "8"])
        assert rc in (0, 1)


class TestThermalThrottleProbe:
    def test_short_window(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["thermal_throttle_probe"](
            ["--output", str(out), "--duration", "0.5", "--interval", "0.1"]
        )
        assert rc in (0, 1)


class TestGpuCapabilityProbe:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["gpu_capability_probe"](["--output", str(out)])
        assert rc == 0


# ---------------------------------------------------------------------------
# Deploy / containers


class TestNetworkPolicySynthesiser:
    def test_synthesises_yaml(self, tmp_path: Path) -> None:
        allow = tmp_path / "allow.txt"
        allow.write_text("https://example.com\n")
        manifest = tmp_path / "policy.yaml"
        out = tmp_path / "c.json"
        TOOL_REGISTRY["network_policy_synthesiser"](
            [
                "--output",
                str(out),
                "--allow-list",
                str(allow),
                "--manifest",
                str(manifest),
            ]
        )
        cert = _load_cert(out)
        assert "egress_rule_count" in cert["body"]


class TestPodSecurityStandardGate:
    def test_compliant_manifest_passes(self, tmp_path: Path) -> None:
        manifest = tmp_path / "deploy.yaml"
        manifest.write_text(
            "kind: Deployment\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: c\n"
            "        securityContext:\n"
            "          runAsNonRoot: true\n"
            "          readOnlyRootFilesystem: true\n"
            "          allowPrivilegeEscalation: false\n"
            "          capabilities:\n"
            "            drop:\n"
            "              - ALL\n"
            "          seccompProfile:\n"
            "            type: RuntimeDefault\n"
        )
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["pod_security_standard_gate"](
            ["--output", str(out), "--manifest", str(manifest)]
        )
        assert rc == 0

    def test_privileged_fails(self, tmp_path: Path) -> None:
        manifest = tmp_path / "bad.yaml"
        manifest.write_text("kind: Pod\nspec:\n  privileged: true\n")
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["pod_security_standard_gate"](
            ["--output", str(out), "--manifest", str(manifest)]
        )
        assert rc == 1


class TestDockerfileLockfileGate:
    def test_unpinned_apt_fails(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(
            "FROM debian:bookworm\n" "RUN apt-get update && apt-get install -y curl\n"
        )
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["dockerfile_lockfile_gate"](
            ["--output", str(out), "--dockerfile", str(dockerfile)]
        )
        assert rc == 1

    def test_pinned_passes(self, tmp_path: Path) -> None:
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text(
            "FROM debian@sha256:" + "0" * 64 + "\n" "RUN apt-get install -y curl=7.88.1-10\n"
        )
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["dockerfile_lockfile_gate"](
            ["--output", str(out), "--dockerfile", str(dockerfile)]
        )
        assert rc == 0


class TestConfigSecretRedactor:
    def test_clean_dir(self, tmp_path: Path) -> None:
        conf = tmp_path / "ok.yaml"
        conf.write_text("logging:\n  level: info\n")
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["config_secret_redactor"](["--output", str(out), str(conf)])
        assert rc == 0


# ---------------------------------------------------------------------------
# Observability


class TestOpenTelemetrySpanEmitter:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["opentelemetry_span_emitter"](["--output", str(out)])
        assert rc in (0, 1)


class TestPrometheusMetricsExporter:
    def test_emits_text(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        metrics_out = tmp_path / "metrics.prom"
        rc = TOOL_REGISTRY["prometheus_metrics_exporter"](
            ["--output", str(out), "--metrics-output", str(metrics_out)]
        )
        assert rc == 0
        assert metrics_out.exists()
        cert = _load_cert(out)
        assert "exposition" in cert["body"]
        assert "mercury_oae_weight_r" in cert["body"]["exposition"]


class TestAuditLogSignerVerifier:
    def test_roundtrip(self, tmp_path: Path) -> None:
        key = "00" * 32
        log = tmp_path / "audit.jsonl"
        for actor in ("alice", "bob", "carol"):
            out = tmp_path / f"sign_{actor}.json"
            TOOL_REGISTRY["audit_log_signer"](
                [
                    "--output",
                    str(out),
                    "--log",
                    str(log),
                    "--actor",
                    actor,
                    "--action",
                    "test",
                    "--key-hex",
                    key,
                ]
            )
        verify_out = tmp_path / "verify.json"
        rc = TOOL_REGISTRY["audit_log_verifier"](
            ["--output", str(verify_out), "--log", str(log), "--key-hex", key]
        )
        assert rc == 0
        cert = _load_cert(verify_out)
        assert cert["body"]["entry_count"] == 3

    def test_tampered_entry_fails(self, tmp_path: Path) -> None:
        key = "11" * 32
        log = tmp_path / "audit.jsonl"
        TOOL_REGISTRY["audit_log_signer"](
            [
                "--output",
                str(tmp_path / "s1.json"),
                "--log",
                str(log),
                "--actor",
                "alice",
                "--action",
                "test",
                "--key-hex",
                key,
            ]
        )
        # Tamper: rewrite the line with a different action.
        lines = log.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["action"] = "tampered"
        lines[0] = json.dumps(entry, sort_keys=True)
        log.write_text("\n".join(lines) + "\n")
        verify_out = tmp_path / "verify.json"
        rc = TOOL_REGISTRY["audit_log_verifier"](
            ["--output", str(verify_out), "--log", str(log), "--key-hex", key]
        )
        assert rc == 1


# ---------------------------------------------------------------------------
# Release / cards


class TestChangelogEnforcer:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["changelog_enforcer"](["--output", str(out)])
        assert rc in (0, 1)


class TestDatasetCardGenerator:
    def test_generates_card(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        md = tmp_path / "card.md"
        rc = TOOL_REGISTRY["dataset_card_generator"](
            ["--output", str(out), "--name", "test_dataset", "--markdown", str(md)]
        )
        assert rc == 0
        assert md.exists()


class TestFederatedRoundSimulatorAdversarial:
    def _run(self, mode: str, tmp_path: Path) -> tuple[int, Path]:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["federated_round_simulator"](
            [
                "--output",
                str(out),
                "--nodes",
                "3",
                "--dim",
                "8",
                "--seed",
                "0",
                "--adversarial",
                mode,
            ]
        )
        return rc, out

    def test_byzantine_mode(self, tmp_path: Path) -> None:
        rc, out = self._run("byzantine", tmp_path)
        if rc == 3:
            pytest.skip("federated extra missing — FederatedAggregator unavailable")
        assert rc in (0, 1)
        cert = _load_cert(out)
        assert cert["body"]["adversarial"]["mode"] == "byzantine"
        assert "deviation_from_honest" in cert["body"]["adversarial"]

    def test_gradient_inversion_mode(self, tmp_path: Path) -> None:
        rc, out = self._run("gradient_inversion", tmp_path)
        if rc == 3:
            pytest.skip("federated extra missing — FederatedAggregator unavailable")
        assert rc in (0, 1)
        cert = _load_cert(out)
        assert cert["body"]["adversarial"]["mode"] == "gradient_inversion"


# ---------------------------------------------------------------------------
# Hardware


class TestHwrngAudit:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["hwrng_audit"](["--output", str(out), "--sample-bytes", "256"])
        # /dev/hwrng absence is non-fatal in dev mode.
        assert rc in (0, 1)
        cert = _load_cert(out)
        assert "urandom" in cert["body"]


class TestTimeSourceProbe:
    def test_runs(self, tmp_path: Path) -> None:
        out = tmp_path / "c.json"
        rc = TOOL_REGISTRY["time_source_probe"](["--output", str(out)])
        # Many CI runners lack chronyc/ntpq; warn (rc=0) is acceptable.
        assert rc in (0, 1)


# ---------------------------------------------------------------------------
# Smoke registry: every tool returns a valid envelope on ``--help``


_HELP_EXEMPT = frozenset(
    {
        # Tools that exec subprocesses or open browsers — too expensive for the smoke test.
    }
)


def _help_exits_cleanly(tool: str) -> int:
    """Run ``tool --help`` and capture the exit code.

    ``--help`` is the standard ``argparse`` short-circuit; every tool
    that wires through ``_base.add_common_arguments`` should support it.
    """
    main = TOOL_REGISTRY[tool]
    try:
        return main(["--help"])
    except SystemExit as exc:  # argparse calls sys.exit
        return int(exc.code or 0)


@pytest.mark.parametrize("tool", sorted(set(TOOL_REGISTRY.names()) - _HELP_EXEMPT))
def test_every_tool_supports_help(tool: str) -> None:
    rc = _help_exits_cleanly(tool)
    assert rc == 0, f"{tool} --help returned {rc}"

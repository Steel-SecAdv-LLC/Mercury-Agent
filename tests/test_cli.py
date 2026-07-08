# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test CLI functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

from omni_mercury_engine.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_help() -> None:
    """Test CLI help command"""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Mercury Agent" in result.output


def test_detect_help() -> None:
    """Test detect command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0


def test_train_help() -> None:
    """Test train command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["train", "--help"])
    assert result.exit_code == 0


def test_biometric_help() -> None:
    """Test biometric command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["biometric", "--help"])
    assert result.exit_code == 0


def test_security_help() -> None:
    """Test security command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--help"])
    assert result.exit_code == 0


def test_stream_help() -> None:
    """Test stream worker command help"""
    runner = CliRunner()
    result = runner.invoke(main, ["stream", "--help"])
    assert result.exit_code == 0


def test_stream_options_match_deployment_manifests() -> None:
    """The `stream` worker must expose the exact flags the k8s/Helm manifests pass.

    The distributed streaming-worker manifest invokes
    ``mercury stream --input-topic ... --output-topic ... --consumer-group ...``,
    so a rename here would silently break the deployment (the failure this guards
    against: a CLI command referenced by a manifest that does not accept its
    arguments). Keep this in lockstep with k8s/overlays/distributed/streaming-workers.yaml.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["stream", "--help"])
    assert result.exit_code == 0
    for option in (
        "--input-topic",
        "--output-topic",
        "--consumer-group",
        "--backend",
        "--metrics-port",
    ):
        assert option in result.output, f"stream command missing {option}"


class TestDetectThreshold:
    """The `detect --threshold` option must actually govern the decision.

    Regression: the flag was parsed and then never used, so `-t 0.9` silently
    ran at the model's default. It now overrides the fusion decision boundary.
    """

    @staticmethod
    def _csv(tmp_path: Path) -> str:
        import numpy as np

        rng = np.random.default_rng(1)
        data = rng.normal(size=(30, 5))
        p = tmp_path / "data.csv"
        p.write_text("\n".join(",".join(str(x) for x in row) for row in data))
        return str(p)

    def test_threshold_zero_forces_anomaly_true_on_fusion(self, tmp_path: Path) -> None:
        import json

        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        csv = self._csv(tmp_path)
        result = CliRunner().invoke(main, ["detect", "-i", csv, "-d", "fusion", "-t", "0.0"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["is_anomaly"] is True
        assert payload["threshold_used"] == 0.0
        assert payload["threshold_source"] == "cli_override"

    def test_threshold_one_forces_anomaly_false_on_fusion(self, tmp_path: Path) -> None:
        import json

        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        csv = self._csv(tmp_path)
        result = CliRunner().invoke(main, ["detect", "-i", csv, "-d", "fusion", "-t", "1.0"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["is_anomaly"] is False
        assert payload["threshold_used"] == 1.0

    def test_threshold_out_of_range_rejected(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        csv = self._csv(tmp_path)
        result = CliRunner().invoke(main, ["detect", "-i", csv, "-d", "fusion", "-t", "1.5"])
        assert result.exit_code != 0
        assert "must be in [0, 1]" in result.output


class TestTierDetect:
    """`mercury-agent tier-detect` runs the torch-free detector-tier ensemble."""

    def test_tier_detect_flags_injected_burst(self, tmp_path: Path) -> None:
        import json

        import numpy as np
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        rng = np.random.default_rng(0)
        series = rng.normal(0, 1, 200)
        series[100:108] += 7.0
        csv = tmp_path / "series.csv"
        np.savetxt(csv, series, delimiter=",")

        result = CliRunner().invoke(
            main,
            [
                "tier-detect",
                "-i",
                str(csv),
                "--subset",
                "spectral_residual,spot_evt,bocpd",
                "--conformal-alpha",
                "0.05",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["n_points"] == 200
        flagged = {i for i, f in enumerate(payload["conformal_flags"]) if f}
        assert flagged & set(range(95, 115))


class TestRca:
    """`mercury-agent rca` localizes a multivariate anomaly to root-cause nodes."""

    def test_rca_ranks_causal_chain_over_independent_node(self, tmp_path: Path) -> None:
        import json

        import numpy as np
        from click.testing import CliRunner

        from omni_mercury_engine.cli import main

        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, (300, 4))
        base[:, 1] += 0.8 * base[:, 0]
        base[:, 2] += 0.8 * base[:, 1]
        obs = base.copy()
        obs[-1, 0] += 8.0
        obs[-1, 1] += 6.0
        obs[-1, 2] += 4.0
        csv = tmp_path / "obs.csv"
        np.savetxt(csv, obs, delimiter=",")

        result = CliRunner().invoke(
            main,
            ["rca", "-i", str(csv), "--node-names", "pump,valve,tank,aux"],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["n_nodes"] == 4
        by_node = {e["node"]: e["attribution"] for e in payload["ranked"]}
        # The independent node 3 is the least likely root cause.
        assert by_node[3] == min(by_node.values())
        assert payload["top_root_cause"]["name"] in {"pump", "valve", "tank"}

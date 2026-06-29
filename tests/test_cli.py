# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Test CLI functionality."""

from __future__ import annotations

from click.testing import CliRunner

from omni_mercury_engine.cli import main


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

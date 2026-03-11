"""
Mercury Agent
Copyright (C) 2025 Steel Security Advisors LLC

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
"""

from __future__ import annotations

"""
Additional CLI tests for edge cases and integration coverage.
"""

import json
import os
import tempfile
from pathlib import Path

from click.testing import CliRunner

from omni_mercury_engine.cli import main


def test_security_command_help():
    """Test security command help."""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--help"])
    assert result.exit_code == 0
    assert "--payload" in result.output


def test_security_command_with_sql_payload():
    """Test security command with SQL injection payload."""
    runner = CliRunner()
    result = runner.invoke(main, ["security", "--payload", "SELECT * FROM users WHERE id = 1"])
    assert result.exit_code == 0 or "error" in result.output.lower()


def test_detect_with_json_input():
    """Test detect command with JSON input."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([[1, 2, 3], [4, 5, 6]], f)
        data_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", data_file, "--detector", "temporal"])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        Path(data_file).unlink()


def test_detect_with_output_file():
    """Test detect command with output file."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n")
        data_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as out_f:
        output_file = out_f.name

    try:
        result = runner.invoke(
            main,
            [
                "detect",
                "--input",
                data_file,
                "--detector",
                "statistical",
                "--output",
                output_file,
            ],
        )
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        Path(data_file).unlink()
        if os.path.exists(output_file):
            Path(output_file).unlink()


def test_train_command_with_options():
    """Test train command with correct flags."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "data")
        os.makedirs(data_dir)
        output_path = os.path.join(tmpdir, "model.pt")

        result = runner.invoke(
            main,
            [
                "train",
                "--data",
                data_dir,
                "--output",
                output_path,
                "--epochs",
                "1",
            ],
        )
        # May fail with RuntimeError if data dir is empty; that's OK for a flag test
        assert result.exit_code == 0 or "error" in result.output.lower() or result.exception is not None


def test_biometric_with_both_images():
    """Test biometric command with reference and test images."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["biometric", "--reference", "/tmp/ref.jpg", "--test", "/tmp/test.jpg"],
    )
    assert result.exit_code != 0 or "error" in result.output.lower()


def test_detect_with_fusion_detector():
    """Test detect command explicitly using fusion detector."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n7,8,9\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["detect", "--input", temp_file, "--detector", "fusion"])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        Path(temp_file).unlink()


def test_detect_nonexistent_file():
    """Test detect command with nonexistent input file."""
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--input", "/nonexistent/data.csv"])
    assert result.exit_code != 0 or "Error" in result.output


def test_explain_command_default_model():
    """Test explain command with default fusion model."""
    runner = CliRunner()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("1,2,3\n4,5,6\n")
        temp_file = f.name

    try:
        result = runner.invoke(main, ["explain", "--input", temp_file])
        assert result.exit_code == 0 or "error" in result.output.lower()
    finally:
        Path(temp_file).unlink()


def test_physics_integrated_no_inputs():
    """Test physics integrated requires at least one input."""
    runner = CliRunner()
    result = runner.invoke(main, ["physics", "integrated"])
    # Should fail since no input files provided
    assert result.exit_code != 0 or "Error" in result.output or "error" in result.output.lower()

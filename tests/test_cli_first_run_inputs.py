# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The inputs a first run hands the CLI are read as written, or refused in one line.

A sweep of every ``mercury-agent`` command with small real files found that a
one-column series was scored as a single 300-feature sample, that a missing,
empty or text-bearing file ended in a Python traceback, that ``physics
dynamics`` interleaved the x/y/z columns into one series, that ``physics
integrated`` failed for every option combination, and that ``physics list``
advertised a binary (``mercury``) that is not installed. Each test below
states what the command did before.
"""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import _csv_header_rows, _load_data, main

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import Result

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the engine and the physics detectors need torch",
)

N = 300
SPIKE = 150


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


def _series() -> np.ndarray:
    values = np.random.default_rng(0).normal(size=N)
    values[SPIKE] += 12.0
    return values


def _column_csv(path: Path, values: np.ndarray, header: str | None = None) -> str:
    lines = [f"{value:.6f}" for value in values]
    if header is not None:
        lines.insert(0, header)
    return _write(path, "\n".join(lines) + "\n")


def _table_csv(path: Path, table: np.ndarray, header: str) -> str:
    rows = [",".join(f"{value:.10f}" for value in row) for row in table]
    return _write(path, header + "\n" + "\n".join(rows) + "\n")


def _json_output(result: Result) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    text = result.stdout
    parsed: dict[str, Any] = json.loads(text[text.index("{") :])
    return parsed


def _assert_one_line_error(result: Result, *fragments: str) -> None:
    """The command refused through click (no uncaught exception) and said why."""
    assert result.exit_code != 0, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)
    assert "Traceback" not in result.output
    for fragment in fragments:
        assert fragment in result.stderr, result.stderr


# --- 1. a single column of N rows is N samples of one feature -----------------


def test_a_single_column_csv_is_n_samples_of_one_feature(tmp_path: Path) -> None:
    """Before: shape (1, 300) -- one sample with 300 features."""
    plain = _load_data(_column_csv(tmp_path / "plain.csv", _series()))
    headed = _load_data(_column_csv(tmp_path / "headed.csv", _series(), header="value"))

    assert plain.shape == (N, 1)
    np.testing.assert_array_equal(headed, plain)


def test_a_one_dimensional_npy_and_a_flat_json_array_are_n_samples(tmp_path: Path) -> None:
    """Before: the ``.npy`` loaded as (1, 300) and the JSON array as (300,).

    The three spellings of one series now load as the same (300, 1) matrix, so
    no detector downstream has to guess what a 1-D array means.
    """
    values = _series()
    np.save(tmp_path / "series.npy", values)
    (tmp_path / "series.json").write_text(json.dumps(values.tolist()), encoding="utf-8")

    from_npy = _load_data(str(tmp_path / "series.npy"))
    from_json = _load_data(str(tmp_path / "series.json"))

    assert from_npy.shape == from_json.shape == (N, 1)
    np.testing.assert_array_equal(from_npy, from_json)


def test_one_row_of_several_values_is_still_one_sample(tmp_path: Path) -> None:
    """A row is a sample, whether the file holds one row or many.

    Kept behaviour for the row; the lone cell is new (before: a 0-d array,
    which no detector accepts).
    """
    assert _load_data(_write(tmp_path / "row.csv", "1,2,3,4\n")).shape == (1, 4)
    assert _load_data(_write(tmp_path / "cell.csv", "7\n")).shape == (1, 1)
    (tmp_path / "row.json").write_text("[[1, 2, 3, 4]]", encoding="utf-8")
    assert _load_data(str(tmp_path / "row.json")).shape == (1, 4)


@requires_torch
@pytest.mark.parametrize("suffix", [".csv", ".npy", ".json"])
def test_detect_flags_the_spike_in_a_one_column_series(tmp_path: Path, suffix: str) -> None:
    """Before: CSV and NPY printed ``"is_anomaly": [false]`` -- one verdict, exit 0.

    Only the JSON spelling of the same 300 values scored 300 rows and flagged
    the spike at row 150.
    """
    values = _series()
    path = tmp_path / f"series{suffix}"
    if suffix == ".csv":
        _column_csv(path, values, header="value")
    elif suffix == ".npy":
        np.save(path, values)
    else:
        path.write_text(json.dumps(values.tolist()), encoding="utf-8")

    payload = _json_output(CliRunner().invoke(main, ["detect", "-i", str(path)]))

    flags = payload["detectors"]["statistical"]["is_anomaly"]
    assert len(flags) == N
    assert flags[SPIKE] is True
    assert payload["is_anomaly"] is True


@requires_torch
def test_train_counts_the_rows_of_a_one_column_file(tmp_path: Path) -> None:
    """Before: ``Training fusion model on 1 samples``, then a fit-size error."""
    data = _column_csv(tmp_path / "series.csv", _series())
    result = CliRunner().invoke(
        main, ["train", "-d", data, "-o", str(tmp_path / "model.pt"), "-e", "1"]
    )
    assert f"Training fusion model on {N} samples" in result.output, result.output


@requires_torch
def test_a_one_column_spectral_reference_is_still_one_recording(tmp_path: Path) -> None:
    """Guard for fix 1: ``--reference`` rows are recordings, a lone column is one.

    The loader used to turn a one-column file into one row, which is what
    ``physics spectral --reference`` relied on. Now that a column loads as a
    column, the command lays it out as the single recording it is; read as
    2048 one-sample recordings the fit has no spectrum to learn from.
    """
    rng = np.random.default_rng(0)
    t = np.arange(2048) / 1000.0
    healthy = np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(t.size)
    probe = np.sin(2 * np.pi * 100 * t) + 0.1 * rng.standard_normal(t.size)
    column = _column_csv(tmp_path / "healthy_column.csv", healthy)
    row = _write(tmp_path / "healthy_row.csv", ",".join(f"{v:.6f}" for v in healthy) + "\n")
    signal = _column_csv(tmp_path / "probe.csv", probe)

    runner = CliRunner()
    as_column = _json_output(
        runner.invoke(main, ["physics", "spectral", "-i", signal, "-r", column])
    )
    as_row = _json_output(runner.invoke(main, ["physics", "spectral", "-i", signal, "-r", row]))

    assert as_column["baseline"] == "reference"
    assert as_column["anomaly_score"] == pytest.approx(as_row["anomaly_score"])
    assert as_column["is_anomaly"] == as_row["is_anomaly"]


# --- 2. one clear line where a traceback was ----------------------------------


@requires_torch
@pytest.mark.parametrize(
    ("content", "fragment"),
    [
        (None, "not found"),
        ("", "no data rows"),
        ("a,b,c\n", "no data rows"),
        ("name,value\ns0,0.1\ns1,0.2\n", "must be a number"),
    ],
    ids=["missing", "empty", "header-only", "text-column"],
)
def test_detect_refuses_an_unreadable_input_in_one_line(
    tmp_path: Path, content: str | None, fragment: str
) -> None:
    """Before: FileNotFoundError, DetectorException and ValueError tracebacks."""
    path = tmp_path / "input.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")

    result = CliRunner().invoke(main, ["detect", "-i", str(path)])

    _assert_one_line_error(result, "--input", "input.csv", fragment)
    assert "is_anomaly" not in result.output


def test_rca_and_tier_detect_refuse_a_missing_input_in_one_line(tmp_path: Path) -> None:
    """Before: ``rca -i nope.csv`` ended in a FileNotFoundError traceback."""
    missing = str(tmp_path / "nope.csv")
    for command in ("rca", "tier-detect"):
        result = CliRunner().invoke(main, [command, "-i", missing])
        _assert_one_line_error(result, "--input", "nope.csv", "not found")


def test_hazard_viz_refuses_a_missing_data_file_in_one_line(tmp_path: Path) -> None:
    """Before: ``hazard-viz -d earthquake --data nope.csv`` raised FileNotFoundError."""
    out = tmp_path / "x.png"
    result = CliRunner().invoke(
        main,
        ["hazard-viz", "-d", "earthquake", "--data", str(tmp_path / "nope.csv"), "-o", str(out)],
    )
    _assert_one_line_error(result, "--data", "nope.csv")
    assert not out.exists()


@requires_torch
@pytest.mark.parametrize("extra", [[], ["--temporal"]], ids=["pc", "granger"])
def test_causal_refuses_a_name_count_that_does_not_match_the_columns(
    tmp_path: Path, extra: list[str]
) -> None:
    """Before: two names for four columns reached an IndexError inside discovery."""
    table = np.random.default_rng(1).normal(size=(60, 4))
    data = _table_csv(tmp_path / "causal.csv", table, "A,B,C,D")

    result = CliRunner().invoke(main, ["causal", "-i", data, "--names", "A,B", *extra])

    assert result.exit_code == 2, result.output
    _assert_one_line_error(result, "--names", "2 names", "4 columns")


def test_stacking_without_labels_is_a_usage_error(tmp_path: Path) -> None:
    """Before: ``ValueError: stacking requires per-point labels`` as a traceback."""
    data = _column_csv(tmp_path / "series.csv", _series())

    result = CliRunner().invoke(main, ["tier-detect", "-i", data, "-m", "stacking"])

    assert result.exit_code == 2, result.output
    _assert_one_line_error(result, "stacking", "--labels")


# --- 3. physics dynamics: one position series per column ----------------------


def _constant_velocity(axes: int = 3) -> np.ndarray:
    steps = np.arange(500)
    return np.stack([0.02 * (axis + 1) * steps for axis in range(axes)], axis=1)


def _dynamics(path: str) -> dict[str, Any]:
    return _json_output(CliRunner().invoke(main, ["physics", "dynamics", "-i", path]))


@requires_torch
def test_dynamics_scores_each_column_as_its_own_axis(tmp_path: Path) -> None:
    """Before: constant velocity in x, y, z scored 0.80, ``is_anomaly: true``.

    ``data.flatten()`` interleaved the columns (x0, y0, z0, x1, ...) into one
    sawtooth series; the same motion as three one-column files scored < 0.001.
    """
    table = _constant_velocity()
    out = _dynamics(_table_csv(tmp_path / "motion.csv", table, "x,y,z"))

    assert out["is_anomaly"] is False
    assert out["anomaly_score"] < 0.05
    assert [axis["column"] for axis in out["axes"]] == [0, 1, 2]
    # Each axis scores what the same column scores as a file of its own.
    for index, axis in enumerate(out["axes"]):
        alone = _dynamics(_column_csv(tmp_path / f"axis{index}.csv", table[:, index]))
        assert "axes" not in alone
        assert axis["anomaly_score"] == pytest.approx(alone["anomaly_score"], abs=1e-6)


@requires_torch
def test_dynamics_reports_the_worst_axis(tmp_path: Path) -> None:
    """A jump on one axis is not averaged away by two steady ones."""
    table = _constant_velocity()
    table[250:, 1] += 3.0
    out = _dynamics(_table_csv(tmp_path / "jump.csv", table, "x,y,z"))

    assert [axis["is_anomaly"] for axis in out["axes"]] == [False, True, False]
    assert out["is_anomaly"] is True
    assert out["anomaly_score"] == pytest.approx(out["axes"][1]["anomaly_score"])


# --- 4. physics integrated runs -----------------------------------------------


@pytest.fixture
def physics_inputs(tmp_path: Path) -> dict[str, str]:
    rng = np.random.default_rng(0)
    t = np.arange(2048) / 1000.0
    vibration = np.sin(2 * np.pi * 50 * t) + 0.05 * rng.normal(size=t.size)
    events = [
        {"timestamp": 0.4 * i, "type": "click", "x": 100 + 7 * i, "y": 200 + 3 * i}
        for i in range(30)
    ]
    interactions = tmp_path / "interactions.json"
    interactions.write_text(json.dumps(events), encoding="utf-8")
    return {
        "vibration": _column_csv(tmp_path / "vibration.csv", vibration),
        "motion": _column_csv(tmp_path / "motion.csv", _constant_velocity(1)[:, 0]),
        "table": _table_csv(tmp_path / "table.csv", _constant_velocity(), "x,y,z"),
        "interactions": str(interactions),
    }


@requires_torch
@pytest.mark.parametrize(
    ("options", "components"),
    [
        (["-s", "vibration"], ["spectral"]),
        (["-d", "motion"], ["dynamics"]),
        (["-u", "interactions"], ["uiux"]),
        (["-s", "vibration", "-d", "vibration"], ["dynamics", "spectral"]),
        (["-s", "vibration", "-u", "interactions"], ["spectral", "uiux"]),
        (
            ["-s", "vibration", "-d", "vibration", "-u", "interactions"],
            ["dynamics", "spectral", "uiux"],
        ),
    ],
    ids=["s", "d", "u", "s+d", "s+u", "s+d+u"],
)
def test_integrated_runs_for_every_option_combination(
    physics_inputs: dict[str, str], options: list[str], components: list[str]
) -> None:
    """Before: exit 1, ``'dict' object has no attribute 'size'`` (or ``'NoneType'
    object has no attribute 'get'`` for ``-u`` alone), whatever the options.

    The command passed ``{"spectral": ..., "dynamics": ..., "uiux": ...}`` under
    the default ``data_type="time_series"``; the detector takes that mapping as
    ``data_type="mixed"`` with the keys ``time_series`` and ``interactions``.
    """
    argv = [physics_inputs.get(option, option) for option in options]
    out = _json_output(CliRunner().invoke(main, ["physics", "integrated", *argv]))

    assert sorted(out["detector_results"]) == components
    assert isinstance(out["is_anomaly"], bool)
    # Before, this key read a field the detector does not return and was 0.0.
    assert 0.0 < out["fused_anomaly_score"] <= 1.0
    for component in out["detector_results"].values():
        assert 0.0 <= component["anomaly_score"] <= 1.0
    assert 0.0 < out["gosnn_governance"]["ethical_scaling"] <= 1.0
    assert set(out["three_r_alignment"]) == {"recursion_score", "resonance_score"}


@requires_torch
def test_integrated_refuses_two_different_time_series(physics_inputs: dict[str, str]) -> None:
    """The detector scores ONE time series with both physics detectors.

    ``-s`` and ``-d`` may name the same recording; two different ones cannot
    be fused by ``PhysicsIntegratedDetector`` and are refused, not flattened
    together or silently reduced to one of them.
    """
    result = CliRunner().invoke(
        main,
        [
            "physics",
            "integrated",
            "-s",
            physics_inputs["vibration"],
            "-d",
            physics_inputs["motion"],
        ],
    )
    assert result.exit_code == 1
    assert "same recording" in result.output
    assert "fused_anomaly_score" not in result.output


@requires_torch
def test_integrated_refuses_a_table_instead_of_interleaving_it(
    physics_inputs: dict[str, str],
) -> None:
    """Before (had the call worked): the x/y/z columns were flattened into one series."""
    result = CliRunner().invoke(main, ["physics", "integrated", "-d", physics_inputs["table"]])
    assert result.exit_code == 1
    assert "one column" in result.output
    assert "fused_anomaly_score" not in result.output


# --- 5. physics list names the installed binary -------------------------------


def test_physics_list_names_the_installed_binary() -> None:
    """Before: four ``Command: mercury physics ...`` lines; ``mercury`` is not installed."""
    result = CliRunner().invoke(main, ["physics", "list"])
    assert result.exit_code == 0, result.output
    commands = [line.strip() for line in result.output.splitlines() if "Command:" in line]
    assert len(commands) == 4
    assert all(line.startswith("Command: mercury-agent physics ") for line in commands)


# --- 6. smaller input checks --------------------------------------------------


def test_a_utf8_bom_does_not_cost_a_headerless_csv_its_first_row(tmp_path: Path) -> None:
    """Before: the BOM made the first row unparseable, so it was dropped as a header.

    A 4-row file exported by Excel or PowerShell ("UTF-8 with BOM") scored 3 rows.
    """
    body = "1.5,2.5\n3,4\n5,6\n7,8\n"
    plain = tmp_path / "plain.csv"
    plain.write_text(body, encoding="utf-8")
    with_bom = tmp_path / "bom.csv"
    with_bom.write_text(body, encoding="utf-8-sig")
    headed = tmp_path / "bom_headed.csv"
    headed.write_text("a,b\n" + body, encoding="utf-8-sig")

    assert _csv_header_rows(with_bom) == 0
    assert _csv_header_rows(headed) == 1
    expected = _load_data(str(plain))
    assert expected.shape == (4, 2)
    np.testing.assert_array_equal(_load_data(str(with_bom)), expected)
    np.testing.assert_array_equal(_load_data(str(headed)), expected)


@pytest.mark.parametrize("step", ["0", "-0.01"])
def test_dynamics_rejects_a_non_positive_time_step(tmp_path: Path, step: str) -> None:
    """Before: ``--time-step 0`` divided by zero and reported a NaN score error."""
    data = _column_csv(tmp_path / "motion.csv", _constant_velocity(1)[:, 0])

    result = CliRunner().invoke(main, ["physics", "dynamics", "-i", data, "--time-step", step])

    assert result.exit_code == 2, result.output
    _assert_one_line_error(result, "--time-step", "greater than 0")


@pytest.mark.parametrize("alpha", ["5", "0", "1", "-0.05"])
def test_causal_rejects_a_significance_outside_the_unit_interval(
    tmp_path: Path, alpha: str
) -> None:
    """Before: ``--significance 5`` was accepted and a graph printed, exit 0."""
    table = np.random.default_rng(1).normal(size=(60, 3))
    data = _table_csv(tmp_path / "causal.csv", table, "A,B,C")

    result = CliRunner().invoke(main, ["causal", "-i", data, "--significance", alpha])

    assert result.exit_code == 2, result.output
    _assert_one_line_error(result, "--significance", "between 0 and 1")

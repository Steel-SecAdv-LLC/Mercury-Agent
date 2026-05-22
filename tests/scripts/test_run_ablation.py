"""Tests for :mod:`scripts.run_ablation`.

These tests exercise the orchestrator end-to-end as a library call
(``main(argv)``) so we get deterministic coverage of:

* successful Lyapunov pre-gate followed by a real subprocess command,
* successful Lyapunov pre-gate with ``--skip-run``,
* missing-config usage error (exit 2),
* Lyapunov pre-gate failure aborting before any subprocess (exit 3),
* missing ``run_command`` (exit 4),
* propagation of the experiment subprocess exit code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_ablation


CANONICAL_CFG = _REPO_ROOT / "configs" / "lyapunov_canonical.yaml"


def _read(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(out.read_text())
    return data


def test_skip_run_succeeds_on_canonical(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    rc = run_ablation.main(
        ["--config", str(CANONICAL_CFG), "--out", str(out), "--skip-run"]
    )
    assert rc == 0
    payload = _read(out)
    assert payload["lyapunov_valid"] is True
    assert payload["skipped_run"] is True


def test_missing_config_exits_2(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    rc = run_ablation.main(
        ["--config", str(tmp_path / "nope.yaml"), "--out", str(out)]
    )
    assert rc == 2
    assert not out.exists()  # we never even validated


def test_lyapunov_failure_exits_3_without_running(tmp_path: Path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        "lambda: 10.0\n"  # impossibly aggressive claim
        "A: [[-0.25, 0.0], [0.0, -0.5]]\n"
        "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        "run_command: 'python -c \"raise SystemExit(99)\"'\n"
    )
    out = tmp_path / "result.json"
    rc = run_ablation.main(["--config", str(cfg), "--out", str(out)])
    assert rc == 3
    payload = _read(out)
    assert payload["lyapunov_valid"] is False
    # If the subprocess had run we'd see run_rc; assert we did NOT run it.
    assert "run_rc" not in payload


def test_missing_run_command_exits_4(tmp_path: Path) -> None:
    cfg = tmp_path / "no_cmd.yaml"
    cfg.write_text(
        "lambda: 0.25\n"
        "A: [[-0.25, 0.0], [0.0, -0.5]]\n"
        "P: [[1.0, 0.0], [0.0, 1.0]]\n"
    )
    out = tmp_path / "result.json"
    rc = run_ablation.main(["--config", str(cfg), "--out", str(out)])
    assert rc == 4
    payload = _read(out)
    assert payload["lyapunov_valid"] is True
    assert payload["run_command"] is None


def test_run_command_exit_code_is_propagated(tmp_path: Path) -> None:
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "lambda: 0.25\n"
        "A: [[-0.25, 0.0], [0.0, -0.5]]\n"
        "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        "run_command: 'python -c \"raise SystemExit(7)\"'\n"
    )
    out = tmp_path / "result.json"
    rc = run_ablation.main(["--config", str(cfg), "--out", str(out)])
    assert rc == 7
    payload = _read(out)
    assert payload["run_rc"] == 7


def test_successful_run_command(tmp_path: Path) -> None:
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        "lambda: 0.25\n"
        "A: [[-0.25, 0.0], [0.0, -0.5]]\n"
        "P: [[1.0, 0.0], [0.0, 1.0]]\n"
        "run_command: 'python -c \"print(42)\"'\n"
    )
    out = tmp_path / "result.json"
    rc = run_ablation.main(["--config", str(cfg), "--out", str(out)])
    assert rc == 0
    payload = _read(out)
    assert payload["run_rc"] == 0
    assert payload["lyapunov_valid"] is True


@pytest.mark.parametrize("argv", [[], ["--out", "x"]])
def test_argparse_requires_config(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        run_ablation.main(argv)

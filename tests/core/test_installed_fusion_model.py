# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The installed-model slot: ``train --install`` makes a model the served default."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from omni_mercury_engine.cli import main
from omni_mercury_engine.engine import (
    FUSION_CHECKPOINT_ENV,
    MODEL_DIR_ENV,
    PREVIOUS_FUSION_FILENAME,
    default_fusion_checkpoint_path,
    install_fusion_checkpoint,
    installed_fusion_checkpoint_path,
)

if TYPE_CHECKING:
    from pathlib import Path


class _Engine:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def save_model(self, path: str) -> None:
        with open(path, "wb") as handle:
            handle.write(self.payload[:2])
            if self.fail:
                raise OSError("disk full")
            handle.write(self.payload[2:])


@pytest.fixture
def model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv(FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path / "models"))
    return tmp_path / "models"


def test_resolution_order(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert default_fusion_checkpoint_path().name == "default_fusion.pt"
    installed = install_fusion_checkpoint(_Engine(b"model-v1"))
    assert installed == installed_fusion_checkpoint_path() == model_dir / "fusion.pt"
    assert default_fusion_checkpoint_path() == installed
    override = model_dir.parent / "operator.pt"
    monkeypatch.setenv(FUSION_CHECKPOINT_ENV, str(override))
    assert default_fusion_checkpoint_path() == override


def test_reinstall_keeps_the_previous_model(model_dir: Path) -> None:
    install_fusion_checkpoint(_Engine(b"model-v1"))
    target = install_fusion_checkpoint(_Engine(b"model-v2"))
    assert target.read_bytes() == b"model-v2"
    assert (model_dir / PREVIOUS_FUSION_FILENAME).read_bytes() == b"model-v1"
    assert sorted(p.name for p in model_dir.iterdir()) == sorted(["fusion.pt", PREVIOUS_FUSION_FILENAME])


def test_a_failed_save_leaves_the_installed_model_intact(model_dir: Path) -> None:
    install_fusion_checkpoint(_Engine(b"model-v1"))
    with pytest.raises(OSError, match="disk full"):
        install_fusion_checkpoint(_Engine(b"model-v2", fail=True))
    assert (model_dir / "fusion.pt").read_bytes() == b"model-v1"
    assert [p.name for p in model_dir.iterdir()] == ["fusion.pt"]


def test_train_needs_a_destination(tmp_path: Path) -> None:
    data = tmp_path / "x.npy"
    data.write_bytes(b"")
    result = CliRunner().invoke(main, ["train", "--data", str(data)])
    assert result.exit_code == 2
    assert "give --output, --install, or both" in result.output

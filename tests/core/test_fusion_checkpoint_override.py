# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``MERCURY_FUSION_CHECKPOINT`` lets an operator replace the bundled checkpoint.

A set-but-missing override must not fall back to the bundled file: serving the
model the operator chose to replace would hide the misconfiguration.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("torch")

from omni_mercury_engine.engine import (
    FUSION_CHECKPOINT_ENV,
    OmniMercuryEngine,
    default_fusion_checkpoint_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bundled(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(FUSION_CHECKPOINT_ENV, raising=False)
    return default_fusion_checkpoint_path()


def test_unset_or_blank_uses_the_bundled_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    bundled = _bundled(monkeypatch)
    assert bundled.name == "default_fusion.pt"
    monkeypatch.setenv(FUSION_CHECKPOINT_ENV, "  ")
    assert default_fusion_checkpoint_path() == bundled


def test_the_override_is_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundled = _bundled(monkeypatch)
    if not bundled.exists():
        pytest.skip("bundled checkpoint not generated")
    override = tmp_path / "operator_fusion.pt"
    shutil.copyfile(bundled, override)
    monkeypatch.setenv(FUSION_CHECKPOINT_ENV, str(override))
    assert default_fusion_checkpoint_path() == override
    engine = OmniMercuryEngine(mode="fusion")
    assert engine.load_default_fusion_checkpoint()
    assert engine._fusion_trained


def test_a_missing_override_does_not_fall_back_to_the_bundled_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "absent.pt"
    monkeypatch.setenv(FUSION_CHECKPOINT_ENV, str(missing))
    assert default_fusion_checkpoint_path() == missing
    engine = OmniMercuryEngine(mode="fusion")
    with caplog.at_level("WARNING"):
        assert engine.load_default_fusion_checkpoint() is False
    assert not engine._fusion_trained
    assert any(FUSION_CHECKPOINT_ENV in r.getMessage() for r in caplog.records)

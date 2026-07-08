# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pin the removal of ``ml.ppo_trainer`` (DEPRECATION.md §6.6).

The module was dead in every supported install profile (stable_baselines3
in no extra, gymnasium absent → the constructor raised unconditionally) and
its surviving paths fabricated results (swallowed ``learn()`` failure logged
as "Pretraining complete"; random actions substituted for a missing model).
Removal follows the §6 correctness-exception policy; these tests pin that
the surface stays gone and the package stays importable without it.
"""

from __future__ import annotations

import importlib.util

import pytest

from omni_mercury_engine import ml


def test_ml_package_imports_without_ppo() -> None:
    """``import omni_mercury_engine.ml`` works with the module removed."""
    assert ml is not None


def test_ppo_names_absent_from_all() -> None:
    for name in (
        "PPOTrainer",
        "MultiEnvPPOTrainer",
        "PPOConfig",
        "TrainingStats",
        "ConvergenceMonitor",
        "CheckpointCallback",
    ):
        assert name not in ml.__all__, f"{name} must stay removed from ml.__all__"
        assert not hasattr(ml, name), f"ml.{name} must not resolve"


def test_ppo_module_gone() -> None:
    assert importlib.util.find_spec("omni_mercury_engine.ml.ppo_trainer") is None
    with pytest.raises(ImportError):
        import omni_mercury_engine.ml.ppo_trainer  # noqa: F401


def test_compat_flag_removed() -> None:
    """The orphaned HAS_STABLE_BASELINES probe went with the module."""
    from omni_mercury_engine import _compat

    assert not hasattr(_compat, "HAS_STABLE_BASELINES")

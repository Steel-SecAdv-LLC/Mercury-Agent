# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Importing the engine must not load libraries only training or validation use.

PyTorch Lightning (with torchmetrics and matplotlib behind it) is needed to
train, and sympy to cross-check constants against symbolic values; neither
is needed to detect. Loading them with the engine cost roughly a fifth of
the engine's import time, paid by every worker, CLI call and container
start. Their public names stay importable and resolve on first use.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("torch")

_HEAVY = ("pytorch_lightning", "lightning_fabric", "torchmetrics", "matplotlib", "sympy")


def test_engine_import_leaves_training_and_validation_libraries_unloaded() -> None:
    probe = (
        "import sys, omni_mercury_engine.engine\n"
        f"loaded = [m for m in {_HEAVY!r} if m in sys.modules]\n"
        "print(','.join(loaded))"
    )
    proc = subprocess.run(
        [sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"loaded with the engine: {proc.stdout.strip()}"


def test_training_names_still_resolve_from_the_ml_package() -> None:
    from omni_mercury_engine import ml
    from omni_mercury_engine.ml import training

    assert ml.FusionTrainer is training.FusionTrainer
    assert ml.LyapunovAnomalyLoss is training.LyapunovAnomalyLoss
    assert ml.ThreeRAnomalyTrainer is training.ThreeRAnomalyTrainer
    assert ml.HAS_PYTORCH_LIGHTNING is training.HAS_PYTORCH_LIGHTNING
    for name in ("FusionTrainer", "HAS_PYTORCH_LIGHTNING"):
        assert name in ml.__all__


def test_star_import_still_provides_the_training_names() -> None:
    namespace: dict[str, object] = {}
    exec("from omni_mercury_engine.ml import *", namespace)  # noqa: S102
    for name in ("FusionTrainer", "LyapunovAnomalyLoss", "ThreeRAnomalyTrainer"):
        assert name in namespace


def test_unknown_ml_attribute_still_raises() -> None:
    from omni_mercury_engine import ml

    with pytest.raises(AttributeError, match="no attribute"):
        ml.DefinitelyNotAnExport  # noqa: B018


def test_constants_still_validate_against_sympy() -> None:
    pytest.importorskip("sympy")
    from omni_mercury_engine.utils import constants

    assert constants.SYMPY_AVAILABLE is True
    assert all(constants.validate_all_constants_with_sympy().values())
    assert constants.sympy is not None  # imported by the first validation

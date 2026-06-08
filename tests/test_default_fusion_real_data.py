# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Network-gated end-to-end tests for the real-data fusion training path.

Guards that the opt-in real (ADBench) path actually produces a discriminating,
calibrated model through the *wired* engine API (``fit_fusion`` /
``fit_fusion_pooled`` + ``score_fusion``) — not just that it runs. Marked
``network`` + ``slow``: ``tests/conftest.py`` auto-skips unless
``MERCURY_NETWORK_TESTS=1`` is set, so default CI lanes never hit the internet.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

pytestmark = [pytest.mark.network, pytest.mark.slow]

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "train_default_fusion.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_default_fusion", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subsample(x: np.ndarray, y: np.ndarray, cap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= cap:
        return x, y
    sel = np.random.default_rng(seed).choice(len(x), cap, replace=False)
    return x[sel], y[sel]


def test_real_adbench_fit_fusion_learns_signal(tmp_path):
    from sklearn.metrics import roc_auc_score

    from omni_mercury_engine.engine import OmniMercuryEngine

    tdf = _load_train_module()
    x, y = tdf._load_adbench("cardio", str(tmp_path))
    x, y = _subsample(x, y, 700, tdf.SEED)

    train_idx, test_idx = tdf._stratified_split(y, train_frac=0.7, seed=tdf.SEED)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        x[train_idx], y[train_idx], epochs=100, batch_size=64, early_stopping_patience=15
    )

    probs = engine.score_fusion(x[test_idx])
    auc = roc_auc_score(y[test_idx], probs)
    assert auc > 0.7, f"real-data fusion underperformed: AUC={auc:.3f}"
    # score_fusion is the true serve path: calibration must be wired, not stored-and-ignored.
    assert engine._fusion_calibrator is not None


def test_real_adbench_pooled_prior_round_trips(tmp_path):
    from omni_mercury_engine.engine import OmniMercuryEngine

    tdf = _load_train_module()
    datasets: list[tuple[np.ndarray, np.ndarray]] = []
    for name in ("cardio", "WBC"):
        try:
            x, y = tdf._load_adbench(name, str(tmp_path))
        except Exception as exc:  # network / availability
            pytest.skip(f"{name} unavailable: {type(exc).__name__}")
        datasets.append(_subsample(x, y, 600, tdf.SEED))

    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    metrics = engine.fit_fusion_pooled(datasets, epochs=80, early_stopping_patience=12)

    assert engine._fusion_trained is True
    assert engine._fusion_calibrator is not None
    assert metrics["pooled_datasets"] == 2
    assert metrics["pooled_groups"], "pooled prior must retain at least one consistent group"

    engine._fusion_provenance = {"source": "adbench", "datasets": ["cardio", "WBC"]}
    ckpt = tmp_path / "real_pooled.pt"
    engine.save_model(str(ckpt))
    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(str(ckpt))
    assert fresh._fusion_provenance == {"source": "adbench", "datasets": ["cardio", "WBC"]}
    assert fresh._fusion_feature_groups == engine._fusion_feature_groups

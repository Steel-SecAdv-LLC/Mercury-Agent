"""Network-gated end-to-end test for the real-data default-fusion training path.

Guards that ``scripts/train_default_fusion.py``'s real (ADBench) path actually
produces a discriminating, calibratable model — not just that it runs. Marked
``network`` + ``slow``: ``tests/conftest.py`` auto-skips it unless
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
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_real_adbench_training_learns_signal(tmp_path):
    import torch
    from sklearn.metrics import roc_auc_score

    from omni_mercury_engine.core.calibration import compute_ece
    from omni_mercury_engine.core.calibration import TemperatureScaling
    from omni_mercury_engine.engine import OmniMercuryEngine
    from omni_mercury_engine.ml.fusion_network import OmniFusionModel

    tdf = _load_train_module()

    # cardio: 1831 x 21, ~9.6% anomalies, genuine external labels. Capped small
    # so the per-sample extraction stays fast enough for a test.
    X, y = tdf._load_adbench("cardio", str(tmp_path))
    rng = np.random.default_rng(tdf.SEED)
    if len(X) > 600:
        sel = rng.choice(len(X), 600, replace=False)
        X, y = X[sel], y[sel]

    engine = OmniMercuryEngine(mode="fusion")
    feats = tdf.extract_inference_features(engine, X)

    # Stratified held-out test split.
    test_mask = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        test_mask[idx[: max(1, int(len(idx) * 0.25))]] = True
    keys = sorted(feats)
    tr = {k: feats[k][~test_mask] for k in keys}
    te = {k: feats[k][test_mask] for k in keys}
    y_tr, y_te = y[~test_mask], y[test_mask]

    model = OmniFusionModel(feature_dims={k: tr[k].shape[1] for k in keys}, hidden_dim=32)
    tdf.train(model, tr, y_tr, epochs=80, seed=tdf.SEED)

    model.eval()
    with torch.no_grad():
        probs = model(te)["anomaly_probs"].cpu().numpy().reshape(-1)

    auc = roc_auc_score(y_te, probs)
    assert auc > 0.8, f"real-data fusion training underperformed: AUC={auc:.3f}"

    # Temperature calibration must not degrade ranking and should not worsen ECE.
    calibrator = TemperatureScaling().fit(probs, y_te)
    cal_probs = calibrator.calibrate(probs)
    assert roc_auc_score(y_te, cal_probs) == pytest.approx(auc, abs=1e-6)
    assert compute_ece(y_te, cal_probs) <= compute_ece(y_te, probs) + 0.02

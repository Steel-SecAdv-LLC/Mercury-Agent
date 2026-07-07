# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""``engine.tune_fusion`` wires Mercury's Bayesian optimizer to fit_fusion.

The AutoML optimizer was reachable only from a gated subagent op with a
meaningless objective. ``tune_fusion`` gives it a real seam: a Bayesian search
over the ``fit_fusion`` hyperparameters scored by held-out ROC-AUC, refitting the
engine on the full data with the winning config. A ``tune`` CLI command exposes
it alongside ``train``.

The core wiring test drives ``tune_fusion`` against a stub engine (no torch, no
construction) so it runs on a base install; a torch end-to-end test and a CLI
smoke test cover the real path.
"""

from __future__ import annotations

import numpy as np
import pytest

_FIT_KEYS = {
    "learning_rate",
    "batch_size",
    "focal_alpha",
    "focal_gamma",
    "early_stopping_patience",
    "symbolic_weight",
}


class _StubEngine:
    """Minimal stand-in exposing the two methods tune_fusion calls."""

    def __init__(self) -> None:
        self.fit_calls: list[dict] = []

    def fit_fusion(self, X, y, **kwargs):  # noqa: ANN001
        self.fit_calls.append({"n": len(X), "config": kwargs})
        return {}

    def score_fusion(self, X):  # noqa: ANN001
        # Feature 0 encodes the label, so the ranking is perfect -> AUC 1.0.
        return np.asarray(X)[:, 0]


def test_tune_fusion_wiring_without_torch() -> None:
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(0)
    labels = (rng.random(80) > 0.5).astype(float)
    # Column 0 == label (+ tiny noise), so score_fusion perfectly separates.
    features = np.column_stack([labels + rng.normal(0, 0.01, 80), rng.normal(size=(80, 5))])

    stub = _StubEngine()
    result = OmniMercuryEngine.tune_fusion(
        stub, features, labels, n_trials=4, tuning_epochs=2, sampler="random", seed=0
    )

    # n_trials trial fits + one final refit on the full dataset.
    assert len(stub.fit_calls) == 4 + 1
    assert stub.fit_calls[-1]["n"] == len(features)
    assert all(call["n"] < len(features) for call in stub.fit_calls[:-1])

    # Best config carries exactly the tuned fit_fusion knobs, coerced to native types.
    assert set(result["best_config"]) == _FIT_KEYS
    first_config = stub.fit_calls[0]["config"]
    assert isinstance(first_config["batch_size"], int)
    assert isinstance(first_config["early_stopping_patience"], int)
    assert isinstance(first_config["learning_rate"], float)

    assert result["best_auc"] == pytest.approx(1.0)
    assert len(result["convergence_history"]) == 4


def test_tune_fusion_requires_both_classes() -> None:
    from omni_mercury_engine.engine import OmniMercuryEngine

    X = np.random.default_rng(0).normal(size=(20, 4))
    y = np.zeros(20)  # single class
    with pytest.raises(ValueError, match="both classes"):
        OmniMercuryEngine.tune_fusion(_StubEngine(), X, y, n_trials=2)


def test_tune_fusion_end_to_end_small() -> None:
    pytest.importorskip("torch")
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1.0, 0.5, (30, 6)), rng.normal(2.0, 0.5, (30, 6))])
    y = np.array([0] * 30 + [1] * 30)

    engine = OmniMercuryEngine(mode="fusion", device="cpu", require_explicit_fit=False)
    result = engine.tune_fusion(X, y, n_trials=2, tuning_epochs=2, sampler="random", seed=0)

    assert result["best_config"]
    assert 0.0 <= result["best_auc"] <= 1.0
    # The engine is refit with the best config and remains usable.
    scores = np.asarray(engine.score_fusion(X[:4]))
    assert scores.shape[0] == 4


def test_tune_cli_smoke(tmp_path, monkeypatch) -> None:
    from click.testing import CliRunner

    from omni_mercury_engine import cli

    rng = np.random.default_rng(0)
    data_path = tmp_path / "X.npy"
    labels_path = tmp_path / "y.npy"
    out_path = tmp_path / "model.pkl"
    np.save(data_path, rng.normal(size=(20, 6)))
    np.save(labels_path, np.array([0, 1] * 10))

    class _CliStub:
        def tune_fusion(self, X, y, **kwargs):  # noqa: ANN001
            return {
                "best_auc": 0.87,
                "best_config": {"learning_rate": 0.01, "batch_size": 32},
                "n_trials": kwargs.get("n_trials"),
                "convergence_history": [],
            }

        def save_model(self, path):  # noqa: ANN001
            open(path, "w").close()

    monkeypatch.setattr(cli, "_get_engine", lambda *a, **k: _CliStub())

    result = CliRunner().invoke(
        cli.main,
        [
            "tune",
            "--data",
            str(data_path),
            "--labels",
            str(labels_path),
            "--output",
            str(out_path),
            "--n-trials",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Best held-out AUC: 0.8700" in result.output
    assert "Tuned model saved" in result.output

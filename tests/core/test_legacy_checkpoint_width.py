# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A checkpoint written before the fitted width was recorded is still checked.

The bundled ``default_fusion.pt`` predates the ``fitted_feature_width`` key, so
the width guard added for fitted engines skipped it: ``mercury-agent detect -d
fusion`` on a 4-column CSV scored those columns against 16-column fitted
statistics and returned ``state: grounded, disposition: clear,
decision_confidence: 1.0`` with p ~ 1e-17 for rows shifted by six standard
deviations, while the drift check in the same result reported the widths
incomparable. The checkpoint still carries its training rows (the spatial
detector's state, the drift reference's source), so its width is recovered
from them on load and every serving surface refuses a mismatched width.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("torch")

import omni_mercury_engine.engine as engine_mod
from omni_mercury_engine.core.calibration import TemperatureScaling
from omni_mercury_engine.detectors.spatial import SpatialAnomalyDetector
from omni_mercury_engine.engine import OmniMercuryEngine, default_fusion_checkpoint_path

if TYPE_CHECKING:
    from pathlib import Path

BUNDLED_WIDTH = 16


@pytest.fixture
def bundled(monkeypatch: pytest.MonkeyPatch) -> OmniMercuryEngine:
    monkeypatch.delenv(engine_mod.FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(engine_mod, "installed_fusion_checkpoint_path", _absent)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    assert engine.load_default_fusion_checkpoint() is True
    return engine


def _absent() -> Path:
    from pathlib import Path

    return Path("/nonexistent/fusion.pt")


def test_the_bundled_checkpoint_predates_the_width_key() -> None:
    import torch

    blob = torch.load(default_fusion_checkpoint_path(), map_location="cpu", weights_only=False)
    assert "fitted_feature_width" not in blob


def test_its_width_is_recovered_from_its_training_rows(bundled: OmniMercuryEngine) -> None:
    assert bundled.fitted_feature_width == BUNDLED_WIDTH
    spatial = bundled.detectors["spatial"]
    assert isinstance(spatial, SpatialAnomalyDetector)
    state = spatial.get_fitted_state()
    assert state is not None
    points = state["lof_points"]
    assert np.asarray(points).shape[1] == BUNDLED_WIDTH


def test_a_mismatched_width_is_refused_not_scored(bundled: OmniMercuryEngine) -> None:
    rng = np.random.default_rng(0)
    narrow = np.vstack([rng.normal(size=(480, 4)), rng.normal(6.0, 1.0, size=(20, 4))])
    with pytest.raises(ValueError, match=r"4 feature column\(s\).*fitted on 16"):
        bundled.detect_with_fusion(narrow)


def test_the_fitted_width_still_scores(
    bundled: OmniMercuryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
    rng = np.random.default_rng(1)
    result = bundled.detect_with_fusion(rng.normal(size=(60, BUNDLED_WIDTH)))
    assert 0.0 <= float(result["anomaly_prob"]) <= 1.0
    assert result["drift_detection"]["status"] != "incomparable"


def test_a_recorded_none_width_stays_unchecked(tmp_path: Path) -> None:
    """``None`` written on purpose (pooled fit, dict input) is not overridden."""
    import torch

    from omni_mercury_engine.ml.mercury_ml import make_classification

    X, y = make_classification(n_samples=120, n_features=6, weights=[0.85, 0.15], random_state=0)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(X.astype(np.float32), y, epochs=3, early_stopping_patience=2)
    path = tmp_path / "pooled_like.pt"
    engine.save_model(str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["fitted_feature_width"] = None
    torch.save(blob, path)
    fresh = OmniMercuryEngine(mode="fusion", device="cpu")
    fresh.load_model(str(path))
    assert fresh.fitted_feature_width is None


def test_a_stale_detector_state_does_not_lend_its_width(
    bundled: OmniMercuryEngine, tmp_path: Path
) -> None:
    """Only rows restored by the load in hand count, never a previous load's."""
    import torch

    from omni_mercury_engine.ml.mercury_ml import make_classification

    X, y = make_classification(n_samples=120, n_features=8, weights=[0.85, 0.15], random_state=0)
    other = OmniMercuryEngine(mode="fusion", device="cpu")
    other.fit_fusion(X.astype(np.float32), y, epochs=3, early_stopping_patience=2)
    path = tmp_path / "legacy_without_detectors.pt"
    other.save_model(str(path))
    blob = torch.load(path, map_location="cpu", weights_only=False)
    del blob["fitted_feature_width"]
    blob["detector_fitted_state"] = {}
    torch.save(blob, path)
    assert bundled.fitted_feature_width == BUNDLED_WIDTH
    bundled.load_model(str(path))
    assert bundled.fitted_feature_width is None


@pytest.mark.parametrize("payload", ["[0, 0.1, 0.2, 0.3, 0.4]", '{"medical": [0.1, 0.2]}'])
def test_cli_refuses_one_dimensional_and_mapping_json_without_a_traceback(
    payload: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from omni_mercury_engine.cli import main

    monkeypatch.delenv(engine_mod.FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(engine_mod, "installed_fusion_checkpoint_path", _absent)
    data = tmp_path / "input.json"
    data.write_text(payload, encoding="utf-8")
    result = CliRunner().invoke(main, ["detect", "-i", str(data), "-d", "fusion"])
    assert result.exit_code == 1
    assert "fitted on 16" in result.output
    assert not isinstance(result.exception, ValueError)  # a usage error, not a crash


def test_cli_refuses_with_the_two_ways_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from click.testing import CliRunner

    from omni_mercury_engine.cli import main

    monkeypatch.delenv(engine_mod.FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(engine_mod, "installed_fusion_checkpoint_path", _absent)
    data = tmp_path / "four.csv"
    np.savetxt(data, np.random.default_rng(2).normal(size=(50, 4)), delimiter=",")
    result = CliRunner().invoke(main, ["detect", "-i", str(data), "-d", "fusion"])
    assert result.exit_code != 0
    assert "fitted on 16" in result.output
    assert "mercury-agent train" in result.output and "-d statistical" in result.output
    assert '"is_anomaly"' not in result.output


def test_mcp_flagship_returns_a_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from omni_mercury_engine.mcp_server import MercuryMCPServer

    monkeypatch.delenv(engine_mod.FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(engine_mod, "installed_fusion_checkpoint_path", _absent)
    server = MercuryMCPServer()
    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mercury_detect_fusion",
                "arguments": {"data": np.random.default_rng(3).normal(size=(30, 5)).tolist()},
            },
        }
    )
    assert response is not None
    result = response["result"]
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert "refused the input" in text and "fitted on 16" in text
    assert "internal tool error" not in text


def test_temperature_sigmoid_does_not_overflow() -> None:
    from scipy.special import expit

    scaler = TemperatureScaling()
    z = np.array([-1000.0, -40.0, 0.0, 40.0, 1000.0])
    with np.errstate(over="raise"):
        p = scaler._sigmoid(z)
    np.testing.assert_allclose(p, expit(z), rtol=0, atol=0)
    assert json.dumps(p.tolist())  # finite, serialisable

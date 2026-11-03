# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The served flagship model states its input width, and the console uses it.

``POST /detect/flagship`` refuses a matrix whose width differs from the served
model's (the bundled model has 16 columns). The Mercury Console's Flagship
card used to send a 16 x 2 built-in sample, which that refusal turns into a
guaranteed 400. ``GET /detect/flagship/model`` now reports the width, and the
card lays its sample out at it and skips a dataset of another width with the
reason.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import omni_mercury_engine.engine as engine_mod
from omni_mercury_engine.api.routes import detection

_CONSOLE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omni_mercury_engine"
    / "api"
    / "frontend_assets"
    / "static"
    / "console"
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi.testclient import TestClient

    from omni_mercury_engine.api.server import app

    monkeypatch.delenv(engine_mod.FUSION_CHECKPOINT_ENV, raising=False)
    monkeypatch.setattr(
        engine_mod, "installed_fusion_checkpoint_path", lambda: Path("/nonexistent/fusion.pt")
    )
    # A fresh served engine, so this test describes the bundled checkpoint.
    monkeypatch.setattr(detection, "_flagship_engine", None)
    return TestClient(app)


def test_the_served_model_reports_its_width(client: Any) -> None:
    response = client.get("/api/v1/detect/flagship/model")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["trained"] is True
    assert body["fitted_feature_width"] == 16
    assert body["provenance"]["source"] == "synthetic_gaussian_mixture"


def test_the_reported_width_is_the_one_detection_enforces(client: Any) -> None:
    width = client.get("/api/v1/detect/flagship/model").json()["fitted_feature_width"]
    narrow = [[0.1] * (width - 1)] * 4
    response = client.post("/api/v1/detect/flagship", json={"data": narrow})
    assert response.status_code == 400
    assert f"fitted on {width}" in response.text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_console_flagship_card_uses_the_served_width(tmp_path: Path) -> None:
    harness = tmp_path / "flagship.js"
    harness.write_text(
        """
global.window = {}; global.performance = { now: () => 0 };
require(process.argv[2]);
const M = window.MercuryCLIMass;
const flag = M.CAPS.filter((c) => c.id === "flagship");
const posted = [];
let contractReads = 0;
const A = {
  get: async (p) => { contractReads++; return { ok: p === "/api/v1/detect/flagship/model", data: { fitted_feature_width: 16 } }; },
  post: async (p, b) => { posted.push(b); return { ok: true, status: 200, data: {}, meta: {} }; },
};
(async () => {
  const sample = await M.run(A, flag, null, {}, () => {}, 1);
  const narrow = await M.run(A, flag, { matrix: [[1, 2, 3, 4]], cols: 4 }, {}, () => {}, 1);
  const wide = await M.run(A, flag, { matrix: [Array(16).fill(0.5)], cols: 16 }, {}, () => {}, 1);
  console.log(JSON.stringify({
    sampleWidths: [...new Set(posted[0].data.map((r) => r.length))],
    sample: sample[0].status, narrow: narrow[0].status, why: narrow[0].error,
    wide: wide[0].status, posts: posted.length, contractReads,
  }));
})();
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["node", str(harness), str(_CONSOLE / "mass.js")],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["sampleWidths"] == [16]
    assert out["sample"] == "ok" and out["wide"] == "ok"
    assert out["narrow"] == "skipped" and "fitted on 16 feature columns" in out["why"]
    assert out["posts"] == 2  # the 4-column dataset was never sent
    assert out["contractReads"] == 3  # once per run

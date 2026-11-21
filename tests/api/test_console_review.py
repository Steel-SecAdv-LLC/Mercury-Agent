# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The console's REVIEW A FILE path: what it sends, and how a review renders.

Browser-free.  Node executes the real generated summariser (``mass.js``) on a
real review result (a committed sample scored by the shipped checkpoint) and
the generated logic and template are read as text, pinning that:

* the picker posts ``{filename, content_base64}`` to ``POST
  /api/v1/review/inline`` -- the route's own request model -- and never to
  the simulated engine;
* the largest file the console sends fits the served request cap once
  base64-encoded;
* a review renders its upload, its flagged segments and where they are, the
  server's three plain-language answers word for word, the certificate's
  scope and regime, and then the complete decision panel -- with nothing
  shown as ``[object Object]``, ``undefined`` or ``NaN``.

The same path in a real browser, with a real file chosen in the picker, is
``tests/api/test_console_review_browser.py``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="fastapi (optional '[api]' extra) is required")
pytest.importorskip("torch")
pytest.importorskip("PIL")

_REPO = Path(__file__).resolve().parents[2]
_CONSOLE = _REPO / "src" / "omni_mercury_engine" / "api" / "frontend_assets" / "static" / "console"
_SAMPLES = _REPO / "examples" / "data" / "review"
_NODE = shutil.which("node")

sys.path.insert(0, str(_REPO))

from omni_mercury_engine import review
from scripts import build_console_port

_HARNESS = """
'use strict';
global.window = { matchMedia: () => ({ matches: false }) };
global.document = { documentElement: {} };
require(process.argv[2] + "/mass.js");
const M = window.MercuryCLIMass;
const results = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [kind, result] of Object.entries(results)) out[kind] = M.summarize(result);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def results() -> dict[str, dict[str, Any]]:
    """Each committed sample, reviewed by the shipped checkpoints."""
    samples = {
        "document": _SAMPLES / "sample_document.txt",
        "image": _SAMPLES / "sample_image.png",
        "audio": _SAMPLES / "sample_recording.wav",
    }
    return {
        kind: review.review_bytes(path.read_bytes(), path.name) for kind, path in samples.items()
    }


@pytest.fixture(scope="module")
def rendered(
    results: dict[str, dict[str, Any]], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, dict[str, Any]]:
    if _NODE is None:
        pytest.skip("node is not installed")
    folder = tmp_path_factory.mktemp("review-panel")
    (folder / "harness.js").write_text(_HARNESS, encoding="utf-8")
    (folder / "results.json").write_text(json.dumps(results, default=str), encoding="utf-8")
    done = subprocess.run(
        [_NODE, str(folder / "harness.js"), str(_CONSOLE), str(folder / "results.json")],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    summaries: dict[str, dict[str, Any]] = json.loads(done.stdout)
    return summaries


def _rows(summary: dict[str, Any]) -> dict[str, str]:
    return {key: value for key, value in summary["kv"]}


class TestWhatTheConsoleSends:
    def test_the_picker_posts_the_routes_own_request_model(self) -> None:
        from omni_mercury_engine.api.routes.review import InlineReviewRequest

        logic = (_CONSOLE / "logic.js").read_text(encoding="utf-8")
        assert (
            'this.A.post("/api/v1/review/inline", { filename: file.name, content_base64: content })'
            in logic
        )
        assert {"filename", "content_base64"} <= set(InlineReviewRequest.model_fields)
        assert "rd.readAsDataURL(file)" in logic

    def test_the_route_it_posts_to_is_served(self) -> None:
        from omni_mercury_engine.api.server import app

        assert "post" in app.openapi()["paths"]["/api/v1/review/inline"]

    def test_the_picker_is_in_the_page_and_labelled(self) -> None:
        template = (_CONSOLE / "template.js").read_text(encoding="utf-8")
        assert "REVIEW A FILE" in template
        assert "{{ onReviewFile }}" in template
        for suffix in (".pdf", ".docx", ".png", ".jpg", ".wav", ".txt"):
            assert suffix in build_console_port.REVIEW_ACCEPT
        assert build_console_port.REVIEW_ACCEPT in template

    def test_the_largest_file_sent_fits_the_served_cap_once_encoded(self) -> None:
        from omni_mercury_engine.serving_limits import load_serving_limits

        largest = build_console_port.REVIEW_MAX_FILE_MIB * 1024 * 1024
        encoded = -(-largest // 3) * 4
        assert encoded + 4096 <= load_serving_limits().review_max_body_bytes
        logic = (_CONSOLE / "logic.js").read_text(encoding="utf-8")
        assert f"const capMib = {build_console_port.REVIEW_MAX_FILE_MIB};" in logic

    def test_a_review_is_never_simulated(self) -> None:
        """Without a live server the console says so; it does not invent a review of a real file."""
        logic = (_CONSOLE / "logic.js").read_text(encoding="utf-8")
        method = logic[logic.index("  reviewFile(file) {") : logic.index("  loadFile(file) {")]
        guard = method.index('startsWith("sim://")')
        assert guard < method.index("if (!this.live() || simulated)") < method.index("this.A.post(")
        assert "never simulated" in method
        sim = (
            (_CONSOLE / "sim.js").read_text(encoding="utf-8")
            if (_CONSOLE / "sim.js").is_file()
            else ""
        )
        assert "/api/v1/review" not in sim


class TestHowAReviewRenders:
    @pytest.mark.parametrize("kind", ["document", "image", "audio"])
    def test_the_headline_states_the_decision(
        self, rendered: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]], kind: str
    ) -> None:
        decision, block = results[kind]["decision"], results[kind]["review"]
        verdict = (
            "ABSTAINED"
            if decision["abstained"]
            else "ANOMALOUS SEGMENT" if decision["decision_label"] else "NO ANOMALOUS SEGMENT"
        )
        assert rendered[kind]["headline"] == (
            f"{verdict} · {block['flagged_count']} of {block['segments']} segments flagged · {kind}"
        )

    @pytest.mark.parametrize("kind", ["document", "image", "audio"])
    def test_the_servers_three_answers_are_shown_word_for_word(
        self, rendered: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]], kind: str
    ) -> None:
        rows, stated = _rows(rendered[kind]), results[kind]["review"]["answers"]
        assert rows["ANSWER 1 · WHY THIS SEGMENT"] == stated["why_this_segment"]["plain"]
        assert rows["ANSWER 2 · HOW SURE"] == stated["how_sure"]["plain"]
        assert rows["ANSWER 3 · THE RECORD"] == stated["the_record"]["plain"]

    @pytest.mark.parametrize("kind", ["document", "image", "audio"])
    def test_the_upload_the_segments_and_the_scope_are_shown(
        self, rendered: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]], kind: str
    ) -> None:
        rows, block = _rows(rendered[kind]), results[kind]["review"]
        assert block["filename"] in rows["REVIEW"] and f"{block['bytes']} bytes" in rows["REVIEW"]
        assert rows["flagged segments"] == f"{block['flagged_count']} of {block['segments']}"
        assert rows["top segment"].startswith(f"#{block['top_segment']['index']} · p ")
        where = {"document": "characters ", "image": " px at (", "audio": " s"}[kind]
        assert where in rows["top segment"]
        for feature in block["top_segment"]["top_features"]:
            assert feature["name"] in rows["sets it apart"]
        assert rows["certificate scope"] == block["certificate_scope"]
        assert rows["basis"] == block["basis"]
        assert rows["sha256"] == block["sha256"]
        assert rows["harm gate"].startswith("passed · read ")
        gate = block["harm_gate"]
        assert rows["harm gate did not inspect"] == gate["not_inspected"]
        flagged_rows = [key for key in rows if key.startswith("· segment #")]
        assert len(flagged_rows) == min(8, len(block["flagged_segments"]))

    @pytest.mark.parametrize("kind", ["document", "image", "audio"])
    def test_the_complete_decision_panel_follows_with_the_certificates_regime_in_words(
        self, rendered: dict[str, dict[str, Any]], results: dict[str, dict[str, Any]], kind: str
    ) -> None:
        keys = [key for key, _value in rendered[kind]["kv"]]
        assert keys.index("ANSWER 3 · THE RECORD") < keys.index("1 · WHY THIS ROW")
        rows = _rows(rendered[kind])
        segments = results[kind]["review"]["segments"]
        assert rows["· certificate"].startswith("batch-level · covers whether this batch of ")
        assert f"{segments} row(s)" in rows["· certificate"]

    @pytest.mark.parametrize("kind", ["document", "image", "audio"])
    def test_nothing_renders_as_a_placeholder(
        self, rendered: dict[str, dict[str, Any]], kind: str
    ) -> None:
        text = (
            rendered[kind]["headline"]
            + "\n"
            + "\n".join(f"{key} {value}" for key, value in rendered[kind]["kv"])
        )
        for placeholder in ("[object Object]", "undefined", "NaN"):
            assert placeholder not in text

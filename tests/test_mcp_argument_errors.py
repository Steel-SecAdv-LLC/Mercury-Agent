# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MCP tools answer a bad argument with what is wrong and what to send instead.

Found by calling every tool over stdio with small valid and invalid inputs:

* ``mercury_tier_detect`` advertises ``minItems: 8``, but a default run on 8 or
  9 points failed from inside one member with ``SPOT calibration needs at
  least 10 samples``, naming neither the argument nor a way out;
* ``mercury_write_document`` with ``format: "pdf"`` reached the generator's
  ``ValueError``, which the dispatcher reports as ``internal tool error``;
* ``mercury_self_consistency`` thresholded ``prob: 7`` as "positive" and
  ``prob: -1`` as "negative".
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.detectors.detection_tier import (
    SPOT_MIN_CALIBRATION_POINTS,
    run_tier_ensemble,
)
from omni_mercury_engine.mcp_server import _DOCUMENT_FORMATS, MercuryMCPServer, ToolError

_SERIES = [0.42, 0.47, 0.44, 0.46, 0.43, 0.45, 3.9, 0.44]
_SECTIONS = [{"heading": "Findings", "body": "One spike at index 6."}]


@pytest.fixture(scope="module")
def server() -> MercuryMCPServer:
    return MercuryMCPServer()


def _schema(server: MercuryMCPServer, name: str) -> dict[str, Any]:
    schema: dict[str, Any] = server._tools[name].input_schema
    return schema


class TestTierMinimum:
    """The default tier needs 10 points because spot_evt does; the message says so."""

    @pytest.mark.parametrize("points", [8, 9])
    def test_too_few_points_for_the_default_tier_names_the_fix(self, points: int) -> None:
        series = (_SERIES + [0.5])[:points]
        with pytest.raises(ValueError, match="spot_evt") as raised:
            run_tier_ensemble(np.asarray(series))
        message = str(raised.value)
        assert f"'data' holds {points}" in message
        assert f"Send {SPOT_MIN_CALIBRATION_POINTS} or more points" in message
        assert "'subset' without spot_evt" in message

    def test_a_short_reference_is_named_as_the_short_series(self) -> None:
        with pytest.raises(ValueError, match="'reference' holds 8"):
            run_tier_ensemble(
                np.asarray(_SERIES + [0.5, 0.41, 0.4, 0.43]), reference=np.asarray(_SERIES)
            )

    def test_ten_points_run_the_full_tier(self) -> None:
        result = run_tier_ensemble(np.asarray(_SERIES + [0.5, 0.41]))
        assert "spot_evt" in result["members"]
        assert result["n_points"] == SPOT_MIN_CALIBRATION_POINTS

    def test_eight_points_still_run_a_subset_without_spot(self) -> None:
        """The declared minimum of 8 stays true for the members that can use it."""
        result = run_tier_ensemble(np.asarray(_SERIES), subset=("spectral_residual", "bocpd"))
        assert result["members"] == ["spectral_residual", "bocpd"]

    def test_the_constant_is_the_detectors_own_limit(self) -> None:
        from omni_mercury_engine.detectors.spot_evt import SPOTDetector

        short = np.linspace(0.0, 1.0, SPOT_MIN_CALIBRATION_POINTS - 1)
        with pytest.raises(ValueError, match="at least 10 samples"):
            SPOTDetector().fit(short)
        SPOTDetector().fit(np.linspace(0.0, 1.0, SPOT_MIN_CALIBRATION_POINTS))

    def test_the_tool_reports_it_as_a_tool_error(self, server: MercuryMCPServer) -> None:
        with pytest.raises(ToolError, match="Send 10 or more points"):
            server._tool_tier_detect({"data": _SERIES})

    def test_the_schema_tells_a_client_before_it_calls(self, server: MercuryMCPServer) -> None:
        data = _schema(server, "mercury_tier_detect")["properties"]["data"]
        assert data["minItems"] == 8
        assert "spot_evt, which needs 10" in data["description"]


class TestDocumentFormat:
    """An unsupported format is the caller's mistake, not a server fault."""

    @pytest.mark.parametrize("fmt", ["pdf", "docx", "txt", ""])
    def test_an_unsupported_format_names_the_choices(
        self, server: MercuryMCPServer, fmt: str
    ) -> None:
        with pytest.raises(ToolError, match="'format' must be one of markdown, html, text"):
            server._tool_write_document({"title": "T", "sections": _SECTIONS, "format": fmt})

    @pytest.mark.parametrize("fmt", _DOCUMENT_FORMATS)
    def test_every_advertised_format_is_produced(self, server: MercuryMCPServer, fmt: str) -> None:
        out = json.loads(
            server._tool_write_document({"title": "T", "sections": _SECTIONS, "format": fmt})
        )
        assert out["format"] == fmt and "Findings" in out["content"]

    def test_the_schema_enum_is_the_checked_list(self, server: MercuryMCPServer) -> None:
        enum = _schema(server, "mercury_write_document")["properties"]["format"]["enum"]
        assert tuple(enum) == _DOCUMENT_FORMATS


class TestSelfConsistencyProbability:
    """A probability outside [0, 1] is refused, not thresholded."""

    @pytest.mark.parametrize("prob", [7, -1, 1.0001, True, "0.5", [0.5]])
    def test_an_impossible_probability_is_refused(
        self, server: MercuryMCPServer, prob: Any
    ) -> None:
        with pytest.raises(ToolError, match=r"'prob' must be a number in \[0,1\]"):
            server._tool_self_consistency({"answers": ["a", "a", "b"], "prob": prob})

    @pytest.mark.parametrize("prob", [0, 0.6, 1])
    def test_a_probability_in_range_is_used(self, server: MercuryMCPServer, prob: float) -> None:
        out = json.loads(server._tool_self_consistency({"answers": ["a", "a", "b"], "prob": prob}))
        assert out["decision"]["decision"] in {"positive", "negative", "abstain"}


def test_the_interconnect_guide_lists_every_tool_the_server_registers(
    server: MercuryMCPServer,
) -> None:
    """The guide's table held 14 of the 20 tools; the six assessment tools were missing."""
    import re
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[1] / "docs" / "INTERCONNECT.md").read_text("utf-8")
    documented = set(re.findall(r"^\| `(mercury_[a-z_]+)` \|", guide, re.MULTILINE))
    assert documented == set(server._tools)


class TestLabelledTier:
    """A labelled run of the default tier works; it failed on every input.

    The ensemble fits its members on the label-0 points only, and ``rca``
    cannot score a series of a different length than its fit, so
    ``tier-detect -l`` and the supervised methods raised ``input has N nodes
    but the detector was fitted on N-1`` whenever one point was labelled 1.
    """

    @staticmethod
    def _labelled() -> tuple[Any, Any]:
        rng = np.random.default_rng(0)
        series = 1.0 + 0.1 * rng.standard_normal(60)
        series[40] = 9.0
        labels = np.zeros(60, dtype=int)
        labels[40] = 1
        return series, labels

    @pytest.mark.parametrize("method", [None, "stacking", "bma"])
    def test_the_default_tier_runs_with_labels(self, method: str | None) -> None:
        series, labels = self._labelled()
        result = run_tier_ensemble(series, labels=labels, method=method)
        assert "rca" not in result["members"]
        assert "label-0 points only" in result["excluded_members"]["rca"]
        assert result["flags"][40] == 1

    def test_asking_for_rca_with_labels_is_refused_by_name(self) -> None:
        series, labels = self._labelled()
        with pytest.raises(ValueError, match="rca cannot be used with labels"):
            run_tier_ensemble(series, labels=labels, subset=("rca", "bocpd"))

    def test_labels_that_mark_nothing_keep_rca(self) -> None:
        """All-zero labels fit on every point, so nothing has to be left out."""
        series, _ = self._labelled()
        result = run_tier_ensemble(series, labels=np.zeros(60, dtype=int), method="average")
        assert "rca" in result["members"] and "excluded_members" not in result

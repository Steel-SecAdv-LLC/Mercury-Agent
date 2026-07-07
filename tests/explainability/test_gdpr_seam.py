# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The GDPR Article 22 report is produced on the live detection path.

The top-level ``explainability`` package (from-scratch Shapley + scipy
counterfactuals + GDPR audit layer) was runtime-orphaned. ``detect_with_fusion``
now attaches its report under ``result["gdpr_report"]`` when opted in, built over
the same ``score_fusion`` serve path the decision reports. These pin that the
report is genuinely produced (not just importable), is JSON-serialisable, and is
absent by default.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")


def _fitted_engine():
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(0)
    x_neg = rng.normal(-1.0, 0.5, (40, 6))
    x_pos = rng.normal(2.0, 0.5, (40, 6))
    x_train = np.vstack([x_neg, x_pos])
    y_train = np.array([0] * 40 + [1] * 40)

    engine = OmniMercuryEngine(mode="fusion", device="cpu", require_explicit_fit=False)
    engine.fit_fusion(x_train, y_train, epochs=4, batch_size=32)
    return engine, x_pos


def test_fit_fusion_captures_shap_background() -> None:
    engine, _ = _fitted_engine()
    assert engine._fusion_background is not None
    assert engine._fusion_background.ndim == 2
    assert engine._fusion_background.shape[1] == 6


def test_gdpr_report_attached_when_requested() -> None:
    engine, x_pos = _fitted_engine()
    result = engine.detect_with_fusion(
        x_pos[:1], domain="general", gdpr_report=True, subject_id="subj_1"
    )
    assert "gdpr_report" in result
    report = result["gdpr_report"]
    # The report is a JSON-serialisable dict carrying real content.
    assert isinstance(report, dict)
    json.dumps(report)  # must not raise

    # It carries a decision and non-empty top factors somewhere in the payload.
    top_factors = report.get("top_factors")
    if top_factors is None:
        top_factors = report.get("explanation", {}).get("top_factors")
    assert top_factors, "expected non-empty top contributing factors"


def test_gdpr_report_absent_by_default() -> None:
    engine, x_pos = _fitted_engine()
    result = engine.detect_with_fusion(x_pos[:1], domain="general")
    assert "gdpr_report" not in result


def test_gdpr_report_generates_unique_id_when_subject_omitted() -> None:
    """Omitting subject_id yields a unique per-report ``anon-`` id, never the old
    constant that collapsed every anonymous audit onto one identifier.
    """
    engine, x_pos = _fitted_engine()
    r1 = engine.detect_with_fusion(x_pos[:1], domain="general", gdpr_report=True)["gdpr_report"]
    r2 = engine.detect_with_fusion(x_pos[:1], domain="general", gdpr_report=True)["gdpr_report"]
    s1, s2 = json.dumps(r1), json.dumps(r2)
    assert "anon-" in s1 and "anon-" in s2
    assert "unspecified_subject" not in s1
    # Two anonymous reports get distinct subject ids rather than a shared constant.
    assert r1 != r2

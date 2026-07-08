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


def test_shap_background_is_bounded_and_sampled_from_training_data() -> None:
    """The background is capped at 100 rows and drawn from the training data.

    Sampling uniformly (not the first 100 rows) avoids biasing the SHAP baseline
    when the training data is ordered (e.g. grouped by label).
    """
    from omni_mercury_engine.engine import OmniMercuryEngine

    rng = np.random.default_rng(0)
    # 160 rows, grouped by label (ordered) -> first-100 would be all class 0.
    x_train = np.vstack([rng.normal(-1.0, 0.5, (80, 6)), rng.normal(2.0, 0.5, (80, 6))])
    y_train = np.array([0] * 80 + [1] * 80)

    engine = OmniMercuryEngine(mode="fusion", device="cpu", require_explicit_fit=False)
    engine.fit_fusion(x_train, y_train, epochs=2, batch_size=32)

    background = engine._fusion_background
    assert background is not None
    assert background.shape == (100, 6)  # capped at 100
    # Every background row is an actual training row (a sampled subset).
    assert all(any(np.allclose(row, tr) for tr in x_train) for row in background[:8])


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


def test_gdpr_report_non_degenerate_without_background() -> None:
    """Without a fit background the report uses a neutral (zero) baseline.

    Using the instance as its own SHAP background would make every attribution
    ~0. Falling back to a zero baseline keeps the report meaningful, so at least
    one top factor must carry a non-zero contribution.
    """
    engine, x_pos = _fitted_engine()
    engine._fusion_background = None  # force the no-background fallback

    result = engine.detect_with_fusion(
        x_pos[:1], domain="general", gdpr_report=True, subject_id="s"
    )
    report = result["gdpr_report"]
    top_factors = report.get("top_factors") or report.get("explanation", {}).get("top_factors")
    assert top_factors, "expected non-empty top factors even without a background"
    contributions = [abs(float(f.get("contribution", 0.0))) for f in top_factors]
    assert any(c > 1e-9 for c in contributions), "attributions collapsed to zero (degenerate)"


def test_gdpr_threshold_zero_is_not_flipped_to_half() -> None:
    """A genuine threshold of 0.0 must not silently become 0.5 (falsy-`or` bug),
    which would flip an adverse decision to 'Normal' in the report."""
    engine, x_pos = _fitted_engine()
    instance = x_pos[0]
    # score 0.3 with threshold 0.0 -> adverse (Anomaly); with 0.5 -> Normal.
    report = engine._gdpr_explain_fusion_decision(
        instance, "s", {"anomaly_prob": 0.3, "threshold_used": 0.0}
    )
    decision = report.get("decision", {})
    value = decision.get("value") or decision.get("decision") or str(report)
    assert "Anomaly" in str(value)


def test_gdpr_report_survives_stale_width_background() -> None:
    """A background from a different-width training run must not crash the report;
    the helper falls back to a neutral zero baseline."""
    engine, x_pos = _fitted_engine()  # width-6 engine
    engine._fusion_background = np.zeros((10, 3))  # stale, wrong width (3 != 6)
    result = engine.detect_with_fusion(
        x_pos[:1], domain="general", gdpr_report=True, subject_id="s"
    )
    assert "gdpr_report" in result  # produced, not crashed


def test_gdpr_report_ignores_mismatched_feature_names() -> None:
    """Drift feature-names of a different width must be ignored, not crash."""
    engine, x_pos = _fitted_engine()  # width-6 instances
    engine._drift_feature_names = [f"f{i}" for i in range(9)]  # wrong length
    result = engine.detect_with_fusion(
        x_pos[:1], domain="general", gdpr_report=True, subject_id="s"
    )
    assert "gdpr_report" in result


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

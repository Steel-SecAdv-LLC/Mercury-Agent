# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is measured, calibrated and served is one function.

``score_fusion`` is the function benchmarks measure and conformal calibration
is fitted on; ``detect_with_fusion`` is the function that is served. Measured
before these fixes they were not the same function:

* serving dropped the features of every domain model whose ``predict``
  refused (three trained groups on a default fit), feeding the network zero
  tokens where training, ``score_fusion`` and calibration fed real features;
* serving floored out-of-support rows at the detector consensus and
  ``score_fusion`` did not, so conformal thresholds were fitted on
  unguarded probabilities and applied to guarded ones;
* ``detect_with_fusion_calibrated`` re-ran the fusion without the guard and
  thresholded probabilities the served verdict never saw.

These tests pin one scoring core behind every one of those surfaces.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.engine import OmniMercuryEngine

WIDTH = 8
N_OUTSIDE = 6


def _labelled(seed: int = 41) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(180, WIDTH)), rng.normal(size=(40, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(180), np.ones(40)])
    order = rng.permutation(len(x))
    return x[order], y[order]


@pytest.fixture(scope="module")
def engine() -> OmniMercuryEngine:
    """A labelled fit without temperature scaling, so probabilities are not saturated."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = OmniMercuryEngine(mode="fusion", device="cpu")
        x, y = _labelled()
        fitted.fit_fusion(x, y, epochs=3, calibrate=False)
    return fitted


@pytest.fixture(scope="module")
def probe() -> np.ndarray[Any, Any]:
    """In-support rows followed by rows far outside the training envelope.

    The outside rows sit opposite the labelled anomalies (which are shifted
    up), where the network extrapolates "normal" and the floor has to act.
    """
    rng = np.random.default_rng(42)
    rows = rng.normal(size=(24, WIDTH)) * 0.5
    rows[-N_OUTSIDE:] -= 9.0
    return rows.astype(np.float32)


def _served(engine: OmniMercuryEngine, rows: np.ndarray[Any, Any]) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result: dict[str, Any] = engine.detect_with_fusion(rows)
    return result


def test_serving_feeds_the_network_every_group_it_was_trained_on(
    engine: OmniMercuryEngine, probe: np.ndarray[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose ``predict`` refuses still contributes its trained features."""
    fed: list[set[str]] = []
    real = engine._timed_fusion_predict

    def recording(features: dict[str, Any]) -> dict[str, Any]:
        fed.append(set(engine._restrict_to_trained_groups(features)))
        return real(features)

    monkeypatch.setattr(engine, "_timed_fusion_predict", recording)
    _served(engine, probe)
    assert engine._fusion_feature_groups is not None
    assert fed, "serving did not run the fusion network"
    missing = set(engine._fusion_feature_groups) - fed[-1]
    assert not missing, f"serving starved trained feature groups: {sorted(missing)}"


def test_score_fusion_returns_the_served_probabilities(
    engine: OmniMercuryEngine, probe: np.ndarray[Any, Any]
) -> None:
    """Out-of-support rows included: benchmarks measure what is served."""
    served = _served(engine, probe)
    served_probs = np.asarray(served["anomaly_probs"], dtype=np.float64)
    outside = served["out_of_support_rows"]
    assert set(range(len(probe) - N_OUTSIDE, len(probe))) <= set(outside)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scored = np.asarray(engine.score_fusion(probe), dtype=np.float64)
        unguarded = np.asarray(
            engine.score_fusion(probe, apply_support_guard=False), dtype=np.float64
        )
    np.testing.assert_allclose(scored, served_probs, rtol=0, atol=1e-9)
    # The floor acts on this probe (otherwise the equality above is vacuous),
    # and only on out-of-support rows.
    inside = np.setdiff1d(np.arange(len(probe)), outside)
    assert np.any(scored[outside] > unguarded[outside] + 1e-6)
    assert np.all(scored[outside] >= unguarded[outside])
    np.testing.assert_array_equal(scored[inside], unguarded[inside])


class _RecordingConformal:
    """Stands in for the conformal classifier; keeps the scores it is fitted on."""

    fitted_on: list[np.ndarray[Any, Any]] = []

    def __init__(self, coverage: float = 0.9) -> None:
        self.coverage = coverage
        self.coverage_achieved = coverage

    def fit(self, probs: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> _RecordingConformal:
        type(self).fitted_on.append(np.asarray(probs, dtype=np.float64).reshape(-1).copy())
        return self

    def coverage_report(self, probs: Any, y: Any) -> dict[str, Any]:
        return {"thresholds": {0: 0.5, 1: 0.5}}


@pytest.mark.parametrize("per_sample", [False, True])
def test_conformal_calibration_is_fitted_on_served_scores(
    engine: OmniMercuryEngine,
    probe: np.ndarray[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    per_sample: bool,
) -> None:
    """The certificate's calibration scores are the scores serving reports.

    Thresholds fitted on unguarded probabilities and applied to guarded ones
    are not exchangeable with what they are compared against.
    """
    from omni_mercury_engine import engine as engine_mod

    if per_sample:
        expected = np.array(
            [_served(engine, probe[i : i + 1])["anomaly_prob"] for i in range(len(probe))]
        )
    else:
        expected = np.asarray(_served(engine, probe)["anomaly_probs"], dtype=np.float64)
    labels = np.zeros(len(probe), dtype=int)
    labels[-N_OUTSIDE:] = 1
    _RecordingConformal.fitted_on = []
    monkeypatch.setattr(engine_mod, "BinaryConformalClassifier", _RecordingConformal)
    monkeypatch.setattr(engine, "_fusion_conformal", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.calibrate_fusion_conformal(probe, labels, per_sample=per_sample)
    assert len(_RecordingConformal.fitted_on) == 1
    np.testing.assert_allclose(_RecordingConformal.fitted_on[0], expected, rtol=0, atol=1e-9)


def test_calibrated_detection_thresholds_the_served_probabilities(
    engine: OmniMercuryEngine, probe: np.ndarray[Any, Any]
) -> None:
    served = np.asarray(_served(engine, probe)["anomaly_probs"], dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrated = engine.detect_with_fusion_calibrated(probe, calibration_method="percentile")
    np.testing.assert_allclose(
        np.asarray(calibrated["anomaly_prob"], dtype=np.float64), served, rtol=0, atol=1e-9
    )
    assert len(calibrated["is_anomaly"]) == len(probe)


def test_serving_feeds_the_domain_encoder_feature_it_was_trained_with(
    probe: np.ndarray[Any, Any],
) -> None:
    """With a co-trained domain encoder, serving and ``score_fusion`` still agree."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = OmniMercuryEngine(mode="fusion", device="cpu")
        x, y = _labelled()
        fitted.fit_fusion(x, y, epochs=2, calibrate=False, domain_encoder=True)
        served = np.asarray(_served(fitted, probe)["anomaly_probs"], dtype=np.float64)
        scored = np.asarray(fitted.score_fusion(probe), dtype=np.float64)
    assert fitted._domain_encoder is not None
    np.testing.assert_allclose(scored, served, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# A detector that did not report is named, and its verdict is not grounded.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def certified_engine() -> OmniMercuryEngine:
    """A labelled fit with its automatic conformal certificate.

    Drift monitoring is off: these tests isolate the partial-evidence
    demotion, and their small probes would otherwise accumulate in the
    cross-call drift window and be deferred for drift instead.
    """
    from omni_mercury_engine.core.config import EngineConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = OmniMercuryEngine(
            mode="fusion", device="cpu", config=EngineConfig(drift_detection=False)
        )
        x, y = _labelled(seed=43)
        fitted.fit_fusion(x, y, epochs=25)
    assert fitted._fusion_conformal is not None
    return fitted


def _normal_rows(seed: int) -> np.ndarray[Any, Any]:
    return (np.random.default_rng(seed).normal(size=(12, WIDTH)) * 0.5).astype(np.float32)


@pytest.mark.parametrize("failure", ["timeout", "saturated"])
def test_a_skipped_trained_detector_is_listed_and_the_verdict_is_not_grounded(
    certified_engine: OmniMercuryEngine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: str,
) -> None:
    """Partial evidence is reported and deferred, not served as a grounded call.

    The skip used to be swallowed with a DEBUG line: the result carried no
    trace of it and the decision layer grounded a verdict on the remainder.
    """
    import logging

    from omni_mercury_engine.core.parallel_detectors import ExecutorSaturatedError

    control = _served(certified_engine, _normal_rows(61))
    assert "skipped_detectors" not in control
    assert control["decision"]["state"] == "grounded", "probe must ground without the fault"

    error: Exception = (
        TimeoutError("statistical exceeded the detector time budget")
        if failure == "timeout"
        else ExecutorSaturatedError("too many overrun tasks are still running")
    )

    def refuse(_data: Any) -> Any:
        raise error

    monkeypatch.setattr(certified_engine.detectors["statistical"], "extract_features", refuse)
    with caplog.at_level(logging.WARNING, logger="omni_mercury_engine.engine"):
        result = _served(certified_engine, _normal_rows(62 if failure == "timeout" else 63))

    skipped = result["skipped_detectors"]
    assert set(skipped) == {"statistical"}
    assert skipped["statistical"]["reason"] == failure
    assert skipped["statistical"]["trained_group"] is True
    assert any(
        record.levelno == logging.WARNING and "statistical" in record.getMessage()
        for record in caplog.records
    ), "a capacity skip must be logged at WARNING"
    decision = result["decision"]
    assert decision["state"] != "grounded"
    assert decision["disposition"] == "defer"
    assert decision["decision_label"] is None
    assert any("partial evidence" in reason for reason in decision["reasons"])
    assert decision["signals"]["skipped_detectors"] == ["statistical"]


def test_interleaved_calls_do_not_see_each_others_skips(
    certified_engine: OmniMercuryEngine,
) -> None:
    """The skip list is per call: a clean call after a faulted one reports nothing."""
    original = certified_engine.detectors["statistical"].extract_features
    calls = {"n": 0}

    def flaky(data: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("statistical exceeded the detector time budget")
        return original(data)

    certified_engine.detectors["statistical"].extract_features = flaky  # type: ignore[method-assign]
    try:
        first = _served(certified_engine, _normal_rows(64))
        second = _served(certified_engine, _normal_rows(65))
    finally:
        del certified_engine.detectors["statistical"].extract_features
    assert "statistical" in first["skipped_detectors"]
    assert "skipped_detectors" not in second


# ---------------------------------------------------------------------------
# One threshold per result: recalibration cannot leave stale derived fields.
# ---------------------------------------------------------------------------


def test_every_derived_field_uses_the_recalibrated_threshold(
    engine: OmniMercuryEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``is_anomaly``, ``is_anomaly_rows``, the drift rows and ``threshold_used`` agree.

    Online recalibration used to overwrite ``threshold_used`` and the scalar
    ``is_anomaly`` after ``is_anomaly_rows`` and the drift payload had been
    computed with the static threshold, so a result's row verdicts did not
    follow from the threshold it reported.
    """
    rows = (np.random.default_rng(71).normal(size=(40, WIDTH)) * 0.6).astype(np.float32)
    static = _served(engine, rows)
    static_threshold = static["threshold_used"]
    probs = np.asarray(static["anomaly_probs"], dtype=np.float64)

    # A recalibrated operating point that splits the batch differently from
    # the static one: just below the batch median, with a step too small to
    # move within one batch.
    target = float(np.sort(probs)[len(probs) // 2 - 1]) + 1e-12
    assert abs(target - static_threshold) > 1e-6
    monkeypatch.setattr(engine, "_adaptive_conformal", None)
    monkeypatch.setattr(engine, "_recalibration_warmup", 30)
    engine.enable_online_recalibration(
        target_coverage=0.5, learning_rate=1e-15, initial_threshold=target, warmup=1
    )
    result = _served(engine, rows)

    threshold = result["threshold_used"]
    assert threshold == pytest.approx(target, abs=1e-9)
    assert result["adaptive_threshold"] == pytest.approx(threshold)
    served = np.asarray(result["anomaly_probs"], dtype=np.float64)
    expected_rows = (served > threshold).tolist()
    assert expected_rows != (served > static_threshold).tolist(), "probe does not discriminate"
    assert result["is_anomaly_rows"] == expected_rows
    assert result["is_anomaly"] is bool(any(expected_rows))
    assert result["is_anomaly"] == (result["anomaly_prob"] > threshold)
    drift = result["drift_detection"]
    assert drift["inlier_fraction"] == pytest.approx(1.0 - float(np.mean(expected_rows)))
    assert result["decision"]["threshold"] == pytest.approx(threshold)


# ---------------------------------------------------------------------------
# Drift is visible to single-sample serving.
# ---------------------------------------------------------------------------


def _shipped_stream(n: int, seed: int, shift: float = 0.0) -> np.ndarray[Any, Any]:
    """Rows from the shipped checkpoint's training distribution, optionally shifted."""
    from scripts.train_default_fusion import N_FEATURES, SEED

    centers = np.random.default_rng(SEED).normal(0.0, 4.0, size=(4, N_FEATURES))
    rng = np.random.default_rng(seed)
    rows = centers[rng.integers(0, 4, size=n)] + rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    return (rows + shift).astype(np.float32)


@pytest.fixture
def shipped() -> OmniMercuryEngine:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = OmniMercuryEngine(mode="fusion", require_explicit_fit=False)
        assert loaded.load_default_fusion_checkpoint()
    return loaded


def test_single_sample_serving_reaches_the_drift_check(shipped: OmniMercuryEngine) -> None:
    """Thirty one-row calls from a shifted stream raise the drift one batch raises.

    The check needs 20 normal-looking rows and nothing accumulated across
    calls, so one-sample serving reported ``insufficient_rows`` on every call
    and the drift deferral could never fire.
    """
    # A one-sigma shift: far enough to drift, near enough that the model still
    # calls the rows normal (only normal-looking rows are tested for drift).
    stream = _shipped_stream(30, seed=81, shift=1.0)
    batch = _served(shipped, stream)["drift_detection"]
    assert batch["status"] == "tested" and batch["is_drift"] is True
    assert "accumulated_rows" not in batch, "a testable batch is tested on its own"

    statuses = []
    for row in stream:
        result = _served(shipped, row.reshape(1, -1))
        statuses.append(result["drift_detection"]["status"])
    last = result["drift_detection"]
    assert statuses[0] == "insufficient_rows"
    assert last["status"] == "tested", statuses
    assert last["is_drift"] is True
    assert last["accumulated_rows"] >= shipped._DRIFT_MIN_ROWS
    assert last["n_tested"] == last["accumulated_rows"]
    assert result["decision"]["disposition"] == "defer"


def test_an_in_distribution_single_sample_stream_is_not_flagged(
    shipped: OmniMercuryEngine,
) -> None:
    result: dict[str, Any] = {}
    for row in _shipped_stream(40, seed=82):
        result = _served(shipped, row.reshape(1, -1))
    drift = result["drift_detection"]
    assert drift["status"] == "tested"
    assert drift["is_drift"] is False and drift["severity"] == "NONE"


def test_the_drift_window_is_bounded_and_dropped_on_a_reload(shipped: OmniMercuryEngine) -> None:
    rows = _shipped_stream(shipped._DRIFT_WINDOW_ROWS + 40, seed=83).astype(np.float64)
    for row in rows:
        payload = shipped._drift_payload(row.reshape(1, -1), None)
    assert payload is not None
    assert payload["accumulated_rows"] == shipped._DRIFT_WINDOW_ROWS
    assert len(shipped._drift_buffer) == shipped._DRIFT_WINDOW_ROWS
    np.testing.assert_array_equal(shipped._drift_buffer[-1], rows[-1])
    np.testing.assert_array_equal(shipped._drift_buffer[0], rows[40])
    assert shipped.load_default_fusion_checkpoint()
    assert len(shipped._drift_buffer) == 0


def test_rows_the_model_calls_anomalous_do_not_enter_the_window(
    shipped: OmniMercuryEngine,
) -> None:
    row = _shipped_stream(1, seed=84).astype(np.float64)
    payload = shipped._drift_payload(row, [True])
    assert payload is not None and payload["accumulated_rows"] == 0
    payload = shipped._drift_payload(np.full_like(row, np.nan), [False])
    assert payload is not None and payload["accumulated_rows"] == 0


def test_concurrent_single_sample_calls_keep_the_window_consistent(
    shipped: OmniMercuryEngine,
) -> None:
    import threading

    rows = _shipped_stream(240, seed=85).astype(np.float64)
    assert shipped._ensure_drift_reference()
    errors: list[BaseException] = []

    def feed(chunk: np.ndarray[Any, Any]) -> None:
        try:
            for row in chunk:
                shipped._drift_payload(row.reshape(1, -1), None)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=feed, args=(rows[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(shipped._drift_buffer) == shipped._DRIFT_WINDOW_ROWS
    assert all(buffered.shape == (rows.shape[1],) for buffered in shipped._drift_buffer)


# ---------------------------------------------------------------------------
# The σ_Immutable testing bypass is a test hook, not a production switch.
# ---------------------------------------------------------------------------


class _GateSpy:
    """Counts σ_Immutable enforcements while delegating to the real gate."""

    def __init__(self, gate: Any) -> None:
        self._gate = gate
        self.enforced = 0

    def enforce(self, *args: Any, **kwargs: Any) -> Any:
        self.enforced += 1
        return self._gate.enforce(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gate, name)


def test_the_testing_bypass_skips_the_gate_in_development(
    engine: OmniMercuryEngine, probe: np.ndarray[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from omni_mercury_engine import engine as engine_mod

    monkeypatch.delenv("MERCURY_ENV", raising=False)
    monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
    spy = _GateSpy(engine._sigma_immutable_gate)
    monkeypatch.setattr(engine, "_sigma_immutable_gate", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = engine.detect_with_fusion(probe, _enable_gosnn=False)
    assert spy.enforced == 0
    assert result["gosnn_metadata"]["sigma_immutable_backend"] == "testing_bypass"


@pytest.mark.parametrize("mode", ["production", " Production ", "prod"])
def test_the_testing_bypass_is_ignored_in_production(
    engine: OmniMercuryEngine,
    probe: np.ndarray[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    mode: str,
) -> None:
    """With ``MERCURY_ENV=production`` the flag changes nothing: the gate runs.

    ``prod`` is not a recognised mode; an unreadable environment is treated
    as production, the fail-closed side of not knowing.
    """
    import logging

    from omni_mercury_engine import engine as engine_mod
    from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError

    monkeypatch.setenv("MERCURY_ENV", mode)
    monkeypatch.setattr(engine_mod, "_GOSNN_TESTING_BYPASS", True)
    spy = _GateSpy(engine._sigma_immutable_gate)
    monkeypatch.setattr(engine, "_sigma_immutable_gate", spy)

    # The boundary gate runs although the flag is set ...
    with caplog.at_level(logging.ERROR, logger="omni_mercury_engine.engine"):
        engine._enforce_ethics_at_boundary(domain=None, data=probe)
    assert spy.enforced == 1
    assert any(
        record.levelno == logging.ERROR and "_GOSNN_TESTING_BYPASS" in record.getMessage()
        for record in caplog.records
    )
    # ... and asking to skip GOSNN fails closed exactly as it does without the flag.
    with pytest.raises(EthicalConstraintViolationError) as refused, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.detect_with_fusion(probe, _enable_gosnn=False)
    assert refused.value.check == "gosnn_unavailable"

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``fit_fusion`` is a function of its data and its seed, not of process history.

Measured before the fix: two engines built in different global-RNG realities
and fitted on the same rows returned different ``score_fusion`` probabilities,
because the train/validation permutation came from the global torch stream
(``torch.randperm`` with no generator), the fusion network was initialised
from it at construction, and dropout drew from it during training. The split
was not stratified either, so with rare positives the validation rows were
often single-class and temperature calibration plus the published
``val_roc_auc`` were skipped without a word; and the training rows were
batched in one fixed order for every epoch.

These tests pin the cure: a ``seed`` (default ``0``) drives a local generator
for the split and the per-epoch shuffle, the network a fresh fit trains is
built under that seed, the caller's global RNG state is left exactly as it
was, the split is stratified by label when both classes are present, and the
batches differ from one epoch to the next.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pytest
import torch

from omni_mercury_engine.engine import OmniMercuryEngine

WIDTH = 8


def _data(seed: int = 11) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(size=(160, WIDTH)), rng.normal(size=(24, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(160), np.ones(24)])
    order = rng.permutation(len(x))
    return x[order], y[order]


def _probe() -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(12)
    return np.vstack([rng.normal(size=(20, WIDTH)), rng.normal(size=(6, WIDTH)) + 3.0])


def _engine_built_under(ambient_seed: int) -> OmniMercuryEngine:
    """An engine constructed in one global-RNG reality (a stand-in for a process)."""
    torch.manual_seed(ambient_seed)
    np.random.seed(ambient_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OmniMercuryEngine(mode="fusion", device="cpu")


def _fit_and_score(ambient_seed: int, **fit_kwargs: Any) -> np.ndarray[Any, Any]:
    engine = _engine_built_under(ambient_seed)
    x, y = _data()
    # Process history between construction and the fit differs as well.
    torch.manual_seed(ambient_seed + 1)
    torch.rand(ambient_seed % 17 + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(x, y, epochs=3, **fit_kwargs)
        return np.asarray(engine.score_fusion(_probe()), dtype=np.float64)


def test_the_default_fit_does_not_depend_on_process_history() -> None:
    """Same data, same (default) seed, different RNG history: identical model."""
    first = _fit_and_score(0)
    second = _fit_and_score(999)
    np.testing.assert_array_equal(first, second)


def test_the_seed_selects_the_fit() -> None:
    """An explicit seed reproduces; a different seed is a different fit."""
    a = _fit_and_score(0, seed=7)
    b = _fit_and_score(999, seed=7)
    c = _fit_and_score(0, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c), "seeds 7 and 8 produced the same fitted model"


def test_a_fit_leaves_the_callers_global_rng_untouched() -> None:
    engine = _engine_built_under(3)
    x, y = _data()
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(x, y, epochs=2)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(4242)
    expected = torch.rand(5)
    torch.random.set_rng_state(before)
    assert torch.equal(expected, torch.rand(5))


def test_seed_none_keeps_the_ambient_stream_in_charge() -> None:
    """``seed=None`` is the opt-out: the global torch stream drives the fit."""
    engine = _engine_built_under(3)
    x, y = _data()
    torch.manual_seed(4242)
    before = torch.random.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(x, y, epochs=2, seed=None)
    assert not torch.equal(before, torch.random.get_rng_state())


def _feature_groups(
    n: int, n_positive: int, data_seed: int = 5
) -> tuple[dict[str, torch.Tensor], np.ndarray[Any, Any]]:
    rng = np.random.default_rng(data_seed)
    y = np.zeros(n)
    y[rng.choice(n, size=n_positive, replace=False)] = 1.0
    stat = rng.normal(size=(n, 10)) + 2.0 * y[:, None]
    spatial = rng.normal(size=(n, 32)) + 2.0 * y[:, None]
    return (
        {
            "statistical": torch.tensor(stat, dtype=torch.float32),
            "spatial": torch.tensor(spatial, dtype=torch.float32),
        },
        y,
    )


@pytest.mark.parametrize("case", range(12))
def test_rare_positives_reach_the_validation_split(case: int) -> None:
    """At 2% positives the validation split holds both classes, case after case.

    An unstratified 20% split of 300 rows with 6 positives carries none of
    them about one time in four, and then temperature calibration and
    ``val_roc_auc`` are skipped; over twelve differently-placed label sets
    (and ambient RNG states) the unstratified split passes this by chance
    about 3% of the time.
    """
    features, y = _feature_groups(300, 6, data_seed=400 + case)
    engine = _engine_built_under(1)
    torch.manual_seed(case)
    metrics = engine._fit_fusion_on_features(features, y, epochs=1)
    assert metrics["val_roc_auc"] is not None, "validation split was single-class"
    assert metrics["ece_before"] is not None, "temperature calibration was skipped"


def test_single_class_labels_fall_back_to_the_plain_split() -> None:
    features, _ = _feature_groups(60, 6)
    engine = _engine_built_under(1)
    metrics = engine._fit_fusion_on_features(features, np.zeros(60), epochs=1)
    assert metrics["epochs_trained"] == 1
    assert metrics["val_roc_auc"] is None


def test_training_batches_are_reshuffled_every_epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first training batch of each epoch holds different rows.

    The row order used to be drawn once, so every epoch replayed identical
    batches. Rows are identified by a marker column the fusion model receives.
    """
    features, y = _feature_groups(120, 20)
    marker = torch.arange(120, dtype=torch.float32)
    features["statistical"][:, 0] = marker
    engine = _engine_built_under(1)
    batches: list[frozenset[int]] = []
    real_forward = type(engine.fusion_model).forward

    def recording_forward(model: Any, groups: dict[str, torch.Tensor], *a: Any, **k: Any) -> Any:
        if model.training:
            batches.append(frozenset(int(v) for v in groups["statistical"][:, 0].tolist()))
        return real_forward(model, groups, *a, **k)

    monkeypatch.setattr(type(engine.fusion_model), "forward", recording_forward)
    engine._fit_fusion_on_features(
        features, y, epochs=3, batch_size=32, early_stopping_patience=10, calibrate=False
    )
    n_train = 120 - int(120 * 0.2)
    per_epoch = -(-n_train // 32)
    assert len(batches) == 3 * per_epoch
    first_batches = [batches[i * per_epoch] for i in range(3)]
    assert len(set(first_batches)) == 3, "every epoch replayed the same first batch"
    # Each epoch still visits every training row exactly once.
    for epoch in range(3):
        rows = [r for b in batches[epoch * per_epoch : (epoch + 1) * per_epoch] for r in b]
        assert len(rows) == n_train == len(set(rows))


def test_tune_fusion_threads_its_seed_into_the_fits(caplog: pytest.LogCaptureFixture) -> None:
    """Two tunes with one seed deliver the same model, whatever the RNG history."""
    x, y = _data()
    results = []
    for ambient in (0, 999):
        engine = _engine_built_under(ambient)
        torch.manual_seed(ambient + 5)
        with warnings.catch_warnings(), caplog.at_level(logging.ERROR):
            warnings.simplefilter("ignore")
            engine.tune_fusion(x, y, n_trials=1, tuning_epochs=1, sampler="random", seed=3)
            results.append(np.asarray(engine.score_fusion(_probe()), dtype=np.float64))
    np.testing.assert_array_equal(results[0], results[1])


# ---------------------------------------------------------------------------
# A second fit refits the detectors it fitted the first time.
# ---------------------------------------------------------------------------


def _fitted_state(engine: OmniMercuryEngine, name: str) -> dict[str, Any]:
    detector: Any = engine.detectors[name]
    state = detector.get_fitted_state()
    assert state is not None, f"{name} is not fitted"
    fitted: dict[str, Any] = state
    return fitted


def _spatial_rows(engine: OmniMercuryEngine) -> np.ndarray[Any, Any]:
    return np.asarray(_fitted_state(engine, "spatial")["lof_points"], dtype=np.float64)


def test_a_second_fit_refits_the_detectors_on_the_new_rows() -> None:
    """``fit_fusion(A)`` then ``fit_fusion(B)`` leaves detectors describing B.

    ``_fit_for_training`` used to return early for any fitted detector, so
    the second fit trained the network on B's features extracted against A's
    reference and served B's rows against A's reference thereafter.
    """
    rng = np.random.default_rng(21)
    first = rng.normal(size=(90, WIDTH))
    second = rng.normal(size=(110, WIDTH)) + 10.0
    engine = _engine_built_under(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(first, epochs=1)
        assert abs(float(_spatial_rows(engine).mean())) < 1.0
        engine.fit_fusion(second, epochs=1)
    rows = _spatial_rows(engine)
    assert (
        abs(float(rows.mean()) - 10.0) < 1.0
    ), "the spatial reference still describes the first fit"
    stat = _fitted_state(engine, "statistical")
    assert abs(float(np.mean(stat["mean"])) - 10.0) < 1.0


def test_a_refit_does_not_serve_features_cached_against_the_old_fit() -> None:
    rng = np.random.default_rng(22)
    first = rng.normal(size=(90, WIDTH))
    second = rng.normal(size=(90, WIDTH)) * 3.0 + 4.0
    probe = rng.normal(size=(12, WIDTH)).astype(np.float32)
    engine = _engine_built_under(2)
    reference = _engine_built_under(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(first, epochs=1)
        engine._extract_detector_features(probe)  # caches features against the first fit
        engine.fit_fusion(second, epochs=1)
        reference.fit_fusion(second, epochs=1)
        served = engine._extract_detector_features(probe)[0]
        expected = reference._extract_detector_features(probe)[0]
    assert set(served) == set(expected)
    for name in sorted(expected):
        np.testing.assert_array_equal(np.asarray(served[name]), np.asarray(expected[name]))


def test_a_fit_the_caller_made_is_kept_across_fits() -> None:
    """Only fits a training call made are replaced; the caller's own stand."""
    rng = np.random.default_rng(23)
    own = rng.normal(size=(70, WIDTH)) - 20.0
    engine = _engine_built_under(2)
    engine.detectors["spatial"].fit(own)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(rng.normal(size=(90, WIDTH)), epochs=1)
        engine.fit_fusion(rng.normal(size=(90, WIDTH)) + 10.0, epochs=1)
    np.testing.assert_array_equal(_spatial_rows(engine), own)


def test_tune_fusion_leaves_the_detectors_fit_on_the_full_data() -> None:
    """The delivered engine's detectors are those of a plain full-data fit.

    They used to stay fitted on the first trial's 75% train split, contrary
    to ``tune_fusion``'s "left fit on the full (X, y)".
    """
    x, y = _data()
    tuned = _engine_built_under(4)
    plain = _engine_built_under(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned.tune_fusion(x, y, n_trials=1, tuning_epochs=1, sampler="random", seed=3)
        plain.fit_fusion(x, y, epochs=1)
    np.testing.assert_array_equal(_spatial_rows(tuned), _spatial_rows(plain))


# ---------------------------------------------------------------------------
# Validation rows are held out of everything fitted for training.
# ---------------------------------------------------------------------------


class _TrainerSpy:
    """Records what ``fit_fusion`` hands the trainer, and the detectors' state then."""

    def __init__(self, engine: OmniMercuryEngine) -> None:
        self.engine = engine
        self.real = engine._fit_fusion_on_features
        self.features: dict[str, torch.Tensor] = {}
        self.labels: np.ndarray[Any, Any] = np.empty(0)
        self.train_rows: np.ndarray[Any, Any] = np.empty(0, dtype=int)
        self.val_rows: np.ndarray[Any, Any] = np.empty(0, dtype=int)
        self.spatial_rows_while_training: np.ndarray[Any, Any] = np.empty(0)

    def __call__(self, features: dict[str, torch.Tensor], y: Any, **kwargs: Any) -> dict[str, Any]:
        self.features = {name: value.clone() for name, value in features.items()}
        self.labels = np.asarray(y, dtype=np.float64).reshape(-1).copy()
        split = kwargs.get("split")
        assert split is not None, "fit_fusion did not draw the split before fitting detectors"
        self.train_rows = np.sort(split[0].numpy())
        self.val_rows = np.sort(split[1].numpy())
        self.spatial_rows_while_training = _spatial_rows(self.engine).copy()
        result: dict[str, Any] = self.real(features, y, **kwargs)
        return result


def _spied_fit(
    x: np.ndarray[Any, Any], monkeypatch: pytest.MonkeyPatch, **fit_kwargs: Any
) -> tuple[OmniMercuryEngine, _TrainerSpy]:
    engine = _engine_built_under(6)
    spy = _TrainerSpy(engine)
    monkeypatch.setattr(engine, "_fit_fusion_on_features", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(x, epochs=1, **fit_kwargs)
    return engine, spy


def test_detectors_are_fitted_on_the_training_rows_while_the_network_trains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No validation row is in a detector's reference during training."""
    x = np.random.default_rng(31).normal(size=(100, WIDTH))
    _, spy = _spied_fit(x, monkeypatch)
    assert len(spy.val_rows) == 20 and len(spy.train_rows) == 80
    np.testing.assert_array_equal(spy.spatial_rows_while_training, x[spy.train_rows])


def test_validation_features_are_what_serving_those_rows_extracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation features equal a served batch from a train-rows-only engine."""
    x = np.random.default_rng(32).normal(size=(100, WIDTH))
    _, spy = _spied_fit(x, monkeypatch)
    held_out = _engine_built_under(6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        held_out._extract_fusion_features(x[spy.train_rows], fit_detectors=True)
        served = held_out._extract_fusion_features(x[spy.val_rows], fit_detectors=False)
    assert set(spy.features) <= set(served)
    for name, value in spy.features.items():
        np.testing.assert_array_equal(value[spy.val_rows].numpy(), served[name].numpy())


def test_pseudo_labels_of_training_rows_ignore_the_validation_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing the validation rows leaves every training row's label alone.

    The consensus threshold used to be a percentile over all rows, so 20
    extreme validation rows took the whole anomaly quota and relabelled the
    training rows.
    """
    x = np.random.default_rng(33).normal(size=(100, WIDTH))
    _, spy = _spied_fit(x, monkeypatch, contamination=0.1)
    replaced = x.copy()
    replaced[spy.val_rows] += 40.0
    _, spy_replaced = _spied_fit(replaced, monkeypatch, contamination=0.1)
    np.testing.assert_array_equal(spy.train_rows, spy_replaced.train_rows)
    np.testing.assert_array_equal(spy.labels[spy.train_rows], spy_replaced.labels[spy.train_rows])
    assert spy.labels[spy.train_rows].sum() > 0


def test_the_served_detectors_are_fitted_on_all_of_the_training_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the fit returns, the documented promise holds: detectors fit on X."""
    x = np.random.default_rng(34).normal(size=(100, WIDTH))
    engine, spy = _spied_fit(x, monkeypatch)
    assert len(spy.spatial_rows_while_training) == 80
    np.testing.assert_array_equal(_spatial_rows(engine), x)
    stat = _fitted_state(engine, "statistical")
    np.testing.assert_allclose(stat["mean"], x.mean(axis=0))


def test_the_domain_scaler_is_a_training_row_statistic(monkeypatch: pytest.MonkeyPatch) -> None:
    x = np.random.default_rng(35).normal(size=(100, WIDTH)) * 2.0 + 1.0
    engine, spy = _spied_fit(x, monkeypatch, domain_encoder=True)
    assert engine._domain_scaler is not None
    np.testing.assert_allclose(
        engine._domain_scaler[0], x[spy.train_rows].astype(np.float32).mean(axis=0), rtol=1e-5
    )


def test_pooled_sources_hold_their_validation_rows_out_of_their_detectors() -> None:
    """Each pooled source's detectors are fitted on that source's training rows."""
    rng = np.random.default_rng(36)
    sources = []
    for shift in (0.0, 5.0):
        x = np.vstack([rng.normal(size=(80, WIDTH)), rng.normal(size=(20, WIDTH)) + 4.0]) + shift
        sources.append((x, np.concatenate([np.zeros(80), np.ones(20)])))
    extractors: list[OmniMercuryEngine] = []

    def factory() -> OmniMercuryEngine:
        extractors.append(_engine_built_under(7))
        return extractors[-1]

    engine = _engine_built_under(7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = engine.fit_fusion_pooled(sources, engine_factory=factory, epochs=1)
    assert metrics["val_roc_auc"] is not None
    for extractor in extractors:
        assert len(_spatial_rows(extractor)) == 80, "a source's detectors saw its validation rows"


def test_refit_detectors_false_serves_the_validated_detectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.random.default_rng(37).normal(size=(100, WIDTH))
    engine, spy = _spied_fit(x, monkeypatch, refit_detectors=False)
    np.testing.assert_array_equal(_spatial_rows(engine), x[spy.train_rows])


# ---------------------------------------------------------------------------
# A fresh fit registers exactly the feature groups it extracted.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_fit() -> tuple[OmniMercuryEngine, list[str]]:
    """A fresh labelled fit plus the fusion-network warnings it logged."""
    engine = _engine_built_under(8)
    x, y = _data()
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    fusion_logger = logging.getLogger("omni_mercury_engine.ml.fusion_network")
    fusion_logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            engine.fit_fusion(x, y, epochs=2)
    finally:
        fusion_logger.removeHandler(handler)
    return engine, records


def test_every_extracted_group_has_a_trained_projection(
    fresh_fit: tuple[OmniMercuryEngine, list[str]],
) -> None:
    """Registry == extracted groups: none dropped, none a never-fed zero token."""
    engine, warnings_logged = fresh_fit
    x, _ = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extracted = engine._extract_fusion_features(x[:40], fit_detectors=False)
    model = engine.fusion_model
    assert set(extracted) == set(model.feature_dims), (
        f"extracted but unregistered: {sorted(set(extracted) - set(model.feature_dims))}; "
        f"registered but never extracted: {sorted(set(model.feature_dims) - set(extracted))}"
    )
    for name, features in extracted.items():
        assert model.feature_dims[name] == int(np.prod(features.shape[1:]))
    assert engine._fusion_feature_groups == sorted(extracted)
    fusion_layer: Any = model.fusion_layer
    assert list(fusion_layer.detector_names) == list(model.feature_dims)
    assert len(model._dynamic_projections) == 0
    assert model.masked_mean_pooling is True
    assert fusion_layer.masked_mean_pooling is True
    dropped = [message for message in warnings_logged if "not registered" in message]
    assert not dropped, f"the fit extracted groups the model then dropped: {dropped}"


def test_a_fresh_fit_round_trips_its_architecture(
    fresh_fit: tuple[OmniMercuryEngine, list[str]], tmp_path: Any
) -> None:
    """save -> load rebuilds the same registry and pooling; scores are identical."""
    engine, _ = fresh_fit
    path = str(tmp_path / "fresh.pt")
    engine.save_model(path)
    restored = _engine_built_under(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        restored.load_model(path)
        probe = _probe()
        expected = np.asarray(engine.score_fusion(probe), dtype=np.float64)
        actual = np.asarray(restored.score_fusion(probe), dtype=np.float64)
        # The raw (pre-temperature) probabilities too: the calibrated ones
        # saturate, which would hide an architecture mismatch.
        saved_calibrator = engine._fusion_calibrator
        engine._fusion_calibrator = None
        restored._fusion_calibrator = None
        try:
            expected_raw = np.asarray(engine.score_fusion(probe), dtype=np.float64)
            actual_raw = np.asarray(restored.score_fusion(probe), dtype=np.float64)
        finally:
            engine._fusion_calibrator = saved_calibrator
    assert list(restored.fusion_model.feature_dims.items()) == list(
        engine.fusion_model.feature_dims.items()
    )
    assert restored.fusion_model.masked_mean_pooling is True
    assert len(restored.fusion_model._dynamic_projections) == 0
    # Fitted detector state round-trips through float32 tensors, so the
    # checkpoint contract is closeness at float32 resolution, as in
    # tests/test_fusion_checkpoint_roundtrip.py.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(actual_raw, expected_raw, rtol=0, atol=1e-6)
    assert float(np.ptp(expected_raw)) > 1e-3, "probe does not exercise the network"


def test_the_shipped_checkpoint_keeps_its_architecture() -> None:
    """Loading ``default_fusion.pt`` still builds the plain-mean, 13-group network."""
    from omni_mercury_engine.engine import default_fusion_checkpoint_path
    from omni_mercury_engine.security.safe_torch import safe_torch_load

    saved = safe_torch_load(str(default_fusion_checkpoint_path()), map_location="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OmniMercuryEngine(mode="fusion", device="cpu", auto_load_checkpoint=True)
    assert engine._fusion_trained is True
    assert engine.fusion_model.feature_dims == saved["feature_dims"]
    assert engine.fusion_model.masked_mean_pooling is False
    fusion_layer: Any = engine.fusion_model.fusion_layer
    assert fusion_layer.masked_mean_pooling is False


def test_loading_the_shipped_checkpoint_over_a_fresh_fit_restores_plain_pooling(
    fresh_fit: tuple[OmniMercuryEngine, list[str]],
) -> None:
    """The pooling mode follows the checkpoint, not the engine that loads it."""
    from omni_mercury_engine.engine import default_fusion_checkpoint_path

    donor, _ = fresh_fit
    engine = _engine_built_under(8)
    x, y = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.fit_fusion(x, y, epochs=1)
        assert engine.fusion_model.masked_mean_pooling is True
        engine.load_model(str(default_fusion_checkpoint_path()))
    assert engine.fusion_model.masked_mean_pooling is False
    assert donor.fusion_model.masked_mean_pooling is True


# ---------------------------------------------------------------------------
# The automatic conformal slice is large enough for the coverage it claims.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("coverage", "rows"), [(0.8, 4), (0.9, 9), (0.95, 19), (0.99, 99)])
def test_rows_needed_is_the_smallest_calibration_size_reaching_the_coverage(
    coverage: float, rows: int
) -> None:
    from omni_mercury_engine.core.conformal_prediction import SplitConformalPredictor

    assert OmniMercuryEngine._conformal_rows_needed(coverage) == rows
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enough = SplitConformalPredictor(coverage=coverage).fit(np.linspace(0.0, 1.0, rows))
        short = SplitConformalPredictor(coverage=coverage).fit(np.linspace(0.0, 1.0, rows - 1))
    assert enough.coverage_achieved == pytest.approx(coverage)
    assert short.coverage_achieved is not None
    assert short.coverage_achieved < coverage


def test_with_ample_data_the_certificate_achieves_the_requested_coverage() -> None:
    """40 positives: the slice holds 9 of them, not 5, so 0.9 is really 0.9.

    The 5-row floor capped the achieved coverage at 5/6 for every class of
    fewer than 90 rows, however much data there was to spare.
    """
    rng = np.random.default_rng(51)
    x = np.vstack([rng.normal(size=(200, WIDTH)), rng.normal(size=(40, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(200), np.ones(40)])
    engine = _engine_built_under(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = engine.fit_fusion(x, y, epochs=1)
    assert engine._fusion_conformal is not None
    assert engine._fusion_conformal.coverage_achieved >= 0.9
    assert metrics["conformal"]["coverage"] == 0.9
    assert metrics["conformal"]["n_calibration"] == 20 + 9
    assert metrics["conformal"]["coverage_achieved"] >= 0.9


def test_a_class_too_small_to_spare_keeps_the_five_row_slice() -> None:
    """14 positives: 5 are held out, 9 train, and the certificate says 5/6."""
    rng = np.random.default_rng(52)
    x = np.vstack([rng.normal(size=(200, WIDTH)), rng.normal(size=(14, WIDTH)) + 3.0])
    y = np.concatenate([np.zeros(200), np.ones(14)])
    engine = _engine_built_under(9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metrics = engine.fit_fusion(x, y, epochs=1)
    assert metrics["conformal"]["n_calibration"] == 20 + 5
    assert metrics["conformal"]["coverage_achieved"] == pytest.approx(5.0 / 6.0)

"""Offline tests for the WS-B design-space sweep verdict logic.

The full sweep trains the fusion path on ADBench (network + minutes) and is run
to produce ``artifacts/domain_encoder_sweep.json``. These tests cover the
math/verdict aggregation with synthetic cells -- no training, no network.
"""

from __future__ import annotations

from benchmarks.domain_encoder_sweep import stratified_verdict


def _cell(config, dataset, family, fraction, delta, confounded):
    return {
        "config": config,
        "dataset": dataset,
        "family": family,
        "fraction": fraction,
        "delta_auc": delta,
        "confounded": confounded,
    }


def test_all_confounded_yields_quarantine() -> None:
    cells = [
        _cell("full_default", "Pima", "hard", 0.25, 0.18, True),
        _cell("spectral_only", "glass", "hard", 0.25, 0.22, True),
    ]
    v = stratified_verdict(cells)
    assert v["any_confound_free_cell_clears_threshold"] is False
    assert "QUARANTINE" in v["verdict"]
    assert v["by_family_and_size"]["hard/low_data"]["mean_clean_delta_auc"] is None
    assert v["by_family_and_size"]["hard/low_data"]["n_confounded"] == 2


def test_clean_subthreshold_stays_quarantine() -> None:
    cells = [
        _cell("full_default", "cardio", "ceiling", 1.0, 0.0005, False),
        _cell("layernorm", "thyroid", "ceiling", 1.0, -0.001, False),
    ]
    v = stratified_verdict(cells)
    assert v["any_confound_free_cell_clears_threshold"] is False
    assert "QUARANTINE" in v["verdict"]
    assert v["by_family_and_size"]["ceiling/full_data"]["mean_clean_delta_auc"] is not None


def test_clean_above_threshold_triggers_investigate() -> None:
    cells = [
        _cell("wide_kernels", "Pima", "hard", 0.25, 0.05, False),  # clean, above noise
        _cell("full_default", "cardio", "ceiling", 1.0, 0.0, False),
    ]
    v = stratified_verdict(cells)
    assert v["any_confound_free_cell_clears_threshold"] is True
    assert "INVESTIGATE" in v["verdict"]
    assert v["best_confound_free_where"] is not None
    assert v["best_confound_free_delta"] >= 0.05


def test_confounded_cells_excluded_from_best() -> None:
    # A huge confounded delta must NOT become the best confound-free delta.
    cells = [
        _cell("spectral_only", "glass", "hard", 0.25, 0.45, True),  # fake +0.45
        _cell("full_default", "cardio", "ceiling", 1.0, 0.001, False),
    ]
    v = stratified_verdict(cells)
    assert v["best_confound_free_delta"] < 0.45
    assert v["any_confound_free_cell_clears_threshold"] is False


def test_stratification_keys() -> None:
    cells = [
        _cell("full_default", "Pima", "hard", 0.25, 0.0, False),
        _cell("full_default", "cardio", "ceiling", 1.0, 0.0, False),
    ]
    v = stratified_verdict(cells)
    assert set(v["by_family_and_size"]) == {"hard/low_data", "ceiling/full_data"}

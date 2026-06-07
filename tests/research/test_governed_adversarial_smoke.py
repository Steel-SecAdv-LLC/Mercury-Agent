from __future__ import annotations

import numpy as np

from research.adversarial.governed_attacks import (
    cubic_mahalanobis_floor,
    worst_case_over_attacks,
)


def test_adversarial_survivability_smoke_reports_worst_case_and_nes() -> None:
    rng = np.random.default_rng(5)
    normal = rng.normal(0.0, 0.3, size=(32, 3))
    shifted = rng.normal(1.4, 0.3, size=(12, 3))
    x = np.vstack([normal, shifted])
    y = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(shifted), dtype=int)])
    loc = normal.mean(axis=0)

    def score_fn(batch: np.ndarray) -> np.ndarray:
        return np.linalg.norm(batch - loc, axis=1)

    report = worst_case_over_attacks(score_fn, x, y, normal_reference=normal, eps=0.4, seed=3)

    assert set(report["per_attack"]) == {"clean", "condmean", "bpda", "nes", "transfer"}
    assert report["worst_attack"] in report["per_attack"]
    assert report["worst_case_auroc"] <= report["clean_auroc"]
    assert isinstance(report["gradient_masking_flag"], bool)


def test_moment_floor_probe_smoke_reports_floor_auc() -> None:
    rng = np.random.default_rng(9)
    normal = rng.normal(0.0, 1.0, size=(24, 4))
    axis_spikes = normal[:8].copy()
    axis_spikes[:, 0] += 3.0
    x = np.vstack([normal, axis_spikes])
    y = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(axis_spikes), dtype=int)])

    report = cubic_mahalanobis_floor(x, y)

    assert 0.0 <= report["mahalanobis_floor_auc"] <= 1.0
    assert 0.0 <= report["sparse_axis_auc"] <= 1.0
    assert np.isfinite(report["mean_gap"])

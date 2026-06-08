from __future__ import annotations

import numpy as np

from research.adversarial.governed_attacks import (
    cubic_moment_escape,
    floor_curve,
    gaussian_floor_score,
    most_informative_channels,
    worst_case_over_attacks,
)


def _blobs(seed: int, d: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 0.3, size=(48, d))
    shifted = rng.normal(1.4, 0.3, size=(16, d))
    x = np.vstack([normal, shifted])
    y = np.concatenate([np.zeros(len(normal), int), np.ones(len(shifted), int)])
    return x, y, normal


def test_worst_case_controlled_budget_and_masking_flag_logic() -> None:
    x, y, normal = _blobs(3)
    loc = normal.mean(axis=0)

    def score_fn(batch: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.linalg.norm(batch - loc, axis=1)
        return out

    controlled = most_informative_channels(x, y, x.shape[1] // 2)
    report = worst_case_over_attacks(
        score_fn, x, y, normal_reference=normal, controlled=controlled, eps=0.4, seed=3
    )

    assert set(report["per_attack"]) == {"condmean", "bpda", "nes", "transfer"}
    assert report["worst_case_auroc"] <= report["clean_auroc"] + 1e-9
    # win_counts is a true per-row tally over the four attacks.
    assert sum(report["win_counts"].values()) == len(x)
    # Masking flag fires iff gradient-free NES beats gradient-based BPDA.
    assert report["gradient_masking_flag"] == (
        report["per_attack"]["nes"]["auroc"] < report["per_attack"]["bpda"]["auroc"]
    )


def test_masking_flag_independent_of_global_worst() -> None:
    """A quantized score masks BPDA's finite-diff gradient; NES still evades."""
    x, y, normal = _blobs(7)
    loc = normal.mean(axis=0)

    def masked_score(batch: np.ndarray) -> np.ndarray:
        # Coarsely quantized -> local finite differences see zero gradient.
        out: np.ndarray = np.round(np.linalg.norm(batch - loc, axis=1) * 2.0) / 2.0
        return out

    controlled = most_informative_channels(x, y, x.shape[1] // 2)
    report = worst_case_over_attacks(
        masked_score, x, y, normal_reference=normal, controlled=controlled, eps=0.6, seed=1
    )
    nes = report["per_attack"]["nes"]["auroc"]
    bpda = report["per_attack"]["bpda"]["auroc"]
    assert report["gradient_masking_flag"] == bool(nes < bpda)


def test_floor_curve_starts_clean_and_reports_each_budget() -> None:
    x, y, normal = _blobs(5, d=8)
    loc = normal.mean(axis=0)

    def score_fn(batch: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.linalg.norm(batch - loc, axis=1)
        return out

    curve = floor_curve(score_fn, x, y, normal_reference=normal, eps=0.5, seed=0)
    assert curve[0]["m"] == 0
    # m=0 is the unperturbed clean AUROC.
    clean = float(np.mean([(score_fn(x)[y == 1][:, None] > score_fn(x)[y == 0][None, :]).mean()]))
    assert abs(curve[0]["worst_case_auroc"] - clean) < 1e-6
    # Every later budget is reported and never exceeds clean.
    for pt in curve[1:]:
        assert pt["worst_case_auroc"] <= curve[0]["worst_case_auroc"] + 1e-9


def test_cubic_moment_vanishes_on_gaussian_control() -> None:
    """Gaussian (mean/cov-shift) anomaly: cubic detector ~= the floor."""
    rng = np.random.default_rng(0)
    d, n, na = 6, 600, 150
    normal = rng.normal(0.0, 1.0, size=(n, d))
    anom = rng.normal(0.0, 1.0, size=(na, d))
    anom[:, : d // 2] += 1.5  # pure mean shift -> Gaussian
    x = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(n, int), np.ones(na, int)])
    r = cubic_moment_escape(x, y)
    # No 3rd-moment structure to exploit: the cubic detector must not beat floor.
    assert r["escape"] <= 0.03


def test_cubic_moment_escapes_on_skewed_anomaly() -> None:
    """On-manifold skew anomaly (matches mean/var): cubic beats the floor."""
    rng = np.random.default_rng(1)
    d, n, na, m = 6, 800, 220, 3
    normal = rng.normal(0.0, 1.0, size=(n, d))
    anom = rng.normal(0.0, 1.0, size=(na, d))
    e = rng.exponential(1.0, size=(na, m))
    anom[:, :m] = (e - e.mean(0)) / e.std(0)  # mean 0, var 1, positive skew
    x = np.vstack([normal, anom])
    y = np.concatenate([np.zeros(n, int), np.ones(na, int)])
    r = cubic_moment_escape(x, y)
    # The Gaussian floor is near chance (mean/cov matched); cubic escapes it.
    assert r["floor_auc"] < 0.6
    assert r["cubic_auc"] > r["floor_auc"]
    # And the floor detector really is the plain Mahalanobis one.
    assert np.all(np.isfinite(gaussian_floor_score(x, normal)))

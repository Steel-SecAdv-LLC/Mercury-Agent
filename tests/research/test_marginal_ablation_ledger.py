# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Marginal-ablation ledger: schema integrity, math correctness, and regression.

Two layers:

* **schema** -- the committed ledger file is well-formed and the
  measure_marginal_ablation module produces records matching the schema.
* **math** -- the lift computation is correct on a synthetic event where the
  ground truth is known: a perfectly informative component must produce a
  large positive marginal lift; a pure-noise component must produce
  approximately zero lift.

The ledger's role is the fitness function the autonomous self-improvement
loop climbs (Phase 3+). Locking the math here means Phase 2's regression
guard sees a trustworthy signal from day one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from research.governed_fusion import measure_marginal_ablation as mma
from research.governed_fusion.measure_marginal_ablation import (
    _COMPONENTS,
    _load_external_label_scores,
    compute_marginal_lift,
    measure,
)
from research.governed_fusion.score_cache import EventScores
from research.governed_fusion.suite import EventData

if TYPE_CHECKING:
    import pytest

_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / "research" / "governed_fusion" / "ablation_ledger.json"
)


def _make_event(
    domain: str,
    event_id: str,
    *,
    n: int = 400,
    pos_frac: float = 0.20,
    informative_components: tuple[int, ...] = (0, 1, 2),
    noise: float = 0.5,
    seed: int = 42,
) -> EventScores:
    """Synthetic event with controllable per-component informativeness."""
    rng = np.random.default_rng(seed)
    n_pos = int(n * pos_frac)
    y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n - n_pos, dtype=int)])
    rng.shuffle(y)
    cols = []
    for j in range(3):
        if j in informative_components:
            # mean shifted by class -- separable signal
            x = rng.normal(loc=y * 2.0, scale=noise)
        else:
            # pure noise
            x = rng.normal(loc=0.0, scale=1.0, size=n)
        cols.append(x)
    return EventScores(
        domain=domain,
        event_id=event_id,
        y=y,
        combined=np.mean(cols, axis=0),
        verdict=(np.mean(cols, axis=0) > 0).astype(int),
        threshold=0.0,
        resonance=cols[0],
        kinematic=cols[1],
        info_geo=cols[2],
        ig_mean=np.zeros(3),
        ig_cov_inv=np.eye(3),
    )


def test_committed_ledger_is_well_formed() -> None:
    with _LEDGER_PATH.open() as fh:
        ledger = json.load(fh)
    assert ledger["schema_version"] == 1
    assert ledger["ledger"] == "governed-fusion marginal ablation"
    assert ledger["components"] == list(_COMPONENTS)
    assert ledger["transparent_fitness_bucket"] == "external_label"
    assert isinstance(ledger["runs"], list)


def test_compute_marginal_lift_recovers_informative_components() -> None:
    """All three components weakly informative -> ablating any costs AUROC.

    A high noise floor ensures the 2-component fusion doesn't already
    saturate at AUROC=1.0; otherwise leave-one-out would show zero lift not
    because the components are uninformative but because their information is
    redundant under saturating signal-to-noise.
    """
    events = [
        _make_event("network_security", "syn_a", noise=3.0, seed=11),
        _make_event("network_security", "syn_b", noise=3.0, seed=22),
    ]
    out = compute_marginal_lift(events)
    full = out["full"]
    assert full["n_events"] == 2
    assert full["auroc"] > 0.70, f"informative fusion AUROC too low: {full}"
    assert (
        full["auroc"] < 0.999
    ), f"informative fusion AUROC saturated; the LOO test would be vacuous: {full}"
    for name in _COMPONENTS:
        d = out["leave_one_out"][name]["delta"]
        assert d["auroc"] > 0.0, f"{name}: expected positive marginal lift, got {d}"


def test_compute_marginal_lift_flags_a_noise_component() -> None:
    """A noise-only component must show a smaller marginal lift than informative.

    The qualitative discipline -- info_geo is uninformative here -- must show
    in the lift ranking: ablating either informative component costs more
    AUROC than ablating the noise one.
    """
    events = [
        _make_event(
            "network_security",
            "syn_a",
            informative_components=(0, 1),  # info_geo is noise
            noise=3.0,
            seed=11,
        ),
        _make_event(
            "network_security",
            "syn_b",
            informative_components=(0, 1),
            noise=3.0,
            seed=22,
        ),
    ]
    out = compute_marginal_lift(events)
    d_res = out["leave_one_out"]["resonance"]["delta"]["auroc"]
    d_kin = out["leave_one_out"]["kinematic"]["delta"]["auroc"]
    d_inf = out["leave_one_out"]["info_geo"]["delta"]["auroc"]
    # Both informative components must rank above the noise one.
    assert d_res > d_inf, (d_res, d_kin, d_inf)
    assert d_kin > d_inf, (d_res, d_kin, d_inf)


def test_measure_handles_missing_cache(tmp_path: Path) -> None:
    """Without a score cache, the script must produce an informational record
    (not crash) so the ledger keeps a chronological account of reachability."""
    rec = measure(cache_dir=str(tmp_path / "nonexistent"))
    assert rec["status"] == "needs_cache"
    assert rec["external_label_events"] == 0
    assert rec["full"] is None
    assert rec["leave_one_out"] is None
    assert rec["components"] == list(_COMPONENTS)


def test_external_label_score_loader_uses_requested_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI's --cache-dir must control the score-cache path it reads."""
    requested: list[str] = []
    event = EventData(
        domain="network_security",
        event_id="nsl_kdd",
        X=np.zeros((8, 3), dtype=np.float64),
        y=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
    )

    def fake_event_scores(
        ev: EventData,
        *,
        cap: int = 6000,
        seed: int = 42,
        cache_dir: str,
    ) -> EventScores:
        requested.append(cache_dir)
        return _make_event(ev.domain, ev.event_id, seed=seed, n=cap // 15)

    monkeypatch.setattr(mma, "build_suite", lambda kind="real": [event])
    monkeypatch.setattr("research.governed_fusion.score_cache.event_scores", fake_event_scores)

    scores = _load_external_label_scores(str(tmp_path))
    assert [(score.domain, score.event_id) for score in scores] == [("network_security", "nsl_kdd")]
    assert requested == [str(tmp_path)]


def test_compute_marginal_lift_schema_is_stable() -> None:
    """The schema downstream gates depend on -- lock its shape."""
    events = [_make_event("network_security", "syn_only", seed=7)]
    out = compute_marginal_lift(events)
    assert set(out) == {"full", "leave_one_out"}
    assert set(out["full"]) == {"auroc", "auprc", "f1", "n_events"}
    assert set(out["leave_one_out"]) == set(_COMPONENTS)
    for name in _COMPONENTS:
        block = out["leave_one_out"][name]
        assert set(block) == {"ablated", "delta"}
        assert set(block["ablated"]) == {"auroc", "auprc", "f1", "n_events"}
        assert set(block["delta"]) == {"auroc", "auprc", "f1"}

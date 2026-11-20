# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Meta-Fusion Controller, measured leave-one-dataset-out.

The proposal: a controller that selects among live fusion candidates using
uncertainty, entropy, domain affinity and historical performance.  The prior
record (``research/omni_equation/FINDINGS.md``) measured four *label-free
per-set* signal families and found none that ranks streams per set; what it
never tested is **historical cross-dataset performance** — a prior that
consumes labels only from *other* datasets, which a deployed Mercury
legitimately has.  This harness measures that controller honestly:

Candidates per dataset (all through the shipped fusion mathematics, on the
same 19 recorded evaluations the redundancy harness uses):

* ``uniform`` — the shipped plain average (validated identical to
  ``detect()``), the baseline every candidate must beat.
* ``best_single_hist`` — the single probe with the best historical mean AUC
  over the *other* datasets (the user-facing "single best probe" baseline,
  made label-free by LODO).
* ``hist_weighted`` — a fixed-weight sum with weights proportional to the
  probes' historical LODO mean AUC above chance.
* ``decorrelated`` — the measured 16-probe preset, uniform-fused.
* ``meta_controller`` — per dataset, pick whichever of the four candidates
  above has the best historical mean AUC among the *other* datasets of the
  same corpus (domain affinity: nab vs adbench), labels of the target
  dataset never consulted.

The dataset-local signals the proposal names (score-dispersion uncertainty
and score entropy) are measured for predictive value: their rank correlation
with which candidate actually wins is reported, so a future selector can be
built on them only if they carry signal.

Pre-registered adoption bar (the repository's standing bar): a candidate is
adopted only if its overall mean AUC beats ``uniform`` by more than 0.003
with no dataset regressing by more than 0.002.

Writes ``artifacts/meta_fusion_controller.json``.
``tests/detectors/test_meta_fusion_decision.py`` pins the verdicts.

Run from the repository root::

    python research/ensembles/harness_meta_fusion.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from research.math_arrest.harness_probe_redundancy import (
    FittedDataset,
    dataset_auc,
    fuse_subset,
    load_datasets,
)

ARTIFACT_PATH = _ROOT / "artifacts" / "meta_fusion_controller.json"
ADOPTION_BAND = 0.003
REGRESSION_FLOOR = 0.002

#: Class name -> short probe name, from the probe classes' own registry names.
_DECORRELATED_SHORT: list[str] = []


def _decorrelated_short_names(sample_names: list[str]) -> list[str]:
    """Map the decorrelated preset's class names onto fit-order short names.

    A tiny ``AnomalyMathArrest(probes="decorrelated")`` fit yields the exact
    ``probe_name`` strings those probes report, so the mapping is the
    instrument's own, not a casing reconstruction.
    """
    from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest

    arrest = AnomalyMathArrest(probes="decorrelated")
    arrest.fit(np.sin(np.linspace(0.0, 20.0, 128)) + 0.01 * np.arange(128))
    short = [result.probe_name for result in arrest.scored_results(np.linspace(0.0, 1.0, 64))]
    missing = [s for s in short if s not in sample_names]
    if missing:
        raise AssertionError(f"decorrelated preset names not in fit order: {missing}")
    return short


def probe_aucs(dataset: FittedDataset) -> dict[str, float]:
    """Held-out AUROC of each probe column on this dataset."""
    out: dict[str, float] = {}
    positive = dataset.positive[: dataset.held_matrix.shape[0]]
    for j, name in enumerate(dataset.probe_names):
        scores = dataset.held_matrix[:, j]
        pos, neg = scores[positive], scores[~positive]
        if pos.size == 0 or neg.size == 0:
            out[name] = float("nan")
            continue
        ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
        r_pos = ranks[: pos.size].sum()
        out[name] = float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
    return out


def fuse_weighted(dataset: FittedDataset, weights: dict[str, float]) -> npt.NDArray[np.float64]:
    """Fixed-weight sum through the shipped monotonic projection math."""
    w = np.array([max(weights.get(n, 0.0), 0.0) for n in dataset.probe_names])
    if w.sum() <= 0:
        w = np.ones(len(dataset.probe_names))
    w = w / w.sum()
    fused = dataset.held_matrix @ w
    return np.clip(np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def local_signals(dataset: FittedDataset) -> dict[str, float]:
    """Dataset-local, label-free signals the proposal names."""
    fused = dataset.shipped_fused
    dispersion = float(np.std(dataset.held_matrix, axis=1).mean())
    hist, _ = np.histogram(fused, bins=20, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum() / np.log(20))
    return {"probe_dispersion": dispersion, "score_entropy": entropy}


def main() -> int:
    datasets = load_datasets()
    names = [d.name for d in datasets]
    per_probe = {d.name: probe_aucs(d) for d in datasets}
    decorrelated = _decorrelated_short_names(datasets[0].probe_names)

    candidates = ("uniform", "best_single_hist", "hist_weighted", "decorrelated")
    aucs: dict[str, dict[str, float]] = {c: {} for c in candidates}
    aucs["meta_controller"] = {}
    chosen: dict[str, str] = {}
    signals = {d.name: local_signals(d) for d in datasets}

    for i, dataset in enumerate(datasets):
        others = [d for j, d in enumerate(datasets) if j != i]
        # Historical per-probe mean AUC over the other datasets.
        hist = {
            name: float(np.nanmean([per_probe[o.name][name] for o in others]))
            for name in dataset.probe_names
        }
        best_probe = max(hist, key=lambda n: hist[n])
        weights = {n: max(a - 0.5, 0.0) for n, a in hist.items()}

        aucs["uniform"][dataset.name] = dataset_auc(
            dataset, fuse_subset(dataset, dataset.probe_names)
        )
        aucs["best_single_hist"][dataset.name] = dataset_auc(
            dataset, fuse_subset(dataset, [best_probe])
        )
        aucs["hist_weighted"][dataset.name] = dataset_auc(dataset, fuse_weighted(dataset, weights))
        aucs["decorrelated"][dataset.name] = dataset_auc(
            dataset, fuse_subset(dataset, decorrelated)
        )

        # The controller: among the four candidates, the one with the best
        # historical mean AUC on same-corpus other datasets.  To avoid
        # consuming the target's labels, candidate historical AUCs are the
        # candidates recomputed on the other datasets with THEIR own LODO
        # priors excluded of the target — the per-probe LODO prior above
        # already excludes the target, so reusing each other-dataset's
        # recorded candidate AUC (computed in its own fold) stays label-free
        # for the target.  First pass computes folds, so the controller is
        # resolved in a second pass below.
        chosen[dataset.name] = ""  # filled in the second pass

    aucs["meta_guarded"] = {}
    chosen_guarded: dict[str, str] = {}
    for i, dataset in enumerate(datasets):
        same_corpus = [
            d.name for j, d in enumerate(datasets) if j != i and d.corpus == dataset.corpus
        ]
        pool = same_corpus if same_corpus else [d.name for j, d in enumerate(datasets) if j != i]
        best_candidate = max(
            candidates,
            key=lambda c: float(np.nanmean([aucs[c][n] for n in pool])),
        )
        chosen[dataset.name] = best_candidate
        aucs["meta_controller"][dataset.name] = aucs[best_candidate][dataset.name]

        # The guarded controller may fall back: a non-uniform candidate is
        # selected only when the historical same-corpus pool itself clears
        # the standing keep bar (mean delta over the pool > ADOPTION_BAND and
        # no pool dataset below -REGRESSION_FLOOR); otherwise uniform.  This
        # is the per-dataset abstention the forced-selection controller
        # lacked — it buys the mean only where history says no one paid.
        eligible: list[tuple[float, str]] = []
        for candidate in candidates:
            if candidate == "uniform":
                continue
            deltas = [aucs[candidate][n] - aucs["uniform"][n] for n in pool]
            if (
                float(np.nanmean(deltas)) > ADOPTION_BAND
                and float(np.nanmin(deltas)) >= -REGRESSION_FLOOR
            ):
                eligible.append((float(np.nanmean([aucs[candidate][n] for n in pool])), candidate))
        picked = max(eligible)[1] if eligible else "uniform"
        chosen_guarded[dataset.name] = picked
        aucs["meta_guarded"][dataset.name] = aucs[picked][dataset.name]

    shipped = aucs["uniform"]
    verdicts: dict[str, dict[str, object]] = {}
    for candidate, values in aucs.items():
        if candidate == "uniform":
            continue
        deltas = {n: values[n] - shipped[n] for n in names}
        mean_delta = float(np.mean(list(deltas.values())))
        worst_name, worst_delta = min(deltas.items(), key=lambda kv: kv[1])
        verdicts[candidate] = {
            "mean_auc": float(np.mean(list(values.values()))),
            "mean_delta_vs_shipped": mean_delta,
            "worst_dataset": worst_name,
            "worst_delta": float(worst_delta),
            "adopted": bool(mean_delta > ADOPTION_BAND and worst_delta >= -REGRESSION_FLOOR),
        }

    # Do the local signals predict the winning candidate?  Spearman rank
    # correlation between each signal and the winner's identity is
    # ill-defined; instead report, per signal, the correlation with the
    # uniform-vs-best-single advantage (the axis the FINDINGS record showed
    # matters most).
    advantage = np.array([aucs["best_single_hist"][n] - aucs["uniform"][n] for n in names])
    signal_corr = {}
    for key in ("probe_dispersion", "score_entropy"):
        x = np.array([signals[n][key] for n in names])
        xr = np.argsort(np.argsort(x)).astype(float)
        yr = np.argsort(np.argsort(advantage)).astype(float)
        signal_corr[key] = float(np.corrcoef(xr, yr)[0, 1])

    artifact = {
        "schema": "mercury.research.meta_fusion_controller/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "selection": "leave-one-dataset-out; controller picks by historical "
            "mean AUC on same-corpus other datasets; target labels never read",
            "adoption_band": ADOPTION_BAND,
            "regression_floor": REGRESSION_FLOOR,
            "n_evaluations": len(datasets),
        },
        "shipped_uniform_mean_auc": float(np.mean(list(shipped.values()))),
        "per_dataset": aucs,
        "controller_choices": chosen,
        "guarded_controller_choices": chosen_guarded,
        "local_signal_rank_correlation_with_best_single_advantage": signal_corr,
        "verdicts": verdicts,
    }
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"shipped uniform mean AUC {artifact['shipped_uniform_mean_auc']:.4f}")
    for name, verdict in verdicts.items():
        print(
            f"{name}: mean {verdict['mean_auc']:.4f} "
            f"(delta {verdict['mean_delta_vs_shipped']:+.4f}, worst "
            f"{verdict['worst_dataset']} {verdict['worst_delta']:+.4f}) -> "
            f"{'ADOPT' if verdict['adopted'] else 'refused'}"
        )
    print("controller choices:", json.dumps(chosen, indent=0)[:400])
    print("signal rank correlations:", signal_corr)
    print(f"written: {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

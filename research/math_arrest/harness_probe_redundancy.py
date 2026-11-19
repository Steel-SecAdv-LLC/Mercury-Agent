# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which of the 21 probes are redundant, and what does removing them cost?

The decorrelator already *measures* redundancy per fit and down-weights
cluster duplicates, but nothing persisted the correlation structure and no
measurement existed of what dropping the redundant members would do to the
ensemble's own mean AUC and mean rank. This harness closes both:

* For every recorded dataset (four NAB files, five ADBench tables x three
  seeds) it fits the full 21-probe ensemble once, records the fit-time
  pairwise correlation matrix (the matrix the shipped decorrelator calibrates
  on), each probe's held-out AUROC, and the shipped fused AUROC.
* Subsets are then evaluated through the *shipped* fusion path — uniform
  weighting, decorrelator recalibrated on the subset's fit-time columns —
  reconstructed exactly from the cached matrices (validated bit-identical to
  ``AnomalyMathArrest.detect`` for the full set on every dataset).
* A correlation-threshold sweep removes the non-dominant members of every
  cluster at each threshold and reports the reduced ensemble's mean AUC and
  its mean rank against the full ensemble, picking the most aggressive
  threshold that does not degrade either.
* Leave-one-out deltas for all 21 probes are recorded alongside.

Everything is written to ``artifacts/math_arrest_probe_correlation.json``;
``tests/detectors/test_probe_redundancy.py`` pins the selection and the
non-degradation bound.

Run from the repository root::

    python research/math_arrest/harness_probe_redundancy.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
from omni_mercury_engine.detectors.math_arrest.base_probe import ProbeResult
from omni_mercury_engine.detectors.math_arrest.domain_affinity import get_affinity_order
from omni_mercury_engine.detectors.math_arrest.fusion import (
    CorrelationAwareDecorrelator,
    PhiWeightedFusion,
)
from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, nab_corpus, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "math_arrest_probe_correlation.json"
SEEDS = (0, 1, 2)
#: Most aggressive first: the sweep keeps the lowest threshold that does not
#: degrade mean AUC or mean rank against the full ensemble.
THRESHOLDS = (0.80, 0.85, 0.90, 0.95, 0.99)


@dataclass
class FittedDataset:
    """One dataset's cached full-ensemble state: everything a subset needs."""

    name: str
    corpus: str  # "nab" | "adbench"
    probe_names: list[str]  # short names, fit order
    fit_matrix: npt.NDArray[np.float64]  # (n_fit, 21) fit-time probe scores
    held_matrix: npt.NDArray[np.float64]  # (n_held, 21) held-out probe scores
    fit_qualities: dict[str, float]
    positive: npt.NDArray[np.bool_]  # held-out labels
    shipped_fused: npt.NDArray[np.float64]  # detect() on held-out, full set


def _columns(
    detector: AnomalyMathArrest, data: npt.NDArray[np.float64]
) -> tuple[list[str], npt.NDArray[np.float64], dict[str, float]]:
    """Per-probe score columns (fit order), plus trajectory fit qualities."""
    names: list[str] = []
    cols: list[npt.NDArray[np.float64]] = []
    qualities: dict[str, float] = {}
    for probe in detector._probes:
        if not probe.is_fitted:
            continue
        result = probe.deviation_score(data)
        names.append(result.probe_name)
        cols.append(np.asarray(result.deviation_scores, dtype=np.float64))
        qualities[result.probe_name] = result.trajectory_fit_quality
    width = min(len(c) for c in cols)
    return names, np.column_stack([c[:width] for c in cols]), qualities


def _fit_one(
    name: str,
    corpus: str,
    train: npt.NDArray[np.float64],
    held: npt.NDArray[np.float64],
    positive: npt.NDArray[np.bool_],
) -> FittedDataset:
    detector = AnomalyMathArrest()
    detector.fit(train)
    probe_names, fit_matrix, fit_qualities = _columns(detector, train)
    _, held_matrix, _ = _columns(detector, held)
    shipped = np.asarray(detector.detect(held), dtype=np.float64).ravel()
    return FittedDataset(
        name=name,
        corpus=corpus,
        probe_names=probe_names,
        fit_matrix=fit_matrix,
        held_matrix=held_matrix,
        fit_qualities=fit_qualities,
        positive=positive,
        shipped_fused=shipped,
    )


def load_datasets() -> list[FittedDataset]:
    """The 19 fitted evaluations: 4 NAB files + 5 ADBench tables x 3 seeds."""
    fitted: list[FittedDataset] = []
    for series in nab_corpus():
        split = series.n_probation
        fitted.append(
            _fit_one(
                series.name,
                "nab",
                series.values[:split],
                series.values[split:],
                series.positive[split:],
            )
        )
    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            fitted.append(
                _fit_one(
                    f"{table}/seed{seed}",
                    "adbench",
                    (features[train] - mean) / spread,
                    (features[test] - mean) / spread,
                    labels[test] == 1,
                )
            )
    return fitted


def fuse_subset(dataset: FittedDataset, keep: list[str]) -> npt.NDArray[np.float64]:
    """The shipped fusion of a probe subset, from the cached matrices.

    Mirrors ``AnomalyMathArrest`` exactly: uniform weighting, the
    decorrelator recalibrated on the subset's fit-time columns, affinity
    order passed (a no-op under uniform weights). For ``keep`` equal to the
    full probe list this reproduces ``detect`` bit-identically, which
    :func:`main` asserts on every dataset.
    """
    indices = [dataset.probe_names.index(name) for name in keep]
    decorrelator = CorrelationAwareDecorrelator()
    decorrelator.calibrate(
        dataset.fit_matrix[:, indices],
        keep,
        {name: dataset.fit_qualities[name] for name in keep},
    )
    results = [
        ProbeResult(
            probe_name=name,
            deviation_scores=dataset.held_matrix[:, idx],
            confidence=dataset.fit_qualities[name],
            trajectory_fit_quality=dataset.fit_qualities[name],
            anomaly_geometry="",
        )
        for name, idx in zip(keep, indices, strict=True)
    ]
    fusion = PhiWeightedFusion(n_probes=len(keep), weighting="uniform")
    order = get_affinity_order("default", keep)
    return fusion.fuse(results, affinity_order=order, decorrelator=decorrelator)


def dataset_auc(dataset: FittedDataset, fused: npt.NDArray[np.float64]) -> float:
    positive = dataset.positive[: len(fused)]
    return auroc(fused[positive], fused[~positive])


def per_dataset_aucs(datasets: list[FittedDataset], keep: list[str]) -> dict[str, float]:
    return {d.name: dataset_auc(d, fuse_subset(d, keep)) for d in datasets}


def corpus_means(datasets: list[FittedDataset], aucs: dict[str, float]) -> dict[str, float]:
    """Mean AUC per corpus, ADBench seeds averaged per table first."""
    nab = [aucs[d.name] for d in datasets if d.corpus == "nab"]
    tables: dict[str, list[float]] = {}
    for d in datasets:
        if d.corpus == "adbench":
            tables.setdefault(d.name.split("/")[0], []).append(aucs[d.name])
    adb = [float(np.mean(v)) for v in tables.values()]
    return {
        "nab": float(np.mean(nab)),
        "adbench": float(np.mean(adb)),
        "overall": float(np.mean(nab + adb)),
    }


def mean_rank(per_config_aucs: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average-rank-on-ties of each configuration across the evaluations.

    The same ranking convention as ``benchmarks/competitive_benchmark.py``:
    rank 1 is best, ties share the average of the ranks they span.
    """
    configs = sorted(per_config_aucs)
    names = sorted(next(iter(per_config_aucs.values())))
    totals = dict.fromkeys(configs, 0.0)
    for name in names:
        values = np.array([per_config_aucs[c][name] for c in configs])
        order = np.argsort(-values, kind="stable")
        ranks = np.empty(len(configs))
        position = 0
        while position < len(configs):
            stop = position
            while stop + 1 < len(configs) and values[order[stop + 1]] == values[order[position]]:
                stop += 1
            ranks[order[position : stop + 1]] = (position + stop) / 2.0 + 1.0
            position = stop + 1
        for i, config in enumerate(configs):
            totals[config] += float(ranks[i])
    return {c: totals[c] / len(names) for c in configs}


def correlation_matrix(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Pairwise Pearson correlation with zero-variance columns as zero rows."""
    stds = matrix.std(axis=0)
    corr = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    valid = np.where(stds >= 1e-10)[0]
    if valid.size >= 2:
        sub = np.corrcoef(matrix[:, valid].T)
        sub = np.nan_to_num(np.atleast_2d(sub), nan=0.0)
        corr[np.ix_(valid, valid)] = sub
    np.fill_diagonal(corr, 1.0)
    return corr


def clusters_at(
    mean_abs_corr: npt.NDArray[np.float64], names: list[str], threshold: float
) -> list[list[str]]:
    """Connected components of |r| >= threshold, as the decorrelator builds them."""
    n = len(names)
    adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if abs(mean_abs_corr[i, j]) >= threshold:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    out: list[list[str]] = []
    for start in range(n):
        if start in seen or not adjacency[start]:
            continue
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node] - seen)
        if len(component) >= 2:
            out.append([names[i] for i in sorted(component)])
    return out


def removal_for_threshold(clusters: list[list[str]], probe_auc_mean: dict[str, float]) -> list[str]:
    """Non-dominant members of every cluster: keep the highest mean held-out AUC."""
    removed: list[str] = []
    for cluster in clusters:
        keep_name = max(cluster, key=lambda n: probe_auc_mean[n])
        removed.extend(sorted(set(cluster) - {keep_name}))
    return sorted(set(removed))


def _git_commit() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    try:
        out = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def main() -> int:
    datasets = load_datasets()
    names = datasets[0].probe_names
    for d in datasets:
        if d.probe_names != names:
            raise RuntimeError(f"{d.name}: probe set differs — {d.probe_names}")
        rebuilt = fuse_subset(d, names)
        if not np.array_equal(rebuilt, d.shipped_fused[: len(rebuilt)]):
            raise RuntimeError(f"{d.name}: cached-path fusion does not reproduce detect()")

    # Per-dataset correlation structure and per-probe held-out AUC.
    per_dataset: dict[str, dict[str, object]] = {}
    corr_stack: list[npt.NDArray[np.float64]] = []
    probe_auc: dict[str, list[float]] = {n: [] for n in names}
    for d in datasets:
        corr = correlation_matrix(d.fit_matrix)
        corr_stack.append(np.abs(corr))
        positive = d.positive[: d.held_matrix.shape[0]]
        aucs = {}
        for i, n in enumerate(names):
            column = d.held_matrix[:, i]
            value = (
                auroc(column[positive], column[~positive])
                if 0 < positive.sum() < len(positive)
                else 0.5
            )
            aucs[n] = round(value, 4)
            probe_auc[n].append(value)
        per_dataset[d.name] = {
            "corpus": d.corpus,
            "n_fit": int(d.fit_matrix.shape[0]),
            "n_held": int(d.held_matrix.shape[0]),
            "correlation": np.round(corr, 4).tolist(),
            "per_probe_auc": aucs,
        }
    mean_abs_corr = np.mean(corr_stack, axis=0)
    probe_auc_mean = {n: float(np.mean(v)) for n, v in probe_auc.items()}

    # Full-ensemble baseline.
    full_aucs = per_dataset_aucs(datasets, names)
    full_means = corpus_means(datasets, full_aucs)

    # Leave-one-out deltas.
    loo: dict[str, dict[str, float]] = {}
    for n in names:
        keep = [m for m in names if m != n]
        means = corpus_means(datasets, per_dataset_aucs(datasets, keep))
        loo[n] = {
            "delta_overall": round(means["overall"] - full_means["overall"], 4),
            "delta_nab": round(means["nab"] - full_means["nab"], 4),
            "delta_adbench": round(means["adbench"] - full_means["adbench"], 4),
        }

    # Threshold sweep: remove non-dominant cluster members, most aggressive first.
    sweep: list[dict[str, object]] = []
    selected: dict[str, object] | None = None
    for threshold in THRESHOLDS:
        cluster_list = clusters_at(mean_abs_corr, names, threshold)
        removed = removal_for_threshold(cluster_list, probe_auc_mean)
        keep = [n for n in names if n not in removed]
        if not removed:
            entry: dict[str, object] = {
                "threshold": threshold,
                "clusters": cluster_list,
                "removed": [],
                "kept": keep,
                "note": "no cluster at this threshold",
            }
            sweep.append(entry)
            continue
        reduced_aucs = per_dataset_aucs(datasets, keep)
        reduced_means = corpus_means(datasets, reduced_aucs)
        ranks = mean_rank({"full_21": full_aucs, "reduced": reduced_aucs})
        non_degrading = (
            reduced_means["overall"] >= full_means["overall"]
            and ranks["reduced"] <= ranks["full_21"]
        )
        entry = {
            "threshold": threshold,
            "clusters": cluster_list,
            "removed": removed,
            "kept": keep,
            "mean_auc": {k: round(v, 4) for k, v in reduced_means.items()},
            "mean_rank": {k: round(v, 4) for k, v in ranks.items()},
            "non_degrading": non_degrading,
        }
        sweep.append(entry)
        if non_degrading and selected is None:
            selected = entry

    artifact = {
        "schema": "mercury.research.math_arrest_probe_correlation/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "probe_names": names,
        "datasets": per_dataset,
        "mean_abs_correlation": np.round(mean_abs_corr, 4).tolist(),
        "per_probe_mean_auc": {n: round(v, 4) for n, v in probe_auc_mean.items()},
        "full_ensemble": {
            "mean_auc": {k: round(v, 4) for k, v in full_means.items()},
            "per_dataset_auc": {k: round(v, 4) for k, v in full_aucs.items()},
        },
        "leave_one_out": loo,
        "threshold_sweep": sweep,
        "selected": selected,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")

    print(f"full 21-probe mean AUC: {full_means}")
    print("\nleave-one-out delta_overall (sorted):")
    for n, d in sorted(loo.items(), key=lambda kv: -kv[1]["delta_overall"]):
        print(
            f"  {n:28s} {d['delta_overall']:+.4f} (nab {d['delta_nab']:+.4f}, adb {d['delta_adbench']:+.4f})"
        )
    print("\nthreshold sweep:")
    for entry in sweep:
        if entry.get("removed"):
            print(
                f"  |r|>={entry['threshold']}: remove {len(entry['removed'])} -> "
                f"mean AUC {entry['mean_auc']}, rank {entry['mean_rank']}, "
                f"non_degrading={entry['non_degrading']}"
            )
        else:
            print(f"  |r|>={entry['threshold']}: no clusters")
    if selected:
        print(f"\nselected: |r|>={selected['threshold']} removes {selected['removed']}")
    else:
        print("\nselected: none (every removal degrades mean AUC or mean rank)")
    print(f"wrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What do Mercury's elaborate fusion mechanisms buy over its own simple ones?

Three families of Mercury's own composition mechanisms are run on the five
recorded ADBench tables (genuine labels, three seeds, the same stratified
splits every recorded measurement uses), against the simplest compositions of
the same internal streams:

* **Trained fusion** (``OmniMercuryEngine.fit_fusion`` — the multi-head
  attention HybridFusionLayer): neural-only (``symbolic_weight=0.0``), the
  adaptive label-scarcity schedule, and fixed symbolic weights 0.05 / 0.1 /
  0.25 / 0.5. The fixed-weight sweep is the measured answer to "what does the
  co-trained symbolic satisfaction term contribute" — enabled, disabled and
  intermediate, through Mercury's own training loop.
* **The R/K/IG detector** (``MercuryAnomalyDetector``): the shipped adaptive
  weights against the plain average of its three component scores, the fixed
  40/30/30 default, and the single best component (chosen on the training
  labels), all through the detector's own scoring pipeline so every condition
  shares calibration, guards and clipping.
* **The 21-probe arrest ensemble**: plain average (shipped), the
  golden-ratio ``phi_affinity`` prior (the fixed-weight sum), and the single
  best probe (chosen on the training labels).
* **Stream composition**: the FibringComposer's φ:1 mode and the 0.7/0.3
  neural-dominant blend over (trained-fusion probability, arrest plain-mean
  score), against the plain average of those two streams and each stream
  alone.

Every stream is Mercury's own; nothing external enters. Results (per-table
AUROC, mean AUROC, mean rank with competitive-benchmark tie handling) are
written to ``artifacts/fusion_mode_ablation.json`` and pinned by
``tests/benchmarks/test_fusion_mode_ablation.py``.

Run from the repository root (needs the AMA-built environment)::

    python research/ensembles/harness_fusion_mode_ablation.py
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tests.recorded_data import ADBENCH_NAMES, adbench, auroc, stratified_split

ARTIFACT_PATH = _ROOT / "artifacts" / "fusion_mode_ablation.json"
SEEDS = (0, 1, 2)
EPOCHS = 20
FIXED_LAMBDAS = (0.05, 0.1, 0.25, 0.5)


def _trained_probs(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    x_test: np.ndarray[Any, Any],
    symbolic_weight: float | str,
    seed: int,
) -> np.ndarray[Any, Any]:
    """One deterministic ``fit_fusion`` run; held-out fusion probabilities."""
    import torch

    from omni_mercury_engine.engine import OmniMercuryEngine

    torch.manual_seed(seed)
    np.random.seed(seed)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=64,
        early_stopping_patience=15,
        symbolic_weight=symbolic_weight,
    )
    return np.asarray(engine.score_fusion(x_test), dtype=np.float64).ravel()


def _detector_scores(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    x_test: np.ndarray[Any, Any],
) -> dict[str, np.ndarray[Any, Any]]:
    """R/K/IG compositions, all through the detector's own scoring pipeline."""
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    detector = MercuryAnomalyDetector()
    detector.fit(x_train)

    def scores_with(weights: np.ndarray[Any, Any] | None) -> np.ndarray[Any, Any]:
        if weights is not None:
            detector._adaptive_weights = weights.copy()
        result = detector.detect(x_test)
        return np.asarray(result["scores"], dtype=np.float64).ravel()

    adaptive = detector._adaptive_weights.copy()
    out = {"mercury_adaptive": scores_with(None)}
    out["mercury_average"] = scores_with(np.array([1 / 3, 1 / 3, 1 / 3]))
    out["mercury_fixed_403030"] = scores_with(np.array([0.4, 0.3, 0.3]))

    # Single best component, chosen on the training labels through the same
    # pipeline (one-hot weight vector per component, best train AUROC wins).
    best_name, best_auc = "", -1.0
    for idx, name in enumerate(("resonance", "kinematic", "infogeo")):
        one_hot = np.zeros(3)
        one_hot[idx] = 1.0
        detector._adaptive_weights = one_hot
        train_scores = np.asarray(detector.detect(x_train)["scores"], dtype=np.float64).ravel()
        value = auroc(train_scores[y_train == 1], train_scores[y_train == 0])
        if value > best_auc:
            best_name, best_auc = name, value
            out["mercury_best_component"] = scores_with(one_hot)
    detector._adaptive_weights = adaptive
    out["_best_component_name"] = best_name  # type: ignore[assignment]
    return out


def _arrest_scores(
    x_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    x_test: np.ndarray[Any, Any],
) -> dict[str, np.ndarray[Any, Any]]:
    """21-probe compositions: plain mean, golden-ratio prior, best single probe."""
    from omni_mercury_engine.detectors.math_arrest.arrest import AnomalyMathArrest
    from omni_mercury_engine.detectors.math_arrest.fusion import PhiWeightedFusion

    detector = AnomalyMathArrest()
    detector.fit(x_train)
    out = {"arrest_uniform": np.asarray(detector.detect(x_test), dtype=np.float64).ravel()}
    detector._fusion = PhiWeightedFusion(n_probes=len(detector._probes), weighting="phi_affinity")
    out["arrest_phi_prior"] = np.asarray(detector.detect(x_test), dtype=np.float64).ravel()

    train_columns = detector.per_probe_scores(x_train)
    test_columns = detector.per_probe_scores(x_test)
    best_name, best_auc = "", -1.0
    for name, column in train_columns.items():
        col = np.asarray(column, dtype=np.float64)
        value = auroc(col[y_train == 1], col[y_train == 0])
        if value > best_auc and name in test_columns:
            best_name, best_auc = name, value
    out["arrest_best_probe"] = np.asarray(test_columns[best_name], dtype=np.float64).ravel()
    out["_best_probe_name"] = best_name  # type: ignore[assignment]
    return out


def _composed_streams(
    neural: np.ndarray[Any, Any], symbolic: np.ndarray[Any, Any]
) -> dict[str, np.ndarray[Any, Any]]:
    """Fibring φ:1, the 0.7/0.3 blend, and the plain average of two streams."""
    from omni_mercury_engine.core.fibring_fusion import FibringComposer

    composer = FibringComposer(domain="default")
    fused = np.empty(len(neural), dtype=np.float64)
    for i, (n, s) in enumerate(zip(neural, symbolic, strict=True)):
        weights = composer.compose(float(n), float(s))
        fused[i] = weights.neural_weight * float(n) + weights.symbolic_weight * float(s)
    return {
        "fibring_phi": fused,
        "blend_070_030": 0.7 * neural + 0.3 * symbolic,
        "stream_average": 0.5 * (neural + symbolic),
    }


def mean_rank(per_config: dict[str, dict[str, float]]) -> dict[str, float]:
    """Average-rank-on-ties across tables (competitive-benchmark convention)."""
    configs = sorted(per_config)
    tables = sorted(next(iter(per_config.values())))
    totals = dict.fromkeys(configs, 0.0)
    for table in tables:
        values = np.array([per_config[c][table] for c in configs])
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
    return {c: round(totals[c] / len(tables), 4) for c in configs}


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


def run() -> dict[str, Any]:
    """The full ablation; returns the artifact payload."""
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    per_seed: dict[str, dict[str, list[float]]] = {}
    chosen: dict[str, list[str]] = {"best_component": [], "best_probe": []}

    def record(config: str, table: str, value: float) -> None:
        per_seed.setdefault(config, {}).setdefault(table, []).append(value)

    for table in ADBENCH_NAMES:
        features, labels = adbench(table)
        for seed in SEEDS:
            train, test = stratified_split(labels, seed)
            mean = features[train].mean(axis=0)
            spread = features[train].std(axis=0)
            spread[spread < 1e-8] = 1.0
            x_train = (features[train] - mean) / spread
            x_test = (features[test] - mean) / spread
            y_train, y_test = labels[train], labels[test]
            held = y_test == 1

            def score(config: str, values: np.ndarray[Any, Any]) -> None:
                record(config, table, auroc(values[held], values[~held]))

            # Trained fusion family (item: attention path + symbolic sweep).
            neural = _trained_probs(x_train, y_train, x_test, 0.0, seed)
            score("trained_neural_only", neural)
            score(
                "trained_symbolic_adaptive",
                _trained_probs(x_train, y_train, x_test, "adaptive", seed),
            )
            for lam in FIXED_LAMBDAS:
                score(
                    f"trained_symbolic_{lam}",
                    _trained_probs(x_train, y_train, x_test, lam, seed),
                )

            # R/K/IG detector family.
            detector_scores = _detector_scores(x_train, y_train, x_test)
            chosen["best_component"].append(str(detector_scores.pop("_best_component_name")))
            for config, values in detector_scores.items():
                score(config, values)

            # 21-probe arrest family.
            arrest_scores = _arrest_scores(x_train, y_train, x_test)
            chosen["best_probe"].append(str(arrest_scores.pop("_best_probe_name")))
            symbolic_stream = arrest_scores["arrest_uniform"]
            for config, values in arrest_scores.items():
                score(config, values)

            # Composed streams over (trained neural, arrest mean).
            for config, values in _composed_streams(neural, symbolic_stream).items():
                score(config, values)
            print(f"  {table}/seed{seed} done", flush=True)

    per_config: dict[str, dict[str, float]] = {
        config: {table: round(float(np.mean(v)), 4) for table, v in tables.items()}
        for config, tables in per_seed.items()
    }
    summary = {
        config: {
            "per_table_auc": tables,
            "mean_auc": round(float(np.mean(list(tables.values()))), 4),
        }
        for config, tables in per_config.items()
    }
    ranks = mean_rank(per_config)
    for config, rank in ranks.items():
        summary[config]["mean_rank"] = rank

    return {
        "schema": "mercury.research.fusion_mode_ablation/v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": _git_commit(),
        "config": {
            "tables": list(ADBENCH_NAMES),
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "fixed_lambdas": list(FIXED_LAMBDAS),
        },
        "selections": {
            "best_component_by_run": chosen["best_component"],
            "best_probe_by_run": chosen["best_probe"],
        },
        "results": summary,
    }


def main() -> int:
    artifact = run()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    rows = sorted(artifact["results"].items(), key=lambda kv: kv[1]["mean_rank"])
    print(f"\n{'config':28s} {'mean AUC':>9s} {'mean rank':>10s}")
    for config, cell in rows:
        print(f"{config:28s} {cell['mean_auc']:9.4f} {cell['mean_rank']:10.2f}")
    print(f"\nwrote {ARTIFACT_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

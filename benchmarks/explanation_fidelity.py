"""
Mercury Agent - Copyright (C) 2025 Steel Security Advisors LLC
Licensed under GNU GPL v3

Explanation-fidelity validation: are the dormant explainers faithful, and do
they recover the features a model actually uses?

`explainability.py` was orphaned and, judged by anomaly-AUC, un-revivable -- it
emits feature attributions, not anomaly scores. The right metric for an explainer
is **faithfulness**: do its top-ranked features actually drive the model's
prediction (comprehensiveness), and do the model's true informative features get
ranked highly (recovery)?

This is the second non-AUC measurement framework (after causal recovery). It is
self-contained and dependency-free -- it exercises the two components of
`explainability.py` that need no optional ``shap``/``lime`` libraries:

* ``IntegratedGradientsExplainer`` -- path-integral attributions via finite
  differences over any callable predict-fn.
* ``FaithfulnessEvaluator`` -- comprehensiveness / sufficiency / monotonicity.

A synthetic classification problem is built with a **known** informative feature
set; a real (small MLP) model is trained on it; then for each test instance the
explainer is run and scored on (a) recovery of the informative features vs
chance, and (b) faithfulness vs a random-importance baseline.

Pre-registered bar: the explainer is *validated* if mean recovery@k clearly beats
chance AND its comprehensiveness clearly beats the random-attribution baseline.

Usage::

    python -m benchmarks.explanation_fidelity --out artifacts/explanation_fidelity.json
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

N_FEATURES = 10
N_INFORMATIVE = 3
N_SAMPLES = 1500
N_EXPLAIN = 60  # test instances to explain


def _make_data(
    seed: int,
) -> tuple[
    np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], set[int]
]:
    """Synthetic binary task; features 0..N_INFORMATIVE-1 drive the label."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N_SAMPLES, N_FEATURES)).astype(np.float32)
    w = np.zeros(N_FEATURES, dtype=np.float32)
    informative = list(range(N_INFORMATIVE))
    w[informative] = rng.uniform(1.5, 3.0, N_INFORMATIVE) * rng.choice([-1.0, 1.0], N_INFORMATIVE)
    logits = x @ w
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(N_SAMPLES) < probs).astype(np.int64)
    n_tr = int(0.7 * N_SAMPLES)
    return x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:], set(informative)


def _train_predict_fn(
    x_tr: np.ndarray[Any, Any], y_tr: np.ndarray[Any, Any], seed: int
) -> Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]]:
    """Train a small MLP; return a numpy predict-fn returning P(y=1)."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(N_FEATURES, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    xt = torch.tensor(x_tr)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    model.train()
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        loss.backward()
        opt.step()
    model.eval()

    def predict_fn(arr: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        with torch.no_grad():
            logits = model(torch.tensor(np.atleast_2d(arr).astype(np.float32)))
            return torch.sigmoid(logits).numpy().reshape(-1)

    return predict_fn


def run(seed: int) -> dict[str, Any]:
    from omni_mercury_engine.cognitive.explainability import (
        Explanation,
        FaithfulnessEvaluator,
        FeatureImportance,
        IntegratedGradientsExplainer,
    )

    x_tr, y_tr, x_te, y_te, informative = _make_data(seed)
    predict_fn = _train_predict_fn(x_tr, y_tr, seed)
    explainer = IntegratedGradientsExplainer(n_steps=32)
    evaluator = FaithfulnessEvaluator()
    rng = np.random.default_rng(seed + 99)

    recovers: list[float] = []
    comp_ig: list[float] = []
    comp_rand: list[float] = []
    k = N_INFORMATIVE
    # Explain only confidently-classified instances (explanations of coin-flips
    # are meaningless); cap at N_EXPLAIN.
    probs = predict_fn(x_te)
    order = np.argsort(-np.abs(probs - 0.5))
    for idx in order[:N_EXPLAIN]:
        instance = x_te[idx]
        expl = explainer.explain(predict_fn, instance, [f"f{i}" for i in range(N_FEATURES)])
        ranked = sorted(expl.feature_importances, key=lambda fi: abs(fi.importance), reverse=True)
        top_idx = {fi.feature_index for fi in ranked[:k]}
        recovers.append(len(top_idx & informative) / k)
        comp_ig.append(float(evaluator.evaluate(predict_fn, instance, expl)["comprehensiveness"]))
        # Random-attribution baseline: same explanation, random importances.
        import dataclasses

        rand_imps = rng.normal(size=N_FEATURES)
        rand_expl = dataclasses.replace(
            expl,
            feature_importances=[
                FeatureImportance(
                    feature_name=f"f{i}",
                    feature_index=i,
                    importance=float(rand_imps[i]),
                    direction="neutral",
                )
                for i in range(N_FEATURES)
            ],
        )
        comp_rand.append(
            float(evaluator.evaluate(predict_fn, instance, rand_expl)["comprehensiveness"])
        )

    return {
        "seed": seed,
        "recovery_at_k": float(np.mean(recovers)),
        "chance_recovery": k / N_FEATURES,
        "comprehensiveness_ig": float(np.mean(comp_ig)),
        "comprehensiveness_random": float(np.mean(comp_rand)),
        "n_explained": len(recovers),
    }


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.getLogger("omni_mercury_engine").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default="artifacts/explanation_fidelity.json", type=str)
    args = parser.parse_args()

    print("Explanation-fidelity validation (IntegratedGradients + FaithfulnessEvaluator)")
    print("-" * 80)
    results = [run(s) for s in args.seeds]
    for r in results:
        print(
            f"  seed={r['seed']}  recovery@{N_INFORMATIVE}={r['recovery_at_k']:.3f} "
            f"(chance {r['chance_recovery']:.3f})  "
            f"comprehensiveness IG={r['comprehensiveness_ig']:.4f} vs random={r['comprehensiveness_random']:.4f}"
        )

    mean_recovery = float(np.mean([r["recovery_at_k"] for r in results]))
    chance = N_INFORMATIVE / N_FEATURES
    mean_comp_ig = float(np.mean([r["comprehensiveness_ig"] for r in results]))
    mean_comp_rand = float(np.mean([r["comprehensiveness_random"] for r in results]))
    passed = bool(mean_recovery > 2 * chance and mean_comp_ig > mean_comp_rand + 0.01)
    verdict = {
        "mean_recovery_at_k": mean_recovery,
        "chance_recovery": chance,
        "mean_comprehensiveness_ig": mean_comp_ig,
        "mean_comprehensiveness_random": mean_comp_rand,
        "passed": passed,
        "verdict": (
            "VALIDATED -- IntegratedGradients recovers the model's informative features "
            "and is faithful above a random baseline; revive the explainer + evaluator"
            if passed
            else "WEAK -- attributions do not clear the recovery/faithfulness bar"
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"results": results, "verdict": verdict}, indent=2, sort_keys=True)
    )
    print("-" * 80)
    print(
        f"recovery@{N_INFORMATIVE}={mean_recovery:.3f} (chance {chance:.3f})  "
        f"comprehensiveness IG={mean_comp_ig:.4f} vs random={mean_comp_rand:.4f}"
    )
    print(f"VERDICT: {verdict['verdict']}")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

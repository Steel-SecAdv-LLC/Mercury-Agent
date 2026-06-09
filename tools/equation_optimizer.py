# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Universal equation optimizer for Mercury mathematical surfaces.

This operator tool implements a reproducible optimization workflow that:

1. inventories equation surfaces from ``docs/MATH_SPEC.md``;
2. freezes an explicit baseline profile preserving original equations;
3. searches a constrained universal candidate family;
4. stress-tests candidates and selects a winner with hard safety gates;
5. emits versioned artifacts plus rollback/revalidation metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_MATH_SPEC = Path("docs/MATH_SPEC.md")
_DEFAULT_OUTPUT_DIR = Path("artifacts/equation_optimization")
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectiveWeights:
    detection_quality: float = 0.26
    calibration: float = 0.18
    stability: float = 0.16
    ethical_compliance: float = 0.16
    robustness: float = 0.10
    latency: float = 0.07
    generalization: float = 0.07

    def normalised(self) -> dict[str, float]:
        raw = asdict(self)
        total = sum(raw.values())
        return {k: float(v / total) for k, v in raw.items()}


@dataclass(frozen=True)
class HardConstraints:
    ethical_min: float = 1.0
    stability_min: float = 1.0
    output_range_min: float = 0.0
    output_range_max: float = 1.0
    contraction_alpha_max: float = 0.999
    lyapunov_lambda_min: float = 1e-6


@dataclass(frozen=True)
class CandidateParams:
    candidate_id: str
    add_weight: float
    mult_weight: float
    inter_weight: float
    w_r: float
    w_h: float
    w_o: float
    w_rh: float
    w_ro: float
    w_ho: float
    ethical_exponent: float


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _inventory_equation_surfaces(math_spec: Path) -> list[dict[str, str]]:
    if not math_spec.exists():
        return []

    lines = math_spec.read_text(encoding="utf-8").splitlines()
    surfaces: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for i, line in enumerate(lines, start=1):
        if line.startswith("### 2."):
            if current is not None:
                surfaces.append(current)
            current = {
                "section": line.strip("# ").strip(),
                "line": str(i),
                "implementation": "",
            }
        elif current is not None and line.startswith("**Implementation:**"):
            current["implementation"] = line.removeprefix("**Implementation:**").strip()

    if current is not None:
        surfaces.append(current)

    return surfaces


def _freeze_baseline_profile() -> dict[str, Any]:
    # Original equations are explicitly preserved as immutable baseline.
    return {
        "profile_id": "baseline_original_v1",
        "description": "Original Mercury equations preserved as canonical baseline",
        "oae": {
            "formula": "A = (w_R*R + w_H*H + w_O*O) * eta^p",
            "weights": {"w_R": 0.4472135955, "w_H": 0.2763932023, "w_O": 0.2763932023},
            "ethical_exponent": 1.6180339887,
        },
        "lyapunov": {"lambda_convergence": 0.25, "epsilon": 1.0},
        "contraction": {"alpha_max": 0.999},
        "preserve_original_equations": True,
    }


def _build_ai_equation_library() -> dict[str, Any]:
    """Return the separated known/in-house equation library for AI operators."""
    return {
        "known_reference_equations": [
            {
                "id": "maut_additive_utility",
                "equation": "U(x)=sum_i w_i u_i(x_i), sum_i w_i=1, w_i>0",
                "purpose": "Composite utility grounding for explicit multi-objective trade-offs.",
                "source": "Keeney & Raiffa, Decisions with Multiple Objectives, 1976",
            },
            {
                "id": "expected_calibration_error",
                "equation": "ECE=sum_m |B_m|/n * |acc(B_m)-conf(B_m)|",
                "purpose": "Probability-calibration check for anomaly and AI confidence outputs.",
                "source": "Guo et al., On Calibration of Modern Neural Networks, ICML 2017",
            },
            {
                "id": "split_conformal_quantile",
                "equation": "q_hat=Quantile_ceil((n+1)(1-alpha))/n({s_i})",
                "purpose": "Finite-sample uncertainty coverage for thresholded decisions.",
                "source": "Vovk, Gammerman & Shafer, Algorithmic Learning in a Random World, 2005",
            },
            {
                "id": "cusum_drift",
                "equation": "S_t=max(0,S_{t-1}+x_t-mu_0-k)",
                "purpose": "Sequential drift/stress signal for continuous revalidation.",
                "source": "Page, Continuous Inspection Schemes, Biometrika 1954",
            },
        ],
        "in_house_equations": [
            {
                "id": "oae_original_baseline",
                "equation": "A=(w_R R+w_H H+w_O O)*eta^p",
                "purpose": "Mercury's preserved 3R anomaly-fusion baseline.",
                "source": "docs/MATH_SPEC.md; src/omni_mercury_engine/core/three_r/fusion.py",
            },
            {
                "id": "ugcm_candidate_family",
                "equation": "G=eta^p*(lambda_a A+lambda_m M+lambda_i I)",
                "purpose": "Universal generalized composite candidate family searched by this tool.",
                "source": "tools/equation_optimizer.py",
            },
            {
                "id": "quiet_horizon_runtime_candidate",
                "equation": (
                    "S=0.70*N+0.30*(eta^sqrt(Phi)*(0.5+0.5*A_RHO)*"
                    "(Phi*R+sqrt(HO)+cuberoot(RHO))/(Phi+2))"
                ),
                "purpose": "Gentle runtime R/H/O agreement profile derived from Mercury's OAE signals.",
                "source": "src/omni_mercury_engine/core/equation_profiles.py",
            },
            {
                "id": "benevolence_phi_index",
                "equation": "B=(Phi*harm_reduction+equity)/(Phi+1)",
                "purpose": "In-house ethical score prioritizing missed-harm reduction.",
                "source": "src/omni_mercury_engine/core/domain_metrics.py",
            },
            {
                "id": "gosnn_hierarchical_geometric_score",
                "equation": "S=prod_c max(s_c,epsilon)^(w_c/sum_j w_j)",
                "purpose": "Neuro-symbolic omni-scalar aggregation that penalizes weak categories.",
                "source": "src/omni_mercury_engine/core/global_omni_scalar_network.py",
            },
        ],
        "separation_rule": (
            "Known reference equations justify validation discipline; Mercury in-house "
            "equations remain preserved baselines or candidate profiles with rollback."
        ),
    }


def _build_default_dataset(seed: int, n: int = 800) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    domains = ("security", "medical", "environmental", "infrastructure", "humanitarian")
    for idx in range(n):
        r = float(np.clip(rng.normal(0.62, 0.18), 0.0, 1.0))
        h = float(np.clip(rng.normal(0.58, 0.20), 0.0, 1.0))
        o = float(np.clip(rng.normal(0.60, 0.16), 0.0, 1.0))
        eta = float(np.clip(rng.normal(0.96, 0.025), 0.90, 1.0))
        gate_ok = bool(eta >= 0.93)
        sigma_ok = bool(rng.random() > 0.01)
        # Synthetic in-house target proxy (stable, deterministic under seed).
        latent = 0.44 * r + 0.30 * h + 0.26 * o
        label = float(np.clip(_sigmoid(6.0 * (latent * eta - 0.48)), 0.0, 1.0))
        rows.append(
            {
                "id": idx,
                "domain": domains[idx % len(domains)],
                "r": r,
                "h": h,
                "o": o,
                "eta": eta,
                "label": label,
                "ood": bool((idx % 7) == 0),
                "latency_ms": float(np.clip(rng.normal(7.5, 2.0), 3.0, 18.0)),
                "sigma_ok": sigma_ok,
                "gate_ok": gate_ok,
                "lyapunov_lambda": float(np.clip(rng.normal(0.30, 0.06), 0.03, 0.8)),
                "alpha": float(np.clip(rng.normal(0.88, 0.06), 0.55, 0.995)),
            }
        )
    return rows


def _load_dataset(dataset_path: Path | None, seed: int) -> tuple[list[dict[str, Any]], str]:
    if dataset_path is None:
        return _build_default_dataset(seed=seed), "synthetic_default"

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("dataset JSON must be a list of records")

    required = {"r", "h", "o", "eta", "label"}
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("dataset entries must be objects")
        missing = required - set(row)
        if missing:
            raise ValueError(f"dataset entry missing fields: {sorted(missing)}")
        rows.append(dict(row))
    return rows, f"file:{dataset_path}"


def _normalise_primary_weights(w_r: float, w_h: float, w_o: float) -> tuple[float, float, float]:
    denom = max(_EPS, abs(w_r) + abs(w_h) + abs(w_o))
    return abs(w_r) / denom, abs(w_h) / denom, abs(w_o) / denom


def _score_candidate(params: CandidateParams, row: dict[str, Any]) -> float:
    r = float(np.clip(row["r"], 0.0, 1.0))
    h = float(np.clip(row["h"], 0.0, 1.0))
    o = float(np.clip(row["o"], 0.0, 1.0))
    eta = float(np.clip(row["eta"], 0.0, 1.0))

    add = params.w_r * r + params.w_h * h + params.w_o * o
    mult = (
        (max(_EPS, r) ** params.w_r) * (max(_EPS, h) ** params.w_h) * (max(_EPS, o) ** params.w_o)
    )
    inter = params.w_rh * r * h + params.w_ro * r * o + params.w_ho * h * o

    combined = params.add_weight * add + params.mult_weight * mult + params.inter_weight * inter
    return float(np.clip(combined, 0.0, 1.0) * (eta**params.ethical_exponent))


def _evaluate_candidate(
    params: CandidateParams,
    rows: list[dict[str, Any]],
    weights: dict[str, float],
    constraints: HardConstraints,
) -> dict[str, Any]:
    preds = np.array([_score_candidate(params, row) for row in rows], dtype=np.float64)
    labels = np.array([float(np.clip(row["label"], 0.0, 1.0)) for row in rows], dtype=np.float64)

    mse = float(np.mean((preds - labels) ** 2))
    mae = float(np.mean(np.abs(preds - labels)))
    detection_quality = float(np.clip(1.0 - mse, 0.0, 1.0))
    calibration = float(np.clip(1.0 - mae, 0.0, 1.0))

    stabilities = [
        bool(float(row.get("alpha", 1.0)) < constraints.contraction_alpha_max)
        and bool(float(row.get("lyapunov_lambda", 1.0)) > constraints.lyapunov_lambda_min)
        for row in rows
    ]
    stability = float(sum(stabilities) / max(1, len(stabilities)))

    monotonic_ok = _check_eta_monotonicity(params)
    ethical_compliance = 1.0 if (monotonic_ok and params.ethical_exponent > 0.0) else 0.0
    observed_gate_health = float(
        sum(bool(row.get("sigma_ok", True)) and bool(row.get("gate_ok", True)) for row in rows)
        / max(1, len(rows))
    )

    rng = np.random.default_rng(0)
    jitter = np.clip(preds + rng.normal(0.0, 0.02, size=preds.shape), 0.0, 1.0)
    robustness = float(np.clip(1.0 - np.mean(np.abs(jitter - preds)) / 0.1, 0.0, 1.0))

    latency_ms = np.array([float(row.get("latency_ms", 8.0)) for row in rows], dtype=np.float64)
    complexity_penalty = 0.25 * (params.mult_weight + params.inter_weight)
    latency = float(
        np.clip(1.0 - ((float(np.mean(latency_ms)) / 25.0) + complexity_penalty), 0.0, 1.0)
    )

    ood_mask = np.array([bool(row.get("ood", False)) for row in rows], dtype=bool)
    if ood_mask.any() and (~ood_mask).any():
        id_mean = float(np.mean(preds[~ood_mask]))
        ood_mean = float(np.mean(preds[ood_mask]))
        generalization = float(np.clip(1.0 - abs(id_mean - ood_mean), 0.0, 1.0))
    else:
        generalization = 0.95

    objective = (
        weights["detection_quality"] * detection_quality
        + weights["calibration"] * calibration
        + weights["stability"] * stability
        + weights["ethical_compliance"] * ethical_compliance
        + weights["robustness"] * robustness
        + weights["latency"] * latency
        + weights["generalization"] * generalization
    )

    metrics = {
        "detection_quality": detection_quality,
        "calibration": calibration,
        "stability": stability,
        "ethical_compliance": ethical_compliance,
        "robustness": robustness,
        "latency": latency,
        "generalization": generalization,
    }

    output_range_ok = bool(
        float(np.min(preds)) >= constraints.output_range_min
        and float(np.max(preds)) <= constraints.output_range_max
    )
    ethical_ok = ethical_compliance >= constraints.ethical_min
    stability_ok = stability >= constraints.stability_min
    finite_ok = bool(np.isfinite(preds).all())
    constraints_ok = output_range_ok and ethical_ok and stability_ok and monotonic_ok and finite_ok

    return {
        "candidate_id": params.candidate_id,
        "objective": float(objective),
        "metrics": metrics,
        "constraints_ok": constraints_ok,
        "constraint_detail": {
            "output_range_ok": output_range_ok,
            "ethical_ok": ethical_ok,
            "stability_ok": stability_ok,
            "monotonic_ok": monotonic_ok,
            "finite_ok": finite_ok,
            "observed_gate_health": observed_gate_health,
        },
        "params": asdict(params),
    }


def _check_eta_monotonicity(params: CandidateParams) -> bool:
    x = np.linspace(0.0, 1.0, 200)
    y = np.power(np.clip(x, 0.0, 1.0), params.ethical_exponent)
    return bool(np.all(np.diff(y) >= -1e-10))


def _build_baseline_candidate() -> CandidateParams:
    return CandidateParams(
        candidate_id="baseline_original_v1",
        add_weight=1.0,
        mult_weight=0.0,
        inter_weight=0.0,
        w_r=0.4472135955,
        w_h=0.2763932023,
        w_o=0.2763932023,
        w_rh=0.0,
        w_ro=0.0,
        w_ho=0.0,
        ethical_exponent=1.6180339887,
    )


def _sample_candidate(rng: np.random.Generator, idx: int) -> CandidateParams:
    raw_r, raw_h, raw_o = [float(rng.uniform(0.01, 1.0)) for _ in range(3)]
    w_r, w_h, w_o = _normalise_primary_weights(raw_r, raw_h, raw_o)

    raw_mix = np.array(
        [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)],
        dtype=np.float64,
    )
    raw_mix = raw_mix / max(_EPS, float(np.sum(raw_mix)))

    return CandidateParams(
        candidate_id=f"candidate_{idx:04d}",
        add_weight=float(raw_mix[0]),
        mult_weight=float(raw_mix[1]),
        inter_weight=float(raw_mix[2]),
        w_r=w_r,
        w_h=w_h,
        w_o=w_o,
        w_rh=float(rng.uniform(-0.35, 0.35)),
        w_ro=float(rng.uniform(-0.35, 0.35)),
        w_ho=float(rng.uniform(-0.35, 0.35)),
        ethical_exponent=float(rng.uniform(1.0, 2.2)),
    )


def _run_search(
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
    objective_weights: dict[str, float],
    constraints: HardConstraints,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    evaluations: list[dict[str, Any]] = []

    baseline = _build_baseline_candidate()
    baseline_eval = _evaluate_candidate(baseline, rows, objective_weights, constraints)
    evaluations.append(baseline_eval)

    for i in range(1, iterations + 1):
        candidate = _sample_candidate(rng, i)
        result = _evaluate_candidate(candidate, rows, objective_weights, constraints)
        evaluations.append(result)

    feasible = [e for e in evaluations if e["constraints_ok"]]
    if feasible:
        best = max(feasible, key=lambda e: float(e["objective"]))
    else:
        best = baseline_eval

    return (
        baseline_eval,
        best,
        sorted(evaluations, key=lambda e: float(e["objective"]), reverse=True),
    )


def _select_winner(
    baseline_eval: dict[str, Any],
    best_eval: dict[str, Any],
    min_delta: float = 0.002,
) -> dict[str, Any]:
    baseline_obj = float(baseline_eval["objective"])
    best_obj = float(best_eval["objective"])
    if not bool(best_eval["constraints_ok"]):
        return baseline_eval
    if best_obj < baseline_obj + min_delta:
        return baseline_eval
    return best_eval


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_pipeline(
    math_spec: Path,
    dataset_path: Path | None,
    output_dir: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    inventory = _inventory_equation_surfaces(math_spec)
    baseline_profile = _freeze_baseline_profile()
    rows, dataset_source = _load_dataset(dataset_path, seed)
    objective_weights = ObjectiveWeights().normalised()
    constraints = HardConstraints()

    baseline_eval, best_eval, ranking = _run_search(
        rows=rows,
        iterations=iterations,
        seed=seed,
        objective_weights=objective_weights,
        constraints=constraints,
    )
    winner_eval = _select_winner(baseline_eval, best_eval)

    top_candidates = ranking[: min(40, len(ranking))]
    winner_constraints_ok = bool(winner_eval["constraints_ok"])
    summary = {
        "ok": winner_constraints_ok,
        "dataset_source": dataset_source,
        "iterations": iterations,
        "seed": seed,
        "preserve_original_equations": True,
        "inventory_count": len(inventory),
        "baseline_objective": float(baseline_eval["objective"]),
        "best_objective": float(best_eval["objective"]),
        "winner_id": winner_eval["candidate_id"],
        "winner_objective": float(winner_eval["objective"]),
        "winner_constraints_ok": winner_constraints_ok,
    }

    _write_json(output_dir / "equation_inventory.json", {"surfaces": inventory})
    _write_json(output_dir / "baseline_profile.json", baseline_profile)
    _write_json(output_dir / "ai_equation_library.json", _build_ai_equation_library())
    _write_json(
        output_dir / "search_space.json",
        {
            "candidate_family": "additive + multiplicative + pairwise-interaction + ethical exponent",
            "constraints": asdict(constraints),
            "objective_weights": objective_weights,
            "includes_original_baseline": True,
        },
    )
    _write_json(output_dir / "candidate_ranking.json", {"candidates": top_candidates})
    _write_json(output_dir / "winner.json", winner_eval)
    _write_json(
        output_dir / "equation_profiles_v1.json",
        {
            "version": "v1",
            "active_profile": winner_eval,
            "baseline_profile": baseline_eval,
            "preserve_original_equations": True,
        },
    )
    _write_json(
        output_dir / "rollback_switch.json",
        {
            "active_profile_id": winner_eval["candidate_id"],
            "rollback_profile_id": "baseline_original_v1",
            "reason": "Immediate rollback to preserved original equations if canary regresses",
        },
    )
    _write_json(
        output_dir / "continuous_revalidation.json",
        {
            "schedule": {
                "nightly_rebenchmark_cron": "0 2 * * *",
                "weekly_full_search_cron": "0 3 * * 0",
            },
            "drift_trigger": {
                "objective_drop_threshold": 0.02,
                "ethical_failure_tolerance": 0.0,
                "stability_failure_tolerance": 0.0,
            },
            "policy": "If drift trigger fires, retain baseline_original_v1 as safe fallback while rerunning constrained search.",
        },
    )
    _write_json(
        output_dir / "decision_ledger.json",
        {
            "decisions": [
                {
                    "decision": "preserve_original_equations",
                    "justification": "Original Mercury equations are retained as immutable baselines.",
                },
                {
                    "decision": "constrained_candidate_search",
                    "justification": "Candidates must clear ethical, stability, finite, monotonic, and output-range gates.",
                },
                {
                    "decision": "multi_axis_objective",
                    "justification": "Detection quality is scored with calibration, robustness, latency, and generalization.",
                },
                {
                    "decision": "rollback_first",
                    "justification": "The preserved baseline remains the immediate fallback for canary or drift regressions.",
                },
            ]
        },
    )
    _write_json(output_dir / "summary.json", summary)
    return summary


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.equation_optimizer",
        description="Optimize Mercury equation family while preserving original equations.",
    )
    parser.add_argument(
        "--math-spec",
        type=Path,
        default=_DEFAULT_MATH_SPEC,
        help="Path to docs/MATH_SPEC.md",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional JSON dataset path (list of records with r,h,o,eta,label).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for optimization artifacts.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Number of random-search candidates (baseline is always included).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Deterministic random seed.",
    )
    args = parser.parse_args(argv)

    try:
        summary = _run_pipeline(
            math_spec=args.math_spec,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            iterations=max(1, args.iterations),
            seed=args.seed,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 2

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if bool(summary["ok"]) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    raise SystemExit(_cli())

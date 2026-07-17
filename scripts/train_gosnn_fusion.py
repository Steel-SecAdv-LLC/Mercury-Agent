# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train the GOSNN attention fusion end-to-end on labelled ADBench outcomes.

Task-grounded objective
=======================
The fusion's consequential job is corroborating detection verdicts: a small
detection head over the 37-dim fused state feeds the decision layer's
disagreement overlay (``DecisionAbstentionResponder``), demoting a grounded
verdict to a deferral when the cross-detector fusion strongly disagrees.
Training therefore optimises the fusion weights and the head **jointly
against real labels**: every harvested ``fuse()`` input is paired with the
ADBench ground-truth label of the exact row the engine was detecting, and
the loss is class-weighted BCE on that label (optionally with a small
masked-reconstruction auxiliary).  The earlier reconstruction-only objective
lives on as a disclosure metric, not the gate.

Architecture parity
===================
The trainable stack is imported from
``omni_mercury_engine.core.attention_fusion_stack`` — the same module the
production ``MultiHeadAttentionFusion`` builds its modules from — so train
and serve semantics cannot drift (the previous private ``_FusionModel``
had already diverged from the serve path).

Detection-metric merit gate (can refuse)
========================================
The consequential head ships only when ALL hold on held-out labelled calls:

1. harvest diversity >= ``MIN_UNIQUE_LISTS`` unique state lists (tripwire
   against the 2026-07-14 degenerate-harvest incident);
2. held-out AUC of the learned fused-state head beats BOTH baselines by
   ``GATE_AUC_MARGIN``: (a) the production phi-weighted reference fusion
   with an identically-trained head, and (b) the raw mean detector score;
3. no representational collapse (fused std >= floor x member std);
4. the disagreement-demotion rule, at thresholds selected on the validation
   split, is *useful* on the test split: it fires, its demotions are
   enriched in wrong engine verdicts (enrichment > 1), and it retains
   >= ``TEST_RETENTION_FLOOR`` of correct verdicts.

On refusal nothing consequential ships: the existing reconstruction-gated
checkpoint (observability-only) is left untouched and the refusal is
recorded in ``artifacts/gosnn_fusion.eval.json`` + the dormancy ledger.

Non-circularity
===============
Labels are ADBench ground truth for rows the engine never fitted on; the
state lists are the production ``fuse()`` inputs recorded verbatim (detector
scores computed *before* the OmniFusionModel produces ``anomaly_prob``, so
the head never sees the verdict it later corroborates).  The engine verdicts
recorded per call are used only to evaluate the decision rule, never as a
training signal.

Run:
    python scripts/train_gosnn_fusion.py            # full harvest + gate
    python scripts/train_gosnn_fusion.py --quick    # smaller harvest (smoke)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.core.attention_fusion_stack import (
    DEFAULT_MAX_DIMENSIONS as MAX_DIMS,
    FusionDetectionHead,
    TrainableFusionStack,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_gosnn_fusion")

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache" / "adbench"
CHECKPOINT_PATH = (
    REPO / "src" / "omni_mercury_engine" / "models" / "checkpoints" / "gosnn_attention_fusion.pt"
)
PROVENANCE_PATH = CHECKPOINT_PATH.with_suffix("").with_suffix(".provenance.json")
EVAL_PATH = REPO / "artifacts" / "gosnn_fusion.eval.json"

SEED = 20260717
#: Learned held-out AUC must beat every baseline by at least this margin.
GATE_AUC_MARGIN = 0.02
COLLAPSE_FLOOR = 0.20
#: Harvest-diversity tripwire (2026-07-14 degenerate-harvest incident).
MIN_UNIQUE_LISTS = 50
#: Validation threshold search keeps at least this share of correct verdicts.
VAL_RETENTION_FLOOR = 0.95
#: The shipped rule must keep at least this share of correct verdicts on test.
TEST_RETENTION_FLOOR = 0.90
#: Weight of the masked-reconstruction auxiliary in the joint candidate
#: (the reconstruction term is normalised by the phi-reference MSE so it is
#: O(1) like the BCE term).
RECON_AUX_WEIGHT = 0.1


@dataclass
class HarvestCall:
    """One production ``fuse()`` invocation with its labelled outcome."""

    states: list[np.ndarray]
    label: int
    engine_prob: float
    engine_verdict: bool
    dataset: str
    masked_index: int = 0


@dataclass
class Pools:
    """Per-role index pools over the harvested calls."""

    train: list[int] = field(default_factory=list)
    val: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)


def _pad(states: list[np.ndarray]) -> np.ndarray:
    """Pad/truncate each member to the production 37-dim layout."""
    out = np.zeros((len(states), MAX_DIMS))
    for i, s in enumerate(states):
        out[i, : min(len(s), MAX_DIMS)] = s[:MAX_DIMS]
    return out


def _phi_reference(members: np.ndarray) -> np.ndarray:
    """Production phi-weighted average over the given members."""
    phi = (1 + 5**0.5) / 2
    weights = np.tile(np.array([phi, 1.0, 1.0 / phi]), len(members) // 3 + 1)[: len(members)]
    weights = weights / weights.sum()
    return np.average(members, axis=0, weights=weights)


def _unique_state_lists(calls: list[HarvestCall]) -> int:
    """Number of distinct harvested state lists (rounded to 9 decimals)."""
    seen = set()
    for call in calls:
        seen.add(np.round(_pad(call.states), 9).ravel().tobytes())
    return len(seen)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based ROC AUC; ``None`` when a class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # Average ranks for ties so identical scores carry no ordering signal.
    sorted_scores = scores[order]
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks_sorted[i : j + 1] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    ranks[order] = ranks_sorted
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _harvest(n_datasets: int, n_fit_rows: int, n_detect_rows: int) -> list[HarvestCall]:
    """Record labelled production ``fuse()`` inputs from a real engine.

    A real ``OmniMercuryEngine`` is fitted on ADBench training rows and run
    over held-out test rows; every dimensional-state list the GOSNN attention
    fusion receives is recorded verbatim, paired with the ground-truth label
    of the exact row being detected and the engine's own verdict on it.
    """
    from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network
    from omni_mercury_engine.engine import OmniMercuryEngine

    recorded_states: list[list[np.ndarray]] = []
    net = get_global_scalar_network()
    fusion = net.attention_fusion
    original_fuse = fusion.fuse

    def recording_fuse(dimensional_states: list[np.ndarray], *args: Any, **kwargs: Any) -> Any:
        recorded_states.append([np.asarray(s, dtype=np.float64).copy() for s in dimensional_states])
        return original_fuse(dimensional_states, *args, **kwargs)

    calls: list[HarvestCall] = []
    fusion.fuse = recording_fuse  # type: ignore[method-assign]
    try:
        caches = sorted(CACHE.glob("*.npz"))[:n_datasets]
        if not caches:
            raise FileNotFoundError(
                f"no cached ADBench datasets under {CACHE}; run "
                "benchmarks/competitive_regression_guard.py once to populate it"
            )
        engine = OmniMercuryEngine()
        rng = np.random.default_rng(SEED)
        for path in caches:
            data = np.load(path)
            x_train = np.asarray(data["train_features"], dtype=np.float64)
            y_train = np.asarray(data["train_labels"], dtype=np.int64)
            x_test = np.asarray(data["test_features"], dtype=np.float64)
            y_test = np.asarray(data["test_labels"], dtype=np.int64)
            fit_idx = rng.choice(len(x_train), size=min(n_fit_rows, len(x_train)), replace=False)
            engine.fit_fusion(x_train[fit_idx], y_train[fit_idx])
            det_idx = rng.choice(len(x_test), size=min(n_detect_rows, len(x_test)), replace=False)
            # One row per call: the GOSNN fusion runs once per detect call,
            # so each recorded state list aligns with exactly one label.
            for i in det_idx:
                before = len(recorded_states)
                result = engine.detect_with_fusion(x_test[i : i + 1])
                if len(recorded_states) != before + 1:
                    raise RuntimeError(
                        "fuse() call count did not advance by exactly 1 for a "
                        f"detect call ({before} -> {len(recorded_states)}); "
                        "state/label alignment would be corrupt -- aborting"
                    )
                calls.append(
                    HarvestCall(
                        states=recorded_states[-1],
                        label=int(y_test[i]),
                        engine_prob=float(result["anomaly_prob"]),
                        engine_verdict=bool(result["is_anomaly"]),
                        dataset=path.stem,
                    )
                )
            logger.info("harvested %s: %d labelled calls so far", path.name, len(calls))
    finally:
        fusion.fuse = original_fuse  # type: ignore[method-assign]
    return calls


def _split_by_dataset(calls: list[HarvestCall], rng: np.random.Generator) -> Pools:
    """70/15/15 split *within each dataset* so no pool is one dataset only."""
    pools = Pools()
    by_dataset: dict[str, list[int]] = {}
    for idx, call in enumerate(calls):
        by_dataset.setdefault(call.dataset, []).append(idx)
    for indices in by_dataset.values():
        shuffled = list(rng.permutation(indices))
        n = len(shuffled)
        pools.train.extend(shuffled[: int(n * 0.7)])
        pools.val.extend(shuffled[int(n * 0.7) : int(n * 0.85)])
        pools.test.extend(shuffled[int(n * 0.85) :])
    return pools


def _head_standardizer(
    calls: list[HarvestCall], train_idx: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train-split standardisation statistics from phi-reference fused states.

    Model-independent (computed from the deterministic reference fusion of the
    raw member states), so the statistics are stable while the fusion weights
    train, and they ship inside the head's ``state_dict`` for exact serve
    parity.
    """
    reference = np.stack([_phi_reference(_pad(calls[i].states)) for i in train_idx])
    mean = torch.tensor(reference.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(reference.std(axis=0), dtype=torch.float32)
    return mean, std


def _train_candidate(
    name: str,
    calls: list[HarvestCall],
    pools: Pools,
    epochs: int,
    ref_mse: float,
) -> tuple[TrainableFusionStack, float | None]:
    """Train one candidate end-to-end; return (best model, best val AUC)."""
    torch.manual_seed(SEED)
    model = TrainableFusionStack()
    mean, std = _head_standardizer(calls, pools.train)
    model.detection_head.set_standardizer(mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    labels_train = np.array([calls[i].label for i in pools.train])
    n_pos = max(int(labels_train.sum()), 1)
    n_neg = max(len(labels_train) - n_pos, 1)
    pos_weight = torch.tensor(float(n_neg) / float(n_pos))
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    recon_weight = RECON_AUX_WEIGHT if name == "detection_recon" else 0.0

    best_auc: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(epochs):
        model.train()
        for i in pools.train:
            call = calls[i]
            optimizer.zero_grad()
            members = torch.tensor(_pad(call.states), dtype=torch.float32)
            fused = model(members)
            loss = bce(model.detection_head(fused), torch.tensor(float(call.label)))
            if recon_weight and len(call.states) >= 2:
                padded = _pad(call.states)
                rest = np.delete(padded, call.masked_index, axis=0)
                masked_fused = model(torch.tensor(rest, dtype=torch.float32))
                recon = nn.functional.mse_loss(
                    masked_fused, torch.tensor(padded[call.masked_index], dtype=torch.float32)
                )
                loss = loss + recon_weight * recon / max(ref_mse, 1e-12)
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()

        model.eval()
        val_auc = _aggregate_auc(_head_scores(model, calls, pools.val), calls, pools.val)
        if val_auc is not None and (best_auc is None or val_auc > best_auc):
            best_auc = val_auc
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        logger.info(
            "%s epoch %d: val AUC %s (best %s)",
            name,
            epoch + 1,
            "n/a" if val_auc is None else f"{val_auc:.4f}",
            "n/a" if best_auc is None else f"{best_auc:.4f}",
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


def _head_scores(
    model: TrainableFusionStack, calls: list[HarvestCall], indices: list[int]
) -> np.ndarray:
    """Detection-head probabilities for the given calls."""
    scores = []
    with torch.no_grad():
        for i in indices:
            members = torch.tensor(_pad(calls[i].states), dtype=torch.float32)
            scores.append(float(torch.sigmoid(model.detection_logit(members)).item()))
    return np.array(scores)


def _aggregate_auc(
    scores: np.ndarray, calls: list[HarvestCall], indices: list[int]
) -> float | None:
    """Mean per-dataset AUC (datasets with both classes), else pooled AUC."""
    labels = np.array([calls[i].label for i in indices])
    datasets = np.array([calls[i].dataset for i in indices])
    per_dataset = [
        _auc(scores[datasets == name], labels[datasets == name]) for name in set(datasets)
    ]
    defined = [a for a in per_dataset if a is not None]
    if len(defined) >= 2:
        return float(np.mean(defined))
    return _auc(scores, labels)


def _per_dataset_auc(
    scores: np.ndarray, calls: list[HarvestCall], indices: list[int]
) -> dict[str, float | None]:
    """Per-dataset AUC map for the eval artifact."""
    labels = np.array([calls[i].label for i in indices])
    datasets = np.array([calls[i].dataset for i in indices])
    return {
        name: _auc(scores[datasets == name], labels[datasets == name])
        for name in sorted(set(datasets))
    }


def _train_reference_head(
    calls: list[HarvestCall], pools: Pools, epochs: int
) -> FusionDetectionHead:
    """The phi-reference baseline: identical head, identically trained,
    on the deterministic reference fusion of the same state lists."""
    torch.manual_seed(SEED)
    head = FusionDetectionHead()
    mean, std = _head_standardizer(calls, pools.train)
    head.set_standardizer(mean, std)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)
    labels_train = np.array([calls[i].label for i in pools.train])
    n_pos = max(int(labels_train.sum()), 1)
    n_neg = max(len(labels_train) - n_pos, 1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(n_neg) / float(n_pos)))
    fused_ref = {
        i: torch.tensor(_phi_reference(_pad(calls[i].states)), dtype=torch.float32)
        for i in pools.train
    }
    for _ in range(epochs):
        head.train()
        for i in pools.train:
            optimizer.zero_grad()
            loss = bce(head(fused_ref[i]), torch.tensor(float(calls[i].label)))
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()
    head.eval()
    return head


def _reference_head_scores(
    head: FusionDetectionHead, calls: list[HarvestCall], indices: list[int]
) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for i in indices:
            fused = torch.tensor(_phi_reference(_pad(calls[i].states)), dtype=torch.float32)
            scores.append(float(torch.sigmoid(head(fused)).item()))
    return np.array(scores)


def _mean_detector_scores(calls: list[HarvestCall], indices: list[int]) -> np.ndarray:
    """The 'mean' baseline: unweighted mean of the per-call detector scores
    (the base member the engine feeds the fusion)."""
    return np.array(
        [float(np.mean(calls[i].states[0])) if len(calls[i].states[0]) else 0.0 for i in indices]
    )


def _decision_rule_metrics(
    scores: np.ndarray,
    calls: list[HarvestCall],
    indices: list[int],
    act_below: float,
    clear_above: float,
) -> dict[str, float]:
    """Verdict-level effect of the disagreement-demotion rule.

    The engine's recorded per-call verdict stands in for the grounded
    decision (no conformal calibrator runs during harvest); wrongness is
    measured against ground truth.  Reported: how often the rule fires, how
    enriched its demotions are in wrong verdicts, and how many correct
    verdicts survive.
    """
    verdicts = np.array([calls[i].engine_verdict for i in indices], dtype=bool)
    labels = np.array([calls[i].label for i in indices], dtype=bool)
    wrong = verdicts != labels
    demoted = (verdicts & (scores <= act_below)) | (~verdicts & (scores >= clear_above))
    n = len(indices)
    n_demoted = int(demoted.sum())
    base_error = float(wrong.mean()) if n else 0.0
    demoted_error = float(wrong[demoted].mean()) if n_demoted else 0.0
    correct = ~wrong
    retention = float((correct & ~demoted).sum() / max(int(correct.sum()), 1))
    return {
        "n": float(n),
        "n_demoted": float(n_demoted),
        "base_error_rate": base_error,
        "demoted_error_rate": demoted_error,
        "enrichment": (demoted_error / base_error) if base_error > 0 else 0.0,
        "correct_retention": retention,
        "wrong_demoted": float((wrong & demoted).sum()),
        "correct_demoted": float((correct & demoted).sum()),
    }


def _select_thresholds(
    scores: np.ndarray, calls: list[HarvestCall], indices: list[int]
) -> tuple[float, float]:
    """Grid-search demotion thresholds on the validation split.

    Maximises (wrong demoted - correct demoted) subject to a
    ``VAL_RETENTION_FLOOR`` correct-verdict retention; falls back to the
    inert-ish (0.05, 0.95) pair when nothing clears the floor (the test-split
    gate will then refuse on its own evidence).
    """
    best = (0.05, 0.95)
    best_gain = -np.inf
    for act_below in np.arange(0.05, 0.50, 0.05):
        for clear_above in np.arange(0.55, 1.00, 0.05):
            m = _decision_rule_metrics(scores, calls, indices, act_below, clear_above)
            if m["correct_retention"] < VAL_RETENTION_FLOOR:
                continue
            gain = m["wrong_demoted"] - m["correct_demoted"]
            if gain > best_gain:
                best_gain = gain
                best = (round(float(act_below), 2), round(float(clear_above), 2))
    return best


def _collapse_metrics(
    model: TrainableFusionStack, calls: list[HarvestCall], indices: list[int]
) -> dict[str, float]:
    """Fused-vs-member std on held-out calls (representational collapse)."""
    fused_stds, member_stds = [], []
    with torch.no_grad():
        for i in indices:
            padded = _pad(calls[i].states)
            fused = model(torch.tensor(padded, dtype=torch.float32)).numpy()
            fused_stds.append(float(np.std(fused)))
            member_stds.append(float(np.std(padded)))
    return {"fused_std": float(np.mean(fused_stds)), "member_std": float(np.mean(member_stds))}


def _recon_disclosure(
    model: TrainableFusionStack, calls: list[HarvestCall], indices: list[int]
) -> dict[str, float]:
    """Masked-member reconstruction of the winner vs the phi-reference.

    Disclosure only (the gate is the detection metric): records how the
    jointly-trained fusion fares on the earlier faithful-fusion task.
    """
    learned_se, reference_se = [], []
    with torch.no_grad():
        for i in indices:
            call = calls[i]
            if len(call.states) < 2:
                continue
            padded = _pad(call.states)
            rest = np.delete(padded, call.masked_index, axis=0)
            target = padded[call.masked_index]
            fused = model(torch.tensor(rest, dtype=torch.float32)).numpy()
            learned_se.append(float(np.mean((fused - target) ** 2)))
            reference_se.append(float(np.mean((_phi_reference(rest) - target) ** 2)))
    return {
        "masked_recon_mse": float(np.mean(learned_se)),
        "reference_recon_mse": float(np.mean(reference_se)),
    }


def main() -> int:
    """Harvest, train candidates, gate on the detection metric, ship or refuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller harvest/epochs (smoke)")
    args = parser.parse_args()
    n_datasets, n_fit, n_detect, epochs = (2, 200, 60, 3) if args.quick else (3, 400, 300, 12)

    calls = _harvest(n_datasets, n_fit, n_detect)
    logger.info("harvested %d labelled production calls", len(calls))
    if len(calls) < 60:
        raise RuntimeError(f"harvest too small to split honestly ({len(calls)} calls)")

    n_unique = _unique_state_lists(calls)
    logger.info("harvest diversity: %d unique state lists of %d", n_unique, len(calls))
    generated_utc = datetime.now(UTC).isoformat()
    if n_unique < MIN_UNIQUE_LISTS:
        verdict_degenerate: dict[str, Any] = {
            "generated_utc": generated_utc,
            "seed": SEED,
            "harvest": {
                "n_calls": len(calls),
                "n_unique_state_lists": n_unique,
                "min_unique_required": MIN_UNIQUE_LISTS,
                "quick": bool(args.quick),
            },
            "consequential": {"shipped": False, "reason": "degenerate harvest"},
            "decision": (
                "REFUSED - DEGENERATE HARVEST: the production fuse() input "
                f"distribution collapsed to {n_unique} unique state list(s) "
                f"across {len(calls)} calls. A held-out gate over a constant "
                "is vacuous, so nothing trains and nothing ships."
            ),
        }
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps(verdict_degenerate, indent=2, sort_keys=True) + "\n")
        logger.warning("degenerate harvest; refusal written to %s", EVAL_PATH)
        return 0

    rng = np.random.default_rng(SEED)
    for call in calls:
        call.masked_index = int(rng.integers(len(call.states))) if len(call.states) > 1 else 0
    pools = _split_by_dataset(calls, rng)
    label_counts = {
        pool: {
            "n": len(indices),
            "n_anomalies": int(sum(calls[i].label for i in indices)),
        }
        for pool, indices in (("train", pools.train), ("val", pools.val), ("test", pools.test))
    }
    logger.info("pools: %s", label_counts)

    # The phi-reference masked-recon MSE on the train split normalises the
    # reconstruction auxiliary to O(1).
    ref_mse_train = float(
        np.mean(
            [
                np.mean(
                    (
                        _phi_reference(np.delete(_pad(calls[i].states), calls[i].masked_index, 0))
                        - _pad(calls[i].states)[calls[i].masked_index]
                    )
                    ** 2
                )
                for i in pools.train
                if len(calls[i].states) >= 2
            ]
        )
    )

    results: dict[str, dict[str, Any]] = {}
    models: dict[str, TrainableFusionStack] = {}
    for name in ("detection", "detection_recon"):
        model, best_val_auc = _train_candidate(name, calls, pools, epochs, ref_mse_train)
        test_scores = _head_scores(model, calls, pools.test)
        metrics: dict[str, Any] = {
            "val_auc": best_val_auc,
            "test_auc": _aggregate_auc(test_scores, calls, pools.test),
            "test_auc_pooled": _auc(test_scores, np.array([calls[i].label for i in pools.test])),
            "test_auc_per_dataset": _per_dataset_auc(test_scores, calls, pools.test),
            **_collapse_metrics(model, calls, pools.test),
            **_recon_disclosure(model, calls, pools.test),
        }
        results[name] = metrics
        models[name] = model
        logger.info("%s: %s", name, json.dumps(metrics, default=str))

    # Baselines on the identical held-out calls.
    reference_head = _train_reference_head(calls, pools, epochs)
    phi_scores_test = _reference_head_scores(reference_head, calls, pools.test)
    mean_scores_test = _mean_detector_scores(calls, pools.test)
    baselines = {
        "phi_reference_head": {
            "test_auc": _aggregate_auc(phi_scores_test, calls, pools.test),
            "test_auc_per_dataset": _per_dataset_auc(phi_scores_test, calls, pools.test),
        },
        "mean_detector_score": {
            "test_auc": _aggregate_auc(mean_scores_test, calls, pools.test),
            "test_auc_per_dataset": _per_dataset_auc(mean_scores_test, calls, pools.test),
        },
    }
    logger.info("baselines: %s", json.dumps(baselines, default=str))

    # Candidate selection on validation AUC; gate on held-out test evidence.
    candidates_with_val = [k for k in results if results[k]["val_auc"] is not None]
    verdict: dict[str, Any] = {
        "generated_utc": generated_utc,
        "seed": SEED,
        "gate": {
            "auc_margin": GATE_AUC_MARGIN,
            "collapse_floor": COLLAPSE_FLOOR,
            "test_retention_floor": TEST_RETENTION_FLOOR,
            "constraint": (
                "held-out detection AUC (mean per-dataset) must beat the "
                "phi-reference-fusion head AND the mean detector score by the "
                "margin; fused std >= floor x member std; the demotion rule "
                "(thresholds selected on validation) must fire on test, be "
                "enriched in wrong verdicts (enrichment > 1), and retain >= "
                "floor of correct verdicts"
            ),
        },
        "harvest": {
            "n_calls": len(calls),
            "n_unique_state_lists": n_unique,
            "pools": label_counts,
            "quick": bool(args.quick),
        },
        "candidates": results,
        "baselines": baselines,
    }

    def _refuse(reason: str) -> int:
        verdict["consequential"] = {"shipped": False, "reason": reason}
        verdict["decision"] = (
            "SHIPPED (observability-only) - consequential role REFUSED: "
            f"{reason}. The existing reconstruction-gated checkpoint keeps "
            "serving observability only; no detection head ships and the "
            "decision layer's GOSNN overlay stays structurally inert."
            if CHECKPOINT_PATH.exists()
            else f"REFUSED - {reason}. No checkpoint ships."
        )
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        logger.info("gate refused (%s); eval written to %s", reason, EVAL_PATH)
        return 0

    if not candidates_with_val:
        return _refuse("validation AUC undefined (a class is absent from the validation pool)")

    best_name = max(candidates_with_val, key=lambda k: results[k]["val_auc"])
    best_model = models[best_name]
    best_metrics = results[best_name]
    test_scores = _head_scores(best_model, calls, pools.test)

    val_scores = _head_scores(best_model, calls, pools.val)
    act_below, clear_above = _select_thresholds(val_scores, calls, pools.val)
    rule_val = _decision_rule_metrics(val_scores, calls, pools.val, act_below, clear_above)
    rule_test = _decision_rule_metrics(test_scores, calls, pools.test, act_below, clear_above)
    verdict["decision_rule"] = {
        "demote_act_below": act_below,
        "demote_clear_above": clear_above,
        "selected_on": "validation split",
        "validation": rule_val,
        "test": rule_test,
    }
    verdict["selected_candidate"] = best_name

    learned_auc = best_metrics["test_auc"]
    phi_auc = baselines["phi_reference_head"]["test_auc"]
    mean_auc = baselines["mean_detector_score"]["test_auc"]
    if learned_auc is None or phi_auc is None or mean_auc is None:
        return _refuse("held-out AUC undefined (a class is absent from the test pool)")
    if learned_auc < max(phi_auc, mean_auc) + GATE_AUC_MARGIN or learned_auc <= 0.5:
        return _refuse(
            f"measured no gain: learned test AUC {learned_auc:.4f} does not beat "
            f"phi-reference head {phi_auc:.4f} / mean detector score {mean_auc:.4f} "
            f"by the {GATE_AUC_MARGIN} margin"
        )
    if best_metrics["fused_std"] < COLLAPSE_FLOOR * best_metrics["member_std"]:
        return _refuse(
            f"representational collapse: fused std {best_metrics['fused_std']:.2f} < "
            f"{COLLAPSE_FLOOR} x member std {best_metrics['member_std']:.2f}"
        )
    if rule_test["n_demoted"] < 1:
        return _refuse("the demotion rule never fires on held-out calls (inert channel)")
    if rule_test["enrichment"] <= 1.0:
        return _refuse(
            f"demotions are not enriched in wrong verdicts on test "
            f"(enrichment {rule_test['enrichment']:.2f} <= 1)"
        )
    if rule_test["correct_retention"] < TEST_RETENTION_FLOOR:
        return _refuse(
            f"correct-verdict retention {rule_test['correct_retention']:.3f} below the "
            f"{TEST_RETENTION_FLOOR} floor (deferral cost too high)"
        )

    payload = {
        "projection": best_model.projection.state_dict(),
        "attention": best_model.attention.state_dict(),
        "output_projection": best_model.output_projection.state_dict(),
        "detection_head": best_model.detection_head.state_dict(),
        "decision_thresholds": {
            "demote_act_below": act_below,
            "demote_clear_above": clear_above,
        },
        "objective": best_name,
        "metrics": {k: v for k, v in best_metrics.items() if not isinstance(v, dict)},
    }
    torch.save(payload, CHECKPOINT_PATH)
    sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    verdict["consequential"] = {
        "shipped": True,
        "reason": (
            f"learned test AUC {learned_auc:.4f} beats phi-reference head "
            f"{phi_auc:.4f} and mean detector score {mean_auc:.4f} by >= "
            f"{GATE_AUC_MARGIN}; demotion rule enrichment "
            f"{rule_test['enrichment']:.2f} with retention "
            f"{rule_test['correct_retention']:.3f}"
        ),
    }
    verdict["decision"] = (
        f"SHIPPED candidate '{best_name}' (consequential: detection head + "
        f"decision thresholds) -> {CHECKPOINT_PATH.name}"
    )
    provenance = {
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_sha256": sha256,
        "created_utc": generated_utc,
        "objective": best_name,
        "training": verdict,
        "data_sources": [
            {
                "description": "labelled production fuse() inputs harvested from a real "
                "OmniMercuryEngine detecting on cached real ADBench splits "
                "(labels are the ADBench ground truth of the exact detected rows)",
                "cache_files": sorted(p.name for p in sorted(CACHE.glob("*.npz"))[:n_datasets]),
                "cache_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(CACHE.glob("*.npz"))[:n_datasets]
                },
            }
        ],
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_PATH.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    logger.info("shipped %s (sha256 %s); eval written to %s", best_name, sha256[:12], EVAL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train MultiHeadAttentionFusion on real harvested scalar-state trajectories.

Non-circular objective
======================
The GOSNN attention fusion has no in-repo ground truth (its consumers are
internal enhancement heuristics), so the only defensible supervision is
**observed data**: the dimensional-state lists the production ``fuse()`` call
actually receives while a real engine detects on real ADBench windows.  The
component's measurable job becomes *faithful fusion*: given a state list with
one member masked, the fused output must reconstruct the masked member better
than the deterministic phi-weighted reference average does on identical
held-out lists.  The target is measured system data, never a rule's output —
training against an in-repo rule (e.g. sigma_Immutable labels) would merely
teach the network to mimic a decision the system already computes.

Multi-candidate merit gate (hazard_training pattern)
====================================================
Two candidates are trained and judged on the same held-out task:

* ``masked_recon`` — pure masked-member reconstruction (denoising fusion).
* ``multitask``    — masked reconstruction + next-call base-state prediction
  (temporal regularisation over the harvested call sequence).

Constraints (all must hold for a candidate to ship):

1. held-out masked-member MSE  <=  ``GATE_MARGIN`` x reference MSE, where the
   reference is the production numpy phi-weighted average of the unmasked
   members — the exact behaviour a default (untrained) ``fuse()`` uses;
2. no representational collapse: fused-output std >= 20% of member std.

"Reference wins, nothing ships" is a valid outcome (schumann precedent); the
eval artifact is committed either way.

Run:
    python scripts/train_gosnn_fusion.py            # full harvest + train + gate
    python scripts/train_gosnn_fusion.py --quick    # smaller harvest (smoke)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train_gosnn_fusion")

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "cache" / "adbench"
CHECKPOINT_PATH = (
    REPO / "src" / "omni_mercury_engine" / "models" / "checkpoints" / "gosnn_attention_fusion.pt"
)
PROVENANCE_PATH = CHECKPOINT_PATH.with_suffix("").with_suffix(".provenance.json")
EVAL_PATH = REPO / "artifacts" / "gosnn_fusion.eval.json"

SEED = 20260714
GATE_MARGIN = 0.95  # learned must beat reference MSE by >= 5%
COLLAPSE_FLOOR = 0.20
MAX_DIMS = 37


def _harvest(n_datasets: int, n_fit_rows: int, n_detect_rows: int) -> list[list[np.ndarray]]:
    """Record production ``fuse()`` inputs from a real engine on real data.

    A real ``OmniMercuryEngine`` is fitted on ADBench training rows and run
    over held-out rows; every dimensional-state list the GOSNN attention
    fusion receives during those detections is recorded verbatim by wrapping
    the instance's ``fuse`` — the harvested distribution IS the production
    input distribution, not a synthetic imitation.
    """
    from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network
    from omni_mercury_engine.engine import OmniMercuryEngine

    recorded: list[list[np.ndarray]] = []
    net = get_global_scalar_network()
    fusion = net.attention_fusion
    original_fuse = fusion.fuse

    def recording_fuse(dimensional_states: list[np.ndarray], *args: Any, **kwargs: Any) -> Any:
        recorded.append([np.asarray(s, dtype=np.float64).copy() for s in dimensional_states])
        return original_fuse(dimensional_states, *args, **kwargs)

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
            fit_idx = rng.choice(len(x_train), size=min(n_fit_rows, len(x_train)), replace=False)
            engine.fit_fusion(x_train[fit_idx], y_train[fit_idx])
            det_idx = rng.choice(len(x_test), size=min(n_detect_rows, len(x_test)), replace=False)
            # One row per call: the GOSNN fusion runs once per detect call,
            # so the harvest volume is the call count, not the row count.
            for i in det_idx:
                engine.detect_with_fusion(x_test[i : i + 1])
            logger.info("harvested %s: %d state lists so far", path.name, len(recorded))
    finally:
        fusion.fuse = original_fuse  # type: ignore[method-assign]
    return recorded


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


class _FusionModel(nn.Module):
    """The exact production fuse() module stack, trainable."""

    def __init__(self, d_model: int = 512, num_heads: int = 32) -> None:
        super().__init__()
        self.projection = nn.Linear(MAX_DIMS, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.output_projection = nn.Linear(d_model, MAX_DIMS)

    def forward(self, members: torch.Tensor) -> torch.Tensor:
        projected = self.projection(members).unsqueeze(0)
        attended, _ = self.attention(projected, projected, projected)
        return self.output_projection(attended.squeeze(0)).mean(dim=0)  # type: ignore[no-any-return, unused-ignore]


def _examples(
    lists: list[list[np.ndarray]], rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """(unmasked members, masked target, next-call base state or None) triples."""
    out = []
    for idx, states in enumerate(lists):
        padded = _pad(states)
        if len(padded) < 2:
            continue
        j = int(rng.integers(len(padded)))
        rest = np.delete(padded, j, axis=0)
        nxt = _pad(lists[idx + 1])[0] if idx + 1 < len(lists) else None
        out.append((rest, padded[j], nxt))
    return out


def _train_candidate(
    name: str,
    train: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    val: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    epochs: int,
) -> tuple[_FusionModel, float]:
    """Train one candidate; return (best model state applied, best val MSE)."""
    torch.manual_seed(SEED)
    model = _FusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    aux_weight = 0.3 if name == "multitask" else 0.0
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for rest, target, nxt in train:
            optimizer.zero_grad()
            fused = model(torch.tensor(rest, dtype=torch.float32))
            loss = nn.functional.mse_loss(fused, torch.tensor(target, dtype=torch.float32))
            if aux_weight and nxt is not None:
                loss = loss + aux_weight * nn.functional.mse_loss(
                    fused, torch.tensor(nxt, dtype=torch.float32)
                )
            loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
            optimizer.step()
        model.eval()
        with torch.no_grad():
            v = float(
                np.mean(
                    [
                        nn.functional.mse_loss(
                            model(torch.tensor(rest, dtype=torch.float32)),
                            torch.tensor(target, dtype=torch.float32),
                        ).item()
                        for rest, target, _ in val
                    ]
                )
            )
        if v < best_val:
            best_val, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
        logger.info("%s epoch %d: val MSE %.6f (best %.6f)", name, epoch + 1, v, best_val)
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best_val


def _test_metrics(
    model: _FusionModel, test: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]]
) -> dict[str, float]:
    """Held-out learned vs reference MSE + collapse diagnostics."""
    model.eval()
    learned_se, reference_se, fused_stds, member_stds = [], [], [], []
    with torch.no_grad():
        for rest, target, _ in test:
            fused = model(torch.tensor(rest, dtype=torch.float32)).numpy()
            learned_se.append(float(np.mean((fused - target) ** 2)))
            reference_se.append(float(np.mean((_phi_reference(rest) - target) ** 2)))
            fused_stds.append(float(np.std(fused)))
            member_stds.append(float(np.std(rest)))
    return {
        "learned_mse": float(np.mean(learned_se)),
        "reference_mse": float(np.mean(reference_se)),
        "fused_std": float(np.mean(fused_stds)),
        "member_std": float(np.mean(member_stds)),
        "n_test": float(len(test)),
    }


def main() -> int:
    """Harvest, train both candidates, merit-gate, ship the winner or refuse."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller harvest/epochs (smoke)")
    args = parser.parse_args()
    n_datasets, n_fit, n_detect, epochs = (2, 200, 60, 4) if args.quick else (3, 400, 150, 12)

    lists = _harvest(n_datasets, n_fit, n_detect)
    logger.info("harvested %d production state lists", len(lists))
    if len(lists) < 60:
        raise RuntimeError(f"harvest too small to split honestly ({len(lists)} lists)")

    rng = np.random.default_rng(SEED)
    examples = _examples(lists, rng)
    n = len(examples)
    train, val, test = (
        examples[: int(n * 0.7)],
        examples[int(n * 0.7) : int(n * 0.85)],
        examples[int(n * 0.85) :],
    )

    results: dict[str, dict[str, float]] = {}
    models: dict[str, _FusionModel] = {}
    for name in ("masked_recon", "multitask"):
        model, best_val = _train_candidate(name, train, val, epochs)
        metrics = _test_metrics(model, test)
        metrics["best_val_mse"] = best_val
        results[name] = metrics
        models[name] = model
        logger.info("%s: %s", name, metrics)

    def _passes(m: dict[str, float]) -> bool:
        return (
            m["learned_mse"] <= GATE_MARGIN * m["reference_mse"]
            and m["fused_std"] >= COLLAPSE_FLOOR * m["member_std"]
        )

    winners = [k for k in results if _passes(results[k])]
    verdict: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "gate": {
            "mse_margin": GATE_MARGIN,
            "collapse_floor": COLLAPSE_FLOOR,
            "constraint": (
                "held-out masked-member MSE <= margin x phi-reference MSE on "
                "identical production-harvested state lists; fused std >= "
                "floor x member std"
            ),
        },
        "harvest": {"n_state_lists": len(lists), "n_examples": n, "quick": bool(args.quick)},
        "candidates": results,
        "winners": winners,
    }
    if not winners:
        verdict["decision"] = (
            "REFERENCE WINS - no candidate beat the phi-weighted average by the "
            "required margin on held-out production state lists; nothing ships "
            "and fuse() keeps the deterministic reference (schumann precedent)."
        )
        EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVAL_PATH.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
        logger.info("reference wins; eval written to %s", EVAL_PATH)
        return 0

    best = min(winners, key=lambda k: results[k]["learned_mse"])
    model = models[best]
    payload = {
        "projection": model.projection.state_dict(),
        "attention": model.attention.state_dict(),
        "output_projection": model.output_projection.state_dict(),
        "objective": best,
        "metrics": results[best],
    }
    torch.save(payload, CHECKPOINT_PATH)
    sha256 = hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()
    verdict["decision"] = f"SHIPPED candidate '{best}' -> {CHECKPOINT_PATH.name}"
    provenance = {
        "checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_sha256": sha256,
        "created_utc": verdict["generated_utc"],
        "objective": best,
        "training": verdict,
        "data_sources": [
            {
                "description": "production fuse() inputs harvested from a real "
                "OmniMercuryEngine detecting on cached real ADBench splits",
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
    logger.info("shipped %s (sha256 %s); eval written to %s", best, sha256[:12], EVAL_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

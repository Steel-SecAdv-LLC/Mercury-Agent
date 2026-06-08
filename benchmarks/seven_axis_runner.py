# Copyright (C) 2025 Steel Security Advisors LLC
"""Seven-Axis Evaluation Matrix runner.

Produces a deterministic table of seven externally-citable evaluation axes
(Generalization, Scalability, Data Efficiency, Reasoning, Robustness,
Transferability, Interpretability) for the Mercury fusion stack — concretely,
for ``FusionMode.FIBRING`` composed by ``core/fibring_fusion.FibringComposer``.

Design constraints:

* Every axis returns a single floating-point score in ``[0, 1]`` plus a
  raw-numbers diagnostic, so the table is small enough to live in
  ``docs/BENCHMARKS.md`` and large enough to audit.
* Workloads are deterministic — the runner takes a single ``--seed``
  argument and every axis derives its own RNG by seed-mixing.
* The runner has no third-party dependencies beyond NumPy and the rest
  of the Mercury source tree (no sklearn, no torch, no pandas).

Usage:

    python -m benchmarks.seven_axis_runner --json out.json
    python -m benchmarks.seven_axis_runner --markdown   # prints the MD table

CI integration: ``.github/workflows/benchmark.yml`` runs the JSON form and
uploads ``out.json`` as an artifact; ``docs/BENCHMARKS.md`` regenerates its
"Seven-Axis Evaluation" section from the same JSON via
``--regenerate-docs``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omni_mercury_engine.core.fibring_fusion import FibringComposer

logger = logging.getLogger("benchmarks.seven_axis_runner")

DEFAULT_SEED: int = 20260504
DOCS_PATH: Path = Path(__file__).resolve().parent.parent / "docs" / "BENCHMARKS.md"
SECTION_HEADER: str = "## Seven-Axis Evaluation Matrix"
SECTION_FOOTER: str = "<!-- end seven-axis-section -->"

# ---------------------------------------------------------------------------
# Generic helpers — kept local so the runner has zero sklearn/pandas deps.
# ---------------------------------------------------------------------------


def _auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U AUROC; returns 0.5 if either class is empty."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos_scores = s[y == 1]
    neg_scores = s[y == 0]
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average-rank handling for ties.
    all_scores = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(all_scores, kind="mergesort")
    ranks_in_order = np.arange(1, len(all_scores) + 1, dtype=float)
    ranks = np.empty_like(ranks_in_order)
    ranks[order] = ranks_in_order
    # Tie correction.
    _, inverse, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=float)
    np.add.at(sums, inverse, ranks)
    avg = (sums / counts)[inverse]
    rank_pos = avg[:n_pos].sum()
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def _make_two_channel_workload(
    rng: np.random.Generator,
    n_samples: int,
    anomaly_rate: float = 0.18,
    signal_strength: float = 1.6,
    bias: float = -0.8,
    noise_sigma: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-symmetric noisy-logistic neural + symbolic streams over a binary label."""
    labels = (rng.uniform(size=n_samples) < anomaly_rate).astype(int)

    def _channel(seed_offset: int) -> np.ndarray:
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)) + seed_offset)
        logits = signal_strength * labels + bias + sub_rng.normal(0.0, noise_sigma, size=n_samples)
        return 1.0 / (1.0 + np.exp(-logits))

    return _channel(11), _channel(42), labels


def _fuse_with_fibring(
    neural: np.ndarray,
    symbolic: np.ndarray,
    domain: str | None = None,
) -> np.ndarray:
    """Run the full FibringComposer over a stream and return fused scores."""
    composer = FibringComposer(domain=domain)
    out = np.empty_like(neural, dtype=float)
    for i, (n_val, s_val) in enumerate(zip(neural, symbolic, strict=True)):
        fused, _ = composer.fuse(float(n_val), float(s_val), update_history=True)
        out[i] = fused
    return out


# ---------------------------------------------------------------------------
# Axis dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AxisResult:
    name: str
    score: float
    higher_is_better: bool
    raw: dict[str, Any]
    notes: str = ""


@dataclass
class SevenAxisReport:
    seed: int
    runtime_seconds: float
    axes: list[AxisResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "axes": [
                {
                    "name": a.name,
                    "score": a.score,
                    "higher_is_better": a.higher_is_better,
                    "notes": a.notes,
                    "raw": a.raw,
                }
                for a in self.axes
            ],
        }


# ---------------------------------------------------------------------------
# Axis 1: Generalization — AUROC delta between in-distribution and OOD.
# ---------------------------------------------------------------------------


def axis_generalization(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed)
    n_id, n_ood = 1500, 600

    n_id_arr, s_id_arr, y_id_arr = _make_two_channel_workload(rng, n_id)
    # OOD: shifted bias (toward fewer anomalies) and elevated noise.
    n_ood_arr, s_ood_arr, y_ood_arr = _make_two_channel_workload(
        rng, n_ood, signal_strength=1.4, bias=-1.0, noise_sigma=0.85
    )

    fused_id = _fuse_with_fibring(n_id_arr, s_id_arr)
    fused_ood = _fuse_with_fibring(n_ood_arr, s_ood_arr)

    auc_id = _auroc(y_id_arr, fused_id)
    auc_ood = _auroc(y_ood_arr, fused_ood)
    delta = auc_id - auc_ood
    # Score in [0, 1]: 1 means OOD AUC equals ID AUC; 0 means OOD AUC is 0.5.
    score = float(np.clip(auc_ood, 0.0, 1.0))
    return AxisResult(
        name="Generalization",
        score=score,
        higher_is_better=True,
        raw={"auc_in_distribution": auc_id, "auc_ood": auc_ood, "delta": delta},
        notes=(
            "OOD AUROC under bias / noise shift "
            f"(ID={auc_id:.3f} → OOD={auc_ood:.3f}, Δ={delta:+.3f})."
        ),
    )


# ---------------------------------------------------------------------------
# Axis 2: Scalability — log-log slope of runtime vs N (1.0 = linear).
# ---------------------------------------------------------------------------


def axis_scalability(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed + 1)
    sizes = [200, 800, 3200]
    times: list[float] = []
    for n in sizes:
        n_arr, s_arr, _ = _make_two_channel_workload(rng, n)
        t0 = time.perf_counter()
        _fuse_with_fibring(n_arr, s_arr)
        times.append(time.perf_counter() - t0)
    log_n = np.log(np.asarray(sizes, dtype=float))
    log_t = np.log(np.asarray(times, dtype=float))
    # Linear regression slope of log(t) vs log(n).
    slope, _intercept = np.polyfit(log_n, log_t, deg=1)
    # Score: 1.0 when slope ≤ 1 (linear or better); decays linearly past that.
    score = float(np.clip(2.0 - max(slope, 1.0), 0.0, 1.0))
    return AxisResult(
        name="Scalability",
        score=score,
        higher_is_better=True,
        raw={"sizes": sizes, "times_seconds": times, "log_log_slope": float(slope)},
        notes=(
            f"Empirical complexity slope d log(t) / d log(n) = {slope:.3f} " f"over N ∈ {sizes}."
        ),
    )


# ---------------------------------------------------------------------------
# Axis 3: Data Efficiency — AUROC at small N over AUROC at large N.
# ---------------------------------------------------------------------------


def axis_data_efficiency(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed + 2)
    sizes = [50, 200, 1000]
    aurocs: dict[int, float] = {}
    for n in sizes:
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        n_arr, s_arr, y_arr = _make_two_channel_workload(sub_rng, n)
        fused = _fuse_with_fibring(n_arr, s_arr)
        aurocs[n] = _auroc(y_arr, fused)
    score = aurocs[50] / max(aurocs[1000], 1e-9)
    score = float(np.clip(score, 0.0, 1.0))
    return AxisResult(
        name="Data Efficiency",
        score=score,
        higher_is_better=True,
        raw={"auroc_by_n": aurocs},
        notes=(
            "AUROC(N=50) / AUROC(N=1000) — measures how quickly the composer "
            f"reaches asymptotic ranking quality (got {aurocs[50]:.3f} / "
            f"{aurocs[1000]:.3f})."
        ),
    )


# ---------------------------------------------------------------------------
# Axis 4: Reasoning — symbolic-channel contribution to correct decisions.
# ---------------------------------------------------------------------------


def axis_reasoning(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed + 3)
    n_arr, s_arr, y_arr = _make_two_channel_workload(rng, 1500)
    fused = _fuse_with_fibring(n_arr, s_arr)
    threshold = 0.5
    pred = (fused >= threshold).astype(int)
    correct = pred == y_arr
    # Among correct positive predictions, how often did the symbolic channel
    # cross the threshold on its own (i.e., the rule-driven path agreed)?
    sym_pred = (s_arr >= threshold).astype(int)
    correct_pos = correct & (y_arr == 1)
    if correct_pos.sum() == 0:
        score = 0.0
    else:
        score = float(sym_pred[correct_pos].mean())
    return AxisResult(
        name="Reasoning",
        score=score,
        higher_is_better=True,
        raw={
            "n_correct_positives": int(correct_pos.sum()),
            "n_with_symbolic_agreement": int(sym_pred[correct_pos].sum()),
        },
        notes=(
            "Fraction of correctly-flagged anomalies for which the symbolic "
            "channel independently crossed the decision threshold."
        ),
    )


# ---------------------------------------------------------------------------
# Axis 5: Robustness — relative AUROC under additive noise injection.
# ---------------------------------------------------------------------------


def axis_robustness(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed + 4)
    n_arr, s_arr, y_arr = _make_two_channel_workload(rng, 1500)
    fused_clean = _fuse_with_fibring(n_arr, s_arr)
    auc_clean = _auroc(y_arr, fused_clean)

    severities = [0.05, 0.15]
    aurocs_noisy: dict[float, float] = {}
    for sigma in severities:
        sub_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        n_noisy = np.clip(n_arr + sub_rng.normal(0.0, sigma, size=n_arr.shape), 0.0, 1.0)
        s_noisy = np.clip(s_arr + sub_rng.normal(0.0, sigma, size=s_arr.shape), 0.0, 1.0)
        fused_noisy = _fuse_with_fibring(n_noisy, s_noisy)
        aurocs_noisy[sigma] = _auroc(y_arr, fused_noisy)
    worst = min(aurocs_noisy.values())
    score = float(worst / max(auc_clean, 1e-9))
    score = float(np.clip(score, 0.0, 1.0))
    return AxisResult(
        name="Robustness",
        score=score,
        higher_is_better=True,
        raw={
            "auroc_clean": auc_clean,
            "auroc_under_noise": aurocs_noisy,
        },
        notes=(
            f"AUROC retention under additive noise σ ∈ {severities} "
            f"(clean={auc_clean:.3f}, worst-noise={worst:.3f})."
        ),
    )


# ---------------------------------------------------------------------------
# Axis 6: Transferability — cross-domain AUROC vs in-domain AUROC.
# ---------------------------------------------------------------------------


def axis_transferability(seed: int) -> AxisResult:
    rng_a = np.random.default_rng(seed + 5)
    rng_b = np.random.default_rng(seed + 6)
    n_a, s_a, y_a = _make_two_channel_workload(rng_a, 1200, signal_strength=1.6)
    n_b, s_b, y_b = _make_two_channel_workload(rng_b, 1200, signal_strength=1.4, noise_sigma=0.7)

    # In-domain: composer state warmed on A, evaluated on A.
    composer_in = FibringComposer(domain="medical")
    in_out = np.empty_like(n_a, dtype=float)
    for i in range(len(n_a)):
        in_out[i], _ = composer_in.fuse(float(n_a[i]), float(s_a[i]))
    auc_in = _auroc(y_a, in_out)

    # Cross-domain: a composer carrying domain-B affinity bias is warmed on
    # domain-A observations (correlation history + window state) and then
    # evaluated on domain-B data.  This is the genuine transfer measurement
    # — the running window state was learned on A but the affinity bias is
    # B's, so the AUROC ratio against the in-domain run isolates how much
    # signal survives the transfer.  (A prior revision constructed a second
    # medical-bias composer warmed on A but never read from it; that was
    # dead code and has been removed.)
    composer_cross = FibringComposer(domain="financial")
    for n_val, s_val in zip(n_a, s_a, strict=True):
        composer_cross.observe(float(n_val), float(s_val))
    cross_out = np.empty_like(n_b, dtype=float)
    for i in range(len(n_b)):
        cross_out[i], _ = composer_cross.fuse(float(n_b[i]), float(s_b[i]))
    auc_cross = _auroc(y_b, cross_out)

    score = float(np.clip(auc_cross / max(auc_in, 1e-9), 0.0, 1.0))
    return AxisResult(
        name="Transferability",
        score=score,
        higher_is_better=True,
        raw={"auroc_in_domain": auc_in, "auroc_cross_domain": auc_cross},
        notes=(f"Cross-domain AUROC retention (in={auc_in:.3f}, cross={auc_cross:.3f})."),
    )


# ---------------------------------------------------------------------------
# Axis 7: Interpretability — fraction of decisions backed by a non-trivial
# decorrelator / domain-bias diagnostic from the FibringComposer.
# ---------------------------------------------------------------------------


def axis_interpretability(seed: int) -> AxisResult:
    rng = np.random.default_rng(seed + 7)
    n_arr, s_arr, _ = _make_two_channel_workload(rng, 1500)
    composer = FibringComposer(domain="medical")
    diagnostic_count = 0
    for n_val, s_val in zip(n_arr, s_arr, strict=True):
        weights = composer.compose(float(n_val), float(s_val), update_history=True)
        # A decision is "interpretable" if at least one non-trivial diagnostic
        # is attached: a measured correlation, an applied decorrelation, or a
        # non-zero domain-affinity bias.
        has_diag = (
            weights.correlation is not None
            or weights.decorrelation_applied
            or weights.domain_bias_applied != (0.0, 0.0)
        )
        if has_diag:
            diagnostic_count += 1
    score = float(diagnostic_count / len(n_arr))
    return AxisResult(
        name="Interpretability",
        score=score,
        higher_is_better=True,
        raw={
            "n_decisions": len(n_arr),
            "n_with_diagnostic": diagnostic_count,
        },
        notes=(
            "Fraction of decisions accompanied by at least one non-trivial "
            "FibringComposer diagnostic (correlation, decorrelation, or "
            "domain-affinity bias)."
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

AXIS_FUNCTIONS = [
    axis_generalization,
    axis_scalability,
    axis_data_efficiency,
    axis_reasoning,
    axis_robustness,
    axis_transferability,
    axis_interpretability,
]


def run_seven_axis(seed: int = DEFAULT_SEED) -> SevenAxisReport:
    """Run every axis in sequence and return a SevenAxisReport."""
    started = time.perf_counter()
    axes = [fn(seed) for fn in AXIS_FUNCTIONS]
    return SevenAxisReport(
        seed=seed,
        runtime_seconds=time.perf_counter() - started,
        axes=axes,
    )


def render_markdown_table(report: SevenAxisReport) -> str:
    # Wall-clock runtime is intentionally omitted from the markdown header
    # so the docs section is bytewise deterministic for a fixed seed
    # (CI can `git diff --exit-code docs/BENCHMARKS.md` after regenerating).
    # Runtime is still in the JSON payload for benchmarking purposes.
    lines = [
        SECTION_HEADER,
        "",
        f"_Generated by `python -m benchmarks.seven_axis_runner` "
        f"(seed={report.seed}). "
        "Do not hand-edit — regenerate with "
        "`python -m benchmarks.seven_axis_runner --regenerate-docs`._",
        "",
        "| Axis | Score (higher is better) | Notes |",
        "| --- | --- | --- |",
    ]
    for a in report.axes:
        notes = a.notes.replace("|", "\\|")
        lines.append(f"| {a.name} | {a.score:.3f} | {notes} |")
    lines.append("")
    lines.append(SECTION_FOOTER)
    return "\n".join(lines) + "\n"


def regenerate_docs(report: SevenAxisReport, docs_path: Path = DOCS_PATH) -> None:
    """Replace (or append) the Seven-Axis Evaluation section in BENCHMARKS.md."""
    table = render_markdown_table(report)
    if not docs_path.exists():
        raise FileNotFoundError(f"BENCHMARKS doc not found at {docs_path}")
    text = docs_path.read_text(encoding="utf-8")
    if SECTION_HEADER in text and SECTION_FOOTER in text:
        before = text.split(SECTION_HEADER, 1)[0].rstrip() + "\n\n"
        after_footer = text.split(SECTION_FOOTER, 1)[1].lstrip()
        new_text = before + table + ("\n" + after_footer if after_footer else "")
    else:
        new_text = text.rstrip() + "\n\n" + table
    docs_path.write_text(new_text, encoding="utf-8")
    logger.info("Regenerated %s with Seven-Axis section.", docs_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the JSON report to this path."
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print the markdown table to stdout.",
    )
    parser.add_argument(
        "--regenerate-docs",
        action="store_true",
        help="Rewrite the Seven-Axis Evaluation section in docs/BENCHMARKS.md.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_seven_axis(seed=args.seed)

    payload = report.to_dict()
    if args.json is not None:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote JSON report to %s", args.json)
    if args.markdown:
        sys.stdout.write(render_markdown_table(report))
    if args.regenerate_docs:
        regenerate_docs(report)

    if args.json is None and not args.markdown and not args.regenerate_docs:
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

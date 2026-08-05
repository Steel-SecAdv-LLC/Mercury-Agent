#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the weapons gate on the held-out adversarial slice.

Reports the confusion matrix, precision/recall/F1, and the Brier score of the
gate's offensive-confidence, both overall and per adversarial axis (paraphrase /
conjunction / obfuscation / out_of_lexicon / hard_benign). Unlike the base-corpus
eval, this is a *generalization* test -- the slice is disjoint from the fit-on
corpus -- so a nonzero FN rate here is the transparent residual, not a regression.

    PYTHONPATH=src:benchmarks python benchmarks/eval_weapons_gate_adversarial.py [--posture default|classifier] [--dump]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from weapons_gate_adversarial import build_adversarial_corpus, rows_as_dicts

from omni_mercury_engine.cognitive.ethical_bounding import assess_weapons_uplift

_CORPUS_JSONL = Path(__file__).resolve().parent / "weapons_gate_adversarial.jsonl"


@dataclass
class AxisMetrics:
    """Confusion matrix + Brier accumulator for one axis (or overall)."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    brier_sum: float = 0.0
    fp_examples: tuple[str, ...] = ()
    fn_examples: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """Total examples."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def precision(self) -> float:
        """TP / (TP + FP); 1.0 when nothing was flagged."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 1.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN); 1.0 when there were no offensive examples."""
        pos = self.tp + self.fn
        return self.tp / pos if pos else 1.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fp_rate(self) -> float:
        """Benign wrongly blocked / all benign."""
        benign = self.tn + self.fp
        return self.fp / benign if benign else 0.0

    @property
    def fn_rate(self) -> float:
        """Offensive wrongly allowed / all offensive."""
        pos = self.tp + self.fn
        return self.fn / pos if pos else 0.0

    @property
    def brier(self) -> float:
        """Mean squared error of the gate's offensive-confidence vs the label."""
        return self.brier_sum / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-safe summary."""
        return {
            "n": self.n,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fp_rate": round(self.fp_rate, 4),
            "fn_rate": round(self.fn_rate, 4),
            "brier": round(self.brier, 4),
            "fp_examples": list(self.fp_examples[:10]),
            "fn_examples": list(self.fn_examples[:10]),
        }


def evaluate(
    *,
    use_classifier: bool = False,
    classifier: object | None = None,
) -> dict[str, dict[str, AxisMetrics] | AxisMetrics]:
    """Run the gate over the adversarial slice; return metrics overall + stratified.

    The return dict has three kinds of top-level entries:

    * ``"overall"`` -- an :class:`AxisMetrics` over the whole slice.
    * one :class:`AxisMetrics` per evasion axis (``"paraphrase"``,
      ``"conjunction"``, ``"obfuscation"``, ``"out_of_lexicon"``,
      ``"hard_benign"``) -- the existing per-axis breakdown.
    * ``"by_domain"`` -- a ``dict[str, AxisMetrics]`` keyed by
      ``row.tags[0]`` (hazard domain), so results can be broken down by
      domain (e.g. "is the gate specifically weak on radiological
      paraphrase evasions vs chemical ones") independent of axis.
    * ``"by_domain_axis"`` -- a ``dict[str, AxisMetrics]`` keyed by
      ``"{domain}/{axis}"``, the joint breakdown.

    Postures:

    * ``classifier=None`` (default) -- the **default posture** (no meaning-level
      model). This is what CI and air-gapped deployments run, so it is the FP
      gate of record and the transparent lexical-only FN floor.
    * ``classifier=<callable>`` -- inject a real ``Callable[[str], float]``
      meaning-level classifier (e.g. ``default_harm_classifier()`` with a real
      model serving). This is the measurement that marks "meaning-level coverage
      met" -- FN cut by the routing rescue while FP stays 0.
    * ``use_classifier=True`` -- a permissive *constant* stand-in (returns 1.0).
      This is an FN-**reachability** probe only: it proves the routing rescue
      wiring reaches the routing-miss cases. It is NOT an FP measurement -- a
      constant classifier has no discrimination and will over-escalate benign
      mechanistic queries, so its hard-benign FP is a stub artifact, not a real
      false positive (a real model scores those low).

    ``classifier`` takes precedence over ``use_classifier`` when both are given.
    """
    if classifier is None and use_classifier:
        classifier = lambda _text: 1.0  # noqa: E731 - reachability probe stand-in
    by_axis: dict[str, dict[str, list[str] | int | float]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "brier": 0.0, "fp_ex": [], "fn_ex": []}
    )
    by_domain: dict[str, dict[str, list[str] | int | float]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "brier": 0.0, "fp_ex": [], "fn_ex": []}
    )
    by_domain_axis: dict[str, dict[str, list[str] | int | float]] = defaultdict(
        lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "brier": 0.0, "fp_ex": [], "fn_ex": []}
    )
    overall: dict[str, list[str] | int | float] = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "brier": 0.0,
        "fp_ex": [],
        "fn_ex": [],
    }

    for row in build_adversarial_corpus():
        assessment = assess_weapons_uplift(row.text, harm_classifier=classifier)
        blocks = assessment.blocks
        p = float(assessment.confidence)  # gate's offensive probability
        y = 1.0 if row.label == "offensive" else 0.0
        want_block = row.expected == "block"
        domain = row.tags[0]
        domain_axis = f"{domain}/{row.axis}"

        for bucket in (
            by_axis[row.axis],
            by_domain[domain],
            by_domain_axis[domain_axis],
            overall,
        ):
            bucket["brier"] = float(bucket["brier"]) + (p - y) ** 2  # type: ignore[arg-type]
            if want_block and blocks:
                bucket["tp"] = int(bucket["tp"]) + 1  # type: ignore[arg-type]
            elif want_block and not blocks:
                bucket["fn"] = int(bucket["fn"]) + 1  # type: ignore[arg-type]
                bucket["fn_ex"].append(row.text)  # type: ignore[union-attr]
            elif not want_block and blocks:
                bucket["fp"] = int(bucket["fp"]) + 1  # type: ignore[arg-type]
                bucket["fp_ex"].append(row.text)  # type: ignore[union-attr]
            else:
                bucket["tn"] = int(bucket["tn"]) + 1  # type: ignore[arg-type]

    def finalize(b: dict[str, list[str] | int | float]) -> AxisMetrics:
        return AxisMetrics(
            tp=int(b["tp"]),  # type: ignore[arg-type]
            tn=int(b["tn"]),  # type: ignore[arg-type]
            fp=int(b["fp"]),  # type: ignore[arg-type]
            fn=int(b["fn"]),  # type: ignore[arg-type]
            brier_sum=float(b["brier"]),  # type: ignore[arg-type]
            fp_examples=tuple(b["fp_ex"]),  # type: ignore[arg-type]
            fn_examples=tuple(b["fn_ex"]),  # type: ignore[arg-type]
        )

    result: dict[str, dict[str, AxisMetrics] | AxisMetrics] = {"overall": finalize(overall)}
    for axis, b in sorted(by_axis.items()):
        result[axis] = finalize(b)
    result["by_domain"] = {domain: finalize(b) for domain, b in sorted(by_domain.items())}
    result["by_domain_axis"] = {
        domain_axis: finalize(b) for domain_axis, b in sorted(by_domain_axis.items())
    }
    return result


def dump_corpus() -> Path:
    """(Re)write the shipped JSONL slice and return its path."""
    with _CORPUS_JSONL.open("w", encoding="utf-8") as fh:
        for row in rows_as_dicts():
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return _CORPUS_JSONL


def main() -> None:
    """CLI: evaluate the adversarial slice and optionally dump the corpus."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--posture",
        default="default",
        choices=["default", "shipped", "classifier"],
        help=(
            "default: lexical-only floor (no meaning-level model). "
            "shipped: the shipped offline classifier -- this is the posture the "
            "published headline figures are measured in. "
            "classifier: a constant 1.0 stand-in; an FN-reachability probe only, "
            "NOT an FP measurement."
        ),
    )
    ap.add_argument("--dump", action="store_true", help="rewrite weapons_gate_adversarial.jsonl")
    args = ap.parse_args()
    if args.dump:
        print(f"wrote {dump_corpus()}")
    # The shipped posture was previously unreachable from the CLI: the published
    # "Reproduce" commands named --posture default, which measures the lexical
    # floor, so following the docs produced a number 61 false negatives away from
    # the one they printed beside it. A published figure whose published repro
    # command does not reproduce it is not evidence.
    if args.posture == "shipped":
        from omni_mercury_engine.cognitive.meaning_level import meaning_level_harm_classifier

        metrics = evaluate(classifier=meaning_level_harm_classifier())
    else:
        metrics = evaluate(use_classifier=(args.posture == "classifier"))

    def _to_json(v: dict[str, AxisMetrics] | AxisMetrics) -> object:
        if isinstance(v, AxisMetrics):
            return v.to_dict()
        return {k: m.to_dict() for k, m in v.items()}

    print(f"adversarial eval (posture={args.posture!r}):")
    print(json.dumps({k: _to_json(v) for k, v in metrics.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

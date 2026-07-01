#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the weapons-gate operating point over the labeled corpus.

Reports a real confusion matrix (TP/TN/FP/FN) and the false-positive /
false-negative *rates* on any corpus split -- the measurement the historical
paired-assertion check could not produce. ``--dump`` (re)writes the shipped
``benchmarks/weapons_gate_corpus.jsonl`` so the corpus is reviewable as data.

    PYTHONPATH=src:benchmarks python benchmarks/eval_weapons_gate.py [--split test|val|train|all] [--dump]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from weapons_gate_corpus import build_corpus, rows_as_dicts

from omni_mercury_engine.cognitive.ethical_bounding import assess_weapons_uplift

_CORPUS_JSONL = Path(__file__).resolve().parent / "weapons_gate_corpus.jsonl"


@dataclass(frozen=True)
class GateMetrics:
    """Confusion matrix and derived rates for one evaluation."""

    tp: int
    tn: int
    fp: int
    fn: int
    fp_examples: tuple[str, ...] = ()
    fn_examples: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        """Total examples evaluated."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def fp_rate(self) -> float:
        """Benign examples wrongly blocked / all benign."""
        benign = self.tn + self.fp
        return self.fp / benign if benign else 0.0

    @property
    def fn_rate(self) -> float:
        """Offensive examples wrongly allowed / all offensive."""
        offensive = self.tp + self.fn
        return self.fn / offensive if offensive else 0.0

    def to_dict(self) -> dict[str, object]:
        """JSON-safe summary (examples truncated for brevity)."""
        return {
            "n": self.n,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "fp_rate": round(self.fp_rate, 4),
            "fn_rate": round(self.fn_rate, 4),
            "fp_examples": list(self.fp_examples[:10]),
            "fn_examples": list(self.fn_examples[:10]),
        }


def evaluate(split: str = "test") -> GateMetrics:
    """Run the gate over ``split`` (or ``"all"``) and return the confusion matrix."""
    tp = tn = fp = fn = 0
    fp_ex: list[str] = []
    fn_ex: list[str] = []
    for row in build_corpus():
        if split != "all" and row.split != split:
            continue
        blocks = assess_weapons_uplift(row.text).blocks
        want_block = row.expected == "block"
        if want_block and blocks:
            tp += 1
        elif want_block and not blocks:
            fn += 1
            fn_ex.append(row.text)
        elif not want_block and blocks:
            fp += 1
            fp_ex.append(row.text)
        else:
            tn += 1
    return GateMetrics(tp, tn, fp, fn, tuple(fp_ex), tuple(fn_ex))


def dump_corpus() -> Path:
    """(Re)write the shipped JSONL corpus and return its path."""
    with _CORPUS_JSONL.open("w", encoding="utf-8") as fh:
        for row in rows_as_dicts():
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return _CORPUS_JSONL


def main() -> None:
    """CLI: evaluate a split and optionally dump the corpus."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--dump", action="store_true", help="rewrite weapons_gate_corpus.jsonl")
    args = ap.parse_args()
    if args.dump:
        print(f"wrote {dump_corpus()}")
    metrics = evaluate(args.split)
    print(f"weapons-gate eval on split={args.split!r}:")
    print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

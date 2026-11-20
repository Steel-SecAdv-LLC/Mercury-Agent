#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per hazard: is the score a probability, does the certificate hold, how often is nothing decided?

The published hazard table (README, ``research/engine_calibration``) says how
well the detector *ranks* the rows of a real event. It says nothing of what a
score of 0.8 means, of whether a "90%" certificate is right nine times in
ten, or of how often the decision layer declines to decide. This measures
those three, per hazard, on the same real events, **leave one event out**:
everything that is fitted (the calibration, the certificate) is fitted on
the hazard's other events and judged on the one it has not seen. That is
the question a deployment faces: the next event is not in the training set.

For each event held out in turn:

1. The detector the hazard table uses (``MercuryAnomalyDetector``, fitted
   unlabelled on the event itself) scores every row.
2. The other events' rows are split in two, every other row of each class.
   On one half an isotonic map from score to probability is fitted
   (``core.calibration.IsotonicCalibration``); on the other, a class-
   conditional conformal classifier at the target coverage
   (``core.conformal_prediction.BinaryConformalClassifier``). Then the
   halves change places, and the event is judged both ways: how far the two
   differ is what the choice of half is worth, and a certificate is said to
   have held or fallen short only where both agree.
3. The held-out event's rows get a probability and a label set, and the
   decision layer (``decision.decider.DecisionAbstentionResponder``, as it
   ships) decides or abstains on each.

What is reported, over the held-out rows of all the hazard's events:

* ``score``: the raw score read as a probability (Brier score, ECE).
* ``probability``: the calibrated one, against a reference that knows only
  the other events' share of anomalies. ``skill`` is ``1 - brier /
  reference brier``: above 0 the probability says more than the base rate.
* ``certificate``: how often the true label was in the set, where the
  target says nine times in ten; overall, and for anomalies and normal rows
  apart.
* ``decision``: the share of rows decided and abstained on, how often a
  decision was right, how often a call of "anomaly" was, and what became of
  the true anomalies (called, abstained on, called normal).

The events are fetched by ``research/engine_calibration/build_cache.py``
from their live sources and are not in the repository; this reads its cache
(``--cache``, or ``MERCURY_CALIBRATION_CACHE``). A hazard with one event
cannot be measured this way and is listed as such, as is one whose source
could not be read when the cache was built.

How a hazard's labels were made matters more than any number here
(``label_source``): ``ground_truth`` labels are independent of what is
scored; ``statistical`` ones are a threshold on a column, and a detector
shown that column, or one that moves with it, is partly marking its own work.

Run::

    python research/hazard_calibration/measure.py --cache DIR           # measure and print
    python research/hazard_calibration/measure.py --cache DIR --write   # and record

Nothing here is random: the same cache gives the same numbers.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence.json"
#: The copy the package ships, which the decision layer and the API read.
SHIPPED = HERE.parents[1] / "src" / "omni_mercury_engine" / "decision" / "hazard_evidence.json"
#: The coverage the certificate is asked for.
COVERAGE = 0.9
#: An event with more rows than this is read at evenly spaced rows, the same ones every run.
ROW_LIMIT = 20_000
#: The loader whose ``LABEL_SOURCE`` says how a hazard's labels were made.
LOADERS = {
    "earthquake": ("earthquake_loader", "EarthquakeLoader"),
    "tsunami": ("tsunami_loader", "TsunamiLoader"),
    "tornado": ("tornado_loader", "TornadoLoader"),
    "marine": ("marine_loader", "MarineLoader"),
    "hurricane": ("hurricane_loader", "HurricaneLoader"),
    "wildfire": ("wildfire_loader", "WildfireLoader"),
    "flood": ("flood_loader", "FloodLoader"),
    "volcanic": ("volcanic_loader", "VolcanicLoader"),
    "landslide": ("landslide_loader", "LandslideLoader"),
    "energy": ("energy_loader", "EnergyLoader"),
    "fema": ("fema_loader", "FEMALoader"),
    "fema_activity": ("fema_activity_loader", "FEMAActivityLoader"),
    "financial": ("financial_loader", "FinancialLoader"),
    "network_security": ("network_security_loader", "NetworkSecurityLoader"),
    "pandemic": ("pandemic_loader", "PandemicLoader"),
    "sepsis": ("sepsis_loader", "SepsisLoader"),
}
#: Label sources that are independent of what is scored: the values of
#: ``omni_mercury_engine.datasets.metadata.GENUINE_LABEL_SOURCES``. FEMA
#: declaration activity (``expert_annotated``: NOAA NCEI's billion-dollar
#: catalogue) is one, beside the two ``ground_truth`` hazards.
INDEPENDENT_LABELS = ("ground_truth", "expert_annotated")
Scored = dict[str, dict[str, dict[str, Any]]]


def default_cache() -> Path:
    """Where ``research/engine_calibration`` keeps the events it fetched."""
    override = os.environ.get("MERCURY_CALIBRATION_CACHE")
    if override:
        return Path(override).expanduser() / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base).expanduser() / "mercury-agent" / "engine_calibration" / "cache"


def label_source(hazard: str) -> str:
    """How the hazard's loader says its labels were made."""
    module, name = LOADERS[hazard]
    loader = getattr(importlib.import_module(f"omni_mercury_engine.loaders.{module}"), name)
    return str(loader.LABEL_SOURCE)


def score_events(cache: Path) -> Scored:
    """Each cached event's rows, scored by the detector the hazard table uses.

    Returns:
        ``{hazard: {event: {"scores", "labels", "rows", "subsampled"}}}``.
        An archive from before the label-derived columns were excluded
        (no ``schema`` in it) and one with a quarantine marker are left out,
        as ``research/engine_calibration`` leaves them out.
    """
    from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

    scored: Scored = {}
    for archive in sorted(cache.glob("*.npz")):
        if archive.name.endswith(".deleak.npz"):
            continue
        if (cache / f"{archive.stem}.quarantined.json").exists():
            continue
        held = np.load(archive)
        if "schema" not in held or int(held["schema"]) < 2:
            continue
        hazard, _, event = archive.stem.partition("__")
        rows, labels = held["X"], held["y"].astype(int)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        if labels.min() == labels.max():
            continue
        subsampled = len(labels) > ROW_LIMIT
        if subsampled:
            keep = np.linspace(0, len(labels) - 1, ROW_LIMIT).astype(int)
            rows, labels = rows[keep], labels[keep]
        detector = MercuryAnomalyDetector()
        detector.fit(rows, unlabeled_reference=True)
        scores = np.asarray(detector.detect(rows)["scores"], dtype=float).reshape(-1)
        scored.setdefault(hazard, {})[event] = {
            "scores": scores,
            "labels": labels,
            "rows": len(held["y"]),
            "subsampled": subsampled,
        }
    return scored


def _brier(p: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> float:
    return float(np.mean((p - y) ** 2))


def _share(mask: np.ndarray[Any, Any]) -> float | None:
    return round(float(mask.mean()), 4) if mask.size else None


def _every_other_of_each_class(labels: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Which rows go in the first half: every other anomaly, and every other normal row.

    Taken by class so that each half holds both whenever there are two of
    each: every other row of the whole put all the anomalies of a series
    that has one every ten rows in the same half.
    """
    first = np.zeros(len(labels), dtype=bool)
    for label in (0, 1):
        rows = np.flatnonzero(labels == label)
        first[rows[::2]] = True
    return first


def leave_one_event_out(
    events: dict[str, dict[str, Any]], coverage: float = COVERAGE
) -> dict[str, Any]:
    """One hazard's account, each event judged by what was fitted on the others.

    The others' rows are halved (:func:`_every_other_of_each_class`), and the
    event is judged twice: with the calibration fitted on the first half and
    the certificate on the second, and the other way round. The account is
    of both together, and ``by_split`` gives the certificate's coverage and
    the share abstained on each way: where the two differ, that difference
    is what the choice of half alone is worth.

    Args:
        events: ``{event: {"scores": ..., "labels": ...}}`` for one hazard.
        coverage: The coverage the certificate is asked for.

    Returns:
        The ``score``, ``probability``, ``certificate`` and ``decision``
        blocks described in this module's docstring, with ``rows``,
        ``anomalies`` and ``folds_skipped`` (events whose others' halves did
        not each hold both classes, so nothing could be fitted). With fewer
        than two events, ``{"not_measured": why}``.
    """
    from omni_mercury_engine.core.calibration import IsotonicCalibration, compute_ece
    from omni_mercury_engine.core.conformal_prediction import BinaryConformalClassifier
    from omni_mercury_engine.decision.decider import DecisionAbstentionResponder

    if len(events) < 2:
        return {"not_measured": "one event: there is no other to calibrate on"}
    decider = DecisionAbstentionResponder()
    raw, truth, reference = [], [], []
    judged: list[dict[str, list[Any]]] = [
        {"probability": [], "covered": [], "decided": [], "called": []} for _ in range(2)
    ]
    skipped: list[str] = []
    for held_out in sorted(events):
        others = [events[name] for name in sorted(events) if name != held_out]
        train_scores = np.concatenate([other["scores"] for other in others])
        train_labels = np.concatenate([other["labels"] for other in others])
        first = _every_other_of_each_class(train_labels)
        if any(len(np.unique(train_labels[half])) < 2 for half in (first, ~first)):
            skipped.append(held_out)
            continue
        scores, labels = events[held_out]["scores"], events[held_out]["labels"]
        for split, fitted_on in zip(judged, (first, ~first), strict=True):
            isotonic = IsotonicCalibration().fit(train_scores[fitted_on], train_labels[fitted_on])
            conformal = BinaryConformalClassifier(coverage=coverage).fit(
                isotonic.calibrate(train_scores[~fitted_on]), train_labels[~fitted_on]
            )
            probability = np.asarray(isotonic.calibrate(scores), dtype=float)
            sets = conformal.predict(probability).label_sets()
            for p, labels_in_set in zip(probability, sets, strict=True):
                record = decider.decide(
                    {
                        "anomaly_prob": float(p),
                        "is_anomaly": bool(p > 0.5),
                        "threshold_used": 0.5,
                        "conformal": {
                            "prediction_set": labels_in_set,
                            "set_size": len(labels_in_set),
                            "coverage": coverage,
                        },
                    }
                )
                split["decided"].append(not record.abstained)
                split["called"].append(
                    -1 if record.decision_label is None else int(record.decision_label)
                )
            split["probability"].append(probability)
            split["covered"].append(
                np.array([int(label) in held for label, held in zip(labels, sets, strict=True)])
            )
        raw.append(scores)
        reference.append(np.full(len(labels), train_labels.mean()))
        truth.append(labels)
    if not truth:
        return {"not_measured": "no event's others held both classes", "folds_skipped": skipped}
    once = np.concatenate(truth)
    y = np.concatenate([once, once])  # each row is judged twice, once each way
    score, base = np.concatenate(raw), np.concatenate(reference)
    p = np.concatenate([np.concatenate(split["probability"]) for split in judged])
    inside = np.concatenate([np.concatenate(split["covered"]) for split in judged])
    was_decided = np.concatenate([np.array(split["decided"]) for split in judged])
    label = np.concatenate([np.array(split["called"]) for split in judged])
    anomaly_calls = was_decided & (label == 1)
    is_anomaly = y == 1
    reference_brier = _brier(np.concatenate([base, base]), y)
    return {
        "rows": len(once),
        "anomalies": int(once.sum()),
        "score": {
            "brier": round(_brier(score, once), 4),
            "ece": round(float(compute_ece(once, score)), 4),
        },
        "probability": {
            "brier": round(_brier(p, y), 4),
            "ece": round(float(compute_ece(y, p)), 4),
            "reference_brier": round(reference_brier, 4),
            "skill": round(1.0 - _brier(p, y) / reference_brier, 4),
        },
        "certificate": {
            "target": coverage,
            "coverage": _share(inside),
            "coverage_of_anomalies": _share(inside[is_anomaly]),
            "coverage_of_normal": _share(inside[~is_anomaly]),
        },
        "decision": {
            "decided": _share(was_decided),
            "abstained": _share(~was_decided),
            "decided_accuracy": _share(label[was_decided] == y[was_decided]),
            "anomaly_calls": int(anomaly_calls.sum()),
            "anomaly_call_precision": _share(y[anomaly_calls] == 1),
            "anomalies_called": _share(anomaly_calls[is_anomaly]),
            "anomalies_abstained": _share(~was_decided[is_anomaly]),
            "anomalies_called_normal": _share((was_decided & (label == 0))[is_anomaly]),
        },
        "by_split": [
            {
                "coverage": _share(np.concatenate(split["covered"])),
                "abstained": _share(~np.array(split["decided"])),
            }
            for split in judged
        ],
        "folds_skipped": skipped,
    }


def verdict(told: dict[str, Any]) -> str:
    """Whether a hazard's certificate held: ``held``, ``fell short`` or ``near``.

    ``held`` and ``fell short`` only where both ways of halving agree; where
    one is at or over the target and the other under, the measurement cannot
    tell, and says ``near``.
    """
    target = told["certificate"]["target"]
    over = [split["coverage"] >= target for split in told["by_split"]]
    return "held" if all(over) else "near" if any(over) else "fell short"


def _named(hazards: list[str]) -> str:
    names = [hazard.replace("_", " ") for hazard in hazards]
    if not names:
        return "none"
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def summary(evidence: dict[str, Any]) -> list[str]:
    """What the evidence says, in sentences made from it and from nothing else."""
    measured = {h: t for h, t in evidence["hazards"].items() if "not_measured" not in t}
    skill = {h: t["probability"]["skill"] for h, t in measured.items()}
    more = sorted(h for h, value in skill.items() if value > 0.05)
    less = sorted(h for h, value in skill.items() if value < -0.02)
    same = sorted(set(measured) - set(more) - set(less))
    by_verdict: dict[str, list[str]] = {"held": [], "near": [], "fell short": []}
    for hazard, told in sorted(measured.items()):
        by_verdict[verdict(told)].append(hazard)
    abstained = {h: t["decision"]["abstained"] for h, t in measured.items()}
    least, most = min(abstained, key=abstained.__getitem__), max(
        abstained, key=abstained.__getitem__
    )
    lines = [
        f"**Is the score a probability?** Calibrated on the hazard's other events, it says "
        f"clearly more than their base rate for {_named(more)}; about as much for "
        f"{_named(same)}; and less for {_named(less)}.",
        f"**Does the certificate hold?** Asked for {evidence['coverage_target']:.0%}, on an event "
        f"it had not seen, it held whichever half it was fitted on for {_named(by_verdict['held'])}"
        f" ({len(by_verdict['held'])} of {len(measured)}); fell short either way for "
        f"{_named(by_verdict['fell short'])}; and was over on one half and under on the other "
        f"for {_named(by_verdict['near'])}.",
        f"**How often is nothing decided?** From {abstained[least]:.1%} of rows "
        f"({least.replace('_', ' ')}) to {abstained[most]:.1%} ({most.replace('_', ' ')}).",
    ]
    for hazard, told in sorted(measured.items()):
        if told["label_source"] not in INDEPENDENT_LABELS:
            continue
        decision, calls = told["decision"], told["decision"]["anomaly_call_precision"]
        lines.append(
            f"**{hazard.replace('_', ' ').capitalize()}, whose labels are independent of what "
            f"is scored:** skill {told['probability']['skill']:+.3f}, coverage "
            f"{told['certificate']['coverage']:.1%}; of its true anomalies "
            f"{decision['anomalies_called']:.1%} were called, "
            f"{decision['anomalies_abstained']:.1%} abstained on and "
            f"{decision['anomalies_called_normal']:.1%} called normal; "
            + (
                "no anomaly call was made."
                if calls is None
                else f"{calls:.1%} of {decision['anomaly_calls']:,} anomaly calls were right."
            )
        )
    return lines


def measure(cache: Path) -> dict[str, Any]:
    """The evidence: every hazard in the cache, and those that could not be fetched."""
    scored = score_events(cache)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    hazards: dict[str, Any] = {}
    for hazard in sorted(scored):
        events = scored[hazard]
        hazards[hazard] = {
            "label_source": label_source(hazard),
            "events": [
                {
                    "event": name,
                    "rows": event["rows"],
                    "anomalies_scored": int(event["labels"].sum()),
                    "subsampled": event["subsampled"],
                }
                for name, event in sorted(events.items())
            ],
            **leave_one_event_out(events),
        }
    return {
        "measured_on": datetime.now(UTC).date().isoformat(),
        "coverage_target": COVERAGE,
        "row_limit": ROW_LIMIT,
        "detector": "omni_mercury_engine.detectors.statistical.MercuryAnomalyDetector",
        "hazards": hazards,
        "not_fetched": sorted(hazard for hazard, entry in manifest.items() if hazard not in scored),
    }


def table(evidence: dict[str, Any]) -> str:
    """The evidence as the table the document prints."""
    lines = [
        "| Hazard | Labels | Events | Rows | Anomalies | Skill | Coverage, target "
        f"{evidence['coverage_target']:.0%} (each half) | Abstained | Decided right | "
        "Anomaly calls right |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for hazard, told in evidence["hazards"].items():
        if "not_measured" in told:
            lines.append(
                f"| {hazard} | {told['label_source']} | {len(told['events'])} | | | "
                f"not measured: {told['not_measured']} | | | | |"
            )
            continue
        decision, calls = told["decision"], told["decision"]["anomaly_call_precision"]
        right = (
            "none decided"
            if decision["decided_accuracy"] is None
            else f"{decision['decided_accuracy']:.1%}"
        )
        splits = " / ".join(f"{split['coverage']:.1%}" for split in told["by_split"])
        lines.append(
            f"| {hazard} | {told['label_source']} | {len(told['events'])} | {told['rows']:,} | "
            f"{told['anomalies']:,} | {told['probability']['skill']:+.3f} | "
            f"{told['certificate']['coverage']:.1%} ({splits}) | {decision['abstained']:.1%} | "
            f"{right} | "
            + ("no calls" if calls is None else f"{calls:.1%} of {decision['anomaly_calls']:,}")
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    """Measure, print, and with ``--write`` record."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache", type=Path, default=None, help="the per-event cache to read")
    parser.add_argument("--write", action="store_true", help="record the evidence")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    cache = args.cache if args.cache is not None else default_cache()
    if not any(cache.glob("*.npz")):
        raise SystemExit(
            f"no events in {cache}: build them with research/engine_calibration/build_cache.py"
        )
    evidence = measure(cache)
    print(table(evidence))
    print("\n".join(summary(evidence)))
    print("not fetched:", ", ".join(evidence["not_fetched"]) or "none")
    if args.write:
        text = json.dumps(evidence, indent=2) + "\n"
        EVIDENCE.write_text(text, encoding="utf-8")
        SHIPPED.write_text(text, encoding="utf-8")
        print(f"wrote {EVIDENCE} and {SHIPPED}")


if __name__ == "__main__":
    main()

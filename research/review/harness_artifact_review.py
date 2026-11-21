# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the artifact-review path does on artifacts where the answer is known.

For each kind (document, image, audio) this builds held-out reference
artifacts (:mod:`research.review.reference_artifacts`), encodes each as an
uploadable file, and sends it through the whole review path
(:func:`omni_mercury_engine.review.review_bytes`: decode, segment,
standardize, fusion, batch-level conformal set, batch-family test, ethical
gates, decision record).  It then counts what came back.

Measured, per kind:

* **coverage** of the conformal set per class, at the artifact level (does a
  clean artifact's set contain "clean"; a spliced one's "anomalous");
* **decisions**: of the clean artifacts, how many were grounded clear, wrongly
  grounded as anomalous, or abstained on; of the spliced, how many were found,
  missed, or abstained on;
* **localisation**: how often the most anomalous segment is inside the splice;
* **row-level ranking** (AUROC of spliced against clean segments);
* **outside the family**: artifacts unlike the reference family (source code
  as documents, this repository's real figures as images, chord-and-onset
  signals as recordings).  The certificate says nothing about these, so what
  matters is how often the path abstains on them.

Bars are fixed here, before any number is looked at, and a bar that is not
met is recorded as not met.  Documents are evaluated on sources the
calibration never saw; that is the only honest setting for them.

Run::

    python research/review/harness_artifact_review.py            # write artifacts/artifact_review.json
    python research/review/harness_artifact_review.py --quick    # a tenth of the artifacts, no file written
    python research/review/harness_artifact_review.py --update-readme   # rewrite the README table from the artifact
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from omni_mercury_engine import review
from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError
from omni_mercury_engine.review.decode import Decoded, ReviewError
from omni_mercury_engine.review.features import extract
from research.review import reference_artifacts as R

ARTIFACT = REPO / "artifacts" / "artifact_review.json"
README = REPO / "README.md"
#: The README table is generated from the artifact, between these markers.
README_BEGIN = "<!-- artifact-review:begin (generated: research/review/harness_artifact_review.py --update-readme) -->"
README_END = "<!-- artifact-review:end -->"
SCHEMA = "mercury.research.artifact_review/v1"

#: Target coverage of the certificate.
COVERAGE = 0.90
#: Seeds: fitting, calibration, the trainer's own held-out check, and this harness
#: never share one.
#: trainer_held_out is the trainer's validation draw (training-seed selection
#: reads it); trainer_report is the disjoint draw the checkpoint's own scope
#: statement quotes; harness is the draw this module's published numbers come
#: from. No draw is read for two of those purposes.
SEEDS = {
    "train": 1101,
    "calibration": 2202,
    "trainer_held_out": 3303,
    "trainer_report": 5505,
    "harness": 4404,
}
#: Fixed before measurement.  With 300 held-out artifacts (150 per class) one
#: standard error of a 0.90 proportion is 0.024; the floor is two below target.
BARS = {
    "coverage_per_class_min": 0.85,
    "grounded_wrong_max": 0.12,
    "row_auroc_min": 0.80,
}


def _decisions(results: list[dict[str, Any]], labels: list[int]) -> dict[str, Any]:
    tally: dict[str, dict[str, int]] = {
        "clean": {"grounded_clear": 0, "grounded_anomalous": 0, "abstained": 0},
        "spliced": {"grounded_anomalous": 0, "grounded_clear": 0, "abstained": 0},
    }
    for result, label in zip(results, labels, strict=True):
        decision = result["decision"]
        side = "spliced" if label else "clean"
        if decision["abstained"]:
            tally[side]["abstained"] += 1
        elif decision["decision_label"] == 1:
            tally[side]["grounded_anomalous"] += 1
        else:
            tally[side]["grounded_clear"] += 1
    return tally


def measure(engine: Any, kind: str, count: int, seed: int) -> dict[str, Any]:
    """Send ``count`` held-out reference artifacts of ``kind`` through the review path.

    Args:
        engine: An engine serving the kind's review checkpoint.
        kind: ``document``, ``image`` or ``audio``.
        count: Artifacts (half of them spliced).
        seed: Seed for the held-out draw.

    Returns:
        The measurements described in the module docstring.
    """
    from omni_mercury_engine.ml.mercury_ml import roc_auc_score

    results: list[dict[str, Any]] = []
    labels: list[int] = []
    recipes: list[str] = []
    row_probs: list[np.ndarray[Any, Any]] = []
    row_labels: list[np.ndarray[Any, Any]] = []
    located: list[bool] = []
    seconds: list[float] = []
    refused = {0: 0, 1: 0}
    for reference in R.references(kind, count, seed, "evaluation"):
        data, name = R.to_bytes(reference)
        started = time.perf_counter()
        try:
            result = review.review_bytes(data, name, engine=engine)
        except EthicalConstraintViolationError:
            # The harm-uplift gate refused this reference artifact's text.
            # A refusal is an outcome of the path, not a crash of the
            # measurement: it is counted, and the artifact takes no part in
            # the coverage, decision or ranking numbers (it got no decision).
            refused[int(reference.spliced)] += 1
            continue
        seconds.append(time.perf_counter() - started)
        # Labels follow the segments the path itself cut from the decoded file.
        segmented = extract(review.decode(data, name))
        truth = R.segment_labels(segmented, reference)
        results.append(result)
        labels.append(int(reference.spliced))
        recipes.append(reference.recipe)
        row_probs.append(np.asarray(result["anomaly_probs"], dtype=np.float64).ravel())
        row_labels.append(truth)
        if reference.spliced and truth.any():
            located.append(bool(truth[int(result["top_row"])]))

    def covered(label: int) -> float:
        sets = [
            r["conformal"]["prediction_set"]
            for r, y in zip(results, labels, strict=True)
            if y == label
        ]
        return float(np.mean([label in s for s in sets])) if sets else float("nan")

    decisions = _decisions(results, labels)
    decisions["clean"]["refused_by_gate"] = refused[0]
    decisions["spliced"]["refused_by_gate"] = refused[1]
    n_clean, n_spliced = labels.count(0), labels.count(1)
    missed: dict[str, int] = {}
    for result, label, recipe in zip(results, labels, recipes, strict=True):
        if (
            label
            and not result["decision"]["abstained"]
            and result["decision"]["decision_label"] == 0
        ):
            for part in recipe.split("+"):
                missed[part] = missed.get(part, 0) + 1
    all_rows, all_truth = np.concatenate(row_probs), np.concatenate(row_labels)
    return {
        # Drawn counts; an artifact the gate refused is in ``decisions`` under
        # ``refused_by_gate`` and in no other number (it got no decision, no
        # prediction set and no ranking).
        "artifacts": {"clean": n_clean + refused[0], "spliced": n_spliced + refused[1]},
        "refused_by_gate": {"clean": refused[0], "spliced": refused[1]},
        "segments_per_artifact_median": int(np.median([len(p) for p in row_probs])),
        "coverage": {"clean": covered(0), "spliced": covered(1), "target": COVERAGE},
        "decisions": decisions,
        "rates": {
            "clean_grounded_clear": decisions["clean"]["grounded_clear"] / max(n_clean, 1),
            "clean_grounded_wrong": decisions["clean"]["grounded_anomalous"] / max(n_clean, 1),
            "spliced_found": decisions["spliced"]["grounded_anomalous"] / max(n_spliced, 1),
            "spliced_grounded_wrong": decisions["spliced"]["grounded_clear"] / max(n_spliced, 1),
            "abstained": (decisions["clean"]["abstained"] + decisions["spliced"]["abstained"])
            / max(len(labels), 1),
        },
        "family_test_deferrals": int(
            sum(bool(r["decision"]["signals"]["drift_detected"]) for r in results)
        ),
        "localisation": {
            "top_segment_in_splice": float(np.mean(located)) if located else float("nan"),
            "n": len(located),
        },
        "row_auroc": float(roc_auc_score(all_truth, all_rows)) if all_truth.any() else float("nan"),
        "missed_by_recipe": dict(sorted(missed.items())),
        "seconds_per_artifact_median": float(np.median(seconds)),
    }


# --------------------------------------------------------------------------- #
# Outside the family
# --------------------------------------------------------------------------- #


def _music(rng: np.random.Generator, seconds: float) -> np.ndarray[Any, Any]:
    """Chords with sharp onsets and decays: nothing like a voice."""
    rate = 16_000
    signal = np.zeros(int(seconds * rate))
    cursor = 0
    while cursor < signal.size:
        length = int(rng.uniform(0.2, 0.9) * rate)
        t = np.arange(min(length, signal.size - cursor)) / rate
        root = rng.choice([110.0, 146.8, 196.0, 261.6, 329.6])
        chord = sum(np.sin(2 * np.pi * root * ratio * t) for ratio in (1.0, 1.26, 1.5, 2.0))
        signal[cursor : cursor + t.size] = 0.2 * chord * np.exp(-3.0 * t)
        cursor += length
    return np.clip(signal + rng.normal(scale=0.003, size=signal.size), -1, 1)


def _outside(kind: str, count: int, seed: int) -> list[tuple[bytes, str]]:
    rng = np.random.default_rng(seed)
    if kind == review.DOCUMENT:
        sources = sorted((REPO / "src" / "omni_mercury_engine" / "core").glob("*.py"))
        picked = [
            sources[int(i)]
            for i in rng.choice(len(sources), size=min(count, len(sources)), replace=False)
        ]
        return [
            (path.read_bytes()[:200_000], path.name)
            for path in picked
            if path.stat().st_size > 12_000
        ]
    if kind == review.IMAGE:
        figures = sorted(
            path
            for path in REPO.glob("docs/**/*.png")
            if "design" not in path.parts and path.stat().st_size > 20_000
        )
        picked = [
            figures[int(i)]
            for i in rng.choice(len(figures), size=min(count, len(figures)), replace=False)
        ]
        return [(path.read_bytes(), path.name) for path in picked]
    out = []
    for _ in range(count):
        reference = R.Reference(
            review.AUDIO,
            Decoded(
                review.AUDIO,
                "wav",
                samples=_music(rng, float(rng.uniform(14, 30))),
                sample_rate=16_000,
            ),
            False,
            [],
            "clean",
        )
        out.append(R.to_bytes(reference))
    return out


def measure_outside(engine: Any, kind: str, count: int, seed: int) -> dict[str, Any]:
    """What the path does with artifacts unlike its reference family.

    Returns:
        How many it abstained on, grounded as clear, grounded as anomalous, or
        could not review, and what the artifacts were.
    """
    described = {
        review.DOCUMENT: "Python source files of this repository, uploaded as documents",
        review.IMAGE: "this repository's real figures (charts and diagrams), not textures",
        review.AUDIO: "generated chords with sharp onsets, not a voice",
    }[kind]
    tally = {
        "abstained": 0,
        "grounded_clear": 0,
        "grounded_anomalous": 0,
        "not_reviewable": 0,
        "refused_by_gate": 0,
    }
    family = 0
    samples = _outside(kind, count, seed)
    for data, name in samples:
        try:
            result = review.review_bytes(data, name, engine=engine)
        except EthicalConstraintViolationError:
            tally["refused_by_gate"] += 1
            continue
        except ReviewError:
            tally["not_reviewable"] += 1
            continue
        decision = result["decision"]
        family += int(bool(decision["signals"]["drift_detected"]))
        if decision["abstained"]:
            tally["abstained"] += 1
        elif decision["decision_label"] == 1:
            tally["grounded_anomalous"] += 1
        else:
            tally["grounded_clear"] += 1
    return {
        "what": described,
        "artifacts": len(samples),
        "outcomes": tally,
        "family_test_deferrals": family,
    }


def scope_statement(
    kind: str, n_calibration: int, held_out: dict[str, Any], outside: dict[str, Any]
) -> str:
    """The certificate's scope in one paragraph, from measured numbers only.

    This is what a review checkpoint carries in its provenance and what every
    decision made with it repeats as a caveat.
    """
    source = {
        review.DOCUMENT: "runs of prose from this repository's documentation",
        review.IMAGE: "procedurally generated textures",
        review.AUDIO: "procedurally generated voiced signals",
    }[kind]
    noun = {review.DOCUMENT: "documents", review.IMAGE: "images", review.AUDIO: "recordings"}[kind]
    clean, spliced = held_out["coverage"]["clean"], held_out["coverage"]["spliced"]
    n = held_out["artifacts"]["clean"] + held_out["artifacts"]["spliced"]
    unseen = " from sources the calibration never saw" if kind == review.DOCUMENT else ""
    short = min(clean, spliced) < BARS["coverage_per_class_min"]
    outcomes = outside["outcomes"]
    return (
        f"calibrated on {n_calibration} reference {noun} ({source}), half with procedurally "
        f"inserted foreign material. On {n} held-out reference {noun}{unseen}, the 90% set "
        f"covered {clean:.1%} of clean and {spliced:.1%} of spliced ones"
        + (
            f" -- BELOW the {BARS['coverage_per_class_min']:.0%} per-class floor set for it, so "
            "treat a flag as a pointer to read, not a finding"
            if short
            else ""
        )
        + f". Real-world {noun} are more varied than this family: of {outside['artifacts']} "
        f"unlike it ({outside['what']}), {outcomes['abstained']} were abstained on, "
        f"{outcomes['grounded_anomalous']} called anomalous and {outcomes['grounded_clear']} called clear."
    )


def readme_table(payload: dict[str, Any]) -> str:
    """The measured table the README carries, rendered from the artifact alone."""
    noun = {"document": "documents", "image": "images", "audio": "recordings"}
    lines = [
        "| Kind | 90% set covered (clean / spliced) | Spliced: found / called clear / abstained "
        "| Clean: clear / called anomalous / abstained | Top segment inside the splice "
        "| Segment AUROC | Unlike the family: abstained / judged | Bars met |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind, entry in payload["kinds"].items():
        held = entry["held_out"]
        clean, spliced = held["decisions"]["clean"], held["decisions"]["spliced"]
        outside = entry["outside_the_family"]
        judged = outside["outcomes"]["grounded_anomalous"] + outside["outcomes"]["grounded_clear"]
        wrong = max(
            spliced["grounded_clear"] / max(held["artifacts"]["spliced"], 1),
            clean["grounded_anomalous"] / max(held["artifacts"]["clean"], 1),
        )
        missed = {
            "coverage_per_class": (
                f"per-class coverage ({min(held['coverage'].get('clean', 0), held['coverage'].get('spliced', 0)):.1%} "
                f"< {payload['bars']['coverage_per_class_min']:.0%})"
            ),
            "grounded_wrong": (
                f"wrong decisions ({wrong:.1%} > {payload['bars']['grounded_wrong_max']:.0%})"
            ),
            "row_auroc": f"segment AUROC ({held['row_auroc']:.3f} < {payload['bars']['row_auroc_min']})",
        }
        unmet = [missed[name] for name, met in entry["bars_met"].items() if not met]
        lines.append(
            f"| {noun[kind]} ({held['artifacts']['clean']} + {held['artifacts']['spliced']}) "
            f"| {held['coverage']['clean']:.1%} / {held['coverage']['spliced']:.1%} "
            f"| {spliced['grounded_anomalous']} / {spliced['grounded_clear']} / {spliced['abstained']} "
            f"| {clean['grounded_clear']} / {clean['grounded_anomalous']} / {clean['abstained']} "
            f"| {held['localisation']['top_segment_in_splice']:.1%} "
            f"| {held['row_auroc']:.3f} "
            f"| {outside['outcomes']['abstained']} / {judged} of {outside['artifacts']} "
            f"| {'all three' if not unmet else 'missed: ' + '; '.join(unmet)} |"
        )
    return "\n".join(lines)


def update_readme(payload: dict[str, Any]) -> bool:
    """Rewrite the README's generated table from ``payload``; True when it changed."""
    text = README.read_text(encoding="utf-8")
    begin, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    updated = text[:begin] + "\n" + readme_table(payload) + "\n" + text[end:]
    if updated == text:
        return False
    README.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    """Measure the shipped review checkpoints and write the artifact."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--quick", action="store_true", help="a tenth of the artifacts; nothing written"
    )
    parser.add_argument("--count", type=int, default=300, help="held-out artifacts per kind")
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="rewrite the README table from the committed artifact; measures nothing",
    )
    args = parser.parse_args(argv)
    if args.update_readme:
        changed = update_readme(json.loads(ARTIFACT.read_text(encoding="utf-8")))
        print("README table rewritten" if changed else "README table is current")
        return 0
    warnings.simplefilter("ignore")
    logging.disable(logging.WARNING)
    count = max(20, args.count // 10) if args.quick else args.count

    kinds: dict[str, Any] = {}
    for kind in review.KINDS:
        engine = review.build_engine(kind)
        path = review.checkpoint_path(kind)
        held_out = measure(engine, kind, count, SEEDS["harness"])
        outside = measure_outside(engine, kind, 24, SEEDS["harness"] + 1)
        coverage = held_out["coverage"]
        wrong = max(
            held_out["rates"]["clean_grounded_wrong"], held_out["rates"]["spliced_grounded_wrong"]
        )
        bars = {
            "coverage_per_class": bool(
                min(coverage["clean"], coverage["spliced"]) >= BARS["coverage_per_class_min"]
            ),
            "grounded_wrong": bool(wrong <= BARS["grounded_wrong_max"]),
            "row_auroc": bool(held_out["row_auroc"] >= BARS["row_auroc_min"]),
        }
        provenance = getattr(engine, "_fusion_provenance", None) or {}
        corpus = None
        if kind == review.DOCUMENT:
            # Reference documents are cut from the repository's documentation,
            # so a checkpoint and a measurement made on different states of it
            # describe different documents. Recorded, and said aloud.
            measured_on, trained_on = R.corpus_digest(), provenance.get("corpus")
            corpus = {
                "measured_on": measured_on,
                "checkpoint_trained_on": trained_on,
                "same": measured_on == trained_on,
            }
            if not corpus["same"]:
                print(
                    "document: the documentation has changed since this checkpoint was "
                    "trained; these numbers are for reference documents it was not "
                    "calibrated on. Retrain (scripts/train_review_checkpoints.py) first."
                )
        kinds[kind] = {
            "checkpoint": path.name,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reference_corpus": corpus,
            "certificate": engine._model_provenance()["conformal"],
            "certificate_regime": engine._fusion_conformal_regime,
            "certificate_scope": provenance.get("certificate_scope"),
            "held_out": held_out,
            "outside_the_family": outside,
            "bars_met": bars,
            "all_bars_met": all(bars.values()),
        }
        print(
            f"{kind}: coverage clean {coverage['clean']:.3f} / spliced {coverage['spliced']:.3f}; "
            f"clean wrong {held_out['rates']['clean_grounded_wrong']:.3f}, spliced found "
            f"{held_out['rates']['spliced_found']:.3f}; row AUROC {held_out['row_auroc']:.4f}; "
            f"bars {bars}"
        )

    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "target_coverage": COVERAGE,
            "held_out_artifacts_per_kind": count,
            "seeds": SEEDS,
            "document_sources": {
                split: [path.name for path in R.prose_files(split)]
                for split in ("train", "calibration", "evaluation")
            },
            "note": (
                "Reference artifacts are procedural (see research/review/reference_artifacts.py): "
                "these numbers say how well inserted foreign material of those kinds is found, "
                "not how well real-world tampering is. The document feature set was chosen after "
                "three variants were compared on the held-out sources, so the document figures "
                "are, if anything, optimistic."
            ),
        },
        "bars": BARS,
        "kinds": kinds,
    }
    if args.quick:
        print(json.dumps({k: v["bars_met"] for k, v in kinds.items()}, indent=1))
        return 0
    ARTIFACT.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {ARTIFACT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

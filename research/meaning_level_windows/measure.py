#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A request the classifier knows, with other text round it: is it still found?

The meaning-level classifier (``cognitive/meaning_level.py``) was trained on
single requests and scores what it is given as one bag of words. This puts
requests inside unrelated text and counts how many the harm gate still blocks,
and how many harmless texts about a hazard it blocks that it did not before.

Three sets, each put through the gate in its shipped posture:

* ``train_offensive`` and ``train_benign``: a fixed sample of the classifier's
  own training corpus (``benchmarks/meaning_level_corpus.py``). The benign
  sample is the rows that name a hazard in a defensive or a definitional
  frame, which are the ones a wider reading could wrongly raise. The window
  sizes in ``ethical_bounding.MEANING_WINDOWS`` were chosen on these two
  sets and on nothing else.
* ``held_out``: the adversarial slice (``benchmarks/weapons_gate_adversarial
  .jsonl``), which no rule or weight was fitted to. It is reported, once the
  windows are fixed, and was not used to choose them.

Two kinds of text are put round each request, the same sentences in the same
order every run (nothing here is random):

* ``neutral``: plain statements with no request in them (:data:`NEUTRAL`).
  This measures the classifier alone: what other words round a request do to
  its score.
* ``procedural``: rows of the training corpus's ``benign_task`` group, which
  are requests ("how do I ...") that name no hazard. This measures something
  else as well, and it is not the classifier's: the lexical axes pair a
  how-to frame with a hazard named anywhere in the same passage, so a
  harmless text that names a hazard is refused when requests about other
  things share its passage.

Run::

    PYTHONPATH=src:benchmarks python research/meaning_level_windows/measure.py
    PYTHONPATH=src:benchmarks python research/meaning_level_windows/measure.py --write

``--windows 48:24,24:12`` tries other window sizes without changing the code;
``--no-windows`` reads each text as a whole only, which is how the gate read
before; ``--units`` and ``--statements`` try the other ways of dividing a text
and of believing the classifier that were measured and not adopted. A trial is
printed and cannot be recorded: the evidence is of the gate as it ships.

The evidence has four parts, and ``--only PART`` measures one again and keeps
the rest: ``blocked`` (both fillers, the gate as it ships), ``as_a_whole``
(the neutral filler with no windows: what they are compared with),
``lexical_only`` (the procedural filler with no classifier at all, which
shows whose refusals those are) and ``articles``.

``--articles NAME=DIR`` (any number of times) also reads the encyclopedia
articles ``research/harm_gate_long_text/measure.py --collect`` leaves in a
directory, each as one request, with the windows and without them, and records
which are refused. Two sets were read: the 80 articles of that study, on
which the pattern for "someone asking" was adjusted after two of their
sentences were raised by it, and 40 more collected afterwards and not looked
at until the code was fixed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

EVIDENCE = Path(__file__).with_name("evidence.json")
REPO = Path(__file__).resolve().parents[2]
#: Words of other text put before a request, and the same again after it.
PADDING = (0, 24, 96, 384, 1536)
SAMPLE = 400
#: The parts of the evidence, each of which can be measured again by itself.
SECTIONS = ("blocked", "as_a_whole", "lexical_only", "articles")
#: Set while the lexical axes are measured with no classifier at all.
LEXICAL_ONLY = False
#: Statements with no request and no hazard in them.
NEUTRAL = (
    "The committee met on Tuesday and approved the minutes of the last meeting",
    "Rain is expected along the coast by the end of the week",
    "The library will close early on Friday for the holiday",
    "Sales in the third quarter were slightly above the forecast",
    "The bridge was repainted last summer after two years of delay",
    "Her train arrived twenty minutes late because of a signal fault",
    "The museum opened a new wing devoted to regional pottery",
    "Most of the orchard was planted by his grandfather in the fifties",
    "The council voted to extend the cycle path to the harbour",
    "A new timetable comes into effect at the start of next month",
    "The school choir sang at the opening of the winter market",
    "Prices for wheat fell again after a good harvest in the north",
    "The ferry runs twice a day in winter and hourly in summer",
    "Their youngest daughter starts at the university in the autumn",
    "The report runs to ninety pages and has four appendices",
    "The old mill by the river has been turned into flats",
    "Attendance at the festival was the highest in ten years",
    "The road to the pass is closed until the snow has been cleared",
    "He kept the accounts for the club for nearly thirty years",
    "The garden is open to visitors on the first Sunday of each month",
)


def _every(items: list[Any], count: int) -> list[Any]:
    """``count`` of ``items``, evenly spaced: the same ones every run."""
    if len(items) <= count:
        return items
    return [items[index * len(items) // count] for index in range(count)]


def _sets() -> tuple[dict[str, list[tuple[str, int]]], dict[str, list[str]]]:
    import meaning_level_corpus  # benchmarks/, on PYTHONPATH

    rows = meaning_level_corpus.build_training_corpus()
    offensive = [(row.text, 1) for row in rows if row.label == 1]
    benign = [(row.text, 0) for row in rows if row.group in ("defensive", "definitional")]
    fillers = {
        "neutral": list(NEUTRAL),
        "procedural": [row.text for row in rows if row.group == "benign_task"],
    }
    held_out = [
        (record["text"], int(record["label"] == "offensive"))
        for record in map(
            json.loads,
            (REPO / "benchmarks" / "weapons_gate_adversarial.jsonl")
            .read_text("utf-8")
            .splitlines(),
        )
    ]
    sample = _every(offensive, SAMPLE)
    sets = {
        "train_offensive": sample,
        # The same requests with a full stop put in the middle of each: what
        # dividing a text at its sentence ends would lose.
        "train_offensive_split": [(_split(text), label) for text, label in sample],
        "train_benign": _every(benign, SAMPLE),
        "held_out": held_out,
    }
    return sets, fillers


def _split(text: str) -> str:
    """``text`` with a full stop after its middle word."""
    words = text.split()
    half = len(words) // 2
    return f"{' '.join(words[:half])}. {' '.join(words[half:])}"


def _padded(text: str, words: int, filler: list[str], turn: int) -> str:
    """``text`` with ``words`` words of filler before it and after it."""
    if not words:
        return text
    before: list[str] = []
    after: list[str] = []
    at = turn
    for side in (before, after):
        count = 0
        while count < words:
            sentence = filler[at % len(filler)]
            at += 1
            side.append(sentence)
            count += len(sentence.split())
    return ". ".join((*before, text, *after))


def _blocked(text: str) -> bool:
    from omni_mercury_engine.cognitive.decision_gate import (
        DecisionSubject,
        enforce_decision_boundary,
    )
    from omni_mercury_engine.cognitive.ethical_bounding import EthicalConstraintViolationError

    subject = DecisionSubject(surface="research.windows", operation="measure", request=text)
    try:
        if LEXICAL_ONLY:
            enforce_decision_boundary(subject, harm_classifier=None)
        else:
            enforce_decision_boundary(subject)
    except EthicalConstraintViolationError:
        return True
    return False


def measure(only: str | None = None) -> dict[str, Any]:
    """Blocked counts for each filler (or ``only`` one), set and amount of padding."""
    sets, fillers = _sets()
    counted: dict[str, Any] = {}
    for kind, filler in fillers.items():
        if only and kind != only:
            continue
        counted[kind] = {}
        for name, rows in sets.items():
            by_label: dict[str, dict[str, Any]] = {}
            for label, key in ((1, "offensive"), (0, "benign")):
                texts = [text for text, is_offensive in rows if is_offensive == label]
                if not texts:
                    continue
                by_label[key] = {
                    "texts": len(texts),
                    "blocked": {
                        str(words): sum(
                            _blocked(_padded(text, words, filler, turn))
                            for turn, text in enumerate(texts)
                        )
                        for words in PADDING
                    },
                }
            counted[kind][name] = by_label
    return counted


def articles(directory: Path) -> dict[str, Any]:
    """Which articles in ``directory`` the gate refuses, with the windows and without."""
    from omni_mercury_engine.cognitive import ethical_bounding

    read: dict[str, Any] = {}
    kept = ethical_bounding.MEANING_WINDOWS
    for group in ("general", "hazard"):
        paths = sorted(directory.glob(f"{group}__*.txt"))
        texts = {path.stem.split("__", 1)[1]: path.read_text(encoding="utf-8") for path in paths}
        refused = {}
        for label, windows in (("without_windows", ()), ("with_windows", kept)):
            ethical_bounding.MEANING_WINDOWS = windows
            refused[label] = sorted(title for title, text in texts.items() if _blocked(text))
        ethical_bounding.MEANING_WINDOWS = kept
        read[group] = {"articles": len(texts), **refused}
    return read


def _show(counted: dict[str, Any], heading: str = "") -> None:
    for kind, sets in counted.items():
        for name, by_label in sets.items():
            for key, cell in by_label.items():
                row = "  ".join(
                    f"{words:>5}: {cell['blocked'][str(words)]:>3}" for words in PADDING
                )
                print(f"{heading}{kind:10s} {name:16s} {key:9s} of {cell['texts']:3d}  {row}")


def main() -> None:
    """Measure, print, and with ``--write`` record."""
    global LEXICAL_ONLY
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"record in {EVIDENCE.name}")
    parser.add_argument("--windows", help="window sizes to try, as words:step,words:step")
    parser.add_argument("--no-windows", action="store_true", help="read each text as a whole only")
    parser.add_argument("--units", choices=("words", "sentences", "pairs"), help="how to divide")
    parser.add_argument("--filler", choices=("neutral", "procedural"), help="one filler only")
    parser.add_argument("--statements", action="store_true", help="believe it of statements too")
    parser.add_argument("--articles", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument(
        "--only",
        action="append",
        choices=SECTIONS,
        help="measure this part of the evidence and keep the rest as recorded",
    )
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)

    from omni_mercury_engine.cognitive import ethical_bounding

    trial = bool(args.windows or args.no_windows or args.units or args.statements or args.filler)
    if trial and args.write:
        parser.error("the evidence is of the gate as it ships: a trial is printed, not recorded")
    if args.no_windows:
        ethical_bounding.MEANING_WINDOWS = ()
    elif args.windows:
        ethical_bounding.MEANING_WINDOWS = tuple(
            (int(size), int(step))
            for size, step in (pair.split(":") for pair in args.windows.split(","))
        )
    if args.units:
        ethical_bounding.MEANING_UNITS = args.units
    if args.statements:
        ethical_bounding.MEANING_WINDOWS_ASKED = False
    chosen = ethical_bounding.MEANING_WINDOWS
    wanted = set(args.only or (("blocked",) if trial else SECTIONS))
    record = json.loads(EVIDENCE.read_text(encoding="utf-8")) if args.only else {}
    record.update(
        windows=[list(pair) for pair in chosen],
        units=ethical_bounding.MEANING_UNITS,
        asked=ethical_bounding.MEANING_WINDOWS_ASKED,
        padding=list(PADDING),
    )
    if "blocked" in wanted:
        record["blocked"] = measure(args.filler)
    if "as_a_whole" in wanted:
        # The same texts read as one thing each, which is how the gate read
        # before: what the windows are measured against.
        ethical_bounding.MEANING_WINDOWS = ()
        record["as_a_whole"] = measure("neutral")
        ethical_bounding.MEANING_WINDOWS = chosen
    if "lexical_only" in wanted:
        # With no classifier at all: what the lexical axes do by themselves
        # when the text round a row is requests about other things.
        LEXICAL_ONLY = True
        record["lexical_only"] = measure("procedural")
        LEXICAL_ONLY = False
    for section in ("blocked", "as_a_whole", "lexical_only"):
        _show(record.get(section, {}), f"{section:12s} ")
    read = record.setdefault("articles", {})
    for named in args.articles if "articles" in wanted else ():
        name, _, directory = named.partition("=")
        read[name] = articles(Path(directory))
    for name, groups in read.items():
        for group, cell in groups.items():
            print(
                f"articles {name:6s} {group:8s} of {cell['articles']:3d}  "
                f"without {len(cell['without_windows']):2d}  with {len(cell['with_windows']):2d}"
            )
    if args.write:
        EVIDENCE.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {EVIDENCE}")


if __name__ == "__main__":
    main()

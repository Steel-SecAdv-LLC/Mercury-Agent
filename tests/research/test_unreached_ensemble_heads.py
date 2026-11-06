# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The unreached ensemble heads: their measured dispositions, pinned.

``docs/DORMANCY_LEDGER.md`` §2c records three of the four unreached ensembles
as measured against the plain mean of real detector scores, and refuses all
three. The full 15-cell measurement lives in
``research/ensembles/harness_unreached_heads.py`` and is too expensive for a
test lane, so what is pinned here is the part that is cheap and that the
dispositions actually rest on:

* the structural defect in ``VotingEnsemble`` -- it publishes no continuous
  fused score, so it cannot rank whatever weights it is given. That is a
  property of the code, checkable in milliseconds, and it is the reason the
  head is refused rather than tuned;
* that the heads remain unreached, so a future wiring has to come past this
  file;
* that the claims removed from the four modules stay removed.

The refusals themselves are evidence-backed, not permanent: §2c names the bar
each head failed and what it would take to clear it.
"""

from __future__ import annotations

import numpy as np


def test_voting_ensemble_confidence_is_not_an_anomaly_score() -> None:
    """``confidence`` is symmetric about the threshold, so it cannot rank.

    ``|weighted_avg - 0.5| * 2`` gives a confidently-normal row and a
    confidently-anomalous row the same value. This is the measured reason
    ranking by it scores 0.7566 against a plain mean's 0.9026: half the
    ordering is inverted, and no re-weighting fixes it.

    Pinned as a property rather than a number so the field cannot quietly be
    adopted as a score by a future caller.
    """
    from omni_mercury_engine.ml.ensemble import VotingEnsemble

    # One clearly normal row and one clearly anomalous row, by every detector.
    outputs = {
        "a": np.array([0.02, 0.98]),
        "b": np.array([0.01, 0.99]),
        "c": np.array([0.03, 0.97]),
    }
    result = VotingEnsemble(voting_type="soft").predict(outputs)
    confidence = np.asarray(result["confidence"], dtype=float)

    # Both rows are near-certain, in opposite directions, so a score would rank
    # them far apart and this symmetric statistic ranks them together.
    assert abs(float(confidence[0]) - float(confidence[1])) < 0.05, (
        "confidence now separates the normal row from the anomalous one "
        f"({confidence[0]:.4f} vs {confidence[1]:.4f}), so it may have become a "
        "usable score -- re-measure it against the plain mean and update "
        "DORMANCY_LEDGER 2c before relying on that"
    )
    # And the binary predictions, which is what its contract does support, do
    # separate them -- so the head is not broken, only unable to rank.
    predictions = np.asarray(result["predictions"], dtype=float)
    assert predictions[1] > predictions[0]


def test_the_unreached_heads_are_still_unreached() -> None:
    """No serving path imports them, which is what makes §2c's bar the live one.

    If one of these is wired into the engine, the API, the CLI or a manifest
    entry, its disposition in §2c is no longer a statement about dormant code
    and the measurement has to be redone against the engine's own
    ``OmniFusionModel`` baseline (§8), which §2c explicitly does not do.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
    unreached = (
        "ml.ensemble",
        "ml.ensemble_coordinator",
        "core.three_r.learnable_fusion",
        "models.foundation.ensemble",
    )
    serving = [
        root / "engine.py",
        *(root / "api").glob("*.py"),
        root / "cli.py",
        root / "core" / "detector_registry.py",
    ]
    sources = {path: path.read_text(encoding="utf-8") for path in serving if path.is_file()}
    assert sources, "no serving-path source found; the check would pass vacuously"

    for module in unreached:
        dotted = f"omni_mercury_engine.{module}"
        tail = module.rsplit(".", 1)[-1]
        for path, text in sources.items():
            assert dotted not in text, (
                f"{path.name} now imports {dotted}; it is recorded as unreached "
                "in DORMANCY_LEDGER 2c and its measurement does not cover a "
                "serving role"
            )
            assert (
                f"import {tail}" not in text or dotted in text
            ), f"{path.name} may now import {module}; confirm and update 2c"


def test_the_withdrawn_claims_stay_withdrawn() -> None:
    """Three unsupported claims were removed; this is the ratchet on them.

    Each described performance no measurement in this repository supports:
    ``ml/ensemble.py`` called itself "particularly effective" two lines after
    stating that no accuracy claim is made for it, ``learnable_fusion.py``
    called itself "state-of-the-art", and ``models/foundation/ensemble.py``
    claimed "improved anomaly detection accuracy".
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "omni_mercury_engine"
    banned = {
        root / "ml" / "ensemble.py": ("particularly effective",),
        root / "core" / "three_r" / "learnable_fusion.py": ("state-of-the-art",),
        root / "models" / "foundation" / "ensemble.py": ("improved anomaly detection",),
    }
    for path, phrases in banned.items():
        text = path.read_text(encoding="utf-8").lower()
        for phrase in phrases:
            # The removal note itself quotes the phrase, so require that any
            # occurrence sits in a sentence that disclaims it.
            if phrase in text:
                index = text.index(phrase)
                window = text[max(0, index - 400) : index + 200]
                assert any(
                    marker in window
                    for marker in ("removed", "no longer", "is the claim", "unsupported")
                ), (
                    f"{path.name} asserts {phrase!r} again without a measurement; "
                    "DORMANCY_LEDGER 2c records why it was withdrawn"
                )

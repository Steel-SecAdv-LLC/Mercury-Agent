# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which rows the governed-fusion suite fits on and which it scores.

An externally labelled event has audited ground truth, so a reference known to
be normal exists for it. The suite used to fit the detector on the whole event
anyway: on ``nsl_kdd`` that reference was 48% attacks, and the figure the
promotion gate reads (0.5427 for that event) measured what the detector does
with such a reference, not whether it finds the attacks (0.97 with a normal
one). A self-labelled event has no such reference and is still scored in place,
declared unlabelled.
"""

from __future__ import annotations

import numpy as np

from research.governed_fusion.score_cache import _scores_path, scored_rows
from research.governed_fusion.suite import EventData


def _event(domain: str, event_id: str) -> EventData:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 4))
    labels = np.zeros(200, dtype=int)
    labels[::4] = 1
    return EventData(domain, event_id, features, labels)


def test_an_externally_labelled_event_is_fitted_on_normal_rows_and_scored_on_the_rest() -> None:
    """Was: fitted on, and scored on, all 200 rows, a quarter of them anomalies."""
    event = _event("network_security", "nsl_kdd")

    reference, scored, labels, in_place = scored_rows(event)

    assert in_place is False
    assert len(reference) == 75  # every other one of the 150 normal rows
    assert len(scored) == len(labels) == 125
    assert int(labels.sum()) == 50  # every anomaly is scored, none is fitted on
    fitted = {row.tobytes() for row in reference}
    assert not fitted & {row.tobytes() for row in scored}
    anomalies = {row.tobytes() for row in event.X[event.y == 1]}
    assert not fitted & anomalies


def test_a_self_labelled_event_is_scored_in_place_and_says_so() -> None:
    event = _event("earthquake", "tohoku_2011")

    reference, scored, labels, in_place = scored_rows(event)

    assert in_place is True
    assert reference is scored
    assert len(scored) == len(labels) == 200


def test_the_two_protocols_do_not_share_cached_scores(tmp_path: object) -> None:
    """Scores cached under the in-place protocol must not be read back for the other."""
    in_place_key = _scores_path("earthquake", "nsl_kdd", 6000, 42, cache_dir=str(tmp_path))
    external_key = _scores_path("network_security", "nsl_kdd", 6000, 42, cache_dir=str(tmp_path))

    assert "normal-reference" not in external_key  # the tag is hashed, not spelled out
    assert in_place_key.rsplit("_", 1)[1] != external_key.rsplit("_", 1)[1]

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the OAE's three streams carry signal, and where they do not.

``docs/MATH_SPEC.md`` §5.1b widened the §5.1a weighting comparison to the two
other ordered corpora this repository records, and the streams turned out to
matter more than the weights. Two properties are pinned here because they are
scope boundaries on the ``/three-r`` route and nothing else measures them:

1. ``R(x)`` is ``clip(std / mean(|x|), 0, 1)``, and that ratio is unbounded
   above, so on an approximately zero-mean signal the clip pins it to exactly
   1.0. It is informative on positive-mean telemetry and a constant off it.
2. ``O(theta)`` is a smoothness statistic, so its polarity is **inverted** for a
   hazard that makes a signal rougher. Bearing faults do.

Neither is a defect in the shipped decision path -- §2.1 records that the OAE is
not the decision rule -- but both bound what the ``/three-r`` route may be
presented as, and a reader who saw only §5.1a's NAB table would not know either.

These use the stream producers from the research harness, which reproduce the
served ``/three_r`` path verbatim, so a drift in the served producers that
changes these properties fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from research.omni_equation.harness_oae_weights import streams
from tests.recorded_data import cwru, eeg, nab_corpus

WINDOW = 128


def _stream_columns(signal: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``(n_windows, 3)`` of (R, H, O) over non-overlapping windows."""
    usable = np.asarray(signal, dtype=float)
    count = usable.size // WINDOW
    assert count > 0, "signal shorter than one window"
    return np.asarray(
        [streams(window) for window in usable[: count * WINDOW].reshape(count, WINDOW)],
        dtype=float,
    )


@pytest.mark.parametrize(
    ("label", "loader"),
    [
        ("cwru_healthy", lambda: cwru("normal_0hp")[: 60 * 2048]),
        ("cwru_inner_race", lambda: cwru("inner_race")[: 20 * 2048]),
        ("eeg_seizure", lambda: eeg("chb01_03")[1]),
    ],
)
def test_the_recursion_stream_saturates_on_zero_mean_signals(label: str, loader: Any) -> None:
    """``R(x)`` is a constant 1.0 off positive-mean telemetry.

    Pinned so the boundary is a property of the repository rather than a
    paragraph. A change that makes ``R`` vary here is an improvement to the
    producer and must be measured and recorded, not slipped in: §5.1b refuses
    the obvious unsaturating repair on the grounds that it buys nothing
    demonstrable, and that refusal is only valid while this holds.
    """
    recursion = _stream_columns(loader())[:, 0]
    assert np.allclose(recursion, 1.0, atol=1e-9), (
        f"{label}: R(x) is no longer saturated "
        f"(min {recursion.min():.4f}, max {recursion.max():.4f}, "
        f"{len(np.unique(np.round(recursion, 6)))} unique). If the producer was "
        "repaired, MATH_SPEC 5.1b's refusal of that repair needs re-deriving."
    )


def test_the_recursion_stream_is_informative_on_positive_mean_telemetry() -> None:
    """The other half of the boundary: ``R(x)`` is not simply broken.

    Without this, the test above could be satisfied by a producer that returns
    a constant everywhere, which is a different and worse defect.
    """
    recursion = _stream_columns(nab_corpus()[3].values)[:, 0]
    assert len(np.unique(np.round(recursion, 6))) > 1, (
        "R(x) is now constant on NAB telemetry too, so it carries no signal "
        "anywhere -- that is a dead producer, not a modality boundary"
    )
    assert recursion.max() < 1.0 - 1e-9, "R(x) now saturates on NAB as well"


def test_the_optimization_stream_is_inverted_for_bearing_faults() -> None:
    """``O(theta)`` ranks rougher fault windows as more normal, by construction.

    It is a sigmoid of log10(SNR + 1) with SNR = var(x) / (var(diff(x)) / 2), so
    it measures smoothness. A bearing fault raises var(diff(x)) nearly as fast
    as var(x), collapsing the ratio. This pins the direction, not a magnitude:
    the point is the sign, and a repair that fixes the sign should fail here and
    be recorded.
    """
    healthy = _stream_columns(
        np.concatenate([cwru("normal_0hp")[: 60 * 2048], cwru("normal_1hp")[: 60 * 2048]])
    )
    fault = _stream_columns(cwru("inner_race")[: 20 * 2048])

    assert fault[:, 2].mean() < healthy[:, 2].mean(), (
        "O(theta) now scores bearing faults above healthy windows; the "
        "inversion MATH_SPEC 5.1b documents has been repaired, and the "
        "section's conclusion about /three-r's scope needs re-deriving"
    )
    assert (
        fault[:, 1].mean() < healthy[:, 1].mean()
    ), "H(omega) now scores bearing faults above healthy windows; see above"


def test_phi_and_uniform_rank_these_corpora_identically() -> None:
    """§5.1b's weighting result, on one unit, cheaply.

    Phi and uniform were identical to four decimal places on all five units.
    That is the evidence the default now rests on, so one unit of it is pinned
    here; the full five-unit table is reproduced by
    ``research/omni_equation/harness_oae_weights_ordered.py``.
    """
    from research.omni_equation.harness_oae_weights import W_PHI, W_UNIFORM
    from tests.recorded_data import auroc

    healthy = _stream_columns(cwru("normal_0hp")[: 60 * 2048])
    fault = _stream_columns(cwru("inner_race")[: 20 * 2048])

    def scored(weights: np.ndarray[Any, Any]) -> float:
        return auroc(fault @ weights, healthy @ weights)

    phi, uniform = scored(W_PHI), scored(W_UNIFORM)
    assert phi == pytest.approx(uniform, abs=5e-4), (
        f"phi ({phi:.4f}) and uniform ({uniform:.4f}) now separate on this "
        "corpus; MATH_SPEC 5.1b's 'identical on every unit' finding, which the "
        "default rests on, needs re-measuring"
    )

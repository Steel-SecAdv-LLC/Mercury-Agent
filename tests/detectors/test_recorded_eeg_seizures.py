# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``neural_cognitive`` on recorded scalp EEG (CHB-MIT, patient chb01), measured and pinned.

``tests/fixtures/eeg`` holds channel FP1-F7 at 256 Hz, in microvolts, from two
recordings: 300 s of background that ends 120 s before an annotated seizure, and
the seizure (40 s and 27 s).  One-second windows (256 samples) are scored by
``NeuralCognitiveModel.predict`` and the AUROC of seizure against background
windows is pinned per recording.

What the number is: the model has no fit step, builds its score from the raw
amplitude and spectral power of a window, and returns it unbounded in the
input's unit (about 1e6 for background and 4e7 for seizure in microvolts).  A
plain per-window standard deviation ranks the same windows at least as well
(0.997 and 0.982), so the model measures amplitude here; that comparison is
pinned too, so a change that makes it more than an amplitude detector is
recorded.

The numbers are the measured behaviour on these recordings, not published
results.  Runs are repeatable.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.models.neural import NeuralCognitiveModel
from tests.recorded_data import auroc, eeg

WINDOW = 256
AUROC_TOLERANCE = 0.03

#: record -> (model AUROC, per-window standard deviation AUROC)
MEASURED = {"chb01_03": (0.991, 0.997), "chb01_04": (0.972, 0.982)}


def _windows(signal: np.ndarray) -> np.ndarray:
    return signal[: len(signal) // WINDOW * WINDOW].reshape(-1, WINDOW)


@pytest.mark.parametrize("record", sorted(MEASURED))
def test_seizure_windows_outrank_background_windows(record: str) -> None:
    background, seizure = (_windows(x) for x in eeg(record))
    model = NeuralCognitiveModel()
    scores_background = np.asarray(model.predict(background)["anomaly_scores"], dtype=float)
    scores_seizure = np.asarray(model.predict(seizure)["anomaly_scores"], dtype=float)
    pinned, baseline = MEASURED[record]
    assert auroc(scores_seizure, scores_background) == pytest.approx(pinned, abs=AUROC_TOLERANCE)
    amplitude = auroc(seizure.std(axis=1), background.std(axis=1))
    assert amplitude == pytest.approx(baseline, abs=AUROC_TOLERANCE)
    assert amplitude >= auroc(scores_seizure, scores_background) - AUROC_TOLERANCE

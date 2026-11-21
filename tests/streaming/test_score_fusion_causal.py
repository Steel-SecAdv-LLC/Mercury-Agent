# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The trained fusion layer's causal fixed-window scoring mode.

``score_fusion`` hands the whole batch to the feature extractors, so a row's
probability can depend on rows after it. ``score_fusion_causal`` runs the
identical trained model over trailing windows. Pinned here on a recorded
ADBench table through Mercury's own training loop:

* causality — a row's probability never changes when later rows change;
* instrument identity — each causal probability equals ``score_fusion`` of
  that row's trailing window, by construction and by assertion;
* detection survives the causal execution — held-out AUROC stays within
  tolerance of the batch mode's on the same split.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.xdist_group("score_fusion_causal"),
    pytest.mark.timeout(900),
]

from omni_mercury_engine.engine import OmniMercuryEngine
from tests.recorded_data import adbench, auroc, stratified_split

WINDOW = 64


@pytest.fixture(scope="module")
def fitted() -> tuple[OmniMercuryEngine, np.ndarray, np.ndarray]:
    """One deterministic fusion fit on the recorded breastw table."""
    features, labels = adbench("breastw")
    train, test = stratified_split(labels, 0)
    torch.manual_seed(0)
    np.random.seed(0)
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    engine.fit_fusion(
        features[train],
        labels[train],
        epochs=10,
        batch_size=64,
        early_stopping_patience=15,
        symbolic_weight=0.0,
    )
    return engine, features[test], labels[test]


def test_causal_scores_ignore_the_future(
    fitted: tuple[OmniMercuryEngine, np.ndarray, np.ndarray],
) -> None:
    engine, x_test, _ = fitted
    base = engine.score_fusion_causal(x_test[:90], window=WINDOW)
    altered = x_test[:90].copy()
    altered[60:] *= 50.0  # violent future rows
    other = engine.score_fusion_causal(altered, window=WINDOW)
    np.testing.assert_array_equal(base[:60], other[:60])


def test_each_row_is_the_instrument_on_its_trailing_window(
    fitted: tuple[OmniMercuryEngine, np.ndarray, np.ndarray],
) -> None:
    engine, x_test, _ = fitted
    rows = x_test[:40]
    causal = engine.score_fusion_causal(rows, window=WINDOW)
    for i in (0, 7, 39):
        lo = max(0, i - WINDOW + 1)
        expected = float(engine.score_fusion(rows[lo : i + 1])[-1])
        assert causal[i] == pytest.approx(expected, abs=1e-9)


def test_detection_survives_the_causal_execution(
    fitted: tuple[OmniMercuryEngine, np.ndarray, np.ndarray],
) -> None:
    """Held-out AUROC under causal windows stays near the batch mode's."""
    engine, x_test, y_test = fitted
    batch = np.asarray(engine.score_fusion(x_test), dtype=float).ravel()
    causal = engine.score_fusion_causal(x_test, window=WINDOW)
    batch_auc = auroc(batch[y_test == 1], batch[y_test == 0])
    causal_auc = auroc(causal[y_test == 1], causal[y_test == 0])
    assert causal_auc >= batch_auc - 0.05, (
        f"causal AUROC {causal_auc:.4f} fell more than 0.05 below the batch "
        f"mode's {batch_auc:.4f}"
    )


def test_mode_and_window_are_validated() -> None:
    engine = OmniMercuryEngine(mode="fusion", device="cpu")
    with pytest.raises(ValueError, match="window must be"):
        engine.score_fusion_causal(np.zeros((4, 3)), window=0)
    plain = OmniMercuryEngine(mode="basic", device="cpu")
    with pytest.raises(ValueError, match="requires mode='fusion'"):
        plain.score_fusion_causal(np.zeros((4, 3)))

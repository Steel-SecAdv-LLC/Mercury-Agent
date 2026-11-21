# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A fit on unlabelled data does not give the kNN stream priority.

The dimensional weight (up to 0.85 at low feature width) was measured with a
reference known to be normal. Fitted on unlabelled rows that contain the
anomalies, density can rank the anomalies as the most normal rows, and at that
weight the fused score follows it. Found on the governed-fusion suite's
externally labelled NSL-KDD event, whose inputs had not changed while its
AUROC fell from 0.6788 to 0.3973 (48% of the capped rows are attacks), and
measured on ADBench: one set scores 0.2021 at the dimensional weight and
0.9956 without the stream.

The cap is chosen on ``benchmarks/results/knn_unlabeled_reference.json``
(47 ADBench sets) and these tests hold the code to that artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.core.config import DataCharacteristics
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector

_ARTIFACT = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "knn_unlabeled_reference.json"
)


def _cloud(seed: int = 0, rows: int = 400, width: int = 6) -> np.ndarray[Any, Any]:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(rows, width))
    data[::40] += 6.0
    return data


def test_an_unlabelled_reference_caps_the_weight_at_equal() -> None:
    """Was: 0.84 at six features whatever the reference was."""
    data = _cloud()
    vetted = MercuryAnomalyDetector().fit(data)
    unlabelled = MercuryAnomalyDetector().fit(data, unlabeled_reference=True)

    assert vetted._knn_fusion_weight == pytest.approx(0.8440, abs=1e-3)
    assert unlabelled._knn_fusion_weight == MercuryAnomalyDetector._KNN_WEIGHT_UNLABELED_CAP == 0.5


def test_the_vetted_path_is_unchanged() -> None:
    """The benchmark's protocol: a plain ``fit`` keeps the dimensional weight."""
    data = _cloud()
    plain = MercuryAnomalyDetector().fit(data)
    explicit = MercuryAnomalyDetector().fit(data, unlabeled_reference=False)

    assert plain._unlabeled_reference is False
    assert plain._knn_fusion_weight == explicit._knn_fusion_weight
    assert np.array_equal(plain.detect(data)["scores"], explicit.detect(data)["scores"])


def test_the_fused_score_is_the_equal_weight_fusion() -> None:
    data = _cloud()
    unlabelled = MercuryAnomalyDetector().fit(data, unlabeled_reference=True)
    pinned = MercuryAnomalyDetector({"knn_stream_weight": 0.5}).fit(data)
    vetted = MercuryAnomalyDetector().fit(data)

    scores = np.asarray(unlabelled.detect(data)["scores"])
    assert np.array_equal(scores, np.asarray(pinned.detect(data)["scores"]))
    assert not np.array_equal(scores, np.asarray(vetted.detect(data)["scores"]))


def test_the_config_key_and_the_argument_agree_and_the_argument_wins() -> None:
    data = _cloud()
    by_config = MercuryAnomalyDetector({"unlabeled_reference": True}).fit(data)
    overridden = MercuryAnomalyDetector({"unlabeled_reference": True}).fit(
        data, unlabeled_reference=False
    )

    assert by_config._knn_fusion_weight == 0.5
    assert overridden._knn_fusion_weight == pytest.approx(0.8440, abs=1e-3)


def test_a_later_vetted_fit_restores_the_dimensional_weight() -> None:
    """An auto-fit on a batch is replaced by the training fit, weight included."""
    data = _cloud()
    detector = MercuryAnomalyDetector().fit(data, unlabeled_reference=True)
    assert detector._knn_fusion_weight == 0.5

    detector.fit(data)

    assert detector._unlabeled_reference is False
    assert detector._knn_fusion_weight == pytest.approx(0.8440, abs=1e-3)


def test_the_cap_never_raises_a_weight() -> None:
    """High width and series data already sit below it; an explicit weight wins."""
    wide = MercuryAnomalyDetector()
    wide._unlabeled_reference = True
    assert wide._knn_dimension_weight(1555) == MercuryAnomalyDetector()._knn_dimension_weight(1555)

    series = MercuryAnomalyDetector()
    series._unlabeled_reference = True
    series._data_type = DataCharacteristics.TEMPORAL
    series._knn_consec_nn_ratio = 1.5
    assert series._knn_dimension_weight(8) == MercuryAnomalyDetector._KNN_WEIGHT_TEMPORAL_CAP

    pinned = MercuryAnomalyDetector({"knn_stream_weight": 0.8})
    pinned._unlabeled_reference = True
    assert pinned._knn_dimension_weight(8) == 0.8


def test_the_cap_is_the_weight_the_measurement_selects() -> None:
    """Equal weight has the best mean and the smallest worst loss on ADBench."""
    artifact = json.loads(_ARTIFACT.read_text(encoding="utf-8"))

    assert artifact["n_datasets"] == 47
    cap = str(MercuryAnomalyDetector._KNN_WEIGHT_UNLABELED_CAP)
    assert max(artifact["mean_auroc"], key=artifact["mean_auroc"].get) == cap
    assert min(artifact["max_regret"], key=artifact["max_regret"].get) == cap
    assert min(artifact["mean_regret"], key=artifact["mean_regret"].get) == cap
    # The failure the cap exists for, as measured.
    assert artifact["max_regret"]["dimensional"] > 0.75
    assert artifact["max_regret"][cap] < 0.08
    # The summary is recomputable from the rows it summarises.
    table = np.array([row["auroc"] for row in artifact["per_dataset"].values()])
    assert np.round(table.mean(axis=0), 4).tolist() == list(artifact["mean_auroc"].values())


def test_the_engine_declares_an_auto_fitted_batch_unlabelled() -> None:
    """Was: the batch became a vetted normal reference at full kNN weight."""
    from omni_mercury_engine.engine import OmniMercuryEngine

    engine = OmniMercuryEngine.__new__(OmniMercuryEngine)
    engine._inference_auto_fit_detectors = set()
    detector = MercuryAnomalyDetector()

    OmniMercuryEngine._auto_fit_on_inference(engine, "statistical", detector, _cloud())

    assert detector._unlabeled_reference is True
    assert detector._knn_fusion_weight == 0.5
    assert engine._inference_auto_fit_detectors == {"statistical"}

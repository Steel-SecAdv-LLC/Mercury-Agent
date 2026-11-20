# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The data-type gate reads what a row IS, not how many rows there are.

``IMAGE`` used to be decided by ``0.3*sqrt(n) <= n_features <= 3*sqrt(n)`` --
a property of the sample count. Across the 76 recorded benchmark sets its only
hit was a 196-feature network-flow table (14 x 14 by coincidence), and the
existing "image is detected" test built 60 features, which can never be
``IMAGE``, and accepted ``IMAGE or TABULAR``. The gate now requires a real
raster: a square 1- or 3-channel layout whose neighbouring pixels cohere both
along a row and down a column.

A class-sorted table is the case no test on the data can settle (its block
order is real autocorrelation), so the row-order meaning is declarable.
"""

from __future__ import annotations

import numpy as np
import pytest

from omni_mercury_engine.core.config import DataCharacteristics
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector


def _smooth_images(n: int, side: int, channels: int, layout: str, seed: int) -> np.ndarray:
    """Flattened smooth 2-D fields: every pixel correlates with its neighbours."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side] / side
    planes = []
    for _ in range(n):
        a, b, c = rng.normal(size=3)
        field = a * np.sin(2 * np.pi * xx) + b * np.cos(2 * np.pi * yy) + c * xx * yy
        field = field + 0.05 * rng.normal(size=field.shape)
        planes.append(np.stack([field * (k + 1) for k in range(channels)]))
    stack = np.asarray(planes)  # [n, C, H, W]
    if layout == "channel_last":
        stack = stack.transpose(0, 2, 3, 1)
    return stack.reshape(n, -1)


def _kind(X: np.ndarray) -> DataCharacteristics:
    return MercuryAnomalyDetector()._detect_data_characteristics(np.asarray(X, dtype=np.float64))


class TestImageGate:
    @pytest.mark.parametrize(
        ("channels", "layout"),
        [(1, "channel_first"), (3, "channel_first"), (3, "channel_last")],
    )
    def test_real_rasters_are_image(self, channels: int, layout: str) -> None:
        X = _smooth_images(n=60, side=16, channels=channels, layout=layout, seed=1)
        # Shuffle rows: an image batch has no row order, and the gate must not
        # need one.
        X = X[np.random.default_rng(2).permutation(len(X))]
        assert _kind(X) is DataCharacteristics.IMAGE

    def test_shape_coincidence_is_not_an_image(self) -> None:
        """196 unrelated features with ~9.8k rows satisfied the old sqrt(n) rule."""
        rng = np.random.default_rng(3)
        # Per-feature offsets give adjacent rows a moderate correlation (as in
        # any real table whose columns differ in level), which is what carried
        # such data past the shuffled-tabular exit and into the old shape rule.
        X = rng.normal(size=196) + rng.normal(size=(9755, 196))
        assert 0.3 * np.sqrt(len(X)) <= X.shape[1] <= 3.0 * np.sqrt(len(X))
        assert _kind(X) is DataCharacteristics.TABULAR

    def test_waveform_rows_are_not_an_image(self) -> None:
        """One 1-D window per row coheres along the row but not one side down."""
        rng = np.random.default_rng(4)
        t = np.arange(1024)
        X = np.asarray(
            [
                np.sin(2 * np.pi * t / rng.uniform(5.0, 9.0) + rng.uniform(0, 6.28))
                + 0.2 * rng.normal(size=1024)
                for _ in range(120)
            ]
        )
        assert _kind(X) is not DataCharacteristics.IMAGE

    @pytest.mark.parametrize("period", [40.0, 64.0, 100.0, 300.0])
    def test_smooth_waveform_rows_are_not_an_image(self, period: float) -> None:
        """A slowly varying 1-D window coheres at EVERY lag, the row stride included.

        What it cannot do is cohere more one stride away than half a stride
        away, which is what a raster does.
        """
        rng = np.random.default_rng(14)
        t = np.arange(256)
        X = np.asarray(
            [
                np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28)) + 0.05 * rng.normal(size=256)
                for _ in range(300)
            ]
        )
        assert _kind(X) is not DataCharacteristics.IMAGE

    def test_non_raster_width_is_never_image(self) -> None:
        rng = np.random.default_rng(5)
        base = rng.normal(size=(300, 1))
        X = base + 0.01 * rng.normal(size=(300, 150))  # coherent, but 150 is no raster
        assert _kind(X) is not DataCharacteristics.IMAGE

    def test_image_rows_get_no_kinematic_weight(self) -> None:
        """Rows of an image batch are unordered: the row-difference term is zeroed."""
        det = MercuryAnomalyDetector()
        det.fit(_smooth_images(n=80, side=12, channels=1, layout="channel_first", seed=6))
        assert det._data_type is DataCharacteristics.IMAGE
        assert det._adaptive_weights[1] == 0.0


class TestDeclaredDataType:
    @staticmethod
    def _class_sorted() -> np.ndarray:
        rng = np.random.default_rng(7)
        return np.vstack([rng.normal(m, 1.0, size=(400, 8)) for m in (0.0, 4.0, 8.0)])

    def test_class_sorted_table_reads_temporal_unless_declared(self) -> None:
        X = self._class_sorted()
        inferred = MercuryAnomalyDetector()
        inferred.fit(X)
        declared = MercuryAnomalyDetector({"data_type": "tabular"})
        declared.fit(X)
        # The inference limit is real and documented; the declaration settles it.
        assert inferred._data_type is DataCharacteristics.TEMPORAL
        assert declared._data_type is DataCharacteristics.TABULAR
        assert declared._adaptive_weights[1] == 0.0

    def test_declaration_overrides_in_both_directions(self) -> None:
        rng = np.random.default_rng(8)
        det = MercuryAnomalyDetector({"data_type": "TEMPORAL"})
        det.fit(rng.normal(size=(200, 4)))
        assert det._data_type is DataCharacteristics.TEMPORAL

    @pytest.mark.parametrize("unset", [None, "", "auto", "AUTO"])
    def test_an_unset_declaration_means_auto(self, unset: str | None) -> None:
        rng = np.random.default_rng(10)
        det = MercuryAnomalyDetector({"data_type": unset})
        det.fit(rng.normal(size=(200, 4)))
        assert det._data_type is DataCharacteristics.TABULAR

    def test_unknown_declaration_is_refused(self) -> None:
        det = MercuryAnomalyDetector({"data_type": "video"})
        with pytest.raises(DetectorException, match="Unknown data_type"):
            det.fit(np.random.default_rng(9).normal(size=(50, 3)))

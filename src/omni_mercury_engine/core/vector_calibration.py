# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration and conformal prediction over a *vector* of Mercury's scores.

Every calibrator and conformal predictor in the repository consumed one
scalar score per sample. The two classes here consume the full vector of
Mercury's own probe and detector scores — the 21 math-arrest probe columns
plus the fused score, or any other internal score matrix — and nothing else:

* :class:`VectorIsotonicCalibrator` learns per-column weights from each
  column's own labelled separation (the ``max(AUC − 0.5, 0) ** power`` margin
  convention the R/K/IG detector already uses), forms the weighted index, and
  fits the in-house strictly-increasing isotonic regression
  (:class:`~omni_mercury_engine.core.calibration.StrictIsotonicCalibration`,
  pure NumPy PAV) from index to probability. Monotone in the index, so the
  index's ranking is preserved exactly.
* :class:`VectorConformalBinary` is a Mondrian (class-conditional) split
  conformal predictor over {normal, anomaly} whose nonconformity is the
  calibrated index: for calibration examples of each class it stores the
  class's nonconformity scores, and a new sample's prediction set is every
  class whose conformal p-value exceeds ``alpha``. The guarantee is the
  standard finite-sample one: under exchangeability, each class's coverage is
  at least ``1 − alpha`` (Vovk, Gammerman & Shafer, *Algorithmic Learning in
  a Random World*, 2005, §4.5 — the (n+1)-rank p-value).

Neither class touches the probes: they post-process score matrices, and the
tests hold the probe outputs bit-identical with and without them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omni_mercury_engine.core.calibration import StrictIsotonicCalibration

__all__ = ["VectorConformalBinary", "VectorIsotonicCalibrator"]

#: The margin-sharpening power the R/K/IG adaptive weighting already uses.
DEFAULT_MARGIN_POWER = 4


def _column_auc(column: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float:
    """Mann-Whitney AUROC of one score column (midranks on ties)."""
    order = np.argsort(column, kind="stable")
    ranks = np.empty(len(column), dtype=np.float64)
    ordered = column[order]
    i = 0
    while i < len(column):
        j = i
        while j + 1 < len(column) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    positive = labels == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class VectorIsotonicCalibrator:
    """Isotonic probability calibration on a weighted index of a score matrix.

    Args:
        margin_power: Exponent sharpening the per-column AUC margins into
            index weights (``max(AUC − 0.5, 0) ** margin_power``). Columns at
            or below chance get weight zero; if no column clears chance the
            weights fall back to uniform.
    """

    def __init__(self, margin_power: int = DEFAULT_MARGIN_POWER) -> None:
        """Initialize the instance."""
        if margin_power < 1:
            raise ValueError(f"margin_power must be >= 1, got {margin_power}")
        self._margin_power = margin_power
        self._weights: npt.NDArray[np.float64] | None = None
        self._isotonic = StrictIsotonicCalibration()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._is_fitted

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """The learned per-column index weights, copied (they sum to 1)."""
        if self._weights is None:
            raise RuntimeError("VectorIsotonicCalibrator has not been fitted.")
        return self._weights.copy()

    def _validate(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 1:
            raise ValueError(f"score matrix must be 2-D with >= 1 column, got {arr.shape}")
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(
        self,
        score_matrix: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> VectorIsotonicCalibrator:
        """Learn the index weights and the isotonic map on labelled scores.

        Args:
            score_matrix: ``(n, k)`` internal scores, one column per probe or
                detector stream.
            labels: Binary ground truth, 1 marking the anomalies.

        Returns:
            ``self``.

        Raises:
            ValueError: On a malformed matrix or a single-class label set.
        """
        matrix = self._validate(score_matrix)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(y) != len(matrix):
            raise ValueError(f"labels ({len(y)}) do not match rows ({len(matrix)})")
        if len(np.unique(y)) < 2:
            raise ValueError("vector calibration needs both classes in the labels")

        margins = np.array(
            [max(_column_auc(matrix[:, j], y) - 0.5, 0.0) for j in range(matrix.shape[1])]
        )
        powered = margins**self._margin_power
        total = float(powered.sum())
        if total <= 0.0:
            weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
        else:
            weights = powered / total
        self._weights = weights
        self._isotonic.fit(matrix @ weights, y)
        self._is_fitted = True
        return self

    def index(self, score_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The weighted scalar index of a score matrix (pre-isotonic)."""
        if self._weights is None:
            raise RuntimeError("VectorIsotonicCalibrator has not been fitted.")
        matrix = self._validate(score_matrix)
        if matrix.shape[1] != len(self._weights):
            raise ValueError(
                f"matrix has {matrix.shape[1]} columns; calibrator was fitted on "
                f"{len(self._weights)}"
            )
        result: npt.NDArray[np.float64] = matrix @ self._weights
        return result

    def calibrate(self, score_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calibrated anomaly probabilities for a score matrix.

        Args:
            score_matrix: ``(n, k)`` scores with the fitted column layout.

        Returns:
            Probabilities in ``[0, 1]``, strictly increasing in the index.
        """
        if not self._is_fitted:
            raise RuntimeError("VectorIsotonicCalibrator has not been fitted.")
        calibrated: npt.NDArray[np.float64] = np.clip(
            np.asarray(self._isotonic.calibrate(self.index(score_matrix)), dtype=np.float64),
            0.0,
            1.0,
        )
        return calibrated

    def to_state(self) -> dict[str, Any]:
        """Plain-array state for checkpointing (round-trips via :meth:`from_state`)."""
        if self._weights is None:
            raise RuntimeError("VectorIsotonicCalibrator has not been fitted.")
        iso = self._isotonic._iso.model
        return {
            "margin_power": self._margin_power,
            "weights": self._weights.copy(),
            "iso_x": None if iso._x is None else iso._x.copy(),
            "iso_y": None if iso._y is None else iso._y.copy(),
            "eps": self._isotonic.eps,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> VectorIsotonicCalibrator:
        """Rebuild a fitted calibrator from :meth:`to_state` output."""
        instance = cls(margin_power=int(state["margin_power"]))
        instance._weights = np.asarray(state["weights"], dtype=np.float64)
        instance._isotonic = StrictIsotonicCalibration(eps=float(state["eps"]))
        iso = instance._isotonic._iso
        if state["iso_x"] is not None:
            iso.model._x = np.asarray(state["iso_x"], dtype=np.float64)
            iso.model._y = np.asarray(state["iso_y"], dtype=np.float64)
            iso._fitted = True
            instance._isotonic._fitted = True
        instance._is_fitted = True
        return instance


class VectorConformalBinary:
    """Mondrian split-conformal prediction sets over {normal, anomaly}.

    Args:
        alpha: Miscoverage level; each class's coverage is guaranteed at
            least ``1 − alpha`` in finite samples under exchangeability.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """Initialize the instance."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._calibrator: VectorIsotonicCalibrator | None = None
        #: Sorted nonconformity scores per class (1: anomaly, 0: normal).
        self._nonconformity: dict[int, npt.NDArray[np.float64]] = {}

    @property
    def alpha(self) -> float:
        """The configured miscoverage level."""
        return self._alpha

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return bool(self._nonconformity) and self._calibrator is not None

    @staticmethod
    def _score(index: npt.NDArray[np.float64], label: int) -> npt.NDArray[np.float64]:
        """Nonconformity of ``index`` values under a hypothesised class."""
        return -index if label == 1 else index

    def fit(
        self,
        score_matrix: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
        calibrator: VectorIsotonicCalibrator,
    ) -> VectorConformalBinary:
        """Store each class's calibration nonconformity scores.

        Args:
            score_matrix: ``(n, k)`` calibration-fold scores.
            labels: Calibration-fold ground truth.
            calibrator: A fitted vector calibrator supplying the index.

        Returns:
            ``self``.

        Raises:
            ValueError: If a class is absent from the calibration fold.
        """
        if not calibrator.is_fitted:
            raise ValueError("the vector calibrator must be fitted first")
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        index = calibrator.index(score_matrix)
        self._nonconformity = {}
        for label in (0, 1):
            member = index[y == label]
            if member.size == 0:
                raise ValueError(f"class {label} absent from the calibration fold")
            self._nonconformity[label] = np.sort(self._score(member, label))
        self._calibrator = calibrator
        return self

    def p_values(self, score_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per-class conformal p-values, shape ``(n, 2)`` (columns: class 0, 1).

        The p-value is the (n+1)-rank one:
        ``(1 + #{calibration scores >= new score}) / (n_class + 1)``.
        """
        if not self.is_fitted or self._calibrator is None:
            raise RuntimeError("VectorConformalBinary has not been fitted.")
        index = self._calibrator.index(score_matrix)
        out = np.empty((len(index), 2), dtype=np.float64)
        for label in (0, 1):
            reference = self._nonconformity[label]
            scores = self._score(index, label)
            # Count of calibration nonconformity >= the new score.
            greater_equal = reference.size - np.searchsorted(reference, scores, side="left")
            out[:, label] = (1.0 + greater_equal) / (reference.size + 1.0)
        return out

    def predict_sets(self, score_matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """Prediction sets over {normal, anomaly}.

        Shape ``(n, 2)``: class ``c`` is in a sample's set iff its conformal
        p-value exceeds ``alpha``.
        """
        return self.p_values(score_matrix) > self._alpha

    def to_state(self) -> dict[str, Any]:
        """Plain-array state for checkpointing (calibrator stored separately)."""
        if not self._nonconformity:
            raise RuntimeError("VectorConformalBinary has not been fitted.")
        return {
            "alpha": self._alpha,
            "nonconformity_0": self._nonconformity[0].copy(),
            "nonconformity_1": self._nonconformity[1].copy(),
        }

    @classmethod
    def from_state(
        cls, state: dict[str, Any], calibrator: VectorIsotonicCalibrator
    ) -> VectorConformalBinary:
        """Rebuild a fitted predictor from :meth:`to_state` output."""
        instance = cls(alpha=float(state["alpha"]))
        instance._nonconformity = {
            0: np.asarray(state["nonconformity_0"], dtype=np.float64),
            1: np.asarray(state["nonconformity_1"], dtype=np.float64),
        }
        instance._calibrator = calibrator
        return instance

    def coverage_report(
        self,
        score_matrix: npt.NDArray[np.float64],
        labels: npt.NDArray[np.int64],
    ) -> dict[str, Any]:
        """Empirical coverage and efficiency on a labelled held-out fold.

        Returns:
            ``target_coverage`` (``1 − alpha``), ``empirical_coverage``
            (fraction of rows whose true class is in the set),
            ``average_set_size`` and per-class coverage.
        """
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        sets = self.predict_sets(score_matrix)
        covered = sets[np.arange(len(y)), y]
        per_class = {}
        for label in (0, 1):
            member = covered[y == label]
            per_class[f"coverage_class_{label}"] = (
                float(member.mean()) if member.size else float("nan")
            )
        return {
            "target_coverage": 1.0 - self._alpha,
            "empirical_coverage": float(covered.mean()),
            "average_set_size": float(sets.sum(axis=1).mean()),
            **per_class,
        }

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Frequent-pattern / association-rule mining anomaly detector.

Association-rule mining (Agrawal & Srikant, *Fast Algorithms for Mining
Association Rules*, VLDB 1994) discovers high-confidence implications ``A ⇒ b``
that hold across a set of transactions. For anomaly detection the learned rules
encode the *co-occurrence grammar* of normal records — categorical event
bundles, feature co-activations, co-firing detectors — and a record is anomalous
when it violates rules it should satisfy: the antecedent is present but the
expected consequent is missing (FP-outlier / rule-violation scoring, He et al.,
2004).

This detector mines frequent itemsets by Apriori and derives single-consequent
rules above a confidence floor from training transactions; each transaction is
scored by its confidence-weighted rule-violation mass, squashed into ``[0, 1]``.
Pure NumPy (always importable); registered as an opt-in BASE detector.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.detectors._calibration import (
    bound_finite,
    finite_features,
    finite_scores,
    squash_scale,
)

if TYPE_CHECKING:
    import torch

__all__ = ["FrequentPatternDetector"]

logger = logging.getLogger(__name__)

_LN2 = float(np.log(2.0))


class FrequentPatternDetector(BaseDetector):
    """Apriori association-rule violation detector for binary transactions.

    Input is a ``(n_transactions, n_items)`` binary/boolean matrix (non-zero =
    item present). :meth:`fit` mines frequent itemsets and single-consequent
    rules ``A ⇒ b`` above the support/confidence floors; :meth:`detect` scores
    each transaction by its confidence-weighted violation mass (antecedent
    present, consequent absent), squashed into ``[0, 1]``.
    """

    def __init__(
        self,
        min_support: float = 0.1,
        min_confidence: float = 0.8,
        max_itemset: int = 3,
        calibration_quantile: float = 0.98,
        max_items: int = 128,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the frequent-pattern detector.

        Args:
            min_support: Minimum fraction of transactions an itemset must appear
                in to be *frequent*. Must be in ``(0, 1]``.
            min_confidence: Minimum rule confidence ``P(b | A)`` to keep a rule.
                Must be in ``(0, 1]``.
            max_itemset: Maximum itemset size mined by Apriori (rule antecedent
                size is up to ``max_itemset - 1``). Must be >= 2.
            calibration_quantile: Training violation-mass quantile at the 0.5
                boundary; ``1 - calibration_quantile`` is the normal-regime FPR.
                Must be in ``(0, 1)``.
            max_items: Per-level cap on the number of frequent itemsets carried
                to the next Apriori join, keeping the highest-support itemsets.
                This bounds candidate generation to ``O(max_items**2)`` per level
                so mining cannot blow up combinatorially on pathological input
                (e.g. a single wide transaction where every item has support
                ``1.0`` and nothing prunes). Must be >= 2. Truncation is logged,
                never silent.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if not 0.0 < min_support <= 1.0:
            raise ValueError(f"min_support must be in (0, 1], got {min_support}")
        if not 0.0 < min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}")
        if max_itemset < 2:
            raise ValueError(f"max_itemset must be >= 2, got {max_itemset}")
        if not 0.0 < calibration_quantile < 1.0:
            raise ValueError(f"calibration_quantile must be in (0, 1), got {calibration_quantile}")
        if max_items < 2:
            raise ValueError(f"max_items must be >= 2, got {max_items}")
        self.min_support = float(min_support)
        self.min_confidence = float(min_confidence)
        self.max_itemset = int(max_itemset)
        self.calibration_quantile = float(calibration_quantile)
        self.max_items = int(max_items)
        # Each rule: (antecedent frozenset, consequent int, confidence float).
        self._rules: list[tuple[frozenset[int], int, float]] = []
        self._scale: float = 1.0

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has mined rules and the scale."""
        return self._is_fitted

    def _to_bool_matrix(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a 2-D boolean transaction matrix."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = bound_finite(np.asarray(data, dtype=np.float64), detector=self.name)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr != 0.0

    def _squash_scale(self, raw: np.ndarray[Any, Any]) -> float:
        """Squash scale anchoring the ``calibration_quantile`` at score 0.5."""
        return squash_scale(raw, self.calibration_quantile)

    def _cap_by_support(
        self, scored: list[tuple[frozenset[int], float]], level: int
    ) -> list[frozenset[int]]:
        """Keep the ``max_items`` highest-support itemsets; log any truncation.

        Ordering is deterministic (support descending, then the sorted item
        tuple) so mining stays reproducible even when the cap bites. Bounding
        each level to ``max_items`` makes the next join ``O(max_items**2)``.
        """
        if len(scored) <= self.max_items:
            return [iset for iset, _ in scored]
        scored.sort(key=lambda kv: (-kv[1], tuple(sorted(kv[0]))))
        dropped = len(scored) - self.max_items
        logger.debug(
            "frequent_pattern: level-%d frequent itemsets truncated to %d "
            "(dropped %d lowest-support) to bound Apriori candidate growth",
            level,
            self.max_items,
            dropped,
        )
        return [iset for iset, _ in scored[: self.max_items]]

    def _mine_frequent(self, mat: np.ndarray[Any, Any]) -> dict[frozenset[int], float]:
        """Apriori frequent-itemset mining; returns itemset → support.

        Each level is capped to the ``max_items`` most-supported itemsets before
        the next join, which bounds candidate generation and guarantees the miner
        terminates quickly on any input, including degenerate single-transaction
        matrices where nothing prunes.
        """
        _, n_items = mat.shape
        support: dict[frozenset[int], float] = {}
        # Level 1: frequent single items (capped to the highest-support ones).
        col_support = mat.mean(axis=0)
        singles = [
            (frozenset([j]), float(col_support[j]))
            for j in range(n_items)
            if col_support[j] >= self.min_support
        ]
        current = self._cap_by_support(singles, 1)
        for iset in current:
            support[iset] = float(col_support[next(iter(iset))])
        # Levels 2..max_itemset: join, score, prune to the support floor, cap.
        k = 2
        while current and k <= self.max_itemset:
            candidates: set[frozenset[int]] = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    union = current[i] | current[j]
                    if len(union) == k:
                        candidates.add(union)
            scored: list[tuple[frozenset[int], float]] = []
            for cand in candidates:
                cols = list(cand)
                sup = float(mat[:, cols].all(axis=1).mean())
                if sup >= self.min_support:
                    scored.append((cand, sup))
            current = self._cap_by_support(scored, k)
            for cand in current:
                cols = list(cand)
                support[cand] = float(mat[:, cols].all(axis=1).mean())
            k += 1
        return support

    def _mine_rules(self, support: dict[frozenset[int], float]) -> None:
        """Derive single-consequent rules ``A ⇒ b`` above the confidence floor."""
        rules: list[tuple[frozenset[int], int, float]] = []
        for itemset, sup in support.items():
            if len(itemset) < 2:
                continue
            for consequent in itemset:
                antecedent = itemset - {consequent}
                ant_sup = support.get(antecedent)
                if not ant_sup:
                    continue
                confidence = sup / ant_sup
                if confidence >= self.min_confidence:
                    rules.append((antecedent, int(consequent), float(confidence)))
        self._rules = rules

    def _violation_mass(self, mat: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Confidence-weighted rule-violation mass per transaction.

        Rules store *absolute* item (column) indices learned at ``fit`` time. A
        ``detect`` batch may be narrower than the training vocabulary (a shorter
        1-D series reshaped to one transaction, or a 2-D matrix with fewer
        columns), so any rule that references a column beyond the current width
        is simply skipped for that batch rather than dereferenced out of bounds.
        """
        n_tx = mat.shape[0]
        out = np.zeros(n_tx, dtype=np.float64)
        if not self._rules or mat.shape[1] == 0:
            return out
        width = mat.shape[1]
        applicable = [
            (ant, con, conf)
            for ant, con, conf in self._rules
            if con < width and all(a < width for a in ant)
        ]
        if not applicable:
            return out
        for t in range(n_tx):
            row = mat[t]
            mass = 0.0
            for antecedent, consequent, confidence in applicable:
                if all(row[a] for a in antecedent) and not row[consequent]:
                    mass += confidence
            out[t] = mass
        return out

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> FrequentPatternDetector:
        """Mine frequent itemsets + rules and set the violation squash scale.

        Args:
            data: Training transactions ``(n_transactions, n_items)`` (binary).

        Returns:
            ``self``.
        """
        mat = self._to_bool_matrix(data)
        support = self._mine_frequent(mat)
        self._mine_rules(support)
        raw = self._violation_mass(mat)
        self._scale = self._squash_scale(raw)
        self._is_fitted = True
        return self

    def extract_features(self, data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Per-transaction fusion feature: the rule-violation mass.

        Args:
            data: Input transactions ``(n_transactions, n_items)``.

        Returns:
            ``(n_transactions, 1)`` float32 violation masses.
        """
        raw = self._violation_mass(self._to_bool_matrix(data))
        return finite_features(raw, detector=self.name).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-transaction anomaly scores in ``[0, 1]`` from rule violations.

        Violation mass is squashed via ``1 - exp(-m / scale)``;
        ``metadata['n_rules']`` reports how many rules were mined.
        """
        raw = self._violation_mass(self._to_bool_matrix(data))
        scale = self._scale if self._is_fitted else self._squash_scale(raw)
        scores = finite_scores(1.0 - np.exp(-raw / scale), detector=self.name).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {"n_rules": len(self._rules)},
        }

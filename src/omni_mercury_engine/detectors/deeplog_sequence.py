# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""DeepLog-style sequence / log-template anomaly detector.

DeepLog (Du et al., *DeepLog: Anomaly Detection and Diagnosis from System Logs
through Deep Learning*, CCS 2017) models a stream of discrete log-template keys
as a language: it learns to predict the next key from a window of preceding
keys, and flags a position as anomalous when the observed key is not among the
model's top-``g`` predictions — i.e. the execution path departs from the grammar
of normal logs.

DeepLog's predictor is an LSTM; this implementation realises the same *detection
principle* with a back-off n-gram transition model, which needs no GPU/PyTorch,
trains in a single pass, and yields a proper next-key distribution
``P(key | context)``. The anomaly score is the model's next-key *miss
probability* ``1 - P(key | context)`` in ``[0, 1]`` (calibration-free: near 0
for confident, in-grammar predictions and near 1 for novel keys/contexts), and
each position also reports whether the key fell outside the top-``g``
predictions. Pure NumPy (always importable); registered as an opt-in BASE
detector.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from omni_mercury_engine.core.base import BaseDetector

if TYPE_CHECKING:
    import torch

__all__ = ["DeepLogSequenceDetector"]



class DeepLogSequenceDetector(BaseDetector):
    """Back-off n-gram next-key surprisal detector for log/event sequences.

    Input is a 1-D sequence of non-negative integer log-template keys. :meth:`fit`
    accumulates n-gram transition counts (with back-off to shorter contexts and a
    global unigram); :meth:`detect` scores each position by the next-key miss
    probability ``1 - P(key | context)`` and reports whether the key fell outside
    the top-``top_g`` predictions.
    """

    def __init__(
        self,
        order: int = 3,
        top_g: int = 3,
        smoothing: float = 1.0,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the DeepLog-style sequence detector.

        Args:
            order: Context length ``n`` of the n-gram model (keys of history).
                Must be >= 1.
            top_g: A key outside the model's top-``g`` next-key predictions is
                marked anomalous (DeepLog's ``g`` parameter). Must be >= 1.
            smoothing: Additive (Laplace) smoothing count. Must be > 0.
            config: Optional ``BaseDetector`` config (``threshold`` ...).

        Raises:
            ValueError: If any parameter is out of its valid range.
        """
        super().__init__(config)
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if top_g < 1:
            raise ValueError(f"top_g must be >= 1, got {top_g}")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be > 0")
        self.order = int(order)
        self.top_g = int(top_g)
        self.smoothing = float(smoothing)
        # counts[k] maps a context tuple of length k -> {next_key: count}.
        self._counts: list[dict[tuple[int, ...], dict[int, int]]] = []
        self._vocab: set[int] = set()
        self._unigram: dict[int, int] = {}

    def is_fitted(self) -> bool:
        """Return ``True`` once :meth:`fit` has accumulated counts."""
        return self._is_fitted

    @staticmethod
    def _to_1d_int(data: np.ndarray[Any, Any] | torch.Tensor) -> np.ndarray[Any, Any]:
        """Coerce numpy/torch input to a finite 1-D non-negative int sequence."""
        detach = getattr(data, "detach", None)
        if callable(detach):
            data = detach().cpu().numpy()
        arr = np.nan_to_num(np.asarray(data)).ravel()
        return np.rint(arr).astype(np.int64)

    def _prob_and_rank(self, context: tuple[int, ...], key: int) -> tuple[float, int]:
        """Back-off ``P(key | context)`` and the rank of ``key`` among next keys.

        Contexts are tried longest-first; the first non-empty context table is
        used. Probabilities are Laplace-smoothed over the observed vocabulary.
        """
        vocab_size = max(len(self._vocab), 1)
        for k in range(min(self.order, len(context)), 0, -1):
            table = self._counts[k - 1].get(tuple(context[-k:]))
            if table:
                total = sum(table.values()) + self.smoothing * vocab_size
                count = table.get(key, 0) + self.smoothing
                prob = count / total
                # Rank of `key` by descending smoothed probability.
                better = sum(1 for c in table.values() if c + self.smoothing > count)
                return prob, better + 1
        # Back-off to the global unigram distribution.
        total = sum(self._unigram.values()) + self.smoothing * vocab_size
        count = self._unigram.get(key, 0) + self.smoothing
        prob = count / total
        better = sum(1 for c in self._unigram.values() if c + self.smoothing > count)
        return prob, better + 1

    def _eval(
        self, seq: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Per-position surprisal, next-key probability, and top-g violations."""
        n = seq.size
        surp = np.zeros(n, dtype=np.float64)
        probs = np.zeros(n, dtype=np.float64)
        flags = np.zeros(n, dtype=bool)
        if n == 0 or not self._counts:
            return surp, probs, flags
        for t in range(n):
            context = tuple(int(v) for v in seq[max(0, t - self.order) : t])
            prob, rank = self._prob_and_rank(context, int(seq[t]))
            probs[t] = prob
            surp[t] = -np.log(max(prob, 1e-12))
            flags[t] = rank > self.top_g
        return surp, probs, flags

    def fit(self, data: np.ndarray[Any, Any] | torch.Tensor) -> DeepLogSequenceDetector:
        """Accumulate back-off n-gram counts and the surprisal squash scale.

        Args:
            data: Training sequence of integer log-template keys.

        Returns:
            ``self``.
        """
        seq = self._to_1d_int(data)
        self._counts = [defaultdict(lambda: defaultdict(int)) for _ in range(self.order)]
        self._vocab = set(int(v) for v in seq.tolist())
        self._unigram = defaultdict(int)
        for v in seq.tolist():
            self._unigram[int(v)] += 1
        for t in range(seq.size):
            key = int(seq[t])
            for k in range(1, self.order + 1):
                if t - k < 0:
                    break
                context = tuple(int(v) for v in seq[t - k : t])
                self._counts[k - 1][context][key] += 1
        # Freeze defaultdicts into plain dicts for stable lookups.
        self._counts = [dict(c) for c in self._counts]
        self._unigram = dict(self._unigram)
        self._is_fitted = True
        return self

    def extract_features(
        self, data: np.ndarray[Any, Any] | torch.Tensor
    ) -> np.ndarray[Any, Any]:
        """Per-position fusion feature: the next-key surprisal.

        Args:
            data: Input integer key sequence.

        Returns:
            ``(n_keys, 1)`` float32 surprisals.
        """
        surp, _, _ = self._eval(self._to_1d_int(data))
        return surp.astype(np.float32).reshape(-1, 1)

    def detect(self, data: np.ndarray[Any, Any] | torch.Tensor) -> dict[str, Any]:
        """Per-position anomaly scores in ``[0, 1]`` from next-key miss probability.

        The score is the model's next-key *miss probability* ``1 - P(key |
        context)`` — calibration-free and naturally near 0 for confident,
        in-grammar predictions and near 1 for novel keys/contexts (unlike a
        squashed surprisal, which degenerates on low-entropy grammars).
        ``metadata['top_g_violations']`` counts positions whose key fell outside
        the model's top-``top_g`` predictions.
        """
        seq = self._to_1d_int(data)
        _, probs, flags = self._eval(seq)
        scores = np.clip(1.0 - probs, 0.0, 1.0).astype(np.float32)
        return {
            "anomaly_score": float(scores.max()) if scores.size else 0.0,
            "scores": scores,
            "is_anomaly": scores > self.threshold,
            "confidence": scores,
            "metadata": {"top_g_violations": int(np.count_nonzero(flags))},
        }

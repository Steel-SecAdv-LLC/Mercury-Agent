# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""SPOT thread-safety / purity tests.

Verifies the refactor that made :meth:`SPOTDetector.detect` a pure read of the
fitted tail state:

* :meth:`SPOTDetector._tail_probability` / ``_threshold_from_tail`` are pure
  functions of their explicit arguments (no instance state);
* ``detect`` is idempotent (repeated calls return identical scores) and mutates
  no fitted attribute;
* two ``detect`` calls running concurrently on the *same* instance return the
  same result as serial calls -- no state corruption.
"""

from __future__ import annotations

import threading

import numpy as np

from omni_mercury_engine.detectors.spot_evt import SPOTDetector

_FITTED_ATTRS = ("_t", "_zq", "_gamma", "_sigma", "_n", "_nt")


def _fitted() -> tuple[SPOTDetector, np.ndarray]:
    rng = np.random.default_rng(7)
    train = rng.normal(0.0, 1.0, 2000)
    test = rng.normal(0.0, 1.0, 1200)
    test[600] = 14.0
    return SPOTDetector(q=1e-3, init_level=0.98).fit(train), test


class TestPurity:
    def test_tail_probability_is_pure(self) -> None:
        # Same arguments -> same result, independent of any instance.
        a = SPOTDetector._tail_probability(3.0, 0.0, 1000, 20, 0.1, 1.0)
        b = SPOTDetector._tail_probability(3.0, 0.0, 1000, 20, 0.1, 1.0)
        assert a == b
        # It reads no instance state: callable as an unbound static method.
        assert 0.0 <= a <= 1.0

    def test_threshold_from_tail_is_pure(self) -> None:
        a = SPOTDetector._threshold_from_tail(0.0, 1e-3, 1000, 20, 0.0, 1.0)
        b = SPOTDetector._threshold_from_tail(0.0, 1e-3, 1000, 20, 0.0, 1.0)
        assert a == b

    def test_detect_does_not_mutate_fitted_state(self) -> None:
        det, test = _fitted()
        snapshot = {a: getattr(det, a) for a in _FITTED_ATTRS}
        det.detect(test)
        after = {a: getattr(det, a) for a in _FITTED_ATTRS}
        assert snapshot == after, "detect() must not mutate the fitted tail state"

    def test_detect_is_idempotent(self) -> None:
        det, test = _fitted()
        r1 = np.asarray(det.detect(test)["scores"])
        r2 = np.asarray(det.detect(test)["scores"])
        r3 = np.asarray(det.detect(test)["scores"])
        assert np.array_equal(r1, r2)
        assert np.array_equal(r2, r3)


class TestConcurrency:
    def test_parallel_detect_no_corruption(self) -> None:
        det, test = _fitted()
        baseline = np.asarray(det.detect(test)["scores"])

        results: dict[int, np.ndarray] = {}
        errors: list[Exception] = []

        def work(k: int) -> None:
            try:
                results[k] = np.asarray(det.detect(test)["scores"])
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(k,)) for k in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent detect raised: {errors}"
        assert len(results) == 16
        for scores in results.values():
            assert np.array_equal(scores, baseline), "concurrent result diverged from serial"

    def test_parallel_detect_on_different_inputs(self) -> None:
        # Two different inputs scored concurrently must each match their serial result.
        det, _ = _fitted()
        rng = np.random.default_rng(99)
        a = rng.normal(size=800)
        a[400] = 12.0
        b = rng.normal(size=800)
        b[100] = -12.0
        exp_a = np.asarray(det.detect(a)["scores"])
        exp_b = np.asarray(det.detect(b)["scores"])
        out: dict[str, np.ndarray] = {}

        def work(key: str, data: np.ndarray) -> None:
            out[key] = np.asarray(det.detect(data)["scores"])

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=work, args=("a", a)))
            threads.append(threading.Thread(target=work, args=("b", b)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert np.array_equal(out["a"], exp_a)
        assert np.array_equal(out["b"], exp_b)

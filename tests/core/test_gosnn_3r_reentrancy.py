# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``GOSNN3RIntegration.adjust_weights`` returns.

It holds the integration's lock and calls ``verify_stability``, which takes the
same lock. The lock was a plain ``threading.Lock``, so the call deadlocked on
itself -- and ``NeuroSymbolicHub.predict`` with its default settings called it
on every prediction and never returned.
"""

from __future__ import annotations

import threading
from typing import Any

from omni_mercury_engine.core.gosnn_3r_integration import GOSNN3RIntegration

_TIMEOUT_S = 20.0


def _call_with_timeout(fn: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run ``fn`` on a daemon thread so a deadlock fails the test instead of hanging it."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - surfaced below
            box["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(_TIMEOUT_S)
    assert not worker.is_alive(), f"{getattr(fn, '__name__', fn)} did not return: deadlock"
    assert "error" not in box, box.get("error")
    result: dict[str, Any] = box["value"]
    return result


def test_adjust_weights_returns_and_reports_stability() -> None:
    integration = GOSNN3RIntegration(enable_sliding_window=False)
    result = _call_with_timeout(
        integration.adjust_weights, neural_score=0.7, symbolic_score=0.2, confidence=0.4
    )
    assert isinstance(result, dict)
    assert result, "adjust_weights returned an empty report"


def test_adjust_weights_is_safe_from_several_threads() -> None:
    integration = GOSNN3RIntegration(enable_sliding_window=False)
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        try:
            for step in range(25):
                integration.adjust_weights(
                    neural_score=((seed + step) % 10) / 10.0,
                    symbolic_score=((seed * 3 + step) % 10) / 10.0,
                    confidence=((seed + 2 * step) % 10) / 10.0,
                )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(seed,), daemon=True) for seed in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads), "adjust_weights deadlocked"
    assert not errors, errors

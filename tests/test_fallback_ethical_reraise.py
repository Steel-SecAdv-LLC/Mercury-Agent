# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression test for FallbackChain ethical-error propagation.

The Wave B dual-gate contract requires that
``EthicalConstraintViolationError`` raised by *any* handler in a
``FallbackChain`` is re-raised unconditionally — regardless of the
``fail_fast`` flag and regardless of whether later handlers might
succeed.  A prior version of ``FallbackChain.execute()`` had a
generic ``except Exception`` that swallowed the ethical refusal and
tried the next handler, effectively masking the hard gate.

This test exercises both ``fail_fast=True`` and ``fail_fast=False``
paths to ensure the ethical gate is never bypassed.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omni_mercury_engine.cognitive.ethical_bounding import (
    EthicalConstraintViolationError,
)
from omni_mercury_engine.integrations.routing.fallback import FallbackChain


def _run(coro: Any) -> Any:
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


class TestFallbackChainEthicalReraise:
    """Verify EthicalConstraintViolationError is never masked."""

    @staticmethod
    def _make_chain(fail_fast: bool) -> FallbackChain:
        chain = FallbackChain(name="ethical-test", fail_fast=fail_fast)

        async def ethical_refuser(*_args: Any, **_kwargs: Any) -> None:
            raise EthicalConstraintViolationError(
                action="test_action",
                score=0.10,
                threshold=0.99,
                check="benevolence",
            )

        async def fallback_handler(*_args: Any, **_kwargs: Any) -> Any:
            return {"status": "degraded"}

        chain.add_handler(ethical_refuser, name="ethical_refuser", priority=0)
        chain.add_handler(fallback_handler, name="safe_fallback", priority=1)
        return chain

    def test_ethical_error_propagates_fail_fast_false(self) -> None:
        """fail_fast=False must NOT suppress EthicalConstraintViolationError."""
        chain = self._make_chain(fail_fast=False)
        with pytest.raises(EthicalConstraintViolationError, match="benevolence"):
            _run(chain.execute())

    def test_ethical_error_propagates_fail_fast_true(self) -> None:
        """fail_fast=True must NOT suppress EthicalConstraintViolationError."""
        chain = self._make_chain(fail_fast=True)
        with pytest.raises(EthicalConstraintViolationError, match="benevolence"):
            _run(chain.execute())

    def test_fallback_handler_never_reached(self) -> None:
        """The safe_fallback handler must never execute after an ethical refusal."""
        reached = {"fallback": False}

        chain = FallbackChain(name="reach-test", fail_fast=False)

        async def ethical_refuser(*_args: Any, **_kwargs: Any) -> None:
            raise EthicalConstraintViolationError(
                action="test_action",
                score=0.05,
                threshold=0.99,
                check="sigma_immutable",
            )

        async def spy_fallback(*_args: Any, **_kwargs: Any) -> Any:
            reached["fallback"] = True
            return {"status": "should-not-reach"}

        chain.add_handler(ethical_refuser, name="ethical_refuser", priority=0)
        chain.add_handler(spy_fallback, name="spy_fallback", priority=1)

        with pytest.raises(EthicalConstraintViolationError):
            _run(chain.execute())

        assert not reached["fallback"], (
            "Fallback handler was reached after EthicalConstraintViolationError; "
            "the hard ethical gate was bypassed."
        )

    def test_non_ethical_error_still_falls_back(self) -> None:
        """Non-ethical exceptions should still trigger normal fallback logic."""
        chain = FallbackChain(name="normal-fallback-test", fail_fast=False)

        async def failing_handler(*_args: Any, **_kwargs: Any) -> None:
            raise ValueError("transient failure")

        async def recovery_handler(*_args: Any, **_kwargs: Any) -> Any:
            return {"recovered": True}

        chain.add_handler(failing_handler, name="failing", priority=0)
        chain.add_handler(recovery_handler, name="recovery", priority=1)

        result = _run(chain.execute())
        assert result.value == {"recovered": True}
        assert result.fallback_count >= 1

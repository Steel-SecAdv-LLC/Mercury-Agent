# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline-first routing across Mercury's reasoning backends.

The router encodes one policy: **offline-first**. A local (offline) backend is
always required and is the default; a remote (network-capable) backend is
optional and is reached only when a call explicitly opts in *and* the router is
not in hard-offline mode. Hard-offline mode is a hard guarantee — a
network-capable backend is never selected and therefore never called — so an
air-gapped deployment can prove it makes zero outbound reasoning calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_mercury_engine.reasoning.backend import ReasoningBackend
    from omni_mercury_engine.reasoning.schemas import (
        Explanation,
        Hypothesis,
        ReasoningContext,
        Report,
    )

__all__ = ["ReasoningRouter"]


class ReasoningRouter:
    """Select and delegate to a reasoning backend under an offline-first policy.

    Attributes:
        local: The required offline backend (the floor; always available).
        remote: An optional network-capable backend.
        hard_offline: When True, the remote backend is never selected.
        prefer_remote: When True, calls default to the remote backend (subject
            to ``hard_offline`` and the remote backend being configured).
    """

    def __init__(
        self,
        *,
        local: ReasoningBackend,
        remote: ReasoningBackend | None = None,
        hard_offline: bool = False,
        prefer_remote: bool = False,
    ) -> None:
        """Initialize the router.

        Args:
            local: Required offline backend. Must report ``is_offline`` True so
                the offline-first floor genuinely never needs the network.
            remote: Optional network-capable backend.
            hard_offline: When True, never select the remote backend.
            prefer_remote: When True, default to the remote backend.

        Raises:
            ValueError: If ``local`` is not offline (it would not be a safe
                floor), or if ``prefer_remote`` is set without a remote backend.
        """
        if not local.is_offline:
            raise ValueError(
                "local backend must be offline (is_offline=True) to serve as "
                "the offline-first floor"
            )
        if prefer_remote and remote is None:
            raise ValueError("prefer_remote=True requires a remote backend")
        self.local = local
        self.remote = remote
        self.hard_offline = hard_offline
        self.prefer_remote = prefer_remote

    def select(self, *, allow_remote: bool = False) -> ReasoningBackend:
        """Return the backend to use for a call.

        The remote backend is chosen only when it exists, the call allows it
        (``allow_remote`` per call, or :attr:`prefer_remote` as the configured
        default), and neither hard-offline mode nor the ``MERCURY_OFFLINE``
        master air-gap is active. With ``prefer_remote=True`` the remote backend
        is the default for every call; otherwise the local backend is the
        default and remote requires per-call ``allow_remote=True``. The air-gap
        always wins — under hard-offline or ``MERCURY_OFFLINE`` the local backend
        is returned regardless of ``prefer_remote``/``allow_remote``.

        Args:
            allow_remote: Opt into remote escalation for this call (moot when
                ``prefer_remote`` already makes remote the default).

        Returns:
            The selected backend.
        """
        from omni_mercury_engine.datasets.exceptions import offline_mode_active

        if self.hard_offline or offline_mode_active() or self.remote is None:
            return self.local
        if allow_remote or self.prefer_remote:
            return self.remote
        return self.local

    def explain(self, context: ReasoningContext, *, allow_remote: bool = False) -> Explanation:
        """Route :meth:`ReasoningBackend.explain` under the offline-first policy."""
        return self.select(allow_remote=allow_remote).explain(context)

    def propose_hypotheses(
        self, evidence: ReasoningContext, *, allow_remote: bool = False
    ) -> list[Hypothesis]:
        """Route :meth:`ReasoningBackend.propose_hypotheses` under the policy."""
        return self.select(allow_remote=allow_remote).propose_hypotheses(evidence)

    def synthesize_report(
        self, findings: ReasoningContext, *, allow_remote: bool = False
    ) -> Report:
        """Route :meth:`ReasoningBackend.synthesize_report` under the policy."""
        return self.select(allow_remote=allow_remote).synthesize_report(findings)

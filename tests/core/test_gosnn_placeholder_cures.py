# Copyright (C) 2025 Steel Security Advisors LLC
"""Phase 2 ITEM 3 regression: GOSNN placeholder cures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Cure 1+2: AttentionProvider replaces the random-tensor fallback.
# ---------------------------------------------------------------------------


class TestAttentionProviderCure:
    def test_no_provider_skips_metric_with_recommendation(self) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import GOSNNOptimizer

        reset_global_network()
        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer()
        result = optimizer.optimize(gosnn)

        # The skip path must surface in recommendations so downstream
        # auditors can see the metric was not computed.
        assert any("attention overhead metric skipped" in r.lower() for r in result.recommendations)

    def test_provider_get_attention_runtime_error_skips_with_recommendation(self) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import (
            AttentionProvider,
            GOSNNOptimizer,
        )

        class _BoomProvider(AttentionProvider):
            def get_attention(self) -> np.ndarray:
                raise RuntimeError("model not yet wired")

        reset_global_network()
        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer(attention_provider=_BoomProvider())
        result = optimizer.optimize(gosnn)

        assert any("attention overhead metric skipped" in r.lower() for r in result.recommendations)

    def test_real_tensors_drive_metric_when_provider_succeeds(self) -> None:
        from omni_mercury_engine.core.global_omni_scalar_network import (
            GlobalOmniScalarNetwork,
            reset_global_network,
        )
        from omni_mercury_engine.core.gosnn_optimizer import (
            AttentionProvider,
            GOSNNOptimizer,
        )

        class _UnitProvider(AttentionProvider):
            def get_attention(self) -> np.ndarray:
                return np.ones((32, 16, 16), dtype=np.float64)

        reset_global_network()
        gosnn = GlobalOmniScalarNetwork()
        optimizer = GOSNNOptimizer(attention_provider=_UnitProvider())
        result = optimizer.optimize(gosnn)

        # When real tensors flow, the recommendation list does NOT
        # contain the skip-marker.  The optimizer either reports
        # in-budget overhead or surfaces an over-budget recommendation
        # — never the placeholder skip text.
        assert not any(
            "attention overhead metric skipped" in r.lower() for r in result.recommendations
        )


# ---------------------------------------------------------------------------
# Cure 3: Conformal prediction failures propagate (no silent swallow).
# ---------------------------------------------------------------------------


class TestConformalPropagation:
    def test_conformal_predict_error_propagates(self) -> None:
        from omni_mercury_engine.core.gosnn_integration import GOSNNIntegration

        integration = GOSNNIntegration(use_conformal=False)
        integration.add_domain(name="test")

        # Inject a predictor whose .predict raises.  The previous
        # behaviour was to swallow this and return
        # ``confidence_intervals=None``; the contract now is to
        # propagate so callers cannot silently miss the failure.
        class _BoomConformal:
            def predict(self, X: Any) -> None:
                raise ValueError("calibration history exhausted")

        integration._conformal = _BoomConformal()  # type: ignore[assignment]
        # Mark as fitted so predict() will exercise the path.
        integration._fitted = True

        with pytest.raises(ValueError, match="calibration history exhausted"):
            integration.detect(np.zeros((4, 3)), use_cache=False)

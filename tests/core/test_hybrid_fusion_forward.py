# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The HybridFusionLayer forward is the dense attention path — pinned.

The layer's docstring used to describe an early+late+sparse-attention
pipeline the forward never ran, and its ``enable_sparse_attention`` /
``enable_entropy_weighting`` flags configure submodules that are constructed
(for checkpoint shape compatibility) but never called. The measured ablation
(``artifacts/fusion_mode_ablation.json``) evaluated the *actual* forward, so
this module pins it: the fused representation is the dense
:class:`AttentionFusion` output whatever the flags say, the early/late side
computations are detached diagnostics, and a flag flip changes nothing about
the output. Wiring an inert module into the path is an architecture change
that must be re-measured; this test is what makes that change visible.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from omni_mercury_engine.core.fusion import HybridFusionLayer

FEATURE_DIMS = {"statistical": 6, "temporal": 4, "spectral": 5}


def _inputs(seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    features = {
        name: torch.randn(8, dim, generator=generator) for name, dim in FEATURE_DIMS.items()
    }
    scores = {name: torch.rand(8, 1, generator=generator) for name in FEATURE_DIMS}
    return features, scores


def _layer(**kwargs: object) -> HybridFusionLayer:
    torch.manual_seed(0)
    return HybridFusionLayer(feature_dims=FEATURE_DIMS, hidden_dim=16, num_heads=2, **kwargs)


def test_the_forward_is_the_dense_attention_output() -> None:
    layer = _layer()
    layer.eval()
    features, scores = _inputs(1)
    with torch.no_grad():
        fused, diagnostics = layer(features, scores)
        projected = [layer.feature_projectors[name](features[name]) for name in FEATURE_DIMS]
        expected, _ = layer.attention(torch.stack(projected, dim=1), return_attention=True)
    torch.testing.assert_close(fused, expected)
    # The side computations exist, detached, in the diagnostics only.
    assert set(diagnostics) >= {"early_contribution", "late_contribution"}
    assert not diagnostics["early_contribution"].requires_grad
    assert not diagnostics["late_contribution"].requires_grad


def test_the_inert_flags_do_not_change_the_output() -> None:
    """The flags configure constructed-but-uncalled submodules only.

    Construction order consumes RNG draws, so the comparison shares one set
    of weights across the variants (state-dict copy; ``strict=False`` where a
    flag removes an inert module entirely) — then the forward must be
    bit-identical whatever the flags say.
    """
    features, scores = _inputs(2)
    baseline = _layer(enable_sparse_attention=True, enable_entropy_weighting=True)
    baseline.eval()
    with torch.no_grad():
        reference, _ = baseline(features, scores)

    for kwargs in (
        {"enable_sparse_attention": False, "enable_entropy_weighting": False},
        {"uncertainty_lambda": 0.9, "resonance_lambda": 0.9},
    ):
        variant = _layer(**kwargs)
        variant.load_state_dict(baseline.state_dict(), strict=False)
        variant.eval()
        with torch.no_grad():
            fused, _ = variant(features, scores)
        np.testing.assert_array_equal(fused.numpy(), reference.numpy())


def test_the_inert_modules_stay_constructed_for_checkpoint_shape() -> None:
    """Removing them breaks shipped state-dict loads; they must stay."""
    layer = _layer()
    names = set(layer.state_dict().keys())
    assert any(key.startswith("uncertainty_fusion") for key in names)
    assert any(key.startswith("resonance_fusion") for key in names)
    assert any(key.startswith("sparse_attention") for key in names)

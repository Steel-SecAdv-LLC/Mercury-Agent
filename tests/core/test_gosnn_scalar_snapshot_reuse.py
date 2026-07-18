# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The advisory and authoritative gates must share one scalar snapshot.

``detect_with_fusion`` used to collect the full 127-scalar operational vector
twice per call: once inside ``get_enhanced_scalars`` (for GOSNN's advisory
ethical gate) and again for the authoritative σ_Immutable gate. Two
independent collections are wasteful (a redundant registry walk under lock)
and a latent signal-integrity gap -- a concurrent registration between them
could leave the two gates evaluating divergent vectors. The enhancement now
carries the snapshot it scored (``EnhancementResult.collected_scalars``) and
the engine reuses it. This suite pins that contract.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from omni_mercury_engine.core.global_omni_scalar_network import get_global_scalar_network


def test_enhancement_carries_the_collected_snapshot() -> None:
    """get_enhanced_scalars exposes the exact scalars it gated against."""
    net = get_global_scalar_network()
    result = net.get_enhanced_scalars(
        requesting_component="test_snapshot_reuse",
        base_scalars={"detector_a_score": 0.3, "detector_b_score": 0.7},
        context={"domain": None},
    )
    assert result.collected_scalars is not None
    # It must equal an immediately-taken fresh collection (single-threaded:
    # nothing registered in between), key-for-key and value-for-value.
    fresh = net._collect_all_scalars()
    assert result.collected_scalars == fresh
    # And it is the vector the advisory gate scored -- non-empty, all finite.
    values = np.array(list(result.collected_scalars.values()), dtype=np.float64)
    assert values.size > 0
    assert np.all(np.isfinite(values))


def test_reused_snapshot_matches_a_fresh_collection_order_and_values() -> None:
    """The reused snapshot yields the identical σ_Immutable scalar vector.

    The authoritative gate builds ``np.array(list(collected.values()))``; the
    reuse must produce the byte-identical vector a fresh collection would, so
    the σ_Immutable score cannot shift from the change.
    """
    net = get_global_scalar_network()
    result = net.get_enhanced_scalars(
        requesting_component="test_snapshot_reuse_vec",
        base_scalars={"detector_a_score": 0.1},
        context={"domain": None},
    )
    assert result.collected_scalars is not None
    reused = np.array(list(result.collected_scalars.values()), dtype=np.float64)
    fresh = np.array(list(net._collect_all_scalars().values()), dtype=np.float64)
    np.testing.assert_array_equal(reused, fresh)

# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scoped intra-op threading control for tiny-tensor torch sections.

The torch-gated tier detectors (srcnn, diffusion_ad) train very small models
on very small tensors — e.g. one full-batch epoch over a ``[n, 1, 33]`` conv
input.  At that size the per-op OpenMP fork/join overhead dominates the
arithmetic: measured on the srcnn fit loop, one epoch costs ~460 ms with 4
intra-op threads and ~34 ms with 1 (a 13x penalty), which made the tier's
auto-fit path pathologically slow on multi-core hosts.  torch exposes only a
process-global knob, so :func:`single_threaded_torch` pins it to 1 for the
duration of a tight loop and restores the previous value afterwards.

Concurrency note: if two registry worker threads overlap, the later restore
may briefly leave the other's section running with the ambient thread count —
a transient performance effect only, never a correctness one (thread count
does not change what is computed, just how reductions are scheduled).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["single_threaded_torch"]


@contextmanager
def single_threaded_torch() -> Iterator[None]:
    """Run a tiny-tensor torch section with intra-op parallelism disabled."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)

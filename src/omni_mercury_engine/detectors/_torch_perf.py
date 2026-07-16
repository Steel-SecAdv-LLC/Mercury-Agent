# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scoped intra-op threading control for tiny-tensor torch sections.

The torch-gated tier detectors (srcnn, diffusion_ad) train very small models
on very small tensors — e.g. one full-batch epoch over a ``[n, 1, 33]`` conv
input.  At that size the per-op OpenMP fork/join overhead dominates the
arithmetic: measured on the srcnn fit loop, one epoch costs ~460 ms with 4
intra-op threads and ~34 ms with 1 (a 13x penalty), which made the tier's
auto-fit pathologically slow on multi-core hosts.  torch exposes only a
process-global knob, so :func:`single_threaded_torch` pins it to 1 for the
duration of a tight loop and restores the previous value afterwards.

``torch`` is imported optionally: a torch-free install (no ``[ml]`` extra)
can still import this module and enter :func:`single_threaded_torch`, which
degrades to a no-op.  This keeps torch-optional importers such as
``detectors.acceleration_dynamics`` importable without the extra.

Numeric note: intra-op parallelism changes how reductions are *scheduled*, not
which algorithm runs, so pinning to one thread never changes results in a
semantically meaningful way.  It can, however, reorder floating-point
reductions and therefore produce differences at the last bit or two — the same
non-determinism any change in thread count carries.  Do not rely on this
context manager for reproducibility; use it only for the latency win.

Concurrency: the num-threads knob is process-global, so a naive
save-set-restore races when registry worker threads overlap — one section's
restore can clobber another's, and interleaving can leave the process
permanently pinned to one thread (A saves 4→sets 1; B saves 1→sets 1; A
restores 4; B restores 1).  The detector registry drives detectors on a
``ThreadPoolExecutor``, so overlap is expected.  :func:`single_threaded_torch`
therefore reference-counts entries under a lock: the first concurrent entrant
records the ambient thread count and pins to 1, further overlapping entrants
share that state, and only the last one to exit restores the recorded value.
The lock is held only around the tiny book-keeping, never across the caller's
body, so overlapping single-threaded sections still run concurrently.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["single_threaded_torch"]

# Guards the reference count and the saved thread value below.  A plain Lock
# suffices because it is never held across the ``yield``; same-thread nesting
# does not re-acquire it while held, so it cannot self-deadlock.
_state_lock = threading.Lock()
_active_depth = 0
_saved_threads: int | None = None


@contextmanager
def single_threaded_torch() -> Iterator[None]:
    """Run a tiny-tensor torch section with intra-op parallelism disabled.

    Reference-counted and thread-safe: concurrent sections share a single
    pin-to-one-thread window and the ambient thread count is restored only
    once every overlapping section has exited.

    Degrades to a no-op when torch is unavailable so torch-optional callers
    stay importable and runnable without the ``[ml]`` extra.
    """
    if not TORCH_AVAILABLE:
        yield
        return
    global _active_depth, _saved_threads
    with _state_lock:
        if _active_depth == 0:
            _saved_threads = torch.get_num_threads()
            torch.set_num_threads(1)
        _active_depth += 1
    try:
        yield
    finally:
        with _state_lock:
            _active_depth -= 1
            if _active_depth == 0 and _saved_threads is not None:
                torch.set_num_threads(_saved_threads)
                _saved_threads = None

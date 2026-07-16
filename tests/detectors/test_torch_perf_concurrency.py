# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Concurrency contract for :func:`single_threaded_torch`.

The intra-op thread knob is process-global, so a naive save/set/restore races
when registry worker threads overlap: one section's restore can clobber
another's and interleaving can leave the process permanently pinned to a single
thread. These tests pin the reference-counted, thread-safe behaviour.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("torch")

import torch

from omni_mercury_engine.detectors._torch_perf import single_threaded_torch


def test_restores_ambient_count_after_single_use() -> None:
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        with single_threaded_torch():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(original)


def test_nested_same_thread_restores_once_at_outer_exit() -> None:
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(5)
        with single_threaded_torch():
            assert torch.get_num_threads() == 1
            with single_threaded_torch():
                assert torch.get_num_threads() == 1
            # Inner exit must NOT restore while an outer section is still active.
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 5
    finally:
        torch.set_num_threads(original)


def test_overlapping_sections_do_not_leave_process_pinned() -> None:
    """Deterministically interleave two sections so a naive implementation would
    leave the process stuck at one thread; the reference-counted version restores
    the ambient count once the last overlapping section exits.
    """
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(4)

        a_entered = threading.Event()
        b_entered = threading.Event()
        a_exited = threading.Event()
        errors: list[str] = []

        def worker_a() -> None:
            with single_threaded_torch():
                a_entered.set()
                if not b_entered.wait(10):  # keep A's section open until B enters
                    errors.append("B never entered")
            a_exited.set()  # A leaves first, while B is still inside

        def worker_b() -> None:
            if not a_entered.wait(10):  # enter only after A has pinned to 1
                errors.append("A never entered")
                return
            with single_threaded_torch():
                b_entered.set()
                if not a_exited.wait(10):  # stay inside until A has fully exited
                    errors.append("A never exited")

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        tb.start()
        ta.join(15)
        tb.join(15)

        assert not errors, errors
        assert not ta.is_alive() and not tb.is_alive()
        # The last section to exit restores the true ambient value, not the
        # transient "1" a mid-flight section would have observed and saved.
        assert torch.get_num_threads() == 4
    finally:
        torch.set_num_threads(original)

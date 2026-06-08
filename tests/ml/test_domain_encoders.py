# Copyright (C) 2025 Steel Security Advisors LLC
"""Unit tests for the differentiable domain encoders (WS-B / Target 2)."""

from __future__ import annotations

from typing import cast

import pytest
from torch import nn

pytest.importorskip("torch")

import torch

from omni_mercury_engine.ml.domain_encoders import (
    DomainEncoderStack,
    FisherEntropyEncoder,
    KinematicEncoder,
    SpectralEncoder,
)

_EncoderClass = type[SpectralEncoder] | type[KinematicEncoder] | type[FisherEntropyEncoder]
_ENCODERS: tuple[_EncoderClass, ...] = (SpectralEncoder, KinematicEncoder, FisherEntropyEncoder)


@pytest.mark.parametrize("cls", _ENCODERS)
@pytest.mark.parametrize("in_dim", [8, 13, 21])
def test_encoder_shape_and_finite(cls: _EncoderClass, in_dim: int) -> None:
    torch.manual_seed(0)
    enc = cls(in_dim, hidden_dim=32, output_dim=64)
    out = enc(torch.randn(16, in_dim))
    assert out.shape == (16, 64)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("cls", _ENCODERS)
def test_encoder_is_differentiable(cls: _EncoderClass) -> None:
    """Gradient must flow back to the input -- the whole point of WS-B."""
    torch.manual_seed(0)
    enc = cls(21, hidden_dim=32, output_dim=64)
    x = torch.randn(8, 21, requires_grad=True)
    enc(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # at least one parameter receives a gradient
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_kinematic_conv_initialised_to_finite_difference() -> None:
    """The kinematic channels start exactly at velocity/accel/jerk operators."""
    enc = KinematicEncoder(21)
    expected = [[-1.0, 1.0], [1.0, -2.0, 1.0], [-1.0, 3.0, -3.0, 1.0]]
    for conv, exp in zip(enc.convs, expected):
        conv1d = cast("nn.Conv1d", conv)
        assert torch.detach(conv1d.weight).view(-1).tolist() == exp


def test_spectral_uses_fft_grad_path() -> None:
    """SpectralEncoder's FFT magnitude path is differentiable w.r.t. the input."""
    torch.manual_seed(1)
    enc = SpectralEncoder(16, hidden_dim=16, output_dim=32)
    x = torch.randn(4, 16, requires_grad=True)
    enc(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


def test_stack_shape_and_eval_determinism() -> None:
    torch.manual_seed(0)
    stack = DomainEncoderStack(21, hidden_dim=32, per_encoder_dim=32, output_dim=128)
    stack.eval()
    x = torch.randn(10, 21)
    a = stack(x)
    b = stack(x)
    assert a.shape == (10, 128)
    assert torch.equal(a, b)  # eval mode (dropout off) is deterministic


def test_stack_rejects_unknown_domain() -> None:
    with pytest.raises(ValueError):
        DomainEncoderStack(21, domains=("spectral", "nonexistent"))

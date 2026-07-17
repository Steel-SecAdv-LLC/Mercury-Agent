# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""The single trainable module stack behind ``MultiHeadAttentionFusion``.

This module is the **only** definition of the GOSNN attention-fusion
architecture and of its forward computation.  Both consumers import it:

* the production serve path
  (:class:`omni_mercury_engine.core.global_omni_scalar_network.MultiHeadAttentionFusion`
  builds its modules with :func:`build_fusion_modules` and fuses through
  :func:`fuse_members`), and
* the training program (``scripts/train_gosnn_fusion.py`` trains a
  :class:`TrainableFusionStack` and ships its state dicts).

History: the training script used to *reimplement* "the exact production
fuse() module stack" as a private ``_FusionModel`` — and the two had already
drifted (the serve path applied a harmonic-synergy output modulation the
training/merit-gate path never saw, so the shipped checkpoint's gate verdict
was measured under different semantics than it served with).  A single shared
definition makes that class of train/serve divergence structurally
impossible.

This module requires torch (import it lazily / behind ``TORCH_AVAILABLE``
guards from torch-optional callers).
"""

from __future__ import annotations

import torch
from torch import nn

#: Canonical architecture constants (the shipped ``gosnn_attention_fusion``
#: checkpoint is trained at exactly these sizes).
DEFAULT_D_MODEL = 512
DEFAULT_NUM_HEADS = 32
DEFAULT_MAX_DIMENSIONS = 37
DETECTION_HIDDEN = 16


def build_fusion_modules(
    d_model: int = DEFAULT_D_MODEL,
    num_heads: int = DEFAULT_NUM_HEADS,
    max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
) -> tuple[nn.Linear, nn.MultiheadAttention, nn.Linear]:
    """Construct the three fusion modules (projection, attention, output).

    Args:
        d_model: Attention model dimension.
        num_heads: Number of attention heads.
        max_dimensions: Width of the padded member state vectors.

    Returns:
        ``(projection, attention, output_projection)`` — the exact module
        stack whose state dicts the shipped checkpoint carries under the
        payload keys ``"projection"`` / ``"attention"`` /
        ``"output_projection"``.
    """
    projection = nn.Linear(max_dimensions, d_model)
    attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
    output_projection = nn.Linear(d_model, max_dimensions)
    return projection, attention, output_projection


def fuse_members(
    projection: nn.Linear,
    attention: nn.MultiheadAttention,
    output_projection: nn.Linear,
    members: torch.Tensor,
) -> torch.Tensor:
    """The canonical fusion forward pass: ``(n_members, W) -> (W,)``.

    Projects each member state vector to ``d_model``, self-attends across
    members, projects back to member width, and mean-pools over members.
    This is the semantics the merit gate measures — the serve path must not
    add computation this function does not have (that was the historical
    train/serve drift).

    Args:
        projection: Member -> model-dimension projection.
        attention: Multi-head self-attention over members.
        output_projection: Model-dimension -> member-width projection.
        members: ``(n_members, max_dimensions)`` stacked member states.

    Returns:
        The fused ``(max_dimensions,)`` state vector.
    """
    projected = projection(members).unsqueeze(0)
    attended, _ = attention(projected, projected, projected)
    fused: torch.Tensor = output_projection(attended.squeeze(0)).mean(dim=0)
    return fused


class FusionDetectionHead(nn.Module):
    """Detection head over the fused state: standardise -> MLP -> logit.

    The head turns the 37-dimensional fused state into a single anomaly
    logit.  Train-split standardisation statistics are registered as buffers
    so they ship inside the head's own ``state_dict`` — serve-time inputs are
    standardised exactly as training inputs were, with no side-channel
    artefact to keep in sync.
    """

    #: Standardisation buffers (annotated so mypy resolves them as tensors
    #: rather than through ``nn.Module.__getattr__``'s ``Tensor | Module``).
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(
        self, max_dimensions: int = DEFAULT_MAX_DIMENSIONS, hidden: int = DETECTION_HIDDEN
    ) -> None:
        """Initialize the head.

        Args:
            max_dimensions: Width of the fused state vector.
            hidden: Hidden layer width.
        """
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.register_buffer("feature_mean", torch.zeros(max_dimensions))
        self.register_buffer("feature_std", torch.ones(max_dimensions))
        self.net = nn.Sequential(
            nn.Linear(max_dimensions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def set_standardizer(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install train-split standardisation statistics.

        Args:
            mean: Per-dimension mean of the training fused states.
            std: Per-dimension std of the training fused states; zero entries
                are replaced with 1.0 (constant dimensions carry no signal and
                must not divide by zero).
        """
        safe_std = torch.where(std > 0, std, torch.ones_like(std))
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(safe_std)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Anomaly logit for fused state(s).

        Args:
            fused: ``(max_dimensions,)`` or ``(batch, max_dimensions)`` fused
                state.

        Returns:
            Logit tensor of shape ``()`` or ``(batch,)``.
        """
        z = (fused - self.feature_mean) / self.feature_std
        out: torch.Tensor = self.net(z).squeeze(-1)
        return out


class TrainableFusionStack(nn.Module):
    """The production fusion stack plus detection head, trainable end-to-end.

    ``forward`` is exactly the serve-path fusion (:func:`fuse_members`);
    :meth:`detection_logit` runs the fused state through the
    :class:`FusionDetectionHead`.  The training program optimises both
    jointly so the fused representation is shaped by the real detection
    objective rather than by a reconstruction proxy alone.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        num_heads: int = DEFAULT_NUM_HEADS,
        max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
        detection_hidden: int = DETECTION_HIDDEN,
    ) -> None:
        """Initialize the stack.

        Args:
            d_model: Attention model dimension.
            num_heads: Number of attention heads.
            max_dimensions: Width of the padded member state vectors.
            detection_hidden: Hidden width of the detection head.
        """
        super().__init__()  # type: ignore[no-untyped-call, unused-ignore]
        self.projection, self.attention, self.output_projection = build_fusion_modules(
            d_model=d_model, num_heads=num_heads, max_dimensions=max_dimensions
        )
        self.detection_head = FusionDetectionHead(
            max_dimensions=max_dimensions, hidden=detection_hidden
        )

    def forward(self, members: torch.Tensor) -> torch.Tensor:
        """Fuse ``(n_members, max_dimensions)`` members into ``(max_dimensions,)``.

        Args:
            members: Stacked member state vectors.

        Returns:
            The fused state vector.
        """
        return fuse_members(self.projection, self.attention, self.output_projection, members)

    def detection_logit(self, members: torch.Tensor) -> torch.Tensor:
        """Anomaly logit for one member stack (fuse, then head).

        Args:
            members: ``(n_members, max_dimensions)`` stacked member states.

        Returns:
            Scalar logit tensor.
        """
        logit: torch.Tensor = self.detection_head(self.forward(members))
        return logit


__all__ = [
    "DEFAULT_D_MODEL",
    "DEFAULT_MAX_DIMENSIONS",
    "DEFAULT_NUM_HEADS",
    "DETECTION_HIDDEN",
    "FusionDetectionHead",
    "TrainableFusionStack",
    "build_fusion_modules",
    "fuse_members",
]

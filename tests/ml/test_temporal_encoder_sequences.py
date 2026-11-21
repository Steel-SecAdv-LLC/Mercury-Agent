# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``TemporalEncoder`` accepts the sequences its docstring promises.

Its LSTM is built with ``input_size=1`` but ``forward`` handed it the
``[batch, seq_len, input_dim]`` tensor as it came, which raises for any
``input_dim`` other than 1. The branch was never reached in the fusion model
(3-D input is flattened first), so nothing failed. The LSTM's shape is in
the released default fusion checkpoint and stays; a multi-feature sequence is
now encoded one feature channel at a time.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
TemporalEncoder = pytest.importorskip("omni_mercury_engine.ml.encoders").TemporalEncoder


def _encoder(input_dim: int = 32, hidden_dim: int = 16, output_dim: int = 24) -> Any:
    torch.manual_seed(0)
    return TemporalEncoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).eval()


class TestSequenceInput:
    def test_a_sequence_of_input_dim_features_is_encoded(self) -> None:
        """``[batch, seq_len, input_dim]`` used to raise in the LSTM."""
        encoder = _encoder(input_dim=32)
        x = torch.randn(4, 10, 32)

        embedding = encoder(x)

        assert embedding.shape == (4, 24)
        assert torch.isfinite(embedding).all()

    @pytest.mark.parametrize("n_features", [1, 3, 32, 64])
    def test_any_number_of_features_per_step_is_encoded(self, n_features: int) -> None:
        encoder = _encoder(input_dim=32)

        embedding = encoder(torch.randn(2, 7, n_features))

        assert embedding.shape == (2, 24)

    def test_a_univariate_sequence_is_the_plain_lstm(self) -> None:
        """One feature per step: the path that already worked is unchanged."""
        encoder = _encoder()
        x = torch.randn(5, 12, 1)

        with torch.no_grad():
            _out, (hidden, _cell) = encoder.lstm(x)
            expected = encoder.projection(hidden[-1])
            actual = encoder(x)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_a_multi_feature_sequence_is_the_mean_of_its_channels(self) -> None:
        encoder = _encoder()
        x = torch.randn(3, 9, 4)

        with torch.no_grad():
            per_channel = torch.stack(
                [encoder.lstm(x[:, :, c : c + 1])[1][0][-1] for c in range(4)], dim=1
            )
            expected = encoder.projection(per_channel.mean(dim=1))
            actual = encoder(x)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_the_embedding_depends_on_the_order_of_the_steps(self) -> None:
        """It is a sequence encoder, not a bag of values."""
        encoder = _encoder()
        x = torch.randn(2, 16, 5)

        with torch.no_grad():
            forward = encoder(x)
            backward = encoder(torch.flip(x, dims=[1]))

        assert not torch.allclose(forward, backward)

    def test_gradients_reach_the_lstm(self) -> None:
        encoder = _encoder().train()

        encoder(torch.randn(3, 6, 32)).sum().backward()

        assert encoder.lstm.weight_ih_l0.grad is not None
        assert float(encoder.lstm.weight_ih_l0.grad.abs().sum()) > 0.0


class TestOtherInputs:
    def test_pre_extracted_embeddings_take_the_projection(self) -> None:
        encoder = _encoder(input_dim=32)
        x = torch.randn(5, 32)

        with torch.no_grad():
            torch.testing.assert_close(encoder(x), encoder.embedding_projection(x))

    def test_other_ranks_are_refused(self) -> None:
        encoder = _encoder()

        with pytest.raises(ValueError, match="2D or 3D"):
            encoder(torch.randn(2, 3, 4, 5))


class TestReleasedCheckpointsStillLoad:
    def test_the_lstm_keeps_the_shape_the_checkpoints_hold(self) -> None:
        """``weight_ih_l0`` is ``[128, 1]`` in the released ``default_fusion.pt``."""
        encoder = TemporalEncoder(input_dim=64, hidden_dim=32, output_dim=64, num_layers=2)

        shapes = {name: tuple(value.shape) for name, value in encoder.state_dict().items()}

        assert shapes["lstm.weight_ih_l0"] == (128, 1)
        assert shapes["lstm.weight_hh_l0"] == (128, 32)
        assert shapes["projection.weight"] == (64, 32)
        assert shapes["embedding_projection.0.weight"] == (64, 64)

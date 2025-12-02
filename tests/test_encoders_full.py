"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Comprehensive encoder tests to boost coverage
"""

import torch
from omni_anomaly_engine.ml.encoders import (
    StatisticalEncoder,
    TemporalEncoder,
    BiometricEncoder,
    QuantumEncoder,
    AstrophysicalEncoder,
    AffectiveEncoder,
)
from omni_anomaly_engine.ml.harmonic_encoder import HarmonicEncoder


def test_statistical_encoder_with_precomputed():
    """Test StatisticalEncoder with precomputed embeddings"""
    encoder = StatisticalEncoder(input_dim=10, output_dim=128)

    embedding = torch.randn(5, 10)
    output = encoder(embedding)

    assert output.shape[0] == 5
    assert output.shape[1] == 128


def test_temporal_encoder_with_precomputed():
    """Test TemporalEncoder with precomputed embeddings"""
    encoder = TemporalEncoder(input_dim=32, output_dim=128)

    embedding = torch.randn(5, 32)
    output = encoder(embedding)

    assert output.shape == (5, 128)


def test_biometric_encoder_with_2d():
    """Test BiometricEncoder with 2D embeddings"""
    encoder = BiometricEncoder(embedding_dim=128, output_dim=128)

    embedding = torch.randn(5, 128)
    output = encoder(embedding)

    assert output.dim() == 2
    assert output.shape[1] == 128


def test_biometric_encoder_with_4d():
    """Test BiometricEncoder with 4D images"""
    encoder = BiometricEncoder(input_channels=3, output_dim=128)

    images = torch.randn(5, 3, 224, 224)
    output = encoder(images)

    assert output.shape[0] == 5
    assert output.shape[1] == 128


def test_quantum_encoder_with_precomputed():
    """Test QuantumEncoder with complex quantum states"""
    encoder = QuantumEncoder(state_dim=16, output_dim=128)

    quantum_states = torch.randn(5, 16, 2)
    output = encoder(quantum_states)

    assert output.shape == (5, 128)


def test_astrophysical_encoder_with_precomputed():
    """Test AstrophysicalEncoder with precomputed embeddings"""
    encoder = AstrophysicalEncoder(input_dim=32, output_dim=128)

    embedding = torch.randn(5, 32)
    output = encoder(embedding)

    assert output.shape == (5, 128)


def test_affective_encoder_with_precomputed():
    """Test AffectiveEncoder with precomputed embeddings"""
    encoder = AffectiveEncoder(input_dim=32, output_dim=128)

    embedding = torch.randn(5, 32)
    output = encoder(embedding)

    assert output.shape == (5, 128)


def test_harmonic_encoder_initialization():
    """Test HarmonicEncoder initialization"""
    encoder = HarmonicEncoder(l_max=5)
    assert encoder.spherical_decomposer.l_max == 5


def test_harmonic_encoder_forward():
    """Test HarmonicEncoder forward pass with signal"""
    encoder = HarmonicEncoder(l_max=3, output_dim=32)

    signal = torch.randn(50)
    output = encoder(signal=signal)

    assert output.shape[0] == 32


def test_encoders_batch_processing():
    """Test all encoders with batch processing"""
    encoders = [
        (StatisticalEncoder(input_dim=32, output_dim=128), torch.randn(10, 32)),
        (TemporalEncoder(input_dim=32, output_dim=128), torch.randn(10, 32)),
        (BiometricEncoder(embedding_dim=128, output_dim=128), torch.randn(10, 128)),
        (QuantumEncoder(state_dim=16, output_dim=128), torch.randn(10, 16, 2)),
        (AstrophysicalEncoder(input_dim=32, output_dim=128), torch.randn(10, 32)),
        (AffectiveEncoder(input_dim=32, output_dim=128), torch.randn(10, 32)),
    ]

    for encoder, batch in encoders:
        output = encoder(batch)
        assert output.shape[0] == 10
        assert output.shape[1] == 128

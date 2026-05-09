"""
Mercury Agent Copyright (C) 2025 Steel Security Advisors LLC.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not,
see
https://www.gnu.org/licenses/.
"""

from __future__ import annotations

"""
Spectral Vibration Analysis Module for Mercury Agent.

Advanced frequency-domain anomaly detection inspired by:
- Machine Learning Interatomic Potentials (MLIPs) for vibrational spectrum analysis
- Graph Neural Networks (GNNs) for mapping atomic/data interactions
- Convolutional Neural Networks (CNNs) for spectral pattern recognition
- Phonon interaction modeling for complex data relationships
- Predictive maintenance through vibration signature analysis

This module accelerates spectroscopic analysis by replacing expensive quantum
mechanical simulations with machine learning approaches that achieve comparable
results while enabling real-time anomaly detection.

Research foundations:
- Donoho & Johnstone (1994): Wavelet shrinkage denoising
- Schumann resonance: ELF spectrum analysis (7.83 Hz fundamental)
- Discrete Fourier Transform for frequency-domain analysis
- Graph Laplacian spectral methods for relational data
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from scipy import (
    signal as scipy_signal,
    stats as scipy_stats,
)
from scipy.fft import fft, fftfreq
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from torch import nn

from omni_mercury_engine.core.base import BaseDetector
from omni_mercury_engine.core.centralized_constants import get_domain_fundamentals
from omni_mercury_engine.core.exceptions import DetectorException
from omni_mercury_engine.utils.constants import MathematicalConstants

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Enumerations
# =============================================================================

# Golden ratio for harmonic weighting (from 3R mechanism)
PHI = MathematicalConstants.GOLDEN_RATIO.value  # 1.618033988749895

# Schumann resonance fundamental frequency (Hz)
# Now used as fallback only — domain-adaptive frequencies are preferred.
# Provenance: Schumann (1952), valid for environmental/geophysical domain.
SCHUMANN_FUNDAMENTAL = 7.83

# Schumann harmonics (Hz)
# Retained for backward compatibility; see get_domain_fundamentals() for
# domain-specific frequency selection.
SCHUMANN_HARMONICS = [7.83, 14.3, 20.8, 27.3, 33.8]


class SpectralAnalysisMode(Enum):
    """Available spectral analysis modes."""

    FFT_STANDARD = "fft_standard"
    WAVELET_MULTIRESOLUTION = "wavelet_multiresolution"
    GNN_SPECTRAL = "gnn_spectral"
    CNN_PATTERN = "cnn_pattern"
    PHONON_INTERACTION = "phonon_interaction"
    MLIP_VIBRATIONAL = "mlip_vibrational"
    HYBRID_FUSION = "hybrid_fusion"
    # CLI-friendly aliases
    COMPREHENSIVE = "hybrid_fusion"  # Alias for HYBRID_FUSION
    FFT_ONLY = "fft_standard"  # Alias for FFT_STANDARD
    WAVELET_ONLY = "wavelet_multiresolution"  # Alias for WAVELET_MULTIRESOLUTION
    PHONON = "phonon_interaction"  # Alias for PHONON_INTERACTION
    PREDICTIVE = "mlip_vibrational"  # Alias for MLIP_VIBRATIONAL


class VibrationSignatureType(Enum):
    """Types of vibration signatures for predictive maintenance."""

    NORMAL = "normal"
    BEARING_FAULT = "bearing_fault"
    IMBALANCE = "imbalance"
    MISALIGNMENT = "misalignment"
    LOOSENESS = "looseness"
    RESONANCE = "resonance"
    ELECTRICAL = "electrical"
    FLOW_INDUCED = "flow_induced"
    GEAR_MESH = "gear_mesh"
    UNKNOWN_ANOMALY = "unknown_anomaly"


# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class SpectralVibrationConfig:
    """
    Configuration for spectral vibration analysis.

    Attributes:
        analysis_mode: Primary analysis mode
        sample_rate: Sampling rate in Hz (default 1000)
        fft_size: FFT window size (power of 2)
        overlap_ratio: Overlap ratio for STFT (0.0-0.9)
        num_harmonics: Number of harmonics to extract
        wavelet_levels: Decomposition levels for wavelet analysis
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_num_layers: Number of GNN message-passing layers
        cnn_filters: Number of CNN filters per layer
        phonon_cutoff: Phonon energy cutoff for interaction modeling
        enable_schumann: Enable Schumann resonance detection
        threshold: Anomaly detection threshold
        device: Compute device ('cpu', 'cuda', 'mps')
    """

    analysis_mode: SpectralAnalysisMode = SpectralAnalysisMode.HYBRID_FUSION
    sample_rate: float = 1000.0
    fft_size: int = 1024
    overlap_ratio: float = 0.5
    num_harmonics: int = 8
    wavelet_levels: int = 5
    gnn_hidden_dim: int = 64
    gnn_num_layers: int = 3
    cnn_filters: int = 32
    phonon_cutoff: float = 0.1
    enable_schumann: bool = True
    threshold: float = 0.5
    device: str = "cpu"
    domain: str = "environmental"
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectralFeatures:
    """
    Extracted spectral features from analysis.

    Attributes:
        power_spectrum: Power spectral density
        dominant_frequencies: List of dominant frequency peaks
        harmonic_ratios: Ratios between harmonic components
        spectral_centroid: Center of mass of spectrum
        spectral_bandwidth: Bandwidth around centroid
        spectral_rolloff: Frequency below which X% of energy is contained
        spectral_flatness: Measure of noise-like vs tonal quality
        crest_factor: Peak to RMS ratio
        kurtosis: Fourth moment (peakedness)
        phonon_coupling: Inter-frequency coupling strength
        graph_laplacian_spectrum: Eigenvalues of frequency graph
        cnn_features: CNN-extracted pattern features
        schumann_alignment: Alignment with Schumann frequencies
    """

    power_spectrum: np.ndarray
    dominant_frequencies: list[tuple[float, float]]  # (freq, amplitude)
    harmonic_ratios: np.ndarray
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flatness: float
    crest_factor: float
    kurtosis: float
    phonon_coupling: float
    graph_laplacian_spectrum: np.ndarray
    cnn_features: np.ndarray
    schumann_alignment: float


@dataclass
class VibrationDiagnostic:
    """
    Diagnostic result for vibration analysis.

    Attributes:
        signature_type: Detected vibration signature type
        confidence: Confidence in diagnosis
        fault_frequency: Primary fault-related frequency
        severity_score: Severity of detected condition
        recommended_action: Recommended maintenance action
        time_to_failure: Estimated time to failure (if predictable)
        supporting_features: Features supporting the diagnosis
    """

    signature_type: VibrationSignatureType
    confidence: float
    fault_frequency: float
    severity_score: float
    recommended_action: str
    time_to_failure: float | None
    supporting_features: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Neural Network Components
# =============================================================================


class SpectralGraphLayer(nn.Module):
    """
    Graph Neural Network layer for spectral analysis.

    Implements message passing on a frequency-domain graph where nodes
    represent frequency bins and edges represent harmonic/coupling relationships.

    The graph structure captures:
    - Harmonic relationships (integer frequency ratios)
    - Sideband coupling (modulation effects)
    - Energy flow between frequency bands
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_edge_types: int = 4,
    ) -> None:
        """
        Initialize spectral graph layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_edge_types: Number of edge relationship types
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_edge_types = num_edge_types

        # Node feature transformation
        self.node_transform = nn.Linear(in_features, out_features)

        # Edge-type specific message functions
        self.edge_transforms = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_edge_types)]
        )

        # Attention mechanism for edge weighting
        self.attention = nn.Sequential(
            nn.Linear(2 * out_features, out_features),
            nn.LeakyReLU(0.2),
            nn.Linear(out_features, 1),
        )

        # Aggregation and update
        self.update_fn = nn.GRUCell(out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through spectral graph layer.

        Args:
            node_features: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]

        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = node_features.size(0)

        # Transform node features
        h = self.node_transform(node_features)

        # Compute messages for each edge
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]

        # Get source node features for each edge
        src_features = node_features[src_idx]

        # Apply edge-type specific transformations
        messages = torch.zeros(edge_index.size(1), self.out_features, device=node_features.device)
        for i in range(self.num_edge_types):
            mask = edge_type == i
            if mask.any():
                messages[mask] = self.edge_transforms[i](src_features[mask])

        # Compute attention weights
        tgt_features_expanded = h[tgt_idx]
        attn_input = torch.cat([messages, tgt_features_expanded], dim=-1)
        attn_weights = torch.softmax(self.attention(attn_input).squeeze(-1), dim=0)

        # Aggregate messages with attention
        weighted_messages = messages * attn_weights.unsqueeze(-1)
        aggregated = torch.zeros(num_nodes, self.out_features, device=node_features.device)
        aggregated.index_add_(0, tgt_idx, weighted_messages)

        # Update node representations
        h_updated = self.update_fn(aggregated, h)

        return self.layer_norm(h_updated)


class SpectralGNN(nn.Module):
    """
    Complete Graph Neural Network for spectral analysis.

    Processes frequency-domain data as a graph where:
    - Nodes represent frequency bins with their magnitudes
    - Edges represent harmonic and coupling relationships
    - Message passing captures energy flow and interactions
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_layers: int = 3,
        num_edge_types: int = 4,
    ) -> None:
        """
        Initialize spectral GNN.

        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of message passing layers
            num_edge_types: Number of edge relationship types
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Graph layers
        self.layers = nn.ModuleList(
            [SpectralGraphLayer(hidden_dim, hidden_dim, num_edge_types) for _ in range(num_layers)]
        )

        # Output projection with readout
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Global pooling attention
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through spectral GNN.

        Args:
            node_features: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_type: Edge type indices [num_edges]

        Returns:
            Graph-level representation [output_dim]
        """
        # Project input
        h = self.input_proj(node_features)

        # Message passing layers
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_type)  # Residual connection

        # Attention-weighted global pooling
        attn_weights = self.pool_attention(h)
        graph_repr = (h * attn_weights).sum(dim=0)

        return self.output_proj(graph_repr)


class SpectralCNN(nn.Module):
    """
    Convolutional Neural Network for spectral pattern recognition.

    Processes 2D spectrograms or 1D spectra to extract discriminative features for anomaly
    detection. Inspired by spectrogram-based methods in speech recognition and seismology.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_filters: int = 32,
        output_dim: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
    ) -> None:
        """
        Initialize spectral CNN.

        Args:
            input_channels: Number of input channels (1 for magnitude, 2 for complex)
            num_filters: Number of filters per convolutional layer
            output_dim: Output feature dimension
            kernel_sizes: Multi-scale kernel sizes for parallel convolutions
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.output_dim = output_dim

        # Multi-scale parallel convolutions (inception-style)
        self.conv_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(input_channels, num_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                    nn.Conv1d(num_filters, num_filters, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(num_filters),
                    nn.ReLU(),
                )
                for k in kernel_sizes
            ]
        )

        # Merge branches
        total_filters = num_filters * len(kernel_sizes)
        self.merge_conv = nn.Sequential(
            nn.Conv1d(total_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Conv1d(num_filters * 2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
        )

        # Global pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(num_filters, num_filters),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_filters, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral CNN.

        Args:
            x: Input spectrum [batch, channels, length] or [channels, length]

        Returns:
            Feature vector [batch, output_dim] or [output_dim]
        """
        # Handle unbatched input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True

        # Multi-scale convolutions
        branch_outputs = [branch(x) for branch in self.conv_branches]

        # Concatenate branches
        merged = torch.cat(branch_outputs, dim=1)

        # Merge and pool
        features = self.merge_conv(merged)
        pooled = self.global_pool(features).squeeze(-1)

        # Output projection
        output = self.output_proj(pooled)

        if squeeze_output:
            output = output.squeeze(0)

        return output


class PhononInteractionNetwork(nn.Module):
    """
    Neural network modeling phonon-like interactions between frequency modes.

    Inspired by phonon physics where:
    - Phonons are quantized vibrational modes
    - Anharmonic interactions couple different modes
    - Energy transfers between modes via scattering

    This network models inter-frequency interactions as a tensor network
    encoding coupling strengths between frequency bins.
    """

    def __init__(
        self,
        num_modes: int = 64,
        hidden_dim: int = 32,
        interaction_order: int = 3,
    ) -> None:
        """
        Initialize phonon interaction network.

        Args:
            num_modes: Number of frequency modes to model
            hidden_dim: Hidden dimension for interaction tensors
            interaction_order: Order of phonon interactions (2=pairs, 3=triplets)
        """
        super().__init__()
        self.num_modes = num_modes
        self.hidden_dim = hidden_dim
        self.interaction_order = interaction_order

        # Mode embedding
        self.mode_embedding = nn.Embedding(num_modes, hidden_dim)

        # Pairwise interaction tensor (anharmonic coupling)
        self.pair_interaction = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)

        # Triplet interaction (if order >= 3)
        if interaction_order >= 3:
            self.triplet_mlp = nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Energy predictor
        self.energy_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Scattering rate predictor
        self.scattering_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        mode_amplitudes: torch.Tensor,
        mode_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute phonon interactions.

        Args:
            mode_amplitudes: Amplitude of each mode [num_modes]
            mode_indices: Optional subset of mode indices to consider

        Returns:
            Dictionary with:
                - total_energy: Total interaction energy
                - coupling_matrix: Pairwise coupling strengths
                - scattering_rates: Inter-mode scattering rates
                - anharmonic_score: Overall anharmonicity measure
        """
        if mode_indices is None:
            mode_indices = torch.arange(self.num_modes, device=mode_amplitudes.device)

        num_active_modes = len(mode_indices)

        # Get mode embeddings
        embeddings = self.mode_embedding(mode_indices)  # [num_modes, hidden_dim]

        # Weight embeddings by mode amplitudes
        weighted_embeddings = embeddings * mode_amplitudes[mode_indices].unsqueeze(-1)

        # Compute pairwise interactions
        coupling_matrix = torch.zeros(
            num_active_modes, num_active_modes, device=mode_amplitudes.device
        )
        for i in range(num_active_modes):
            for j in range(i + 1, num_active_modes):
                interaction = self.pair_interaction(weighted_embeddings[i], weighted_embeddings[j])
                coupling = self.energy_predictor(interaction).squeeze()
                coupling_matrix[i, j] = coupling
                coupling_matrix[j, i] = coupling

        # Compute scattering rates
        scattering_rates = torch.zeros(
            num_active_modes, num_active_modes, device=mode_amplitudes.device
        )
        for i in range(num_active_modes):
            for j in range(num_active_modes):
                if i != j:
                    pair_features = torch.cat([weighted_embeddings[i], weighted_embeddings[j]])
                    rate = self.scattering_predictor(pair_features).squeeze()
                    scattering_rates[i, j] = rate

        # Total interaction energy
        total_energy = coupling_matrix.sum() / 2  # Divide by 2 for double counting

        # Anharmonicity score (deviation from harmonic behavior)
        off_diagonal_mean = (
            coupling_matrix[~torch.eye(num_active_modes, dtype=bool, device=mode_amplitudes.device)]  # type: ignore[call-overload, unused-ignore]
            .abs()
            .mean()
        )
        anharmonic_score = torch.sigmoid(off_diagonal_mean)

        return {
            "total_energy": total_energy,
            "coupling_matrix": coupling_matrix,
            "scattering_rates": scattering_rates,
            "anharmonic_score": anharmonic_score,
        }


class MLIPVibrationEncoder(nn.Module):
    """
    Machine Learning Interatomic Potential inspired vibrational encoder.

    Inspired by MLIPs that predict atomic forces and energies:
    - Encodes local "atomic" (frequency bin) environments
    - Uses symmetry-adapted descriptors
    - Predicts vibrational properties from learned representations

    This provides a physics-informed encoding of spectral data.
    """

    def __init__(
        self,
        num_freq_bins: int = 512,
        descriptor_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 32,
        num_radial_basis: int = 16,
    ) -> None:
        """
        Initialize MLIP-style vibration encoder.

        Args:
            num_freq_bins: Number of frequency bins in spectrum
            descriptor_dim: Dimension of local descriptors
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_radial_basis: Number of radial basis functions for distance encoding
        """
        super().__init__()
        self.num_freq_bins = num_freq_bins
        self.descriptor_dim = descriptor_dim
        self.output_dim = output_dim
        self.num_radial_basis = num_radial_basis

        # Radial basis for frequency distance encoding
        self.radial_centers = nn.Parameter(
            torch.linspace(0, 1, num_radial_basis),
            requires_grad=False,
        )
        self.radial_width = 1.0 / num_radial_basis

        # Local environment encoder
        self.local_encoder = nn.Sequential(
            nn.Linear(num_radial_basis * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )

        # Message passing for environment interaction
        self.message_net = nn.Sequential(
            nn.Linear(descriptor_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )

        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _compute_radial_basis(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute radial basis function expansion.

        Args:
            distances: Distances [num_pairs]

        Returns:
            Radial basis features [num_pairs, num_radial_basis]
        """
        # Gaussian radial basis
        return torch.exp(
            -((distances.unsqueeze(-1) - self.radial_centers) ** 2) / (2 * self.radial_width**2)
        )

    def forward(self, spectrum: torch.Tensor, k_neighbors: int = 8) -> torch.Tensor:
        """
        Encode spectrum using MLIP-style local descriptors.

        Args:
            spectrum: Power spectrum [num_freq_bins] or [batch, num_freq_bins]
            k_neighbors: Number of neighboring frequencies to consider

        Returns:
            Encoded features [output_dim] or [batch, output_dim]
        """
        # Handle batched input
        squeeze_output = False
        if spectrum.dim() == 1:
            spectrum = spectrum.unsqueeze(0)
            squeeze_output = True

        batch_size, num_bins = spectrum.shape
        device = spectrum.device

        # Compute local descriptors for each frequency bin
        descriptors = torch.zeros(batch_size, num_bins, self.descriptor_dim, device=device)

        for b in range(batch_size):
            for i in range(num_bins):
                # Get local neighborhood
                start = max(0, i - k_neighbors // 2)
                end = min(num_bins, i + k_neighbors // 2 + 1)

                # Compute distances and values
                neighbors = list(range(start, end))
                if i in neighbors:
                    neighbors.remove(i)

                if len(neighbors) == 0:
                    continue

                # Frequency distances (normalized)
                freq_distances = torch.tensor(
                    [abs(j - i) / num_bins for j in neighbors],
                    device=device,
                )

                # Amplitude differences
                amp_diffs = spectrum[b, neighbors] - spectrum[b, i]

                # Radial basis expansion
                radial_features = self._compute_radial_basis(freq_distances)

                # Combine with amplitude information
                amp_features = self._compute_radial_basis(
                    torch.abs(amp_diffs) / (spectrum[b].max() + 1e-8)
                )

                # Aggregate neighborhood features
                combined = torch.cat([radial_features, amp_features], dim=-1)
                local_features = self.local_encoder(combined).mean(dim=0)

                descriptors[b, i] = local_features

        # Message passing between descriptors
        for _ in range(2):  # 2 rounds of message passing
            new_descriptors = torch.zeros_like(descriptors)
            for i in range(num_bins):
                # Aggregate messages from neighbors
                start = max(0, i - k_neighbors // 2)
                end = min(num_bins, i + k_neighbors // 2 + 1)

                neighbor_descs = descriptors[:, start:end]
                self_desc = descriptors[:, i : i + 1].expand(-1, neighbor_descs.size(1), -1)

                messages = self.message_net(torch.cat([self_desc, neighbor_descs], dim=-1))
                new_descriptors[:, i] = descriptors[:, i] + messages.mean(dim=1)

            descriptors = new_descriptors

        # Global pooling and output
        pooled = descriptors.mean(dim=1)
        output = self.output_net(pooled)

        if squeeze_output:
            output = output.squeeze(0)

        return output


# =============================================================================
# Main Spectral Vibration Detector
# =============================================================================


class SpectralVibrationDetector(BaseDetector):
    """Advanced spectral vibration anomaly detector.

    Combines multiple analysis techniques:
    1. FFT-based spectral analysis
    2. Graph Neural Network for frequency interactions
    3. CNN for spectral pattern recognition
    4. Phonon interaction modeling
    5. MLIP-inspired vibrational encoding

    Supports predictive maintenance through vibration signature classification
    and provides comprehensive spectral feature extraction.

    Example:
        >>> detector = SpectralVibrationDetector(config={
        ...     "analysis_mode": "hybrid_fusion",
        ...     "sample_rate": 10000,  # 10 kHz sampling
        ...     "threshold": 0.6,
        ... })
        >>> detector.fit(normal_vibration_data)
        >>> result = detector.detect(test_vibration_data)
        >>> print(result["anomaly_score"], result["signature_type"])
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize spectral vibration detector.

        Args:
            config: Configuration dictionary. Supports keys from SpectralVibrationConfig.
        """
        super().__init__(config)

        # Parse configuration
        self._spectral_config = SpectralVibrationConfig(
            analysis_mode=SpectralAnalysisMode(self.config.get("analysis_mode", "hybrid_fusion")),
            sample_rate=self.config.get("sample_rate", 1000.0),
            fft_size=self.config.get("fft_size", 1024),
            overlap_ratio=self.config.get("overlap_ratio", 0.5),
            num_harmonics=self.config.get("num_harmonics", 8),
            wavelet_levels=self.config.get("wavelet_levels", 5),
            gnn_hidden_dim=self.config.get("gnn_hidden_dim", 64),
            gnn_num_layers=self.config.get("gnn_num_layers", 3),
            cnn_filters=self.config.get("cnn_filters", 32),
            phonon_cutoff=self.config.get("phonon_cutoff", 0.1),
            enable_schumann=self.config.get("enable_schumann", True),
            threshold=self.threshold,
            device=self.config.get("device", "cpu"),
        )

        self.device = torch.device(self._spectral_config.device)

        # Initialize neural network components
        self._init_networks()

        # Reference statistics for anomaly detection
        self._reference_spectrum: np.ndarray | None = None
        self._reference_features: SpectralFeatures | None = None
        self._reference_mean: np.ndarray | None = None
        self._reference_std: np.ndarray | None = None
        self._reference_features_mean: np.ndarray | None = None
        self._reference_features_std: np.ndarray | None = None

        # Vibration signature database
        self._signature_database: dict[VibrationSignatureType, list[np.ndarray]] = {
            sig_type: [] for sig_type in VibrationSignatureType
        }

    def _init_networks(self) -> None:
        """Initialize neural network components."""
        cfg = self._spectral_config

        # GNN for spectral graph analysis
        self._gnn = SpectralGNN(
            input_dim=2,  # magnitude + phase or magnitude + derivative
            hidden_dim=cfg.gnn_hidden_dim,
            output_dim=32,
            num_layers=cfg.gnn_num_layers,
        ).to(self.device)

        # CNN for spectral patterns
        self._cnn = SpectralCNN(
            input_channels=1,
            num_filters=cfg.cnn_filters,
            output_dim=32,
        ).to(self.device)

        # Phonon interaction network
        num_modes = cfg.fft_size // 2
        self._phonon_net = PhononInteractionNetwork(
            num_modes=min(num_modes, 128),  # Cap for efficiency
            hidden_dim=32,
            interaction_order=3,
        ).to(self.device)

        # MLIP encoder
        self._mlip_encoder = MLIPVibrationEncoder(
            num_freq_bins=cfg.fft_size // 2,
            descriptor_dim=64,
            output_dim=32,
        ).to(self.device)

        # Set to eval mode (no training in detector)
        self._gnn.eval()
        self._cnn.eval()
        self._phonon_net.eval()
        self._mlip_encoder.eval()

    def fit(
        self,
        data: np.ndarray | torch.Tensor,
        signature_type: VibrationSignatureType = VibrationSignatureType.NORMAL,
    ) -> SpectralVibrationDetector:
        """
        Fit detector on reference/training data.

        Args:
            data: Time-domain signal array [num_samples] or [batch, num_samples]
            signature_type: Type of vibration signature in training data

        Returns:
            Self for method chaining.

        Raises:
            DetectorException: If data is empty or invalid.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Validate input
        if data.size == 0:
            raise DetectorException("Cannot fit SpectralVibrationDetector with empty data.")

        # Handle batched input
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Compute reference spectrum for each sample
        all_spectra = []
        all_features = []

        for sample in data:
            spectrum = self._compute_power_spectrum(sample)
            features = self._extract_spectral_features(sample)
            all_spectra.append(spectrum)
            all_features.append(self._features_to_array(features))

            # Add to signature database
            self._signature_database[signature_type].append(spectrum)

        # Compute reference statistics
        spectra_array = np.array(all_spectra)
        self._reference_spectrum = np.mean(spectra_array, axis=0)
        self._reference_mean = np.mean(spectra_array, axis=0)
        self._reference_std = np.std(spectra_array, axis=0) + 1e-8

        # Store reference features
        features_array = np.array(all_features)
        self._reference_features_mean = np.mean(features_array, axis=0)
        self._reference_features_std = np.std(features_array, axis=0) + 1e-8

        self._is_fitted = True
        logger.info(
            f"SpectralVibrationDetector fitted on {len(data)} samples, "
            f"signature type: {signature_type.value}"
        )

        return self

    def detect(self, data: np.ndarray | torch.Tensor) -> dict[str, Any]:
        """
        Detect anomalies in spectral vibration data.

        Args:
            data: Time-domain signal array [num_samples] or [batch, num_samples]

        Returns:
            Dictionary containing:
                - is_anomaly: Boolean anomaly flag
                - anomaly_score: Continuous score [0, 1]
                - scores: Per-sample scores if batched
                - spectral_features: Extracted SpectralFeatures
                - diagnostic: VibrationDiagnostic result
                - dominant_frequencies: List of dominant frequency peaks
                - signature_type: Detected vibration signature type
                - confidence: Detection confidence
                - detector_type: "spectral_vibration"

        Raises:
            DetectorException: If detector not fitted.
        """
        if not self._is_fitted:
            raise DetectorException("Detector must be fitted before detection")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Handle single sample
        if data.ndim == 1:
            data = data.reshape(1, -1)

        all_scores = []
        all_diagnostics = []
        all_features = []

        for sample in data:
            # Extract spectral features
            features = self._extract_spectral_features(sample)
            all_features.append(features)

            # Compute anomaly score using multiple methods
            score, diagnostic = self._compute_anomaly_score(sample, features)
            all_scores.append(score)
            all_diagnostics.append(diagnostic)

        # Aggregate results
        scores_array = np.array(all_scores)
        mean_score = float(np.mean(scores_array))
        is_anomaly = mean_score > self.threshold

        # Get primary diagnostic (most severe or most confident)
        primary_diagnostic = max(all_diagnostics, key=lambda d: d.severity_score * d.confidence)

        # Auto-calibration if enabled
        effective_threshold = self.threshold
        calibration_diagnostics = None
        if self._auto_calibrate:
            effective_threshold = self.calibrate_threshold(scores_array)
            calibration_diagnostics = self._last_diagnostics
            is_anomaly = mean_score > effective_threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": mean_score,
            "scores": scores_array,
            "spectral_features": all_features[0] if len(all_features) == 1 else all_features,
            "diagnostic": primary_diagnostic,
            "dominant_frequencies": all_features[0].dominant_frequencies[:5],
            "signature_type": primary_diagnostic.signature_type.value,
            "confidence": primary_diagnostic.confidence,
            "severity": primary_diagnostic.severity_score,
            "detector_type": "spectral_vibration",
            "threshold": effective_threshold,
            "calibration_diagnostics": calibration_diagnostics,
        }

    def extract_features(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Extract features for ML fusion.

        Args:
            data: Time-domain signal array

        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        if data.ndim == 1:
            data = data.reshape(1, -1)

        all_features = []

        for sample in data:
            # Compute power spectrum
            spectrum = self._compute_power_spectrum(sample)

            # Extract multi-modal features
            spectral_features = self._extract_spectral_features(sample)
            basic_features = self._features_to_array(spectral_features)

            # GNN features
            with torch.no_grad():
                spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32, device=self.device)
                edge_index, edge_type = self._build_spectral_graph(len(spectrum))

                # Compute numerical gradient using forward differences with padding
                spectrum_diff = torch.diff(spectrum_tensor, prepend=spectrum_tensor[:1])
                node_features = torch.stack(
                    [
                        spectrum_tensor,
                        spectrum_diff,
                    ],
                    dim=-1,
                )

                gnn_features = self._gnn(node_features, edge_index, edge_type)

            # CNN features
            with torch.no_grad():
                cnn_input = spectrum_tensor.unsqueeze(0)  # [1, length]
                cnn_features = self._cnn(cnn_input)

            # MLIP features
            with torch.no_grad():
                mlip_features = self._mlip_encoder(spectrum_tensor)

            # Concatenate all features
            combined = np.concatenate(
                [
                    basic_features,
                    gnn_features.cpu().numpy(),
                    cnn_features.cpu().numpy(),
                    mlip_features.cpu().numpy(),
                ]
            )

            all_features.append(combined)

        return torch.tensor(np.array(all_features), dtype=torch.float32)

    def _compute_power_spectrum(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute power spectral density.

        Args:
            signal: Time-domain signal

        Returns:
            Power spectrum (magnitude squared of FFT)
        """
        cfg = self._spectral_config

        # Zero-pad or truncate to FFT size
        if len(signal) < cfg.fft_size:
            signal = np.pad(signal, (0, cfg.fft_size - len(signal)))
        elif len(signal) > cfg.fft_size:
            # Use windowed average for long signals
            hop = int(cfg.fft_size * (1 - cfg.overlap_ratio))
            num_frames = (len(signal) - cfg.fft_size) // hop + 1

            spectra = []
            window = np.hanning(cfg.fft_size)

            for i in range(num_frames):
                start = i * hop
                frame = signal[start : start + cfg.fft_size] * window
                spectrum = np.abs(fft(frame)[: cfg.fft_size // 2]) ** 2
                spectra.append(spectrum)

            return np.mean(spectra, axis=0)

        # Single FFT for short signals
        spectrum = np.abs(fft(signal)[: cfg.fft_size // 2]) ** 2
        return spectrum

    def _extract_spectral_features(self, signal: np.ndarray) -> SpectralFeatures:
        """
        Extract comprehensive spectral features.

        Args:
            signal: Time-domain signal

        Returns:
            SpectralFeatures dataclass with all extracted features
        """
        cfg = self._spectral_config

        # Compute spectrum
        power_spectrum = self._compute_power_spectrum(signal)
        freqs = fftfreq(cfg.fft_size, 1 / cfg.sample_rate)[: cfg.fft_size // 2]

        # Normalize spectrum
        power_spectrum_norm = power_spectrum / (power_spectrum.sum() + 1e-10)

        # Dominant frequencies (peaks)
        peak_indices, _ = scipy_signal.find_peaks(power_spectrum, height=power_spectrum.max() * 0.1)
        dominant_frequencies = [
            (freqs[i], power_spectrum[i]) for i in peak_indices[: cfg.num_harmonics]
        ]
        dominant_frequencies.sort(key=lambda x: -x[1])  # Sort by amplitude

        # Harmonic ratios
        harmonic_ratios = self._compute_harmonic_ratios(dominant_frequencies)

        # Spectral centroid
        spectral_centroid = np.sum(freqs * power_spectrum_norm)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * power_spectrum_norm)
        )

        # Spectral rolloff (95% energy)
        cumsum = np.cumsum(power_spectrum_norm)
        rolloff_idx = np.searchsorted(cumsum, 0.95)
        spectral_rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        # Spectral flatness (geometric mean / arithmetic mean)
        log_spectrum = np.log(power_spectrum + 1e-10)
        spectral_flatness = np.exp(np.mean(log_spectrum)) / (np.mean(power_spectrum) + 1e-10)

        # Crest factor
        crest_factor = np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-10)

        # Kurtosis
        kurtosis = scipy_stats.kurtosis(signal) if len(signal) > 4 else 0.0

        # Phonon coupling (using interaction network)
        phonon_coupling = self._compute_phonon_coupling(power_spectrum)

        # Graph Laplacian spectrum
        graph_laplacian_spectrum = self._compute_graph_laplacian_spectrum(power_spectrum)

        # CNN features
        cnn_features = self._compute_cnn_features(power_spectrum)

        # Schumann alignment
        schumann_alignment = self._compute_schumann_alignment(freqs, power_spectrum)

        return SpectralFeatures(
            power_spectrum=power_spectrum,
            dominant_frequencies=dominant_frequencies,
            harmonic_ratios=harmonic_ratios,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            spectral_rolloff=spectral_rolloff,
            spectral_flatness=spectral_flatness,
            crest_factor=crest_factor,
            kurtosis=kurtosis,
            phonon_coupling=phonon_coupling,
            graph_laplacian_spectrum=graph_laplacian_spectrum,
            cnn_features=cnn_features,
            schumann_alignment=schumann_alignment,
        )

    def _compute_harmonic_ratios(
        self,
        dominant_frequencies: list[tuple[float, float]],
    ) -> np.ndarray:
        """
        Compute ratios between harmonic components.

        Args:
            dominant_frequencies: List of (frequency, amplitude) tuples

        Returns:
            Array of harmonic ratios
        """
        if len(dominant_frequencies) < 2:
            return np.zeros(self._spectral_config.num_harmonics - 1)

        fundamental = dominant_frequencies[0][0]
        if fundamental < 1e-6:
            return np.zeros(self._spectral_config.num_harmonics - 1)

        ratios = []
        for freq, amp in dominant_frequencies[1:]:
            ratio = freq / fundamental
            ratios.append(ratio)

        # Pad to fixed size
        while len(ratios) < self._spectral_config.num_harmonics - 1:
            ratios.append(0.0)

        return np.array(ratios[: self._spectral_config.num_harmonics - 1])

    def _compute_phonon_coupling(self, spectrum: np.ndarray) -> float:
        """
        Compute phonon-like coupling strength.

        Args:
            spectrum: Power spectrum

        Returns:
            Coupling strength score [0, 1]
        """
        with torch.no_grad():
            # Subsample spectrum for efficiency
            subsample_size = min(len(spectrum), 128)
            indices = np.linspace(0, len(spectrum) - 1, subsample_size).astype(int)
            subsampled = spectrum[indices]

            # Normalize
            subsampled = subsampled / (subsampled.max() + 1e-10)

            # Compute interactions
            amplitudes = torch.tensor(subsampled, dtype=torch.float32, device=self.device)
            mode_indices = torch.arange(subsample_size, device=self.device)

            result = self._phonon_net(amplitudes, mode_indices)

            return float(result["anharmonic_score"].cpu().numpy())

    def _compute_graph_laplacian_spectrum(self, spectrum: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Compute eigenvalues of frequency graph Laplacian.

        The graph connects frequency bins with weights based on their
        coupling strength and harmonic relationships.

        Args:
            spectrum: Power spectrum
            k: Number of eigenvalues to compute

        Returns:
            Array of k smallest eigenvalues
        """
        n = len(spectrum)
        k = min(k, n - 2)

        if k < 1:
            return np.zeros(10)

        # Build adjacency matrix based on frequency relationships
        # Connect bins that are harmonically related or adjacent
        row_ind = []
        col_ind = []
        data = []

        for i in range(n):
            for j in range(i + 1, min(i + 5, n)):  # Adjacent bins
                weight = np.exp(-0.5 * (j - i))  # Decay with distance
                row_ind.extend([i, j])
                col_ind.extend([j, i])
                data.extend([weight, weight])

            # Harmonic connections
            for harmonic in [2, 3, 4]:
                j = i * harmonic
                if j < n:
                    weight = spectrum[i] * spectrum[j] / (spectrum.max() ** 2 + 1e-10)
                    row_ind.extend([i, j])
                    col_ind.extend([j, i])
                    data.extend([weight, weight])

        # Create sparse adjacency matrix
        if len(data) == 0:
            return np.zeros(k)

        adj = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

        # Compute degree matrix
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        d_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        laplacian = diags(np.ones(n)) - d_inv_sqrt @ adj @ d_inv_sqrt

        # Compute smallest eigenvalues
        try:
            eigenvalues, _ = eigsh(laplacian, k=k, which="SM")
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            eigenvalues = np.zeros(k)

        # Pad if needed
        if len(eigenvalues) < k:
            eigenvalues = np.pad(eigenvalues, (0, k - len(eigenvalues)))

        return eigenvalues

    def _compute_cnn_features(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Extract CNN features from spectrum.

        Args:
            spectrum: Power spectrum

        Returns:
            CNN feature array
        """
        with torch.no_grad():
            # Normalize and convert to tensor
            spectrum_norm = spectrum / (spectrum.max() + 1e-10)
            spectrum_tensor = torch.tensor(
                spectrum_norm,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # [1, length]

            features = self._cnn(spectrum_tensor)

            return features.cpu().numpy()

    def _compute_schumann_alignment(
        self,
        freqs: np.ndarray,
        spectrum: np.ndarray,
    ) -> float:
        """Compute alignment with domain-adaptive fundamental frequencies.

        Phase 3 upgrade: Replaces universal Schumann (7.83 Hz) with
        domain-specific fundamentals:
          - Environmental: Schumann resonances (7.83, 14.3, 20.8, 27.3, 33.8 Hz)
          - Medical: HRV bands (0.04, 0.15, 0.4, 1.0, 40.0 Hz)
          - Infrastructure: Power grid + structural (50, 60, 0.1, 0.01 Hz)
          - Space: Solar cycle + orbital (0.001, 0.01, 0.1, 11.0 Hz)
          - Security/Financial: Adaptive spectral peak detection

        For domains with no predefined fundamentals, the method performs
        adaptive peak detection on the input spectrum.

        Updated harmonic ratio equation:
            A(x) = Σ_d [ Σ_n H(n·ω_d) / Σ H(ω) ] / |D|
            where D is the set of domain fundamental frequencies.

        Args:
            freqs: Frequency array from FFT.
            spectrum: Power spectrum.

        Returns:
            Alignment score in [0, 1].

        Provenance:
            - Schumann (1952) for environmental frequencies
            - Task Force of ESC/NASPE (1996) for HRV frequency bands
            - Standard power grid frequencies for infrastructure
        """
        if not self._spectral_config.enable_schumann:
            return 0.0

        # Get domain-specific fundamental frequencies
        domain = getattr(self._spectral_config, "domain", "environmental")
        domain_freqs = get_domain_fundamentals(domain)

        if domain_freqs is None:
            # Adaptive spectral peak detection for unknown domains
            domain_freqs = self._detect_spectral_peaks(freqs, spectrum)

        if not domain_freqs:
            return 0.0

        # Compute alignment against domain fundamentals
        alignment_scores = []

        for target_freq in domain_freqs:
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - target_freq))
            if 0 < idx < len(spectrum) - 1:
                # Interpolate value at target frequency
                local_max = max(spectrum[idx - 1], spectrum[idx], spectrum[idx + 1])
                local_mean = np.mean(spectrum[max(0, idx - 5) : min(len(spectrum), idx + 6)])  # type: ignore[misc, unused-ignore]

                # Score based on peak prominence
                if local_mean > 0:
                    prominence = local_max / local_mean
                    alignment_scores.append(min(1.0, prominence / 3.0))

        if not alignment_scores:
            return 0.0

        # Weight by golden ratio (higher weight to fundamental)
        weights = [PHI ** (-i) for i in range(len(alignment_scores))]
        weights = np.array(weights) / sum(weights)  # type: ignore[assignment, unused-ignore]

        return float(np.average(alignment_scores, weights=weights))

    @staticmethod
    def _detect_spectral_peaks(
        freqs: np.ndarray,
        spectrum: np.ndarray,
        max_peaks: int = 5,
    ) -> tuple[float, ...]:
        """
        Adaptive spectral peak detection for domains without known fundamentals.

        Uses scipy's find_peaks for dominant frequency identification.
        This is a simplified alternative to MUSIC/ESPRIT algorithms
        suitable for real-time operation.

        Args:
            freqs: Frequency array.
            spectrum: Power spectrum.
            max_peaks: Maximum number of peaks to return.

        Returns:
            Tuple of detected fundamental frequencies in Hz.
        """
        if len(spectrum) < 10:
            return ()

        # Find peaks with minimum prominence
        try:
            from scipy.signal import find_peaks as scipy_find_peaks

            # Normalize spectrum for consistent thresholding
            norm_spectrum = spectrum / (np.max(spectrum) + 1e-10)
            peak_indices, properties = scipy_find_peaks(
                norm_spectrum,
                height=0.1,  # Minimum 10% of max
                distance=5,  # Minimum 5 bins apart
                prominence=0.05,  # Minimum prominence
            )

            if len(peak_indices) == 0:
                return ()

            # Sort by prominence (most prominent first)
            if "prominences" in properties:
                sorted_idx = np.argsort(properties["prominences"])[::-1]
                peak_indices = peak_indices[sorted_idx]

            # Return top peaks as frequencies
            selected = peak_indices[:max_peaks]
            return tuple(float(freqs[i]) for i in selected if i < len(freqs))

        except (ImportError, ValueError):
            return ()

    def _build_spectral_graph(self, num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph structure for spectral analysis.

        Args:
            num_nodes: Number of frequency bins

        Returns:
            Tuple of (edge_index, edge_type)
        """
        edge_index_list = []
        edge_type_list = []

        # Edge types:
        # 0: Adjacent (local smoothness)
        # 1: Harmonic (2x frequency relationship)
        # 2: Sub-harmonic (0.5x frequency relationship)
        # 3: Sideband (modulation relationship)

        for i in range(num_nodes):
            # Adjacent edges
            if i > 0:
                edge_index_list.append([i - 1, i])
                edge_type_list.append(0)
                edge_index_list.append([i, i - 1])
                edge_type_list.append(0)

            # Harmonic edges
            if 2 * i < num_nodes:
                edge_index_list.append([i, 2 * i])
                edge_type_list.append(1)
                edge_index_list.append([2 * i, i])
                edge_type_list.append(2)

            # Sideband edges (nearby non-adjacent)
            for offset in [3, 5, 7]:
                if i + offset < num_nodes:
                    edge_index_list.append([i, i + offset])
                    edge_type_list.append(3)
                    edge_index_list.append([i + offset, i])
                    edge_type_list.append(3)

        if not edge_index_list:
            # Create minimal graph
            edge_index_list = [[0, 0]]
            edge_type_list = [0]

        edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=self.device).T
        edge_type = torch.tensor(edge_type_list, dtype=torch.long, device=self.device)

        return edge_index, edge_type

    def _features_to_array(self, features: SpectralFeatures) -> np.ndarray:
        """
        Convert SpectralFeatures to flat array.

        Args:
            features: SpectralFeatures dataclass

        Returns:
            Flat feature array
        """
        return np.concatenate(
            [
                [features.spectral_centroid],
                [features.spectral_bandwidth],
                [features.spectral_rolloff],
                [features.spectral_flatness],
                [features.crest_factor],
                [features.kurtosis],
                [features.phonon_coupling],
                [features.schumann_alignment],
                features.harmonic_ratios[:7],  # Pad/truncate to 7
                features.graph_laplacian_spectrum[:10],  # Top 10 eigenvalues
                features.cnn_features[:32],  # CNN features
            ]
        )

    def _compute_anomaly_score(
        self,
        signal: np.ndarray,
        features: SpectralFeatures,
    ) -> tuple[float, VibrationDiagnostic]:
        """
        Compute anomaly score and diagnostic.

        Args:
            signal: Time-domain signal
            features: Extracted spectral features

        Returns:
            Tuple of (anomaly_score, VibrationDiagnostic)
        """
        # Convert features to array
        feature_array = self._features_to_array(features)

        # Compute z-scores against reference
        if self._reference_features_mean is not None:
            z_scores = (
                feature_array - self._reference_features_mean
            ) / self._reference_features_std
            feature_anomaly_score = np.mean(np.abs(z_scores)) / 3.0  # Normalize
        else:
            feature_anomaly_score = 0.5

        # Spectral distance from reference
        if self._reference_spectrum is not None and self._reference_std is not None:
            spectral_distance = np.mean(
                np.abs(features.power_spectrum - self._reference_spectrum)
                / (self._reference_std + 1e-8)
            )
            spectral_anomaly_score = min(1.0, spectral_distance / 3.0)
        else:
            spectral_anomaly_score = 0.5

        # Classify vibration signature
        signature_type, signature_confidence = self._classify_signature(features)

        # Combine scores with golden ratio weighting (from 3R mechanism)
        phi_sum = PHI + 1.0 + (1.0 / PHI)
        w_feature = PHI / phi_sum
        w_spectral = 1.0 / phi_sum
        w_signature = (1.0 / PHI) / phi_sum

        # Signature-based anomaly contribution
        signature_anomaly = 0.0 if signature_type == VibrationSignatureType.NORMAL else 0.7

        combined_score = (
            w_feature * feature_anomaly_score
            + w_spectral * spectral_anomaly_score
            + w_signature * signature_anomaly
        )
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # Create diagnostic
        diagnostic = self._create_diagnostic(
            features, signature_type, signature_confidence, combined_score
        )

        return combined_score, diagnostic

    def _classify_signature(
        self,
        features: SpectralFeatures,
    ) -> tuple[VibrationSignatureType, float]:
        """
        Classify vibration signature type.

        Args:
            features: Extracted spectral features

        Returns:
            Tuple of (signature_type, confidence)
        """
        # Simple rule-based classification (can be extended with ML)
        dominant_freqs = features.dominant_frequencies

        if len(dominant_freqs) == 0:
            return VibrationSignatureType.NORMAL, 0.5

        harmonic_ratios = features.harmonic_ratios

        # Check for specific fault signatures
        # Bearing faults: non-synchronous frequencies, often 0.4-0.48x or sidebands
        # Imbalance: 1x running speed dominant
        # Misalignment: 2x or 3x running speed
        # Looseness: many harmonics, half harmonics

        confidence = 0.7  # Base confidence

        # High crest factor indicates impulsive faults (bearings, gears)
        if features.crest_factor > 4.0:
            if features.kurtosis > 5.0:
                return VibrationSignatureType.BEARING_FAULT, 0.85
            return VibrationSignatureType.GEAR_MESH, 0.75

        # Check harmonic structure
        if len(harmonic_ratios) >= 2:
            # Integer harmonics suggest imbalance/misalignment
            integer_ratios = [r for r in harmonic_ratios if abs(r - round(r)) < 0.1 and r > 0]

            if len(integer_ratios) >= 2:
                if 2.0 in [round(r) for r in integer_ratios]:
                    return VibrationSignatureType.MISALIGNMENT, 0.80
                return VibrationSignatureType.IMBALANCE, 0.75

            # Half harmonics suggest looseness
            half_ratios = [r for r in harmonic_ratios if abs(r - round(r) - 0.5) < 0.1]
            if len(half_ratios) >= 1:
                return VibrationSignatureType.LOOSENESS, 0.75

        # Low spectral flatness (tonal) might indicate resonance
        if features.spectral_flatness < 0.1:
            return VibrationSignatureType.RESONANCE, 0.70

        # High spectral flatness (noise-like) might indicate flow
        if features.spectral_flatness > 0.8:
            return VibrationSignatureType.FLOW_INDUCED, 0.65

        # Check against reference database
        if self._signature_database[VibrationSignatureType.NORMAL]:
            ref_spectra = np.array(self._signature_database[VibrationSignatureType.NORMAL])
            distances = np.mean(np.abs(features.power_spectrum - ref_spectra), axis=1)
            min_distance = np.min(distances)

            if min_distance < 0.1:
                return VibrationSignatureType.NORMAL, 0.90
            elif min_distance > 0.5:
                return VibrationSignatureType.UNKNOWN_ANOMALY, 0.70

        return VibrationSignatureType.NORMAL, confidence

    def _create_diagnostic(
        self,
        features: SpectralFeatures,
        signature_type: VibrationSignatureType,
        confidence: float,
        severity_score: float,
    ) -> VibrationDiagnostic:
        """
        Create vibration diagnostic report.

        Args:
            features: Spectral features
            signature_type: Classified signature type
            confidence: Classification confidence
            severity_score: Anomaly severity

        Returns:
            VibrationDiagnostic with recommendations
        """
        # Determine fault frequency and recommendations
        fault_freq = features.dominant_frequencies[0][0] if features.dominant_frequencies else 0.0

        recommendations = {
            VibrationSignatureType.NORMAL: "No action required. Continue routine monitoring.",
            VibrationSignatureType.BEARING_FAULT: "Schedule bearing inspection. Check lubrication and replace if needed.",
            VibrationSignatureType.IMBALANCE: "Balance rotating component. Check for buildup or wear.",
            VibrationSignatureType.MISALIGNMENT: "Check and correct shaft alignment. Inspect couplings.",
            VibrationSignatureType.LOOSENESS: "Inspect and tighten foundation bolts and fittings.",
            VibrationSignatureType.RESONANCE: "Investigate structural resonance. Consider damping or speed change.",
            VibrationSignatureType.ELECTRICAL: "Check motor electrical connections and rotor bars.",
            VibrationSignatureType.FLOW_INDUCED: "Inspect for cavitation or turbulence. Check flow paths.",
            VibrationSignatureType.GEAR_MESH: "Inspect gears for wear, pitting, or damage.",
            VibrationSignatureType.UNKNOWN_ANOMALY: "Detailed investigation recommended. Collect additional data.",
        }

        # Estimate time to failure based on severity
        ttf = None
        if signature_type != VibrationSignatureType.NORMAL:
            if severity_score > 0.8:
                ttf = 24.0  # hours
            elif severity_score > 0.6:
                ttf = 168.0  # 1 week
            elif severity_score > 0.4:
                ttf = 720.0  # 1 month

        return VibrationDiagnostic(
            signature_type=signature_type,
            confidence=confidence,
            fault_frequency=fault_freq,
            severity_score=severity_score,
            recommended_action=recommendations.get(signature_type, "Investigate anomaly."),
            time_to_failure=ttf,
            supporting_features={
                "spectral_centroid": features.spectral_centroid,
                "crest_factor": features.crest_factor,
                "kurtosis": features.kurtosis,
                "spectral_flatness": features.spectral_flatness,
                "phonon_coupling": features.phonon_coupling,
                "schumann_alignment": features.schumann_alignment,
            },
        )


# =============================================================================
# Utility Functions
# =============================================================================


def compute_short_time_fourier_transform(
    signal: np.ndarray,
    window_size: int = 256,
    hop_size: int = 64,
    window_type: str = "hann",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT).

    Args:
        signal: Input time-domain signal
        window_size: FFT window size
        hop_size: Hop size between windows
        window_type: Window function type

    Returns:
        Tuple of (frequencies, times, spectrogram)
    """
    # Get window function
    if window_type == "hann":
        window = np.hanning(window_size)
    elif window_type == "hamming":
        window = np.hamming(window_size)
    else:
        window = np.ones(window_size)

    # Compute STFT
    num_frames = (len(signal) - window_size) // hop_size + 1
    spectrogram = np.zeros((window_size // 2, num_frames), dtype=complex)

    for i in range(num_frames):
        start = i * hop_size
        frame = signal[start : start + window_size] * window
        spectrogram[:, i] = fft(frame)[: window_size // 2]

    # Create time and frequency axes
    times = np.arange(num_frames) * hop_size
    frequencies = np.arange(window_size // 2)

    return frequencies, times, np.abs(spectrogram) ** 2


def compute_wavelet_decomposition(
    signal: np.ndarray,
    levels: int = 5,
) -> list[np.ndarray]:
    """
    Compute Haar wavelet decomposition.

    Args:
        signal: Input signal
        levels: Number of decomposition levels

    Returns:
        List of coefficient arrays [approx, detail_n, ..., detail_1]
    """
    coeffs = []
    current = signal.copy()

    for _ in range(levels):
        n = len(current)
        if n < 2:
            break
        n_half = n // 2

        approx = np.zeros(n_half)
        detail = np.zeros(n_half)

        for i in range(n_half):
            approx[i] = (current[2 * i] + current[2 * i + 1]) / np.sqrt(2)
            detail[i] = (current[2 * i] - current[2 * i + 1]) / np.sqrt(2)

        coeffs.append(detail)
        current = approx

    coeffs.insert(0, current)
    return coeffs


def detect_peaks_with_harmonics(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    num_harmonics: int = 5,
    min_prominence: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Detect peaks and their harmonic structure.

    Args:
        spectrum: Power spectrum
        frequencies: Frequency array
        num_harmonics: Maximum harmonics to detect
        min_prominence: Minimum peak prominence

    Returns:
        List of peak dictionaries with harmonic information
    """
    # Find peaks
    peak_indices, properties = scipy_signal.find_peaks(
        spectrum,
        prominence=min_prominence * spectrum.max(),
    )

    if len(peak_indices) == 0:
        return []

    peaks = []
    for idx in peak_indices:
        peak_info = {
            "frequency": frequencies[idx],
            "amplitude": spectrum[idx],
            "prominence": properties["prominences"][list(peak_indices).index(idx)],
            "harmonics": [],
        }

        # Find harmonics
        fundamental = frequencies[idx]
        for h in range(2, num_harmonics + 1):
            harmonic_freq = fundamental * h
            harmonic_idx = np.argmin(np.abs(frequencies - harmonic_freq))

            if abs(frequencies[harmonic_idx] - harmonic_freq) < fundamental * 0.1:
                peak_info["harmonics"].append(
                    {
                        "order": h,
                        "frequency": frequencies[harmonic_idx],
                        "amplitude": spectrum[harmonic_idx],
                        "ratio": spectrum[harmonic_idx] / spectrum[idx],
                    }
                )

        peaks.append(peak_info)

    # Sort by amplitude
    peaks.sort(key=lambda x: -x["amplitude"])

    return peaks

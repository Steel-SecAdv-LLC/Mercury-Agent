"""
Mercury Agent
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

from __future__ import annotations

"""
Cortical-Laminated Neural Network Architecture

Implements a biologically-inspired neural network based on the 6-layer
organization of the mammalian neocortex. This architecture provides:

1. Hierarchical information processing mimicking cortical columns
2. Feedback connections for top-down modulation
3. Sparse activation patterns for efficient computation
4. Lateral inhibition for winner-take-all competition

Cortical Layers (Brodmann's classification):
    Layer I   (Molecular)          - Apical dendrites, horizontal connections
    Layer II  (External Granular)  - Small pyramidal cells, local processing
    Layer III (External Pyramidal) - Medium pyramidal cells, cortico-cortical output
    Layer IV  (Internal Granular)  - Stellate cells, receives thalamic input
    Layer V   (Internal Pyramidal) - Large pyramidal cells, subcortical output
    Layer VI  (Multiform)          - Diverse cells, corticothalamic feedback

Brain Staining Techniques (analysis modules):
    Golgi Stain   - Reveals complete neuron morphology (architecture analysis)
    Nissl Stain   - Shows cell bodies/organization (activation patterns)
    Weigert Stain - Highlights myelin/connections (weight analysis)

Reference:
    - Mountcastle, V.B. (1997). The columnar organization of the neocortex
    - Douglas, R.J. & Martin, K.A. (2004). Neuronal circuits of the neocortex
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "CorticalColumn",
    "CorticalConfig",
    "CorticalLaminatedNetwork",
    "CorticalLayer",
    "GolgiAnalyzer",
    "HebbianLearningRule",
    "LateralInhibition",
    "NisslAnalyzer",
    "SparseCoding",
    "ThalamocorticalGate",
    "WeigertAnalyzer",
]


class CorticalLayer(Enum):
    """Enumeration of cortical layers with their properties."""

    MOLECULAR = 1  # Layer I - dendrites, horizontal connections
    EXTERNAL_GRANULAR = 2  # Layer II - small pyramidal cells
    EXTERNAL_PYRAMIDAL = 3  # Layer III - cortico-cortical output
    INTERNAL_GRANULAR = 4  # Layer IV - thalamic input (primary sensory)
    INTERNAL_PYRAMIDAL = 5  # Layer V - subcortical output (motor)
    MULTIFORM = 6  # Layer VI - feedback to thalamus


@dataclass
class CorticalConfig:
    """Configuration for cortical network.

    Attributes:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for each layer
        output_dim: Final output dimension
        sparsity: Target sparsity level (0.0-1.0, default 0.1)
        lateral_inhibition_strength: Strength of lateral inhibition (default 0.5)
        feedback_strength: Top-down feedback strength (default 0.3)
        hebbian_learning_rate: Learning rate for Hebbian updates (default 0.01)
        dropout: Dropout probability (default 0.1)
        use_layer_norm: Whether to use LayerNorm (default True)
    """

    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    sparsity: float = 0.1
    lateral_inhibition_strength: float = 0.5
    feedback_strength: float = 0.3
    hebbian_learning_rate: float = 0.01
    dropout: float = 0.1
    use_layer_norm: bool = True


class SparseCoding(nn.Module):
    """Sparse coding module implementing k-winner-take-all activation.

    Biologically, cortical neurons exhibit sparse activation patterns where
    only ~10% of neurons are active at any time. This improves:
    - Memory efficiency (fewer active units)
    - Generalization (distributed representations)
    - Noise robustness (redundant encoding)

    Uses soft top-k selection to maintain differentiability.
    """

    def __init__(self, sparsity: float = 0.1, temperature: float = 1.0) -> None:
        """Initialize sparse coding.

        Args:
            sparsity: Target fraction of active neurons (0.0-1.0)
            temperature: Softmax temperature for soft selection
        """
        super().__init__()
        self.sparsity = sparsity
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sparse activation with soft top-k selection.

        Args:
            x: Input tensor [batch_size, features]

        Returns:
            Sparsified tensor with approximately sparsity*features active
        """
        batch_size, features = x.shape
        k = max(1, int(features * self.sparsity))

        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x, k, dim=-1)

        # Create sparse mask using softmax over top-k (differentiable)
        sparse_weights = F.softmax(topk_values / self.temperature, dim=-1)

        # Scatter back to full dimension
        output = torch.zeros_like(x)
        output.scatter_(1, topk_indices, topk_values * sparse_weights)

        return output


class LateralInhibition(nn.Module):
    """Lateral inhibition module implementing competitive dynamics.

    Models the inhibitory interneuron connections in cortical columns that
    create winner-take-all dynamics. Strong activations suppress neighboring
    neurons, sharpening the response selectivity.

    Implements Mexican hat (difference of Gaussians) inhibition pattern:
    - Short-range excitation (self-enhancement)
    - Long-range inhibition (competition suppression)
    """

    def __init__(
        self,
        features: int,
        strength: float = 0.5,
        sigma_exc: float = 1.0,
        sigma_inh: float = 3.0,
    ) -> None:
        """Initialize lateral inhibition.

        Args:
            features: Number of features/neurons
            strength: Overall inhibition strength (0.0-1.0)
            sigma_exc: Excitatory Gaussian width
            sigma_inh: Inhibitory Gaussian width
        """
        super().__init__()
        self.features = features
        self.strength = strength

        # Pre-compute lateral interaction kernel (Mexican hat)
        positions = torch.arange(features).float()
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        excitatory = torch.exp(-distances.pow(2) / (2 * sigma_exc**2))
        inhibitory = torch.exp(-distances.pow(2) / (2 * sigma_inh**2))
        kernel = excitatory - 0.5 * inhibitory

        # Normalize and register as buffer (not a parameter)
        kernel = kernel / kernel.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply lateral inhibition.

        Args:
            x: Input tensor [batch_size, features]

        Returns:
            Tensor with lateral interactions applied
        """
        # Apply lateral interaction kernel
        inhibited = torch.matmul(x, self.kernel)

        # Blend original with inhibited based on strength
        return (1 - self.strength) * x + self.strength * inhibited


class HebbianLearningRule(nn.Module):
    """Hebbian learning module implementing "neurons that fire together, wire together".

    Implements Oja's rule (normalized Hebbian):
        dw = eta * y * (x - y * w)

    This biologically-plausible learning rule:
    - Strengthens connections between co-active neurons
    - Includes weight decay to prevent unbounded growth
    - Extracts principal components of the input

    Can be used as auxiliary loss during training for biological plausibility.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
    ) -> None:
        """Initialize Hebbian learning.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            learning_rate: Hebbian learning rate (eta)
            weight_decay: Weight decay factor
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Hebbian weight matrix (separate from main network weights)
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian transformation.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        return F.linear(x, self.weight)

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Hebbian weight update (Oja's rule).

        Args:
            x: Pre-synaptic activations [batch_size, input_dim]
            y: Post-synaptic activations [batch_size, output_dim]

        Returns:
            Weight update delta [output_dim, input_dim]
        """
        # Oja's rule: dw = eta * y * (x - y * w)
        # Prevents weight explosion while extracting principal components
        batch_size = x.shape[0]

        # y^T @ x gives outer product averaged over batch
        outer = torch.matmul(y.t(), x) / batch_size  # [output_dim, input_dim]

        # y^T @ y @ w gives the normalization term
        y_squared = torch.matmul(y.t(), y) / batch_size  # [output_dim, output_dim]
        normalization = torch.matmul(y_squared, self.weight)  # [output_dim, input_dim]

        # Oja's update
        delta = self.learning_rate * (outer - normalization)

        # Apply weight decay
        delta = delta - self.weight_decay * self.weight

        return delta

    @torch.no_grad()
    def apply_hebbian_update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Apply Hebbian weight update in-place.

        Args:
            x: Pre-synaptic activations
            y: Post-synaptic activations
        """
        delta = self.hebbian_update(x, y)
        self.weight.add_(delta)


class ThalamocorticalGate(nn.Module):
    """Thalamocortical gating mechanism for attention.

    Models the thalamus as a relay station that gates information flow to cortex.
    The thalamus receives:
    - Feedforward sensory input
    - Feedback from cortical Layer VI
    - Modulatory input from brainstem (arousal)

    Implements multiplicative gating similar to GRU/LSTM but biologically motivated.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        feedback_dim: int | None = None,
    ) -> None:
        """Initialize thalamocortical gate.

        Args:
            input_dim: Dimension of feedforward input (sensory)
            hidden_dim: Hidden/output dimension
            feedback_dim: Dimension of feedback from Layer VI (defaults to hidden_dim)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feedback_dim = feedback_dim or hidden_dim

        # Thalamic relay cell (processes feedforward input)
        self.relay = nn.Linear(input_dim, hidden_dim)

        # Reticular nucleus (provides inhibition/gating)
        self.reticular = nn.Sequential(
            nn.Linear(input_dim + self.feedback_dim, hidden_dim),
            nn.Sigmoid(),
        )

        # Feedback integration from cortical Layer VI
        self.feedback_proj = nn.Linear(self.feedback_dim, hidden_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        sensory_input: torch.Tensor,
        cortical_feedback: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gate sensory input based on cortical feedback.

        Args:
            sensory_input: Feedforward sensory input [batch_size, input_dim]
            cortical_feedback: Top-down feedback [batch_size, feedback_dim] (optional)

        Returns:
            Gated output to Layer IV [batch_size, hidden_dim]
        """
        # Process feedforward input through relay cells
        relay_output = self.relay(sensory_input)

        # Compute gating signal from reticular nucleus
        if cortical_feedback is not None:
            gate_input = torch.cat([sensory_input, cortical_feedback], dim=-1)
        else:
            # Use zeros for feedback if not provided
            batch_size = sensory_input.shape[0]
            zeros = torch.zeros(batch_size, self.feedback_dim, device=sensory_input.device)
            gate_input = torch.cat([sensory_input, zeros], dim=-1)

        gate = self.reticular(gate_input)

        # Apply gating (multiplicative modulation)
        gated_output = relay_output * gate

        # Add feedback modulation if available
        if cortical_feedback is not None:
            feedback_signal = self.feedback_proj(cortical_feedback)
            gated_output = gated_output + 0.3 * feedback_signal

        return self.layer_norm(gated_output)


class CorticalColumn(nn.Module):
    """Single cortical column implementing all 6 layers.

    A cortical column is the fundamental processing unit of the neocortex,
    consisting of ~100 neurons organized in 6 layers. Each column processes
    information through:

    1. Layer IV receives thalamic input (sensory data)
    2. Layers II/III process and send to other cortical areas
    3. Layer V sends output to subcortical structures (motor commands)
    4. Layer VI sends feedback to thalamus (attention modulation)
    5. Layer I provides horizontal connections between columns
    """

    def __init__(self, config: CorticalConfig) -> None:
        """Initialize cortical column.

        Args:
            config: Configuration for the column
        """
        super().__init__()
        self.config = config

        # Layer dimensions (biologically-inspired proportions)
        # Layer IV is largest (receives input), Layer I is smallest
        layer_dims = {
            CorticalLayer.MOLECULAR: config.hidden_dim // 8,
            CorticalLayer.EXTERNAL_GRANULAR: config.hidden_dim // 4,
            CorticalLayer.EXTERNAL_PYRAMIDAL: config.hidden_dim // 2,
            CorticalLayer.INTERNAL_GRANULAR: config.hidden_dim,  # Main input layer
            CorticalLayer.INTERNAL_PYRAMIDAL: config.hidden_dim // 2,
            CorticalLayer.MULTIFORM: config.hidden_dim // 4,
        }
        self.layer_dims = layer_dims

        # Layer IV (Internal Granular) - Receives thalamic input
        # This is the primary input layer in sensory cortex
        self.layer_iv = nn.Sequential(
            nn.Linear(config.input_dim, layer_dims[CorticalLayer.INTERNAL_GRANULAR]),
            (
                nn.LayerNorm(layer_dims[CorticalLayer.INTERNAL_GRANULAR])
                if config.use_layer_norm
                else nn.Identity()
            ),
            nn.GELU(),  # Smoother than ReLU, more biologically plausible
            nn.Dropout(config.dropout),
        )

        # Layer II/III (External Granular/Pyramidal) - Cortico-cortical processing
        # These layers communicate with other cortical areas
        self.layer_ii_iii = nn.Sequential(
            nn.Linear(
                layer_dims[CorticalLayer.INTERNAL_GRANULAR],
                layer_dims[CorticalLayer.EXTERNAL_PYRAMIDAL],
            ),
            (
                nn.LayerNorm(layer_dims[CorticalLayer.EXTERNAL_PYRAMIDAL])
                if config.use_layer_norm
                else nn.Identity()
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            SparseCoding(sparsity=config.sparsity),
        )

        # Layer V (Internal Pyramidal) - Subcortical output
        # Large pyramidal cells that send output to brainstem, spinal cord
        self.layer_v = nn.Sequential(
            nn.Linear(
                layer_dims[CorticalLayer.EXTERNAL_PYRAMIDAL],
                layer_dims[CorticalLayer.INTERNAL_PYRAMIDAL],
            ),
            (
                nn.LayerNorm(layer_dims[CorticalLayer.INTERNAL_PYRAMIDAL])
                if config.use_layer_norm
                else nn.Identity()
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Layer VI (Multiform) - Corticothalamic feedback
        # Sends feedback to thalamus for attention modulation
        self.layer_vi = nn.Sequential(
            nn.Linear(
                layer_dims[CorticalLayer.INTERNAL_PYRAMIDAL],
                layer_dims[CorticalLayer.MULTIFORM],
            ),
            (
                nn.LayerNorm(layer_dims[CorticalLayer.MULTIFORM])
                if config.use_layer_norm
                else nn.Identity()
            ),
            nn.GELU(),
        )

        # Layer I (Molecular) - Horizontal connections
        # Apical dendrites from deeper layers, horizontal axons
        self.layer_i = nn.Linear(
            layer_dims[CorticalLayer.MULTIFORM],
            layer_dims[CorticalLayer.MOLECULAR],
        )

        # Lateral inhibition within layers II/III
        self.lateral_inhibition = LateralInhibition(
            features=layer_dims[CorticalLayer.EXTERNAL_PYRAMIDAL],
            strength=config.lateral_inhibition_strength,
        )

        # Feedback projection (Layer VI -> Layer IV modulation)
        self.feedback_proj = nn.Linear(
            layer_dims[CorticalLayer.MULTIFORM],
            layer_dims[CorticalLayer.INTERNAL_GRANULAR],
        )

        # Output projection
        self.output_proj = nn.Linear(
            layer_dims[CorticalLayer.INTERNAL_PYRAMIDAL], config.output_dim
        )

        # Store activations for analysis
        self.activations: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: torch.Tensor,
        feedback: torch.Tensor | None = None,
        return_layer_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass through cortical column.

        Args:
            x: Input tensor [batch_size, input_dim]
            feedback: Optional feedback from higher areas [batch_size, feedback_dim]
            return_layer_activations: Whether to return intermediate activations

        Returns:
            Output tensor [batch_size, output_dim], optionally with layer activations
        """
        # Layer IV - Receive input (with optional feedback modulation)
        h_iv = self.layer_iv(x)
        if feedback is not None:
            feedback_signal = self.feedback_proj(feedback)
            h_iv = h_iv + self.config.feedback_strength * feedback_signal

        # Layer II/III - Local processing with lateral inhibition
        h_ii_iii = self.layer_ii_iii(h_iv)
        h_ii_iii = self.lateral_inhibition(h_ii_iii)

        # Layer V - Output to subcortical structures
        h_v = self.layer_v(h_ii_iii)

        # Layer VI - Feedback to thalamus
        h_vi = self.layer_vi(h_v)

        # Layer I - Horizontal spread (for inter-column communication)
        h_i = self.layer_i(h_vi)

        # Final output from Layer V
        output = self.output_proj(h_v)

        if return_layer_activations:
            activations = {
                "layer_i": h_i,
                "layer_ii_iii": h_ii_iii,
                "layer_iv": h_iv,
                "layer_v": h_v,
                "layer_vi": h_vi,
            }
            self.activations = activations
            return output, activations

        return output

    def get_feedback_signal(self) -> torch.Tensor | None:
        """Get Layer VI output for feedback to thalamus."""
        return self.activations.get("layer_vi")


class CorticalLaminatedNetwork(nn.Module):
    """Multi-column cortical network with thalamocortical input gating.

    Implements a hierarchical network of cortical columns with:
    - Thalamocortical gating for attention
    - Inter-column connections via Layer I
    - Top-down feedback from higher to lower areas
    - Hebbian learning for biological plausibility

    Architecture:
        Sensory Input
            |
        [Thalamocortical Gate] <-- Feedback from Column 3
            |
        [Cortical Column 1] (Primary sensory)
            |
        [Cortical Column 2] (Secondary association)
            |
        [Cortical Column 3] (Higher association)
            |
        Output
    """

    def __init__(
        self,
        config: CorticalConfig,
        num_columns: int = 3,
        use_thalamic_gate: bool = True,
        use_hebbian: bool = True,
    ) -> None:
        """Initialize cortical network.

        Args:
            config: Configuration for cortical columns
            num_columns: Number of hierarchical columns (default 3)
            use_thalamic_gate: Whether to use thalamocortical gating
            use_hebbian: Whether to include Hebbian learning module
        """
        super().__init__()
        self.config = config
        self.num_columns = num_columns
        self._prev_feedback: torch.Tensor | None = None

        # Thalamocortical gate for input gating
        self.use_thalamic_gate = use_thalamic_gate
        if use_thalamic_gate:
            self.thalamic_gate = ThalamocorticalGate(
                input_dim=config.input_dim,
                hidden_dim=config.input_dim,
                feedback_dim=config.hidden_dim // 4,  # From Layer VI
            )

        # Stack of cortical columns
        self.columns = nn.ModuleList()
        for i in range(num_columns):
            col_config = CorticalConfig(
                input_dim=config.input_dim if i == 0 else config.output_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.output_dim,
                sparsity=config.sparsity,
                lateral_inhibition_strength=config.lateral_inhibition_strength,
                feedback_strength=config.feedback_strength,
                dropout=config.dropout,
                use_layer_norm=config.use_layer_norm,
            )
            self.columns.append(CorticalColumn(col_config))

        # Hebbian learning module (optional)
        self.use_hebbian = use_hebbian
        if use_hebbian:
            self.hebbian = HebbianLearningRule(
                input_dim=config.output_dim,
                output_dim=config.output_dim,
                learning_rate=config.hebbian_learning_rate,
            )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.output_dim, config.output_dim),
            nn.LayerNorm(config.output_dim) if config.use_layer_norm else nn.Identity(),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_activations: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        """Forward pass through cortical network.

        Args:
            x: Input tensor [batch_size, input_dim]
            return_all_activations: Whether to return all intermediate activations

        Returns:
            Output tensor [batch_size, output_dim] or dict with activations
        """
        all_activations = {}

        # Get feedback from last column for thalamic gating (if available)
        feedback = None
        if hasattr(self, "_prev_feedback"):
            feedback = self._prev_feedback

        # Apply thalamocortical gating
        if self.use_thalamic_gate:
            x = self.thalamic_gate(x, feedback)

        # Process through hierarchical columns
        h = x
        for i, column in enumerate(self.columns):
            # Get feedback from next column (top-down)
            col_feedback = None
            if i < len(self.columns) - 1 and hasattr(self.columns[i + 1], "activations"):
                col_feedback = self.columns[i + 1].activations.get("layer_vi")

            h, activations = column(h, feedback=col_feedback, return_layer_activations=True)
            all_activations[f"column_{i}"] = activations

        # Store Layer VI from last column for next forward pass (thalamic feedback)
        self._prev_feedback = self.columns[-1].activations.get("layer_vi")

        # Apply Hebbian learning if enabled (during training)
        if self.use_hebbian and self.training:
            hebbian_output = self.hebbian(h)
            self.hebbian.apply_hebbian_update(h.detach(), hebbian_output.detach())

        # Final output
        output = self.output_proj(h)

        if return_all_activations:
            return {
                "output": output,
                "activations": all_activations,
                "thalamic_gated": x if self.use_thalamic_gate else None,
            }

        return output


# ==============================================================================
# Brain Stain-Inspired Analysis Modules
# ==============================================================================


class GolgiAnalyzer:
    """Golgi stain-inspired analysis of network morphology.

    The Golgi stain reveals complete neuron morphology including:
    - Dendritic arbor structure
    - Axonal projections
    - Cell body shape

    This analyzer examines network architecture:
    - Weight connectivity patterns
    - Layer-wise receptive fields
    - Information flow pathways
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize Golgi analyzer.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model

    def analyze_connectivity(self) -> dict[str, Any]:
        """Analyze network connectivity patterns.

        Returns:
            Dict containing connectivity metrics:
                - layer_connectivity: Connection density per layer
                - receptive_field_sizes: Effective receptive field per layer
                - pathway_strengths: Information flow strength between layers
        """
        connectivity = {}
        receptive_fields = {}
        pathway_strengths = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach()

                # Connection density (fraction of non-zero weights)
                density = (weight.abs() > 1e-6).float().mean().item()
                connectivity[name] = density

                # Effective receptive field (input units with significant weights)
                input_importance = weight.abs().mean(dim=0)
                threshold = input_importance.mean() + input_importance.std()
                rf_size = (input_importance > threshold).sum().item()
                receptive_fields[name] = rf_size

                # Pathway strength (mean absolute weight)
                pathway_strengths[name] = weight.abs().mean().item()

        return {
            "layer_connectivity": connectivity,
            "receptive_field_sizes": receptive_fields,
            "pathway_strengths": pathway_strengths,
        }

    def visualize_dendrite_tree(self, layer_name: str) -> np.ndarray:
        """Create dendritic tree visualization for a layer.

        Args:
            layer_name: Name of layer to visualize

        Returns:
            Weight matrix as numpy array (for visualization)
        """
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, nn.Linear):
                return module.weight.detach().cpu().numpy()

        raise ValueError(f"Layer '{layer_name}' not found or not a Linear layer")


class NisslAnalyzer:
    """Nissl stain-inspired analysis of activation patterns.

    The Nissl stain shows:
    - Cell body locations and density
    - Layer organization
    - Neuron type distributions

    This analyzer examines:
    - Activation sparsity patterns
    - Layer-wise activation statistics
    - Neuron "type" clustering based on response profiles
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize Nissl analyzer.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.activation_history: dict[str, list[torch.Tensor]] = {}
        self._hooks: list[Any] = []

    def register_hooks(self) -> None:
        """Register forward hooks to capture activations."""

        def make_hook(name: str) -> Any:
            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                if name not in self.activation_history:
                    self.activation_history[name] = []
                self.activation_history[name].append(output.detach().cpu())

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def analyze_activations(self) -> dict[str, Any]:
        """Analyze captured activation patterns.

        Returns:
            Dict containing activation metrics:
                - sparsity: Fraction of near-zero activations per layer
                - mean_activation: Mean activation value per layer
                - activation_distribution: Histogram of activations
        """
        results = {}

        for name, activations in self.activation_history.items():
            # Concatenate all batches
            all_acts = torch.cat(activations, dim=0)

            # Compute metrics
            sparsity = (all_acts.abs() < 0.01).float().mean().item()
            mean_act = all_acts.mean().item()
            std_act = all_acts.std().item()

            # Activation distribution (histogram)
            hist, bin_edges = np.histogram(all_acts.numpy().flatten(), bins=50, density=True)

            results[name] = {
                "sparsity": sparsity,
                "mean": mean_act,
                "std": std_act,
                "histogram": hist,
                "bin_edges": bin_edges,
            }

        return results

    def clear_history(self) -> None:
        """Clear activation history."""
        self.activation_history = {}


class WeigertAnalyzer:
    """Weigert stain-inspired analysis of connection strengths.

    The Weigert stain reveals myelinated fibers (fast connections):
    - Axon pathways
    - White matter organization
    - Connection efficiency

    This analyzer examines:
    - Weight magnitude distributions
    - Strong vs weak connection ratios
    - Information bottlenecks
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize Weigert analyzer.

        Args:
            model: PyTorch model to analyze
        """
        self.model = model

    def analyze_connections(self, threshold: float = 0.1) -> dict[str, Any]:
        """Analyze connection strength patterns.

        Args:
            threshold: Threshold for "strong" connections (relative to max)

        Returns:
            Dict containing connection metrics:
                - strong_connection_ratio: Fraction of strong connections
                - weight_distribution: Statistics of weight magnitudes
                - bottleneck_layers: Layers with limited information flow
        """
        results = {}
        layer_capacities = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach()

                # Weight magnitude analysis
                mag = weight.abs()
                max_mag = mag.max().item()
                threshold_val = threshold * max_mag

                strong_ratio = (mag > threshold_val).float().mean().item()
                weak_ratio = (mag < threshold_val * 0.1).float().mean().item()

                # Information capacity (approximate)
                # Higher entropy = more diverse weights = higher capacity
                weight_flat = weight.flatten()
                hist, _ = np.histogram(weight_flat.cpu().numpy(), bins=100, density=True)
                hist = hist + 1e-10  # Avoid log(0)
                entropy = -np.sum(hist * np.log(hist)) * (hist[1] - hist[0])

                results[name] = {
                    "strong_connection_ratio": strong_ratio,
                    "weak_connection_ratio": weak_ratio,
                    "mean_weight_magnitude": mag.mean().item(),
                    "max_weight_magnitude": max_mag,
                    "weight_entropy": entropy,
                }

                layer_capacities[name] = entropy

        # Identify bottleneck layers (lowest capacity)
        if layer_capacities:
            min_capacity = min(layer_capacities.values())
            bottlenecks = [
                name for name, cap in layer_capacities.items() if cap < min_capacity * 1.5
            ]
        else:
            bottlenecks = []

        return {
            "layer_metrics": results,
            "bottleneck_layers": bottlenecks,
        }


# ==============================================================================
# Biologically-Plausible Loss Functions
# ==============================================================================


class CorticalLoss(nn.Module):
    """Biologically-plausible loss combining multiple cortical constraints.

    Combines:
    1. Task loss (classification/regression)
    2. Sparsity constraint (sparse coding)
    3. Hebbian correlation loss (local learning signal)
    4. Lateral inhibition energy (winner-take-all)
    """

    def __init__(
        self,
        task_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        hebbian_weight: float = 0.01,
        target_sparsity: float = 0.1,
    ) -> None:
        """Initialize cortical loss.

        Args:
            task_weight: Weight for main task loss
            sparsity_weight: Weight for sparsity constraint
            hebbian_weight: Weight for Hebbian correlation loss
            target_sparsity: Target activation sparsity
        """
        super().__init__()
        self.task_weight = task_weight
        self.sparsity_weight = sparsity_weight
        self.hebbian_weight = hebbian_weight
        self.target_sparsity = target_sparsity

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        activations: dict[str, torch.Tensor] | None = None,
        task_type: str = "classification",
    ) -> dict[str, torch.Tensor]:
        """Compute cortical loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            activations: Dict of layer activations (for regularization)
            task_type: "classification" or "regression"

        Returns:
            Dict containing loss components and total loss
        """
        # Task loss
        if task_type == "classification":
            # Check if multi-class classification (predictions have multiple classes)
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                # Multi-class: use cross_entropy with class indices
                if targets.dim() > 1 and targets.shape[1] > 1:
                    # One-hot encoded targets
                    task_loss = F.cross_entropy(predictions, targets.argmax(dim=1))
                else:
                    # Class index targets
                    task_loss = F.cross_entropy(predictions, targets.long())
            else:
                # Binary classification
                task_loss = F.binary_cross_entropy_with_logits(
                    predictions.squeeze(), targets.float()
                )
        else:
            task_loss = F.mse_loss(predictions.squeeze(), targets.float())

        # Sparsity loss (encourage sparse activations)
        sparsity_loss = torch.tensor(0.0, device=predictions.device)
        if activations:
            for name, act in activations.items():
                if "layer" in name:
                    # L1 sparsity penalty
                    actual_sparsity = (act.abs() < 0.01).float().mean()
                    sparsity_loss = sparsity_loss + (actual_sparsity - self.target_sparsity).abs()
            sparsity_loss = sparsity_loss / max(len(activations), 1)

        # Hebbian correlation loss (encourage decorrelated representations)
        hebbian_loss = torch.tensor(0.0, device=predictions.device)
        if activations:
            for name, act in activations.items():
                if act.dim() == 2 and act.shape[0] > 1:
                    # Covariance matrix of activations
                    act_centered = act - act.mean(dim=0, keepdim=True)
                    cov = torch.mm(act_centered.t(), act_centered) / (act.shape[0] - 1)

                    # Off-diagonal penalty (encourage decorrelation)
                    mask = ~torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
                    off_diag = cov[mask]
                    hebbian_loss = hebbian_loss + off_diag.pow(2).mean()

            hebbian_loss = hebbian_loss / max(len(activations), 1)

        # Total loss
        total = (
            self.task_weight * task_loss
            + self.sparsity_weight * sparsity_loss
            + self.hebbian_weight * hebbian_loss
        )

        return {
            "total": total,
            "task": task_loss,
            "sparsity": sparsity_loss,
            "hebbian": hebbian_loss,
        }


class SpikeTimingDependentPlasticity(nn.Module):
    """STDP-inspired learning signal for temporal sequences.

    Spike-timing dependent plasticity strengthens connections when
    pre-synaptic activity precedes post-synaptic activity (causal),
    and weakens connections otherwise (anti-causal).

    Implements a continuous approximation of STDP for gradient-based training.
    """

    def __init__(
        self,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
    ) -> None:
        """Initialize STDP.

        Args:
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms)
            a_plus: Learning rate for potentiation
            a_minus: Learning rate for depression
        """
        super().__init__()
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    def forward(self, pre_times: torch.Tensor, post_times: torch.Tensor) -> torch.Tensor:
        """Compute STDP weight update.

        Args:
            pre_times: Pre-synaptic spike times [batch, neurons_pre]
            post_times: Post-synaptic spike times [batch, neurons_post]

        Returns:
            Weight update matrix [neurons_pre, neurons_post]
        """
        # Time differences: post - pre (positive = causal, potentiation)
        dt = post_times.unsqueeze(1) - pre_times.unsqueeze(2)  # [batch, pre, post]

        # STDP kernel with proper masking
        # Potentiation for dt > 0 (pre before post) - causal timing
        potentiation_mask = (dt > 0).float()
        potentiation = self.a_plus * torch.exp(-dt.abs() / self.tau_plus) * potentiation_mask

        # Depression for dt < 0 (post before pre) - anti-causal timing
        depression_mask = (dt < 0).float()
        depression = -self.a_minus * torch.exp(-dt.abs() / self.tau_minus) * depression_mask

        # Combine and average over batch
        delta_w = (potentiation + depression).mean(dim=0)

        return delta_w

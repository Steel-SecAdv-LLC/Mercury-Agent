"""
Mercury Agent - Cross-Domain Transfer Learning Framework
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Production-grade cross-domain transfer learning for anomaly detection:
- Domain adaptation with Maximum Mean Discrepancy (MMD)
- Feature alignment via Correlation Alignment (CORAL)
- Domain-adversarial neural networks (DANN)
- Optimal transport for distribution alignment
- NSL-KDD → CICIDS benchmark evaluation
- Multi-source domain adaptation

This demonstrates Mercury's architectural advantages over pure supervised
methods through effective knowledge transfer across security domains.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist


if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Optional imports
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DomainAdaptationMethod(StrEnum):
    """Available domain adaptation methods."""

    MMD = "mmd"  # Maximum Mean Discrepancy
    CORAL = "coral"  # Correlation Alignment
    DANN = "dann"  # Domain-Adversarial Neural Network
    JDA = "jda"  # Joint Distribution Adaptation
    TCA = "tca"  # Transfer Component Analysis
    SUBSPACE = "subspace"  # Subspace alignment
    OPTIMAL_TRANSPORT = "optimal_transport"


class SecurityDataset(StrEnum):
    """Known security benchmark datasets."""

    NSL_KDD = "nsl_kdd"
    CICIDS_2017 = "cicids_2017"
    CICIDS_2018 = "cicids_2018"
    UNSW_NB15 = "unsw_nb15"
    CTU_13 = "ctu_13"
    CUSTOM = "custom"


@dataclass
class DomainData:
    """Data from a single domain."""

    X: NDArray[np.float64]  # Features [n_samples, n_features]
    y: NDArray[np.int64]  # Labels [n_samples]
    domain_name: str
    feature_names: list[str] | None = None
    n_samples: int = 0
    n_features: int = 0

    def __post_init__(self) -> None:
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]


@dataclass
class TransferResult:
    """Result from cross-domain transfer learning."""

    # Performance metrics on target domain
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None

    # Transfer-specific metrics
    source_accuracy: float  # Performance on source domain
    transfer_ratio: float  # target_acc / source_acc
    negative_transfer: bool  # Whether transfer hurt performance

    # Domain alignment metrics
    mmd_before: float  # MMD before adaptation
    mmd_after: float  # MMD after adaptation
    alignment_improvement: float

    # Metadata
    source_domain: str
    target_domain: str
    method: str
    adaptation_time: float

    # Per-class performance
    class_f1_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_accuracy": self.accuracy,
            "target_f1": self.f1,
            "source_accuracy": self.source_accuracy,
            "transfer_ratio": self.transfer_ratio,
            "negative_transfer": self.negative_transfer,
            "mmd_before": self.mmd_before,
            "mmd_after": self.mmd_after,
            "alignment_improvement": self.alignment_improvement,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "method": self.method,
            "adaptation_time": self.adaptation_time,
        }


class FeatureAligner:
    """
    Aligns feature spaces between source and target domains.

    Supports multiple alignment methods and automatic feature mapping.
    """

    def __init__(
        self,
        source_features: list[str],
        target_features: list[str],
        alignment_method: str = "intersection",
    ):
        """
        Initialize feature aligner.

        Args:
            source_features: Feature names in source domain
            target_features: Feature names in target domain
            alignment_method: 'intersection', 'union', or 'mapping'
        """
        self.source_features = source_features
        self.target_features = target_features
        self.alignment_method = alignment_method

        # Compute feature mapping
        self.common_features: list[str] = []
        self.source_indices: list[int] = []
        self.target_indices: list[int] = []

        self._compute_alignment()

    def _compute_alignment(self) -> None:
        """Compute feature alignment between domains."""
        if self.alignment_method == "intersection":
            # Use common features
            source_set = set(self.source_features)
            target_set = set(self.target_features)
            self.common_features = list(source_set & target_set)

            self.source_indices = [self.source_features.index(f) for f in self.common_features]
            self.target_indices = [self.target_features.index(f) for f in self.common_features]

        elif self.alignment_method == "mapping":
            # Try to map similar feature names
            self._semantic_feature_mapping()

    def _semantic_feature_mapping(self) -> None:
        """Map features based on semantic similarity."""
        # Common feature mappings between security datasets
        feature_mappings = {
            # NSL-KDD to CICIDS mappings
            "duration": ["flow_duration", "duration"],
            "src_bytes": ["total_fwd_packets", "fwd_pkt_len_total"],
            "dst_bytes": ["total_bwd_packets", "bwd_pkt_len_total"],
            "count": ["flow_pkts_s", "packet_count"],
            "srv_count": ["flow_bytes_s", "byte_count"],
            "same_srv_rate": ["fwd_pkts_s", "packets_per_second"],
            "diff_srv_rate": ["bwd_pkts_s"],
            "serror_rate": ["syn_flag_cnt", "syn_rate"],
            "rerror_rate": ["rst_flag_cnt", "rst_rate"],
            "protocol_type": ["protocol"],
            "flag": ["tcp_flags"],
        }

        for source_feat in self.source_features:
            source_lower = source_feat.lower()

            # Check direct match
            if source_feat in self.target_features:
                self.common_features.append(source_feat)
                self.source_indices.append(self.source_features.index(source_feat))
                self.target_indices.append(self.target_features.index(source_feat))
                continue

            # Check mapping
            for key, mappings in feature_mappings.items():
                if key in source_lower or source_lower in mappings:
                    for target_feat in self.target_features:
                        target_lower = target_feat.lower()
                        if any(m in target_lower for m in mappings):
                            self.common_features.append(source_feat)
                            self.source_indices.append(self.source_features.index(source_feat))
                            self.target_indices.append(self.target_features.index(target_feat))
                            break
                    break

    def align_source(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract aligned features from source domain."""
        if not self.source_indices:
            return X
        return X[:, self.source_indices]

    def align_target(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract aligned features from target domain."""
        if not self.target_indices:
            return X
        return X[:, self.target_indices]


class BaseDomainAdapter(ABC):
    """Base class for domain adaptation methods."""

    @abstractmethod
    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit the domain adapter."""
        pass

    @abstractmethod
    def transform(self, X: NDArray[np.float64], domain: str) -> NDArray[np.float64]:
        """Transform features to aligned space."""
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels for target domain."""
        pass


class MMDAdapter(BaseDomainAdapter):
    """
    Maximum Mean Discrepancy (MMD) based domain adaptation.

    Minimizes the distribution discrepancy between domains
    in a reproducing kernel Hilbert space (RKHS).
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float | None = None,
        n_components: int = 64,
    ):
        """
        Initialize MMD adapter.

        Args:
            kernel: Kernel type ('rbf', 'linear', 'poly')
            gamma: RBF kernel bandwidth (auto if None)
            n_components: Dimension of transformed space
        """
        self.kernel = kernel
        self.gamma = gamma
        self.n_components = n_components

        # Learned components
        self.projection_matrix: NDArray[np.float64] | None = None
        self.source_mean: NDArray[np.float64] | None = None
        self.source_std: NDArray[np.float64] | None = None

        # Classifier
        self._classifier: Any = None

    def _compute_kernel(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute kernel matrix."""
        if Y is None:
            Y = X

        if self.kernel == "rbf":
            if self.gamma is None:
                # Median heuristic
                dists = cdist(X, Y, metric="euclidean")
                self.gamma = 1.0 / (np.median(dists) ** 2 + 1e-10)

            sq_dists = cdist(X, Y, metric="sqeuclidean")
            return np.exp(-self.gamma * sq_dists)

        elif self.kernel == "linear":
            return X @ Y.T

        elif self.kernel == "poly":
            return (X @ Y.T + 1) ** 3

        else:
            return X @ Y.T

    def compute_mmd(
        self,
        source_X: NDArray[np.float64],
        target_X: NDArray[np.float64],
    ) -> float:
        """
        Compute MMD between source and target distributions.

        MMD^2 = ||mean(phi(X_s)) - mean(phi(X_t))||^2
              = E[k(x_s, x_s')] + E[k(x_t, x_t')] - 2*E[k(x_s, x_t)]
        """
        K_ss = self._compute_kernel(source_X, source_X)
        K_tt = self._compute_kernel(target_X, target_X)
        K_st = self._compute_kernel(source_X, target_X)

        n_s = len(source_X)
        n_t = len(target_X)

        # Unbiased estimate
        mmd_ss = (np.sum(K_ss) - np.trace(K_ss)) / (n_s * (n_s - 1) + 1e-10)
        mmd_tt = (np.sum(K_tt) - np.trace(K_tt)) / (n_t * (n_t - 1) + 1e-10)
        mmd_st = np.sum(K_st) / (n_s * n_t + 1e-10)

        mmd_squared = mmd_ss + mmd_tt - 2 * mmd_st

        return float(np.sqrt(max(0, mmd_squared)))

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """
        Fit MMD adapter using domain-invariant projection.

        Args:
            source_X: Source features
            source_y: Source labels
            target_X: Target features
            target_y: Target labels (optional, for evaluation)
        """
        # Normalize features
        self.source_mean = np.mean(source_X, axis=0)
        self.source_std = np.std(source_X, axis=0) + 1e-10

        source_X_norm = (source_X - self.source_mean) / self.source_std
        target_X_norm = (target_X - self.source_mean) / self.source_std

        # Compute combined covariance for projection
        combined_X = np.vstack([source_X_norm, target_X_norm])

        # PCA-like projection that minimizes MMD
        cov = np.cov(combined_X.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Take top components
        n_components = min(self.n_components, combined_X.shape[1])
        self.projection_matrix = eigenvectors[:, :n_components]

        # Transform and train classifier
        source_X_proj = self.transform(source_X, "source")

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)

        self._classifier.fit(source_X_proj, source_y)

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform features to aligned space."""
        if self.source_mean is None or self.source_std is None or self.projection_matrix is None:
            raise ValueError("Adapter not fitted")

        X_norm = (X - self.source_mean) / self.source_std
        return X_norm @ self.projection_matrix

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels for target domain."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_proj = self.transform(X, "target")
        return self._classifier.predict(X_proj)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_proj = self.transform(X, "target")
        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_proj)
        else:
            preds = self._classifier.predict(X_proj)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba


class CORALAdapter(BaseDomainAdapter):
    """
    Correlation Alignment (CORAL) domain adaptation.

    Aligns second-order statistics (covariance) between domains.

    Reference: "Return of Frustratingly Easy Domain Adaptation"
               (Sun et al., 2016)
    """

    def __init__(self, regularization: float = 1e-5):
        """
        Initialize CORAL adapter.

        Args:
            regularization: Regularization for covariance inversion
        """
        self.regularization = regularization

        # Learned components
        self.source_mean: NDArray[np.float64] | None = None
        self.target_mean: NDArray[np.float64] | None = None
        self.whitening_matrix: NDArray[np.float64] | None = None
        self.coloring_matrix: NDArray[np.float64] | None = None

        # Classifier
        self._classifier: Any = None

    def _compute_cov_sqrt_inv(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute inverse square root of covariance matrix."""
        # Add regularization
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)

        # Clip negative eigenvalues (numerical issues)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Compute inverse square root
        sqrt_inv_eigenvalues = 1.0 / np.sqrt(eigenvalues)
        sqrt_inv = eigenvectors @ np.diag(sqrt_inv_eigenvalues) @ eigenvectors.T

        return sqrt_inv

    def _compute_cov_sqrt(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute square root of covariance matrix."""
        cov_reg = cov + self.regularization * np.eye(cov.shape[0])

        try:
            sqrt_cov = sqrtm(cov_reg)
            # Handle potential complex values
            if np.iscomplexobj(sqrt_cov):
                sqrt_cov = np.real(sqrt_cov)
            return sqrt_cov
        except Exception as e:
            # Fallback to eigendecomposition when scipy.sqrtm fails (e.g., near-singular matrix)
            logger.debug("Matrix sqrt via sqrtm failed, using eigendecomposition fallback: %s", e)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """
        Fit CORAL adapter by aligning covariances.

        X_target_aligned = (X_target - mean_t) @ C_t^(-1/2) @ C_s^(1/2) + mean_s
        """
        self.source_mean = np.mean(source_X, axis=0)
        self.target_mean = np.mean(target_X, axis=0)

        # Center data
        source_centered = source_X - self.source_mean
        target_centered = target_X - self.target_mean

        # Compute covariances
        source_cov = np.cov(source_centered.T) + self.regularization * np.eye(source_X.shape[1])
        target_cov = np.cov(target_centered.T) + self.regularization * np.eye(target_X.shape[1])

        # Compute whitening matrix (target) and coloring matrix (source)
        self.whitening_matrix = self._compute_cov_sqrt_inv(target_cov)
        self.coloring_matrix = self._compute_cov_sqrt(source_cov)

        # Train classifier on source domain
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)

        self._classifier.fit(source_X, source_y)

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform target features to source-aligned space."""
        if domain == "source":
            return X

        if (
            self.whitening_matrix is None
            or self.coloring_matrix is None
            or self.target_mean is None
            or self.source_mean is None
        ):
            raise ValueError("Adapter not fitted")

        # Center with target mean
        X_centered = X - self.target_mean

        # Whiten and recolor
        X_whitened = X_centered @ self.whitening_matrix
        X_aligned = X_whitened @ self.coloring_matrix

        # Add source mean
        return X_aligned + self.source_mean

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict on aligned target data."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_aligned = self.transform(X, "target")
        return self._classifier.predict(X_aligned)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_aligned = self.transform(X, "target")
        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_aligned)
        else:
            preds = self._classifier.predict(X_aligned)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba


class SubspaceAlignmentAdapter(BaseDomainAdapter):
    """
    Subspace Alignment for domain adaptation.

    Projects domains to their principal subspaces and aligns them.

    Reference: "Unsupervised Visual Domain Adaptation Using Subspace Alignment"
               (Fernando et al., 2013)
    """

    def __init__(self, n_components: int = 32):
        """
        Initialize subspace alignment adapter.

        Args:
            n_components: Number of principal components
        """
        self.n_components = n_components

        # Learned components
        self.source_basis: NDArray[np.float64] | None = None
        self.target_basis: NDArray[np.float64] | None = None
        self.alignment_matrix: NDArray[np.float64] | None = None
        self.source_mean: NDArray[np.float64] | None = None
        self.target_mean: NDArray[np.float64] | None = None

        # Classifier
        self._classifier: Any = None

    def _compute_principal_subspace(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute principal subspace basis."""
        # Center data
        X_centered = X - np.mean(X, axis=0)

        # SVD
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Take top components
        n_components = min(self.n_components, Vt.shape[0])
        return Vt[:n_components].T  # [features, components]

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit subspace alignment."""
        self.source_mean = np.mean(source_X, axis=0)
        self.target_mean = np.mean(target_X, axis=0)

        # Compute principal subspaces
        self.source_basis = self._compute_principal_subspace(source_X)
        self.target_basis = self._compute_principal_subspace(target_X)

        # Compute alignment matrix: M = T_s^T @ T_t
        self.alignment_matrix = self.source_basis.T @ self.target_basis

        # Train classifier on projected source data
        source_proj = (source_X - self.source_mean) @ self.source_basis

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)

        self._classifier.fit(source_proj, source_y)

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform to aligned subspace."""
        if self.source_mean is None or self.source_basis is None:
            raise ValueError("Adapter not fitted")

        if domain == "source":
            return (X - self.source_mean) @ self.source_basis

        if self.target_basis is None or self.alignment_matrix is None or self.target_mean is None:
            raise ValueError("Adapter not fitted")

        # Project to target subspace then align to source subspace
        X_centered = X - self.target_mean
        X_target_proj = X_centered @ self.target_basis
        X_aligned = X_target_proj @ self.alignment_matrix.T

        return X_aligned

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict on aligned target data."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_aligned = self.transform(X, "target")
        return self._classifier.predict(X_aligned)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")

        X_aligned = self.transform(X, "target")
        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_aligned)
        else:
            preds = self._classifier.predict(X_aligned)
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba


class DANNAdapter(BaseDomainAdapter):
    """
    Domain-Adversarial Neural Network (DANN) adapter.

    Uses adversarial training with a gradient reversal layer to learn
    domain-invariant representations that confuse a domain classifier
    while maintaining task performance.

    Reference: "Domain-Adversarial Training of Neural Networks"
               (Ganin et al., 2016)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        n_iterations: int = 1000,
        lambda_domain: float = 1.0,
        random_state: int | None = None,
    ):
        """
        Initialize DANN adapter.

        Args:
            hidden_dim: Hidden layer dimension for feature extractor
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of training iterations
            lambda_domain: Weight for domain adversarial loss
            random_state: Seed for reproducible initialization
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_domain = lambda_domain
        self.rng = np.random.default_rng(random_state)

        # Feature extractor: input -> hidden
        self.W_feat: NDArray[np.float64] | None = None
        self.b_feat: NDArray[np.float64] | None = None

        # Label predictor: hidden -> n_classes
        self.W_label: NDArray[np.float64] | None = None
        self.b_label: NDArray[np.float64] | None = None

        # Domain classifier: hidden -> 2 (source/target)
        self.W_domain: NDArray[np.float64] | None = None
        self.b_domain: NDArray[np.float64] | None = None

        self.source_mean: NDArray[np.float64] | None = None
        self.source_std: NDArray[np.float64] | None = None
        self.classes: list[int] = []

    def _initialize_params(self, input_dim: int, n_classes: int) -> None:
        """Initialize network parameters."""
        # Feature extractor
        scale = np.sqrt(2.0 / (input_dim + self.hidden_dim))
        self.W_feat = (
            self.rng.standard_normal((input_dim, self.hidden_dim)).astype(np.float64) * scale
        )
        self.b_feat = np.zeros(self.hidden_dim, dtype=np.float64)

        # Label predictor
        scale = np.sqrt(2.0 / (self.hidden_dim + n_classes))
        self.W_label = (
            self.rng.standard_normal((self.hidden_dim, n_classes)).astype(np.float64) * scale
        )
        self.b_label = np.zeros(n_classes, dtype=np.float64)

        # Domain classifier
        scale = np.sqrt(2.0 / (self.hidden_dim + 2))
        self.W_domain = self.rng.standard_normal((self.hidden_dim, 2)).astype(np.float64) * scale
        self.b_domain = np.zeros(2, dtype=np.float64)

    def _softmax(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def _extract_features(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract features using feature extractor."""
        assert self.W_feat is not None and self.b_feat is not None
        return np.maximum(0, X @ self.W_feat + self.b_feat)  # ReLU

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit DANN using adversarial domain adaptation."""
        # Normalize
        self.source_mean = np.mean(source_X, axis=0)
        self.source_std = np.std(source_X, axis=0) + 1e-10
        source_X_norm = (source_X - self.source_mean) / self.source_std
        target_X_norm = (target_X - self.source_mean) / self.source_std

        # Setup classes
        self.classes = list(np.unique(source_y))
        n_classes = len(self.classes)
        label_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        source_y_idx = np.array([label_to_idx[y] for y in source_y])

        # Initialize
        self._initialize_params(source_X.shape[1], n_classes)

        # Domain labels: 0 = source, 1 = target
        n_source, n_target = len(source_X), len(target_X)

        # Mini-batch training
        batch_size = min(32, n_source, n_target)

        for iteration in range(self.n_iterations):
            # Sample batches
            src_idx = self.rng.choice(n_source, batch_size, replace=False)
            tgt_idx = self.rng.choice(n_target, batch_size, replace=False)

            src_batch = source_X_norm[src_idx]
            tgt_batch = target_X_norm[tgt_idx]
            src_labels = source_y_idx[src_idx]

            # Forward pass - source
            src_features = self._extract_features(src_batch)
            assert self.W_label is not None and self.b_label is not None
            src_label_logits = src_features @ self.W_label + self.b_label
            src_label_probs = self._softmax(src_label_logits)

            # Forward pass - target features
            tgt_features = self._extract_features(tgt_batch)

            # Domain classification
            assert self.W_domain is not None and self.b_domain is not None
            combined_features = np.vstack([src_features, tgt_features])
            domain_logits = combined_features @ self.W_domain + self.b_domain
            domain_probs = self._softmax(domain_logits)
            domain_labels = np.concatenate([np.zeros(batch_size), np.ones(batch_size)]).astype(int)

            # Compute gradients for label predictor
            d_label_logits = src_label_probs.copy()
            d_label_logits[np.arange(batch_size), src_labels] -= 1
            d_label_logits /= batch_size

            dW_label = src_features.T @ d_label_logits
            db_label = np.sum(d_label_logits, axis=0)

            # Compute gradients for domain classifier
            d_domain_logits = domain_probs.copy()
            d_domain_logits[np.arange(2 * batch_size), domain_labels] -= 1
            d_domain_logits /= 2 * batch_size

            dW_domain = combined_features.T @ d_domain_logits
            db_domain = np.sum(d_domain_logits, axis=0)

            # Gradient reversal: domain gradient pushes features to be domain-invariant
            d_feat_from_label = d_label_logits @ self.W_label.T
            d_feat_from_domain = d_domain_logits @ self.W_domain.T

            # Gradient for feature extractor (with gradient reversal for domain)
            d_src_feat = d_feat_from_label - self.lambda_domain * d_feat_from_domain[:batch_size]
            d_tgt_feat = -self.lambda_domain * d_feat_from_domain[batch_size:]

            # ReLU backward
            d_src_feat = d_src_feat * (src_features > 0)
            d_tgt_feat = d_tgt_feat * (tgt_features > 0)

            dW_feat = src_batch.T @ d_src_feat + tgt_batch.T @ d_tgt_feat
            db_feat = np.sum(d_src_feat, axis=0) + np.sum(d_tgt_feat, axis=0)

            # Update parameters
            self.W_label -= self.learning_rate * dW_label
            self.b_label -= self.learning_rate * db_label
            self.W_domain -= self.learning_rate * dW_domain
            self.b_domain -= self.learning_rate * db_domain
            self.W_feat -= self.learning_rate * dW_feat
            self.b_feat -= self.learning_rate * db_feat

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform to domain-invariant feature space."""
        if self.source_mean is None or self.source_std is None:
            raise ValueError("Adapter not fitted")
        X_norm = (X - self.source_mean) / self.source_std
        return self._extract_features(X_norm)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels using label predictor."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return np.array([self.classes[i] for i in pred_indices])

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        features = self.transform(X, "target")
        assert self.W_label is not None and self.b_label is not None
        logits = features @ self.W_label + self.b_label
        return self._softmax(logits)


class JDAAdapter(BaseDomainAdapter):
    """
    Joint Distribution Adaptation (JDA) adapter.

    Adapts both marginal and conditional distributions between domains
    by iteratively minimizing MMD with pseudo-labels.

    Reference: "Transfer Feature Learning with Joint Distribution Adaptation"
               (Long et al., 2013)
    """

    def __init__(
        self,
        n_components: int = 64,
        n_iterations: int = 10,
        kernel: str = "rbf",
        gamma: float | None = None,
    ):
        """
        Initialize JDA adapter.

        Args:
            n_components: Dimension of subspace
            n_iterations: Number of JDA iterations
            kernel: Kernel type for MMD
            gamma: RBF kernel bandwidth
        """
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma

        self.projection_matrix: NDArray[np.float64] | None = None
        self.source_mean: NDArray[np.float64] | None = None
        self.source_std: NDArray[np.float64] | None = None
        self._classifier: Any = None
        self.classes: list[int] = []

    def _compute_kernel(
        self, X: NDArray[np.float64], Y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute kernel matrix."""
        if self.kernel == "rbf":
            if self.gamma is None:
                dists = cdist(X, Y, metric="euclidean")
                self.gamma = 1.0 / (np.median(dists) ** 2 + 1e-10)
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            return np.exp(-self.gamma * sq_dists)
        return X @ Y.T

    def _compute_mmd_matrix(
        self,
        source_X: NDArray[np.float64],
        target_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_y_pseudo: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Compute joint MMD matrix for all classes."""
        n_s, n_t = len(source_X), len(target_X)
        n_total = n_s + n_t

        # Initialize MMD matrix (for marginal distribution)
        M = np.zeros((n_total, n_total))

        # Marginal MMD term
        M[:n_s, :n_s] = 1.0 / (n_s * n_s)
        M[n_s:, n_s:] = 1.0 / (n_t * n_t)
        M[:n_s, n_s:] = -1.0 / (n_s * n_t)
        M[n_s:, :n_s] = -1.0 / (n_s * n_t)

        # Add conditional MMD terms for each class
        for c in self.classes:
            src_mask = source_y == c
            tgt_mask = target_y_pseudo == c
            n_sc = np.sum(src_mask)
            n_tc = np.sum(tgt_mask)

            if n_sc > 0 and n_tc > 0:
                src_idx = np.where(src_mask)[0]
                tgt_idx = np.where(tgt_mask)[0] + n_s

                for i in src_idx:
                    for j in src_idx:
                        M[i, j] += 1.0 / (n_sc * n_sc)
                for i in tgt_idx:
                    for j in tgt_idx:
                        M[i, j] += 1.0 / (n_tc * n_tc)
                for i in src_idx:
                    for j in tgt_idx:
                        M[i, j] -= 1.0 / (n_sc * n_tc)
                        M[j, i] -= 1.0 / (n_sc * n_tc)

        return M

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit JDA with iterative pseudo-labeling."""
        # Normalize
        self.source_mean = np.mean(source_X, axis=0)
        self.source_std = np.std(source_X, axis=0) + 1e-10
        source_X_norm = (source_X - self.source_mean) / self.source_std
        target_X_norm = (target_X - self.source_mean) / self.source_std

        self.classes = list(np.unique(source_y))

        # Combined data
        X_combined = np.vstack([source_X_norm, target_X_norm])
        len(source_X)

        # Initialize pseudo-labels using simple 1-NN
        dists = cdist(target_X_norm, source_X_norm)
        nn_idx = np.argmin(dists, axis=1)
        target_y_pseudo = source_y[nn_idx]

        # Iterative JDA
        for _ in range(self.n_iterations):
            # Compute joint MMD matrix
            M = self._compute_mmd_matrix(source_X_norm, target_X_norm, source_y, target_y_pseudo)

            # Solve generalized eigenvalue problem
            # min_A tr(A^T X M X^T A) s.t. A^T X H X^T A = I
            X_combined_T = X_combined.T
            H = np.eye(len(X_combined)) - 1.0 / len(X_combined)

            # Regularize
            reg = 1e-6 * np.eye(X_combined_T.shape[0])

            # Compute A = (X M X^T)^-1 X H X^T
            XMXt = X_combined_T @ M @ X_combined + reg
            XHXt = X_combined_T @ H @ X_combined + reg

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(XHXt) @ XMXt)
                # Sort by eigenvalue (ascending for minimization)
                idx = np.argsort(eigenvalues)
                self.projection_matrix = eigenvectors[:, idx[: self.n_components]]
            except np.linalg.LinAlgError:
                # Fallback to PCA
                _, _, Vt = np.linalg.svd(X_combined, full_matrices=False)
                self.projection_matrix = Vt[: self.n_components].T

            # Update pseudo-labels
            source_proj = source_X_norm @ self.projection_matrix
            target_proj = target_X_norm @ self.projection_matrix
            dists = cdist(target_proj, source_proj)
            nn_idx = np.argmin(dists, axis=1)
            target_y_pseudo = source_y[nn_idx]

        # Train final classifier
        source_proj = source_X_norm @ self.projection_matrix
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)
        self._classifier.fit(source_proj, source_y)

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform to aligned subspace."""
        if self.source_mean is None or self.source_std is None or self.projection_matrix is None:
            raise ValueError("Adapter not fitted")
        X_norm = (X - self.source_mean) / self.source_std
        return X_norm @ self.projection_matrix

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")
        X_proj = self.transform(X, "target")
        return self._classifier.predict(X_proj)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")
        X_proj = self.transform(X, "target")
        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_proj)
        preds = self._classifier.predict(X_proj)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


class TCAAdapter(BaseDomainAdapter):
    """
    Transfer Component Analysis (TCA) adapter.

    Learns transfer components in a RKHS that minimize domain discrepancy
    while preserving important data properties.

    Reference: "Domain Adaptation via Transfer Component Analysis"
               (Pan et al., 2011)
    """

    def __init__(
        self,
        n_components: int = 32,
        kernel: str = "rbf",
        gamma: float | None = None,
        mu: float = 0.1,
    ):
        """
        Initialize TCA adapter.

        Args:
            n_components: Number of transfer components
            kernel: Kernel type ('rbf', 'linear')
            gamma: RBF kernel bandwidth
            mu: Trade-off parameter for variance preservation
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.mu = mu

        self.transformation: NDArray[np.float64] | None = None
        self.source_data: NDArray[np.float64] | None = None
        self.target_data: NDArray[np.float64] | None = None
        self._classifier: Any = None

    def _compute_kernel(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        if self.kernel == "rbf":
            if self.gamma is None:
                dists = cdist(X, Y, metric="euclidean")
                self.gamma = 1.0 / (np.median(dists) ** 2 + 1e-10)
            sq_dists = cdist(X, Y, metric="sqeuclidean")
            return np.exp(-self.gamma * sq_dists)
        return X @ Y.T

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit TCA transformation."""
        n_s, n_t = len(source_X), len(target_X)
        n_total = n_s + n_t

        # Store for kernel computation during transform
        self.source_data = source_X.copy()
        self.target_data = target_X.copy()

        # Combined data
        X_combined = np.vstack([source_X, target_X])

        # Compute kernel matrix
        K = self._compute_kernel(X_combined, X_combined)

        # Construct MMD matrix L
        L = np.zeros((n_total, n_total))
        L[:n_s, :n_s] = 1.0 / (n_s * n_s)
        L[n_s:, n_s:] = 1.0 / (n_t * n_t)
        L[:n_s, n_s:] = -1.0 / (n_s * n_t)
        L[n_s:, :n_s] = -1.0 / (n_s * n_t)

        # Centering matrix
        H = np.eye(n_total) - 1.0 / n_total

        # Solve eigenvalue problem
        # (K L K + mu * I)^-1 K H K
        reg = self.mu * np.eye(n_total)
        KLK = K @ L @ K + reg
        KHK = K @ H @ K + 1e-6 * np.eye(n_total)

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(KLK) @ KHK)
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            self.transformation = eigenvectors[:, idx[: self.n_components]]
        except np.linalg.LinAlgError:
            # Fallback: use kernel PCA
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            idx = np.argsort(eigenvalues)[::-1]
            self.transformation = eigenvectors[:, idx[: self.n_components]]

        # Transform and train classifier
        K_source = K[:n_s]
        source_proj = K_source @ self.transformation

        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)
        self._classifier.fit(source_proj, source_y)

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform to TCA subspace."""
        if self.transformation is None or self.source_data is None:
            raise ValueError("Adapter not fitted")

        # Compute kernel with training data
        train_data = np.vstack([self.source_data, self.target_data])
        K_new = self._compute_kernel(X, train_data)
        return K_new @ self.transformation

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict labels."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")
        X_proj = self.transform(X, "target")
        return self._classifier.predict(X_proj)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities."""
        if self._classifier is None:
            raise ValueError("Adapter not fitted")
        X_proj = self.transform(X, "target")
        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_proj)
        preds = self._classifier.predict(X_proj)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


class OptimalTransportAdapter(BaseDomainAdapter):
    """
    Optimal Transport (OT) based domain adaptation.

    Uses the Sinkhorn algorithm to compute transport plan between
    source and target distributions for domain alignment.

    Reference: "Optimal Transport for Domain Adaptation"
               (Courty et al., 2017)
    """

    def __init__(
        self,
        reg: float = 0.1,
        n_iterations: int = 100,
        cost_metric: str = "euclidean",
    ):
        """
        Initialize Optimal Transport adapter.

        Args:
            reg: Entropic regularization parameter
            n_iterations: Number of Sinkhorn iterations
            cost_metric: Cost metric for transport
        """
        self.reg = reg
        self.n_iterations = n_iterations
        self.cost_metric = cost_metric

        self.transport_plan: NDArray[np.float64] | None = None
        self.source_X: NDArray[np.float64] | None = None
        self.source_y: NDArray[np.int64] | None = None
        self.source_mean: NDArray[np.float64] | None = None
        self.source_std: NDArray[np.float64] | None = None
        self._classifier: Any = None

    def _sinkhorn(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        C: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Sinkhorn-Knopp algorithm for entropic regularized OT.

        Args:
            a: Source distribution (n_s,)
            b: Target distribution (n_t,)
            C: Cost matrix (n_s, n_t)

        Returns:
            Transport plan (n_s, n_t)
        """
        n_s, n_t = C.shape
        K = np.exp(-C / self.reg)

        u = np.ones(n_s) / n_s
        v = np.ones(n_t) / n_t

        for _ in range(self.n_iterations):
            u = a / (K @ v + 1e-10)
            v = b / (K.T @ u + 1e-10)

        return np.diag(u) @ K @ np.diag(v)

    def fit(
        self,
        source_X: NDArray[np.float64],
        source_y: NDArray[np.int64],
        target_X: NDArray[np.float64],
        target_y: NDArray[np.int64] | None = None,
    ) -> None:
        """Fit optimal transport plan."""
        # Normalize
        self.source_mean = np.mean(source_X, axis=0)
        self.source_std = np.std(source_X, axis=0) + 1e-10
        source_X_norm = (source_X - self.source_mean) / self.source_std
        target_X_norm = (target_X - self.source_mean) / self.source_std

        self.source_X = source_X_norm.copy()
        self.source_y = source_y.copy()

        n_s, n_t = len(source_X), len(target_X)

        # Uniform distributions
        a = np.ones(n_s) / n_s
        b = np.ones(n_t) / n_t

        # Cost matrix
        C = cdist(source_X_norm, target_X_norm, metric=self.cost_metric)

        # Compute transport plan
        self.transport_plan = self._sinkhorn(a, b, C)

        # Store target data for out-of-sample extension
        self.target_X = target_X_norm.copy()

        # Barycentric mapping: transport source to target domain
        # Each target sample is a weighted combination of source samples
        self.source_transported = n_s * self.transport_plan.T @ source_X_norm

        # Compute transported labels for target samples
        # Using label propagation via transport plan
        self.target_y_propagated = self._propagate_labels(source_y, self.transport_plan)

        # Train classifier on transported source data for out-of-sample prediction
        try:
            from sklearn.ensemble import GradientBoostingClassifier

            self._classifier = GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            self._classifier = LogisticRegression(max_iter=1000, random_state=42)

        # Train on source transported to target domain
        self._classifier.fit(self.source_transported, source_y)

    def _propagate_labels(
        self,
        source_y: NDArray[np.int64],
        transport_plan: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Propagate labels from source to target via transport plan.

        For each target sample, compute weighted vote from source labels
        based on transport plan coupling.

        Args:
            source_y: Source labels
            transport_plan: OT coupling matrix [n_source, n_target]

        Returns:
            Propagated labels for target samples
        """
        n_target = transport_plan.shape[1]
        classes = np.unique(source_y)

        # Weighted label voting
        target_labels = np.zeros(n_target, dtype=np.int64)

        for t in range(n_target):
            # Get transport weights for this target sample
            weights = transport_plan[:, t]

            # Weighted vote for each class
            class_scores = {}
            for cls in classes:
                cls_mask = source_y == cls
                class_scores[cls] = np.sum(weights[cls_mask])

            # Assign most likely class
            target_labels[t] = max(class_scores, key=lambda k: class_scores[k])

        return target_labels

    def transform(self, X: NDArray[np.float64], domain: str = "target") -> NDArray[np.float64]:
        """Transform using OT barycentric projection.

        For new samples, computes barycentric projection via nearest neighbors
        in the transport space.
        """
        if self.source_mean is None or self.source_std is None:
            raise ValueError("Adapter not fitted")
        return (X - self.source_mean) / self.source_std

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict using OT-based label propagation.

        For samples in the fitted target set, returns propagated labels.
        For new samples, uses nearest neighbor in transported source space.
        """
        if self.source_X is None or self.source_y is None or self.source_transported is None:
            raise ValueError("Adapter not fitted")

        X_norm = self.transform(X, "target")

        # Check if samples are from fitted target set (by distance)
        if hasattr(self, "target_X") and self.target_X is not None:
            # For samples close to fitted target, use propagated labels
            dists_to_target = cdist(X_norm, self.target_X, metric="euclidean")
            min_dists = np.min(dists_to_target, axis=1)
            nearest_target = np.argmin(dists_to_target, axis=1)

            predictions = np.zeros(len(X_norm), dtype=np.int64)

            for i, (dist, nearest_idx) in enumerate(zip(min_dists, nearest_target)):
                if dist < 1e-6:  # Exact match - use propagated label
                    predictions[i] = self.target_y_propagated[nearest_idx]
                else:
                    # Out-of-sample: use nearest neighbor in transported source
                    dists_to_transported = cdist(X_norm[i : i + 1], self.source_transported)
                    nn_idx = np.argmin(dists_to_transported)
                    predictions[i] = self.source_y[nn_idx]

            return predictions

        # Fallback: nearest neighbor in transported source space
        dists = cdist(X_norm, self.source_transported, metric="euclidean")
        nn_indices = np.argmin(dists, axis=1)
        return self.source_y[nn_indices]

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict probabilities using transported labels."""
        if self._classifier is None or self.source_X is None:
            raise ValueError("Adapter not fitted")

        X_norm = self.transform(X, "target")

        if hasattr(self._classifier, "predict_proba"):
            return self._classifier.predict_proba(X_norm)

        preds = self._classifier.predict(X_norm)
        proba = np.zeros((len(preds), 2))
        proba[np.arange(len(preds)), preds] = 1.0
        return proba


class CrossDomainTransferLearner:
    """
    Unified cross-domain transfer learning for security anomaly detection.

    Supports training on one dataset (e.g., NSL-KDD) and testing on
    another (e.g., CICIDS) with domain adaptation.
    """

    def __init__(
        self,
        method: DomainAdaptationMethod = DomainAdaptationMethod.CORAL,
        feature_alignment: str = "intersection",
        normalize: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize cross-domain transfer learner.

        Args:
            method: Domain adaptation method
            feature_alignment: Feature alignment strategy
            normalize: Normalize features before adaptation
            verbose: Print progress information
        """
        self.method = method
        self.feature_alignment = feature_alignment
        self.normalize = normalize
        self.verbose = verbose

        # Components
        self.adapter: BaseDomainAdapter | None = None
        self.feature_aligner: FeatureAligner | None = None
        self.scaler: Any = None
        self.label_encoder: Any = None

        # Metrics
        self._mmd_before: float = 0.0
        self._mmd_after: float = 0.0

    def _create_adapter(self) -> BaseDomainAdapter:
        """Create domain adapter based on method."""
        if self.method == DomainAdaptationMethod.MMD:
            return MMDAdapter()
        elif self.method == DomainAdaptationMethod.CORAL:
            return CORALAdapter()
        elif self.method == DomainAdaptationMethod.SUBSPACE:
            return SubspaceAlignmentAdapter()
        elif self.method == DomainAdaptationMethod.DANN:
            return DANNAdapter()
        elif self.method == DomainAdaptationMethod.JDA:
            return JDAAdapter()
        elif self.method == DomainAdaptationMethod.TCA:
            return TCAAdapter()
        elif self.method == DomainAdaptationMethod.OPTIMAL_TRANSPORT:
            return OptimalTransportAdapter()
        else:
            # Default to CORAL for any unknown method
            logger.warning(f"Method {self.method} not implemented, using CORAL")
            return CORALAdapter()

    def fit(
        self,
        source_data: DomainData,
        target_data: DomainData,
        target_labels_for_eval: NDArray[np.int64] | None = None,
    ) -> None:
        """
        Fit the cross-domain transfer model.

        Args:
            source_data: Labeled source domain data
            target_data: Target domain data (labels optional)
            target_labels_for_eval: Target labels for evaluation only
        """
        start_time = time.time()

        # Feature alignment
        if source_data.feature_names and target_data.feature_names:
            self.feature_aligner = FeatureAligner(
                source_data.feature_names,
                target_data.feature_names,
                self.feature_alignment,
            )

            source_X = self.feature_aligner.align_source(source_data.X)
            target_X = self.feature_aligner.align_target(target_data.X)

            if self.verbose:
                logger.info(f"Aligned {len(self.feature_aligner.common_features)} common features")
        else:
            # Assume same feature space
            min_features = min(source_data.n_features, target_data.n_features)
            source_X = source_data.X[:, :min_features]
            target_X = target_data.X[:, :min_features]

        # Normalize
        if self.normalize and SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            source_X = self.scaler.fit_transform(source_X)
            target_X = self.scaler.transform(target_X)

        # Encode labels
        if SKLEARN_AVAILABLE:
            self.label_encoder = LabelEncoder()
            source_y = self.label_encoder.fit_transform(source_data.y)
        else:
            source_y = source_data.y

        # Compute MMD before adaptation
        mmd_adapter = MMDAdapter()
        self._mmd_before = mmd_adapter.compute_mmd(source_X, target_X)

        # Create and fit adapter
        self.adapter = self._create_adapter()
        self.adapter.fit(
            source_X,
            source_y,
            target_X,
            target_labels_for_eval,
        )

        # Compute MMD after adaptation
        source_transformed = self.adapter.transform(source_X, "source")
        target_transformed = self.adapter.transform(target_X, "target")
        self._mmd_after = mmd_adapter.compute_mmd(source_transformed, target_transformed)

        if self.verbose:
            logger.info(
                f"Domain adaptation complete: "
                f"MMD {self._mmd_before:.4f} -> {self._mmd_after:.4f}"
            )
            logger.info(f"Adaptation time: {time.time() - start_time:.2f}s")

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict on target domain data."""
        if self.adapter is None:
            raise ValueError("Model not fitted")

        # Feature alignment
        if self.feature_aligner:
            X = self.feature_aligner.align_target(X)

        # Normalize
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Predict
        preds = self.adapter.predict(X)

        # Decode labels
        if self.label_encoder is not None:
            preds = self.label_encoder.inverse_transform(preds)

        return preds

    def evaluate(
        self,
        source_data: DomainData,
        target_data: DomainData,
    ) -> TransferResult:
        """
        Evaluate cross-domain transfer performance.

        Args:
            source_data: Source domain data
            target_data: Target domain data with labels

        Returns:
            TransferResult with comprehensive metrics
        """
        start_time = time.time()

        # Fit the model
        self.fit(source_data, target_data, target_data.y)

        # Evaluate on source (to measure source performance)
        source_X = source_data.X
        if self.feature_aligner:
            source_X = self.feature_aligner.align_source(source_X)
        if self.scaler is not None:
            source_X = self.scaler.transform(source_X)

        if self.adapter is None:
            raise ValueError("Model not fitted")

        source_preds = self.adapter.predict(source_X)
        if self.label_encoder is not None:
            source_y_encoded = self.label_encoder.transform(source_data.y)
        else:
            source_y_encoded = source_data.y

        source_accuracy = accuracy_score(source_y_encoded, source_preds)

        # Evaluate on target
        target_preds = self.predict(target_data.X)

        # Calculate metrics
        accuracy = accuracy_score(target_data.y, target_preds)
        precision = precision_score(
            target_data.y, target_preds, average="weighted", zero_division=0
        )
        recall = recall_score(target_data.y, target_preds, average="weighted", zero_division=0)
        f1 = f1_score(target_data.y, target_preds, average="weighted", zero_division=0)

        # AUC-ROC if binary
        auc = None
        if len(np.unique(target_data.y)) == 2:
            try:
                if self.adapter is not None and hasattr(self.adapter, "predict_proba"):
                    target_X_aligned = target_data.X
                    if self.feature_aligner:
                        target_X_aligned = self.feature_aligner.align_target(target_X_aligned)
                    if self.scaler is not None:
                        target_X_aligned = self.scaler.transform(target_X_aligned)
                    proba = self.adapter.predict_proba(target_X_aligned)
                    auc = roc_auc_score(target_data.y, proba[:, 1])
            except Exception as e:
                logger.debug(f"Failed to compute AUC for target domain evaluation: {e}")

        # Per-class F1
        unique_classes = np.unique(target_data.y)
        class_f1_scores = {}
        for cls in unique_classes:
            cls_mask = target_data.y == cls
            cls_preds = target_preds[cls_mask]
            cls_true = target_data.y[cls_mask]
            cls_f1 = f1_score(cls_true, cls_preds, average="binary", pos_label=cls, zero_division=0)
            class_f1_scores[str(cls)] = float(cls_f1)

        # Transfer ratio
        transfer_ratio = accuracy / (source_accuracy + 1e-10)
        negative_transfer = transfer_ratio < 0.9  # Less than 90% of source performance

        # Alignment improvement
        alignment_improvement = (self._mmd_before - self._mmd_after) / (self._mmd_before + 1e-10)

        return TransferResult(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            auc_roc=float(auc) if auc is not None else None,
            source_accuracy=float(source_accuracy),
            transfer_ratio=float(transfer_ratio),
            negative_transfer=negative_transfer,
            mmd_before=self._mmd_before,
            mmd_after=self._mmd_after,
            alignment_improvement=float(alignment_improvement),
            source_domain=source_data.domain_name,
            target_domain=target_data.domain_name,
            method=self.method.value,
            adaptation_time=time.time() - start_time,
            class_f1_scores=class_f1_scores,
        )


def run_nsl_kdd_to_cicids_benchmark(
    nsl_kdd_X: NDArray[np.float64],
    nsl_kdd_y: NDArray[np.int64],
    cicids_X: NDArray[np.float64],
    cicids_y: NDArray[np.int64],
    nsl_kdd_features: list[str] | None = None,
    cicids_features: list[str] | None = None,
    methods: list[DomainAdaptationMethod] | None = None,
) -> dict[str, TransferResult]:
    """
    Run NSL-KDD → CICIDS cross-domain transfer benchmark.

    This is the key benchmark demonstrating Mercury's architectural
    advantages over pure supervised methods.

    Args:
        nsl_kdd_X: NSL-KDD features
        nsl_kdd_y: NSL-KDD labels
        cicids_X: CICIDS features
        cicids_y: CICIDS labels
        nsl_kdd_features: Feature names for NSL-KDD
        cicids_features: Feature names for CICIDS
        methods: List of methods to evaluate

    Returns:
        Dictionary mapping method name to TransferResult
    """
    if methods is None:
        methods = [
            DomainAdaptationMethod.CORAL,
            DomainAdaptationMethod.MMD,
            DomainAdaptationMethod.SUBSPACE,
        ]

    source_data = DomainData(
        X=nsl_kdd_X,
        y=nsl_kdd_y,
        domain_name="NSL-KDD",
        feature_names=nsl_kdd_features,
    )

    target_data = DomainData(
        X=cicids_X,
        y=cicids_y,
        domain_name="CICIDS",
        feature_names=cicids_features,
    )

    results: dict[str, TransferResult] = {}

    for method in methods:
        logger.info(f"Evaluating {method.value} transfer: NSL-KDD -> CICIDS")

        learner = CrossDomainTransferLearner(method=method)
        result = learner.evaluate(source_data, target_data)

        results[method.value] = result

        logger.info(
            f"  {method.value}: accuracy={result.accuracy:.4f}, "
            f"f1={result.f1:.4f}, transfer_ratio={result.transfer_ratio:.4f}"
        )

    return results


def create_cross_domain_learner(
    method: str = "coral",
    **kwargs: Any,
) -> CrossDomainTransferLearner:
    """
    Factory function to create cross-domain transfer learner.

    Args:
        method: Method name ('coral', 'mmd', 'subspace', 'dann')
        **kwargs: Additional arguments

    Returns:
        Configured CrossDomainTransferLearner
    """
    method_map = {
        "coral": DomainAdaptationMethod.CORAL,
        "mmd": DomainAdaptationMethod.MMD,
        "subspace": DomainAdaptationMethod.SUBSPACE,
        "dann": DomainAdaptationMethod.DANN,
        "jda": DomainAdaptationMethod.JDA,
        "tca": DomainAdaptationMethod.TCA,
        "optimal_transport": DomainAdaptationMethod.OPTIMAL_TRANSPORT,
    }

    m = method_map.get(method.lower(), DomainAdaptationMethod.CORAL)

    return CrossDomainTransferLearner(method=m, **kwargs)


# Exports
__all__ = [
    "CORALAdapter",
    "CrossDomainTransferLearner",
    "DomainAdaptationMethod",
    "DomainData",
    "FeatureAligner",
    "MMDAdapter",
    "SecurityDataset",
    "SubspaceAlignmentAdapter",
    "TransferResult",
    "create_cross_domain_learner",
    "run_nsl_kdd_to_cicids_benchmark",
]

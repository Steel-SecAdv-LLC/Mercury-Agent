"""
Mercury Agent - Topological Data Analysis for Anomaly Detection
Copyright (C) 2025 Steel Security Advisors LLC

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

Implements persistent homology for topological feature extraction:
- Vietoris-Rips filtration on point cloud data
- 0D and 1D persistent homology via union-find
- Persistence diagrams, entropy, and landscape features
- Wasserstein and bottleneck distances for diagram comparison
- TopologicalAnomalyDetector for TDA-based anomaly detection

Reference: Edelsbrunner & Harer (2010) "Computational Topology"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

__all__ = [
    "PersistenceDiagram",
    "TopologicalAnomalyDetector",
    "VietorisRipsFiltration",
    "bottleneck_distance",
    "wasserstein_distance_pd",
]


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set) data structure for connected-component tracking
# ---------------------------------------------------------------------------


class _UnionFind:
    """Weighted union-find with path compression.

    Tracks connected components during filtration.  Each element starts in
    its own component; ``union`` merges two components and ``find`` returns
    the canonical representative with full path compression.

    Attributes:
        parent: Mapping from element to its parent.
        rank: Mapping from root element to tree rank (for union by rank).
        birth: Mapping from root element to its birth time (filtration value).
    """

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}
        self.birth: dict[int, float] = {}

    def make_set(self, x: int, birth_time: float = 0.0) -> None:
        """Create a new singleton set for element *x*.

        Args:
            x: Element identifier.
            birth_time: Filtration value at which *x* appears.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.birth[x] = birth_time

    def find(self, x: int) -> int:
        """Return the root representative of the set containing *x*.

        Applies full path compression so subsequent queries are O(1)
        amortised.

        Args:
            x: Element to look up.

        Returns:
            Root representative of the component.
        """
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, x: int, y: int) -> tuple[int, int] | None:
        """Merge the sets containing *x* and *y*.

        Uses union-by-rank.  The component with the *earlier* birth time
        (smaller filtration value) survives; the other component dies.

        Args:
            x: First element.
            y: Second element.

        Returns:
            ``(survivor_root, dying_root)`` if a merge happened, or
            ``None`` if *x* and *y* already belong to the same set.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return None

        # The component born earlier survives (elder rule).
        if self.birth[rx] > self.birth[ry]:
            rx, ry = ry, rx

        # Union by rank - attach smaller tree under larger.
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
            # Re-check elder rule after rank swap: elder must survive.
            if self.birth[rx] > self.birth[ry]:
                rx, ry = ry, rx

        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

        return rx, ry


# ---------------------------------------------------------------------------
# Persistence diagram dataclass
# ---------------------------------------------------------------------------


@dataclass
class PersistenceDiagram:
    """Container for persistence diagram data.

    A persistence diagram stores (birth, death) pairs for topological
    features detected during a filtration.  Dimension-0 pairs correspond to
    connected components; dimension-1 pairs correspond to loops / cycles.

    Attributes:
        pairs_dim0: Array of shape ``(k, 2)`` with birth/death pairs for
            connected components (H0).  A pair ``(b, inf)`` means a
            component born at filtration value *b* that never dies.
        pairs_dim1: Array of shape ``(m, 2)`` with birth/death pairs for
            1-dimensional cycles (H1).
        filtration_max: Maximum filtration value used.  Infinite-death
            features are clamped to this value for numerical computation.
    """

    pairs_dim0: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    pairs_dim1: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    filtration_max: float = 0.0

    # -- convenience helpers -------------------------------------------------

    def lifetimes(self, dim: int = 0) -> np.ndarray:
        """Return the persistence (death - birth) for each pair in *dim*.

        Args:
            dim: Homology dimension (0 or 1).

        Returns:
            1-D array of non-negative lifetimes.
        """
        pairs = self.pairs_dim0 if dim == 0 else self.pairs_dim1
        if pairs.size == 0:
            return np.array([], dtype=np.float64)
        deaths = np.where(np.isinf(pairs[:, 1]), self.filtration_max, pairs[:, 1])
        return deaths - pairs[:, 0]

    def betti_at(self, epsilon: float, dim: int = 0) -> int:
        """Compute the Betti number at filtration value *epsilon*.

        The Betti number counts how many features are *alive* at a given
        scale: born at or before *epsilon* and dying strictly after it.

        Args:
            epsilon: Filtration scale parameter.
            dim: Homology dimension (0 or 1).

        Returns:
            Non-negative integer Betti number.
        """
        pairs = self.pairs_dim0 if dim == 0 else self.pairs_dim1
        if pairs.size == 0:
            return 0
        alive = (pairs[:, 0] <= epsilon) & ((pairs[:, 1] > epsilon) | np.isinf(pairs[:, 1]))
        return int(np.sum(alive))


# ---------------------------------------------------------------------------
# Vietoris-Rips filtration
# ---------------------------------------------------------------------------


class VietorisRipsFiltration:
    """Build a Vietoris-Rips simplicial complex and compute persistent homology.

    Given a point cloud of *n* points in R^d, computes the pairwise distance
    matrix and constructs a filtered Vietoris-Rips complex.  Persistent
    homology in dimensions 0 (connected components) and 1 (cycles) is then
    extracted via a union-find algorithm for H0 and an edge-triangle
    reduction for H1.

    Args:
        max_edge_length: Maximum filtration radius.  Edges longer than this
            are discarded.  ``None`` means use the maximum pairwise distance.
        max_dimension: Maximum homology dimension to compute (0 or 1).
        metric: Distance metric passed to :func:`scipy.spatial.distance.pdist`.

    Example::

        filtration = VietorisRipsFiltration(max_edge_length=2.0)
        diagram = filtration.build(point_cloud)
    """

    def __init__(
        self,
        max_edge_length: float | None = None,
        max_dimension: int = 1,
        metric: str = "euclidean",
    ) -> None:
        self.max_edge_length = max_edge_length
        self.max_dimension = min(max_dimension, 1)  # cap at H1
        self.metric = metric
        self._distance_matrix: np.ndarray | None = None

    # -- public API ----------------------------------------------------------

    def build(self, point_cloud: np.ndarray) -> PersistenceDiagram:
        """Compute the persistence diagram for *point_cloud*.

        Args:
            point_cloud: Array of shape ``(n_points, n_features)``.

        Returns:
            :class:`PersistenceDiagram` with H0 and (optionally) H1 pairs.
        """
        point_cloud = np.asarray(point_cloud, dtype=np.float64)

        # -- Edge cases ------------------------------------------------------
        if point_cloud.ndim == 1:
            point_cloud = point_cloud.reshape(-1, 1)

        n_points = point_cloud.shape[0]

        if n_points == 0:
            logger.debug("Empty point cloud; returning trivial diagram.")
            return PersistenceDiagram()

        if n_points == 1:
            logger.debug("Single point; returning one infinite H0 feature.")
            return PersistenceDiagram(
                pairs_dim0=np.array([[0.0, np.inf]]),
                filtration_max=0.0,
            )

        # -- Distance matrix -------------------------------------------------
        condensed = pdist(point_cloud, metric=self.metric)
        self._distance_matrix = squareform(condensed)

        filt_max = float(np.max(condensed)) if condensed.size > 0 else 0.0
        if self.max_edge_length is not None:
            filt_max = min(filt_max, self.max_edge_length)

        # -- Build sorted edge list ------------------------------------------
        edges = self._sorted_edges(n_points, filt_max)

        # -- H0 via union-find ----------------------------------------------
        pairs_dim0 = self._compute_h0(n_points, edges, filt_max)

        # -- H1 via edge-triangle reduction ----------------------------------
        pairs_dim1: np.ndarray
        if self.max_dimension >= 1:
            pairs_dim1 = self._compute_h1(n_points, edges, filt_max)
        else:
            pairs_dim1 = np.empty((0, 2))

        diagram = PersistenceDiagram(
            pairs_dim0=pairs_dim0,
            pairs_dim1=pairs_dim1,
            filtration_max=filt_max,
        )
        logger.debug(
            "VR filtration: n=%d, H0 pairs=%d, H1 pairs=%d",
            n_points,
            len(pairs_dim0),
            len(pairs_dim1),
        )
        return diagram

    @property
    def distance_matrix(self) -> np.ndarray | None:
        """Return the last computed pairwise distance matrix, or ``None``."""
        return self._distance_matrix

    # -- internal helpers ----------------------------------------------------

    def _sorted_edges(self, n_points: int, filt_max: float) -> list[tuple[float, int, int]]:
        """Return edges ``(weight, i, j)`` sorted by ascending weight.

        Args:
            n_points: Number of vertices.
            filt_max: Maximum edge length to include.

        Returns:
            Sorted list of ``(distance, vertex_i, vertex_j)`` tuples.
        """
        assert self._distance_matrix is not None
        dm = self._distance_matrix
        edges: list[tuple[float, int, int]] = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                w = dm[i, j]
                if w <= filt_max:
                    edges.append((w, i, j))
        edges.sort(key=lambda e: e[0])
        return edges

    def _compute_h0(
        self,
        n_points: int,
        edges: list[tuple[float, int, int]],
        filt_max: float,
    ) -> np.ndarray:
        """Compute 0-dimensional persistent homology (connected components).

        Every vertex is born at filtration value 0.  When an edge merges two
        components the younger component dies.  The last surviving component
        has infinite death time.

        Args:
            n_points: Number of vertices.
            edges: Sorted edge list from :meth:`_sorted_edges`.
            filt_max: Maximum filtration value.

        Returns:
            Array of shape ``(n_pairs, 2)`` with (birth, death) rows.
        """
        uf = _UnionFind()
        for i in range(n_points):
            uf.make_set(i, birth_time=0.0)

        pairs: list[tuple[float, float]] = []
        for w, u, v in edges:
            result = uf.union(u, v)
            if result is not None:
                _survivor, dying = result
                birth = uf.birth.get(dying, 0.0)
                pairs.append((birth, w))

        # Count surviving components (those that never merged away)
        roots = {uf.find(i) for i in range(n_points)}
        for root in roots:
            pairs.append((uf.birth[root], np.inf))

        if not pairs:
            return np.empty((0, 2))
        return np.array(pairs, dtype=np.float64)

    def _compute_h1(
        self,
        n_points: int,
        edges: list[tuple[float, int, int]],
        filt_max: float,
    ) -> np.ndarray:
        """Compute 1-dimensional persistent homology (cycles / loops).

        Uses an incremental edge-triangle approach: for each edge ``(u, v)``
        that does *not* merge two components (i.e. both endpoints already
        connected), check whether a triangle closes through a common
        neighbour.  If so, the cycle is born at the edge weight and dies
        at the weight of the heaviest triangle edge.

        This is a lightweight heuristic suitable for small-to-medium point
        clouds.  For large-scale problems a full boundary-matrix reduction
        would be needed.

        Args:
            n_points: Number of vertices.
            edges: Sorted edge list.
            filt_max: Maximum filtration value.

        Returns:
            Array of shape ``(m, 2)`` with (birth, death) rows for H1.
        """
        # Rebuild adjacency with edge weights for triangle detection
        adjacency: dict[int, dict[int, float]] = {i: {} for i in range(n_points)}
        uf = _UnionFind()
        for i in range(n_points):
            uf.make_set(i, birth_time=0.0)

        pairs: list[tuple[float, float]] = []

        for w, u, v in edges:
            # Record edge in adjacency
            adjacency[u][v] = w
            adjacency[v][u] = w

            merge_result = uf.union(u, v)
            if merge_result is not None:
                # Edge merges components -- this is an H0 event, not H1.
                continue

            # Edge connects already-connected vertices: potential cycle.
            # Look for shortest closing triangle through a common neighbour.
            common_neighbours = set(adjacency[u].keys()) & set(adjacency[v].keys())
            if not common_neighbours:
                # No triangle closes yet; record a long-lived cycle.
                pairs.append((w, filt_max))
                continue

            # The cycle dies when the triangle is fully formed.  The death
            # time is the maximum of the three edge weights of the triangle.
            best_death = filt_max
            for nb in common_neighbours:
                triangle_max = max(w, adjacency[u][nb], adjacency[v][nb])
                best_death = min(best_death, triangle_max)

            if best_death > w:
                pairs.append((w, best_death))
            # If best_death == w the cycle is born and immediately killed;
            # we omit zero-persistence pairs.

        if not pairs:
            return np.empty((0, 2))
        return np.array(pairs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Diagram distance functions
# ---------------------------------------------------------------------------


def bottleneck_distance(
    dgm_a: PersistenceDiagram,
    dgm_b: PersistenceDiagram,
    dim: int = 0,
) -> float:
    """Approximate bottleneck distance between two persistence diagrams.

    The bottleneck distance is the infimum over all bijections between
    diagram points (including diagonal projections) of the maximum L-inf
    cost.  This implementation uses a greedy nearest-neighbour matching
    which provides an upper bound and is exact for many practical cases.

    Args:
        dgm_a: First persistence diagram.
        dgm_b: Second persistence diagram.
        dim: Homology dimension to compare (0 or 1).

    Returns:
        Non-negative bottleneck distance approximation.
    """
    pts_a = _diagram_points(dgm_a, dim)
    pts_b = _diagram_points(dgm_b, dim)

    if pts_a.size == 0 and pts_b.size == 0:
        return 0.0

    # Augment both sets with diagonal projections of the other set
    aug_a = _augment_with_diagonal(pts_a, pts_b)
    aug_b = _augment_with_diagonal(pts_b, pts_a)

    # Greedy nearest-neighbour matching in L-inf
    n = max(len(aug_a), len(aug_b))
    # Pad shorter set with diagonal points at the origin
    while len(aug_a) < n:
        aug_a = np.vstack([aug_a, [0.0, 0.0]])
    while len(aug_b) < n:
        aug_b = np.vstack([aug_b, [0.0, 0.0]])

    # Compute pairwise L-inf costs
    cost_matrix = np.max(np.abs(aug_a[:, np.newaxis, :] - aug_b[np.newaxis, :, :]), axis=2)

    # Greedy assignment: pick the minimum-cost pairing row by row.
    used_cols: set[int] = set()
    max_cost = 0.0
    for i in range(n):
        best_j = -1
        best_c = np.inf
        for j in range(n):
            if j not in used_cols and cost_matrix[i, j] < best_c:
                best_c = cost_matrix[i, j]
                best_j = j
        if best_j >= 0:
            used_cols.add(best_j)
            max_cost = max(max_cost, best_c)

    return float(max_cost)


def wasserstein_distance_pd(
    dgm_a: PersistenceDiagram,
    dgm_b: PersistenceDiagram,
    dim: int = 0,
    p: float = 2.0,
) -> float:
    """Compute the p-Wasserstein distance between two persistence diagrams.

    Uses a greedy nearest-neighbour assignment as an efficient
    approximation.  Diagonal projections are included so that diagrams of
    different cardinality can be compared.

    .. math::

        W_p(D_1, D_2) = \\left( \\inf_{\\gamma} \\sum_{x} \\|x - \\gamma(x)\\|_\\infty^p \\right)^{1/p}

    Args:
        dgm_a: First persistence diagram.
        dgm_b: Second persistence diagram.
        dim: Homology dimension to compare (0 or 1).
        p: Wasserstein exponent (default 2).

    Returns:
        Non-negative Wasserstein distance.
    """
    if p <= 0:
        raise ValueError("Wasserstein exponent p must be positive.")

    pts_a = _diagram_points(dgm_a, dim)
    pts_b = _diagram_points(dgm_b, dim)

    if pts_a.size == 0 and pts_b.size == 0:
        return 0.0

    aug_a = _augment_with_diagonal(pts_a, pts_b)
    aug_b = _augment_with_diagonal(pts_b, pts_a)

    n = max(len(aug_a), len(aug_b))
    while len(aug_a) < n:
        aug_a = np.vstack([aug_a, [0.0, 0.0]])
    while len(aug_b) < n:
        aug_b = np.vstack([aug_b, [0.0, 0.0]])

    cost_matrix = np.max(np.abs(aug_a[:, np.newaxis, :] - aug_b[np.newaxis, :, :]), axis=2)

    # Greedy assignment
    used_cols: set[int] = set()
    total_cost = 0.0
    for i in range(n):
        best_j = -1
        best_c = np.inf
        for j in range(n):
            if j not in used_cols and cost_matrix[i, j] < best_c:
                best_c = cost_matrix[i, j]
                best_j = j
        if best_j >= 0:
            used_cols.add(best_j)
            total_cost += best_c**p

    return float(total_cost ** (1.0 / p))


# -- helpers for diagram distances -------------------------------------------


def _diagram_points(dgm: PersistenceDiagram, dim: int) -> np.ndarray:
    """Extract finite (birth, death) points from a diagram.

    Infinite death values are clamped to ``dgm.filtration_max``.

    Args:
        dgm: Source persistence diagram.
        dim: Homology dimension.

    Returns:
        Array of shape ``(k, 2)``.
    """
    pairs = dgm.pairs_dim0 if dim == 0 else dgm.pairs_dim1
    if pairs.size == 0:
        return np.empty((0, 2))
    pts = pairs.copy()
    inf_mask = np.isinf(pts[:, 1])
    pts[inf_mask, 1] = dgm.filtration_max
    return pts


def _augment_with_diagonal(pts: np.ndarray, other_pts: np.ndarray) -> np.ndarray:
    """Augment *pts* with diagonal projections of *other_pts*.

    For each point ``(b, d)`` in *other_pts* the closest point on the
    diagonal is ``((b+d)/2, (b+d)/2)``.

    Args:
        pts: Original diagram points of shape ``(k, 2)``.
        other_pts: Points from the opposing diagram.

    Returns:
        Concatenation of *pts* and the diagonal projections.
    """
    if other_pts.size == 0:
        return pts if pts.size > 0 else np.empty((0, 2))
    mid = (other_pts[:, 0] + other_pts[:, 1]) / 2.0
    diag_pts = np.column_stack([mid, mid])
    if pts.size == 0:
        return diag_pts
    return np.vstack([pts, diag_pts])


# ---------------------------------------------------------------------------
# Topological anomaly detector
# ---------------------------------------------------------------------------


class TopologicalAnomalyDetector:
    """TDA-based anomaly detector using persistent homology features.

    Builds a topological profile of a *reference* (normal) dataset and
    then scores new windows / batches against that reference.  Anomaly
    scores are derived from:

    * **Betti-number deviation** -- shift in the number of connected
      components (b0) or loops (b1) relative to the reference.
    * **Persistence entropy** -- information-theoretic complexity of the
      persistence diagram.
    * **Landscape norm** -- L2 norm of the persistence landscape (a stable
      functional summary of the diagram).
    * **Wasserstein distance** -- optimal-transport distance between the
      test diagram and the reference diagram.

    Args:
        max_edge_length: Maximum filtration radius forwarded to
            :class:`VietorisRipsFiltration`.
        metric: Pairwise distance metric (default ``"euclidean"``).
        n_reference_samples: Number of sub-samples to draw when building
            the reference profile (``None`` = use all data).
        anomaly_threshold: Score above which a window is flagged as
            anomalous.  ``None`` means auto-calibrate from reference.
        seed: Random seed for reproducibility.

    Example::

        detector = TopologicalAnomalyDetector(max_edge_length=3.0)
        detector.fit(reference_data)
        result = detector.score(test_window)
    """

    def __init__(
        self,
        max_edge_length: float | None = None,
        metric: str = "euclidean",
        n_reference_samples: int | None = None,
        anomaly_threshold: float | None = None,
        seed: int = 42,
    ) -> None:
        self.max_edge_length = max_edge_length
        self.metric = metric
        self.n_reference_samples = n_reference_samples
        self.anomaly_threshold = anomaly_threshold
        self.seed = seed

        self._filtration = VietorisRipsFiltration(
            max_edge_length=max_edge_length,
            max_dimension=1,
            metric=metric,
        )
        self._reference_diagram: PersistenceDiagram | None = None
        self._reference_features: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
        self._fitted = False

    # -- public API ----------------------------------------------------------

    def fit(self, reference_data: np.ndarray) -> TopologicalAnomalyDetector:
        """Build the reference topological profile.

        Args:
            reference_data: Array of shape ``(n_samples, n_features)``
                representing normal / in-distribution data.

        Returns:
            Self for method chaining.
        """
        reference_data = np.asarray(reference_data, dtype=np.float64)
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)

        if reference_data.shape[0] == 0:
            logger.warning("Empty reference data; detector will not be fitted.")
            return self

        rng = np.random.RandomState(self.seed)
        data = reference_data
        if (
            self.n_reference_samples is not None
            and reference_data.shape[0] > self.n_reference_samples
        ):
            idx = rng.choice(
                reference_data.shape[0],
                size=self.n_reference_samples,
                replace=False,
            )
            data = reference_data[idx]

        self._reference_diagram = self._filtration.build(data)
        self._reference_features = self._extract_features(self._reference_diagram)

        # Auto-calibrate threshold from reference feature variability
        if self.anomaly_threshold is None:
            self._calibrate_threshold(reference_data, rng)

        self._fitted = True
        logger.info(
            "TopologicalAnomalyDetector fitted on %d samples (%d features).",
            data.shape[0],
            data.shape[1],
        )
        return self

    def score(self, test_data: np.ndarray) -> dict[str, Any]:
        """Score a test window / batch against the reference profile.

        Args:
            test_data: Array of shape ``(n_samples, n_features)``.

        Returns:
            Dictionary with keys:

            - ``anomaly_score``: Aggregate anomaly score in [0, 1].
            - ``is_anomaly``: Boolean flag based on threshold.
            - ``betti_0``, ``betti_1``: Betti numbers of the test diagram.
            - ``persistence_entropy_h0``, ``persistence_entropy_h1``:
              Entropy values.
            - ``landscape_norm_h0``, ``landscape_norm_h1``: Landscape L2
              norms.
            - ``wasserstein_h0``, ``wasserstein_h1``: Wasserstein distances
              to the reference diagram.
            - ``features``: Full feature dictionary.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self._reference_diagram is None:
            raise RuntimeError("Must call fit() before score().")

        test_data = np.asarray(test_data, dtype=np.float64)
        if test_data.ndim == 1:
            test_data = test_data.reshape(-1, 1)

        if test_data.shape[0] == 0:
            return self._empty_score()

        test_diagram = self._filtration.build(test_data)
        test_features = self._extract_features(test_diagram)

        # Compute Wasserstein distances to reference
        w_h0 = wasserstein_distance_pd(self._reference_diagram, test_diagram, dim=0)
        w_h1 = wasserstein_distance_pd(self._reference_diagram, test_diagram, dim=1)
        test_features["wasserstein_h0"] = w_h0
        test_features["wasserstein_h1"] = w_h1

        # Aggregate anomaly score: normalised feature deviation
        anomaly_score = self._aggregate_score(test_features)

        threshold = self.anomaly_threshold if self.anomaly_threshold is not None else 0.5

        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(anomaly_score > threshold),
            "threshold": float(threshold),
            "betti_0": test_features.get("betti_0", 0),
            "betti_1": test_features.get("betti_1", 0),
            "persistence_entropy_h0": test_features.get("entropy_h0", 0.0),
            "persistence_entropy_h1": test_features.get("entropy_h1", 0.0),
            "landscape_norm_h0": test_features.get("landscape_norm_h0", 0.0),
            "landscape_norm_h1": test_features.get("landscape_norm_h1", 0.0),
            "wasserstein_h0": w_h0,
            "wasserstein_h1": w_h1,
            "features": test_features,
            "method": "topological_persistent_homology",
        }

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """Return binary anomaly predictions for each row treated as a batch.

        Convenience wrapper compatible with the Mercury Agent detector
        pattern.  Each call scores the full ``test_data`` array as a
        single topological window.

        Args:
            test_data: Array of shape ``(n_samples, n_features)``.

        Returns:
            1-D integer array with 1 for anomaly, 0 for normal.  All
            rows receive the same label (topological analysis is a
            *set*-level, not point-level, method).
        """
        result = self.score(test_data)
        n = test_data.shape[0] if test_data.ndim > 1 else len(test_data)
        label = 1 if result["is_anomaly"] else 0
        return np.full(n, label, dtype=int)

    # -- feature extraction --------------------------------------------------

    @staticmethod
    def _extract_features(diagram: PersistenceDiagram) -> dict[str, float]:
        """Derive topological features from a persistence diagram.

        Args:
            diagram: Source persistence diagram.

        Returns:
            Dictionary of named feature values.
        """
        features: dict[str, float] = {}

        # Betti numbers at the median filtration scale
        mid = diagram.filtration_max / 2.0 if diagram.filtration_max > 0 else 0.0
        features["betti_0"] = float(diagram.betti_at(mid, dim=0))
        features["betti_1"] = float(diagram.betti_at(mid, dim=1))

        # Persistence entropy
        features["entropy_h0"] = _persistence_entropy(diagram, dim=0)
        features["entropy_h1"] = _persistence_entropy(diagram, dim=1)

        # Landscape norms
        features["landscape_norm_h0"] = _landscape_norm(diagram, dim=0)
        features["landscape_norm_h1"] = _landscape_norm(diagram, dim=1)

        # Summary statistics of lifetimes
        for dim_label, dim_val in [("h0", 0), ("h1", 1)]:
            lt = diagram.lifetimes(dim_val)
            if lt.size > 0:
                features[f"lifetime_mean_{dim_label}"] = float(np.mean(lt))
                features[f"lifetime_max_{dim_label}"] = float(np.max(lt))
                features[f"lifetime_std_{dim_label}"] = float(np.std(lt))
                features[f"n_features_{dim_label}"] = float(len(lt))
            else:
                features[f"lifetime_mean_{dim_label}"] = 0.0
                features[f"lifetime_max_{dim_label}"] = 0.0
                features[f"lifetime_std_{dim_label}"] = 0.0
                features[f"n_features_{dim_label}"] = 0.0

        return features

    # -- scoring helpers -----------------------------------------------------

    def _aggregate_score(self, test_features: dict[str, float]) -> float:
        """Combine feature deviations into a single anomaly score in [0, 1].

        Uses z-score deviations from the reference features, weighted by
        inverse variance when available, then passed through a sigmoid for
        bounded output.

        Args:
            test_features: Feature dictionary for the test window.

        Returns:
            Scalar anomaly score.
        """
        if not self._reference_features:
            return 0.5

        deviations: list[float] = []
        for key in self._reference_features:
            ref_val = self._reference_features[key]
            test_val = test_features.get(key, 0.0)
            std = self._feature_stds.get(key, 0.0)
            if std < 1e-6:
                # Feature had near-zero variance in reference sub-samples.
                # Use absolute difference scaled by reference magnitude + 1
                # to avoid division by near-zero.
                scale = abs(ref_val) + 1.0
                deviations.append(abs(test_val - ref_val) / scale)
            else:
                deviations.append(abs(test_val - ref_val) / std)

        if not deviations:
            return 0.0

        mean_dev = float(np.mean(deviations))
        # Sigmoid mapping: 0 deviation -> 0.0, large deviation -> ~1.0
        # Scaled so that a 3-sigma deviation maps to roughly 0.75.
        score = 2.0 / (1.0 + np.exp(-0.5 * mean_dev)) - 1.0
        return float(np.clip(score, 0.0, 1.0))

    def _calibrate_threshold(self, reference_data: np.ndarray, rng: np.random.RandomState) -> None:
        """Auto-calibrate the anomaly threshold from reference data.

        Splits the reference into overlapping windows, scores each, and
        sets the threshold at the 95th percentile of reference scores.

        Args:
            reference_data: Full reference dataset.
            rng: Random state for reproducibility.
        """
        n = reference_data.shape[0]
        if n < 4:
            self.anomaly_threshold = 0.5
            self._feature_stds = dict.fromkeys(self._reference_features, 1.0)
            return

        # Compute features on several random sub-samples
        n_trials = min(10, n // 2)
        window_size = max(3, n // 3)
        all_features: dict[str, list[float]] = {k: [] for k in self._reference_features}

        for _ in range(n_trials):
            idx = rng.choice(n, size=min(window_size, n), replace=False)
            sub_data = reference_data[idx]
            dgm = self._filtration.build(sub_data)
            feats = self._extract_features(dgm)
            for k in all_features:
                all_features[k].append(feats.get(k, 0.0))

        # Record feature standard deviations for z-scoring
        for k, vals in all_features.items():
            arr = np.array(vals)
            self._feature_stds[k] = float(np.std(arr)) if len(arr) > 1 else 1.0

        # Score each sub-sample against the reference
        scores: list[float] = []
        for trial_idx in range(n_trials):
            trial_feats = {k: all_features[k][trial_idx] for k in all_features}
            trial_feats["wasserstein_h0"] = 0.0
            trial_feats["wasserstein_h1"] = 0.0
            scores.append(self._aggregate_score(trial_feats))

        if scores:
            self.anomaly_threshold = float(np.percentile(scores, 95))
        else:
            self.anomaly_threshold = 0.5

        logger.debug("Auto-calibrated anomaly threshold: %.4f", self.anomaly_threshold)

    def _empty_score(self) -> dict[str, Any]:
        """Return a neutral score dict for empty input."""
        threshold = self.anomaly_threshold if self.anomaly_threshold is not None else 0.5
        return {
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "threshold": float(threshold),
            "betti_0": 0,
            "betti_1": 0,
            "persistence_entropy_h0": 0.0,
            "persistence_entropy_h1": 0.0,
            "landscape_norm_h0": 0.0,
            "landscape_norm_h1": 0.0,
            "wasserstein_h0": 0.0,
            "wasserstein_h1": 0.0,
            "features": {},
            "method": "topological_persistent_homology",
        }


# ---------------------------------------------------------------------------
# Topological feature utilities
# ---------------------------------------------------------------------------


def _persistence_entropy(diagram: PersistenceDiagram, dim: int = 0) -> float:
    """Compute persistence entropy of a diagram dimension.

    Persistence entropy is an information-theoretic measure of the
    complexity of a persistence diagram:

    .. math::

        H = -\\sum_i p_i \\log(p_i), \\quad p_i = \\frac{l_i}{\\sum_j l_j}

    where *l_i* is the lifetime (persistence) of the *i*-th feature.

    Args:
        diagram: Source persistence diagram.
        dim: Homology dimension.

    Returns:
        Non-negative entropy value.  Returns 0.0 for empty diagrams or
        diagrams with only zero-persistence features.
    """
    lifetimes = diagram.lifetimes(dim)
    if lifetimes.size == 0:
        return 0.0

    # Filter out zero-lifetime features
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0

    total = np.sum(lifetimes)
    if total <= 0:
        return 0.0

    probs = lifetimes / total
    # Use the convention 0 * log(0) = 0
    log_probs = np.log(probs, where=probs > 0, out=np.zeros_like(probs))
    entropy = -float(np.sum(probs * log_probs))
    return max(entropy, 0.0)


def _landscape_norm(
    diagram: PersistenceDiagram,
    dim: int = 0,
    n_grid: int = 100,
    k: int = 1,
) -> float:
    """Compute the L2 norm of the k-th persistence landscape.

    The persistence landscape is a stable vectorisation of a persistence
    diagram.  The *k*-th landscape function at parameter *t* is the *k*-th
    largest value of the tent functions centred at each diagram point.

    Args:
        diagram: Source persistence diagram.
        dim: Homology dimension.
        n_grid: Number of grid points for numerical integration.
        k: Landscape layer (1-indexed; 1 = top envelope).

    Returns:
        Non-negative L2 norm of the landscape function.
    """
    pairs = diagram.pairs_dim0 if dim == 0 else diagram.pairs_dim1
    if pairs.size == 0:
        return 0.0

    # Clamp infinite deaths
    pts = pairs.copy()
    inf_mask = np.isinf(pts[:, 1])
    pts[inf_mask, 1] = diagram.filtration_max

    births = pts[:, 0]
    deaths = pts[:, 1]

    t_min = float(np.min(births))
    t_max = float(np.max(deaths))
    if t_max <= t_min:
        return 0.0

    grid = np.linspace(t_min, t_max, n_grid)
    landscape_values = np.zeros(n_grid)

    for idx, t in enumerate(grid):
        # Tent function for each pair: min(t - b, d - t), clipped at 0
        tent_vals = np.minimum(t - births, deaths - t)
        tent_vals = np.maximum(tent_vals, 0.0)

        # k-th largest value (k=1 is the maximum)
        if len(tent_vals) >= k:
            # Partial sort for the k-th largest
            partitioned = np.partition(tent_vals, -min(k, len(tent_vals)))
            landscape_values[idx] = partitioned[-k]
        else:
            landscape_values[idx] = 0.0

    # L2 norm via trapezoidal integration
    dt = (t_max - t_min) / max(n_grid - 1, 1)
    # np.trapezoid (numpy >=2.0) replaced the older np.trapz
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if _trapz is not None:
        integral = float(_trapz(landscape_values**2, dx=dt))
    else:
        # Manual fallback for unusual numpy builds
        integral = float(np.sum(landscape_values**2) * dt)
    l2_norm = float(np.sqrt(max(integral, 0.0)))
    return l2_norm

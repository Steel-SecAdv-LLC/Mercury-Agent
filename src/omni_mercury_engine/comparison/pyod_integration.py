# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""PyOD integration: a real, executable comparison layer against PyOD baselines.

This module runs actual PyOD detectors (not descriptions of them) so Mercury's
tier and fusion engines can be measured head-to-head against the standard
open-source competition under one protocol. It is consumed by
``benchmarks/competitive_benchmark.py`` (the ADBench head-to-head harness) and
usable directly for ad-hoc comparisons.

Design rules (transparent-by-construction):

* **Library defaults only.** Every PyOD detector is instantiated with PyOD's
  own defaults; the single exception is fixing ``random_state`` for the
  stochastic detectors so runs are reproducible. No per-dataset tuning for
  anyone -- Mercury runs its defaults, PyOD runs its defaults.
* **Identical data.** Callers pass the same ``(X_train, X_test)`` split to
  every method. This layer never re-splits, re-scales, or re-orders data, so
  whatever preprocessing/shuffling the harness applies is applied to everyone.
* **No silent drops.** A detector that raises is recorded as
  ``{"error": ...}`` in the result dict, never omitted.

PyOD is an optional dependency (``pip install mercury-agent[benchmark]``);
importing this module without PyOD installed is safe -- only building/running
detectors raises, with an actionable message.

Research sources:
- PyOD GitHub (github.com/yzhao062/pyod)
- Zhao et al. "PyOD: A Python Toolbox" (JMLR 2019)

Note: This runs PyOD as an external library, it doesn't copy PyOD code.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_BASELINES",
    "CombinationMethod",
    "PyODAlgorithm",
    "PyODComparison",
    "build_pyod_detector",
    "pyod_available",
    "pyod_version",
    "run_pyod_baselines",
]


class PyODAlgorithm(Enum):
    """PyOD algorithms available for comparison."""

    ISOLATION_FOREST = "isolation_forest"
    LOF = "local_outlier_factor"
    COPOD = "copod"
    ECOD = "ecod"
    OCSVM = "one_class_svm"
    KNN = "knn"
    HBOS = "hbos"
    AUTOENCODER = "autoencoder"
    PCA = "pca"


#: The baseline set the competitive benchmark runs by default: the standard,
#: cheap, CPU-fair PyOD detectors. Deep baselines (``AUTOENCODER``) are
#: deliberately excluded from the default set: under the defaults-only /
#: no-tuning / CPU-only budget rule they train a torch network with PyOD's
#: stock 10 epochs and no per-dataset architecture search, which measures the
#: budget cap rather than the method -- reporting that as "the deep baseline"
#: would be a strawman. They remain buildable via :func:`build_pyod_detector`
#: for callers that grant them a fair budget.
DEFAULT_BASELINES: tuple[PyODAlgorithm, ...] = (
    PyODAlgorithm.ISOLATION_FOREST,
    PyODAlgorithm.ECOD,
    PyODAlgorithm.COPOD,
    PyODAlgorithm.LOF,
    PyODAlgorithm.KNN,
    PyODAlgorithm.HBOS,
)


class CombinationMethod(Enum):
    """Ensemble combination methods from PyOD."""

    AVERAGE = "average"
    MAXIMUM = "maximum"
    AOM = "average_of_maximum"
    MOA = "maximum_of_average"


def pyod_available() -> bool:
    """Return True when the optional ``pyod`` package is importable."""
    try:
        import pyod  # noqa: F401
    except ImportError:
        return False
    return True


def pyod_version() -> str | None:
    """Return the installed PyOD version string, or ``None`` if unavailable.

    Metadata-only helper: a broken/partial install can raise a non-``ImportError``
    on ``import pyod`` (binary incompatibility, C-extension load failure), and
    version stamping must never crash the benchmark. Any failure returns
    ``None`` -- the same defensive posture as ``_version()`` in
    ``benchmarks/competitive_benchmark.py``. (``pyod_available()`` deliberately
    stays narrow: a broken install should surface loudly at the gate, not be
    silently reported as "not installed".)
    """
    try:
        import pyod

        return str(pyod.__version__)
    except Exception:
        return None


def build_pyod_detector(algorithm: PyODAlgorithm, *, seed: int = 42) -> Any:
    """Instantiate a PyOD detector with library defaults (seeded where stochastic).

    Args:
        algorithm: Which PyOD algorithm to build.
        seed: ``random_state`` for the stochastic detectors (IForest, PCA,
            AutoEncoder). Deterministic detectors (ECOD, COPOD, LOF, KNN,
            HBOS, OCSVM) take no seed -- passing one would deviate from
            defaults for no reproducibility gain.

    Returns:
        An unfitted PyOD detector instance.

    Raises:
        ImportError: PyOD is not installed
            (``pip install mercury-agent[benchmark]``).
        ValueError: ``algorithm`` is not a supported :class:`PyODAlgorithm`.
    """
    try:
        if algorithm is PyODAlgorithm.ISOLATION_FOREST:
            from pyod.models.iforest import IForest

            return IForest(random_state=seed)
        if algorithm is PyODAlgorithm.LOF:
            from pyod.models.lof import LOF

            return LOF()
        if algorithm is PyODAlgorithm.COPOD:
            from pyod.models.copod import COPOD

            return COPOD()
        if algorithm is PyODAlgorithm.ECOD:
            from pyod.models.ecod import ECOD

            return ECOD()
        if algorithm is PyODAlgorithm.OCSVM:
            from pyod.models.ocsvm import OCSVM

            return OCSVM()
        if algorithm is PyODAlgorithm.KNN:
            from pyod.models.knn import KNN

            return KNN()
        if algorithm is PyODAlgorithm.HBOS:
            from pyod.models.hbos import HBOS

            return HBOS()
        if algorithm is PyODAlgorithm.PCA:
            from pyod.models.pca import PCA

            return PCA(random_state=seed)
        if algorithm is PyODAlgorithm.AUTOENCODER:
            from pyod.models.auto_encoder import AutoEncoder

            return AutoEncoder(random_state=seed, verbose=0)
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            f"Could not import the PyOD backend for {algorithm!r}: {exc}. "
            "This usually means PyOD itself is not installed -- install it with: "
            "pip install 'mercury-agent[benchmark]' (or pip install pyod). It can "
            "also be a missing optional/transitive dependency of the specific "
            "model (e.g. AutoEncoder requires torch); the original ImportError "
            "above names the exact missing module."
        ) from exc
    raise ValueError(f"Unsupported PyOD algorithm: {algorithm!r}")


def run_pyod_baselines(
    X_train: np.ndarray[Any, Any],
    X_test: np.ndarray[Any, Any],
    *,
    algorithms: tuple[PyODAlgorithm, ...] | list[PyODAlgorithm] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Fit each PyOD baseline on ``X_train`` (unlabelled) and score ``X_test``.

    This is the unsupervised protocol: detectors see only the (typically
    normal-only) training rows and must rank the test rows. Higher score means
    more anomalous (PyOD's ``decision_function`` convention).

    Args:
        X_train: Training rows ``(n_train, n_features)``. Passed to ``fit``
            exactly as given -- preprocessing is the caller's job so every
            compared method (Mercury included) sees identical input.
        X_test: Test rows ``(n_test, n_features)`` scored by every detector.
        algorithms: Baselines to run (default :data:`DEFAULT_BASELINES`).
        seed: Seed forwarded to :func:`build_pyod_detector`.

    Returns:
        Mapping ``algorithm.value -> result``. A successful result carries
        ``{"scores": np.ndarray (n_test,), "fit_seconds": float,
        "score_seconds": float}``; a failed one carries ``{"error": str}``
        (recorded, never silently dropped).

    Raises:
        ImportError: PyOD is not installed.
    """
    if not pyod_available():
        raise ImportError(
            "PyOD is required to run comparison baselines. "
            "Install it with: pip install 'mercury-agent[benchmark]' (or pip install pyod)."
        )
    chosen = tuple(algorithms) if algorithms is not None else DEFAULT_BASELINES
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    results: dict[str, dict[str, Any]] = {}
    for algorithm in chosen:
        # Stable result key derived up front so the "record, never drop" error
        # path below can never itself raise: a caller passing a non-enum value
        # gets its failure recorded (under str(value)) instead of an
        # AttributeError on ``.value`` crashing the whole baseline run.
        key = algorithm.value if isinstance(algorithm, PyODAlgorithm) else str(algorithm)
        try:
            detector = build_pyod_detector(algorithm, seed=seed)
            t0 = time.perf_counter()
            detector.fit(X_train)
            fit_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = np.asarray(detector.decision_function(X_test), dtype=np.float64).ravel()
            score_seconds = time.perf_counter() - t0
            if scores.shape[0] != X_test.shape[0]:
                raise ValueError(f"scores length {scores.shape[0]} != n_test {X_test.shape[0]}")
            results[key] = {
                "scores": scores,
                "fit_seconds": float(fit_seconds),
                "score_seconds": float(score_seconds),
            }
        except Exception as exc:  # record, never drop
            results[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def _score_metrics(y_test: np.ndarray[Any, Any], scores: np.ndarray[Any, Any]) -> dict[str, float]:
    """ROC-AUC + Average Precision via the in-repo (sklearn-free) metric kernels."""
    from omni_mercury_engine.ml.mercury_ml import average_precision_score, roc_auc_score

    y = np.asarray(y_test, dtype=np.int64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    out: dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y, s))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["average_precision"] = float(average_precision_score(y, s))
    except ValueError:
        out["average_precision"] = float("nan")
    return out


class PyODComparison:
    """Run Mercury-vs-PyOD comparisons on a shared train/test split.

    The caller supplies Mercury's test scores (however they were produced --
    tier ensemble, fusion, a single detector) plus the identical
    ``(X_train, X_test, y_test)`` protocol data; this class runs the PyOD
    baselines on the same split and reports ROC-AUC / Average Precision for
    every method side by side, wins/losses included.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.algorithm_characteristics = self._init_algorithm_profiles()
        self.benchmark_results: dict[str, Any] = {}

    def _init_algorithm_profiles(self) -> dict[PyODAlgorithm, dict[str, Any]]:
        """Initialize algorithm characteristics for selection guidance."""
        return {
            PyODAlgorithm.ISOLATION_FOREST: {
                "type": "tree_based",
                "strengths": [
                    "Efficient",
                    "No assumptions on data distribution",
                    "Handles high dimensions",
                ],
                "weaknesses": ["May struggle with local anomalies", "Not interpretable"],
                "best_for": ["High-dimensional data", "Global anomalies", "Large datasets"],
                "complexity": "O(n log n)",
                "parameters": ["n_estimators", "max_samples", "contamination"],
            },
            PyODAlgorithm.LOF: {
                "type": "density_based",
                "strengths": [
                    "Captures local density deviations",
                    "Intuitive concept",
                    "Works well for clusters",
                ],
                "weaknesses": [
                    "Sensitive to k parameter",
                    "Computationally expensive O(n²)",
                    "Memory intensive",
                ],
                "best_for": ["Datasets with varying density", "Local anomalies", "Clustered data"],
                "complexity": "O(n²)",
                "parameters": ["n_neighbors", "contamination"],
            },
            PyODAlgorithm.COPOD: {
                "type": "statistical",
                "strengths": ["Fast O(n)", "Parameter-free", "Empirical distribution", "Scalable"],
                "weaknesses": ["May miss complex patterns", "Assumes feature independence"],
                "best_for": ["Quick baseline", "Large-scale data", "When speed is critical"],
                "complexity": "O(n)",
                "parameters": ["contamination"],
            },
            PyODAlgorithm.ECOD: {
                "type": "statistical",
                "strengths": ["No hyperparameters", "Fast", "Works on raw data", "Interpretable"],
                "weaknesses": ["Assumes feature independence", "May not capture dependencies"],
                "best_for": ["Quick deployment", "No tuning budget", "Interpretability needed"],
                "complexity": "O(n log n)",
                "parameters": [],
            },
            PyODAlgorithm.KNN: {
                "type": "distance_based",
                "strengths": ["Simple", "Strong ranking baseline", "Few assumptions"],
                "weaknesses": ["O(n²) neighbour search", "Curse of dimensionality"],
                "best_for": ["Small/medium tabular data", "Local + global anomalies"],
                "complexity": "O(n²)",
                "parameters": ["n_neighbors", "method"],
            },
            PyODAlgorithm.HBOS: {
                "type": "statistical",
                "strengths": ["Very fast", "Histogram-based", "Scales linearly"],
                "weaknesses": ["Assumes feature independence", "Bin sensitivity"],
                "best_for": ["Large datasets", "Speed-critical baselines"],
                "complexity": "O(n)",
                "parameters": ["n_bins", "alpha"],
            },
        }

    def recommend_algorithm(
        self, data_characteristics: dict[str, Any], constraints: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Recommend best algorithm(s) based on data characteristics.

        Args:
            data_characteristics: Data properties dict
            constraints: Optional constraints dict

        Returns:
            Recommended algorithms with rationale
        """
        recommendations = []

        num_samples = data_characteristics.get("num_samples", 1000)
        max_time = (
            constraints.get("max_time_seconds", float("inf")) if constraints else float("inf")
        )

        if num_samples > 100000 and max_time < 60:
            recommendations.append(
                {
                    "algorithm": PyODAlgorithm.COPOD,
                    "rationale": "Fast O(n), parameter-free, scales to large datasets",
                    "priority": 1,
                }
            )
        else:
            recommendations.append(
                {
                    "algorithm": PyODAlgorithm.ISOLATION_FOREST,
                    "rationale": "General-purpose, robust default",
                    "priority": 1,
                }
            )

        def get_priority(x: dict[str, object]) -> int:
            return int(str(x["priority"]))

        recommendations.sort(key=get_priority)

        return {
            "recommendations": recommendations,
            "data_summary": data_characteristics,
            "constraints": constraints,
        }

    def combine_predictions(
        self,
        predictions: dict[str, np.ndarray[Any, Any]],
        method: CombinationMethod = CombinationMethod.AVERAGE,
    ) -> np.ndarray[Any, Any]:
        """Combine predictions from multiple detectors using PyOD-inspired methods.

        Args:
            predictions: {detector_name: anomaly_scores} for multiple detectors
            method: Combination method (Average, Maximum, AOM, MOA)

        Returns:
            Combined anomaly scores
        """
        scores_matrix = np.array(list(predictions.values()))

        if method == CombinationMethod.AVERAGE:
            return np.asarray(np.mean(scores_matrix, axis=0))

        elif method == CombinationMethod.MAXIMUM:
            return np.asarray(np.max(scores_matrix, axis=0))

        elif method == CombinationMethod.AOM:
            num_detectors = len(predictions)
            k = max(1, num_detectors // 2)

            partitions = np.array_split(scores_matrix, num_detectors // k)

            max_scores = [np.max(partition, axis=0) for partition in partitions]

            return np.asarray(np.mean(max_scores, axis=0))

        elif method == CombinationMethod.MOA:
            num_detectors = len(predictions)
            k = max(1, num_detectors // 2)

            partitions = np.array_split(scores_matrix, num_detectors // k)

            avg_scores = [np.mean(partition, axis=0) for partition in partitions]

            return np.asarray(np.max(avg_scores, axis=0))

        return np.asarray(np.mean(scores_matrix, axis=0))

    def benchmark_against_pyod(
        self,
        mercury_scores: dict[str, np.ndarray[Any, Any]],
        X_train: np.ndarray[Any, Any],
        X_test: np.ndarray[Any, Any],
        y_test: np.ndarray[Any, Any],
        algorithms: tuple[PyODAlgorithm, ...] | list[PyODAlgorithm] | None = None,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Benchmark Mercury score vectors against real PyOD baselines.

        Every PyOD baseline is fitted on ``X_train`` and scored on ``X_test``
        -- the *same* rows, in the *same* order, that produced
        ``mercury_scores``. The caller is responsible for that protocol
        (contamination-free normal-only train split, fixed-seed test-row
        shuffle, train-fitted scaling); this method compares, it does not
        re-split.

        Args:
            mercury_scores: ``{method_name: scores}`` for Mercury methods
                (e.g. ``{"mercury_tier": ..., "mercury_fusion": ...}``), each
                aligned to ``X_test`` rows, higher = more anomalous.
            X_train: The (typically normal-only) training rows.
            X_test: The evaluation rows.
            y_test: Ground-truth 0/1 labels aligned to ``X_test``.
            algorithms: PyOD baselines (default :data:`DEFAULT_BASELINES`).
            seed: Seed for stochastic PyOD detectors.

        Returns:
            Dict with per-method metrics (``mercury`` + ``pyod`` sections,
            each ``{name: {roc_auc, average_precision, ...}}``) and a
            ``comparison_summary`` of measured wins/losses per Mercury method
            vs each baseline.

        Raises:
            ImportError: PyOD is not installed.
        """
        mercury_metrics: dict[str, dict[str, Any]] = {}
        for name, scores in mercury_scores.items():
            mercury_metrics[name] = _score_metrics(y_test, scores)

        pyod_runs = run_pyod_baselines(X_train, X_test, algorithms=algorithms, seed=seed)
        pyod_metrics: dict[str, dict[str, Any]] = {}
        for name, run in pyod_runs.items():
            if "error" in run:
                pyod_metrics[name] = {"error": run["error"]}
                continue
            pyod_metrics[name] = {
                **_score_metrics(y_test, run["scores"]),
                "fit_seconds": run["fit_seconds"],
                "score_seconds": run["score_seconds"],
            }

        results = {
            "mercury": mercury_metrics,
            "pyod": pyod_metrics,
            "comparison_summary": self._generate_comparison_summary(mercury_metrics, pyod_metrics),
        }
        self.benchmark_results = results
        return results

    def _generate_comparison_summary(
        self,
        mercury_metrics: dict[str, dict[str, Any]],
        pyod_metrics: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Measured wins/losses per Mercury method vs each PyOD baseline.

        A "win" is a strictly higher ROC-AUC; ties count to neither side.
        Losses are reported with the same prominence as wins -- the summary is
        a measurement, not a recommendation.
        """
        summary: dict[str, Any] = {}
        for mercury_name, mm in mercury_metrics.items():
            mercury_auc = mm.get("roc_auc")
            entry: dict[str, Any] = {"roc_auc": mercury_auc, "vs": {}}
            if mercury_auc is None or np.isnan(mercury_auc):
                entry["note"] = "mercury ROC-AUC not measurable on this split"
                summary[mercury_name] = entry
                continue
            for pyod_name, pm in pyod_metrics.items():
                if "error" in pm or np.isnan(pm.get("roc_auc", float("nan"))):
                    entry["vs"][pyod_name] = {"result": "baseline_error"}
                    continue
                delta = float(mercury_auc - pm["roc_auc"])
                entry["vs"][pyod_name] = {
                    "baseline_roc_auc": pm["roc_auc"],
                    "auc_delta": delta,
                    "result": "win" if delta > 0 else ("loss" if delta < 0 else "tie"),
                }
            summary[mercury_name] = entry
        return summary

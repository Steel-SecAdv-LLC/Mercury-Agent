# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared infrastructure for the hazard-checkpoint training pipeline.

Everything here enforces the project's honesty rules for shipped neural
weights:

* **Real data only** -- fetch helpers cache raw source bytes on disk and
  record SHA-256 digests so every training run is traceable to the exact
  upstream files it consumed.
* **Temporal splits only** -- :class:`TemporalSplit` splits by calendar year
  and refuses interleaved years. Random splits leak future information into
  training for every geophysical series this pipeline touches.
* **Merit gate** -- :func:`ship_checkpoint` refuses to write a checkpoint into
  the shipped-models directory unless the learned model beat the physics
  fallback on the held-out test years (:class:`EvaluationOutcome`).
* **Fail loud** -- hooks whose real training data is not obtainable in this
  environment raise :class:`HazardDataUnavailableError` with the documented
  data requirement instead of fabricating a dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omni_mercury_engine.datasets.base import http_get_with_retry
from omni_mercury_engine.models.checkpoint_paths import checkpoints_dir

logger = logging.getLogger(__name__)

#: Default on-disk cache for downloaded training data (gitignored via /data/).
DEFAULT_DATA_DIR = Path("data") / "hazard_training"


@dataclass
class PipelineContext:
    """Runtime options threaded through every pipeline stage.

    Attributes:
        data_dir: On-disk cache/workspace for fetched data and candidates.
        seed: Deterministic seed for torch + numpy.
        max_epochs: Upper bound on training epochs (early stopping may end
            sooner).
        ship_dir: Override for the shipped checkpoints directory; None means
            the packaged ``models/checkpoints`` directory.
        limit_samples: Optional cap on dataset size (tiny fixture runs in
            tests); None means use everything fetched.
    """

    data_dir: Path = DEFAULT_DATA_DIR
    seed: int = 20260709
    max_epochs: int = 60
    ship_dir: Path | None = None
    limit_samples: int | None = None


class HazardDataUnavailableError(RuntimeError):
    """Raised when a hook's real training data cannot be obtained here.

    This is the honest terminal state for category (b)/(c) hooks: the message
    carries the full data requirement so an operator with the missing
    credentials or archives can run the same pipeline stage themselves.
    """


class MeritGateError(RuntimeError):
    """Raised when a checkpoint fails the learned-beats-physics merit gate."""


def seed_everything(seed: int) -> np.random.Generator:
    """Seed torch and return a fresh numpy generator for reproducible runs.

    Args:
        seed: Seed applied to torch (CPU) and used for the numpy generator.

    Returns:
        A ``numpy.random.Generator`` seeded with ``seed``.
    """
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_fetch(url: str, dest: Path, *, timeout: float = 120.0) -> Path:
    """Download ``url`` to ``dest`` unless a cached copy already exists.

    The transport is the project's :func:`http_get_with_retry`, so the host
    must be on the trusted-endpoint allowlist and HTTPS-only rules apply.

    Args:
        url: Source URL (must be allowlisted).
        dest: Cache path; parent directories are created.
        timeout: Total per-URL timeout in seconds.

    Returns:
        ``dest`` once the file exists.

    Raises:
        RuntimeError: If the server returns an empty body (a silent truncation
            must not be cached as if it were data).
    """
    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("cache hit: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = http_get_with_retry(url, timeout=timeout)
    if not body:
        raise RuntimeError(f"empty response body from {url}; refusing to cache")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(dest)
    logger.info("fetched %s -> %s (%d bytes)", url, dest, len(body))
    return dest


@dataclass(frozen=True)
class TemporalSplit:
    """A by-year train/validation/test split for time-series training.

    Years must be strictly ordered ``train < val < test`` with no overlap so
    no future information can leak backwards into model fitting.
    """

    train_years: tuple[int, ...]
    val_years: tuple[int, ...]
    test_years: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate ordering and disjointness."""
        for name, years in (
            ("train_years", self.train_years),
            ("val_years", self.val_years),
            ("test_years", self.test_years),
        ):
            if not years:
                raise ValueError(f"{name} must not be empty")
        if not (max(self.train_years) < min(self.val_years)):
            raise ValueError("temporal split violated: max(train) must be < min(val)")
        if not (max(self.val_years) < min(self.test_years)):
            raise ValueError("temporal split violated: max(val) must be < min(test)")

    @property
    def all_years(self) -> tuple[int, ...]:
        """All years covered by the split, ascending."""
        return tuple(sorted({*self.train_years, *self.val_years, *self.test_years}))

    def masks(self, years: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return boolean (train, val, test) masks for a per-sample year array."""
        train = np.isin(years, self.train_years)
        val = np.isin(years, self.val_years)
        test = np.isin(years, self.test_years)
        return train, val, test


@dataclass
class EvaluationOutcome:
    """Learned-vs-physics comparison on the held-out test years.

    Attributes:
        hook: Registry name of the hook being evaluated.
        primary_metric: Key (present in both metric dicts) that the merit
            gate compares.
        higher_is_better: Direction of the primary metric.
        learned: Metrics for the detector running the trained checkpoint.
        physics: Metrics for the detector's physics fallback (or the
            documented physics baseline where the fallback abstains).
        n_test_samples: Number of held-out cases both models saw.
        test_years: The held-out years.
        extras: Additional context recorded into provenance verbatim.
    """

    hook: str
    primary_metric: str
    higher_is_better: bool
    learned: dict[str, float]
    physics: dict[str, float]
    n_test_samples: int
    test_years: tuple[int, ...]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def learned_beats_physics(self) -> bool:
        """True when the learned model wins the primary metric outright."""
        lv = self.learned[self.primary_metric]
        pv = self.physics[self.primary_metric]
        if not (np.isfinite(lv) and np.isfinite(pv)):
            return False
        return bool(lv > pv) if self.higher_is_better else bool(lv < pv)

    def to_json(self) -> dict[str, Any]:
        """Serializable form for the evaluation record and provenance."""
        return {
            "hook": self.hook,
            "primary_metric": self.primary_metric,
            "higher_is_better": self.higher_is_better,
            "learned": self.learned,
            "physics": self.physics,
            "learned_beats_physics": self.learned_beats_physics,
            "n_test_samples": self.n_test_samples,
            "test_years": list(self.test_years),
            "extras": self.extras,
        }


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC for binary labels without sklearn.

    Args:
        labels: 0/1 array.
        scores: Real-valued scores, higher = more positive.

    Returns:
        AUC in [0, 1]; NaN if either class is absent.
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks over ties so equal scores contribute 0.5.
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    rank_sum_pos = float(ranks[labels].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    """Mean squared error between probabilities and binary outcomes."""
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))


def _git_commit() -> str | None:
    """Current git commit hash, or None when not in a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def candidate_paths(data_dir: Path, hook: str) -> tuple[Path, Path]:
    """(candidate checkpoint, evaluation record) paths for a hook."""
    cand_dir = data_dir / "candidates"
    return cand_dir / f"{hook}.pt", cand_dir / f"{hook}.eval.json"


def save_candidate(
    data_dir: Path, hook: str, payload: dict[str, Any], train_record: dict[str, Any]
) -> Path:
    """Persist a trained-but-not-shipped candidate checkpoint.

    Args:
        data_dir: Pipeline data directory.
        hook: Hook registry name.
        payload: Torch-saveable checkpoint payload (state dicts + metadata).
        train_record: Training metadata merged into the payload under
            ``"training"``.

    Returns:
        The candidate checkpoint path.
    """
    cand_path, _ = candidate_paths(data_dir, hook)
    cand_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["training"] = train_record
    torch.save(payload, cand_path)
    logger.info("candidate checkpoint written: %s", cand_path)
    return cand_path


def save_evaluation(data_dir: Path, outcome: EvaluationOutcome) -> Path:
    """Persist an evaluation outcome next to the candidate checkpoint."""
    _, eval_path = candidate_paths(data_dir, outcome.hook)
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(outcome.to_json(), indent=2, sort_keys=True))
    logger.info("evaluation record written: %s", eval_path)
    return eval_path


def load_evaluation(data_dir: Path, hook: str) -> EvaluationOutcome:
    """Load a previously saved evaluation outcome for ``hook``.

    Raises:
        FileNotFoundError: If the evaluate stage has not been run.
    """
    _, eval_path = candidate_paths(data_dir, hook)
    if not eval_path.exists():
        raise FileNotFoundError(
            f"no evaluation record at {eval_path}; run the --evaluate stage first"
        )
    raw = json.loads(eval_path.read_text())
    return EvaluationOutcome(
        hook=raw["hook"],
        primary_metric=raw["primary_metric"],
        higher_is_better=raw["higher_is_better"],
        learned=raw["learned"],
        physics=raw["physics"],
        n_test_samples=raw["n_test_samples"],
        test_years=tuple(raw["test_years"]),
        extras=raw.get("extras", {}),
    )


def ship_checkpoint(
    *,
    hook: str,
    checkpoint_name: str,
    data_dir: Path,
    outcome: EvaluationOutcome,
    data_sources: list[dict[str, Any]],
    seed: int,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Promote a candidate checkpoint to the shipped models directory.

    Refuses -- loudly, with a nonzero-exit-worthy exception -- unless the
    evaluation shows the learned model beat the physics fallback on the
    primary metric for the held-out test years.

    Args:
        hook: Hook registry name.
        checkpoint_name: Basename (without extension) for the shipped file.
        data_dir: Pipeline data directory holding the candidate.
        outcome: The held-out evaluation this promotion is judged on.
        data_sources: Provenance entries, each with at least ``url`` /
            ``sha256`` / ``description`` keys.
        seed: Training seed recorded into provenance.
        out_dir: Override for the shipped checkpoints directory (tests).

    Returns:
        Tuple of (shipped checkpoint path, provenance sidecar path).

    Raises:
        MeritGateError: If the learned model did not beat physics.
        FileNotFoundError: If the candidate checkpoint is missing.
    """
    if not outcome.learned_beats_physics:
        direction = "higher" if outcome.higher_is_better else "lower"
        raise MeritGateError(
            f"MERIT GATE REFUSED for hook '{hook}': learned "
            f"{outcome.primary_metric}={outcome.learned[outcome.primary_metric]:.6g} does not "
            f"beat physics {outcome.primary_metric}="
            f"{outcome.physics[outcome.primary_metric]:.6g} ({direction} is better) on held-out "
            f"years {list(outcome.test_years)}. The physics fallback stays in charge; the "
            "honest deliverable is this evaluation record, not a shipped checkpoint."
        )
    cand_path, _ = candidate_paths(data_dir, hook)
    if not cand_path.exists():
        raise FileNotFoundError(f"no candidate checkpoint at {cand_path}; run --train first")

    target_dir = out_dir if out_dir is not None else checkpoints_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    shipped = target_dir / f"{checkpoint_name}.pt"
    shipped.write_bytes(cand_path.read_bytes())

    provenance = {
        "hook": hook,
        "checkpoint": shipped.name,
        "checkpoint_sha256": sha256_file(shipped),
        "created_utc": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "seed": seed,
        "data_sources": data_sources,
        "evaluation": outcome.to_json(),
    }
    sidecar = target_dir / f"{checkpoint_name}.provenance.json"
    sidecar.write_text(json.dumps(provenance, indent=2, sort_keys=True))
    logger.info("shipped %s (+ provenance %s)", shipped, sidecar)
    return shipped, sidecar

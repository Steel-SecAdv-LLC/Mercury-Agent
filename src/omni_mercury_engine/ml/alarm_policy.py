# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A learned alarm policy for streaming anomaly detection, trained with PPO on NAB.

The streaming worker (``mercury stream``) must decide, message by message,
whether to raise an alarm. NAB (:mod:`omni_mercury_engine.evaluation.nab_scoring`)
scores exactly that decision: early alarms inside a labelled anomaly window earn
credit once per window, and every alarm outside one costs. This module learns
the decision as a policy and serves it without PyTorch.

**Features** (:class:`ValueFeatureState`, causal and per numeric field): fast and
slow EMA z-scores, rolling median/MAD robust z-scores over 100 and 1,000
values, the value's two-sided percentile-rank extremeness among the last 500,
and an EMA z-score of the first difference. Each value is scored against
statistics it has not yet moved.

**Triage** (:class:`AlarmHistory`): only *candidate* points, whose rank
extremeness exceeds ``gate``, reach the policy. It sees the features plus its
own recent history (time since its last alarm and last candidate, candidate
and alarm counts), and decides alarm or not. Every other point is not alarmed.
Gating keeps the decision rare enough to learn: on the raw stream, PPO learned
never to alarm.

**Training** (:func:`train_alarm_policy`): :class:`~omni_mercury_engine.ml.ppo_trainer.PPOTrainer`
on :class:`TriageEnv`, whose reward is the exact NAB score increment of each
decision, so an episode's return is the stream's raw NAB score plus a constant.

**Serving** (:class:`AlarmPolicy`, :class:`StreamingAlarmDetector`): the trained
network's policy tower exported to a NumPy archive with a SHA-256-pinned
provenance sidecar; the deterministic decision is the more probable action.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from omni_mercury_engine.evaluation.nab_scoring import NABStream, SweepTable
    from omni_mercury_engine.ml.ppo_trainer import PPOConfig, PPOTrainer

__all__ = [
    "FEATURE_NAMES",
    "HISTORY_NAMES",
    "SHIPPED_POLICY",
    "AlarmHistory",
    "AlarmPolicy",
    "StreamingAlarmDetector",
    "TriageEnv",
    "ValueFeatureState",
    "export_alarm_policy",
    "load_alarm_policy",
    "run_alarm_policy",
    "stream_features",
    "train_alarm_policy",
]

#: Per-value features, in observation order.
FEATURE_NAMES = (
    "z_fast",
    "z_slow",
    "robust_z",
    "robust_z_long",
    "rank_extremeness",
    "z_diff",
)
#: Alarm-history features appended to the value features.
HISTORY_NAMES = (
    "since_alarm",
    "since_candidate",
    "recent_candidates",
    "recent_alarms",
)
#: Index of the feature the candidate gate reads.
GATE_FEATURE = FEATURE_NAMES.index("rank_extremeness")
#: Archive format of an exported policy.
POLICY_FORMAT = "mercury.alarm_policy"
POLICY_VERSION = 1
#: The policy the streaming worker ships with.
SHIPPED_POLICY = (
    Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "streaming_alarm_policy.npz"
)

_WARMUP = 10
_Z_CAP = 10.0


class ValueFeatureState:
    """Causal features of one numeric stream, updated one value at a time."""

    def __init__(self) -> None:
        """Start empty (every feature is 0 until 10 values have been seen)."""
        self.count = 0
        self._ema_fast = [0.0, 0.0]
        self._ema_slow = [0.0, 0.0]
        self._ema_diff = [0.0, 0.0]
        self._previous: float | None = None
        self._short: deque[float] = deque(maxlen=100)
        self._long: deque[float] = deque(maxlen=1000)
        self._rank: deque[float] = deque(maxlen=500)

    @staticmethod
    def _ema_z(state: list[float], x: float, alpha: float, ready: bool) -> float:
        mean, var = state
        z = abs(x - mean) / math.sqrt(var + 1e-12) if ready else 0.0
        delta = x - mean
        state[0] = mean + alpha * delta
        state[1] = (1.0 - alpha) * (var + alpha * delta * delta)
        return z

    @staticmethod
    def _robust_z(window: deque[float], x: float) -> float:
        if len(window) < _WARMUP:
            return 0.0
        values = np.fromiter(window, dtype=np.float64, count=len(window))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return abs(x - median) / (1.4826 * mad + 1e-9 * (abs(median) + 1.0))

    def update(self, x: float) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Score ``x`` against the past, then add it; returns :data:`FEATURE_NAMES` values.

        z-scores are compressed as ``log1p(z)`` and every feature is capped at
        10; rank extremeness is ``|2 F - 1|`` for the value's empirical CDF
        position ``F`` among the last 500 values, in [0, 1].
        """
        x = float(x)
        ready = self.count >= _WARMUP
        if self.count == 0:
            self._ema_fast[0] = self._ema_slow[0] = x
        z_fast = self._ema_z(self._ema_fast, x, 0.1, ready)
        z_slow = self._ema_z(self._ema_slow, x, 0.01, ready)
        robust = self._robust_z(self._short, x)
        robust_long = self._robust_z(self._long, x)
        if len(self._rank) >= _WARMUP:
            past = np.fromiter(self._rank, dtype=np.float64, count=len(self._rank))
            position = ((past < x).sum() + 0.5 * (past == x).sum()) / past.size
            extremeness = abs(2.0 * position - 1.0)
        else:
            extremeness = 0.0
        if self._previous is None:
            z_diff = 0.0
        else:
            z_diff = self._ema_z(self._ema_diff, x - self._previous, 0.1, self.count >= _WARMUP + 1)
        self._previous = x
        self._short.append(x)
        self._long.append(x)
        self._rank.append(x)
        self.count += 1
        z = np.log1p(np.minimum([z_fast, z_slow, robust, robust_long, 0.0, z_diff], 1e6))
        z[GATE_FEATURE] = extremeness
        return np.minimum(z, _Z_CAP).astype(np.float32)


def stream_features(values: Sequence[float] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """:class:`ValueFeatureState` features for a whole series, ``[n, 6]`` float32."""
    state = ValueFeatureState()
    rows = [state.update(float(v)) for v in values]
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack(rows)


@dataclass
class AlarmHistory:
    """The policy's memory of its own recent decisions on one stream."""

    last_alarm: int = -(10**9)
    last_candidate: int = -(10**9)
    candidates: deque[int] = field(default_factory=deque)
    alarms: deque[int] = field(default_factory=deque)

    def features(self, t: int) -> list[float]:
        """:data:`HISTORY_NAMES` values at time ``t`` (before deciding at ``t``)."""
        while self.candidates and self.candidates[0] <= t - 100:
            self.candidates.popleft()
        while self.alarms and self.alarms[0] <= t - 1000:
            self.alarms.popleft()
        return [
            min(1.0, math.log1p(t - self.last_alarm) / math.log1p(10_000)),
            min(1.0, math.log1p(t - self.last_candidate) / math.log1p(1000)),
            len(self.candidates) / 20.0,
            len(self.alarms) / 10.0,
        ]

    def record(self, t: int, alarm: bool) -> None:
        """Remember the decision taken at candidate time ``t``."""
        if alarm:
            self.last_alarm = t
            self.alarms.append(t)
        self.last_candidate = t
        self.candidates.append(t)


@dataclass(frozen=True)
class AlarmPolicy:
    """A trained policy network, as NumPy arrays (no PyTorch needed).

    Attributes:
        obs_mean: Observation normalization mean.
        obs_var: Observation normalization variance.
        clip_obs: Bound on normalized observations.
        weights: Hidden-layer weights, each ``[in, out]`` (policy tower).
        biases: Hidden-layer biases.
        head_weight: Output weights ``[latent, 2]``.
        head_bias: Output biases ``[2]``.
        activation: ``"tanh"`` or ``"relu"``.
        gate: Rank-extremeness threshold for candidates.
        provenance: Training record (empty when built in memory).
    """

    obs_mean: np.ndarray[Any, Any]
    obs_var: np.ndarray[Any, Any]
    clip_obs: float
    weights: tuple[np.ndarray[Any, Any], ...]
    biases: tuple[np.ndarray[Any, Any], ...]
    head_weight: np.ndarray[Any, Any]
    head_bias: np.ndarray[Any, Any]
    activation: str
    gate: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def alarm_probability(self, obs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """``P(alarm)`` for observations ``[B, 10]`` (or one ``[10]``)."""
        x = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        x = np.clip(
            (x - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -self.clip_obs, self.clip_obs
        )
        for w, b in zip(self.weights, self.biases, strict=True):
            x = x @ w + b
            x = np.tanh(x) if self.activation == "tanh" else np.maximum(x, 0.0)
        logits = x @ self.head_weight + self.head_bias
        p = 1.0 / (1.0 + np.exp(logits[:, 0] - logits[:, 1]))
        return p if np.ndim(obs) > 1 else p[:1]

    def decide(self, obs: np.ndarray[Any, Any]) -> tuple[bool, float]:
        """The deterministic decision (the more probable action) and ``P(alarm)``."""
        p = float(self.alarm_probability(obs)[0])
        return p > 0.5, p


def run_alarm_policy(policy: AlarmPolicy, features: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Alarms the policy raises over one stream's feature rows (``[n]`` bool)."""
    alarms = np.zeros(features.shape[0], dtype=bool)
    history = AlarmHistory()
    for t in np.flatnonzero(features[:, GATE_FEATURE] > policy.gate):
        obs = np.concatenate([features[t], history.features(int(t))])
        alarm, _ = policy.decide(obs)
        alarms[t] = alarm
        history.record(int(t), alarm)
    return alarms


class StreamingAlarmDetector:
    """The streaming pipeline's detector: the learned alarm policy, per numeric field.

    A ``dict -> dict`` callable for
    :class:`~omni_mercury_engine.infrastructure.streaming.StreamingAnomalyPipeline`.
    Each finite numeric field keeps its own :class:`ValueFeatureState` and
    :class:`AlarmHistory`; a message alarms when the policy alarms on any field.

    Args:
        policy: The policy (default: the shipped one, verified on load).
    """

    def __init__(self, policy: AlarmPolicy | None = None) -> None:
        """Load (and verify) the policy."""
        self.policy = policy or load_alarm_policy()
        self._states: dict[str, ValueFeatureState] = {}
        self._histories: dict[str, AlarmHistory] = {}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Decide on one message; the result schema of the pipeline's detectors."""
        feature_scores: dict[str, float] = {}
        anomaly_features: list[str] = []
        candidates = 0
        for key, value in data.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if not math.isfinite(value):
                continue
            name = str(key)
            state = self._states.setdefault(name, ValueFeatureState())
            history = self._histories.setdefault(name, AlarmHistory())
            t = state.count
            features = state.update(float(value))
            if features[GATE_FEATURE] <= self.policy.gate:
                feature_scores[name] = 0.0
                continue
            candidates += 1
            alarm, p = self.policy.decide(np.concatenate([features, history.features(t)]))
            history.record(t, alarm)
            feature_scores[name] = round(p, 4)
            if alarm:
                anomaly_features.append(name)
        score = max(feature_scores.values(), default=0.0)
        return {
            "input": data,
            "is_anomaly": bool(anomaly_features),
            "score": score,
            "detector": "learned_alarm_policy",
            "anomaly_features": anomaly_features,
            "feature_scores": feature_scores,
            "statistical_context": {
                "policy": self.policy.provenance.get("name", "in-memory"),
                "gate": self.policy.gate,
                "features_analyzed": len(feature_scores),
                "candidates": candidates,
            },
        }


# -- training (needs PyTorch) ----------------------------------------------------------


class TriageEnv:
    """Vector environment over candidate points of NAB streams.

    Each copy plays one stream's candidates in order; the reward of a decision
    is its exact NAB score increment (an alarm in a window lifts that window's
    best score from ``-fnWeight``; one outside costs its sweep score).

    Args:
        features: Feature rows per stream (:func:`stream_features`).
        tables: NAB sweep tables per stream (the same keys).
        gate: Rank-extremeness threshold for candidates.
        num_envs: Copies stepped together.
        seed: Stream-order seed.
    """

    def __init__(
        self,
        features: Mapping[str, np.ndarray[Any, Any]],
        tables: Mapping[str, SweepTable],
        gate: float,
        num_envs: int = 16,
        seed: int | None = None,
    ) -> None:
        """Index every stream's candidate points."""
        from omni_mercury_engine.ml.rl_envs import Box, Discrete

        self.features, self.tables, self.gate = features, tables, gate
        self.candidates = {
            n: np.flatnonzero(f[:, GATE_FEATURE] > gate) for n, f in features.items()
        }
        self.names = [n for n in features if self.candidates[n].size]
        if not self.names:
            raise ValueError("no stream has a candidate point at this gate")
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        size = len(FEATURE_NAMES) + len(HISTORY_NAMES)
        self.observation_space = Box(
            np.full(size, -np.inf, np.float32), np.full(size, np.inf, np.float32)
        )
        self.action_space = Discrete(2)
        self._episodes: list[dict[str, Any]] = [{} for _ in range(num_envs)]

    def _start(self, i: int) -> None:
        name = self.names[int(self._rng.integers(len(self.names)))]
        self._episodes[i] = {"name": name, "k": 0, "history": AlarmHistory(), "best": {}}

    def _obs(self, i: int) -> np.ndarray[Any, Any]:
        episode = self._episodes[i]
        t = int(self.candidates[episode["name"]][episode["k"]])
        history: AlarmHistory = episode["history"]
        return np.concatenate([self.features[episode["name"]][t], history.features(t)]).astype(
            np.float32
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Start every copy on a random stream."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for i in range(self.num_envs):
            self._start(i)
        return np.stack([self._obs(i) for i in range(self.num_envs)]), {}

    def step(self, actions: np.ndarray[Any, Any]) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        dict[str, Any],
    ]:
        """Apply one decision per copy; a finished stream restarts on another."""
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        for i, action in enumerate(np.asarray(actions).reshape(-1)):
            episode = self._episodes[i]
            name = episode["name"]
            t = int(self.candidates[name][episode["k"]])
            table = self.tables[name]
            alarm = bool(action)
            if alarm and table.scorable[t]:
                window = int(table.window_id[t])
                if window >= 0:
                    before = episode["best"].get(window, -table.profile.fn)
                    after = max(before, float(table.weighted[t]))
                    rewards[i] = after - before
                    episode["best"][window] = after
                else:
                    rewards[i] = float(table.weighted[t])
            episode["history"].record(t, alarm)
            episode["k"] += 1
            if episode["k"] >= self.candidates[name].size:
                terminated[i] = True
                self._start(i)
        obs = np.stack([self._obs(i) for i in range(self.num_envs)])
        # Streams end (terminated); no truncation ever needs a bootstrap value.
        return (
            obs,
            rewards,
            terminated,
            np.zeros(self.num_envs, dtype=bool),
            {"final_observation": obs},
        )


def default_training_config(seed: int = 0) -> PPOConfig:
    """The PPO settings the shipped policy was trained with."""
    from omni_mercury_engine.ml.ppo_trainer import PPOConfig

    return PPOConfig(
        seed=seed,
        verbose=0,
        n_envs=16,
        n_steps=128,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.03,
        learning_rate=3e-4,
        lr_schedule="linear",
        normalize_observations=True,
        net_arch=(64, 64),
    )


def train_alarm_policy(
    features: Mapping[str, np.ndarray[Any, Any]],
    tables: Mapping[str, SweepTable],
    *,
    gate: float,
    total_timesteps: int,
    config: PPOConfig | None = None,
) -> PPOTrainer:
    """Train the triage policy with PPO on the given streams.

    Args:
        features: Feature rows per stream.
        tables: NAB sweep tables per stream.
        gate: Rank-extremeness threshold for candidates.
        total_timesteps: Candidate decisions to train on.
        config: PPO settings (default: :func:`default_training_config`).

    Returns:
        The trained trainer.
    """
    from omni_mercury_engine.ml.ppo_trainer import PPOTrainer

    cfg = config or default_training_config()
    env = TriageEnv(features, tables, gate, num_envs=cfg.n_envs, seed=cfg.seed)
    trainer = PPOTrainer(env, cfg)
    trainer.learn(total_timesteps)
    return trainer


def _policy_arrays(trainer: PPOTrainer, gate: float) -> dict[str, np.ndarray[Any, Any]]:
    policy = trainer.policy
    if not policy.discrete or policy.action_dim != 2:
        raise ValueError("an alarm policy has exactly two discrete actions")
    if trainer.obs_rms is None:
        raise ValueError("an alarm policy is trained with observation normalization")
    arrays: dict[str, np.ndarray[Any, Any]] = {
        "format": np.asarray(POLICY_FORMAT),
        "version": np.asarray(POLICY_VERSION),
        "activation": np.asarray(trainer.config.activation),
        "gate": np.asarray(gate, dtype=np.float64),
        "clip_obs": np.asarray(trainer.config.clip_observations, dtype=np.float64),
        "obs_mean": trainer.obs_rms.mean.astype(np.float64),
        "obs_var": trainer.obs_rms.var.astype(np.float64),
        "n_hidden": np.asarray(policy.n_hidden),
        "head_weight": policy.tensors["pi.weight"].cpu().numpy().astype(np.float64),
        "head_bias": policy.tensors["pi.bias"].cpu().numpy().astype(np.float64),
    }
    for k in range(policy.n_hidden):
        # Tower 0 of each stacked hidden layer is the policy network.
        arrays[f"weight_{k}"] = (
            policy.tensors[f"hidden.{k}.weight"][0].cpu().numpy().astype(np.float64)
        )
        arrays[f"bias_{k}"] = (
            policy.tensors[f"hidden.{k}.bias"][0, 0].cpu().numpy().astype(np.float64)
        )
    return arrays


def _from_arrays(
    arrays: Mapping[str, np.ndarray[Any, Any]], provenance: Mapping[str, Any]
) -> AlarmPolicy:
    if str(arrays["format"]) != POLICY_FORMAT or int(arrays["version"]) != POLICY_VERSION:
        raise ValueError("not a Mercury alarm policy of a supported version")
    n_hidden = int(arrays["n_hidden"])
    policy = AlarmPolicy(
        obs_mean=np.asarray(arrays["obs_mean"], dtype=np.float64),
        obs_var=np.asarray(arrays["obs_var"], dtype=np.float64),
        clip_obs=float(arrays["clip_obs"]),
        weights=tuple(np.asarray(arrays[f"weight_{k}"], dtype=np.float64) for k in range(n_hidden)),
        biases=tuple(np.asarray(arrays[f"bias_{k}"], dtype=np.float64) for k in range(n_hidden)),
        head_weight=np.asarray(arrays["head_weight"], dtype=np.float64),
        head_bias=np.asarray(arrays["head_bias"], dtype=np.float64),
        activation=str(arrays["activation"]),
        gate=float(arrays["gate"]),
        provenance=dict(provenance),
    )
    size = len(FEATURE_NAMES) + len(HISTORY_NAMES)
    widths = [size, *(w.shape[1] for w in policy.weights)]
    shapes_ok = (
        policy.obs_mean.shape == (size,)
        and policy.obs_var.shape == (size,)
        and all(
            w.shape == (a, b) for w, a, b in zip(policy.weights, widths, widths[1:], strict=False)
        )
        and all(
            b.shape == (w.shape[1],) for w, b in zip(policy.weights, policy.biases, strict=True)
        )
        and policy.head_weight.shape == (widths[-1], 2)
        and policy.head_bias.shape == (2,)
        and policy.activation in ("tanh", "relu")
    )
    if not shapes_ok:
        raise ValueError("alarm policy arrays have inconsistent shapes")
    return policy


def policy_from_trainer(trainer: PPOTrainer, gate: float) -> AlarmPolicy:
    """The NumPy policy of a trained :class:`PPOTrainer` (no file involved)."""
    return _from_arrays(_policy_arrays(trainer, gate), {})


def export_alarm_policy(
    trainer: PPOTrainer,
    path: str | Path,
    *,
    gate: float,
    provenance: Mapping[str, Any],
) -> Path:
    """Write the policy archive and its provenance sidecar (pinning its SHA-256).

    Args:
        trainer: The trained trainer.
        path: Destination ``.npz``.
        gate: The candidate gate it was trained with.
        provenance: Training record (data, settings, measured results).

    Returns:
        The sidecar's path (``<name>.provenance.json``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        arrays: dict[str, Any] = _policy_arrays(trainer, gate)
        np.savez(handle, allow_pickle=False, **arrays)
    sidecar = path.with_name(f"{path.stem}.provenance.json")
    record = {
        **provenance,
        "name": path.stem,
        "checkpoint": path.name,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar


def load_alarm_policy(path: str | Path = SHIPPED_POLICY) -> AlarmPolicy:
    """Load a policy archive after verifying it against its provenance sidecar.

    Args:
        path: The ``.npz`` (default: the shipped policy).

    Returns:
        The policy.

    Raises:
        FileNotFoundError: The archive or its sidecar is missing.
        RuntimeError: The archive does not match the sidecar's SHA-256.
        ValueError: Not an alarm policy, or inconsistent arrays.
    """
    path = Path(path)
    sidecar = path.with_name(f"{path.stem}.provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != provenance.get("checkpoint_sha256"):
        raise RuntimeError(
            f"{path} does not match its provenance sidecar (sha256 {actual}); "
            "refusing to load unverifiable weights"
        )
    with np.load(path, allow_pickle=False) as arrays:
        return _from_arrays({k: arrays[k] for k in arrays.files}, provenance)


def nab_features_and_tables(
    streams: Sequence[NABStream], profile: str = "standard"
) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, SweepTable]]:
    """Feature rows and NAB sweep tables for each stream."""
    from omni_mercury_engine.evaluation.nab_scoring import sweep_table

    features = {s.name: stream_features(s.values) for s in streams}
    tables = {s.name: sweep_table(s.values.size, s.windows, profile) for s in streams}
    return features, tables

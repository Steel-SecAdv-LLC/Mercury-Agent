# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proximal Policy Optimization (PPO), implemented natively in PyTorch.

PPO-Clip (Schulman et al., 2017, arXiv:1707.06347) with generalized advantage
estimation (Schulman et al., 2016, arXiv:1506.02438) and the implementation
choices found to matter by Engstrom et al. (2020, arXiv:2005.12729),
Andrychowicz et al. (2021, arXiv:2006.05990) and Huang et al. (2022, "The 37
Implementation Details of Proximal Policy Optimization", ICLR blog track):

* separate policy and value networks with orthogonal initialization (gain
  sqrt(2) on hidden layers, 0.01 on the policy head, 1 on the value head) and
  Adam with eps = 1e-5;
* per-minibatch advantage normalization, global gradient-norm clipping, an
  entropy bonus, optional value-loss clipping, linear learning-rate and
  clip-range annealing, and approximate-KL early stopping;
* value bootstrapping through time-limit truncations (a truncated episode is
  not a terminal state);
* running observation normalization and return-scale reward normalization;
* generalized state-dependent exploration (gSDE; Raffin et al., CoRL 2021,
  arXiv:2005.05719) for continuous actions.

It needs no reinforcement-learning library. Environments come from
:mod:`omni_mercury_engine.ml.rl_envs`: batched NumPy versions of the
classic-control benchmarks (bit-identical to Gymnasium's), or any environment
with the Gymnasium API. Discrete and one-dimensional box action spaces are
supported, over box or discrete observations.

Checkpoints hold only tensors and primitives and load through
:func:`~omni_mercury_engine.security.safe_torch.safe_torch_load` (the
restricted ``weights_only`` unpickler), so loading one cannot run code.
With ``PPOConfig.seed`` set, training is reproducible: parameter
initialization, action sampling, minibatch order and environment resets are
all seeded, and the process-wide random state is left untouched.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from omni_mercury_engine.ml.rl_envs import SyncVectorEnv, VectorEnv, make_vec_env
from omni_mercury_engine.security.safe_torch import safe_torch_load

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ActorCritic",
    "BaseCallback",
    "CheckpointCallback",
    "ConvergenceMonitor",
    "MultiEnvPPOTrainer",
    "PPOConfig",
    "PPOTrainer",
    "RunningMeanStd",
    "TrainingStats",
    "compute_gae",
    "create_multi_env_trainer",
    "create_ppo_trainer",
    "ppo_loss",
]

#: Identifies a checkpoint written by :meth:`PPOTrainer.save_model`.
CHECKPOINT_FORMAT = "mercury.ppo"
CHECKPOINT_VERSION = 1

_SCHEDULES = ("constant", "linear")
_ACTIVATIONS = ("tanh", "relu")
_LOG_2PI = math.log(2.0 * math.pi)
#: Variance floor of the gSDE action distribution (as in gSDE's reference code).
_SDE_EPS = 1e-6
#: Numerical floor inside the running-statistics square roots.
_NORM_EPS = 1e-8
#: Episodes in the moving window behind the reward statistics.
_WINDOW = 100
#: Upper bound on the copies :meth:`PPOTrainer.evaluate` steps together.
_MAX_EVAL_ENVS = 16

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _finite(name: str, value: Any, *, low: float | None = None, strict: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if low is not None and (value <= low if strict else value < low):
        raise ValueError(f"{name} must be {'>' if strict else '>='} {low}, got {value!r}")


@dataclass
class PPOConfig:
    """Hyperparameters for :class:`PPOTrainer`.

    The fields through ``verbose`` are the 2.0.0 configuration, with the same
    defaults: the PPO paper's MuJoCo settings (Schulman et al., 2017, table 3)
    plus an entropy coefficient of 0.01. The remaining fields extend it.

    Attributes:
        learning_rate: Adam step size (initial value when annealed).
        n_steps: Steps collected from each environment copy per rollout.
        batch_size: Minibatch size (a rollout holds ``n_steps * n_envs``).
        n_epochs: Passes over each rollout.
        gamma: Discount factor in [0, 1].
        gae_lambda: GAE bias/variance trade-off in [0, 1].
        clip_range: PPO ratio clipping epsilon (initial value when annealed).
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value-loss coefficient.
        max_grad_norm: Global gradient-norm clipping threshold.
        target_kl: Stop a rollout's epochs once the approximate KL divergence
            of a minibatch exceeds 1.5 times this value (``None``: never).
        verbose: 0 silent, 1 start/end summaries, 2 a line per rollout.
        n_envs: Copies built when the environment is given by name.
        lr_schedule: ``"constant"`` or ``"linear"`` (to 0 over a run).
        clip_range_schedule: ``"constant"`` or ``"linear"``.
        clip_range_vf: Value-prediction clipping around the rollout's values,
            pessimistic as in the original implementation (``None``: off).
        normalize_advantage: Normalize advantages within each minibatch.
        normalize_observations: Running mean/variance observation scaling.
        normalize_rewards: Divide rewards by the running standard deviation
            of the discounted return.
        clip_observations: Bound on normalized observations.
        clip_rewards: Bound on normalized rewards.
        net_arch: Hidden-layer widths, used for both networks (any sequence;
            stored as a tuple).
        activation: ``"tanh"`` or ``"relu"``.
        ortho_init: Orthogonal initialization (see the module docstring).
        log_std_init: Initial log standard deviation (continuous actions).
        use_sde: gSDE exploration (continuous actions only).
        sde_sample_freq: Resample the gSDE noise every this many steps
            (-1: once per rollout).
        seed: Seed for everything random in training (``None``: unseeded).
        device: Torch device for the networks.
    """

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    verbose: int = 1
    n_envs: int = 1
    lr_schedule: str = "constant"
    clip_range_schedule: str = "constant"
    clip_range_vf: float | None = None
    normalize_advantage: bool = True
    normalize_observations: bool = False
    normalize_rewards: bool = False
    clip_observations: float = 10.0
    clip_rewards: float = 10.0
    net_arch: Sequence[int] = (64, 64)
    activation: str = "tanh"
    ortho_init: bool = True
    log_std_init: float = 0.0
    use_sde: bool = False
    sde_sample_freq: int = -1
    seed: int | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            ValueError: A field is out of range or of the wrong kind.
        """
        _finite("learning_rate", self.learning_rate, low=0.0, strict=True)
        for name in ("n_steps", "batch_size", "n_epochs", "n_envs"):
            _positive_int(name, getattr(self, name))
        for name in ("gamma", "gae_lambda"):
            value = getattr(self, name)
            _finite(name, value, low=0.0)
            if value > 1:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        _finite("clip_range", self.clip_range, low=0.0, strict=True)
        _finite("ent_coef", self.ent_coef)
        _finite("vf_coef", self.vf_coef, low=0.0)
        _finite("max_grad_norm", self.max_grad_norm, low=0.0, strict=True)
        for name in ("target_kl", "clip_range_vf"):
            if getattr(self, name) is not None:
                _finite(name, getattr(self, name), low=0.0, strict=True)
        _finite("clip_observations", self.clip_observations, low=0.0, strict=True)
        _finite("clip_rewards", self.clip_rewards, low=0.0, strict=True)
        _finite("log_std_init", self.log_std_init)
        for name in ("lr_schedule", "clip_range_schedule"):
            if getattr(self, name) not in _SCHEDULES:
                raise ValueError(f"{name} must be one of {_SCHEDULES}, got {getattr(self, name)!r}")
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}")
        self.net_arch = tuple(self.net_arch)
        for width in self.net_arch:
            _positive_int("net_arch width", width)
        if isinstance(self.sde_sample_freq, bool) or not isinstance(self.sde_sample_freq, int):
            raise ValueError("sde_sample_freq must be an integer")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0
        ):
            raise ValueError(f"seed must be a non-negative integer or None, got {self.seed!r}")

    def to_dict(self) -> dict[str, Any]:
        """The configuration as primitives (``net_arch`` as a list)."""
        data = dataclasses.asdict(self)
        data["net_arch"] = list(self.net_arch)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PPOConfig:
        """Rebuild a configuration from :meth:`to_dict` output.

        Raises:
            ValueError: An unknown field, or an invalid value.
        """
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown PPOConfig fields: {sorted(unknown)}")
        values = dict(data)
        if "net_arch" in values:
            values["net_arch"] = tuple(values["net_arch"])
        return cls(**values)


@dataclass
class TrainingStats:
    """Cumulative statistics of a trainer.

    Reward statistics cover the most recent 100 completed training episodes,
    in the environment's own (unnormalized) reward. The loss and diagnostic
    fields describe the most recent update.

    Attributes:
        total_timesteps: Environment steps taken in training (all copies).
        total_episodes: Training episodes completed.
        mean_reward: Mean episodic return over the recent window.
        std_reward: Standard deviation of that window.
        mean_episode_length: Mean episode length over the window.
        convergence_metric: The :class:`ConvergenceMonitor` metric, when one ran.
        best_reward: Highest windowed mean return seen at a rollout's end.
        training_time: Wall-clock seconds spent training.
        checkpoints_saved: Checkpoints written.
        n_updates: Gradient steps taken.
        learning_rate: Learning rate of the latest update.
        clip_range: Clip range of the latest update.
        policy_loss: Mean clipped-surrogate loss of the latest update.
        value_loss: Mean value loss of the latest update.
        entropy: Mean policy entropy of the latest update.
        approx_kl: Mean approximate KL divergence of the latest update.
        clip_fraction: Fraction of ratios outside the clip range.
        explained_variance: Of the returns by the value function, per rollout.
        eval_mean_reward: Mean return of the latest :meth:`PPOTrainer.evaluate`.
        eval_std_reward: Its standard deviation.
    """

    total_timesteps: int = 0
    total_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    convergence_metric: float = 0.0
    best_reward: float = float("-inf")
    training_time: float = 0.0
    checkpoints_saved: int = 0
    n_updates: int = 0
    learning_rate: float = 0.0
    clip_range: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    eval_mean_reward: float | None = None
    eval_std_reward: float | None = None


class RunningMeanStd:
    """Running mean and variance, merged batch by batch (Chan et al., 1979).

    Args:
        shape: Shape of one sample.
        epsilon: Initial pseudo-count, keeping early estimates finite.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        """Start at mean 0, variance 1."""
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch: np.ndarray[Any, Any]) -> None:
        """Merge a batch of samples (leading axis = samples)."""
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = (
            self.var * self.count
            + batch.var(axis=0) * batch_count
            + np.square(delta) * self.count * batch_count / total
        )
        self.mean = np.asarray(self.mean + delta * batch_count / total)
        self.var = np.asarray(m2 / total)
        self.count = total

    def state_dict(self) -> dict[str, Any]:
        """Tensors and primitives, for a checkpoint."""
        return {
            "mean": torch.from_numpy(self.mean.copy()),
            "var": torch.from_numpy(self.var.copy()),
            "count": self.count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state_dict` output.

        Raises:
            ValueError: A shape that does not match this instance.
        """
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError(f"running-statistics shape {mean.shape} != {self.mean.shape}")
        self.mean, self.var, self.count = mean, var, float(state["count"])


def compute_gae(
    rewards: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    dones: np.ndarray[Any, Any],
    last_values: np.ndarray[Any, Any],
    gamma: float,
    gae_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Generalized advantage estimates and value targets.

    Args:
        rewards: ``[T, N]`` rewards (time-limit bootstraps already added).
        values: ``[T, N]`` value estimates of the observations acted on.
        dones: ``[T, N]`` whether step ``t`` ended its episode.
        last_values: ``[N]`` value estimates of the observations after step
            ``T - 1``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        ``(advantages, returns)``, both ``[T, N]`` float32, where
        ``returns = advantages + values``.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    not_done = 1.0 - np.asarray(dones, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    next_values = np.asarray(last_values, dtype=np.float32)
    running = np.zeros_like(next_values)
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_values * not_done[t] - values[t]
        running = delta + gamma * gae_lambda * not_done[t] * running
        advantages[t] = running
        next_values = values[t]
    return advantages, advantages + values


def _normal_log_prob(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    z = (actions - mean) / std
    return (-0.5 * z.square() - std.log() - 0.5 * _LOG_2PI).sum(-1)


class ActorCritic(nn.Module):
    """Separate policy and value MLPs over a flat observation vector.

    Discrete actions use a categorical distribution over the policy head's
    logits; continuous actions a diagonal Gaussian with a learned,
    state-independent log standard deviation, or, with ``use_sde``, gSDE:
    the noise is a linear function of the policy's last hidden layer, with
    weights resampled periodically, so exploration is smooth in time.

    The two networks have the same hidden widths, so each hidden layer of
    both is one batched matrix product (weights ``[2, in, out]``: index 0
    policy, 1 value), and every parameter is a view into one flat vector,
    :attr:`flat`. :meth:`ppo_gradients` computes the PPO loss gradient by
    hand straight into ``flat.grad``, about twice as fast as autograd on the
    same network (64-unit layers, one CPU thread); :func:`ppo_loss` is the
    same loss through autograd, and the tests hold the two equal. The state dict is the single tensor
    ``flat``; :attr:`tensors` names its parts.

    Args:
        obs_dim: Observation length.
        action_dim: Number of discrete actions, or the action vector length.
        discrete: Whether actions are discrete.
        net_arch: Hidden-layer widths for both networks.
        activation: ``"tanh"`` or ``"relu"``.
        ortho_init: Orthogonal initialization (gain sqrt(2) hidden, 0.01
            policy head, 1 value head; zero biases); otherwise PyTorch's
            default ``nn.Linear`` initialization.
        log_std_init: Initial log standard deviation (continuous only).
        use_sde: gSDE (continuous only).

    Raises:
        ValueError: gSDE with discrete actions, or an unknown activation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        discrete: bool,
        net_arch: Sequence[int] = (64, 64),
        activation: str = "tanh",
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        use_sde: bool = False,
    ) -> None:
        """Lay out and initialize the parameters."""
        super().__init__()
        if use_sde and discrete:
            raise ValueError("gSDE needs a continuous action space")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}")
        self.discrete = discrete
        self.use_sde = use_sde
        self.activation = activation
        self.obs_dim, self.action_dim = obs_dim, action_dim
        widths = [obs_dim, *net_arch]
        self.n_hidden = len(net_arch)
        self.latent_dim = widths[-1]
        shapes: list[tuple[str, tuple[int, ...]]] = []
        for k, (n_in, n_out) in enumerate(itertools.pairwise(widths)):
            shapes += [
                (f"hidden.{k}.weight", (2, n_in, n_out)),
                (f"hidden.{k}.bias", (2, 1, n_out)),
            ]
        shapes += [
            ("pi.weight", (self.latent_dim, action_dim)),
            ("pi.bias", (action_dim,)),
            ("vf.weight", (self.latent_dim, 1)),
            ("vf.bias", (1,)),
        ]
        if not discrete:
            shapes.append(("log_std", (self.latent_dim, action_dim) if use_sde else (action_dim,)))
        self._shapes = shapes
        self.flat = nn.Parameter(torch.zeros(sum(math.prod(shape) for _, shape in shapes)))
        self.flat.grad = torch.zeros_like(self.flat)
        with torch.no_grad():
            self._initialize(ortho_init, log_std_init)

    # -- parameter layout ---------------------------------------------------

    def _views(self, source: torch.Tensor) -> dict[str, torch.Tensor]:
        views, offset = {}, 0
        for name, shape in self._shapes:
            size = math.prod(shape)
            views[name] = source[offset : offset + size].view(shape)
            offset += size
        return views

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        """Named views of the parameters (no autograd history).

        Derived from ``flat`` on every access, so a move or cast (which
        replaces the storage) can never leave them pointing at the old one.
        """
        return self._views(self.flat.detach())

    def _initialize(self, ortho_init: bool, log_std_init: float) -> None:
        t = self.tensors
        for k in range(self.n_hidden):
            weight, bias = t[f"hidden.{k}.weight"], t[f"hidden.{k}.bias"]
            for tower in range(2):
                self._init_linear(weight[tower], bias[tower], math.sqrt(2.0), ortho_init)
        self._init_linear(t["pi.weight"], t["pi.bias"], 0.01, ortho_init)
        self._init_linear(t["vf.weight"], t["vf.bias"], 1.0, ortho_init)
        if not self.discrete:
            t["log_std"].fill_(float(log_std_init))

    @staticmethod
    def _init_linear(weight: torch.Tensor, bias: torch.Tensor, gain: float, ortho: bool) -> None:
        if ortho:
            nn.init.orthogonal_(weight, gain=gain)
            bias.zero_()
        else:
            # nn.Linear's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both.
            bound = 1.0 / math.sqrt(weight.shape[0])
            nn.init.uniform_(weight, -bound, bound)
            nn.init.uniform_(bias, -bound, bound)

    # -- forward ----------------------------------------------------------

    def _act(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z) if self.activation == "tanh" else torch.relu(z)

    def _forward(
        self, params: dict[str, torch.Tensor], obs: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Both towers; returns (hidden activations per layer, head output, values)."""
        h = obs.unsqueeze(0).expand(2, *obs.shape)
        hidden = [h]
        for k in range(self.n_hidden):
            h = self._act(
                torch.baddbmm(params[f"hidden.{k}.bias"], h, params[f"hidden.{k}.weight"])
            )
            hidden.append(h)
        out = torch.addmm(params["pi.bias"], h[0], params["pi.weight"])
        values = torch.addmm(params["vf.bias"], h[1], params["vf.weight"]).squeeze(-1)
        return hidden, out, values

    def _std(self, log_std: torch.Tensor, latent: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        if self.use_sde:
            return torch.sqrt(latent.detach().square() @ torch.exp(2.0 * log_std) + _SDE_EPS)
        return log_std.exp().expand_as(mean)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """State values, shape ``[B]`` (no autograd history)."""
        return self._forward(self.tensors, obs)[2]

    def sample_sde_noise(self, n: int, generator: torch.Generator | None) -> torch.Tensor:
        """Fresh gSDE exploration matrices, one per environment copy.

        Returns:
            ``[n, latent, action_dim]`` draws from N(0, exp(log_std)^2).
        """
        std = self.tensors["log_std"].exp()
        return std * torch.randn(
            (n, *std.shape), generator=generator, device=std.device, dtype=std.dtype
        )

    def act(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
        sde_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw actions (no autograd history).

        Args:
            obs: ``[B, obs_dim]``.
            deterministic: The mode (argmax / mean) instead of a sample.
            generator: Random source for the sample.
            sde_noise: gSDE matrices from :meth:`sample_sde_noise`, ``[B, ...]``
                (fresh ones are drawn if omitted).

        Returns:
            ``(actions, log_probs, values)``.
        """
        hidden, out, values = self._forward(self.tensors, obs)
        if self.discrete:
            log_p = torch.log_softmax(out, dim=-1)
            if deterministic:
                actions = out.argmax(dim=-1)
            else:
                # Gumbel-max: argmax(logits + Gumbel noise) is an exact sample.
                uniform = torch.rand(out.shape, generator=generator, device=out.device)
                actions = (out - uniform.log_().neg_().log_()).argmax(dim=-1)
            return actions, log_p.gather(-1, actions.unsqueeze(-1)).squeeze(-1), values
        latent = hidden[-1][0]
        std = self._std(self.tensors["log_std"], latent, out)
        if deterministic:
            actions = out
        elif self.use_sde:
            if sde_noise is None:
                sde_noise = self.sample_sde_noise(out.shape[0], generator)
            actions = out + torch.bmm(latent.unsqueeze(1), sde_noise).squeeze(1)
        else:
            noise = torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype)
            actions = out + std * noise
        return actions, _normal_log_prob(actions, out, std), values

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-probabilities, entropies and values, differentiable by autograd.

        Returns:
            ``(log_probs, entropy, values)``, each ``[B]``.
        """
        params = self._views(self.flat)
        hidden, out, values = self._forward(params, obs)
        if self.discrete:
            log_p = torch.log_softmax(out, dim=-1)
            log_prob = log_p.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
            return log_prob, -(log_p.exp() * log_p).sum(-1), values
        std = self._std(params["log_std"], hidden[-1][0], out)
        entropy = (0.5 + 0.5 * _LOG_2PI + std.log()).sum(-1)
        return _normal_log_prob(actions, out, std), entropy, values

    # -- the PPO gradient, by hand ---------------------------------------------

    @torch.no_grad()
    def ppo_gradients(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        *,
        clip_range: float,
        clip_range_vf: float | None,
        ent_coef: float,
        vf_coef: float,
    ) -> torch.Tensor:
        """Write the gradient of :func:`ppo_loss` into ``flat.grad``.

        Every element of ``flat.grad`` is overwritten (no zeroing needed).

        Returns:
            ``[policy_loss, value_loss, entropy, approx_kl, clip_fraction]``.
        """
        if self.flat.grad is None:  # zero_grad(set_to_none=True) dropped the buffer
            self.flat.grad = torch.zeros_like(self.flat)
        # Views of the current buffers: an autograd backward or a move may
        # have replaced either since the last call.
        p, g = self.tensors, self._views(self.flat.grad)
        n = obs.shape[0]
        hidden, out, values = self._forward(p, obs)
        latent_pi, latent_vf = hidden[-1][0], hidden[-1][1]
        # Policy term: d(-min(A r, A clip(r)))/d(log pi) = -A r where the
        # unclipped branch is the active one, else 0.
        if self.discrete:
            log_p = torch.log_softmax(out, dim=-1)
            probs = log_p.exp()
            one_hot = torch.zeros_like(out).scatter_(-1, actions.long().unsqueeze(-1), 1.0)
            log_prob = (log_p * one_hot).sum(-1)
            entropy = -(probs * log_p).sum(-1)
        else:
            std = self._std(p["log_std"], latent_pi, out)
            z = (actions - out) / std
            log_prob = (-0.5 * z.square() - std.log() - 0.5 * _LOG_2PI).sum(-1)
            entropy = (0.5 + 0.5 * _LOG_2PI + std.log()).sum(-1)
        log_ratio = log_prob - old_log_probs
        ratio = log_ratio.exp()
        surrogate = advantages * ratio
        clipped_surrogate = advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
        active = surrogate <= clipped_surrogate
        g_log_prob = surrogate * active * (-1.0 / n)
        g_entropy = -ent_coef / n
        if self.discrete:
            # d log pi(a)/d logits = onehot - p;  d H/d logits = -p (log p + H).
            g_out = g_log_prob.unsqueeze(-1) * (one_hot - probs) - g_entropy * probs * (
                log_p + entropy.unsqueeze(-1)
            )
        else:
            # d log pi/d mean = z / std;  d log pi/d std = (z^2 - 1) / std;
            # d H/d std = 1 / std.
            g_out = g_log_prob.unsqueeze(-1) * z / std
            g_std_scaled = g_log_prob.unsqueeze(-1) * (z.square() - 1.0) + g_entropy
            if self.use_sde:
                # std^2 = latent^2 @ exp(2 s) + eps (latent detached).
                g_variance = g_std_scaled / (2.0 * std.square())
                torch.mul(
                    latent_pi.square().t() @ g_variance,
                    2.0 * torch.exp(2.0 * p["log_std"]),
                    out=g["log_std"],
                )
            else:
                torch.sum(g_std_scaled, dim=0, out=g["log_std"])
        # Value term.
        error = values - returns
        if clip_range_vf is None:
            value_loss = error.square().mean()
            g_values = error * (2.0 * vf_coef / n)
        else:
            delta = values - old_values
            clipped_error = old_values + delta.clamp(-clip_range_vf, clip_range_vf) - returns
            use_clipped = clipped_error.square() > error.square()
            value_loss = torch.where(use_clipped, clipped_error, error).square().mean()
            inside = delta.abs() <= clip_range_vf
            g_values = torch.where(use_clipped, clipped_error * inside, error) * (2.0 * vf_coef / n)
        # Heads.
        torch.mm(latent_pi.t(), g_out, out=g["pi.weight"])
        torch.sum(g_out, dim=0, out=g["pi.bias"])
        g_values = g_values.unsqueeze(-1)
        torch.mm(latent_vf.t(), g_values, out=g["vf.weight"])
        torch.sum(g_values, dim=0, out=g["vf.bias"])
        # Hidden layers, both towers at once.
        if self.n_hidden:
            g_h = torch.stack([g_out @ p["pi.weight"].t(), g_values @ p["vf.weight"].t()])
            for k in range(self.n_hidden - 1, -1, -1):
                h_out = hidden[k + 1]
                if self.activation == "tanh":
                    g_z = torch.addcmul(g_h, g_h * h_out, h_out, value=-1.0)
                else:
                    g_z = g_h * (h_out > 0)
                torch.bmm(hidden[k].transpose(1, 2), g_z, out=g[f"hidden.{k}.weight"])
                torch.sum(g_z, dim=1, keepdim=True, out=g[f"hidden.{k}.bias"])
                if k:
                    g_h = torch.bmm(g_z, p[f"hidden.{k}.weight"].transpose(1, 2))
        policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
        return torch.stack([policy_loss, value_loss, entropy.mean(), approx_kl, clip_fraction])


def ppo_loss(
    policy: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_range: float,
    clip_range_vf: float | None,
    ent_coef: float,
    vf_coef: float,
) -> torch.Tensor:
    """The PPO loss through autograd (reference for :meth:`ActorCritic.ppo_gradients`).

    ``-mean(min(A r, A clip(r, 1 - eps, 1 + eps))) - ent_coef * mean(H)
    + vf_coef * mean(value error^2)``, the value error optionally clipped
    pessimistically around ``old_values``. ``advantages`` are used as given
    (normalize them first if wanted).
    """
    log_prob, entropy, values = policy.evaluate_actions(obs, actions)
    log_ratio = log_prob - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = -torch.min(
        advantages * ratio, advantages * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    ).mean()
    if clip_range_vf is None:
        value_loss = (returns - values).square().mean()
    else:
        clipped = old_values + (values - old_values).clamp(-clip_range_vf, clip_range_vf)
        value_loss = torch.max((returns - values).square(), (returns - clipped).square()).mean()
    return policy_loss - ent_coef * entropy.mean() + vf_coef * value_loss


class BaseCallback:
    """Hook into :meth:`PPOTrainer.learn`.

    Subclasses override ``_on_step`` (return ``False`` to stop training) and
    optionally ``_on_training_start``, ``_on_rollout_end`` (called after each
    rollout, before its update) and ``_on_training_end``.

    Attributes:
        verbose: Verbosity level.
        locals: For the current step: ``episode`` (a list of ``{"r": return,
            "l": length}`` for episodes that finished), ``rewards``, ``dones``
            and ``infos``.
        n_calls: Steps seen (one per vector-environment step).
        num_timesteps: The trainer's environment-step count.
        model: The trainer.
    """

    def __init__(self, verbose: int = 0) -> None:
        """Initialize the instance."""
        self.verbose = verbose
        self.locals: dict[str, Any] = {}
        self.n_calls = 0
        self.num_timesteps = 0
        self.model: Any = None

    def init_callback(self, model: Any) -> None:
        """Attach to a trainer at the start of :meth:`PPOTrainer.learn`."""
        self.model = model
        self.num_timesteps = model.num_timesteps
        self._on_training_start()

    def on_step(self, locals_: dict[str, Any]) -> bool:
        """Record a step; ``False`` stops training."""
        self.locals = locals_
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()

    def on_rollout_end(self) -> None:
        """Called once a rollout is collected, before the update on it."""
        self._on_rollout_end()

    def on_training_end(self) -> None:
        """Called when :meth:`PPOTrainer.learn` returns."""
        self._on_training_end()

    def _on_training_start(self) -> None:
        return None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        return None

    def _on_training_end(self) -> None:
        return None


class ConvergenceMonitor(BaseCallback):
    """Stop training once episodic return plateaus.

    After each completed episode (once ``window`` have completed), the mean
    return of the last ``window`` episodes is compared with the best such
    mean so far. It counts as an improvement only if it beats the best by
    more than ``(1 - convergence_threshold) * |best|``; ``patience``
    consecutive episodes without one mark the run converged, and training
    stops.

    Args:
        convergence_threshold: In (0, 1]; 0.999 tolerates changes under 0.1%.
        patience: Episodes without improvement before stopping.
        verbose: Verbosity level.
        window: Episodes in the moving mean.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.999,
        patience: int = 10,
        verbose: int = 0,
        window: int = _WINDOW,
    ) -> None:
        """Initialize the monitor.

        Raises:
            ValueError: ``convergence_threshold`` outside (0, 1], or a
                non-positive ``patience`` or ``window``.
        """
        super().__init__(verbose)
        _finite("convergence_threshold", convergence_threshold, low=0.0, strict=True)
        if convergence_threshold > 1:
            raise ValueError("convergence_threshold must be in (0, 1]")
        _positive_int("patience", patience)
        _positive_int("window", window)
        self.convergence_threshold = float(convergence_threshold)
        self.patience = patience
        self.window = window
        self.reward_history: list[float] = []
        self.best_mean_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.converged = False

    def _on_step(self) -> bool:
        for episode in self.locals.get("episode", ()):
            self.reward_history.append(float(episode["r"]))
            if len(self.reward_history) < self.window:
                continue
            mean = float(np.mean(self.reward_history[-self.window :]))
            tolerance = (1.0 - self.convergence_threshold) * abs(self.best_mean_reward)
            if not math.isfinite(self.best_mean_reward) or mean > self.best_mean_reward + tolerance:
                self.best_mean_reward = mean
                self.episodes_without_improvement = 0
            else:
                self.episodes_without_improvement += 1
            if self.episodes_without_improvement >= self.patience:
                self.converged = True
                if self.verbose:
                    logger.info("return plateaued at %.4f; stopping", mean)
                return False
        return True

    def get_convergence_metric(self) -> float:
        """How close the windowed mean is to the best, in [0, 1].

        ``1 - (best - mean) / |best|``, clipped to [0, 1]; 0 until ``window``
        episodes have completed.
        """
        if len(self.reward_history) < self.window:
            return 0.0
        mean = float(np.mean(self.reward_history[-self.window :]))
        scale = max(abs(self.best_mean_reward), 1e-8)
        return float(np.clip(1.0 - (self.best_mean_reward - mean) / scale, 0.0, 1.0))


class CheckpointCallback(BaseCallback):
    """Save periodic and best checkpoints during training.

    Writes ``<name_prefix>_step_<timesteps>.pt`` each time the trainer's step
    count crosses a multiple of ``save_freq``, and ``<name_prefix>_best.pt``
    whenever the windowed mean training return beats its best so far
    (checked after each rollout, before the policy that earned it is
    updated).

    Args:
        save_path: Directory for checkpoints (created if missing).
        save_freq: Environment steps between periodic checkpoints.
        name_prefix: File-name prefix.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_path: str | os.PathLike[str],
        save_freq: int = 10000,
        name_prefix: str = "ppo_model",
        verbose: int = 0,
    ) -> None:
        """Initialize the callback and create ``save_path``."""
        super().__init__(verbose)
        _positive_int("save_freq", save_freq)
        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.best_mean_reward = float("-inf")
        self.checkpoints_saved = 0
        self._periods_saved = 0

    def _on_training_start(self) -> None:
        self._periods_saved = self.num_timesteps // self.save_freq

    def _save(self, path: Path) -> None:
        self.model.save_model(path)
        self.checkpoints_saved += 1
        if self.verbose:
            logger.info("saved checkpoint %s", path)

    def _on_step(self) -> bool:
        periods = self.num_timesteps // self.save_freq
        if periods > self._periods_saved:
            self._periods_saved = periods
            self._save(self.save_path / f"{self.name_prefix}_step_{self.num_timesteps}.pt")
        return True

    def _on_rollout_end(self) -> None:
        returns = self.model.episode_returns
        if returns:
            mean = float(np.mean(returns))
            if mean > self.best_mean_reward:
                self.best_mean_reward = mean
                self._save(self.save_path / f"{self.name_prefix}_best.pt")


@dataclass
class _Rollout:
    """One rollout: ``[T, N, ...]`` arrays plus the values after its last step."""

    obs: FloatArray
    actions: np.ndarray[Any, Any]
    log_probs: FloatArray
    values: FloatArray
    rewards: FloatArray
    dones: np.ndarray[Any, Any]
    last_values: FloatArray
    episodes: int


def _space_kind(space: Any) -> str:
    """``"discrete"`` for a finite space, ``"box"`` for a real box."""
    if hasattr(space, "n") and tuple(getattr(space, "shape", ()) or ()) == ():
        return "discrete"
    if hasattr(space, "low") and hasattr(space, "high"):
        return "box"
    raise TypeError(f"unsupported space {space!r}: PPO needs a Discrete or Box space")


def _as_vector_env(env: Any, n_envs: int, seed: int | None) -> Any:
    if isinstance(env, str):
        return make_vec_env(env, n_envs, seed=seed)
    if isinstance(env, VectorEnv):
        # A Gymnasium vector environment resets on the *next* step by default,
        # which would feed the learner a transition that never happened.
        mode = getattr(env, "metadata", {}).get("autoreset_mode")
        if mode is not None and getattr(mode, "value", mode) != "SameStep":
            raise ValueError(
                "a Gymnasium vector environment needs autoreset_mode=AutoresetMode.SAME_STEP"
            )
        return env
    if hasattr(env, "reset") and hasattr(env, "step"):
        return SyncVectorEnv([env])
    if callable(env):
        return make_vec_env(env, n_envs, seed=seed)
    raise TypeError(f"{env!r} is neither an environment nor a registered environment name")


class PPOTrainer:
    """Train a policy with PPO.

    Example:
        trainer = PPOTrainer("CartPole-v1", PPOConfig(n_envs=8, seed=0))
        stats = trainer.learn(total_timesteps=100_000)
        mean_reward, std_reward = trainer.evaluate(num_episodes=10)

    Args:
        env: An environment name for
            :func:`~omni_mercury_engine.ml.rl_envs.make_vec_env` (built with
            ``config.n_envs`` copies), a zero-argument factory of one
            Gymnasium-API environment (likewise), a vector environment
            (Gymnasium's with ``autoreset_mode=SAME_STEP``), or one
            Gymnasium-API environment.
        config: Hyperparameters (defaults: :class:`PPOConfig`).
        checkpoint_dir: Where :meth:`pretrain` writes checkpoints.
        tensorboard_log: Directory for TensorBoard scalars (needs the
            ``tensorboard`` package), or ``None``.

    Raises:
        TypeError: An unsupported environment or space.
        ValueError: ``batch_size`` larger than a rollout, gSDE requested for
            discrete actions, or a next-step-autoreset vector environment.
        ImportError: ``tensorboard_log`` given without TensorBoard installed.
    """

    def __init__(
        self,
        env: Any,
        config: PPOConfig | None = None,
        checkpoint_dir: str | os.PathLike[str] = "./checkpoints",
        tensorboard_log: str | os.PathLike[str] | None = None,
    ) -> None:
        """Build the environment, networks and optimizer."""
        self.config = config or PPOConfig()
        self.env = _as_vector_env(env, self.config.n_envs, self.config.seed)
        self.num_envs = int(self.env.num_envs)
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_log = tensorboard_log
        self._writer: Any = None
        if tensorboard_log is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "tensorboard_log needs the 'tensorboard' package (pip install tensorboard)"
                ) from exc
            self._writer = SummaryWriter(log_dir=str(tensorboard_log))
        self._read_spaces()
        self.device = torch.device(self.config.device)
        self.stats = TrainingStats()
        self.num_timesteps = 0
        self.convergence_monitor: ConvergenceMonitor | None = None
        self.checkpoint_callback: CheckpointCallback | None = None
        self._build()
        self._last_obs: FloatArray | None = None
        self._reset_seed = self.config.seed
        self._ep_returns: deque[float] = deque(maxlen=_WINDOW)
        self._ep_lengths: deque[int] = deque(maxlen=_WINDOW)
        if self.config.verbose:
            logger.info(
                "PPO trainer: %d %s, %s actions, %d parameters",
                self.num_envs,
                "copy" if self.num_envs == 1 else "copies",
                self._action_kind,
                sum(p.numel() for p in self.policy.parameters()),
            )

    # -- construction -----------------------------------------------------

    def _read_spaces(self) -> None:
        # Gymnasium vector environments name the per-copy spaces single_*.
        obs_space = getattr(self.env, "single_observation_space", self.env.observation_space)
        self._obs_kind = _space_kind(obs_space)
        self._obs_shape: tuple[int, ...] = ()
        if self._obs_kind == "discrete":
            self._obs_n = int(obs_space.n)
            self._obs_start = int(getattr(obs_space, "start", 0))
            self.obs_dim = self._obs_n
        else:
            self._obs_shape = tuple(obs_space.shape)
            self.obs_dim = math.prod(self._obs_shape)
        act_space = getattr(self.env, "single_action_space", self.env.action_space)
        self._action_kind = _space_kind(act_space)
        if self._action_kind == "discrete":
            self.action_dim = int(act_space.n)
            self._action_start = int(getattr(act_space, "start", 0))
        else:
            if len(act_space.shape) != 1:
                raise TypeError(f"box actions must be one-dimensional, got shape {act_space.shape}")
            self.action_dim = int(act_space.shape[0])
            self._low = np.asarray(act_space.low, dtype=np.float32)
            self._high = np.asarray(act_space.high, dtype=np.float32)
        self._check_rollout_size()

    def _check_rollout_size(self) -> None:
        rollout = self.config.n_steps * self.num_envs
        if self.config.batch_size > rollout:
            raise ValueError(
                f"batch_size {self.config.batch_size} exceeds the rollout size "
                f"n_steps * num_envs = {rollout}"
            )

    def _build(self) -> None:
        cfg = self.config
        seed = (
            cfg.seed if cfg.seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.policy = ActorCritic(
                self.obs_dim,
                self.action_dim,
                discrete=self._action_kind == "discrete",
                net_arch=cfg.net_arch,
                activation=cfg.activation,
                ortho_init=cfg.ortho_init,
                log_std_init=cfg.log_std_init,
                use_sde=cfg.use_sde,
            ).to(self.device)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed + 1)
        # One flat parameter vector: Adam (eps 1e-5) is elementwise, so this is
        # the per-tensor update in a single fused kernel.
        self.optimizer = torch.optim.Adam(
            [self.policy.flat], lr=cfg.learning_rate, eps=1e-5, fused=True
        )
        self.obs_rms = RunningMeanStd((self.obs_dim,)) if cfg.normalize_observations else None
        self.ret_rms = RunningMeanStd(()) if cfg.normalize_rewards else None
        self._discounted = np.zeros(self.num_envs, dtype=np.float64)

    # -- observation and reward processing ----------------------------------

    def _flatten(self, raw: Any) -> FloatArray:
        raw = np.asarray(raw)
        if self._obs_kind == "discrete":
            index = raw.reshape(-1).astype(np.int64) - self._obs_start
            return np.eye(self._obs_n, dtype=np.float32)[index]
        return raw.reshape(raw.shape[0], -1).astype(np.float32, copy=False)

    def _process_obs(self, raw: Any, *, update: bool) -> FloatArray:
        obs = self._flatten(raw)
        if self.obs_rms is None:
            return obs
        if update:
            self.obs_rms.update(obs)
        limit = self.config.clip_observations
        scaled = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + _NORM_EPS)
        return np.clip(scaled, -limit, limit).astype(np.float32)

    def _scale_rewards(
        self, rewards: np.ndarray[Any, Any], dones: np.ndarray[Any, Any]
    ) -> FloatArray:
        if self.ret_rms is None:
            return rewards.astype(np.float32)
        self._discounted = self._discounted * self.config.gamma + rewards
        self.ret_rms.update(self._discounted)
        limit = self.config.clip_rewards
        scaled = np.clip(rewards / np.sqrt(self.ret_rms.var + _NORM_EPS), -limit, limit)
        self._discounted[dones] = 0.0
        return scaled.astype(np.float32)

    def _to_env_actions(self, actions: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self._action_kind == "discrete":
            return actions + self._action_start if self._action_start else actions
        return np.clip(actions, self._low, self._high)

    @staticmethod
    def _final_rows(info: dict[str, Any], rows: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Pre-reset observations of ``rows`` (``final_observation`` or ``final_obs``)."""
        final = info.get("final_observation", info.get("final_obs"))
        if final is None:
            raise KeyError("a truncated step must report info['final_observation']")
        return np.stack([np.asarray(final[i]) for i in np.flatnonzero(rows)])

    # -- rollouts -----------------------------------------------------------

    def _reset_env(self) -> None:
        obs, _ = self.env.reset(seed=self._reset_seed)
        self._reset_seed = None
        self._ep_return = np.zeros(self.num_envs, dtype=np.float64)
        self._ep_length = np.zeros(self.num_envs, dtype=np.int64)
        self._discounted = np.zeros(self.num_envs, dtype=np.float64)
        self._last_obs = self._process_obs(obs, update=True)

    def _collect(
        self,
        n_steps: int,
        callbacks: Sequence[BaseCallback],
        min_episodes: int | None = None,
    ) -> _Rollout | None:
        """Collect ``n_steps`` per copy (fewer once ``min_episodes`` finish).

        Returns ``None`` when a callback stops training.
        """
        if self._last_obs is None:
            self._reset_env()
        cfg, policy, device, gen = self.config, self.policy, self.device, self._generator
        obs = self._last_obs
        assert obs is not None
        sde = policy.use_sde
        noise = policy.sample_sde_noise(self.num_envs, gen) if sde else None
        buffers: tuple[list[Any], ...] = ([], [], [], [], [], [])
        b_obs, b_act, b_logp, b_val, b_rew, b_done = buffers
        episodes = 0
        for step in range(n_steps):
            if sde and cfg.sde_sample_freq > 0 and step > 0 and step % cfg.sde_sample_freq == 0:
                noise = policy.sample_sde_noise(self.num_envs, gen)
            with torch.inference_mode():
                actions_t, log_probs_t, values_t = policy.act(
                    torch.as_tensor(obs, device=device),
                    deterministic=False,
                    generator=gen,
                    sde_noise=noise,
                )
            actions = actions_t.cpu().numpy()
            raw, rewards, terminated, truncated, info = self.env.step(self._to_env_actions(actions))
            rewards = np.asarray(rewards, dtype=np.float64)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            dones = terminated | truncated
            self.num_timesteps += self.num_envs
            self._ep_return += rewards
            self._ep_length += 1
            finished = []
            if dones.any():
                for i in np.flatnonzero(dones):
                    finished.append({"r": float(self._ep_return[i]), "l": int(self._ep_length[i])})
                    self._ep_returns.append(float(self._ep_return[i]))
                    self._ep_lengths.append(int(self._ep_length[i]))
                self._ep_return[dones] = 0.0
                self._ep_length[dones] = 0
                episodes += len(finished)
                self.stats.total_episodes += len(finished)
            step_rewards = self._scale_rewards(rewards, dones)
            next_obs = self._process_obs(raw, update=True)
            bootstrap = truncated & ~terminated
            if bootstrap.any():
                final = self._process_obs(self._final_rows(info, bootstrap), update=False)
                with torch.inference_mode():
                    final_values = policy.value(torch.as_tensor(final, device=device)).cpu().numpy()
                step_rewards[bootstrap] += cfg.gamma * final_values
            b_obs.append(obs)
            b_act.append(actions)
            b_logp.append(log_probs_t.cpu().numpy())
            b_val.append(values_t.cpu().numpy())
            b_rew.append(step_rewards)
            b_done.append(dones)
            obs = next_obs
            if callbacks:
                step_locals = {
                    "episode": finished,
                    "rewards": rewards,
                    "dones": dones,
                    "infos": info,
                }
                # Every callback sees every step, even after one says stop.
                keep_going = [cb.on_step(step_locals) for cb in callbacks]
                if not all(keep_going):
                    self._last_obs = obs
                    return None
            if min_episodes is not None and episodes >= min_episodes:
                break
        self._last_obs = obs
        with torch.inference_mode():
            last_values = policy.value(torch.as_tensor(obs, device=device)).cpu().numpy()
        return _Rollout(
            obs=np.stack(b_obs),
            actions=np.stack(b_act),
            log_probs=np.stack(b_logp),
            values=np.stack(b_val),
            rewards=np.stack(b_rew),
            dones=np.stack(b_done),
            last_values=last_values,
            episodes=episodes,
        )

    # -- the update -----------------------------------------------------------

    def _update(self, rollout: _Rollout, progress_remaining: float) -> None:
        cfg, device = self.config, self.device
        progress = min(max(progress_remaining, 0.0), 1.0)
        lr = cfg.learning_rate * (progress if cfg.lr_schedule == "linear" else 1.0)
        clip = cfg.clip_range * (progress if cfg.clip_range_schedule == "linear" else 1.0)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        advantages, returns = compute_gae(
            rollout.rewards,
            rollout.values,
            rollout.dones,
            rollout.last_values,
            cfg.gamma,
            cfg.gae_lambda,
        )

        def merged(array: np.ndarray[Any, Any]) -> torch.Tensor:
            """``[T, N, ...]`` -> ``[T * N, ...]`` on the device."""
            return torch.as_tensor(array.reshape(-1, *array.shape[2:]), device=device)

        obs, actions = merged(rollout.obs), merged(rollout.actions)
        old_log_probs, old_values = merged(rollout.log_probs), merged(rollout.values)
        b_adv, b_ret = merged(advantages), merged(returns)
        total = obs.shape[0]
        batch = min(cfg.batch_size, total)
        params = self.policy.flat
        history: list[torch.Tensor] = []
        stop = False
        for _ in range(cfg.n_epochs):
            if batch == total:
                # One minibatch is the whole rollout: order cannot matter.
                batches: Iterable[torch.Tensor | None] = (None,)
            else:
                perm = torch.randperm(total, generator=self._generator, device=device)
                batches = [perm[start : start + batch] for start in range(0, total, batch)]
            for idx in batches:
                tensors = (obs, actions, old_log_probs, old_values, b_adv, b_ret)
                mb = tensors if idx is None else tuple(t[idx] for t in tensors)
                mb_obs, mb_act, mb_old_lp, mb_old_v, mb_adv, mb_ret = mb
                if cfg.normalize_advantage and mb_adv.shape[0] > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                metrics = self.policy.ppo_gradients(
                    mb_obs,
                    mb_act,
                    mb_old_lp,
                    mb_old_v,
                    mb_adv,
                    mb_ret,
                    clip_range=clip,
                    clip_range_vf=cfg.clip_range_vf,
                    ent_coef=cfg.ent_coef,
                    vf_coef=cfg.vf_coef,
                )
                history.append(metrics)
                if cfg.target_kl is not None and float(metrics[3]) > 1.5 * cfg.target_kl:
                    stop = True
                    break
                grad = params.grad
                assert grad is not None
                # Global-norm clipping, as torch.nn.utils.clip_grad_norm_.
                norm = torch.linalg.vector_norm(grad)
                grad.mul_(torch.clamp(cfg.max_grad_norm / (norm + 1e-6), max=1.0))
                self.optimizer.step()
                self.stats.n_updates += 1
            if stop:
                break
        ret_var = float(np.var(returns))
        explained = (
            float(1.0 - np.var(returns - rollout.values) / ret_var) if ret_var > 0 else float("nan")
        )
        means = torch.stack(history).mean(0).tolist()
        stats = self.stats
        stats.learning_rate, stats.clip_range = lr, clip
        stats.policy_loss, stats.value_loss, stats.entropy = means[0], means[1], means[2]
        stats.approx_kl, stats.clip_fraction = means[3], means[4]
        stats.explained_variance = explained

    # -- training entry points -------------------------------------------------

    @property
    def episode_returns(self) -> list[float]:
        """Returns of the most recent (up to 100) training episodes."""
        return list(self._ep_returns)

    def _refresh_reward_stats(self, callbacks: Sequence[BaseCallback]) -> None:
        stats = self.stats
        stats.total_timesteps = self.num_timesteps
        if self._ep_returns:
            stats.mean_reward = float(np.mean(self._ep_returns))
            stats.std_reward = float(np.std(self._ep_returns))
            stats.mean_episode_length = float(np.mean(self._ep_lengths))
            stats.best_reward = max(stats.best_reward, stats.mean_reward)
        for cb in callbacks:
            if isinstance(cb, ConvergenceMonitor):
                stats.convergence_metric = cb.get_convergence_metric()

    def _log_rollout(self) -> None:
        stats = self.stats
        if self.config.verbose >= 2:
            logger.info(
                "steps=%d episodes=%d mean_reward=%.3f kl=%.5f clip_frac=%.3f ev=%.3f",
                self.num_timesteps,
                stats.total_episodes,
                stats.mean_reward,
                stats.approx_kl,
                stats.clip_fraction,
                stats.explained_variance,
            )
        if self._writer is not None:
            for name in (
                "mean_reward",
                "policy_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "explained_variance",
                "learning_rate",
            ):
                self._writer.add_scalar(f"ppo/{name}", getattr(stats, name), self.num_timesteps)

    def _train(
        self,
        callback: BaseCallback | Sequence[BaseCallback] | None,
        *,
        steps: int | None = None,
        episodes: tuple[int, int] | None = None,
    ) -> TrainingStats:
        """Train for ``steps`` steps, or for ``episodes = (budget, update_freq)``."""
        callbacks = [callback] if isinstance(callback, BaseCallback) else list(callback or ())
        for cb in callbacks:
            cb.init_callback(self)
        start_steps, start_episodes = self.num_timesteps, self.stats.total_episodes
        started = time.perf_counter()
        try:
            while True:
                if episodes is None:
                    assert steps is not None
                    if self.num_timesteps - start_steps >= steps:
                        break
                    wanted = None
                else:
                    budget, update_freq = episodes
                    remaining = budget - (self.stats.total_episodes - start_episodes)
                    if remaining <= 0:
                        break
                    wanted = min(update_freq, remaining)
                rollout = self._collect(self.config.n_steps, callbacks, min_episodes=wanted)
                self._refresh_reward_stats(callbacks)
                if rollout is None:
                    break
                for cb in callbacks:
                    cb.on_rollout_end()
                if episodes is None:
                    assert steps is not None
                    progress = 1.0 - (self.num_timesteps - start_steps) / steps
                else:
                    progress = 1.0 - (self.stats.total_episodes - start_episodes) / budget
                self._update(rollout, progress)
                self._log_rollout()
        finally:
            self.stats.training_time += time.perf_counter() - started
            self._refresh_reward_stats(callbacks)
            for cb in callbacks:
                cb.on_training_end()
        if self.config.verbose:
            logger.info(
                "PPO trained %d steps (%d episodes) in %.2fs; mean_reward=%.4f",
                self.num_timesteps - start_steps,
                self.stats.total_episodes - start_episodes,
                time.perf_counter() - started,
                self.stats.mean_reward,
            )
        return self.stats

    def learn(
        self,
        total_timesteps: int,
        callback: BaseCallback | Sequence[BaseCallback] | None = None,
    ) -> TrainingStats:
        """Train for ``total_timesteps`` environment steps (all copies).

        Training runs in whole rollouts, so it may overshoot by up to one
        rollout. Linear schedules anneal over this call's budget. Calling
        again continues from the current policy, optimizer and environments.

        Args:
            total_timesteps: Step budget (>= 1).
            callback: One callback or several.

        Returns:
            :attr:`stats`.
        """
        _positive_int("total_timesteps", total_timesteps)
        return self._train(callback, steps=total_timesteps)

    def pretrain(
        self,
        total_timesteps: int = 100000,
        convergence_threshold: float = 0.999,
        save_checkpoints: bool = True,
        *,
        patience: int = 10,
    ) -> TrainingStats:
        """Train until ``total_timesteps`` or until return plateaus.

        Runs :meth:`learn` with a :class:`ConvergenceMonitor` and, when
        ``save_checkpoints``, a :class:`CheckpointCallback` writing
        ``ppo_pretrain_*`` files every 10,000 steps into ``checkpoint_dir``.

        Args:
            total_timesteps: Step budget.
            convergence_threshold: See :class:`ConvergenceMonitor`.
            save_checkpoints: Write checkpoints.
            patience: See :class:`ConvergenceMonitor`.

        Returns:
            :attr:`stats`.
        """
        verbose = 1 if self.config.verbose else 0
        self.convergence_monitor = ConvergenceMonitor(convergence_threshold, patience, verbose)
        callbacks: list[BaseCallback] = [self.convergence_monitor]
        if save_checkpoints:
            self.checkpoint_callback = CheckpointCallback(
                self.checkpoint_dir, save_freq=10000, name_prefix="ppo_pretrain", verbose=verbose
            )
            callbacks.append(self.checkpoint_callback)
        return self.learn(total_timesteps, callback=callbacks)

    def train_online(
        self,
        num_episodes: int = 100,
        update_freq: int = 10,
        *,
        callback: BaseCallback | Sequence[BaseCallback] | None = None,
    ) -> TrainingStats:
        """Train for ``num_episodes`` episodes, updating every ``update_freq``.

        Each update uses the experience of the last ``update_freq`` completed
        episodes, or of ``n_steps`` steps per copy if those episodes run
        longer. Linear schedules anneal over the episode budget. Episodes
        must end: an environment that never terminates or truncates keeps
        training without completing the budget.

        Args:
            num_episodes: Episodes to train for (>= 1).
            update_freq: Completed episodes between policy updates (>= 1).
            callback: One callback or several.

        Returns:
            :attr:`stats`.
        """
        _positive_int("num_episodes", num_episodes)
        _positive_int("update_freq", update_freq)
        return self._train(callback, episodes=(num_episodes, update_freq))

    # -- inference and evaluation ---------------------------------------------

    def predict(self, observation: Any, deterministic: bool = True) -> np.ndarray[Any, Any]:
        """Actions for one observation or a batch, in the environment's terms.

        Observations are normalized with the frozen training statistics;
        continuous actions are clipped to the action bounds.

        Args:
            observation: One observation, or a batch along the first axis.
            deterministic: The policy's mode instead of a sample.

        Returns:
            One action, or a batch.
        """
        raw = np.asarray(observation)
        single = raw.ndim == len(self._obs_shape)
        batch = raw[None] if single else raw
        obs = self._process_obs(batch, update=False)
        with torch.inference_mode():
            actions_t, _, _ = self.policy.act(
                torch.as_tensor(obs, device=self.device),
                deterministic=deterministic,
                generator=self._generator,
            )
        actions = self._to_env_actions(actions_t.cpu().numpy())
        return actions[0] if single else actions

    def _evaluation_env(self, num_episodes: int, seed: int | None) -> tuple[Any, bool]:
        """A fresh copy of the training environment when one can be made."""
        spawnable = not (isinstance(self.env, SyncVectorEnv) and self.env.factory is None)
        spawn = getattr(self.env, "spawn", None)
        if spawnable and callable(spawn):
            return spawn(min(num_episodes, _MAX_EVAL_ENVS), seed=seed), False
        return self.env, True

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        *,
        env: Any = None,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Mean and standard deviation of episodic return.

        Runs on ``env`` if given, else on a fresh copy of the training
        environment (or, when none can be made, on the training environment
        itself, which the next training call then resets). Episodes are
        split evenly across the copies, so short episodes are not
        over-represented.

        Args:
            num_episodes: Episodes to run (>= 1).
            deterministic: The policy's mode instead of samples.
            env: A vector environment, or one Gymnasium-API environment.
            seed: Reset seed (default: derived from ``PPOConfig.seed``).

        Returns:
            ``(mean_reward, std_reward)``, also stored in :attr:`stats`.
        """
        _positive_int("num_episodes", num_episodes)
        if seed is None and self.config.seed is not None:
            seed = self.config.seed + 10_000
        if env is not None:
            eval_env, shared = _as_vector_env(env, 1, seed), False
        else:
            eval_env, shared = self._evaluation_env(num_episodes, seed)
        n = int(eval_env.num_envs)
        targets = np.array([(num_episodes + i) // n for i in range(n)])
        counts = np.zeros(n, dtype=np.int64)
        running = np.zeros(n, dtype=np.float64)
        returns: list[float] = []
        raw, _ = eval_env.reset(seed=seed)
        while (counts < targets).any():
            actions = self.predict(raw, deterministic=deterministic)
            raw, rewards, terminated, truncated, _ = eval_env.step(actions)
            running += np.asarray(rewards, dtype=np.float64)
            for i in np.flatnonzero(np.asarray(terminated) | np.asarray(truncated)):
                if counts[i] < targets[i]:
                    returns.append(float(running[i]))
                    counts[i] += 1
                running[i] = 0.0
        if shared:
            self._last_obs = None
        mean, std = float(np.mean(returns)), float(np.std(returns))
        self.stats.eval_mean_reward, self.stats.eval_std_reward = mean, std
        if self.config.verbose:
            logger.info("evaluation over %d episodes: %.4f +/- %.4f", num_episodes, mean, std)
        return mean, std

    # -- persistence -------------------------------------------------------------

    def _action_spec(self) -> dict[str, Any]:
        if self._action_kind == "discrete":
            return {"kind": "discrete", "n": self.action_dim, "start": self._action_start}
        return {
            "kind": "box",
            "low": torch.from_numpy(self._low.copy()),
            "high": torch.from_numpy(self._high.copy()),
        }

    def save_model(self, path: str | os.PathLike[str]) -> None:
        """Write a checkpoint (atomically: a crash leaves the old file intact).

        It holds the configuration, network and optimizer state, normalization
        statistics, step count and sampling-generator state, all as tensors
        and primitives.

        Args:
            path: Destination file; parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "config": self.config.to_dict(),
            "obs_dim": self.obs_dim,
            "action": self._action_spec(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": None if self.obs_rms is None else self.obs_rms.state_dict(),
            "ret_rms": None if self.ret_rms is None else self.ret_rms.state_dict(),
            "num_timesteps": self.num_timesteps,
            "generator": self._generator.get_state(),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(checkpoint, handle)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.stats.checkpoints_saved += 1
        if self.config.verbose:
            logger.info("PPO checkpoint saved: %s", path)

    def load_model(self, path: str | os.PathLike[str]) -> None:
        """Restore a checkpoint written by :meth:`save_model`.

        The checkpoint's configuration replaces this trainer's (so its
        network shape is rebuilt), and training can continue from it.

        Args:
            path: The checkpoint file.

        Raises:
            ValueError: Not a PPO checkpoint, or one made for a different
                observation or action space.
            UnsafeCheckpointError: The file is not a plain-data checkpoint.
        """
        data = safe_torch_load(path, map_location=self.device)
        if not isinstance(data, dict) or data.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a Mercury PPO checkpoint")
        if data.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"unsupported PPO checkpoint version {data.get('version')!r}")
        if data["obs_dim"] != self.obs_dim or not _same_action_spec(
            data["action"], self._action_spec()
        ):
            raise ValueError("checkpoint was made for a different observation or action space")
        config = PPOConfig.from_dict(data["config"])
        config.device = self.config.device
        self.config = config
        self._check_rollout_size()
        self._build()
        self.policy.load_state_dict(data["policy"])
        self.optimizer.load_state_dict(data["optimizer"])
        # The rebuilt trainer has the checkpoint's configuration, so it keeps
        # running statistics exactly when the checkpoint holds them.
        if self.obs_rms is not None:
            self.obs_rms.load_state_dict(data["obs_rms"])
        if self.ret_rms is not None:
            self.ret_rms.load_state_dict(data["ret_rms"])
        self._generator.set_state(data["generator"].cpu())
        self.num_timesteps = self.stats.total_timesteps = int(data["num_timesteps"])
        self._last_obs = None
        if self.config.verbose:
            logger.info("PPO checkpoint loaded: %s", path)

    def get_training_summary(self) -> dict[str, Any]:
        """Training statistics as a flat dictionary."""
        stats = self.stats
        return {
            "total_timesteps": stats.total_timesteps,
            "total_episodes": stats.total_episodes,
            "mean_reward": stats.mean_reward,
            "std_reward": stats.std_reward,
            "best_reward": stats.best_reward,
            "convergence_metric": stats.convergence_metric,
            "training_time": stats.training_time,
            "checkpoints_saved": stats.checkpoints_saved,
            "converged": bool(self.convergence_monitor and self.convergence_monitor.converged),
            "n_updates": stats.n_updates,
            "eval_mean_reward": stats.eval_mean_reward,
            "eval_std_reward": stats.eval_std_reward,
        }


def _same_action_spec(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["kind"] != b["kind"]:
        return False
    if a["kind"] == "discrete":
        return bool(a["n"] == b["n"] and a["start"] == b["start"])
    return bool(
        torch.equal(torch.as_tensor(a["low"]).cpu(), b["low"])
        and torch.equal(torch.as_tensor(a["high"]).cpu(), b["high"])
    )


class MultiEnvPPOTrainer(PPOTrainer):
    """One policy trained across several environments at once.

    The environments are stepped together as one vector environment, so they
    must share observation and action spaces (for example, one task over
    different data). Without an explicit ``config``, observations and
    rewards are normalized, since the environments' scales may differ.

    Args:
        envs: Gymnasium-API environments (at least one).
        config: Hyperparameters.
        checkpoint_dir: See :class:`PPOTrainer`.
        tensorboard_log: See :class:`PPOTrainer`.

    Raises:
        ValueError: No environments, or differing spaces.
    """

    def __init__(
        self,
        envs: Sequence[Any],
        config: PPOConfig | None = None,
        checkpoint_dir: str | os.PathLike[str] = "./checkpoints",
        tensorboard_log: str | os.PathLike[str] | None = None,
    ) -> None:
        """Combine the environments and build the trainer."""
        if not envs:
            raise ValueError("MultiEnvPPOTrainer needs at least one environment")
        if config is None:
            config = PPOConfig(normalize_observations=True, normalize_rewards=True)
        super().__init__(SyncVectorEnv(list(envs)), config, checkpoint_dir, tensorboard_log)
        self.benchmark_trainers: dict[str, PPOTrainer] = {}

    def pretrain_on_benchmarks(
        self,
        benchmark_names: Sequence[str],
        timesteps_per_benchmark: int = 20000,
        *,
        eval_episodes: int = 10,
    ) -> dict[str, TrainingStats]:
        """Train and evaluate a separate policy on each named benchmark.

        Each benchmark gets a fresh :class:`PPOTrainer` with this trainer's
        configuration (kept in :attr:`benchmark_trainers`), trained for
        ``timesteps_per_benchmark`` steps and then evaluated.

        Args:
            benchmark_names: Environment names for
                :func:`~omni_mercury_engine.ml.rl_envs.make_vec_env`.
            timesteps_per_benchmark: Step budget per benchmark.
            eval_episodes: Evaluation episodes per benchmark.

        Returns:
            Each benchmark's :class:`TrainingStats` (with evaluation results).

        Raises:
            KeyError: An unknown name (checked before any training).
        """
        _positive_int("timesteps_per_benchmark", timesteps_per_benchmark)
        envs = {
            name: make_vec_env(name, self.config.n_envs, seed=self.config.seed)
            for name in benchmark_names
        }
        results: dict[str, TrainingStats] = {}
        for name, env in envs.items():
            trainer = PPOTrainer(env, self.config, checkpoint_dir=Path(self.checkpoint_dir) / name)
            trainer.learn(timesteps_per_benchmark)
            trainer.evaluate(eval_episodes)
            self.benchmark_trainers[name] = trainer
            results[name] = trainer.stats
        return results


def create_ppo_trainer(
    env: Any,
    config: PPOConfig | None = None,
    **kwargs: Any,
) -> PPOTrainer:
    """Build a :class:`PPOTrainer` (``kwargs`` go to its constructor)."""
    return PPOTrainer(env=env, config=config, **kwargs)


def create_multi_env_trainer(
    envs: Sequence[Any],
    config: PPOConfig | None = None,
    **kwargs: Any,
) -> MultiEnvPPOTrainer:
    """Build a :class:`MultiEnvPPOTrainer` (``kwargs`` go to its constructor)."""
    return MultiEnvPPOTrainer(envs=envs, config=config, **kwargs)

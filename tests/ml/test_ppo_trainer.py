# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ml.ppo_trainer``: the native PPO implementation.

Covers the hand-written gradient against autograd, GAE and the running
statistics against reference formulas, the update's controls (clipping,
schedules, KL early stop, truncation bootstrapping), reproducibility,
checkpoints through the restricted loader, the callbacks, every 2.0.0 entry
point, and learning on CartPole and Pendulum within a CI budget.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch

from omni_mercury_engine import ml
from omni_mercury_engine.ml.ppo_trainer import (
    ActorCritic,
    BaseCallback,
    CheckpointCallback,
    ConvergenceMonitor,
    MultiEnvPPOTrainer,
    PPOConfig,
    PPOTrainer,
    RunningMeanStd,
    TrainingStats,
    compute_gae,
    create_multi_env_trainer,
    create_ppo_trainer,
    ppo_loss,
)
from omni_mercury_engine.ml.rl_envs import Box, Discrete, SyncVectorEnv
from omni_mercury_engine.security.safe_torch import UnsafeCheckpointError

if TYPE_CHECKING:
    from pathlib import Path

#: The rl-baselines3-zoo CartPole-v1 settings.
CARTPOLE: dict[str, Any] = {
    "n_envs": 8,
    "n_steps": 32,
    "batch_size": 256,
    "gae_lambda": 0.8,
    "gamma": 0.98,
    "n_epochs": 20,
    "ent_coef": 0.0,
    "learning_rate": 0.001,
    "lr_schedule": "linear",
    "clip_range": 0.2,
    "clip_range_schedule": "linear",
}
#: The rl-baselines3-zoo Pendulum-v1 settings.
PENDULUM: dict[str, Any] = {
    "n_envs": 4,
    "n_steps": 1024,
    "batch_size": 64,
    "gae_lambda": 0.95,
    "gamma": 0.9,
    "n_epochs": 10,
    "ent_coef": 0.0,
    "learning_rate": 1e-3,
    "clip_range": 0.2,
    "use_sde": True,
    "sde_sample_freq": 4,
    "normalize_observations": True,
    "normalize_rewards": True,
}
SMALL: dict[str, Any] = {"n_envs": 2, "n_steps": 16, "batch_size": 16, "n_epochs": 2, "verbose": 0}


def small(**overrides: Any) -> PPOConfig:
    return PPOConfig(**{**SMALL, "seed": 0, **overrides})


class NotPlainData:
    """A picklable object that is not a tensor or primitive."""


class CountdownEnv:
    """Discrete-action toy: observation = step count, terminates after ``length``."""

    observation_space: Any = Box(np.array([0.0], np.float32), np.array([100.0], np.float32))
    action_space: Any = Discrete(2)

    def __init__(self, length: int = 4, truncate: bool = False) -> None:
        self.length, self.truncate, self.t = length, truncate, 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.t += 1
        done = self.t >= self.length
        obs = np.array([float(self.t)], np.float32)
        return obs, 1.0, done and not self.truncate, done and self.truncate, {}


@pytest.fixture(autouse=True)
def _one_thread() -> Any:
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


# -- the hand-written gradient ---------------------------------------------------------

GRID = [
    (discrete, sde, act, arch, vclip, ortho)
    for discrete, sde, act, arch, vclip, ortho in itertools.product(
        [True, False],
        [False, True],
        ["tanh", "relu"],
        [(64, 64), (16,), ()],
        [None, 0.2],
        [True, False],
    )
    if not (discrete and sde)
]


@pytest.mark.parametrize(("discrete", "sde", "activation", "arch", "clip_vf", "ortho"), GRID)
def test_hand_gradient_equals_autograd(
    discrete: bool,
    sde: bool,
    activation: str,
    arch: tuple[int, ...],
    clip_vf: float | None,
    ortho: bool,
) -> None:
    torch.manual_seed(1)
    policy = ActorCritic(
        5,
        3,
        discrete=discrete,
        net_arch=arch,
        activation=activation,
        use_sde=sde,
        ortho_init=ortho,
        log_std_init=-0.5,
    )
    with torch.no_grad():
        policy.flat.add_(0.3 * torch.randn_like(policy.flat))
    n = 97
    obs = torch.randn(n, 5)
    actions = torch.randint(0, 3, (n,)) if discrete else torch.randn(n, 3)
    with torch.no_grad():
        log_prob, _, values = policy.evaluate_actions(obs, actions)
    batch = (
        obs,
        actions,
        log_prob + 0.3 * torch.randn(n),
        values + 0.3 * torch.randn(n),
        torch.randn(n),
        torch.randn(n),
    )
    coefs: dict[str, Any] = {
        "clip_range": 0.2,
        "clip_range_vf": clip_vf,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
    }
    policy.flat.grad = None
    ppo_loss(policy, *batch, **coefs).backward()
    assert policy.flat.grad is not None
    expected = policy.flat.grad.clone()
    policy.flat.grad.fill_(float("nan"))
    metrics = policy.ppo_gradients(*batch, **coefs)
    got = policy.flat.grad
    assert got is not None
    assert not torch.isnan(got).any(), "every gradient element must be written"
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-6 * float(expected.abs().max()))
    assert metrics.shape == (5,) and torch.isfinite(metrics).all()


def test_hand_gradient_follows_a_replaced_gradient_buffer() -> None:
    policy = ActorCritic(3, 2, discrete=True)
    policy.flat.grad = None  # e.g. optimizer.zero_grad(set_to_none=True)
    obs = torch.randn(8, 3)
    batch = (
        obs,
        torch.zeros(8, dtype=torch.long),
        torch.zeros(8),
        torch.zeros(8),
        torch.randn(8),
        torch.randn(8),
    )
    policy.ppo_gradients(*batch, clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5)
    assert policy.flat.grad is not None and policy.flat.grad.abs().sum() > 0


def test_metrics_report_the_loss_terms() -> None:
    torch.manual_seed(0)
    policy = ActorCritic(4, 2, discrete=True)
    obs, actions = torch.randn(64, 4), torch.randint(0, 2, (64,))
    with torch.no_grad():
        log_prob, entropy, values = policy.evaluate_actions(obs, actions)
    adv, ret = torch.randn(64), torch.randn(64)
    metrics = policy.ppo_gradients(
        obs,
        actions,
        log_prob,
        values,
        adv,
        ret,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
    )
    # At the rollout policy the ratio is 1: no KL, nothing clipped.
    assert metrics[0].item() == pytest.approx(-adv.mean().item(), abs=1e-6)
    assert metrics[1].item() == pytest.approx((ret - values).square().mean().item(), rel=1e-5)
    assert metrics[2].item() == pytest.approx(entropy.mean().item(), rel=1e-5)
    assert metrics[3].item() == pytest.approx(0.0, abs=1e-7) and metrics[4].item() == 0.0


def test_parameters_are_views_of_one_flat_vector_and_follow_moves() -> None:
    policy = ActorCritic(4, 2, discrete=False, use_sde=True)
    sizes = sum(t.numel() for t in policy.tensors.values())
    assert sizes == policy.flat.numel() and list(policy.state_dict()) == ["flat"]
    policy.double()
    assert policy.flat.dtype == torch.float64
    for view in policy.tensors.values():
        assert view.dtype == torch.float64
        assert view.untyped_storage().data_ptr() == policy.flat.untyped_storage().data_ptr()


def test_orthogonal_initialization_gains() -> None:
    policy = ActorCritic(8, 3, discrete=True, net_arch=(64, 64))
    w = policy.tensors["hidden.1.weight"][0]
    assert torch.allclose(w.t() @ w, 2.0 * torch.eye(64), atol=1e-4)  # gain sqrt(2)
    head = policy.tensors["pi.weight"]
    assert torch.allclose(head.t() @ head, 1e-4 * torch.eye(3), atol=1e-7)  # gain 0.01
    assert policy.tensors["hidden.0.bias"].abs().sum() == 0


def test_gumbel_sampling_matches_the_softmax() -> None:
    policy = ActorCritic(2, 3, discrete=True, net_arch=())
    with torch.no_grad():
        policy.tensors["pi.bias"].copy_(torch.tensor([0.0, 1.0, -1.0]))
    gen = torch.Generator().manual_seed(0)
    actions, log_prob, _ = policy.act(torch.zeros(100_000, 2), deterministic=False, generator=gen)
    expected = torch.softmax(torch.tensor([0.0, 1.0, -1.0]), 0)
    freq = torch.bincount(actions, minlength=3) / 100_000
    assert torch.allclose(freq, expected, atol=0.005)
    assert torch.allclose(log_prob, expected.log()[actions])


@pytest.mark.parametrize("sde", [False, True])
def test_rollout_and_update_log_probs_agree(sde: bool) -> None:
    torch.manual_seed(0)
    policy = ActorCritic(3, 2, discrete=False, use_sde=sde, log_std_init=-1.0)
    gen = torch.Generator().manual_seed(1)
    obs = torch.randn(32, 3)
    noise = policy.sample_sde_noise(32, gen) if sde else None
    actions, log_prob, values = policy.act(obs, deterministic=False, generator=gen, sde_noise=noise)
    replay, _, replay_values = policy.evaluate_actions(obs, actions)
    assert torch.allclose(log_prob, replay, atol=1e-5) and torch.allclose(values, replay_values)
    mode, _, _ = policy.act(obs, deterministic=True, generator=gen)
    assert torch.allclose(mode, policy._forward(policy.tensors, obs)[1])


# -- GAE and running statistics --------------------------------------------------------


def _reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Schulman et al. (2016) eq. 16, computed per environment from scratch."""
    steps, envs = rewards.shape
    out = np.zeros_like(rewards, dtype=np.float64)
    for e in range(envs):
        for t in range(steps):
            total, weight = 0.0, 1.0
            for k in range(t, steps):
                next_v = last[e] if k == steps - 1 else values[k + 1, e]
                delta = rewards[k, e] + gamma * next_v * (1 - dones[k, e]) - values[k, e]
                total += weight * delta
                if dones[k, e]:
                    break
                weight *= gamma * lam
            out[t, e] = total
    return out


@pytest.mark.parametrize(("gamma", "lam"), [(0.99, 0.95), (0.9, 0.0), (1.0, 1.0)])
def test_gae_matches_the_definition(gamma: float, lam: float) -> None:
    rng = np.random.default_rng(0)
    rewards, values = rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    dones = rng.random((12, 3)) < 0.2
    last = rng.normal(size=3)
    adv, ret = compute_gae(rewards, values, dones, last, gamma, lam)
    expected = _reference_gae(rewards, values, dones, last, gamma, lam)
    assert np.allclose(adv, expected, atol=1e-5)
    assert np.allclose(ret, adv + values.astype(np.float32))


def test_running_mean_std_equals_the_batch_statistics() -> None:
    rng = np.random.default_rng(1)
    batches = [rng.normal(3.0, 2.0, size=(n, 4)) for n in (1, 7, 50, 3)]
    rms = RunningMeanStd((4,), epsilon=0.0)
    for b in batches:
        rms.update(b)
    data = np.concatenate(batches)
    assert np.allclose(rms.mean, data.mean(0)) and np.allclose(rms.var, data.var(0))
    assert rms.count == len(data)
    restored = RunningMeanStd((4,))
    restored.load_state_dict(rms.state_dict())
    assert np.array_equal(restored.mean, rms.mean) and restored.count == rms.count
    with pytest.raises(ValueError):
        RunningMeanStd((3,)).load_state_dict(rms.state_dict())
    with pytest.raises(ValueError):  # the variance alone mismatched
        RunningMeanStd((4,)).load_state_dict({**rms.state_dict(), "var": torch.ones(3)})


# -- configuration ------------------------------------------------------------


def test_config_keeps_the_2_0_0_fields_and_defaults() -> None:
    cfg = PPOConfig()
    released = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "verbose": 1,
    }
    assert {k: getattr(cfg, k) for k in released} == released
    assert [f.name for f in dataclasses.fields(PPOConfig)][:12] == list(released)


@pytest.mark.parametrize(
    "bad",
    [
        {"learning_rate": 0.0},
        {"n_steps": 0},
        {"batch_size": True},
        {"gamma": 1.5},
        {"gae_lambda": -0.1},
        {"clip_range": 0.0},
        {"max_grad_norm": float("inf")},
        {"target_kl": -1.0},
        {"lr_schedule": "cosine"},
        {"activation": "gelu"},
        {"net_arch": (64, 0)},
        {"seed": 1.5},
        {"seed": -1},
        {"vf_coef": -1.0},
        {"clip_range_vf": 0.0},
    ],
)
def test_config_rejects_bad_values(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**bad)


def test_config_round_trips_through_primitives() -> None:
    cfg = PPOConfig(net_arch=[32, 16], use_sde=True, seed=3)  # type: ignore[arg-type]
    assert cfg.net_arch == (32, 16)
    assert PPOConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown"):
        PPOConfig.from_dict({**cfg.to_dict(), "ethics_score": 1.0})


def test_training_stats_keep_the_2_0_0_fields() -> None:
    names = {f.name for f in dataclasses.fields(TrainingStats)}
    assert {
        "total_timesteps",
        "total_episodes",
        "mean_reward",
        "std_reward",
        "mean_episode_length",
        "convergence_metric",
        "best_reward",
        "training_time",
        "checkpoints_saved",
    } <= names
    assert "ethics_score" not in names  # never computed in 2.0.0


def test_trainer_rejects_unusable_setups() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        PPOTrainer("CartPole-v1", small(batch_size=33))
    with pytest.raises(ValueError, match="gSDE"):
        PPOTrainer("CartPole-v1", small(use_sde=True))
    with pytest.raises(TypeError):
        PPOTrainer(42)
    env = CountdownEnv()
    env.action_space = Box(np.zeros((2, 2), np.float32), np.ones((2, 2), np.float32))
    with pytest.raises(TypeError, match="one-dimensional"):
        PPOTrainer(env, small(n_envs=1))


# -- training behavior -----------------------------------------------------------


def _flat(trainer: PPOTrainer) -> torch.Tensor:
    return trainer.policy.flat.detach().clone()


def test_training_is_reproducible_and_leaves_global_rng_alone() -> None:
    torch_state, np_state = torch.get_rng_state(), np.random.get_state()
    a = PPOTrainer(
        "Pendulum-v1",
        small(use_sde=True, sde_sample_freq=4, normalize_observations=True, normalize_rewards=True),
    )
    a.learn(96)
    b = PPOTrainer(
        "Pendulum-v1",
        small(use_sde=True, sde_sample_freq=4, normalize_observations=True, normalize_rewards=True),
    )
    b.learn(96)
    assert torch.equal(_flat(a), _flat(b))
    assert dataclasses.replace(a.stats, training_time=0.0) == dataclasses.replace(
        b.stats, training_time=0.0
    )
    c = PPOTrainer("Pendulum-v1", small(seed=1, use_sde=True))
    c.learn(96)
    assert not torch.equal(_flat(a), _flat(c))
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.random.get_state()[1].tolist() == np_state[1].tolist()


def test_learn_counts_steps_episodes_and_updates() -> None:
    trainer = PPOTrainer(lambda: CountdownEnv(4), small(n_envs=2))
    assert isinstance(trainer.env, SyncVectorEnv)
    stats = trainer.learn(100)
    rollouts = math.ceil(100 / 32)
    assert stats.total_timesteps == trainer.num_timesteps == rollouts * 32
    assert stats.total_episodes == rollouts * 32 // 4
    assert (
        stats.mean_reward == 4.0 and stats.mean_episode_length == 4.0 and stats.best_reward == 4.0
    )
    assert stats.n_updates == rollouts * 2 * 2  # epochs x minibatches
    assert stats.training_time > 0
    trainer.learn(32)  # continues
    assert trainer.num_timesteps == (rollouts + 1) * 32


def test_linear_schedules_anneal_over_the_call() -> None:
    trainer = PPOTrainer("CartPole-v1", small(lr_schedule="linear", clip_range_schedule="linear"))
    trainer.learn(64)  # two rollouts of 32: the last update runs at progress 0
    assert trainer.stats.learning_rate == 0.0 and trainer.stats.clip_range == 0.0
    trainer = PPOTrainer("CartPole-v1", small(lr_schedule="linear"))
    trainer.learn(128)
    assert trainer.stats.learning_rate == pytest.approx(0.0)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0
    constant = PPOTrainer("CartPole-v1", small())
    constant.learn(64)
    assert constant.stats.learning_rate == 3e-4 and constant.stats.clip_range == 0.2


def test_target_kl_stops_after_the_first_step_of_each_rollout() -> None:
    """The first minibatch is at the rollout policy (KL 0); the next exceeds any tiny target."""
    trainer = PPOTrainer("CartPole-v1", small(target_kl=1e-12, batch_size=8, learning_rate=1e-2))
    trainer.learn(96)
    assert trainer.stats.n_updates == 3


def test_truncation_bootstraps_the_value_of_the_final_observation() -> None:
    trainer = PPOTrainer(
        lambda: CountdownEnv(3, truncate=True), small(n_envs=1, n_steps=6, batch_size=6, gamma=0.9)
    )
    rollout = trainer._collect(6, [])
    assert rollout is not None
    bootstrap = 0.9 * trainer.policy.value(torch.tensor([[3.0]])).item()
    assert rollout.rewards[:, 0].tolist() == pytest.approx([1.0, 1.0, 1.0 + bootstrap] * 2)
    assert rollout.dones[:, 0].tolist() == [False, False, True] * 2
    terminal = PPOTrainer(lambda: CountdownEnv(3), small(n_envs=1, n_steps=6, batch_size=6))
    terminal_rollout = terminal._collect(6, [])
    assert terminal_rollout is not None
    assert terminal_rollout.rewards[:, 0].tolist() == [1.0] * 6


def test_observation_and_reward_normalization() -> None:
    trainer = PPOTrainer("Pendulum-v1", small(normalize_observations=True, normalize_rewards=True))
    rollout = trainer._collect(16, [])
    assert rollout is not None
    assert trainer.obs_rms is not None and trainer.obs_rms.count > 30
    assert np.abs(rollout.obs).max() <= 10.0
    assert trainer.ret_rms is not None and trainer.ret_rms.count > 30
    raw_scale = 16.2736  # Pendulum's largest per-step cost
    assert np.abs(rollout.rewards).max() < raw_scale


def test_discrete_observations_are_one_hot() -> None:
    env = CountdownEnv(3)
    env.observation_space = Discrete(5)
    trainer = PPOTrainer(env, small(n_envs=1))
    assert trainer.obs_dim == 5
    assert trainer._flatten(np.array([2])).tolist() == [[0.0, 0.0, 1.0, 0.0, 0.0]]


def test_discrete_action_offset_reaches_the_environment() -> None:
    @dataclasses.dataclass(frozen=True)
    class Offset:
        n: int = 2
        start: int = 10

    seen: list[int] = []

    class Env(CountdownEnv):
        action_space: Any = Offset()

        def step(self, action: Any) -> Any:
            seen.append(int(action))
            return super().step(action)

    trainer = PPOTrainer(Env(), small(n_envs=1))
    trainer.learn(16)
    assert set(seen) <= {10, 11} and seen
    assert trainer.predict(np.array([0.0], np.float32)) in (10, 11)


# -- callbacks --------------------------------------------------------------------


def test_a_callback_can_stop_training_and_sees_episodes() -> None:
    class StopAtFirstEpisode(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.rollouts = 0
            self.started = self.ended = False

        def _on_training_start(self) -> None:
            self.started = True

        def _on_step(self) -> bool:
            return not self.locals["episode"]

        def _on_rollout_end(self) -> None:
            self.rollouts += 1

        def _on_training_end(self) -> None:
            self.ended = True

    cb = StopAtFirstEpisode()
    trainer = PPOTrainer(lambda: CountdownEnv(5), small(n_envs=1))
    trainer.learn(1000, callback=cb)
    assert cb.n_calls == 5 and trainer.num_timesteps == 5 and cb.model is trainer
    assert cb.started and cb.ended and cb.rollouts == 0
    assert cb.locals["episode"] == [{"r": 5.0, "l": 5}]


def test_every_callback_sees_the_step_that_stops_training() -> None:
    class Stop(BaseCallback):
        def _on_step(self) -> bool:
            return False

    counter = BaseCallback()
    trainer = PPOTrainer(lambda: CountdownEnv(5), small(n_envs=1))
    trainer.learn(100, callback=[Stop(), counter])
    assert counter.n_calls == 1


def test_convergence_monitor_detects_a_plateau() -> None:
    monitor = ConvergenceMonitor(convergence_threshold=0.99, patience=2, window=2)
    returns = [1.0, 2.0, 3.0, 3.01, 3.0, 3.02]  # means 1.5, 2.5, 3.005, 3.005, 3.01
    results = []
    for r in returns:
        monitor.locals = {"episode": [{"r": r, "l": 1}]}
        results.append(monitor._on_step())
    assert results == [True, True, True, True, True, False]
    assert monitor.converged and monitor.episodes_without_improvement == 2
    assert monitor.best_mean_reward == pytest.approx(3.005)
    assert monitor.get_convergence_metric() == 1.0


def test_convergence_monitor_handles_negative_returns() -> None:
    monitor = ConvergenceMonitor(convergence_threshold=0.999, patience=100, window=1)
    for r in (-200.0, -300.0):
        monitor.locals = {"episode": [{"r": r, "l": 1}]}
        monitor._on_step()
    assert monitor.best_mean_reward == -200.0
    assert monitor.get_convergence_metric() == pytest.approx(0.5)
    assert ConvergenceMonitor().get_convergence_metric() == 0.0
    for bad in ({"convergence_threshold": 0.0}, {"convergence_threshold": 1.5}, {"patience": 0}):
        with pytest.raises(ValueError):
            ConvergenceMonitor(**bad)


def test_checkpoint_callback_writes_periodic_and_best(tmp_path: Path) -> None:
    cb = CheckpointCallback(tmp_path / "ckpt", save_freq=40, name_prefix="run")
    trainer = PPOTrainer(lambda: CountdownEnv(4), small(n_envs=2))
    trainer.learn(96, callback=cb)
    names = sorted(p.name for p in (tmp_path / "ckpt").glob("*.pt"))
    assert names == ["run_best.pt", "run_step_40.pt", "run_step_80.pt"]
    assert cb.checkpoints_saved == trainer.stats.checkpoints_saved == 3  # best once: returns stay 4
    assert not list((tmp_path / "ckpt").glob(".*tmp"))


# -- 2.0.0 entry points --------------------------------------------------------------


def test_pretrain_stops_on_a_plateau_and_checkpoints(tmp_path: Path) -> None:
    trainer = PPOTrainer(lambda: CountdownEnv(2), small(n_envs=2), checkpoint_dir=tmp_path)
    stats = trainer.pretrain(total_timesteps=100_000, convergence_threshold=0.999)
    assert trainer.convergence_monitor is not None and trainer.convergence_monitor.converged
    assert stats.total_timesteps < 1000  # 100 episodes to fill the window, then 10 flat
    assert stats.convergence_metric == 1.0
    assert (tmp_path / "ppo_pretrain_best.pt").exists()
    summary = trainer.get_training_summary()
    assert summary["converged"] is True and summary["total_timesteps"] == stats.total_timesteps


def test_train_online_trains_for_the_episode_budget() -> None:
    trainer = PPOTrainer(lambda: CountdownEnv(3), small(n_envs=1, n_steps=64, batch_size=8))
    stats = trainer.train_online(num_episodes=12, update_freq=4)
    assert stats.total_episodes == 12 and stats.total_timesteps == 36
    assert stats.n_updates > 0
    with pytest.raises(ValueError):
        trainer.train_online(num_episodes=0)


def test_evaluate_splits_episodes_across_copies() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    mean, std = trainer.evaluate(num_episodes=5)
    assert trainer.stats.eval_mean_reward == mean and trainer.stats.eval_std_reward == std
    assert 8 <= mean <= 500
    # With no factory the training environment is used, and reset for training.
    shared = PPOTrainer(SyncVectorEnv([CountdownEnv(3)]), small(n_envs=1))
    shared.learn(16)
    assert shared.evaluate(num_episodes=4) == (3.0, 0.0)
    assert shared._last_obs is None
    assert shared.evaluate(num_episodes=2, env=CountdownEnv(6)) == (6.0, 0.0)


def test_predict_handles_single_and_batched_observations() -> None:
    trainer = PPOTrainer("Pendulum-v1", small(normalize_observations=True))
    trainer.learn(32)
    single = trainer.predict(np.array([1.0, 0.0, 0.0], np.float32))
    batch = trainer.predict(np.zeros((5, 3), np.float32))
    assert single.shape == (1,) and batch.shape == (5, 1)
    assert np.all(np.abs(trainer.predict(np.zeros((50, 3)), deterministic=False)) <= 2.0)
    assert trainer.predict(np.zeros((2, 3))).tolist() == trainer.predict(np.zeros((2, 3))).tolist()


# -- checkpoints ---------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cfg = small(use_sde=True, normalize_observations=True, normalize_rewards=True, net_arch=(32,))
    trainer = PPOTrainer("Pendulum-v1", cfg)
    trainer.learn(64)
    trainer.save_model(tmp_path / "m.pt")
    other = PPOTrainer("Pendulum-v1", small())  # different architecture: rebuilt on load
    other.load_model(tmp_path / "m.pt")
    assert other.config == trainer.config and torch.equal(_flat(other), _flat(trainer))
    assert other.num_timesteps == trainer.num_timesteps == other.stats.total_timesteps
    assert other.obs_rms is not None and trainer.obs_rms is not None
    assert np.array_equal(other.obs_rms.mean, trainer.obs_rms.mean)
    obs = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
    assert np.array_equal(other.predict(obs), trainer.predict(obs))
    assert torch.equal(
        other.optimizer.state_dict()["state"][0]["exp_avg"],
        trainer.optimizer.state_dict()["state"][0]["exp_avg"],
    )
    other.learn(32)  # training continues from the checkpoint


def test_load_refuses_foreign_or_unsafe_files(tmp_path: Path) -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    trainer.save_model(tmp_path / "cartpole.pt")
    with pytest.raises(ValueError, match="different observation or action space"):
        PPOTrainer("Pendulum-v1", small()).load_model(tmp_path / "cartpole.pt")
    torch.save({"weights": torch.zeros(2)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a Mercury PPO checkpoint"):
        trainer.load_model(tmp_path / "other.pt")
    torch.save({"format": "mercury.ppo", "payload": NotPlainData()}, tmp_path / "evil.pt")
    with pytest.raises(UnsafeCheckpointError):
        trainer.load_model(tmp_path / "evil.pt")


# -- multi-environment training ------------------------------------------------------


def test_multi_env_trainer_combines_environments() -> None:
    trainer = create_multi_env_trainer([CountdownEnv(3), CountdownEnv(5)], small(n_envs=7))
    assert trainer.num_envs == 2 and isinstance(trainer.env, SyncVectorEnv)
    trainer.learn(32)
    assert trainer.stats.total_episodes > 0
    default = MultiEnvPPOTrainer([CountdownEnv()])
    assert default.config.normalize_observations and default.config.normalize_rewards
    with pytest.raises(ValueError):
        MultiEnvPPOTrainer([])


def test_pretrain_on_benchmarks_trains_and_evaluates_each(tmp_path: Path) -> None:
    trainer = MultiEnvPPOTrainer([CountdownEnv()], small(), checkpoint_dir=tmp_path)
    with pytest.raises(KeyError):
        trainer.pretrain_on_benchmarks(["CartPole-v1", "NoSuchEnvironment-v0"])
    assert trainer.benchmark_trainers == {}
    results = trainer.pretrain_on_benchmarks(
        ["CartPole-v1", "Pendulum-v1"], timesteps_per_benchmark=64, eval_episodes=2
    )
    assert set(results) == {"CartPole-v1", "Pendulum-v1"}
    assert (
        results["CartPole-v1"].total_timesteps == 64
        and results["CartPole-v1"].eval_mean_reward is not None
    )
    assert trainer.benchmark_trainers["Pendulum-v1"].action_dim == 1


def test_factory_and_package_exports() -> None:
    assert isinstance(create_ppo_trainer("CartPole-v1", small()), PPOTrainer)
    for name in (
        "PPOTrainer",
        "MultiEnvPPOTrainer",
        "PPOConfig",
        "TrainingStats",
        "ConvergenceMonitor",
        "CheckpointCallback",
        "create_ppo_trainer",
        "create_multi_env_trainer",
    ):
        assert name in ml.__all__ and getattr(ml, name) is not None


def test_tensorboard_log_needs_tensorboard(tmp_path: Path) -> None:
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tensorboard"):
            PPOTrainer("CartPole-v1", small(), tensorboard_log=tmp_path)
        return
    trainer = PPOTrainer("CartPole-v1", small(), tensorboard_log=tmp_path)
    trainer.learn(32)
    assert list(tmp_path.iterdir())


def test_gymnasium_vector_environments() -> None:
    gym = pytest.importorskip("gymnasium")
    from gymnasium.vector import AutoresetMode

    def make(mode: Any) -> Any:
        return gym.make_vec(
            "CartPole-v1",
            num_envs=2,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": mode},
        )

    with pytest.raises(ValueError, match="SAME_STEP"):
        PPOTrainer(make(AutoresetMode.NEXT_STEP), small())
    trainer = PPOTrainer(make(AutoresetMode.SAME_STEP), small())
    assert trainer.obs_dim == 4 and trainer.action_dim == 2
    trainer.learn(64)
    assert trainer.stats.total_episodes > 0
    single = PPOTrainer(gym.make("Pendulum-v1"), small(n_envs=1))
    single.learn(32)
    assert single.action_dim == 1


# -- learning ------------------------------------------------------------------------


def test_learns_cartpole_with_the_zoo_settings() -> None:
    trainer = PPOTrainer("CartPole-v1", PPOConfig(seed=0, verbose=0, **CARTPOLE))
    trainer.learn(30_000)
    mean, _ = trainer.evaluate(num_episodes=10)
    assert mean >= 195.0  # the classic "solved" bar; 500 is the cap


def test_learns_pendulum_with_gsde() -> None:
    trainer = PPOTrainer("Pendulum-v1", PPOConfig(seed=0, verbose=0, **PENDULUM))
    before, _ = trainer.evaluate(num_episodes=10)
    trainer.learn(16_384)
    after, _ = trainer.evaluate(num_episodes=10)
    assert after > before + 200.0


# -- pinned arithmetic and contracts ---------------------------------------------------
# Each test below closes mutants the first full mutation enumeration of this
# module left alive: defaults, validation edges and arithmetic no behavioural
# test above distinguished.


def _defaults(fn: Any) -> dict[str, Any]:
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_every_default_is_pinned() -> None:
    assert PPOConfig() == PPOConfig(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        verbose=1,
        n_envs=1,
        lr_schedule="constant",
        clip_range_schedule="constant",
        clip_range_vf=None,
        normalize_advantage=True,
        normalize_observations=False,
        normalize_rewards=False,
        clip_observations=10.0,
        clip_rewards=10.0,
        net_arch=(64, 64),
        activation="tanh",
        ortho_init=True,
        log_std_init=0.0,
        use_sde=False,
        sde_sample_freq=-1,
        seed=None,
        device="cpu",
    )
    stats = TrainingStats()
    assert dataclasses.astuple(stats) == (
        0, 0, 0.0, 0.0, 0.0, 0.0, float("-inf"), 0.0, 0, 0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None,
    )  # fmt: skip
    assert _defaults(ActorCritic.__init__) == {
        "net_arch": (64, 64),
        "activation": "tanh",
        "ortho_init": True,
        "log_std_init": 0.0,
        "use_sde": False,
    }
    assert _defaults(RunningMeanStd.__init__) == {"shape": (), "epsilon": 1e-4}
    assert RunningMeanStd().count == 1e-4
    assert _defaults(BaseCallback.__init__) == {"verbose": 0}
    assert _defaults(ConvergenceMonitor.__init__) == {
        "convergence_threshold": 0.999,
        "patience": 10,
        "verbose": 0,
        "window": 100,
    }
    assert _defaults(CheckpointCallback.__init__) == {
        "save_freq": 10000,
        "name_prefix": "ppo_model",
        "verbose": 0,
    }
    assert _defaults(PPOTrainer.pretrain) == {
        "total_timesteps": 100000,
        "convergence_threshold": 0.999,
        "save_checkpoints": True,
        "patience": 10,
    }
    assert _defaults(PPOTrainer.train_online) == {
        "num_episodes": 100,
        "update_freq": 10,
        "callback": None,
    }
    assert _defaults(PPOTrainer.evaluate) == {
        "num_episodes": 10,
        "deterministic": True,
        "env": None,
        "seed": None,
    }
    assert _defaults(PPOTrainer.predict) == {"deterministic": True}
    assert _defaults(MultiEnvPPOTrainer.pretrain_on_benchmarks) == {
        "timesteps_per_benchmark": 20000,
        "eval_episodes": 10,
    }
    monitor = ConvergenceMonitor()
    assert monitor.episodes_without_improvement == 0 and monitor.converged is False


@pytest.mark.parametrize(
    "edge",
    [
        {"gamma": 0.0},
        {"gamma": 0.05},
        {"gamma": 1.0},
        {"gae_lambda": 0.0},
        {"gae_lambda": 1.0},
        {"vf_coef": 0.0},
        {"vf_coef": 0.05},
        {"ent_coef": -0.01},
        {"clip_range": 0.05},
        {"max_grad_norm": 0.05},
        {"clip_observations": 0.05},
        {"clip_rewards": 0.05},
        {"target_kl": 0.05},
        {"clip_range_vf": 0.05},
        {"learning_rate": 0.05},
        {"seed": 0},
        {"sde_sample_freq": 0},
    ],
)
def test_config_accepts_its_boundary_values(edge: dict[str, Any]) -> None:
    PPOConfig(**edge)


@pytest.mark.parametrize(
    "bad",
    [
        {"max_grad_norm": 0.0},
        {"clip_observations": 0.0},
        {"clip_rewards": 0.0},
        {"gamma": 1.0000001},
        {"sde_sample_freq": True},
        {"sde_sample_freq": 1.5},
        {"n_envs": 0},
    ],
)
def test_config_rejects_values_just_past_the_boundary(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**bad)


def test_convergence_monitor_threshold_edges() -> None:
    ConvergenceMonitor(convergence_threshold=1.0)
    ConvergenceMonitor(convergence_threshold=0.05)
    # Threshold 1 means zero tolerance: an exactly flat window never improves.
    monitor = ConvergenceMonitor(convergence_threshold=1.0, patience=2, window=1)
    results = []
    for r in (5.0, 5.0, 5.0):
        monitor.locals = {"episode": [{"r": r, "l": 1}]}
        results.append(monitor._on_step())
    assert results == [True, True, False]


def test_convergence_monitor_tolerance_is_relative_to_best() -> None:
    # Threshold 0.99: a 5% gain is an improvement, a 0.5% gain is not.
    monitor = ConvergenceMonitor(convergence_threshold=0.99, patience=5, window=1)
    for r in (10.0, 10.5, 10.55):
        monitor.locals = {"episode": [{"r": r, "l": 1}]}
        monitor._on_step()
    assert monitor.best_mean_reward == 10.5 and monitor.episodes_without_improvement == 1


def test_convergence_metric_edges() -> None:
    monitor = ConvergenceMonitor(window=2)
    for r in (0.01, 0.0):
        monitor.reward_history.append(r)
    monitor.best_mean_reward = 0.01
    # Exactly a full window counts, and the scale is |best| even when tiny.
    assert monitor.get_convergence_metric() == pytest.approx(0.5)
    monitor.reward_history[:] = [-100.0, -100.0]
    monitor.best_mean_reward = 100.0
    assert monitor.get_convergence_metric() == 0.0  # clipped at 0


def test_reward_window_keeps_the_last_100_episodes() -> None:
    trainer = PPOTrainer(lambda: CountdownEnv(1), small(n_envs=1, n_steps=150, batch_size=8))
    trainer.learn(150)
    assert len(trainer.episode_returns) == 100 and trainer.stats.total_episodes == 150


@pytest.mark.parametrize("sde", [False, True])
def test_gaussian_log_prob_and_entropy_match_torch_distributions(sde: bool) -> None:
    torch.manual_seed(0)
    policy = ActorCritic(3, 2, discrete=False, use_sde=sde, log_std_init=-0.7)
    obs, actions = torch.randn(16, 3), torch.randn(16, 2)
    with torch.no_grad():
        log_prob, entropy, values = policy.evaluate_actions(obs, actions)
        hidden, mean, _ = policy._forward(policy.tensors, obs)
        std = policy._std(policy.tensors["log_std"], hidden[-1][0], mean)
    dist = torch.distributions.Normal(mean, std)
    assert torch.allclose(log_prob, dist.log_prob(actions).sum(-1), atol=1e-5)
    assert torch.allclose(entropy, dist.entropy().sum(-1), atol=1e-5)
    metrics = policy.ppo_gradients(
        obs, actions, log_prob, values, torch.randn(16), torch.randn(16),
        clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5,
    )  # fmt: skip
    assert metrics[2].item() == pytest.approx(entropy.mean().item(), rel=1e-5)


def test_gsde_standard_deviation_formula() -> None:
    policy = ActorCritic(2, 1, discrete=False, use_sde=True, net_arch=())
    with torch.no_grad():
        policy.tensors["log_std"].copy_(torch.tensor([[0.5], [0.25]]).log())
    mean = torch.zeros(2, 1)
    latent = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    std = policy._std(policy.tensors["log_std"], latent, mean)
    # std^2 = latent^2 @ exp(2 log_std) + 1e-6
    assert std[0, 0].item() == pytest.approx(math.sqrt(0.25 + 4 * 0.0625 + 1e-6), rel=1e-6)
    assert std[1, 0].item() == pytest.approx(1e-3, rel=1e-4)


def test_gsde_noise_scale_and_actions() -> None:
    torch.manual_seed(0)
    policy = ActorCritic(3, 2, discrete=False, use_sde=True, log_std_init=math.log(0.1))
    gen = torch.Generator().manual_seed(0)
    noise = policy.sample_sde_noise(4000, gen)
    assert noise.shape == (4000, 64, 2)
    assert noise.std().item() == pytest.approx(0.1, rel=0.02) and abs(noise.mean().item()) < 0.002
    obs = torch.randn(5, 3)
    actions, _, _ = policy.act(obs, deterministic=False, generator=gen, sde_noise=noise[:5])
    hidden, mean, _ = policy._forward(policy.tensors, obs)
    expected = mean + torch.bmm(hidden[-1][0].unsqueeze(1), noise[:5]).squeeze(1)
    assert torch.allclose(actions, expected)
    drawn, _, _ = policy.act(obs, deterministic=False, generator=gen)  # fresh noise per row
    assert drawn.shape == (5, 2) and actions.shape == (5, 2)


def test_gaussian_actions_are_mean_plus_scaled_noise() -> None:
    torch.manual_seed(0)
    policy = ActorCritic(3, 2, discrete=False, log_std_init=-1.0)
    obs = torch.randn(6, 3)
    actions, _, _ = policy.act(obs, deterministic=False, generator=torch.Generator().manual_seed(4))
    noise = torch.randn((6, 2), generator=torch.Generator().manual_seed(4))
    mean = policy._forward(policy.tensors, obs)[1]
    assert torch.allclose(actions, mean + math.exp(-1.0) * noise)


def test_metrics_match_their_definitions() -> None:
    torch.manual_seed(0)
    policy = ActorCritic(4, 3, discrete=True)
    obs, actions = torch.randn(64, 4), torch.randint(0, 3, (64,))
    with torch.no_grad():
        log_prob, _, values = policy.evaluate_actions(obs, actions)
    old = log_prob + torch.linspace(-0.5, 0.5, 64)
    adv, ret = torch.randn(64), torch.randn(64)
    metrics = policy.ppo_gradients(
        obs, actions, old, values, adv, ret,
        clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5,
    )  # fmt: skip
    log_ratio = log_prob - old
    ratio = log_ratio.exp()
    assert metrics[0].item() == pytest.approx(
        -torch.min(adv * ratio, adv * ratio.clamp(0.8, 1.2)).mean().item(), rel=1e-5
    )
    assert metrics[3].item() == pytest.approx(((ratio - 1) - log_ratio).mean().item(), rel=1e-5)
    assert metrics[4].item() == ((ratio - 1).abs() > 0.2).float().mean().item()
    assert 0.0 < metrics[4].item() < 1.0


def test_default_initialization_matches_nn_linear_bounds() -> None:
    torch.manual_seed(0)
    policy = ActorCritic(4, 2, discrete=True, ortho_init=False)
    first = policy.tensors["hidden.0.weight"]  # fan-in 4: U(-0.5, 0.5)
    assert 0.45 < first.abs().max().item() <= 0.5
    second = policy.tensors["hidden.1.weight"]  # fan-in 64: U(-0.125, 0.125)
    assert 0.12 < second.abs().max().item() <= 0.125
    assert policy.tensors["hidden.0.bias"].abs().max().item() <= 0.5


# -- the update's arithmetic, observed through a stand-in gradient --------------------


class _GradientProbe:
    """Stands in for ``ActorCritic.ppo_gradients``: records calls, writes a set gradient."""

    def __init__(
        self, policy: ActorCritic, grad: float = 0.0, metrics: Any = (1, 2, 3, 4, 5)
    ) -> None:
        self.policy, self.grad, self.metrics = (
            policy,
            grad,
            torch.tensor(metrics, dtype=torch.float32),
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        names = ("obs", "actions", "old_log_probs", "old_values", "advantages", "returns")
        self.calls.append({**dict(zip(names, args, strict=True)), **kwargs})
        assert self.policy.flat.grad is not None
        self.policy.flat.grad.fill_(self.grad)
        return self.metrics


def _probe(trainer: PPOTrainer, **kwargs: Any) -> _GradientProbe:
    probe = _GradientProbe(trainer.policy, **kwargs)
    trainer.policy.ppo_gradients = probe  # type: ignore[method-assign]
    return probe


def _steps(trainer: PPOTrainer) -> list[dict[str, float]]:
    """Record the gradient norm and learning rate at every optimizer step."""
    seen: list[dict[str, float]] = []
    step = trainer.optimizer.step

    def recording_step(*args: Any, **kwargs: Any) -> Any:
        grad = trainer.policy.flat.grad
        assert grad is not None
        seen.append({"norm": float(grad.norm()), "lr": trainer.optimizer.param_groups[0]["lr"]})
        return step(*args, **kwargs)

    trainer.optimizer.step = recording_step  # type: ignore[method-assign]
    return seen


def test_advantages_are_normalized_per_minibatch() -> None:
    trainer = PPOTrainer("CartPole-v1", small(batch_size=32, n_epochs=1))
    probe = _probe(trainer)
    rollout = trainer._collect(16, [])
    assert rollout is not None
    trainer._update(rollout, 1.0)
    adv, _ = compute_gae(
        rollout.rewards, rollout.values, rollout.dones, rollout.last_values, 0.99, 0.95
    )
    flat_adv = torch.as_tensor(adv.reshape(-1))
    expected = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    assert torch.allclose(probe.calls[0]["advantages"], expected)
    raw = PPOTrainer("CartPole-v1", small(batch_size=32, n_epochs=1, normalize_advantage=False))
    raw_probe = _probe(raw)
    raw_rollout = raw._collect(16, [])
    assert raw_rollout is not None
    raw._update(raw_rollout, 1.0)
    raw_adv, _ = compute_gae(
        raw_rollout.rewards,
        raw_rollout.values,
        raw_rollout.dones,
        raw_rollout.last_values,
        0.99,
        0.95,
    )
    assert torch.equal(raw_probe.calls[0]["advantages"], torch.as_tensor(raw_adv.reshape(-1)))
    single = PPOTrainer("CartPole-v1", small(batch_size=1, n_epochs=1, n_steps=2, n_envs=1))
    single_probe = _probe(single)
    single.learn(2)
    assert all(torch.isfinite(c["advantages"]).all() for c in single_probe.calls)
    pairs = PPOTrainer("CartPole-v1", small(batch_size=2, n_epochs=1, n_steps=2, n_envs=1))
    pair_probe = _probe(pairs)
    pairs.learn(2)
    assert pair_probe.calls[0]["advantages"].abs().tolist() == pytest.approx(
        [2**-0.5] * 2, rel=1e-4
    )


def test_minibatches_cover_every_sample_once_per_epoch() -> None:
    trainer = PPOTrainer("CartPole-v1", small(batch_size=10, n_epochs=3))
    probe = _probe(trainer)
    trainer.learn(32)
    sizes = [c["obs"].shape[0] for c in probe.calls]
    assert sizes == [10, 10, 10, 2] * 3


def test_gradient_norm_clipping() -> None:
    trainer = PPOTrainer("CartPole-v1", small(max_grad_norm=0.5))
    _probe(trainer, grad=1.0)
    seen = _steps(trainer)
    trainer.learn(32)
    norm = math.sqrt(trainer.policy.flat.numel())
    assert seen[0]["norm"] == pytest.approx(0.5 * norm / (norm + 1e-6), rel=1e-5)
    small_grad = PPOTrainer("CartPole-v1", small(max_grad_norm=0.5))
    _probe(small_grad, grad=1e-4)
    seen = _steps(small_grad)
    small_grad.learn(32)
    assert seen[0]["norm"] == pytest.approx(
        1e-4 * math.sqrt(small_grad.policy.flat.numel()), rel=1e-5
    )


@pytest.mark.parametrize(("kl", "steps_taken"), [(0.375, 4), (0.39, 0)])
def test_kl_stop_is_strictly_above_one_and_a_half_targets(kl: float, steps_taken: int) -> None:
    trainer = PPOTrainer("CartPole-v1", small(target_kl=0.25))
    probe = _probe(trainer, metrics=(0.0, 0.0, 0.0, kl, 0.0))
    seen = _steps(trainer)
    trainer.learn(32)
    assert len(seen) == steps_taken
    assert len(probe.calls) == (4 if steps_taken else 1)  # a stop ends every epoch


def test_update_statistics_map_the_metrics() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    _probe(trainer, metrics=(1.0, 2.0, 3.0, 4.0, 5.0))
    stats = trainer.learn(32)
    assert (stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl, stats.clip_fraction) == (
        1.0, 2.0, 3.0, 4.0, 5.0,
    )  # fmt: skip


def test_explained_variance_matches_its_definition() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    _probe(trainer)
    rollout = trainer._collect(16, [])
    assert rollout is not None
    trainer._update(rollout, 1.0)
    _, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.dones, rollout.last_values, 0.99, 0.95
    )
    expected = 1.0 - np.var(returns - rollout.values) / np.var(returns)
    assert trainer.stats.explained_variance == pytest.approx(float(expected), rel=1e-6)


def test_linear_schedule_restarts_with_each_learn_call() -> None:
    trainer = PPOTrainer("CartPole-v1", small(n_epochs=1, lr_schedule="linear"))
    seen = _steps(trainer)
    trainer.learn(64)
    trainer.learn(64)
    assert [s["lr"] for s in seen] == pytest.approx([1.5e-4, 1.5e-4, 0.0, 0.0] * 2)


def test_train_online_budget_arithmetic() -> None:
    trainer = PPOTrainer(
        lambda: CountdownEnv(2),
        small(n_envs=1, n_steps=64, batch_size=8, n_epochs=1, lr_schedule="linear"),
    )
    seen = _steps(trainer)
    stats = trainer.train_online(num_episodes=10, update_freq=4)
    assert stats.total_episodes == 10  # rollouts of 4, 4, then exactly 2
    assert [round(s["lr"] / 3e-4, 6) for s in seen] == [0.6, 0.2, 0.0]
    trainer.train_online(num_episodes=3, update_freq=4)
    assert trainer.stats.total_episodes == 13
    del seen[:]
    trainer.train_online(num_episodes=8, update_freq=4)  # anneals over this call's budget
    assert [round(s["lr"] / 3e-4, 6) for s in seen] == [0.5, 0.0]
    trainer.train_online(num_episodes=5, update_freq=4)  # 4, then the last 1
    assert trainer.stats.total_episodes == 26


def test_train_online_updates_once_per_rollout() -> None:
    trainer = PPOTrainer(lambda: CountdownEnv(3), small(n_envs=1, n_steps=64, batch_size=8))
    stats = trainer.train_online(num_episodes=12, update_freq=4)
    assert stats.n_updates == 3 * 2 * 2  # 3 rollouts of 12 samples: 2 minibatches, 2 epochs


def test_training_time_is_wall_clock() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    started = time.perf_counter()
    trainer.learn(64)
    assert 0.0 < trainer.stats.training_time <= time.perf_counter() - started


def test_a_plain_callback_does_not_stop_training() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    callback = BaseCallback()
    trainer.learn(64, callback=callback)
    assert trainer.num_timesteps == 64 and callback.n_calls == 32


def test_observation_normalization_arithmetic() -> None:
    trainer = PPOTrainer("CartPole-v1", small(normalize_observations=True, clip_observations=5.0))
    assert trainer.obs_rms is not None
    trainer.obs_rms.mean = np.array([1.0, -2.0, 0.5, 0.0])
    trainer.obs_rms.var = np.array([4.0, 0.25, 0.0, 1.0])
    raw = np.array([[3.0, -1.0, 0.5, 100.0]], np.float32)
    out = trainer._process_obs(raw, update=False)
    expected = np.array([[2.0 / math.sqrt(4 + 1e-8), 1.0 / math.sqrt(0.25 + 1e-8), 0.0, 5.0]])
    assert np.allclose(out, expected, rtol=1e-6)
    assert trainer.obs_rms.count == 1e-4  # update=False left the statistics alone


def test_reward_scaling_arithmetic() -> None:
    trainer = PPOTrainer("CartPole-v1", small(n_envs=1, normalize_rewards=True, gamma=0.5))
    assert trainer.ret_rms is not None
    outputs = []
    for done in (False, False, True, False):
        out = trainer._scale_rewards(np.array([1.0]), np.array([done]))
        outputs.append(float(out[0]))
    reference = RunningMeanStd(())
    discounted, expected = 0.0, []
    for done in (False, False, True, False):
        discounted = discounted * 0.5 + 1.0
        reference.update(np.array([discounted]))
        expected.append(min(1.0 / math.sqrt(reference.var + 1e-8), 10.0))  # clip_rewards
        if done:
            discounted = 0.0
    assert outputs == pytest.approx(expected, rel=1e-6)
    assert trainer._discounted.tolist() == [1.0]  # restarted after the episode ended


def test_reset_updates_and_final_observations_do_not() -> None:
    trainer = PPOTrainer(
        "Pendulum-v1", small(n_envs=2, n_steps=200, batch_size=8, normalize_observations=True)
    )
    trainer._reset_env()
    assert trainer.obs_rms is not None and trainer.obs_rms.count == pytest.approx(1e-4 + 2)
    rollout = trainer._collect(200, [])  # the 200th step truncates both copies
    assert rollout is not None and rollout.dones[-1].all()
    assert trainer.obs_rms.count == pytest.approx(1e-4 + 2 + 2 * 200)


@pytest.mark.parametrize(("freq", "draws"), [(-1, 1), (0, 1), (4, 3), (1, 10)])
def test_gsde_noise_is_resampled_every_sde_sample_freq_steps(freq: int, draws: int) -> None:
    trainer = PPOTrainer(
        "Pendulum-v1", small(use_sde=True, sde_sample_freq=freq, n_steps=10, batch_size=8)
    )
    calls = []
    sample = trainer.policy.sample_sde_noise

    def counting(n: int, generator: Any) -> torch.Tensor:
        calls.append(n)
        return sample(n, generator)

    trainer.policy.sample_sde_noise = counting  # type: ignore[method-assign]
    trainer._collect(10, [])
    assert calls == [2] * draws


def test_discrete_observation_offset() -> None:
    @dataclasses.dataclass(frozen=True)
    class Shifted:
        n: int = 3
        start: int = 10

    env = CountdownEnv(3)
    env.observation_space = Shifted()
    trainer = PPOTrainer(env, small(n_envs=1))
    assert trainer._flatten(np.array([12, 10])).tolist() == [[0, 0, 1], [1, 0, 0]]


def test_unsupported_spaces_and_objects_are_refused() -> None:
    @dataclasses.dataclass(frozen=True)
    class MultiBinary:
        n: int = 3
        shape: tuple[int, ...] = (3,)

    @dataclasses.dataclass(frozen=True)
    class LowOnly:
        low: float = 0.0

    for space in (MultiBinary(), LowOnly()):
        env = CountdownEnv()
        env.action_space = space
        with pytest.raises(TypeError, match="unsupported space"):
            PPOTrainer(env, small(n_envs=1))

    class ResetOnly:
        def reset(self) -> None:
            return None

    with pytest.raises(TypeError, match="neither"):
        PPOTrainer(ResetOnly())


def test_logging(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="omni_mercury_engine.ml.ppo_trainer"):
        PPOTrainer("CartPole-v1", small(n_envs=1, verbose=1))
        PPOTrainer("CartPole-v1", small(n_envs=2, verbose=1)).learn(32)
        assert "1 copy," in caplog.text and "2 copies," in caplog.text
        assert "trained 32 steps" in caplog.text and "steps=32 " not in caplog.text
        PPOTrainer("CartPole-v1", small(verbose=2)).learn(32)
        assert "steps=32 " in caplog.text
        caplog.clear()
        again = PPOTrainer(lambda: CountdownEnv(4), small(n_envs=2, verbose=1))
        again.learn(32)
        again.learn(32)
        match = re.search(
            r"trained (\d+) steps \((\d+) episodes\) in ([\d.]+)s", caplog.records[-1].message
        )
        assert match is not None
        assert (int(match[1]), int(match[2])) == (32, 8) and float(match[3]) < 60


# -- evaluation, persistence, helpers ---------------------------------------------------


class RecordingEnv(CountdownEnv):
    """Records the seeds it is reset with; episode return = its length."""

    def __init__(self, length: int) -> None:
        super().__init__(length)
        self.seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.seeds.append(seed)
        return super().reset(seed)


def test_evaluate_seeds_and_episode_split() -> None:
    trainer = PPOTrainer(CountdownEnv(3), small(n_envs=1, seed=5))
    env = RecordingEnv(3)
    trainer.evaluate(num_episodes=1, env=env)
    trainer.evaluate(num_episodes=1, env=env, seed=123)
    assert env.seeds[0] == 10_005 and env.seeds[2] == 123
    # Copy 0 owes 1 episode (return 3), copy 1 owes 2 (return 5 each).
    mean, _ = trainer.evaluate(
        num_episodes=3, env=SyncVectorEnv([CountdownEnv(3), CountdownEnv(5)])
    )
    assert mean == pytest.approx(13 / 3)


def test_evaluate_leaves_training_state_alone_when_it_can() -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    trainer.learn(32)
    obs = trainer._last_obs
    sizes = []
    spawn = trainer.env.spawn

    def recording_spawn(num_envs: int, seed: int | None = None) -> Any:
        sizes.append(num_envs)
        return spawn(num_envs, seed=seed)

    trainer.env.spawn = recording_spawn
    trainer.evaluate(num_episodes=40)
    trainer.evaluate(num_episodes=3)
    trainer.evaluate(num_episodes=2, env="CartPole-v1")
    assert sizes == [16, 3] and trainer._last_obs is obs


def test_save_creates_directories_and_failures_leave_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = PPOTrainer("CartPole-v1", small())
    target = tmp_path / "a" / "b" / "m.pt"
    trainer.save_model(target)
    before = target.read_bytes()

    def failing_save(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        trainer.save_model(target)
    assert target.read_bytes() == before
    assert [p.name for p in target.parent.iterdir()] == ["m.pt"]


def test_checkpoint_format_version_is_checked(tmp_path: Path) -> None:
    from omni_mercury_engine.security.safe_torch import safe_torch_load

    trainer = PPOTrainer("CartPole-v1", small())
    trainer.save_model(tmp_path / "m.pt")
    data = safe_torch_load(tmp_path / "m.pt")
    assert (data["format"], data["version"]) == ("mercury.ppo", 1)
    data["version"] = 2
    torch.save(data, tmp_path / "future.pt")
    with pytest.raises(ValueError, match="version"):
        trainer.load_model(tmp_path / "future.pt")


def test_load_refuses_a_different_observation_with_the_same_actions(tmp_path: Path) -> None:
    PPOTrainer(CountdownEnv(), small(n_envs=1)).save_model(tmp_path / "toy.pt")
    with pytest.raises(ValueError, match="different observation"):
        PPOTrainer("CartPole-v1", small()).load_model(tmp_path / "toy.pt")


def test_action_spec_comparison() -> None:
    from omni_mercury_engine.ml.ppo_trainer import _same_action_spec

    discrete = {"kind": "discrete", "n": 2, "start": 0}
    assert _same_action_spec(discrete, dict(discrete))
    assert not _same_action_spec(discrete, {**discrete, "n": 3})
    assert not _same_action_spec(discrete, {**discrete, "start": 1})
    box = {"kind": "box", "low": torch.zeros(1), "high": torch.ones(1)}
    assert _same_action_spec(box, dict(box))
    assert not _same_action_spec(box, {**box, "low": -torch.ones(1)})
    assert not _same_action_spec(box, {**box, "high": 2 * torch.ones(1)})
    assert not _same_action_spec(box, discrete)


def test_summary_reports_not_converged_until_the_monitor_says_so(tmp_path: Path) -> None:
    trainer = PPOTrainer("CartPole-v1", small(), checkpoint_dir=tmp_path)
    trainer.pretrain(total_timesteps=32, save_checkpoints=False)
    assert trainer.convergence_monitor is not None and not trainer.convergence_monitor.converged
    assert trainer.get_training_summary()["converged"] is False
    assert trainer.checkpoint_callback is None


def test_pretrain_wires_its_callbacks(tmp_path: Path) -> None:
    trainer = PPOTrainer("CartPole-v1", small(verbose=2), checkpoint_dir=tmp_path)
    trainer.pretrain(total_timesteps=32)
    cb = trainer.checkpoint_callback
    assert cb is not None and (cb.save_freq, cb.name_prefix, cb.verbose) == (
        10000,
        "ppo_pretrain",
        1,
    )
    assert trainer.convergence_monitor is not None and trainer.convergence_monitor.verbose == 1
    quiet = PPOTrainer("CartPole-v1", small(verbose=0), checkpoint_dir=tmp_path)
    quiet.pretrain(total_timesteps=32)
    assert quiet.convergence_monitor is not None and quiet.convergence_monitor.verbose == 0


def test_checkpoint_callback_reuses_an_existing_directory(tmp_path: Path) -> None:
    CheckpointCallback(tmp_path / "ckpt")
    CheckpointCallback(tmp_path / "ckpt")
    CheckpointCallback(tmp_path / "nested" / "deeper")
    assert (tmp_path / "ckpt").is_dir() and (tmp_path / "nested" / "deeper").is_dir()

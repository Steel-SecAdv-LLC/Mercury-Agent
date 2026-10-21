# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``ml.rl_envs``: batched environments bit-identical to Gymnasium's.

The fixture holds trajectories recorded from Gymnasium itself
(``scripts/record_gymnasium_trajectories.py``); replaying their actions
through the batched environments must reproduce every observation, reward
and flag exactly, across episode boundaries.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from omni_mercury_engine.ml.rl_envs import (
    Box,
    CartPoleVecEnv,
    Discrete,
    PendulumVecEnv,
    SyncVectorEnv,
    VectorEnv,
    make_vec_env,
    registered_envs,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rl_envs" / "gymnasium_trajectories.npz"
)


@pytest.fixture(scope="module")
def recordings() -> Any:
    return np.load(FIXTURE, allow_pickle=False)


class CountdownEnv:
    """Gymnasium-API environment: observation = step count, ends after ``length``."""

    observation_space: Any = Box(np.array([0.0], np.float32), np.array([100.0], np.float32))
    action_space: Any = Discrete(2)

    def __init__(self, length: int = 3) -> None:
        self.length = length
        self.t = 0
        self.seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.seeds.append(seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.t += 1
        return (
            np.array([float(self.t)], np.float32),
            float(action),
            self.t >= self.length,
            False,
            {},
        )


# -- bit-identity with Gymnasium ------------------------------------------------


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
def test_replays_gymnasium_bit_for_bit(recordings: Any, env_id: str) -> None:
    get = lambda key: recordings[f"{env_id}/{key}"]  # noqa: E731
    env = make_vec_env(env_id, num_envs=1)
    obs, _ = env.reset(seed=int(get("seed")))
    resets = iter(get("resets"))
    assert np.array_equal(obs[0], next(resets))
    episodes = 1
    for t, action in enumerate(get("actions")):
        obs, reward, terminated, truncated, info = env.step(np.asarray(action)[None])
        assert np.array_equal(info["final_observation"][0], get("observations")[t]), t
        assert reward[0] == np.float32(get("rewards")[t]), t
        assert bool(terminated[0]) == bool(get("terminated")[t]), t
        assert bool(truncated[0]) == bool(get("truncated")[t]), t
        if terminated[0] or truncated[0]:
            assert np.array_equal(obs[0], next(resets)), t
            episodes += 1
        else:
            assert np.array_equal(obs[0], get("observations")[t]), t
    assert episodes == len(get("resets"))
    assert next(resets, None) is None


def test_fixture_metadata(recordings: Any) -> None:
    meta = json.loads(str(recordings["meta"]))
    assert meta["envs"] == ["CartPole-v1", "Pendulum-v1"]
    assert recordings["Pendulum-v1/truncated"].sum() == 5


@pytest.mark.parametrize("cls", [CartPoleVecEnv, PendulumVecEnv])
def test_batched_rows_match_single_copies(cls: type) -> None:
    """Stepping N copies together equals stepping each alone from the same state."""
    rng = np.random.default_rng(3)
    batch = cls(num_envs=6, seed=0)
    batch.reset()
    batch.max_episode_steps = 10**9
    states = batch._state.copy()
    for _ in range(40):
        if cls is CartPoleVecEnv:
            actions = rng.integers(0, 2, size=6)
        else:
            actions = rng.uniform(-2, 2, size=(6, 1)).astype(np.float32)
        _, rewards, terminated, _, info = batch.step(actions)
        for i in range(6):
            single = cls(num_envs=1)
            single.reset()
            single.max_episode_steps = 10**9
            single._state = states[i : i + 1].copy()
            _, r, term, _, single_info = single.step(actions[i : i + 1])
            assert np.array_equal(single_info["final_observation"][0], info["final_observation"][i])
            assert r[0] == rewards[i] and term[0] == terminated[i]
        states = batch._state.copy()


# -- time limits, autoreset, validation ----------------------------------------------


def test_cartpole_truncates_at_the_limit_and_resets() -> None:
    env = CartPoleVecEnv(num_envs=2, seed=0)
    env.max_episode_steps = 3
    env.reset()
    env._state[:] = 0.0  # balanced: no termination within 3 steps
    for step in range(1, 4):
        obs, _, terminated, truncated, info = env.step(np.array([0, 1]))
        assert not terminated.any()
        assert truncated.all() == (step == 3)
    assert np.all(np.abs(obs) <= 0.05)  # fresh initial states
    assert not np.array_equal(obs, info["final_observation"])
    assert (env._elapsed == 0).all()


def test_cartpole_terminating_on_the_last_step_reports_both_flags() -> None:
    """As Gymnasium's TimeLimit does: terminated and truncated together."""
    env = CartPoleVecEnv(num_envs=1, seed=0)
    env.max_episode_steps = 1
    env.reset()
    env._state[:] = [2.399, 5.0, 0.0, 0.0]
    _, reward, terminated, truncated, _ = env.step(np.array([1]))
    assert terminated[0] and truncated[0] and reward[0] == 1.0


def test_pendulum_never_terminates_and_truncates_at_200() -> None:
    env = PendulumVecEnv(num_envs=3, seed=1)
    env.reset()
    for step in range(1, 201):
        _, _, terminated, truncated, _ = env.step(np.zeros((3, 1), np.float32))
        assert not terminated.any()
        assert truncated.all() == (step == 200)


def test_pendulum_clips_torque() -> None:
    a, b = PendulumVecEnv(1, seed=5), PendulumVecEnv(1, seed=5)
    a.reset(), b.reset()
    ra = a.step(np.array([[50.0]], np.float32))
    rb = b.step(np.array([[2.0]], np.float32))
    assert np.array_equal(ra[0], rb[0]) and ra[1][0] == rb[1][0]


@pytest.mark.parametrize(
    ("env", "actions"),
    [
        (CartPoleVecEnv(2), np.array([0, 2])),
        (CartPoleVecEnv(2), np.array([0])),
        (PendulumVecEnv(2), np.zeros(3)),
    ],
)
def test_bad_actions_raise(env: Any, actions: np.ndarray) -> None:
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(actions)


def test_reset_seed_is_reproducible_and_spawn_is_independent() -> None:
    env = CartPoleVecEnv(4)
    first, _ = env.reset(seed=9)
    again, _ = env.reset(seed=9)
    assert np.array_equal(first, again)
    copy = env.spawn(num_envs=2, seed=9)
    assert copy is not env and copy.num_envs == 2
    assert np.array_equal(copy.reset(seed=9)[0], first[:2])


# -- SyncVectorEnv and make_vec_env -----------------------------------------------------


def test_sync_vector_env_autoresets_and_seeds_each_copy() -> None:
    envs = [CountdownEnv(2), CountdownEnv(3)]
    vec = SyncVectorEnv(envs)
    obs, _ = vec.reset(seed=10)
    assert [e.seeds for e in envs] == [[10], [11]]
    assert obs.shape == (2, 1)
    obs, reward, terminated, truncated, info = vec.step(np.array([1, 0]))
    assert reward.tolist() == [1.0, 0.0] and not terminated.any()
    obs, _, terminated, _, info = vec.step(np.array([0, 0]))
    assert terminated.tolist() == [True, False]
    assert info["final_observation"][:, 0].tolist() == [2.0, 2.0]
    assert obs[:, 0].tolist() == [0.0, 2.0]
    assert envs[0].seeds == [10, None]


def test_sync_vector_env_rejects_mixed_spaces_and_empty_lists() -> None:
    other = CountdownEnv()
    other.action_space = Discrete(3)
    with pytest.raises(ValueError, match="identical spaces"):
        SyncVectorEnv([CountdownEnv(), other])
    with pytest.raises(ValueError):
        SyncVectorEnv([])


def test_sync_vector_env_spawn_needs_a_factory() -> None:
    with pytest.raises(RuntimeError):
        SyncVectorEnv([CountdownEnv()]).spawn()
    vec = make_vec_env(lambda: CountdownEnv(4), num_envs=2, seed=1)
    assert isinstance(vec, SyncVectorEnv)
    fresh = vec.spawn(num_envs=3)
    assert isinstance(fresh, SyncVectorEnv) and fresh.num_envs == 3


def test_make_vec_env_names_and_validation() -> None:
    assert registered_envs() == ("CartPole-v1", "Pendulum-v1")
    env = make_vec_env("CartPole-v1", num_envs=5, seed=0)
    assert isinstance(env, CartPoleVecEnv) and env.num_envs == 5
    assert isinstance(env, VectorEnv) and isinstance(SyncVectorEnv([CountdownEnv()]), VectorEnv)
    with pytest.raises(KeyError):
        make_vec_env("NoSuchEnvironment-v0")
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            make_vec_env("CartPole-v1", num_envs=bad)


def test_spaces_validate_and_compare() -> None:
    with pytest.raises(ValueError):
        Discrete(0)
    with pytest.raises(ValueError):
        Box(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        Box(np.ones(2), np.zeros(2))
    inf = np.array([np.inf, 1.0])
    assert Box(-inf, inf) == Box(-inf, inf)
    assert Box(-inf, inf) != Box(np.zeros(2), np.ones(2))
    assert Box(np.zeros(3), np.ones(3)).shape == (3,)


# -- pinned constants and edges ----------------------------------------------------------


def test_spaces_and_limits_equal_gymnasiums() -> None:
    """The values Gymnasium 1.4.0 declares (checked against it when recorded)."""
    cart = CartPoleVecEnv()
    theta = 12 * 2 * math.pi / 360
    high = np.array([4.8, np.inf, 2 * theta, np.inf], dtype=np.float32)
    assert cart.num_envs == 1 and cart.observation_space == Box(-high, high)
    assert cart.action_space == Discrete(2) and cart.max_episode_steps == 500
    pend = PendulumVecEnv()
    speed = np.array([1.0, 1.0, 8.0], dtype=np.float32)
    assert pend.num_envs == 1 and pend.observation_space == Box(-speed, speed)
    two = np.array([2.0], dtype=np.float32)
    assert pend.action_space == Box(-two, two) and pend.max_episode_steps == 200
    assert inspect.signature(make_vec_env).parameters["num_envs"].default == 1


def test_cartpole_runs_500_steps_when_balanced() -> None:
    env = CartPoleVecEnv(1, seed=0)
    env.reset()
    for step in range(1, 501):
        env._state[:] = 0.0
        _, _, terminated, truncated, _ = env.step(np.array([step % 2]))
        assert not terminated[0]
        assert truncated[0] == (step == 500)


@pytest.mark.parametrize(
    "state",
    [[2.4, 0.0, 0.0, 0.0], [-2.4, 0.0, 0.0, 0.0], [0.0, 0.0, 12 * 2 * math.pi / 360, 0.0]],
)
def test_termination_is_strictly_beyond_the_threshold(state: list[float]) -> None:
    """Exactly at |x| = 2.4 or |theta| = 12 degrees the episode continues."""
    env = CartPoleVecEnv(1, seed=0)
    env.reset()
    env._state[:] = state
    env.tau = 0.0  # freeze the state on the threshold for one step
    _, _, terminated, _, _ = env.step(np.array([0]))
    assert not terminated[0]
    env._state[:] = np.array(state) * 1.0001
    _, _, terminated, _, _ = env.step(np.array([0]))
    assert terminated[0]


def test_pendulum_speed_is_clipped_to_eight() -> None:
    env = PendulumVecEnv(1, seed=0)
    env.reset()
    env._state[:] = [math.pi / 2, 7.99]
    obs, *_ = env.step(np.array([[2.0]], np.float32))
    assert obs[0, 2] == 8.0


def test_space_edges_and_immutability() -> None:
    assert Discrete(1).n == 1
    Box(np.zeros(2), np.zeros(2))  # low == high is a valid (degenerate) box
    assert Box(np.zeros(2), np.ones(2)) != Box(np.zeros(2), 2 * np.ones(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        Discrete(2).n = 3  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        Box(np.zeros(1), np.ones(1)).low = np.ones(1)  # type: ignore[misc]
    for bad in (0, -1):
        with pytest.raises(ValueError):
            CartPoleVecEnv(bad)


def test_space_comparison_across_kinds() -> None:
    from omni_mercury_engine.ml.rl_envs import _same_space

    @dataclasses.dataclass(frozen=True)
    class MultiBinary:
        n: int = 2
        shape: tuple[int, ...] = (2,)

    assert _same_space(Discrete(2), Discrete(2))
    assert not _same_space(Discrete(2), Box(np.zeros(2), np.ones(2)))
    assert not _same_space(Discrete(2), MultiBinary())
    assert not _same_space(Box(np.zeros(2), np.ones(2)), Box(np.ones(2), np.ones(2)))
    assert not _same_space(Box(np.zeros(2), np.ones(2)), Box(np.zeros(2), 2 * np.ones(2)))

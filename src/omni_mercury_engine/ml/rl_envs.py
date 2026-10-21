# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vectorized environments for :mod:`omni_mercury_engine.ml.ppo_trainer`.

Two kinds of vector environment share one interface (:class:`VectorEnv`):

* **Batched environments** step every copy in one NumPy operation, with no
  per-copy Python loop: :class:`CartPoleVecEnv` and :class:`PendulumVecEnv`
  reproduce Gymnasium's ``CartPole-v1`` and ``Pendulum-v1`` dynamics, reward,
  termination and time limits exactly (pinned against recorded Gymnasium
  trajectories in the tests), so results compare one-to-one with the
  published baselines on those tasks.
* :class:`SyncVectorEnv` steps a list of ordinary single environments with
  the Gymnasium API (``reset(seed=...) -> (obs, info)``,
  ``step(action) -> (obs, reward, terminated, truncated, info)``); Gymnasium
  itself is not required, any object with that interface works.

Every vector environment resets a finished copy in the same ``step`` call
and reports the observation it finished on in ``info["final_observation"]``
(row-aligned, valid where ``terminated | truncated``), so a learner can
bootstrap the value of a time-limit truncation instead of treating it as a
terminal state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Box",
    "CartPoleVecEnv",
    "Discrete",
    "PendulumVecEnv",
    "SyncVectorEnv",
    "VectorEnv",
    "make_vec_env",
    "registered_envs",
]

FloatArray = np.ndarray[Any, np.dtype[np.float32]]


@dataclass(frozen=True)
class Discrete:
    """A finite action space ``{0, ..., n - 1}``.

    Attributes:
        n: Number of actions (>= 1).
    """

    n: int

    def __post_init__(self) -> None:
        """Validate the space.

        Raises:
            ValueError: ``n`` is not a positive integer.
        """
        if not isinstance(self.n, int | np.integer) or self.n < 1:
            raise ValueError(f"Discrete space needs n >= 1, got {self.n!r}")


@dataclass(frozen=True, eq=False)
class Box:
    """A box in R^d (bounds may be infinite).

    Attributes:
        low: Lower bounds, shape ``shape`` (stored as float32).
        high: Upper bounds, shape ``shape`` (stored as float32).
    """

    low: np.ndarray[Any, Any]
    high: np.ndarray[Any, Any]

    def __post_init__(self) -> None:
        """Validate the bounds.

        Raises:
            ValueError: Mismatched shapes or ``low > high`` anywhere.
        """
        low = np.asarray(self.low, dtype=np.float32)
        high = np.asarray(self.high, dtype=np.float32)
        if low.shape != high.shape:
            raise ValueError(f"Box bounds differ in shape: {low.shape} vs {high.shape}")
        if np.any(low > high):
            raise ValueError("Box needs low <= high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of one element."""
        return tuple(self.low.shape)

    def __eq__(self, other: object) -> bool:
        """Equal bounds (infinities included)."""
        if not isinstance(other, Box):
            return NotImplemented
        return bool(np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high))

    __hash__ = object.__hash__


@runtime_checkable
class VectorEnv(Protocol):
    """What the PPO trainer needs from a vector environment."""

    num_envs: int
    observation_space: Any
    action_space: Any

    def reset(self, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Reset every copy; return ``(obs[num_envs, ...], info)``."""
        ...  # pragma: no cover - protocol signature

    def step(self, actions: np.ndarray[Any, Any]) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        dict[str, Any],
    ]:
        """Step every copy; finished copies reset in place (see module docstring)."""
        ...  # pragma: no cover - protocol signature


class _BatchedEnv:
    """Shared bookkeeping for the batched environments: seeding and time limits."""

    max_episode_steps: int

    def __init__(self, num_envs: int, seed: int | None) -> None:
        if not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
        self.num_envs = num_envs
        self._rng = np.random.default_rng(seed)
        self._elapsed = np.zeros(num_envs, dtype=np.int64)

    def _seed(self, seed: int | None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def spawn(self, num_envs: int | None = None, seed: int | None = None) -> Self:
        """A fresh copy of this environment (for evaluation), independently seeded."""
        return type(self)(num_envs or self.num_envs, seed=seed)


class CartPoleVecEnv(_BatchedEnv):
    """Gymnasium ``CartPole-v1``, batched: balance a pole on a moving cart.

    Barto, Sutton & Anderson (1983) dynamics with Gymnasium's constants and
    explicit-Euler integration (tau = 0.02 s); +1 reward per step including
    the step that terminates; termination past |x| = 2.4 or |theta| = 12
    degrees; truncation at 500 steps (also reported when the 500th step
    terminates, as Gymnasium's ``TimeLimit`` does); initial state uniform in
    [-0.05, 0.05].

    Args:
        num_envs: Number of copies stepped together.
        seed: Seed for the initial-state generator.
    """

    max_episode_steps = 500
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    theta_threshold_radians = 12 * 2 * math.pi / 360
    x_threshold = 2.4

    def __init__(self, num_envs: int = 1, seed: int | None = None) -> None:
        """Create the batch (call :meth:`reset` before stepping)."""
        super().__init__(num_envs, seed)
        high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high)
        self.action_space = Discrete(2)
        self._state = np.zeros((num_envs, 4), dtype=np.float64)

    def _initial(self, count: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._rng.uniform(-0.05, 0.05, size=(count, 4))

    def reset(self, seed: int | None = None) -> tuple[FloatArray, dict[str, Any]]:
        """Reset every copy.

        Args:
            seed: Reseed the initial-state generator first.

        Returns:
            ``(obs[num_envs, 4], {})``.
        """
        self._seed(seed)
        self._state = self._initial(self.num_envs)
        self._elapsed[:] = 0
        return self._state.astype(np.float32), {}

    def step(
        self, actions: np.ndarray[Any, Any]
    ) -> tuple[FloatArray, FloatArray, np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
        """Advance every copy by one step; finished copies reset in place.

        Args:
            actions: Integer array of shape ``(num_envs,)`` with values 0 or 1.

        Returns:
            ``(obs, reward, terminated, truncated, info)``; ``info`` holds
            ``final_observation`` (the pre-reset observation of every copy).

        Raises:
            ValueError: Wrong shape, or an action outside {0, 1}.
        """
        actions = np.asarray(actions).reshape(-1)
        if actions.shape != (self.num_envs,) or np.any((actions != 0) & (actions != 1)):
            raise ValueError(f"CartPole needs {self.num_envs} actions in {{0, 1}}")
        x, x_dot, theta, theta_dot = self._state.T
        force = np.where(actions == 1, self.force_mag, -self.force_mag)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * np.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self._state = np.stack([x, x_dot, theta, theta_dot], axis=1)
        self._elapsed += 1
        terminated = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        truncated = self._elapsed >= self.max_episode_steps
        reward = np.ones(self.num_envs, dtype=np.float32)
        final = self._state.astype(np.float32)
        done = terminated | truncated
        if done.any():
            self._state[done] = self._initial(int(done.sum()))
            self._elapsed[done] = 0
        return (
            self._state.astype(np.float32),
            reward,
            terminated,
            truncated,
            {"final_observation": final},
        )


class PendulumVecEnv(_BatchedEnv):
    """Gymnasium ``Pendulum-v1``, batched: swing a pendulum up and hold it.

    Gymnasium's constants (g = 10, m = l = 1, dt = 0.05 s, |torque| <= 2,
    |angular velocity| <= 8); cost ``theta^2 + 0.1 theta_dot^2 + 0.001 u^2``
    with theta wrapped to [-pi, pi); observation ``(cos theta, sin theta,
    theta_dot)``; never terminates, truncates at 200 steps; initial theta
    uniform in [-pi, pi], theta_dot uniform in [-1, 1].

    Args:
        num_envs: Number of copies stepped together.
        seed: Seed for the initial-state generator.
    """

    max_episode_steps = 200
    max_speed = 8.0
    max_torque = 2.0
    dt = 0.05
    g = 10.0
    m = 1.0
    l = 1.0  # noqa: E741 - Gymnasium's name for the pendulum length

    def __init__(self, num_envs: int = 1, seed: int | None = None) -> None:
        """Create the batch (call :meth:`reset` before stepping)."""
        super().__init__(num_envs, seed)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.observation_space = Box(-high, high)
        self.action_space = Box(
            np.array([-self.max_torque], dtype=np.float32),
            np.array([self.max_torque], dtype=np.float32),
        )
        self._state = np.zeros((num_envs, 2), dtype=np.float64)

    def _initial(self, count: int) -> np.ndarray[Any, np.dtype[np.float64]]:
        high = np.array([np.pi, 1.0])
        return self._rng.uniform(-high, high, size=(count, 2))

    def _obs(self, state: np.ndarray[Any, np.dtype[np.float64]]) -> FloatArray:
        th, thdot = state[:, 0], state[:, 1]
        return np.stack([np.cos(th), np.sin(th), thdot], axis=1).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[FloatArray, dict[str, Any]]:
        """Reset every copy.

        Args:
            seed: Reseed the initial-state generator first.

        Returns:
            ``(obs[num_envs, 3], {})``.
        """
        self._seed(seed)
        self._state = self._initial(self.num_envs)
        self._elapsed[:] = 0
        return self._obs(self._state), {}

    def step(
        self, actions: np.ndarray[Any, Any]
    ) -> tuple[FloatArray, FloatArray, np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
        """Advance every copy by one step; finished copies reset in place.

        Args:
            actions: Torques, shape ``(num_envs, 1)`` or ``(num_envs,)``;
                clipped to the torque limit as Gymnasium does.

        Returns:
            ``(obs, reward, terminated, truncated, info)``; ``info`` holds
            ``final_observation``.

        Raises:
            ValueError: Wrong number of actions.
        """
        u = np.asarray(actions).reshape(-1)
        if u.shape != (self.num_envs,):
            raise ValueError(f"Pendulum needs {self.num_envs} torques")
        # Keep the caller's float precision, as Gymnasium does (a float32 torque
        # rounds its terms in float32 before they meet the float64 state).
        u = np.clip(
            u.astype(np.result_type(u.dtype, np.float32)), -self.max_torque, self.max_torque
        )
        th, thdot = self._state[:, 0], self._state[:, 1]
        costs = ((th + np.pi) % (2 * np.pi) - np.pi) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        newthdot = (
            thdot
            + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        )
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * self.dt
        self._state = np.stack([newth, newthdot], axis=1)
        self._elapsed += 1
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = self._elapsed >= self.max_episode_steps
        final = self._obs(self._state)
        if truncated.any():
            self._state[truncated] = self._initial(int(truncated.sum()))
            self._elapsed[truncated] = 0
        return (
            self._obs(self._state),
            (-costs).astype(np.float32),
            terminated,
            truncated,
            {"final_observation": final},
        )


def _same_space(a: Any, b: Any) -> bool:
    """Whether two spaces (Gymnasium's or this module's) describe the same set."""
    if hasattr(a, "n") != hasattr(b, "n"):
        return False
    if hasattr(a, "n"):
        return bool(
            int(a.n) == int(b.n)
            and int(getattr(a, "start", 0)) == int(getattr(b, "start", 0))
            and tuple(getattr(a, "shape", ()) or ()) == tuple(getattr(b, "shape", ()) or ())
        )
    return bool(
        np.array_equal(np.asarray(a.low), np.asarray(b.low))
        and np.array_equal(np.asarray(a.high), np.asarray(b.high))
    )


class SyncVectorEnv:
    """Step a list of single Gymnasium-API environments as one vector environment.

    Each copy is reset in the same :meth:`step` call that finishes it, and its
    last observation is kept in ``info["final_observation"]``.

    Args:
        envs: The environments (at least one), all with the same spaces.
        factory: Zero-argument callable building one more such environment;
            enables :meth:`spawn`.

    Raises:
        ValueError: An empty list, or environments with differing spaces.
    """

    def __init__(self, envs: Sequence[Any], factory: Callable[[], Any] | None = None) -> None:
        """Wrap the environments."""
        if not envs:
            raise ValueError("SyncVectorEnv needs at least one environment")
        self.envs = list(envs)
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        for env in self.envs[1:]:
            if not (
                _same_space(env.observation_space, self.observation_space)
                and _same_space(env.action_space, self.action_space)
            ):
                raise ValueError("SyncVectorEnv needs environments with identical spaces")
        self.factory = factory

    def spawn(self, num_envs: int | None = None, seed: int | None = None) -> SyncVectorEnv:
        """A fresh vector environment built by ``factory`` (seeded on reset).

        Raises:
            RuntimeError: No ``factory`` was given.
        """
        if self.factory is None:
            raise RuntimeError("this SyncVectorEnv has no factory to build fresh copies")
        return SyncVectorEnv(
            [self.factory() for _ in range(num_envs or self.num_envs)], factory=self.factory
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Reset every copy, seeding copy ``i`` with ``seed + i`` when ``seed`` is given."""
        observations = []
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=None if seed is None else seed + i)
            observations.append(np.asarray(obs))
        return np.stack(observations), {}

    def step(self, actions: np.ndarray[Any, Any]) -> tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        dict[str, Any],
    ]:
        """Step every copy with its row of ``actions``; finished copies reset in place."""
        observations, finals = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        discrete = hasattr(self.action_space, "n")
        for i, env in enumerate(self.envs):
            action = int(actions[i]) if discrete else np.asarray(actions[i])
            obs, reward, term, trunc, _ = env.step(action)
            rewards[i] = float(reward)
            terminated[i] = bool(term)
            truncated[i] = bool(trunc)
            finals.append(np.asarray(obs))
            if term or trunc:
                obs, _ = env.reset()
            observations.append(np.asarray(obs))
        return (
            np.stack(observations),
            rewards,
            terminated,
            truncated,
            {"final_observation": np.stack(finals)},
        )


_REGISTRY: dict[str, Callable[[int, int | None], VectorEnv]] = {
    "CartPole-v1": lambda n, seed: CartPoleVecEnv(n, seed=seed),
    "Pendulum-v1": lambda n, seed: PendulumVecEnv(n, seed=seed),
}


def registered_envs() -> tuple[str, ...]:
    """Names :func:`make_vec_env` builds without any third-party package."""
    return tuple(sorted(_REGISTRY))


def make_vec_env(
    env: str | Callable[[], Any], num_envs: int = 1, seed: int | None = None
) -> VectorEnv:
    """Build a vector environment by name or from a single-environment factory.

    Args:
        env: A registered name (:func:`registered_envs`; batched), another
            Gymnasium environment id (needs the ``gymnasium`` package), or a
            zero-argument factory returning one Gymnasium-API environment.
        num_envs: Number of copies (>= 1).
        seed: Seed (batched: the initial-state generator; otherwise copy ``i``
            is reset with ``seed + i`` on the first reset).

    Returns:
        The vector environment.

    Raises:
        KeyError: A name neither registered nor known to an installed Gymnasium.
        ValueError: ``num_envs`` below 1.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError(f"num_envs must be a positive integer, got {num_envs!r}")
    if isinstance(env, str):
        if env in _REGISTRY:
            return _REGISTRY[env](num_envs, seed)
        factory = _gymnasium_factory(env)
    else:
        factory = env
    vec = SyncVectorEnv([factory() for _ in range(num_envs)], factory=factory)
    if seed is not None:
        vec.reset(seed=seed)
    return vec


def _gymnasium_factory(env_id: str) -> Callable[[], Any]:
    try:
        import gymnasium
    except ImportError:
        raise KeyError(
            f"unknown environment {env_id!r}; registered: {registered_envs()} "
            "(other ids need the 'gymnasium' package)"
        ) from None
    if env_id not in gymnasium.registry:
        raise KeyError(f"unknown environment {env_id!r}; registered: {registered_envs()}")
    return lambda: gymnasium.make(env_id)

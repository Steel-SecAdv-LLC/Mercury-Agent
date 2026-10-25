#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record Gymnasium trajectories as test fixtures (tests/fixtures/rl_envs).

``omni_mercury_engine.ml.rl_envs`` re-implements Gymnasium's ``CartPole-v1``
and ``Pendulum-v1`` as batched NumPy environments; its tests replay these
recordings through the batched versions and require every observation,
reward and flag to match bit for bit. Each environment is reset once with a
fixed seed and then driven by seeded random actions, across many episode
boundaries (each later reset unseeded, as an autoresetting vector
environment does).

Needs ``gymnasium`` (not a Mercury dependency); the version used is written
into the fixture's metadata.

Usage::

    python scripts/record_gymnasium_trajectories.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "rl_envs"
    / "gymnasium_trajectories.npz"
)

#: env id -> (reset seed, steps, action sampler)
RECORDINGS: dict[str, tuple[int, int, Any]] = {
    "CartPole-v1": (7, 3000, lambda rng: rng.integers(0, 2)),
    "Pendulum-v1": (11, 1000, lambda rng: rng.uniform(-2.5, 2.5, size=1).astype(np.float32)),
}


def record(env_id: str, seed: int, steps: int, sample: Any) -> dict[str, np.ndarray]:
    """One trajectory: per-step action, observation, reward and flags, plus resets."""
    env = gym.make(env_id)
    rng = np.random.default_rng(0)
    first, _ = env.reset(seed=seed)
    actions, observations, rewards, terminated, truncated, resets = [], [], [], [], [], [first]
    for _ in range(steps):
        action = sample(rng)
        obs, reward, term, trunc, _ = env.step(action)
        actions.append(np.asarray(action))
        observations.append(obs)
        rewards.append(np.float64(reward))
        terminated.append(term)
        truncated.append(trunc)
        if term or trunc:
            obs, _ = env.reset()
            resets.append(obs)
    return {
        "seed": np.int64(seed),
        "actions": np.stack(actions),
        "observations": np.stack(observations),
        "rewards": np.asarray(rewards),
        "terminated": np.asarray(terminated),
        "truncated": np.asarray(truncated),
        "resets": np.stack(resets),
    }


def main() -> None:
    """Record every environment into the fixture file."""
    arrays: dict[str, np.ndarray] = {}
    for env_id, (seed, steps, sample) in RECORDINGS.items():
        for key, value in record(env_id, seed, steps, sample).items():
            arrays[f"{env_id}/{key}"] = value
    meta = {"gymnasium": gym.__version__, "numpy": np.__version__, "envs": sorted(RECORDINGS)}
    arrays["meta"] = np.asarray(json.dumps(meta))
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes): {meta}")


if __name__ == "__main__":
    main()

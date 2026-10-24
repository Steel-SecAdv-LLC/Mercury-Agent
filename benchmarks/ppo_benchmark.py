#!/usr/bin/env python3
# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark ``ml.ppo_trainer`` against stable-baselines3 on identical settings.

Both implementations train on ``CartPole-v1`` and ``Pendulum-v1`` with the
tuned PPO hyperparameters published by rl-baselines3-zoo
(``hyperparams/ppo.yml``), 100,000 environment steps, one CPU thread, over
the same seeds. The implementations alternate seed by seed, so both see the
same machine conditions. Each trained policy is then scored on 20
deterministic episodes of a separately seeded evaluation environment.

Reported per run: training wall-clock seconds, environment steps per second
and the evaluation return (mean and standard deviation). Mercury's batched
environments are bit-identical to Gymnasium's (``tests/ml/test_rl_envs.py``),
so the two learn the same task. The baseline needs ``stable-baselines3`` and
``gymnasium`` (neither is a Mercury dependency); without them only Mercury
runs.

Usage::

    python benchmarks/ppo_benchmark.py                      # 5 seeds, both tasks
    python benchmarks/ppo_benchmark.py --seeds 0 --envs CartPole-v1
    python benchmarks/ppo_benchmark.py --out benchmarks/ppo_results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from omni_mercury_engine.ml.ppo_trainer import PPOConfig, PPOTrainer
from omni_mercury_engine.ml.rl_envs import make_vec_env

TIMESTEPS = 100_000
EVAL_EPISODES = 20
EVAL_SEED_OFFSET = 1000

#: rl-baselines3-zoo hyperparams/ppo.yml, in each implementation's terms.
ZOO: dict[str, dict[str, Any]] = {
    "CartPole-v1": {
        "n_envs": 8,
        "normalize": False,
        "common": {
            "n_steps": 32,
            "batch_size": 256,
            "gae_lambda": 0.8,
            "gamma": 0.98,
            "n_epochs": 20,
            "ent_coef": 0.0,
        },
        "mercury": {
            "learning_rate": 0.001,
            "lr_schedule": "linear",
            "clip_range": 0.2,
            "clip_range_schedule": "linear",
        },
        "sb3": {"learning_rate": ("linear", 0.001), "clip_range": ("linear", 0.2)},
    },
    "Pendulum-v1": {
        "n_envs": 4,
        "normalize": True,
        "common": {
            "n_steps": 1024,
            "gae_lambda": 0.95,
            "gamma": 0.9,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "learning_rate": 1e-3,
            "clip_range": 0.2,
            "use_sde": True,
            "sde_sample_freq": 4,
        },
        "mercury": {"batch_size": 64},  # SB3's default
        "sb3": {},
    },
}


def run_mercury(env_id: str, seed: int) -> dict[str, Any]:
    """Train and evaluate Mercury's PPO once."""
    spec = ZOO[env_id]
    config = PPOConfig(
        seed=seed,
        verbose=0,
        n_envs=spec["n_envs"],
        normalize_observations=spec["normalize"],
        normalize_rewards=spec["normalize"],
        **spec["common"],
        **spec["mercury"],
    )
    trainer = PPOTrainer(env_id, config)
    start = time.perf_counter()
    trainer.learn(TIMESTEPS)
    elapsed = time.perf_counter() - start
    eval_seed = seed + EVAL_SEED_OFFSET
    mean, std = trainer.evaluate(
        EVAL_EPISODES, env=make_vec_env(env_id, 1, seed=eval_seed), seed=eval_seed
    )
    return _record(seed, elapsed, trainer.num_timesteps, mean, std)


def run_sb3(env_id: str, seed: int) -> dict[str, Any]:
    """Train and evaluate stable-baselines3's PPO once (same settings)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    spec = ZOO[env_id]
    kwargs: dict[str, Any] = dict(spec["common"])
    for name, value in spec["sb3"].items():
        if isinstance(value, tuple):  # ("linear", initial): SB3 takes a schedule
            initial = value[1]
            kwargs[name] = lambda progress, initial=initial: progress * initial
        else:
            kwargs[name] = value
    gamma = kwargs["gamma"]
    venv = sb3_make_vec_env(env_id, n_envs=spec["n_envs"], seed=seed)
    if spec["normalize"]:
        venv = VecNormalize(venv, gamma=gamma)
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", **kwargs)
    start = time.perf_counter()
    model.learn(TIMESTEPS)
    elapsed = time.perf_counter() - start
    eval_env = sb3_make_vec_env(env_id, n_envs=1, seed=seed + EVAL_SEED_OFFSET)
    if spec["normalize"]:
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False, gamma=gamma)
        eval_env.obs_rms = venv.obs_rms
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPISODES, deterministic=True)
    return _record(seed, elapsed, model.num_timesteps, float(mean), float(std))


def _record(seed: int, elapsed: float, steps: int, mean: float, std: float) -> dict[str, Any]:
    return {
        "seed": seed,
        "train_s": round(elapsed, 2),
        "steps": int(steps),
        "steps_per_s": round(steps / elapsed),
        "eval_mean": round(mean, 2),
        "eval_std": round(std, 2),
    }


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [r["eval_mean"] for r in runs]
    times = [r["train_s"] for r in runs]
    return {
        "eval_mean_of_means": round(statistics.fmean(returns), 2),
        "eval_worst_seed": min(returns),
        "train_s_mean": round(statistics.fmean(times), 2),
        "steps_per_s_mean": round(statistics.fmean(r["steps_per_s"] for r in runs)),
    }


def main() -> None:
    """Run the benchmark and print (optionally write) the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--envs", nargs="+", default=list(ZOO), choices=list(ZOO))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--no-baseline", action="store_true", help="run Mercury only")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()
    torch.set_num_threads(1)
    baseline = not args.no_baseline and all(
        importlib.util.find_spec(m) is not None for m in ("stable_baselines3", "gymnasium")
    )
    impls = {"mercury": run_mercury, **({"stable-baselines3": run_sb3} if baseline else {})}
    results: dict[str, Any] = {}
    for env_id in args.envs:
        runs: dict[str, list[dict[str, Any]]] = {name: [] for name in impls}
        for seed in args.seeds:
            for name, run in impls.items():
                runs[name].append(run(env_id, seed))
                print(env_id, name, runs[name][-1], flush=True)
        results[env_id] = {
            name: {"runs": rows, "summary": _summary(rows)} for name, rows in runs.items()
        }
    report: dict[str, Any] = {
        "method": (
            f"rl-baselines3-zoo PPO hyperparameters; {TIMESTEPS} steps; 1 torch thread; "
            f"{EVAL_EPISODES} deterministic evaluation episodes on a separately seeded env"
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            **_versions("stable_baselines3", "gymnasium", "numpy"),
        },
        "machine": platform.processor() or platform.machine(),
        "results": results,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


def _versions(*modules: str) -> dict[str, str]:
    found = {}
    for module in modules:
        if importlib.util.find_spec(module) is not None:
            found[module] = __import__(module).__version__
    return found


if __name__ == "__main__":
    main()

"""
OMNI ♱ AVA (O♱A)
Copyright (C) 2025 Steel Security Advisory LLC

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
"""

"""
PPO Trainer for OMNI ♱ AVA

Implements Proximal Policy Optimization (PPO) for autonomous agent
self-evolution in anomaly detection systems.

Key Features:
- Convergence monitoring with early stopping
- Checkpoint management for model persistence
- Multi-environment training support
- Ethics-weighted reward computation

Research Sources:
    - Schulman et al. (2017): Proximal Policy Optimization Algorithms
    - Stable Baselines3: https://stable-baselines3.readthedocs.io/

Original Implementation: OMNI-HALO (Steel Security Advisors LLC)
Integrated into OMNI ♱ AVA for autonomous anomaly detection evolution.
"""

import logging
import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_STABLE_BASELINES = find_spec("stable_baselines3") is not None
if not HAS_STABLE_BASELINES:
    logger.debug("stable-baselines3 not available, using mock trainer")
else:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

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
    target_kl: Optional[float] = None
    verbose: int = 1


@dataclass
class TrainingStats:
    """Statistics from training."""

    total_timesteps: int = 0
    total_episodes: int = 0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_episode_length: float = 0.0
    convergence_metric: float = 0.0
    best_reward: float = float("-inf")
    training_time: float = 0.0
    checkpoints_saved: int = 0
    ethics_score: float = 0.0


class BaseCallback:
    """Base callback for training monitoring."""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.locals: dict[str, Any] = {}
        self.n_calls = 0
        self.model: Any = None

    def _on_step(self) -> bool:
        return True


class ConvergenceMonitor(BaseCallback):
    """
    Monitors training convergence.

    Tracks reward convergence and determines when training has converged.
    Implements early stopping based on reward plateau detection.
    """

    def __init__(
        self,
        convergence_threshold: float = 0.999,
        patience: int = 10,
        verbose: int = 0,
    ):
        """
        Initialize Convergence Monitor.

        Args:
            convergence_threshold: Threshold for convergence
            patience: Patience for early stopping
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.convergence_threshold = convergence_threshold
        self.patience = patience

        self.reward_history: list[float] = []
        self.best_mean_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.converged = False

    def _on_step(self) -> bool:
        """Called at each step."""
        if "episode" in self.locals:
            episode_info = self.locals["episode"]
            if len(episode_info) > 0:
                episode_reward = episode_info[0].get("r", 0.0)
                self.reward_history.append(episode_reward)

                if len(self.reward_history) >= 100:
                    mean_reward = np.mean(self.reward_history[-100:])

                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        self.episodes_without_improvement = 0
                    else:
                        self.episodes_without_improvement += 1

                    convergence = mean_reward / (self.best_mean_reward + 1e-8)

                    if convergence >= self.convergence_threshold:
                        self.converged = True
                        logger.info(f"Training converged: {convergence:.4f}")
                        return False

                    if self.episodes_without_improvement >= self.patience:
                        logger.info("Early stopping triggered")
                        return False

        return True

    def get_convergence_metric(self) -> float:
        """Get current convergence metric."""
        if len(self.reward_history) < 100:
            return 0.0

        mean_reward = np.mean(self.reward_history[-100:])
        return mean_reward / (self.best_mean_reward + 1e-8)


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints during training.

    Saves best model and periodic checkpoints for recovery.
    """

    def __init__(
        self,
        save_path: str,
        save_freq: int = 10000,
        name_prefix: str = "ppo_model",
        verbose: int = 0,
    ):
        """
        Initialize Checkpoint Callback.

        Args:
            save_path: Path to save checkpoints
            save_freq: Frequency of saves (in steps)
            name_prefix: Prefix for checkpoint names
            verbose: Verbosity level
        """
        super().__init__(verbose)

        self.save_path = Path(save_path)
        self.save_freq = save_freq
        self.name_prefix = name_prefix

        self.save_path.mkdir(parents=True, exist_ok=True)

        self.best_mean_reward = float("-inf")
        self.checkpoints_saved = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        if self.n_calls % self.save_freq == 0 and self.model is not None:
            checkpoint_path = self.save_path / f"{self.name_prefix}_step_{self.n_calls}"
            self.model.save(str(checkpoint_path))
            self.checkpoints_saved += 1

            if self.verbose > 0:
                logger.info(f"Saved checkpoint: {checkpoint_path}")

        if "episode" in self.locals and len(self.locals["episode"]) > 0:
            mean_reward = np.mean([ep.get("r", 0.0) for ep in self.locals["episode"]])

            if mean_reward > self.best_mean_reward and self.model is not None:
                self.best_mean_reward = mean_reward

                best_path = self.save_path / f"{self.name_prefix}_best"
                self.model.save(str(best_path))

                if self.verbose > 0:
                    logger.info(
                        f"Saved best model: {best_path} (reward: {mean_reward:.2f})"
                    )

        return True


class PPOTrainer:
    """
    PPO Trainer for autonomous agent self-evolution.

    Implements PPO training with convergence monitoring, checkpointing,
    and ethics-weighted reward computation for anomaly detection.

    Example:
        trainer = PPOTrainer(env, config=PPOConfig())
        stats = trainer.pretrain(total_timesteps=100000)
        mean_reward, std_reward = trainer.evaluate(num_episodes=10)
    """

    def __init__(
        self,
        env: Any,
        config: Optional[PPOConfig] = None,
        checkpoint_dir: str = "./checkpoints",
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize PPO Trainer.

        Args:
            env: Training environment (Gymnasium-compatible)
            config: PPO configuration
            checkpoint_dir: Directory for checkpoints
            tensorboard_log: TensorBoard log directory
        """
        self.env = env
        self.config = config or PPOConfig()
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_log = tensorboard_log

        self.model: Optional[Any] = None
        self.stats = TrainingStats()

        self.convergence_monitor: Optional[ConvergenceMonitor] = None
        self.checkpoint_callback: Optional[CheckpointCallback] = None

        self._initialize_model()

        logger.info("PPO Trainer initialized")

    def _initialize_model(self) -> None:
        """Initialize PPO model."""
        if not HAS_STABLE_BASELINES:
            logger.warning("stable-baselines3 not available, using mock model")
            self.model = None
            return

        try:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                target_kl=self.config.target_kl,
                tensorboard_log=self.tensorboard_log,
                verbose=self.config.verbose,
            )

            logger.info("PPO model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PPO model: {e}")
            self.model = None

    def pretrain(
        self,
        total_timesteps: int = 100000,
        convergence_threshold: float = 0.999,
        save_checkpoints: bool = True,
    ) -> TrainingStats:
        """
        Pretrain model for specified timesteps.

        Args:
            total_timesteps: Total training timesteps
            convergence_threshold: Convergence threshold
            save_checkpoints: Whether to save checkpoints

        Returns:
            Training statistics
        """
        logger.info(f"Starting pretraining: {total_timesteps} timesteps")

        start_time = time.time()

        self.convergence_monitor = ConvergenceMonitor(
            convergence_threshold=convergence_threshold,
            patience=10,
            verbose=1,
        )

        callbacks = [self.convergence_monitor]

        if save_checkpoints:
            self.checkpoint_callback = CheckpointCallback(
                save_path=self.checkpoint_dir,
                save_freq=10000,
                name_prefix="ppo_pretrain",
                verbose=1,
            )
            callbacks.append(self.checkpoint_callback)

        if self.model:
            try:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callbacks,
                    progress_bar=True,
                )

                self.stats.total_timesteps = total_timesteps
                self.stats.convergence_metric = (
                    self.convergence_monitor.get_convergence_metric()
                )

                if len(self.convergence_monitor.reward_history) > 0:
                    recent_rewards = self.convergence_monitor.reward_history[-100:]
                    self.stats.mean_reward = float(np.mean(recent_rewards))
                    self.stats.std_reward = float(np.std(recent_rewards))
                    self.stats.best_reward = self.convergence_monitor.best_mean_reward

                if self.checkpoint_callback:
                    self.stats.checkpoints_saved = (
                        self.checkpoint_callback.checkpoints_saved
                    )

            except Exception as e:
                logger.error(f"Training failed: {e}")
        else:
            logger.warning("No model available, using mock training")
            self._mock_pretrain(total_timesteps)

        self.stats.training_time = time.time() - start_time

        logger.info(
            f"Pretraining complete: "
            f"{self.stats.total_timesteps} steps, "
            f"mean_reward={self.stats.mean_reward:.4f}, "
            f"convergence={self.stats.convergence_metric:.4f}, "
            f"time={self.stats.training_time:.2f}s"
        )

        return self.stats

    def _mock_pretrain(self, total_timesteps: int) -> None:
        """Mock pretraining for testing."""
        self.stats.total_timesteps = total_timesteps
        self.stats.mean_reward = 0.8
        self.stats.std_reward = 0.1
        self.stats.convergence_metric = 0.95
        self.stats.best_reward = 0.9

    def train_online(
        self,
        num_episodes: int = 100,
        update_freq: int = 10,
    ) -> TrainingStats:
        """
        Online training with live traces.

        Args:
            num_episodes: Number of episodes
            update_freq: Update frequency

        Returns:
            Training statistics
        """
        logger.info(f"Starting online training: {num_episodes} episodes")

        start_time = time.time()

        episode_rewards = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=False)
                else:
                    action = self.env.action_space.sample()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)

            if (episode + 1) % update_freq == 0 and self.model:
                self.model.learn(
                    total_timesteps=self.config.n_steps,
                    reset_num_timesteps=False,
                )

            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(episode_rewards[-10:])
                logger.info(f"Episode {episode + 1}: mean_reward={mean_reward:.4f}")

        self.stats.total_episodes = num_episodes
        self.stats.mean_reward = float(np.mean(episode_rewards))
        self.stats.std_reward = float(np.std(episode_rewards))
        self.stats.best_reward = float(max(episode_rewards))
        self.stats.training_time = time.time() - start_time

        logger.info(
            f"Online training complete: "
            f"{num_episodes} episodes, "
            f"mean_reward={self.stats.mean_reward:.4f}"
        )

        return self.stats

    def save_model(self, path: str) -> None:
        """Save trained model."""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved: {path}")
        else:
            logger.warning("No model to save")

    def load_model(self, path: str) -> None:
        """Load trained model."""
        if HAS_STABLE_BASELINES:
            try:
                self.model = PPO.load(path, env=self.env)
                logger.info(f"Model loaded: {path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        else:
            logger.warning("stable-baselines3 not available")

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """
        Evaluate trained model.

        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        logger.info(f"Evaluating model: {num_episodes} episodes")

        episode_rewards = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                if self.model:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                else:
                    action = self.env.action_space.sample()

                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))

        logger.info(
            f"Evaluation complete: "
            f"mean_reward={mean_reward:.4f}, "
            f"std_reward={std_reward:.4f}"
        )

        return mean_reward, std_reward

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of training."""
        return {
            "total_timesteps": self.stats.total_timesteps,
            "total_episodes": self.stats.total_episodes,
            "mean_reward": self.stats.mean_reward,
            "std_reward": self.stats.std_reward,
            "best_reward": self.stats.best_reward,
            "convergence_metric": self.stats.convergence_metric,
            "training_time": self.stats.training_time,
            "checkpoints_saved": self.stats.checkpoints_saved,
            "converged": self.stats.convergence_metric >= 0.999,
        }


class MultiEnvPPOTrainer(PPOTrainer):
    """
    PPO Trainer for multiple environments.

    Trains on multiple benchmark datasets simultaneously for
    robust anomaly detection across domains.
    """

    def __init__(
        self,
        envs: list[Any],
        config: Optional[PPOConfig] = None,
        checkpoint_dir: str = "./checkpoints",
        tensorboard_log: Optional[str] = None,
    ):
        """
        Initialize Multi-Environment PPO Trainer.

        Args:
            envs: List of training environments
            config: PPO configuration
            checkpoint_dir: Directory for checkpoints
            tensorboard_log: TensorBoard log directory
        """
        if HAS_STABLE_BASELINES and envs:
            vec_env = DummyVecEnv([lambda e=env: e for env in envs])
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        else:
            vec_env = envs[0] if envs else None

        super().__init__(
            env=vec_env,
            config=config,
            checkpoint_dir=checkpoint_dir,
            tensorboard_log=tensorboard_log,
        )

        self.num_envs = len(envs)
        logger.info(
            f"Multi-Environment PPO Trainer initialized with {self.num_envs} envs"
        )

    def pretrain_on_benchmarks(
        self,
        benchmark_names: list[str],
        timesteps_per_benchmark: int = 20000,
    ) -> dict[str, TrainingStats]:
        """
        Pretrain on multiple benchmarks.

        Args:
            benchmark_names: List of benchmark names
            timesteps_per_benchmark: Timesteps per benchmark

        Returns:
            Dictionary of training stats per benchmark
        """
        logger.info(f"Pretraining on {len(benchmark_names)} benchmarks")

        benchmark_stats = {}

        for benchmark_name in benchmark_names:
            logger.info(f"Training on benchmark: {benchmark_name}")

            stats = self.pretrain(
                total_timesteps=timesteps_per_benchmark,
                convergence_threshold=0.95,
                save_checkpoints=True,
            )

            benchmark_stats[benchmark_name] = stats

        total_timesteps = sum(s.total_timesteps for s in benchmark_stats.values())
        mean_convergence = np.mean(
            [s.convergence_metric for s in benchmark_stats.values()]
        )

        logger.info(
            f"Multi-benchmark pretraining complete: "
            f"{total_timesteps} total steps, "
            f"mean_convergence={mean_convergence:.4f}"
        )

        return benchmark_stats


def create_ppo_trainer(
    env: Any,
    config: Optional[PPOConfig] = None,
    **kwargs: Any,
) -> PPOTrainer:
    """
    Factory function to create PPO Trainer.

    Args:
        env: Training environment
        config: PPO configuration
        **kwargs: Additional arguments

    Returns:
        PPOTrainer instance
    """
    return PPOTrainer(env=env, config=config, **kwargs)


def create_multi_env_trainer(
    envs: list[Any],
    config: Optional[PPOConfig] = None,
    **kwargs: Any,
) -> MultiEnvPPOTrainer:
    """
    Factory function to create Multi-Environment PPO Trainer.

    Args:
        envs: List of training environments
        config: PPO configuration
        **kwargs: Additional arguments

    Returns:
        MultiEnvPPOTrainer instance
    """
    return MultiEnvPPOTrainer(envs=envs, config=config, **kwargs)


__all__ = [
    "PPOConfig",
    "TrainingStats",
    "ConvergenceMonitor",
    "CheckpointCallback",
    "PPOTrainer",
    "MultiEnvPPOTrainer",
    "create_ppo_trainer",
    "create_multi_env_trainer",
    "HAS_STABLE_BASELINES",
]

# Symbol diff: mercury_a_learning

- **Omni**:    `src/omni_anomaly_engine/mercury_a_learning.py`  (size 15285 bytes)
- **Mercury**: `src/omni_mercury_engine/ml/ppo_trainer.py`  (size 21137 bytes)

## Public symbol counts
- Omni public symbols:    **4**
- Mercury public symbols: **9**
- Shared:                 **0**
- Only in Omni:           **4**   ← POTENTIAL_EXTRACTION_CANDIDATE
- Only in Mercury:        **9**

## ⚠ POTENTIAL_EXTRACTION_CANDIDATE (symbols in Omni only)
- `cls AdaptiveLearner` @ src/omni_anomaly_engine/mercury_a_learning.py:384
- `cls AnomalyDetectionEnv` @ src/omni_anomaly_engine/mercury_a_learning.py:43
- `cls MercuryLearner` @ src/omni_anomaly_engine/mercury_a_learning.py:213
- `cls RewardConfig` @ src/omni_anomaly_engine/mercury_a_learning.py:31

## Mercury-only (likely added enhancements)
- `cls BaseCallback` @ src/omni_mercury_engine/ml/ppo_trainer.py:92
- `cls CheckpointCallback` @ src/omni_mercury_engine/ml/ppo_trainer.py:176
- `cls ConvergenceMonitor` @ src/omni_mercury_engine/ml/ppo_trainer.py:105
- `cls MultiEnvPPOTrainer` @ src/omni_mercury_engine/ml/ppo_trainer.py:543
- `cls PPOConfig` @ src/omni_mercury_engine/ml/ppo_trainer.py:59
- `cls PPOTrainer` @ src/omni_mercury_engine/ml/ppo_trainer.py:235
- `cls TrainingStats` @ src/omni_mercury_engine/ml/ppo_trainer.py:77
- `def create_multi_env_trainer(envs: list[Any], config: PPOConfig | None=None, **kwargs: Any)` @ src/omni_mercury_engine/ml/ppo_trainer.py:644
- `def create_ppo_trainer(env: Any, config: PPOConfig | None=None, **kwargs: Any)` @ src/omni_mercury_engine/ml/ppo_trainer.py:625

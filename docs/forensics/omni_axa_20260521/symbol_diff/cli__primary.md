# Symbol diff: cli__primary

- **Omni**:    `src/omni_anomaly_engine/cli.py`  (size 3765 bytes)
- **Mercury**: `src/omni_mercury_engine/cli.py`  (size 39815 bytes)

## Public symbol counts
- Omni public symbols:    **6**
- Mercury public symbols: **14**
- Shared:                 **6**
- Only in Omni:           **0**   ← POTENTIAL_EXTRACTION_CANDIDATE
- Only in Mercury:        **8**

## Mercury-only (likely added enhancements)
- `def physics()` @ src/omni_mercury_engine/cli.py:182
- `def physics_dynamics(input: str, output: str, threshold: float, time_step: float, jerk_sensitivity: float, chaos_threshold: float)` @ src/omni_mercury_engine/cli.py:278
- `def physics_integrated(spectral_input: str, dynamics_input: str, uiux_input: str, output: str, threshold: float, fusion_weights: str)` @ src/omni_mercury_engine/cli.py:502
- `def physics_list()` @ src/omni_mercury_engine/cli.py:663
- `def physics_spectral(input: str, output: str, threshold: float, mode: str, sample_rate: float)` @ src/omni_mercury_engine/cli.py:199
- `def physics_uiux(input: str, output: str, threshold: float, rage_threshold: float, bot_threshold: float)` @ src/omni_mercury_engine/cli.py:361
- `def serve(host: str, port: int, workers: int, reload: bool, log_level: str)` @ src/omni_mercury_engine/cli.py:754
- `def voice(domain: str, model: str, offline: bool)` @ src/omni_mercury_engine/cli.py:811

## Shared symbols — AST signature diff
### `biometric` — = IDENTICAL
- Omni:    `def biometric(reference: str, test: str)`  @ src/omni_anomaly_engine/cli.py:47
- Mercury: `def biometric(reference: str, test: str)`  @ src/omni_mercury_engine/cli.py:90
### `detect` — = IDENTICAL
- Omni:    `def detect(input: str, detector: str, output: str, threshold: float)`  @ src/omni_anomaly_engine/cli.py:26
- Mercury: `def detect(input: str, detector: str, output: str, threshold: float)`  @ src/omni_mercury_engine/cli.py:69
### `explain` — = IDENTICAL
- Omni:    `def explain(input: str, model: str)`  @ src/omni_anomaly_engine/cli.py:85
- Mercury: `def explain(input: str, model: str)`  @ src/omni_mercury_engine/cli.py:132
### `main` — = IDENTICAL
- Omni:    `def main()`  @ src/omni_anomaly_engine/cli.py:17
- Mercury: `def main()`  @ src/omni_mercury_engine/cli.py:59
  - decorators differ:  omni=['click.group()', "click.version_option(version='1.0.0')"]  mercury=['click.group()', "click.version_option(version='1.7.0')"]
### `security` — = IDENTICAL
- Omni:    `def security(payload: str)`  @ src/omni_anomaly_engine/cli.py:58
- Mercury: `def security(payload: str)`  @ src/omni_mercury_engine/cli.py:101
### `train` — = IDENTICAL
- Omni:    `def train(data: str, output: str, epochs: int)`  @ src/omni_anomaly_engine/cli.py:71
- Mercury: `def train(data: str, output: str, epochs: int)`  @ src/omni_mercury_engine/cli.py:114

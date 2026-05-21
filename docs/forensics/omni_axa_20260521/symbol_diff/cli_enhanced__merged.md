# Symbol diff: cli_enhanced__merged

- **Omni**:    `src/omni_anomaly_engine/cli_enhanced.py`  (size 22468 bytes)
- **Mercury**: `src/omni_mercury_engine/cli.py`  (size 39815 bytes)

## Public symbol counts
- Omni public symbols:    **10**
- Mercury public symbols: **14**
- Shared:                 **4**
- Only in Omni:           **6**   ← POTENTIAL_EXTRACTION_CANDIDATE
- Only in Mercury:        **10**

## ⚠ POTENTIAL_EXTRACTION_CANDIDATE (symbols in Omni only)
- `def run_chemistry(analysis_type: str, sample_file: str, output: Optional[str], report: bool)` @ src/omni_anomaly_engine/cli_enhanced.py:239
- `def run_demo(demo_type: str)` @ src/omni_anomaly_engine/cli_enhanced.py:288
- `def run_humanitarian(crisis_type: str, data_file: str, output: Optional[str], report: bool)` @ src/omni_anomaly_engine/cli_enhanced.py:158
- `def run_medical(subspecialty: str, ecg_file: Optional[str], vitals_file: Optional[str], biomarkers_file: Optional[str], patient_file: Optional[str], output: Optional[str], report: bool)` @ src/omni_anomaly_engine/cli_enhanced.py:59
- `def run_schumann(resonance_file: str, seismic_file: Optional[str], output: Optional[str], report: bool)` @ src/omni_anomaly_engine/cli_enhanced.py:179
- `def run_security(intel_type: str, threat_file: Optional[str], network_file: Optional[str], spectrum_file: Optional[str], output: Optional[str], report: bool)` @ src/omni_anomaly_engine/cli_enhanced.py:112

## Mercury-only (likely added enhancements)
- `def explain(input: str, model: str)` @ src/omni_mercury_engine/cli.py:132
- `def physics()` @ src/omni_mercury_engine/cli.py:182
- `def physics_dynamics(input: str, output: str, threshold: float, time_step: float, jerk_sensitivity: float, chaos_threshold: float)` @ src/omni_mercury_engine/cli.py:278
- `def physics_integrated(spectral_input: str, dynamics_input: str, uiux_input: str, output: str, threshold: float, fusion_weights: str)` @ src/omni_mercury_engine/cli.py:502
- `def physics_list()` @ src/omni_mercury_engine/cli.py:663
- `def physics_spectral(input: str, output: str, threshold: float, mode: str, sample_rate: float)` @ src/omni_mercury_engine/cli.py:199
- `def physics_uiux(input: str, output: str, threshold: float, rage_threshold: float, bot_threshold: float)` @ src/omni_mercury_engine/cli.py:361
- `def security(payload: str)` @ src/omni_mercury_engine/cli.py:101
- `def serve(host: str, port: int, workers: int, reload: bool, log_level: str)` @ src/omni_mercury_engine/cli.py:754
- `def voice(domain: str, model: str, offline: bool)` @ src/omni_mercury_engine/cli.py:811

## Shared symbols — AST signature diff
### `biometric` — = IDENTICAL
- Omni:    `def biometric(reference: str, test: str)`  @ src/omni_anomaly_engine/cli_enhanced.py:260
- Mercury: `def biometric(reference: str, test: str)`  @ src/omni_mercury_engine/cli.py:90
### `detect` — ≠ CHANGED
- Omni:    `def detect(input: str, detector: str, output: str, threshold: float, report: bool)`  @ src/omni_anomaly_engine/cli_enhanced.py:30
- Mercury: `def detect(input: str, detector: str, output: str, threshold: float)`  @ src/omni_mercury_engine/cli.py:69
  - decorators differ:  omni=['main.command()', "click.option('--input', '-i', required=True, help='Input data file (CSV/JSON)')", "click.option('--detector', '-d', default='fusion', help='Detector type')", "click.option('--output', '-o', help='Output file for results')", "click.option('--threshold', '-t', default=0.5, type=float, help='Anomaly threshold')", "click.option('--report', '-r', is_flag=True, help='Generate plain English report')"]  mercury=['main.command()', "click.option('--input', '-i', required=True, help='Input data file (CSV/JSON)')", "click.option('--detector', '-d', default='fusion', help='Detector type')", "click.option('--output', '-o', help='Output file for results')", "click.option('--threshold', '-t', default=0.5, type=float, help='Anomaly threshold')"]
### `main` — = IDENTICAL
- Omni:    `def main()`  @ src/omni_anomaly_engine/cli_enhanced.py:20
- Mercury: `def main()`  @ src/omni_mercury_engine/cli.py:59
  - decorators differ:  omni=['click.group()', "click.version_option(version='1.0.0')"]  mercury=['click.group()', "click.version_option(version='1.7.0')"]
### `train` — = IDENTICAL
- Omni:    `def train(data: str, output: str, epochs: int)`  @ src/omni_anomaly_engine/cli_enhanced.py:271
- Mercury: `def train(data: str, output: str, epochs: int)`  @ src/omni_mercury_engine/cli.py:114

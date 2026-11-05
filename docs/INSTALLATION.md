# Installation

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-07-24.

## Requirements

- Python >= 3.11 (3.12 recommended; 3.13 and 3.14 supported)
- pip >= 26.1 (the CVE-2026-6357 floor enforced across every install
  path in CI by `tests/security/test_cve_2026_6357_regression.py`)
- GCC >= 12 and CMake >= 4.4.4 for the AMA Cryptography native PQC
  build (see "Post-Quantum Cryptography backend" below). Mercury does
  not import without real AMA/PQC.

## Production-mode primitives (`MERCURY_ENV` + AMA/PQC)

Production deployments should set these environment variables before
importing `omni_mercury_engine`:

```bash
export MERCURY_ENV=production
export AMA_REQUIRE_REAL_PQC=true
# AMA_REQUIRE_CONSTANT_TIME is superseded by the pinned AMA v5.0.0
# (native-only constant-time operation is unconditional; AMA warns if it is
# set, and Mercury reads it only for diagnostics); leave it unset.
```

`AMA_REQUIRE_REAL_PQC` is retained for legacy workflow readability, but
AMA/PQC is mandatory regardless of its value:

- `MERCURY_ENV` (added in v1.7) is consumed by every collaborator that
  has a mock/stub fallback (`narrative.voice.MercuryVoice`, more to
  come). In `production`, the absence of a real adapter raises
  `MercuryProductionConfigError`; in `development` (the default) it
  warns and downgrades. An unknown value (e.g.
  `MERCURY_ENV=prod`) also raises — typos must be loud. See
  [`MIGRATION-1.6-to-1.7.md`](MIGRATION-1.6-to-1.7.md) §3.
- The import-time PQC check
  (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`) always
  requires `ama_cryptography.pqc_backends` plus ML-DSA-65,
  Kyber-1024, and SPHINCS+ native availability. Missing or
  partially-built AMA Cryptography raises `RuntimeError` at
  `import omni_mercury_engine` time before any other package state is
  materialised.

## Tier 0 minimal (hardened) configuration

The **Tier 0** posture is the smallest set of variables that makes Mercury refuse
to start or serve in a cryptographically- or safety-degraded state. All of it is
fail-closed: an omission or mismatch stops the process, it never silently
downgrades.

| Variable | Value | Purpose |
|---|---|---|
| `AMA_CRYPTO_VERSION` | `5.0.0` | Pinned PQC backend version; the import gate refuses a different *release* (`_pqc_gate._enforce_ama_version`). Matched PEP 440-tolerantly: `v4.0.0`, `4.0.0.post1`, `4.0` are accepted; `3.3.0` is not. |
| `LD_LIBRARY_PATH` | native AMA lib dir | Only for a manual out-of-tree AMA build; unneeded when `scripts/build_ama_native.sh` co-locates the `.so`. |
| `MERCURY_ENV` | `production` | Mock/stub collaborators raise instead of downgrading. |
| `AMA_REQUIRE_CONSTANT_TIME` | unset | **Superseded compatibility flag with the pinned AMA v5.0.0.** AMA enforces native-only operation unconditionally (INVARIANT-7 revised), so its constant-time C primitives are always in use; setting this variable changes no cryptographic behavior on a healthy install (AMA logs a deprecation warning), and on an install without the native backend it fails closed redundantly with Mercury's mandatory import-time PQC gate. Mercury still reads it for diagnostics — `security.pqc_backends.require_constant_time()`, surfaced by `get_pqc_capabilities()` / `validate_pqc_environment()`. Leave it unset. |
| `MERCURY_GATE_AUDIT_LOG` | durable path | Persistent JSONL sink for every harm-gate decision. |
| `MERCURY_GATE_AUDIT_SECURELOG` | `1` | Also forward to the hash-chained, tamper-evident audit. |
| `MERCURY_REQUIRE_REAL_HARM_CLASSIFIER` | `1` | Fail closed unless a real meaning-level classifier is serving. |
| `MERCURY_MODEL_ENDPOINT` | `http://127.0.0.1:11434` | Loopback served-model endpoint for the meaning-level classifier. |
| `MERCURY_<PROVIDER>_MODEL` | unset | Per-provider LLM model selection (`MERCURY_OLLAMA_MODEL`, `MERCURY_ANTHROPIC_MODEL`, `MERCURY_OPENAI_MODEL`, `MERCURY_GEMINI_MODEL`, …). Mercury ships **no default model for any provider**: explicit `LLMConfig.model_name` wins, this variable is the env fallback, and with neither the adapter reports itself unavailable (the chain serves the deterministic template). Every provider gets the identical convention; none is privileged. |
| `CI_MEANING_LEVEL_ENABLED` | `true` | *(CI repository variable, not a runtime env var.)* The `ci/meaning-level` adversarial FN-budget lane runs on **every** PR via a validated stdlib model double, so this variable does **not** switch the lane on or off. It is the documented marker for opting into a deeper *real-served-model* run, applied by swapping in the real-Ollama serving block (see the header comment in `.github/workflows/tier0-safety.yml`). |

```bash
# Tier 0 minimal production env
export MERCURY_ENV=production
export AMA_CRYPTO_VERSION=5.0.0
# AMA_REQUIRE_CONSTANT_TIME omitted: superseded by AMA v5.0.0 (constant-time is unconditional).
export MERCURY_GATE_AUDIT_LOG=/var/lib/mercury/audit/gate_decisions.jsonl
export MERCURY_GATE_AUDIT_SECURELOG=1
export MERCURY_REQUIRE_REAL_HARM_CLASSIFIER=1
export MERCURY_MODEL_ENDPOINT=http://127.0.0.1:11434
```

The four Tier 0 CI lanes (`ci/tier0-verify`, `ci/gate-unit`, `ci/meaning-level`,
`ci/rolling-corpus-eval`) live in `.github/workflows/tier0-safety.yml`. To make
them **required for PRs touching gate logic**, add those check names under the
repository's branch-protection rules. See the escalation/audit operations guide
in [`ESCALATION_AUDIT_RUNBOOK.md`](ESCALATION_AUDIT_RUNBOOK.md).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Steel-SecAdv-LLC/Mercury-Agent.git
cd Mercury-Agent

# Install core (anomaly detection, no PyTorch required)
pip install -e .

# REQUIRED: build the AMA-Cryptography native PQC backend. The package
# fail-closes at import without it (no dev/CI escape hatch), so `pip install`
# alone leaves it non-importable. See "Post-Quantum Cryptography backend" below.
bash scripts/build_ama_native.sh

# Or install with the full ML stack (still requires the AMA build above)
pip install -e ".[all]"
```

### Installing with uv (locked)

`uv.lock` is one cross-platform resolution of every extra (Linux, macOS and
Windows; Python 3.11 to 3.14). `uv sync --locked` installs exactly that set,
resolving and downloading in parallel, and uses uv's global hardlinked cache, so
a second install on the same machine is near-instant and the first is much
faster than pip. The Dockerfile builder and the CI test lanes install this way.

```bash
# core + the ML stack, the API and the developer tools, from the lockfile
uv sync --locked --extra ml --extra api --extra dev
source .venv/bin/activate

# everything the Docker image ships
uv sync --locked --extra all --extra dev

# the AMA backend is still built (or downloaded) by the same script; it works
# in a uv environment, which has no pip, through uv's pip front end
bash scripts/build_ama_native.sh
```

`--locked` refuses a lockfile that no longer matches `pyproject.toml`. After
changing a dependency, run `uv lock` and commit `uv.lock`; the Workflow
Hardening lane runs `uv lock --check` and names the drift on the pull request.

## Installation Profiles

> **OpenCV moved from `[ml]` into the core dependencies in v3.** The FINDΩYOU
> face path needs it for decoding, colour conversion, resize and the
> legacy-noise blur, and on the 4.x line it also supplies the Haar cascade that
> detects a face without PyTorch. While it was optional the path degraded
> silently: the age filter returned its input unchanged, and a missing decoder
> was reported as "Face detection failed". Image *decoding* does have in-house
> parity and keeps it (`vision/imaging` falls back to Pillow), so that path does
> not depend on the pin.
>
> **The core pin alone does not guarantee a face _detector_.** OpenCV 5.0 moved
> `CascadeClassifier` and the bundled cascades into `opencv_contrib`, so the
> 5.x `-headless` wheel (which the `<6` ceiling admits) carries neither — a
> core-only install resolving 5.x has no detector at all. That case raises
> `FaceDetectorUnavailableError` naming the installed state (whether
> `facenet_pytorch` imports, whether PyTorch is present, the OpenCV version,
> whether it has `CascadeClassifier`, which cascade XML was found) and only the
> remedies that apply, rather than returning `None` and reading as "no face in
> the image". The remedies that really supply a detector:
>
> * **MTCNN** needs the `facenet-pytorch` package, which no Mercury extra
>   declares, together with PyTorch from `[ml]`:
>   `pip install facenet-pytorch "mercury-agent-global[ml]"`. Installing `[ml]`
>   alone supplies PyTorch and leaves MTCNN unavailable.
> * **Haar cascade** needs `cv2.CascadeClassifier` *and* a cascade XML. On the
>   5.x `-headless` wheel the classifier itself is absent, so XMLs cannot help;
>   install an OpenCV that carries it (`pip install "opencv-python-headless<5"`,
>   or a 5.x `opencv-contrib-python-headless` wheel). When the classifier is
>   present but no XML is bundled, place the XMLs in `MERCURY_HAARCASCADE_DIR`
>   or in `${DEEPFACE_HOME:-$HOME}/.deepface/weights` as described under
>   [Biometric image analysis (DeepFace) assets](#biometric-image-analysis-deepface-assets)
>   below; the engine searches the configured directory, OpenCV's bundled
>   `cv2/data`, then the DeepFace weights directory.

| Profile | Command | Includes |
|---------|---------|----------|
| **Core** | `pip install -e .` | numpy, scipy, pandas, OpenCV (headless), MercuryAnomalyDetector, face detection and age progression |
| **ML** | `pip install -e ".[ml]"` | Core + PyTorch, torchvision, pytorch-lightning, timm, h5py, numcodecs, netCDF4, metpy, obspy |
| **Visual** | `pip install -e ".[visual]"` | ML + visual anomaly detectors |
| **VLM** | `pip install -e ".[vlm]"` | transformers, accelerate. Ships SHA-pinned local BLIP backends (`blip_vqa`, `blip_caption`) that run the AnyAnomaly/LAVAD detectors on CPU; larger LVLMs (Qwen2-VL, MiniCPM-V, LLaVA) are configurable with an operator-supplied revision pin. |
| **Foundation** | `pip install -e ".[foundation]"` | ML + foundation-model adapters: `chronos-forecasting` (Amazon Chronos, local inference), `nixtla` (TimeGPT API client), `stumpy` (matrix profile) |
| **Explainability** | `pip install -e ".[explainability]"` | `shap`. The default explainer (IntegratedGradients + faithfulness evaluator) is self-contained and needs no extra. `lime` is deliberately not declared — its only release cannot build on modern setuptools; the LIME adapter uses the library if manually installed and otherwise falls back to its in-repo linear surrogate. |
| **API** | `pip install -e ".[api]"` | FastAPI, httpx, uvicorn, python-multipart |
| **PQC** | `pip install -e ".[pqc]"` | AMA Cryptography (pinned to `v5.0.0`) |
| **Compliance** | `pip install -e ".[compliance]"` | NIST CSF live-fetcher dependency (`openpyxl`) |
| **All** | `pip install -e ".[all]"` | Every runtime feature extra (ml, visual, vlm, foundation, medical, face, api, sota, llm, drift, fairness, streaming, optimization, benchmark, domains, gui, explainability, compliance). `[pqc]` and `[dev]` install separately. |
| **Dev** | `pip install -e ".[dev]"` | Tooling only: pytest (+ asyncio/cov/timeout/mock/xdist), hypothesis, black, mypy, ruff, pre-commit. Combine with `[all]` for the full stack. |

The `[compliance]` extra installs `openpyxl`, used only by
`NISTCSFReferenceFetcher` when parsing the live NIST CSRC reference
XLSX. The OSHA and TLP 2.0 modules in `omni_mercury_engine.compliance`
have no extra dependencies beyond core.

PyTorch Lightning, torchmetrics, matplotlib, SymPy, SHAP, LIME and
DeepFace/TensorFlow are imported the first time the feature that needs them
runs, not when `omni_mercury_engine` is imported (a cold
`import omni_mercury_engine.engine` took about 3 s on the 4-CPU host it was
measured on, by `benchmarks/fusion_latency.py`). A missing library is therefore
noticed, or replaced by the in-repo fallback where one exists, when that
feature first runs.

### Biometric image analysis (DeepFace) assets

DeepFace is installed only on Python < 3.14 (TensorFlow has no 3.14 wheel), so
the container image, which runs 3.14, uses the harmonic fallback instead. On
other installs the first image analysis downloads, into
`$DEEPFACE_HOME/.deepface/weights` (default `~/.deepface/weights`):

| Asset | Size | Needed by |
|---|---|---|
| `age_model_weights.h5`, `gender_model_weights.h5`, `facial_expression_model_weights.h5` | 539 MB, 537 MB, 6 MB | `predict()` on an image |
| `facenet_weights.h5` | 92 MB | `extract_features()` and reference-vs-test matching |
| `haarcascade_frontalface_default.xml`, `haarcascade_eye.xml` | 930 KB, 341 KB | face detection when the installed OpenCV has `CascadeClassifier` but no bundled cascades (a 5.x `opencv-contrib` build); 4.x wheels ship them in `cv2/data`; `MERCURY_HAARCASCADE_DIR` names another directory |

A failed download does not raise: `predict()` returns the visual assessment with
an `error` key and `attributes_measured: false`. On a host without GitHub
access, copy a populated `weights` directory into place, and fetch the two
cascades (pinned to the OpenCV 4.9.0 tag DeepFace uses) with:

```bash
cd "${DEEPFACE_HOME:-$HOME}/.deepface/weights"
base=https://raw.githubusercontent.com/opencv/opencv/4.9.0/data/haarcascades
curl -fLO "$base/haarcascade_frontalface_default.xml" -O "$base/haarcascade_eye.xml"
sha256sum -c <<'SUMS'
0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0  haarcascade_frontalface_default.xml
71cc64fc305a355dc60067880f6fbbd43dd155bd63ee3844661a1bda34b2fd8c  haarcascade_eye.xml
SUMS
```

The unit tests stub DeepFace; the end-to-end test needs `[face]` installed and
`MERCURY_NETWORK_TESTS=1`, and no CI lane runs it yet.

## Core Dependencies

The core anomaly detection path (`MercuryAnomalyDetector`) requires only:

- `numpy >= 2.4.0` (required for Python 3.12/3.13 wheels and the strict-mypy type contract)
- `scipy >= 1.10.0`

scikit-learn is **not** required for core detection and is not part of any
profile above. It ships in the `[benchmark-comparison]` extra and is used for
cross-domain transfer, calibration baselines, and benchmark comparisons.

## Post-Quantum Cryptography backend

Mercury Agent uses **AMA Cryptography** as the sole supported PQC
backend (see `SECURITY.md` and PRs #144, #162). Package import is
guarded unconditionally: `security/pqc_backends.py` imports the real
`ama_cryptography.pqc_backends` surface, and the import-time gate
(`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`,
invoked from `__init__.py`) requires ML-DSA-65, Kyber-1024, and
SPHINCS+ native availability. Missing or partially-built AMA
Cryptography raises `RuntimeError` before any other package state is
materialised.

For local and production installs, build and install the native library with the
canonical helper Mercury ships, **`scripts/build_ama_native.sh`** (the `cmake`
step operates on the AMA-Cryptography checkout the script clones, **not** on the
Mercury-Agent repo, which has no `CMakeLists.txt` of its own). It clones the
pinned `AMA_REF` (`v5.0.0`, matching pyproject's `ama-cryptography` git pin and
`.github/actions/build-ama-cryptography`), builds the native PQC library,
installs the Python package, **co-locates the shared object inside the installed
`ama_cryptography` package** so it loads with no `LD_LIBRARY_PATH`, and fails
loudly unless ML-DSA-65 + Kyber-1024 + SPHINCS+ all load:

```bash
# Downloads the prebuilt backend for this OS and CPU when the release carries
# one; otherwise requires git and gcc/g++ >= 12 and installs cmake >= 4.4.4 itself.
bash scripts/build_ama_native.sh
# AMA_REQUIRE_CONSTANT_TIME is superseded by AMA v5.0.0 (native-only
# constant-time is unconditional; AMA warns if it is set); no need to export it.
```

**Prebuilt first.** The cmake build is the slow phase of every install, so the
script first runs `scripts/fetch_ama_prebuilt.sh`, which downloads
`ama-cryptography-<ref>-<os>-<arch>.tar.gz` (os: `linux`, `macos`, `windows`;
arch: `x86_64`, `aarch64`) from the `ama-native-<ref>` release of this
repository, checks it against the release's `SHA256SUMS`, and unpacks the
`ama_cryptography` wheels for Python 3.11 to 3.14. AMA bundles the native
library inside each wheel, signs the library's SHA3-256 into it at build time
and verifies that at import, and tags every wheel for the interpreter that built
it; the script installs the wheel matching the active interpreter (`pip install
--no-index --find-links`) and runs the usual capability probe. Outcomes:

| Result | Script behaviour |
|---|---|
| archive verified, a wheel matches this Python, its backend loads | installed; no clone, no compiler, no cmake |
| no archive for this platform, no network, `AMA_PREBUILT=0` | the source build below runs unchanged |
| archive fails `SHA256SUMS` | the script stops; a corrupt or tampered release is never built around silently |
| archive verified but no wheel for this Python, or the library does not load here | the source build runs, with a notice |

`AMA_PREBUILT_BASE_URL` points the download elsewhere (a `file://` mirror, or
the AMA repository's own release once it carries the same assets). The archives
are produced by `.github/workflows/ama-prebuilt.yml` (manual dispatch with the
AMA tag): one job per OS and CPU builds a wheel with each of the four Pythons
(the first pays for the cmake build, the others reuse it), proves every wheel in
a fresh environment of its Python, and the release is published with its
`SHA256SUMS`. A platform that fails to build is simply absent from the release
and its installs build from source.

Override the ref/repo/scratch dir via `AMA_REF`, `AMA_REPO`, `AMA_BUILD_DIR`.
The source build performs, in order: install the PEP 517 build floors
(`setuptools>=84.0.0`, `wheel>=0.48.0`, `cmake>=4.4.4` — AMA v5.0.0 raised the
setuptools and cmake floors; older toolchains fail its preflight); `git clone
--branch v5.0.0`; `cmake -DAMA_USE_NATIVE_PQC=ON -DAMA_BUILD_SHARED=ON` + build;
`AMA_NO_CYTHON=1 pip install --no-build-isolation --force-reinstall --no-deps .`
(through `uv pip` in a uv environment, which has no pip);
co-locate `libama_cryptography.so*`; verify via `get_pqc_backend_info()`.

**Manual out-of-tree build (only if you do not co-locate the `.so`).** If you
build AMA yourself and leave the shared object in the build tree, export its
directory on `LD_LIBRARY_PATH` before importing Mercury:

```bash
git clone --depth 1 --branch v5.0.0 \
    https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
cd /tmp/ama-cryptography
pip install --upgrade "setuptools>=84.0.0" "wheel>=0.48.0" "cmake>=4.4.4"
cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
AMA_NO_CYTHON=1 pip install --no-build-isolation .
export LD_LIBRARY_PATH=/tmp/ama-cryptography/build/lib:/tmp/ama-cryptography/build:${LD_LIBRARY_PATH:-}
export AMA_CRYPTO_VERSION=5.0.0
```

The import-time gate additionally enforces the pinned **version**: it refuses
unless the installed `ama_cryptography.__version__` resolves to release `5.0.0`
and, when set, `AMA_CRYPTO_VERSION` agrees. Both are matched PEP 440-tolerantly
(`v5.0.0`, `5.0.0.post1`, `5.0.0+cpu`, and even `5.0` / `5` are accepted — a
shorter string is zero-padded to the pinned width, so `4` becomes `(4, 0, 0)`
and matches, whereas under the old `3.3.0` pin `3` padded to `(3, 0, 0)` and was
refused; a different release such as `3.3.0` is refused), so a valid post/local
build is not misdiagnosed as a failure — see
`omni_mercury_engine/_pqc_gate.py::_enforce_ama_version` / `_release_matches`.

**Docker / Kubernetes:** the production image builds this automatically — the
Dockerfile builder stage runs `scripts/build_ama_native.sh`, the `.so` travels
into the runtime image with the virtualenv, and a build-time
`import omni_mercury_engine.engine` smoke test fails the build if the backend
is missing. No extra steps are required to deploy a PQC-complete image.

The gate (`omni_mercury_engine._pqc_gate._enforce_pqc_production_gate`) is
scoped to the agent surfaces: it runs at import of
`omni_mercury_engine.engine` (which the CLI and API serve),
`omni_mercury_engine.crypto`, and every AMA-backed `security` export, and
refuses to proceed if the native library is unloadable — those imports raise
`RuntimeError` before any agent state is materialised. There is no stub path
and no fallback chain to a non-AMA backend. Only the standalone analytic
leaves (dataset loaders, individual detectors, metrics — the quickstart path,
which cannot decide, serve, sign, or encrypt) import without AMA; the split
is pinned by `tests/test_pqc_startup_gate.py::TestScopedGateContract`.

### Build options that are deliberately not enabled

Two AMA build knobs come up when looking for throughput, and both are settled
rather than open. Recorded here because the alternative is each operator
re-deriving them blind.

**Cython bindings are skipped on purpose, and it is not a performance loss.**
`scripts/build_ama_native.sh` exports `AMA_NO_CYTHON=1`. The Python API loads
the shared library through `ctypes` at runtime, so the optional Cython
extension is not on the call path at all -- building it would add a build
dependency and change nothing Mercury executes.

**The AVX-512 SHA3 4-way kernel (`-DAMA_ENABLE_AVX512=ON`) is available, and
its benefit is not established.** Measured 2026-11-05 on a 4-vCPU Xeon with
`avx512f/bw/cd/dq/vl`, comparing two builds that differ only in that flag,
using AMA's own `benchmark_c_raw` (thousands of iterations, medians):

* Across **four** passes nothing survives as a reproducible speedup above 20%,
  and nothing as a reproducible regression above 10%.
* Two passes showed SHA3-256/512 at roughly **-53%** and ML-KEM-1024 at
  **-53%**; two later, quieter passes showed SHA3-256 at -18.8% then +0.9%.
* One pass reported AES-256-GCM at **-45%**, which an AVX-512 *SHA3* kernel
  cannot cause -- so that pass was measuring host noise, not the kernel.
* The only family consistent across both of the quietest passes is
  **ML-KEM-1024 (-12%)** and **ML-DSA-65 Verify (-11%)**.

Through Mercury's own integration path the effect is smaller still: repeated
runs of `tools/pqc_capability_probe.py` disagree in sign between median and
minimum on the primitives that matter, because those round trips are dominated
by FFI and single-shot keygen rather than bulk SHA3.

So the flag is **not** added to the build helper, for two reasons. The evidence
does not meet this repository's bar for adopting an optimisation, and a library
built with it requires AVX-512 at runtime -- which would make a helper intended
to work on any host produce a non-portable artifact. An operator on dedicated
hardware who wants it should pass the flag and measure on their own host; a
shared vCPU cannot answer the question.

## Verify Installation

```bash
python -c "
from omni_mercury_engine.detectors.statistical import MercuryAnomalyDetector
import numpy as np
d = MercuryAnomalyDetector()
d.fit(np.random.randn(100, 5))
r = d.detect(np.random.randn(20, 5))
print(f'Scores shape: {r[\"scores\"].shape}')
print('Installation OK')
"
```

## Running Tests

```bash
# Core ensemble tests (fast, no extra deps)
pytest tests/validation/test_ensemble_replacement.py -v

# Full test suite (requires dev dependencies)
pip install -e ".[dev]"
pytest tests/ -x -k "not mimic and not swat and not wadi"
```

## Running Benchmarks

```bash
# Mercury benchmark on real datasets (requires ML deps for data loading)
pip install -e ".[ml]"
python benchmarks/mercury_benchmark.py
```

Results are written to `benchmarks/mercury_benchmark_results.json`.

## v1.7 cycle additions

- **Medical integrations.** The `EndocrinologyDetector` and
  `AnesthesiologyPredictor` are integration-ready (no vendor
  credentials in tree). See [`medical/SETUP.md`](medical/SETUP.md) for
  Dexcom v3 / FHIR R4 wiring and the `CGMDataSource` /
  `VitalsDataSource` adapter contract.
- **Drone telemetry.** The `DroneAnomalyDetector` is transport-agnostic;
  populate `DroneState` instances from your ingest layer (PX4 ULog via
  `pyulog`, MAVLink via `pymavlink`, or vendor SDK). See
  [`drone/SETUP.md`](drone/SETUP.md).
- **Compliance modules.** NIST CSF 2.0, FIRST.org TLP 2.0, and
  OSHA / eCFR live under `omni_mercury_engine.compliance`. See
  [`COMPLIANCE.md`](COMPLIANCE.md).
- **Performance profiling.** Six entry points (`@profile_func`,
  `@profile_memory`, `@profile_time`, `@profile_time_async`,
  `@profile_complete`, `PerformanceBenchmark`) plus `benchmark_function`
  ship in `omni_mercury_engine.utils.profiling` and are gated by
  `set_profiling_enabled(True)` at runtime. See
  [`PROFILING.md`](PROFILING.md).

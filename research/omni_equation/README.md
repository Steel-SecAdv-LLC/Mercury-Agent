# research/omni_equation

Measurement-first audit of the "omni-equation" direction for Mercury Agent —
run against the **real** detector, **real** streams, **real** equation, and
**real** live-API data. Built to decide, not to decorate.

## Files
- `FINDINGS.md` — the measured results and the reframe they force.
- `PROMPT.md` — the corrected, measurement-grounded build prompt (gated, with
  kill criteria). This is the direction to build to.
- `harness_multidomain.py` — per-stream harness on real live-domain data
  (earthquake/hurricane/marine/tornado/tsunami …). The evidence behind FINDINGS.
- `harness_tabular.py` — per-stream harness for tabular ADRepository/ODDS
  datasets. NOTE: those sources are blocked by the sandbox SSRF allowlist and
  fall back to a trivially-separable synthetic (AUROC 1.0) — only meaningful in
  an environment where ADRepository is reachable.
- `real_results.json` — raw per-event metrics from the multi-domain run.

## Reproduce
The real engine requires the AMA-Cryptography PQC backend (hard import gate, no
env bypass). Build it once (as CI does), then point the env at it:

```bash
# AMA backend (one-time)
git clone --depth 1 --branch v3.3.0 \
  https://github.com/Steel-SecAdv-LLC/AMA-Cryptography.git /tmp/ama-cryptography
cd /tmp/ama-cryptography && cmake -B build -DAMA_USE_NATIVE_PQC=ON && cmake --build build
AMA_NO_CYTHON=1 pip install --no-build-isolation .

# run the real measurement
cd Mercury-Agent
export PYTHONPATH=.:/tmp/ama-cryptography
export AMA_CRYPTO_LIB_PATH=/tmp/ama-cryptography/build/lib/libama_cryptography.so
export LD_LIBRARY_PATH=/tmp/ama-cryptography/build/lib
export MERCURY_ALLOW_SYNTHETIC=0     # real data only; unreachable domains skip
python research/omni_equation/harness_multidomain.py
```

Deps beyond the engine: `numpy scipy` (already in the engine's runtime stack).
Metrics come from Mercury's own `omni_mercury_engine.ml.mercury_ml`, **not**
scikit-learn — `src/` and `research/` are guarded against any `sklearn` import
(`tests/test_no_sklearn_in_src.py`). Data is fetched live from allowlisted
USGS/NOAA endpoints.

## Headline
Ensemble AUC **0.836** vs best-single **0.909** (fusion *dilutes*); streams
|corr| **0.66**; equation inert (**−0.002**); η^Φ gate inert (**0.003** flip);
earthquake AUC **0.94** / F1 **0.09** (calibration is the bottleneck). See
`FINDINGS.md`.

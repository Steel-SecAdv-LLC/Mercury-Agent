# Governed Fusion Substrate (PR #278) — measurement environment.
#
# Source this before running research/governed_fusion/measure_*.py so every
# figure in FINDINGS.md reproduces against the REAL AMA/PQC native backend
# (hard import gate) on the REAL reachable suite (MERCURY_ALLOW_SYNTHETIC=0):
#
#     source research/governed_fusion/gf_env.sh
#     python research/governed_fusion/measure_baseline.py
#     python research/governed_fusion/measure_conformal.py
#     python research/governed_fusion/measure_reliability_fusion.py
#     python research/governed_fusion/measure_survivability.py
#     python research/governed_fusion/build_manifest.py
#
# The AMA native backend is built by .github/actions/build-ama-cryptography
# (ama-ref v5.0.0) into ${AMA_HOME}/build/lib/libama_cryptography.so.  Override
# AMA_HOME / GF_CACHE_DIR / GF_RESULTS_DIR if your layout differs.

# Repo root = two levels up from this script (research/governed_fusion/).
_gf_src="${BASH_SOURCE[0]:-$0}"
GF_REPO_ROOT="$(cd "$(dirname "${_gf_src}")/../.." && pwd)"

# AMA cryptography native backend location.
AMA_HOME="${AMA_HOME:-/tmp/ama-cryptography}"

# NB: ${AMA_HOME} itself must NOT go on PYTHONPATH. The composite pip-installs
# the ama_cryptography package, and that installed copy is the sanctioned
# import: its integrity artifact was generated against the .so the wheel was
# linked to, so the AMA_CRYPTO_LIB_PATH override below verifies and loads.
# Importing from the source checkout instead fails closed under AMA >= 5.0.0:
# the checkout carries the release's *signed* INTEGRITY_NATIVE_DIGEST_HEX,
# which a locally built libama_cryptography.so can never match, so the FIPS
# 140-3 POST refuses to initialise (observed on the Leakage-split gate lane
# after the v4.0.0 -> v5.0.0 pin bump).
export PYTHONPATH="${GF_REPO_ROOT}:${GF_REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export AMA_CRYPTO_LIB_PATH="${AMA_HOME}/build/lib/libama_cryptography.so"
export LD_LIBRARY_PATH="${AMA_HOME}/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Real-data discipline.  NB: the loaders/ package does NOT honour this flag (only
# the datasets/ package does), so suite.py makes the live/reconstructed split
# explicit in code instead of relying on it.  Kept for the datasets/ paths.
export MERCURY_ALLOW_SYNTHETIC=0

# The reconstructed-from-live loaders (tsunami BPR, energy Kp, ebola curve) now
# derive their RNG seed from hashlib.sha256 (process-stable), so the reconstructed
# group reproduces byte-identically across processes without this pin.  It is kept
# as defense-in-depth and because the datasets/ package consults it; the 23-event
# live headline suite is hash-independent regardless.
export PYTHONHASHSEED=0

# Heavy per-event (X,y) + score .npz cache (regenerated from the live loaders
# when absent; NOT committed — fingerprinted instead by manifest.json).
export GF_CACHE_DIR="${GF_CACHE_DIR:-/home/user/gf_cache}"

# Committed, auditable per-event results JSON + data manifest.
export GF_RESULTS_DIR="${GF_RESULTS_DIR:-${GF_REPO_ROOT}/research/governed_fusion/results}"

#!/usr/bin/env bash
# Regenerate the hash-locked CI requirement files in requirements/ci/.
#
# Every install a CI lane, the Docker image or a setup script runs is pinned
# and hash-verified (OpenSSF Scorecard Pinned-Dependencies):
#
#   * the project's own dependencies and the CI tool groups come from uv.lock,
#     exported at run time (`uv export --frozen ... | pip install
#     --require-hashes`), so Dependabot's uv updates reach CI unaided;
#   * what cannot come from uv.lock -- pip and uv themselves, tools that run in
#     their own environment, the deliberate numpy-floor install -- is compiled
#     here, universally for every supported Python, with hashes.
#
# Run after editing any requirements/ci/*.in. tests/scripts/
# test_ci_requirements_pinned.py fails if a .txt is out of step with its .in.
set -euo pipefail
cd "$(dirname "$0")/.."
# With arguments, relock only those specs (``requirements/ci/torch-cpu.in``);
# without, relock every spec.
specs=("$@")
[[ ${#specs[@]} -gt 0 ]] || specs=(requirements/ci/*.in)
for spec in "${specs[@]}"; do
  out="${spec%.in}.txt"
  # A spec may carry extra compile flags on a ``# uv-compile-args:`` line (the
  # CPU torch index, a platform). Everything else is resolved for every
  # platform and every supported Python. --no-config: these files serve
  # environments the project's own [tool.uv] ``environments`` does not cover
  # (the AMA prebuilt matrix builds on Intel macOS and Windows), so the
  # project's settings must not narrow them.
  read -r -a extra <<< "$(sed -n 's/^# uv-compile-args: //p' "$spec")"
  # Compile from scratch: uv reuses an existing output's pins *and hashes* as
  # preferences, which keeps hashes for the wrong interpreter after a spec's
  # compile arguments change.
  rm -f "$out"
  # A spec that names a platform is compiled for exactly one interpreter: uv
  # then emits hashes only for the wheels that interpreter can install, so it
  # must also name the consuming lane's --python-version.
  platform_args=(--universal)
  version_args=(--python-version 3.11)
  for arg in "${extra[@]}"; do
    [[ "$arg" == "--python-platform" ]] && platform_args=()
    [[ "$arg" == "--python-version" ]] && version_args=()
  done
  uv pip compile "$spec" \
    --no-config \
    "${platform_args[@]}" \
    "${version_args[@]}" \
    --generate-hashes \
    --no-annotate \
    --quiet \
    --custom-compile-command "bash scripts/lock_ci_requirements.sh" \
    "${extra[@]}" \
    -o "$out"
  echo "locked ${out}"
done

# Hardware Test Harness

This document describes how to obtain reproducible performance numbers
for the Mercury Agent Lyapunov validation pipeline using
`scripts/run_hardware_benchmark.py`.  The goal is *scientific*
reproducibility: a number quoted in the documentation or in a release
note must be paired with the environment that produced it, and any
reviewer must be able to re-derive that number with a single command.

## Quick start

```bash
# Default: canonical config, 2000 iterations, 200 warmup, JSON at artifacts/hwbench.json.
python scripts/run_hardware_benchmark.py

# Custom configuration / output / iteration budget:
python scripts/run_hardware_benchmark.py \
    --config configs/lyapunov_canonical.yaml \
    --iters 5000 --warmup 500 \
    --out artifacts/hwbench-$(date -u +%Y%m%dT%H%M%SZ).json

# Treat a measured throughput regression as a CI failure:
python scripts/run_hardware_benchmark.py \
    --min-ops-per-sec 1500
```

The harness exits non-zero if:

| Code | Meaning                                                                |
|------|------------------------------------------------------------------------|
| 2    | Configuration file missing or invalid.                                 |
| 3    | The Lyapunov claim itself failed validation. Performance is undefined. |
| 4    | Measured throughput fell below `--min-ops-per-sec`.                    |

## Output schema

The JSON report contains four sections — all required:

```jsonc
{
  "config": "configs/lyapunov_canonical.yaml",
  "environment": {
    "python": "3.12.3",
    "numpy": "2.x.y",
    "platform": "Linux-...-x86_64-with-glibc2.39",
    "machine": "x86_64",
    "processor": "x86_64",
    "cpu_count": 4,
    "cpu_affinity": [0, 1, 2, 3]
  },
  "validation": {
    "ok": true,
    "claimed_lambda": 0.25,
    "computed_lambda": 0.5,
    "max_generalized_eig": -0.5,
    "mode": "quadratic",
    "tol": 1e-08
  },
  "timing": {
    "iters": 2000,
    "warmup": 200,
    "samples": 1800,
    "mean_s": 1.2e-4,
    "stdev_s": 1.3e-5,
    "p50_s": 1.1e-4,
    "p95_s": 1.5e-4,
    "p99_s": 1.7e-4,
    "max_s": 1.8e-4,
    "ops_per_sec": 8056
  }
}
```

Two reports are comparable **only if** their `environment` blocks
match on `python`, `numpy`, `platform`, and `cpu_count`.  Anything
else is an apples-to-oranges comparison and the harness's caller is
responsible for refusing it.

## Reducing variance on shared / CI runners

GitHub-hosted runners are noisy because the host is shared.  The
following practices materially reduce measurement variance and should
be applied whenever a number is going to be quoted publicly:

1. **Pin the CPU set.** On Linux:
   ```bash
   taskset -c 0,1 python scripts/run_hardware_benchmark.py
   ```
   This excludes other cores from the measured workload and is
   visible in the report under `environment.cpu_affinity`.

2. **Raise process priority** when permitted (`sudo nice -n -10 ...`
   or `chrt -f 50 ...`).  Do **not** do this on shared infrastructure
   without authorization.

3. **Disable Turbo / dynamic frequency scaling** on bare metal:
   ```bash
   sudo cpupower frequency-set --governor performance
   ```

4. **Run with a larger iteration budget**: at least `--iters 5000
   --warmup 500` for any number that will be quoted.  The default
   2000 / 200 is suitable for CI smoke runs, not for publication.

5. **Repeat at least three times** and report the median of the
   `mean_s` figures; standard deviation across runs must be reported
   alongside.

## Integrating into CI as a regression gate

The recommended pattern is to commit a baseline JSON report and
compare against it on every PR.  The minimum viable gate is:

```bash
python scripts/run_hardware_benchmark.py \
    --iters 1000 --warmup 100 \
    --min-ops-per-sec "$(jq '.timing.ops_per_sec * 0.5' baselines/hwbench.json)"
```

The `* 0.5` factor accommodates the noise floor of GitHub-hosted
runners; tighten it (e.g. `* 0.8`) when the harness is moved to
dedicated benchmarking hardware.

## Dedicated hardware

For numbers cited in release notes, run the harness on a dedicated
host that:

* is not shared with any other workload during measurement;
* has SMT/Hyper-Threading either disabled or fully reserved;
* has its scaling governor set to `performance`;
* has the harness Python and NumPy versions pinned exactly (record
  them under `environment.python` / `environment.numpy`).

Store the resulting JSON report in `baselines/hwbench-<host>.json`
together with the exact command used to produce it.  Future
regression diffs can then use `jq` to compare specific keys without
re-running.

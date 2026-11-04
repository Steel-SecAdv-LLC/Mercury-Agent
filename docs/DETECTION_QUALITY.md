# Detection quality of the served routes under adversarial conditions

Measured, not asserted: every number here is produced by
`python research/adversarial/served_routes.py` from seeded scenarios built to
defeat a naive detector, and `tests/research/test_served_routes_ledger.py`
fails when the committed ledger (`research/adversarial/served_routes_results.json`)
no longer matches what the code produces. Run the script to refresh both.

Series length 300, seeds [0, 1]; kernel blob `411ff9dfab39`.

## Shape: AUROC / recall at the route's flags / false-positive rate

| Scenario | univariate_zscore | univariate_iqr | multivariate | fusion_rows | tier_average | fusion series verdict |
|---|---|---|---|---|---|---|
| `point_spike` | 1.000 / 1.00 / 0.002 | 1.000 / 1.00 / 0.000 | 1.000 / 1.00 / 0.002 | 1.000 / 1.00 / 0.018 | 0.997 / 1.00 / 0.047 | 2 of 2 flagged |
| `slow_ramp` | 0.979 / 0.40 / 0.000 | 0.973 / 0.17 / 0.000 | 0.979 / 0.40 / 0.000 | 0.999 / 1.00 / 0.089 | 0.773 / 0.12 / 0.043 | 2 of 2 flagged |
| `level_shift` | 0.968 / 0.00 / 0.000 | 0.723 / 0.00 / 0.000 | 0.968 / 0.00 / 0.000 | 1.000 / 1.00 / 0.013 | 0.619 / 0.09 / 0.033 | 2 of 2 flagged |
| `seasonal_masking` | 1.000 / 1.00 / 0.002 | 1.000 / 1.00 / 0.000 | 1.000 / 1.00 / 0.002 | 1.000 / 1.00 / 0.005 | 1.000 / 1.00 / 0.047 | 2 of 2 flagged |
| `variance_burst` | 1.000 / 1.00 / 0.027 | 1.000 / 1.00 / 0.017 | 1.000 / 1.00 / 0.027 | 1.000 / 1.00 / 0.067 | 1.000 / 1.00 / 0.047 | 2 of 2 flagged |
| `bounded_evasion` | 0.994 / 0.40 / 0.000 | 0.994 / 0.10 / 0.000 | 0.994 / 0.40 / 0.000 | 0.999 / 1.00 / 0.051 | 0.944 / 0.30 / 0.046 | 2 of 2 flagged |
| `distribution_shift` | 0.998 / 1.00 / 0.008 | 0.998 / 1.00 / 0.007 | 0.998 / 1.00 / 0.008 | 0.997 / 1.00 / 0.042 | 0.996 / 1.00 / 0.044 | 2 of 2 flagged |

Scenarios: a lone 8-sigma spike; a linear ramp through the last fifth (labels on
its second half); a 4-sigma level shift over the last third; a spike of one
seasonal amplitude on a crest of a strong season; a point anomaly inside a stretch
of tripled variance; an 8-sigma spike spread by an attacker over five rows;
Gaussian first half, heavy-tailed (t3) second half with two planted spikes.

## Width: false positives on pure noise

| Width | multivariate per-row FPR | fusion series flagged |
|---|---|---|
| 1 | 0.0000 | 0 of 2 |
| 8 | 0.0000 | 0 of 2 |
| 64 | 0.0000 | 0 of 2 |

The nominal per-row rate at the default sensitivity is 0.00047 (two-sided z 3.5).

## Rate independence: the detector tier's flags against `contamination`

True anomaly fraction 0.05. The flag rate's slope against the
parameter is **1.00** (1.0: the flag count is
the parameter, by construction; the tier thresholds at the contamination quantile).
The conformal flags (`conformal_alpha` = 0.05) do not move with the parameter, but
without labels they are calibrated on the scored series itself, so they flag about
`conformal_alpha` of it by construction: their rate is set by alpha, not by evidence.
Evidence needs a reference independent of the batch (a fit on known-normal data).

| contamination | flag rate | recall | FPR | conformal flag rate | conformal FPR |
|---|---|---|---|---|---|
| 0.01 | 0.010 | 0.200 | 0.000 | 0.047 | 0.007 |
| 0.05 | 0.050 | 0.867 | 0.007 | 0.047 | 0.007 |
| 0.1 | 0.100 | 1.000 | 0.053 | 0.047 | 0.007 |
| 0.2 | 0.200 | 1.000 | 0.158 | 0.047 | 0.007 |

## Attribution: what each tier member adds over the rest

Ensemble (average of all members) AUROC over the shape scenarios: **0.902**. A negative marginal gain means the
ensemble scores higher without the member on these shapes.

| Member | own AUROC | ensemble without it | marginal gain |
|---|---|---|---|
| `energy_based` | 0.986 | 0.878 | +0.024 |
| `deep_svdd` | 0.998 | 0.879 | +0.024 |
| `spiking` | 0.819 | 0.887 | +0.015 |
| `deeplog_sequence` | 0.653 | 0.902 | +0.000 |
| `rca` | 0.500 | 0.902 | -0.000 |
| `frequent_pattern` | 0.500 | 0.902 | -0.000 |
| `spot_evt` | 0.789 | 0.902 | -0.000 |
| `spectral_residual` | 0.848 | 0.903 | -0.001 |
| `particle_filter` | 0.816 | 0.904 | -0.002 |
| `survival` | 0.764 | 0.906 | -0.004 |
| `digital_twin` | 0.869 | 0.907 | -0.005 |
| `hawkes` | 0.833 | 0.907 | -0.005 |
| `echo_state` | 0.861 | 0.907 | -0.005 |
| `imm` | 0.808 | 0.907 | -0.005 |
| `gaussian_process` | 0.827 | 0.910 | -0.008 |
| `bocpd` | 0.708 | 0.915 | -0.013 |

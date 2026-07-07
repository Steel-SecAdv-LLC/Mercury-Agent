# Detector tier -- real NAB before/after analysis

- **Data source:** omni_mercury_engine.datasets.timeseries.NABLoader
- **Series measured:** 30  |  **seed:** 0  |  **max_len:** 6000

## Ensemble vs best single detector (mean ROC-AUC over all series)

| pipeline | combiner | calibration | best single detector | best-single AUC | ensemble AUC | ensemble - best | ensemble F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| before (pre-PR) | `average` | `none` | echo_state | 0.6230 | 0.6240 | 0.0009 | 0.2882 |
| after (this PR) | `consensus` | `rank` | echo_state | 0.6230 | 0.6347 | 0.0117 | 0.2895 |

**Calibrated consensus ensemble beats the best single detector by 0.0117 ROC-AUC** (acceptance threshold > 0.003): PASS.

The `before` row is the pre-PR pipeline: raw-score averaging with no per-detector calibration (`OMNI_ENSEMBLE_CALIBRATION=none`, combiner `average`). The `after` row is this PR's default pipeline: per-detector empirical-CDF calibration (`rank`) plus the robust high-quantile `consensus` combiner. Both rows are the *same detectors on the same real NAB series*; only the ensemble's calibration + combination differ. A plain mean of anomaly scores is known to be dominated by robust rank aggregation for outlier ensembles (Aggarwal & Sathe, 2017); the consensus combiner is not dragged toward 0.5 by the uninformative members a mean averages in.
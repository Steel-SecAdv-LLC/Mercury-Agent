# Cross-Domain Frequency Correlation Analysis

## Schumann Resonance / Seismic Precursor Validation Path

### Overview

Mercury Agent's SpectralDomainOracle defines frequency bands across 7
domains. The environmental domain includes Schumann resonance bands
(fundamental at 7.83 Hz and harmonics), while the space domain includes
`schumann_coupling` (0.1--8.0 Hz). When USGS earthquake data and NOAA
space weather data simultaneously show anomalies in overlapping frequency
ranges, the `CrossDomainFrequencyCorrelator` flags the coincidence for
human review.

### Methodology

1. **Data Sources**:
   - USGS FDSNWS API (seismic event data) via `USGSEarthquakeLoader`
   - NOAA SWPC JSON (solar weather data) via `SolarDynamicsLoader`

2. **Spectral Analysis**:
   - Each data stream is processed by a domain-specific Oracle instance
   - Oracle produces a `FrequencyInfluenceVector` with per-band anomaly
     scores, spectral flux, phase coherence, and cepstral peak ratio

3. **Cross-Domain Correlation**:
   - `CrossDomainFrequencyCorrelator` identifies overlapping frequency
     bands between domain pairs
   - Minimum overlap width: 0.5 Hz (configurable)
   - Minimum band anomaly score: 0.3 (configurable)
   - Aggregate correlation via bandwidth-weighted geometric mean of
     per-band pair scores

4. **Alert Levels**:
   - HIGH: correlation score >= 0.8
   - MEDIUM: correlation score >= 0.6
   - LOW: correlation score >= 0.3
   - NONE: below threshold

### Relevant Frequency Bands

| Domain | Band Label | Range (Hz) | Weight |
|--------|-----------|------------|--------|
| environmental | schumann_fundamental | 7.83--8.5 | 0.20 |
| environmental | schumann_harmonic_1 | 8.5--14.3 | 0.15 |
| environmental | schumann_harmonics_upper | 14.3--33.8 | 0.10 |
| space | schumann_coupling | 0.1--8.0 | 0.15 |
| space | ionospheric | 0.01--0.1 | 0.15 |
| infrastructure | seismic_structural | 0.5--5.0 | 0.12 |

### Results

**Status: Pending empirical validation.**

The `CrossDomainFrequencyCorrelator` module is implemented and ready for
data. The following steps are required to produce empirical results:

1. Run full benchmark with `--live-only` flag to collect Oracle outputs
   from both USGS and NOAA loaders simultaneously
2. Feed concurrent Oracle results into the correlator
3. Document band-by-band correlation scores
4. Compute statistical significance (Fisher's method on per-band p-values)

### Preliminary Observations

No empirical cross-domain correlation results are available yet. This
section will be updated once live data from both USGS seismic and NOAA
space weather streams have been processed through concurrent Oracle
instances.

**Null results are scientifically valuable.** If no significant
correlation is found between Schumann resonance anomalies and seismic
precursors, that negative finding will be documented with the same rigour
as a positive one.

### Limitations

1. **Correlation != Causation**: The correlator detects spectral overlap
   coincidences. It does not and cannot establish causal relationships.

2. **Temporal Resolution**: The current implementation compares
   simultaneous snapshots. Lagged correlation (e.g., solar activity
   preceding seismic events by hours/days) is not yet implemented.

3. **Selection Bias**: The Oracle's band definitions were designed for
   domain-specific anomaly detection, not cross-domain correlation. Band
   boundaries may not optimally capture cross-domain phenomena.

4. **Sample Size**: Initial results will be based on limited concurrent
   observations. Statistical power increases with observation duration.

### Scientific Context

The hypothesis that electromagnetic precursors (including Schumann
resonance perturbations) may correlate with seismic activity has been
explored in the geophysics literature:

- Hayakawa, M. & Molchanov, O.A. (2002). *Seismo-electromagnetics:
  Lithosphere-Atmosphere-Ionosphere Coupling*. TERRAPUB.
- Balser, M. & Wagner, C.A. (1960). Observations of Earth-ionosphere
  cavity resonances. *Nature*, 188(4751), 638-641.
- Schumann, W.O. (1952). On the free oscillations of a conducting sphere
  which is surrounded by an air layer and an ionosphere shell. *Z.
  Naturforsch. A*, 7(2), 149-154.

The evidence for seismo-electromagnetic precursors remains **contested**
in the scientific community. Mercury Agent's role is to provide an
empirical observation channel, not to validate or refute the hypothesis.
All findings require expert human assessment.

### Conservative Conclusion

The cross-domain frequency correlation module provides infrastructure for
empirical observation of spectral coincidences across domain boundaries.
No claims of predictive capability are made. The system flags correlations
for human review with mandatory disclaimer:

> "Correlated spectral anomaly detected -- requires human assessment."

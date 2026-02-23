# Cross-Domain Frequency Analysis

## Overview

The `CrossDomainFrequencyCorrelator` module
(`src/omni_mercury_engine/detectors/cross_domain_frequency.py`)
identifies overlapping significant frequency bands across concurrent
SpectralDomainOracle instances running on different data domains.

## Methodology

1. Extract per-band anomaly scores from each domain's
   `FrequencyInfluenceVector`.
2. For every domain pair, check Hz ranges for band overlap.
3. Compute correlation strength as the geometric mean of the two
   band anomaly scores.
4. Report significant overlaps (strength >= 0.3 by default).

## Schumann Resonance Bands

The environmental Oracle defines Schumann resonance bands:

| Band | Hz Range | Weight |
|------|----------|--------|
| sub_schumann | 1.0 - 7.83 | 0.10 |
| schumann_fundamental | 7.83 - 8.5 | 0.20 |
| schumann_harmonic_1 | 8.5 - 14.3 | 0.15 |
| schumann_harmonics_upper | 14.3 - 33.8 | 0.10 |

The space Oracle includes a `schumann_coupling` band (0.1 - 8.0 Hz)
that overlaps with the environmental sub-Schumann and fundamental
Schumann bands.

## Expected Cross-Domain Overlaps

| Domain A | Band A | Domain B | Band B | Overlap Hz |
|----------|--------|----------|--------|------------|
| environmental | sub_schumann (1.0-7.83) | space | schumann_coupling (0.1-8.0) | 1.0-7.83 |
| environmental | schumann_fundamental (7.83-8.5) | space | schumann_coupling (0.1-8.0) | 7.83-8.0 |
| environmental | sub_schumann (1.0-7.83) | infrastructure | seismic_structural (0.5-5.0) | 1.0-5.0 |

## Results

**Status: Awaiting live benchmark data.**

When the benchmark runs with USGS Earthquake and NOAA SWPC (SolarDynamics)
loaders active, cross-domain frequency correlation results will be recorded
here.

### Null Results

If no significant overlaps are detected, that is a valid and valuable
finding. It means the frequency-domain anomaly signatures in the tested
domains are statistically independent during the observation period.

## Interpretation Constraints

- **Correlation != causation.** Overlapping anomalous bands across
  domains indicate simultaneous spectral anomalies, not causal links.
- **Correlation != prediction.** Never "earthquake predicted" or
  "solar event imminent."
- **Requires human assessment.** Every correlator output includes this
  disclaimer. Automated action based solely on cross-domain correlation
  is prohibited.
- **Environmental factors.** Instrument noise, seasonal patterns, and
  anthropogenic interference can produce spurious overlaps.

# Oracle Noise Color Calibration

Applies to Mercury Agent **v2.1.x**. Last updated: 2026-05-20.

## Theory

Natural signals follow colored noise — their power spectral density (PSD) scales as a power law:

```
PSD(f) ∝ f^(-β)
```

Where β (beta) is the noise color exponent:

| β | Color | Physical Origin |
|---|-------|----------------|
| ≈ 0 | White | Independent samples, flat spectrum |
| ≈ 1 | Pink (1/f) | Self-organized criticality, turbulence |
| ≈ 2 | Brown (1/f²) | Brownian motion, random walks |
| < 0 | Blue/Violet | Differentiated processes |

## Problem

When the Oracle assumes white noise (flat PSD) but the data is colored, every low-frequency band in pink/brown noise appears anomalously powerful. The systematic bias masks real anomalies and causes significance tests to rarely fire.

## Solution: Noise Color Estimation

The Oracle estimates β from the reference (training) spectrum:

1. Compute PSD via FFT: `PSD = |FFT(signal)|²`
2. Fit log-log regression: `log(PSD) = -β·log(f) + C`
3. Classify: β < 0.5 → white, 0.5-1.5 → pink, > 1.5 → brown

The R² of the fit indicates confidence in the noise model.

## Band Power Correction

For each frequency band [f_lo, f_hi], the expected power fraction under the noise model is:

```
Expected(band) = ∫_{f_lo}^{f_hi} f^(-β) df / ∫_{f_min}^{f_max} f^(-β) df
```

The corrected power ratio = observed / expected. This removes the systematic bias from colored noise, allowing genuine anomalies to stand out.

## Domain-Expected Colors

| Domain | Expected β | Reasoning |
|--------|-----------|-----------|
| Environmental | 0.5-2.0 | Geophysical processes follow pink/brown |
| Ocean | 1.0-2.5 | Wave dynamics, tidal Brownian motion |
| Space | 0.5-2.0 | Solar wind, cosmic processes |
| Security | -0.5-0.5 | Independent packet arrivals ≈ white |
| Medical | 0.5-1.5 | Heart rate variability ≈ pink |
| Climate | 0.5-2.0 | Weather patterns, slow drift |

## Implementation

See `src/omni_mercury_engine/detectors/spectral_domain_oracle.py`:

- `SpectralDomainOracle._estimate_noise_color()` — β estimation
- `SpectralDomainOracle._expected_band_power()` — model-based expected power
- `SpectralDomainOracle._compute_band_anomaly()` — corrected z-scores
- `SpectralDomainOracle._compute_adaptive_alpha()` — window-aware thresholds

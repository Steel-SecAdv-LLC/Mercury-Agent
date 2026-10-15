# Visual anomaly analysis

`omni_mercury_engine.vision` is Mercury's image-anomaly analyser. It learns
what *normal* looks like from defect-free images, then detects, localises,
classifies and explains deviations. Each step is a stated, checkable
measurement, and decisions come with a calibrated false-alarm rate. It uses
no pretrained weights, so it runs anywhere NumPy, SciPy and OpenCV run.

It backs `BiometricAnomalyModel.predict` (image inputs),
`AgeProgressionEngine.assess_progression`, and the FINDOYOU evidence bridge
(see [`TRIO_INTEGRATION.md`](TRIO_INTEGRATION.md)).

## Capability map

| Requirement | How it is met | Where |
|---|---|---|
| Detect anomalies across image types | Any path or array (grey, RGB, RGBA, up to 16 channels, channel-first, uint/float/bool, NaN-masked) becomes one canonical form with recorded provenance. Inputs it cannot handle are refused with a stated reason. | `vision/imaging.py::to_canonical` |
| Learn from normal data | Normal-model scoring: a PatchCore-style k-centre coreset memory (kNN distance), a Ledoit-Wolf-shrunk PaDiM-style Mahalanobis model, and optional per-position Gaussians for aligned inputs. | `VisualAnomalyAnalyzer.fit` |
| Analyse visual features | 31 named, interpretable patch features per pyramid scale (28 for greyscale), across 9 groups: intensity, colour, gradient, orientation, LBP texture, frequency bands, noise, edges and lattice symmetry. At the default two scales that is 62 dimensions, all exactly invariant to a global gain. | `vision/features.py` |
| Localise anomalies precisely | Patch scores are spread back by coverage averaging and smoothed. Hysteresis then seeds regions at the calibrated image threshold and grows them over a pixel threshold, with half-maximum extents. | `VisualAnomalyAnalyzer._segment` |
| Recognise and classify anomalies | Self-supervised prototypes, one per polarity, of seven physical anomaly types simulated *on the user's own normal images*. The classifier uses pooled-covariance distances with an open-set bound, so an unfamiliar defect is reported as unclassified, not forced into a type. A soft-AND physical rule table is the fallback. Labelled examples can be added through `learn_anomaly_types`. | `vision/simulation.py`, `_learn_type_prototypes` |
| Provide interpretable results | Each region reports: box, peak, area, type with posterior and distance, the largest feature deviations in robust σ, and forensic alerts. `to_dict()` is JSON-ready. | `VisualAnomalyReport` |
| Handle varied environments | Robust per-tile capture statistics (exposure, contrast, noise, blur and others) are compared with the normal envelope. Out-of-envelope images get a *withheld* decision (`unavailable`) or a flagged one; `recalibrate()` re-anchors the threshold for a new session. | `vision/imaging.py::assess_environment` |
| Real time or batch | `analyze_batch` for batch work. `StreamingVisualMonitor` adds frame persistence and p50/p95 latency, and sheds forensics when over budget. `AnalyzerConfig.fast()` is a single-scale, sparser profile. | `vision/analyzer.py` |

### Forensics (separate from the content decision)

These screens never change `is_anomalous`. The content decision keeps its
conformal guarantee, and the screens report separately through
`manipulation_suspected` and `forensic_alerts`:

- **Noise-level inconsistency:** a block-wise Immerkær noise estimate
  (texture-robust), compared across the image.
- **JPEG grid phase:** detects re-compression on a shifted 8×8 grid. Both axes
  must show blockiness of at least 1.2.
- **Copy-move:** quantised 4×4 DCT block descriptors with shift voting at a
  step of one pixel.
- **Periodic spectral peaks:** moiré and halftone cues, which signal a screen
  or print replay.

### The decision and its guarantee

Split conformal prediction gives p = (1 + #{calibration scores ≥ s}) / (n + 1).
An image is anomalous when p ≤ α. For calibration images exchangeable with
the test images, this bounds the false-alarm rate by α (marginally over
calibration draws). Two consequences are measured below rather than assumed:

- **Acquisition shift breaks exchangeability.** When the test session differs
  from the training session, call `recalibrate()` on a few normals from the
  new session.
- **Small calibration sets limit α.** With n calibration images, α below
  1/(n+1) cannot be certified, and the analyser says so.

### Lattice symmetry (periodic textures)

Meshes, weaves and tilings are defined by their translation symmetry. A bend,
a broken wire or a misplaced element is a local break of that symmetry, which
purely local statistics cannot see. For each image and scale the analyser
does the following:

1. **Finds the lattice.** It computes a Hann-windowed, zero-padded
   autocorrelation, normalised by the window's own autocorrelation, so the
   two shortest basis vectors are found at any rotation. A parabolic peak fit
   gives sub-pixel accuracy, which matters because periods are rarely whole
   pixels.
2. **Compares each pixel with its neighbours one period ahead and one behind.**
   The comparison runs on log-luminance minus a separable 9-pixel median
   background, so a lighting or exposure change is not a break. It keeps the
   smaller difference, so a defect is blamed on itself, not on its neighbour.
3. **Reports no evidence near the border.** Within 6 px of the border, and
   where both neighbours fall outside the image, the image's median residual
   is reported instead of a break, because mirrored samples are not the
   lattice.

`lattice_any_axis` and `lattice_both_axes` are the per-pixel max and min
over the two basis vectors. The tests in
`tests/vision/test_features_forensics.py::TestLatticeSymmetry` show:
- a 7.3 × 11 px lattice is recovered to within 0.05 px;
- a lattice rotated 20° is recovered to within 0.5°;
- a foreign patch scores more than 10× its surroundings;
- a 0.45× exposure step does not register as a break;
- borders and corner pockets stay below 0.05 at the patch level.

## Age progression

`AgeProgressionEngine.progress_face` combines geometry and texture:

- **Geometry:** cardioidal-strain craniofacial growth (Todd, Mark, Shaw &
  Pittenger, 1980), R′ = R(1 + k(1 − cos θ)), resampled with cubic
  interpolation. It applies only to children. The growth schedule
  g(age) = 1 − exp(−age/6) is a stated, unfitted assumption.
- **Texture:** a deterministic difference-of-Gaussians mid-band detail model
  that injects no noise.

`assess_progression` audits the result: an SSIM below 0.6, a noise ratio above
1.5 or a sharpness ratio below 0.5 is flagged, so a damaging progression cannot
pass silently.

## `BiometricAnomalyModel.predict`

- **Image inputs:**
  - Once `fit_normal` has been called, predict returns the full visual report.
  - Without a fitted model it runs the environment and forensic screens,
    including presentation-attack cues (moiré, noise outliers, misaligned JPEG
    grids).
  - Face attributes appear only when DeepFace actually measured them
    (`attributes_measured`); no age or confidence value is invented.
- **Tabular inputs:** these keep the harmonic score, labelled through
  `anomaly_score_basis`.

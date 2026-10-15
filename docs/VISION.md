# Visual anomaly analysis

`omni_mercury_engine.vision` is Mercury's image-anomaly analyser. It learns
what *normal* looks like from defect-free images, then detects, localises,
classifies and explains deviations. Each step is a stated, checkable
measurement, and decisions come with a calibrated false-alarm rate. It uses
no pretrained weights, so it runs anywhere NumPy, SciPy and OpenCV run.

It backs `BiometricAnomalyModel.predict` (image inputs),
`AgeProgressionEngine.assess_progression`, and the FINDΩYOU™ evidence bridge
(`integrations/findoyou_bridge.py`).

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

## Measured results

Both benchmarks are committed and seeded, and their outputs live in
`benchmarks/results/`. The figures below are from those files, produced by
the code in this repository. The texture images are 256 px, and the analyser
was fitted on 100 defect-free training images per category (60 fit, 37
conformal calibration, 3 normalisation).

### MVTec AD textures (real defects, `benchmarks/mvtec_visual_benchmark.py`)

The data are MVTec AD (Bergmann et al., CVPR 2019; CC BY-NC-SA 4.0). They are
downloaded at run time from the `Voxel51/mvtec-ad` mirror under a pinned
manifest digest, and nothing is redistributed. No pretrained weights are used.

| Category | Lattice group | Image AUROC | Pixel AUROC | Strict FAR (α 0.05) | Strict detection | Recalibrated FAR (α 0.1) | Recalibrated detection |
|---|---|---|---|---|---|---|---|
| carpet | on | 0.965 | 0.982 | 0.429 | 0.989 | 0.056 | 0.888 |
| grid | on | 0.996 | 0.984 | 0.000 | 0.614 | 0.182 | 0.982 |
| leather | off | 0.999 | 0.996 | 0.594 | 1.000 | 0.000 | 0.978 |
| tile | off | 0.999 | 0.975 | 0.000 | 0.929 | 0.000 | 0.976 |
| wood | off | 0.965 | 0.972 | 0.211 | 0.983 | 0.100 | 0.950 |
| **mean** | | **0.985** | **0.982** | 0.247 | 0.903 | **0.068** | **0.955** |

The same pipeline without the lattice group (measured during development) is
the comparison:

| Metric | Without lattice | With lattice |
|---|---|---|
| Mean image AUROC | 0.914 | 0.985 |
| Mean pixel AUROC | 0.958 | 0.982 |
| Mean recalibrated detection | 0.819 | 0.955 |
| Mean recalibrated FAR | 0.072 | 0.068 |
| Grid image AUROC | 0.656 | 0.996 |
| Grid recalibrated detection | 0.316 | 0.982 |
| Grid "bent" detected (strict) | 0.00 | 0.83 |

#### Read these honestly

- **The strict false-alarm rate is not controlled at α on MVTec.** The rate
  measured on MVTec's good *test* images is 0.247 on average, against a
  target of 0.05. Those test images are measurably shifted from the good
  *training* images: on carpet they separated at AUROC 0.66 when measured
  with the pre-lattice features. That breaks the
  exchangeability the guarantee needs. Recalibrating on 10 good images from
  the test session brings the false-alarm rate to 0.068 at α 0.1. In
  deployment, call `recalibrate()` on a few normals from each new capture
  session. The self-model records this as a stated limitation.
- **Small-sample rates are coarse.** Grid's recalibrated false-alarm rate is
  2 images of 11, and strict thresholds come from 37 calibration images (the
  threshold is the largest of them at α 0.05).
- **Weakest defect types:**
  - grid `glue`: 0.09 detected at the strict threshold
  - grid `broken`: 0.50
  - tile `glue_strip`: 0.72
- **Defect typing on real data is coarse.** MVTec's defect names are not the
  analyser's physical types, so typing is not scored there; see the synthetic
  benchmark below.

### Held-out synthetic textures (exact types, `benchmarks/vision_synthetic_heldout.py`)

Fit on 40 normals, then evaluated on unseen images and seeds at α 0.1:

| Domain | Held-out FAR (n=60) | Forensic false positives | Same-definition typing | Cross-generator typing | Detection (8 injectors) | Image AUROC (min over injectors) | Pixel AUROC (min) |
|---|---|---|---|---|---|---|---|
| woven (periodic) | 0.033 | 0.000 | 99.3% (139/140) | 85.4% (205/240) | 100% | 1.000 | 0.988 |
| leather (aperiodic) | 0.050 | 0.000 | 95.0% (133/140) | 77.5% (186/240) | 100% | 0.983 | 0.994 |

The cross-generator injectors are written independently of the analyser's
simulator. The misses are physically ambiguous cases:
- a flat dark occluder is typed as an exposure anomaly (10 of 30 on woven;
  all 30 on leather are typed as blur);
- structured foreign matter on a weave is typed as a structural change (25 of
  30), which is physically defensible because it breaks the weave's lattice.

### Latency

Measured on a 4-core container, single process, 256 px:

| Profile | Median | p95 |
|---|---|---|
| Default (two scales) | 240–270 ms | 320–360 ms |
| `AnalyzerConfig.fast()` (one coarse scale) | 79–83 ms | 93–102 ms |

The fast profile runs at about 12 frames per second.

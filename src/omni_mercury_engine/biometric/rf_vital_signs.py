# Copyright (C) 2025 Steel Security Advisors LLC
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contactless respiration and heart-rate estimation from Wi-Fi CSI or radar.

Consent and scope. Contactless sensing of people requires their informed
consent. Every public entry point that processes a recording of a person
(``estimate_vital_signs`` and ``track_vital_signs``) takes a mandatory
keyword ``consent_confirmed`` and raises ``ConsentRequiredError`` unless it is
exactly ``True`` or a ``truth_decipher.ConsentRecord`` whose scope includes
``VITAL_SIGNS_CONSENT_SCOPE``. ``ConsentRequiredError`` is the class defined
in ``omni_mercury_engine.truth_decipher`` (re-exported here), so a caller
that catches it from either module catches every consent refusal. This module performs no person identification, no
localisation, no imaging and no pose reconstruction. It estimates two rates
(breaths per minute and beats per minute) for one cooperative, stationary
subject. Multiple people and moving subjects are out of scope: the estimator
looks for evidence of either (a second periodic source in the respiration
band, body-motion artefacts) and abstains when it finds it.

Inputs. Either Wi-Fi channel state information (an amplitude array of shape
``(T, S)`` or ``(T, S, A)`` over ``S`` subcarriers and ``A`` receive antennas,
and/or the matching phase array), or a radar series of shape ``(T,)``: chest
displacement in metres, or CW / FMCW phase in radians with the carrier
wavelength. All inputs are sampled at ``fs`` Hz, or carry packet timestamps.

Method (every step is cited; every default without a citation is declared a
configurable default with its rationale in ``VitalSignConfig``):

1. Irregular packet times are resampled onto a uniform grid by linear
   interpolation. The grid is never coarser than the median packet rate (it
   is the smallest integer multiple of the output rate at or above it), so
   interference above the output Nyquist frequency is removed by the
   anti-alias filter of step 4 instead of being folded into the vital-sign
   bands by the interpolation. Lost samples (non-finite rows, timestamp gaps) in runs no
   longer than ``max_gap_s`` are bridged by interpolation; longer runs are
   marked missing and excluded rather than invented.
2. Phase sanitisation. Single-antenna CSI phase is unwrapped across
   subcarriers and the per-packet linear term in subcarrier index (timing and
   phase offsets, Sen et al. 2012, section 4.1) is removed. Sen et al. take
   the slope from the two edge subcarriers; this module fits the same linear
   model by least squares over all subcarriers. With two or more antennas the
   phase difference between antennas of the same receiver is used instead,
   which cancels the offsets common to both RF chains (Wang, Yang and Mao,
   PhaseBeat, 2017). CSI phase streams are re-wrapped about their circular
   mean instead of being unwrapped in time, so noise-induced cycle slips
   cannot accumulate. Radar phase is unwrapped and converted to displacement
   with ``d = wavelength * phase / (4 pi)`` (Droitcour et al. 2004).
3. Outliers are removed per stream with the Hampel identifier (Hampel 1974;
   Davies and Gather 1993; Pearson 2002), as Liu et al. (2015, 2018) do for
   CSI amplitude. The scale constant is 1.4826, the reciprocal of the
   standard-normal upper quartile, which makes the MAD a consistent estimate
   of the standard deviation for Gaussian data.
4. Streams are anti-alias filtered and resampled to ``analysis_fs`` with a
   polyphase FIR (``scipy.signal.resample_poly``), then linearly detrended.
5. Body motion is detected from short-time statistics, following the
   short-time energy of moving variance of Liu et al. (2018, section VI-A;
   short-time energy, Rabiner and Schafer 1978), made coherent across
   streams by using the largest eigenvalue of each frame's covariance.
   Frames above ``motion_threshold`` times the recording's median frame are
   excluded, and only the longest contiguous clean segment of a window is
   analysed, as Vital-Radio stops estimating while the person moves (Adib et
   al. 2015). Displacement inputs also have an absolute per-frame limit.
6. Stream selection ranks streams by the fraction of their power inside the
   respiration band, a periodicity score. Liu et al. (2015, 2018) rank
   subcarriers by amplitude variance instead; that criterion is available as
   ``selection="variance"``. The respiration waveform of a multi-stream input
   is the first principal component of the selected, band-passed streams
   (PCA as used for CSI by Wang et al. 2015, CARM).
7. Respiration: zero-phase Butterworth band-pass (default 0.1 to 0.5 Hz, i.e.
   6 to 30 breaths/min). The rate is the peak of the Hann-windowed,
   zero-padded periodogram (Harris 1978), averaged over selected streams after
   normalising each to unit in-band power (the mean PSD across subcarriers of
   Liu et al. 2018, section V), refined by parabolic interpolation of the
   log spectrum (Smith and Serra 1987). It is cross-checked by counting
   waveform peaks with a minimum spacing set by the band's upper edge (the
   peak-to-peak method of Liu et al. 2018, section IV-C).
8. Heart rate: band-pass (default 0.8 to 2.0 Hz, i.e. 48 to 120 beats/min),
   then removal of the respiration harmonics that fall in the heart band
   (up to the fifth, as Tazen, Sasaoka and Okamoto 2023 handle). The default
   is a least-squares canceller with references phase-locked to the measured
   respiration phase, a batch form of the adaptive noise canceller with
   periodic references (Widrow et al. 1975) that follows breath-to-breath
   rate variability; cascaded IIR notches (Tazen et al. 2023; notch design
   of Orfanidis 1996) are the alternative. The rate is the spectral peak, as
   in step 7, withheld when it coincides with any respiration harmonic to
   within the window's resolution. Breathing faster than the respiration
   band puts its second harmonic in the heart band, so the fundamental used
   for harmonic removal and the coincidence test is also searched up to just
   below the heart band and preferred when it lies above the respiration
   band; when no fundamental passes the SNR test and a band maximum lies on
   an upper edge, a heart peak at a harmonic of the unresolved frequency
   range is withheld (``HARMONIC_AMBIGUITY``). A breathing fundamental found
   above the respiration band also withholds the respiration rate, and a
   heart peak whose half lies on a suppressed respiration harmonic (the
   heartbeat's fundamental may have been cancelled with it) is withheld.
9. Quality is the in-band SNR: the power within a main-lobe-wide region
   around the peak over the remaining in-band power. It is tested against
   ``null_snr_threshold_db``, the level that Gaussian noise alone exceeds
   with probability ``false_alarm`` for the effective number of averaged
   streams (the participation ratio of their in-band coherence outside the
   peak region, so noise common to all subcarriers counts once) and the
   segment length (a Beta
   model whose dof scale was fitted by Monte Carlo to white noise, plus a
   margin covering 1/f noise). The
   estimator abstains (``rate_bpm=None`` with an ``AbstainReason``) on low
   SNR, on a peak at the band edge, on a second resolvable non-harmonic
   respiration peak (possible second person), when the spectral and
   peak-count respiration rates disagree, on a heart peak at a respiration
   harmonic, and when the data is flat, too short or motion-contaminated.

Physiological scale (used for the radar motion check and the synthetic
validation, not as a prior inside the estimator): tidal-breathing chest-wall
displacement of about 4 to 12 mm (De Groote et al. 1997) and cardiac
chest-wall displacement of about 0.2 to 0.5 mm (Ramachandran and Singh 1989),
both as reported by Kim and Jeong (2020). Adult resting rates are usually
quoted as 12 to 20 breaths/min and 60 to 100 beats/min (Kebe et al. 2020);
the default bands are wider so that bradypnoea, tachypnoea, athletic
bradycardia and arousal-driven tachycardia stay in range.

Validation. No accuracy claim is made beyond the following.

* Reproducible from this repository: the seeded synthetic model and the
  noise-only checks in ``tests/biometric/test_rf_vital_signs.py`` (a small
  version of the model described below), including the white-noise
  false-alarm record
  ``TestSpectral::test_null_threshold_false_alarm_on_white_noise``, the
  common-mode-noise false-alarm record
  ``TestAbstention::test_common_mode_noise_respects_the_false_alarm_rate``
  and the respiration-harmonic regression
  ``TestEstimateRadar::test_fast_breathing_harmonic_is_never_reported``.
* Not reproducible from this repository: the figures below were produced by
  an out-of-tree validation (a larger version of the synthetic model, and a
  script that demodulated public radar recordings). Neither the script nor
  its outputs are in this repository or its history, and they predate later
  changes to the estimator (the effective-stream noise gate, the wide-band
  respiration-harmonic search and the timestamp anti-alias grid), so treat
  them as unverified indications, not as measured properties of this code.
  Noise only (1000 windows each of white and 1/f Gaussian noise, one radar
  stream or 30 independent CSI subcarriers): no heart rate reported; a
  respiration rate in 0 of 1000 windows except 6 of 1000 for 1/f CSI noise.
  Synthetic model (static multipath plus a body path at 5.32 GHz, 24 GHz
  radar phase, rate variability, 3 s motion bursts in half of the
  recordings, 400 windows per condition): respiration mean absolute error
  0.19 to 0.23 breaths/min with 70 to 95 % of windows reported at receiver
  SNRs of 0 to 30 dB (CSI) and 20 to 400 um noise (radar); heart rate on
  radar 0.38 to 0.71 beats/min with 57 to 70 % reported, CSI amplitude
  0.73 beats/min at 30 dB (56 % reported), CSI phase difference 1.04
  beats/min at 30 dB (48 % reported), essentially withheld at 10 dB and
  below. Real radar recordings (Schellenberger et al. 2020; 24 GHz CW,
  resting scenario; I/Q correction and demodulation done outside this
  module): on three subjects (176 windows) heart rate reported in 18 % of
  windows with a mean absolute error of 0.75 beats/min against ECG, and
  respiration in 21 % with 0.10 breaths/min against thoracic impedance;
  coverage was low because several subjects breathed at 6 to 8 breaths/min,
  on the default band edge, and the harmonic-coincidence test withheld
  half of the heart estimates.

Not validated: real Wi-Fi CSI recordings, multiple antennas on real hardware,
subjects other than healthy adults at rest, breathing slower than the band
edge, any clinical or forensic decision. Known limits: amplitude-only CSI
barely registers torso motion while breathing already sweeps the body
path's phase; motion is missed at 5 dB CSI SNR and below; the relative
motion test also flags irregular breathing depth.

Temporal resolution. A spectral rate estimate averages over its analysis
window, so a window of ``W`` seconds cannot resolve rate changes faster than
about ``W``. In 10 s windows respiration is never reported (the SNR test is
undefined there), and heart rate is withheld whenever its peak lies within
about 0.13 to 0.18 Hz of a suppressed respiration harmonic
(``cancellation_guard_bins``), which at ordinary breathing rates covers
most of the heart band: in the test module's stepped 10 s radar track no
window is reported. Callers that need stimulus-locked responses should use the
respiration waveform excerpts (``extract_waveform_excerpt``) and compare
windowed rates against a pre-stimulus baseline only with ``W`` no longer than
the interval of interest.

In particular this module cannot, as configured, supply per-stimulus heart
rates for the Concealed Information Test (CIT) of
``omni_mercury_engine.truth_decipher``: its shortest analysed segment (``min_clean_s``, default
20 s, and ``track_vital_signs`` windows no shorter than that) is longer than
the CIT's default 3 s heart-rate baseline and 10 s response windows, so a
rate from this module mixes several stimuli and the inter-stimulus interval.

References:
    Adib, F., Mao, H., Kabelac, Z., Katabi, D., Miller, R. C. (2015). Smart
    homes that monitor breathing and heart rate. Proc. ACM CHI 2015.

    Davies, L., Gather, U. (1993). The identification of multiple outliers.
    J. Amer. Statist. Assoc. 88(423), 782-792.

    De Groote, A., et al. (1997). Chest wall motion during tidal breathing.
    J. Appl. Physiol. 83(5), 1531-1537.

    Droitcour, A. D., Boric-Lubecke, O., Lubecke, V. M., Lin, J., Kovacs, G.
    T. A. (2004). Range correlation and I/Q performance benefits in
    single-chip silicon Doppler radars for noncontact cardiopulmonary
    monitoring. IEEE Trans. Microw. Theory Tech. 52(3), 838-848.

    Hampel, F. R. (1974). The influence curve and its role in robust
    estimation. J. Amer. Statist. Assoc. 69(346), 383-393.

    Harris, F. J. (1978). On the use of windows for harmonic analysis with
    the discrete Fourier transform. Proc. IEEE 66(1), 51-83.

    Kebe, M., et al. (2020). Human vital signs detection methods and
    potential using radars: a review. Sensors 20(5), 1454.

    Kim, H., Jeong, J. (2020). Non-contact measurement of human respiration
    and heartbeat using W-band Doppler radar sensor. Sensors 20(18), 5209.

    Liu, J., Wang, Y., Chen, Y., Yang, J., Chen, X., Cheng, J. (2015).
    Tracking vital signs during sleep leveraging off-the-shelf WiFi. Proc.
    ACM MobiHoc 2015, 267-276.

    Liu, J., Chen, Y., Wang, Y., Chen, X., Cheng, J., Yang, J. (2018).
    Monitoring vital signs and postures during sleep using WiFi signals.
    IEEE Internet Things J. 5(3), 2071-2084.

    Orfanidis, S. J. (1996). Introduction to Signal Processing. Prentice
    Hall.

    Pearson, R. K. (2002). Outliers in process modeling and identification.
    IEEE Trans. Control Syst. Technol. 10(1), 55-63.

    Rabiner, L. R., Schafer, R. W. (1978). Digital Processing of Speech
    Signals. Prentice Hall.

    Ramachandran, G., Singh, M. (1989). Three-dimensional reconstruction of
    cardiac displacement patterns on the chest wall during the P, QRS and
    T-segments of the ECG by laser speckle interferometry. Med. Biol. Eng.
    Comput. 27, 525-530.

    Schellenberger, S., Shi, K., Steigleder, T., Malessa, A., Michler, F.,
    Hameyer, L., Neumann, N., Lurz, F., Weigel, R., Ostgathe, C., Koelpin, A.
    (2020). A dataset of clinically recorded radar vital signs with
    synchronised reference sensor signals. Sci. Data 7, 291.

    Sen, S., Radunovic, B., Choudhury, R. R., Minka, T. (2012). You are
    facing the Mona Lisa: spot localization using PHY layer information.
    Proc. ACM MobiSys 2012, 183-196.

    Smith, J. O., Serra, X. (1987). PARSHL: an analysis/synthesis program
    for non-harmonic sounds based on a sinusoidal representation. Proc.
    International Computer Music Conference 1987.

    Tazen, M., Sasaoka, N., Okamoto, Y. (2023). Non-contact heart rate
    measurement based on adaptive notch filter and elimination of
    respiration harmonics. IEEE Access 11, 46107-46119.

    Wang, W., Liu, A. X., Shahzad, M., Ling, K., Lu, S. (2015).
    Understanding and modeling of WiFi signal based human activity
    recognition. Proc. ACM MobiCom 2015, 65-76.

    Wang, X., Yang, C., Mao, S. (2017). PhaseBeat: exploiting CSI phase data
    for vital sign monitoring with commodity WiFi devices. Proc. IEEE ICDCS
    2017, 1230-1239.

    Widrow, B., Glover, J. R., McCool, J. M., Kaunitz, J., Williams, C. S.,
    Hearn, R. H., et al. (1975). Adaptive noise cancelling: principles and
    applications. Proc. IEEE 63(12), 1692-1716.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray
from scipy import signal, special

from omni_mercury_engine.truth_decipher import ConsentRecord, ConsentRequiredError

__all__ = [
    "MAD_TO_SIGMA",
    "VITAL_SIGNS_CONSENT_SCOPE",
    "AbstainReason",
    "ConsentRequiredError",
    "RateEstimate",
    "SpectralPeak",
    "VitalSignConfig",
    "VitalSignEstimate",
    "VitalSignTrack",
    "WaveformExcerpt",
    "antenna_phase_difference",
    "band_power_ratio",
    "cancel_respiration_harmonics",
    "centre_wrapped_phase",
    "count_cycle_rate",
    "detect_motion",
    "dominant_frequency",
    "effective_stream_count",
    "estimate_vital_signs",
    "extract_waveform_excerpt",
    "first_principal_component",
    "hampel_filter",
    "null_snr_threshold_db",
    "rate_series",
    "rates_within",
    "resample_uniform",
    "respiration_harmonics",
    "respiration_phase",
    "respiration_series",
    "sanitize_csi_phase",
    "suppress_harmonics",
    "track_vital_signs",
    "windows_within",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

#: Consistency constant of the MAD for Gaussian data: 1 / Phi^{-1}(3/4).
MAD_TO_SIGMA = 1.4826

_EPS = 1e-300

#: Scope a ``truth_decipher.ConsentRecord`` must include to authorise
#: contactless vital-sign sensing.
VITAL_SIGNS_CONSENT_SCOPE = "contactless_vital_signs"


class AbstainReason(StrEnum):
    """Machine-readable reason a rate estimate was withheld."""

    INSUFFICIENT_DATA = "insufficient_data"
    MOTION = "motion"
    FLAT_SIGNAL = "flat_signal"
    LOW_SNR = "low_snr"
    PEAK_AT_BAND_EDGE = "peak_at_band_edge"
    MULTIPLE_SOURCES = "multiple_sources"
    COUNT_DISAGREEMENT = "count_disagreement"
    HARMONIC_AMBIGUITY = "harmonic_ambiguity"


@dataclass(frozen=True)
class VitalSignConfig:
    """Configurable defaults for the estimator.

    Every value without a citation in the module docstring is a configurable
    default whose rationale is given here.

    Attributes:
        respiration_band_hz: Respiration pass band. Default 0.1 to 0.5 Hz
            (6 to 30 breaths/min), wider than the 12 to 20 breaths/min adult
            resting range (Kebe et al. 2020) to keep bradypnoea and tachypnoea
            in range.
        heart_band_hz: Heart pass band. Default 0.8 to 2.0 Hz (48 to 120
            beats/min), wider than the 60 to 100 beats/min resting range
            (Kebe et al. 2020) to keep athletic bradycardia and
            arousal-driven tachycardia in range.
        analysis_fs: Rate (Hz) streams are resampled to before analysis, or
            ``None`` to keep the input rate. Default 20 Hz: five times the
            Nyquist rate of the 2 Hz heart-band edge, and enough to place the
            fifth respiration harmonic of a 0.5 Hz breath below Nyquist.
        filter_order: Butterworth prototype order of each band-pass (the
            band-pass has twice this order). Default 4: a common compromise
            between roll-off and ringing for zero-phase filtering.
        hampel_half_width: Hampel window half-width in input samples.
            Default 3 (a 7-sample window): CSI outliers are typically single
            corrupted packets, and a short window keeps the cost linear.
        hampel_n_sigmas: Hampel threshold in robust standard deviations.
            Default 3, the conventional "three-sigma" rule (Pearson 2002).
        max_gap_s: Longest run of lost samples bridged by interpolation
            (a gap between packet timestamps, or consecutive non-finite
            samples); longer runs are excluded from analysis. Default 0.25 s,
            five samples at the default analysis rate.
        zero_pad_factor: Periodogram length as a multiple of the segment
            length (rounded up to a power of two). Default 8: the
            interpolated peak then moves by well under 0.1 breaths/min
            between padding choices.
        n_streams: Number of streams kept by selection. Default 10, a third
            of the 30 subcarrier groups an Intel 5300 reports per antenna
            pair, so that weak subcarriers near a Fresnel-zone null are
            dropped while spectral averaging still has several streams.
        selection: ``"band_power"`` (fraction of power inside the
            respiration band) or ``"variance"`` (Liu et al. 2015).
        respiration_signal_halfwidth_hz: Half-width of the region counted
            as signal around the respiration peak. The effective value is
            never below the Hann main-lobe half-width ``2 / duration``
            (Harris 1978). Default 0.05 Hz: covers a 10 % respiration-rate
            drift at 0.5 Hz within a window.
        heart_signal_halfwidth_hz: Same for the heart peak. Default 0.1 Hz:
            covers a 6 beats/min drift within a window.
        respiration_min_snr_db: Fixed respiration SNR threshold (dB), or
            ``None`` (default) to use ``null_snr_threshold_db`` for the
            effective number of averaged streams
            (``effective_stream_count``) and the segment length.
        heart_min_snr_db: Same for the heart band.
        false_alarm: Target probability that noise alone passes the SNR
            test when the thresholds come from the null model. Default 0.01.
        null_dof_scale: Degrees-of-freedom scale of the Beta null model in
            ``null_snr_threshold_db``. Default 0.45, fitted by Monte Carlo to
            the 99th percentile of the statistic under white Gaussian noise
            (median residual +0.06 dB over 1 to 10 streams and 15 to 120 s).
        null_colour_margin_db: Margin added to the white-noise threshold so
            that 1/f noise is also covered. Default 2.6 dB, the largest
            shortfall of the white-noise model against 1/f noise on the same
            Monte Carlo grid.
        multi_source_power_ratio: A second resolvable, non-harmonic
            respiration peak holding at least this fraction of the main
            peak's power means more than one periodic source. Default 0.5.
        count_tolerance_bpm: Largest accepted difference between the
            spectral and peak-count respiration rates. Default 3 breaths/min
            (a quarter of a 12 breaths/min rate): a larger gap means the
            spectral peak is not the breathing cycle.
        count_prominence_fraction: Minimum peak prominence for the breath
            count, as a fraction of the waveform's 5th-to-95th percentile
            range. Default 0.25.
        harmonic_suppression: How respiration harmonics in the heart band
            are removed: ``"regression"`` (default; phase-locked least
            squares, ``cancel_respiration_harmonics``), ``"notch"``
            (cascaded IIR notches as in Tazen et al. 2023) or ``"none"``.
        harmonic_poly_degree: Legendre degree of the breathing-depth
            modulation allowed in the regression. Default 2 (constant,
            linear and quadratic change of depth over a window), capped at
            ``floor(T / 15 s)`` for a segment of ``T`` seconds: each degree
            lets a reference spread by roughly one cycle per window, and in
            short windows that spread absorbs nearby heart components and
            leaves structured residue (measured: degree 2 on 10 s windows
            produced 20 beats/min errors that degree 0 did not).
        max_harmonic_order: Highest respiration harmonic suppressed. Default
            5, the order Tazen et al. (2023) report handling.
        notch_width_hz: -3 dB width of each harmonic notch, and the margin
            (half of it) by which harmonics just outside the heart band are
            still suppressed. Default 0.1 Hz: spans a 5 % respiration-rate
            drift up to the fourth harmonic of 0.25 Hz.
        harmonic_coincidence_fraction: The heart estimate is withheld
            (``HARMONIC_AMBIGUITY``) when its peak lies within this fraction
            of the Rayleigh resolution ``1 / T`` of any respiration harmonic
            ``k >= 2`` (suppressed or not: a peak there may be residue),
            widened by ``k * respiration_precision_bins / T``. Default 0.5: two components closer
            than half the resolution of the window cannot be told apart.
            0 disables the check.
        respiration_precision_bins: Assumed precision of the respiration
            peak, in DFT bins ``1 / T``; the ``k``-th harmonic's position is
            uncertain by ``k`` times it. Default 0.1: the 95th percentile of
            the interpolated-peak error measured on noise-free breathing
            waveforms (fundamental plus second harmonic) over 20 to 30 s
            windows (0.07 to 0.11 bins).
        cancellation_guard_bins: The heart estimate is withheld
            (``HARMONIC_AMBIGUITY``) when its peak lies within this many DFT
            bins ``1 / T`` of a respiration harmonic that was suppressed,
            widened by ``k * respiration_precision_bins / T``. Suppression
            removes or distorts a heartbeat that close to the harmonic, so
            the residual peak there is biased rather than absent. Default
            1.25: on 30 and 40 s radar windows (both suppression modes) a
            heartbeat exactly on a suppressed harmonic left a residual peak
            up to 1.13 bins away, and peaks biased by more than 0.5 beats/min
            lay within 1.2 bins. 0 disables the check.
        motion_frame_s: Frame length of the short-time motion statistics.
            Default 1 s, shorter than one breath at 30 breaths/min.
        motion_threshold: Frame statistic, relative to the recording's
            median frame, above which a frame is motion. Default 4: steady
            breathing varies a frame's statistic by about the square root
            of two between the steepest and flattest part of a cycle.
        motion_guard_s: Samples within this many seconds of a motion frame
            are also excluded. Default 1 s, about the settling time of the
            band-pass filters' edge transients at the heart band.
        max_motion_fraction: Abstain with ``MOTION`` when more than this
            fraction of a window is motion. Default 0.3.
        max_radar_excursion_m: For displacement inputs, a frame whose
            peak-to-peak displacement exceeds this is motion regardless of
            the relative test. Default 0.025 m, twice the upper end of the
            tidal-breathing range (De Groote et al. 1997).
        min_clean_s: Shortest clean segment analysed. Default 20 s: two
            breaths at the 6 breaths/min band edge.
    """

    respiration_band_hz: tuple[float, float] = (0.1, 0.5)
    heart_band_hz: tuple[float, float] = (0.8, 2.0)
    analysis_fs: float | None = 20.0
    filter_order: int = 4
    hampel_half_width: int = 3
    hampel_n_sigmas: float = 3.0
    max_gap_s: float = 0.25
    zero_pad_factor: int = 8
    n_streams: int = 10
    selection: Literal["band_power", "variance"] = "band_power"
    respiration_signal_halfwidth_hz: float = 0.05
    heart_signal_halfwidth_hz: float = 0.1
    respiration_min_snr_db: float | None = None
    heart_min_snr_db: float | None = None
    false_alarm: float = 0.01
    null_dof_scale: float = 0.45
    null_colour_margin_db: float = 2.6
    multi_source_power_ratio: float = 0.5
    count_tolerance_bpm: float = 3.0
    count_prominence_fraction: float = 0.25
    harmonic_suppression: Literal["regression", "notch", "none"] = "regression"
    harmonic_poly_degree: int = 2
    max_harmonic_order: int = 5
    notch_width_hz: float = 0.1
    harmonic_coincidence_fraction: float = 0.5
    respiration_precision_bins: float = 0.1
    cancellation_guard_bins: float = 1.25
    motion_frame_s: float = 1.0
    motion_threshold: float = 4.0
    motion_guard_s: float = 1.0
    max_motion_fraction: float = 0.3
    max_radar_excursion_m: float | None = 0.025
    min_clean_s: float = 20.0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any value is outside its admissible range.
        """
        for name in ("respiration_band_hz", "heart_band_hz"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError(f"{name} must satisfy 0 < low < high, got {(low, high)}")
        if self.analysis_fs is not None and not (
            math.isfinite(self.analysis_fs) and self.analysis_fs > 0.0
        ):
            raise ValueError("analysis_fs must be positive and finite, or None")
        positive_ints = ("filter_order", "zero_pad_factor", "n_streams")
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.hampel_half_width < 0:
            raise ValueError("hampel_half_width must be non-negative")
        if self.max_harmonic_order < 1:
            raise ValueError("max_harmonic_order must be at least 1")
        non_negative = ("hampel_n_sigmas", "count_prominence_fraction")
        for name in non_negative:
            if not float(getattr(self, name)) >= 0.0:
                raise ValueError(f"{name} must be non-negative")
        positive = (
            "max_gap_s",
            "respiration_signal_halfwidth_hz",
            "heart_signal_halfwidth_hz",
            "multi_source_power_ratio",
            "count_tolerance_bpm",
            "notch_width_hz",
            "motion_frame_s",
            "motion_threshold",
            "min_clean_s",
            "null_dof_scale",
        )
        for name in (
            "harmonic_coincidence_fraction",
            "respiration_precision_bins",
            "cancellation_guard_bins",
        ):
            if not float(getattr(self, name)) >= 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in positive:
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.false_alarm < 1.0:
            raise ValueError("false_alarm must lie in (0, 1)")
        if not math.isfinite(self.null_colour_margin_db):
            raise ValueError("null_colour_margin_db must be finite")
        for name in ("respiration_min_snr_db", "heart_min_snr_db"):
            value = getattr(self, name)
            if value is not None and math.isnan(value):
                raise ValueError(f"{name} must be a number or None")
        if not self.motion_guard_s >= 0.0:
            raise ValueError("motion_guard_s must be non-negative")
        if not 0.0 <= self.max_motion_fraction <= 1.0:
            raise ValueError("max_motion_fraction must lie in [0, 1]")
        if self.max_radar_excursion_m is not None and not self.max_radar_excursion_m > 0.0:
            raise ValueError("max_radar_excursion_m must be positive or None")
        if self.selection not in ("band_power", "variance"):
            raise ValueError("selection must be 'band_power' or 'variance'")
        if self.harmonic_suppression not in ("regression", "notch", "none"):
            raise ValueError("harmonic_suppression must be 'regression', 'notch' or 'none'")
        if self.harmonic_poly_degree < 0:
            raise ValueError("harmonic_poly_degree must be non-negative")


@dataclass(frozen=True)
class RateEstimate:
    """A rate estimate, or an abstention with its reason.

    Attributes:
        rate_bpm: Estimated rate per minute, or ``None`` when abstaining.
        reason: Why the estimate was withheld, or ``None`` when it was made.
        detail: Human-readable explanation with the measured quantities.
        peak_hz: Interpolated spectral peak, when one was found (also
            reported on some abstentions, e.g. low SNR).
        snr_db: In-band SNR of the peak; ``nan`` when it was not computed.
        snr_threshold_db: Threshold the SNR was tested against (``nan``
            when no test was made).
        count_rate_bpm: Peak-count cross-check (respiration only).
    """

    rate_bpm: float | None
    reason: AbstainReason | None
    detail: str
    peak_hz: float | None = None
    snr_db: float = math.nan
    count_rate_bpm: float | None = None
    snr_threshold_db: float = math.nan

    @property
    def abstained(self) -> bool:
        """Return whether the estimate was withheld."""
        return self.rate_bpm is None


@dataclass(frozen=True)
class SpectralPeak:
    """The dominant in-band spectral peak of one or more streams.

    Attributes:
        frequency_hz: Interpolated peak frequency, or ``None`` when the
            maximum lies on a band edge or the band holds no power.
        snr_db: Power within ``halfwidth_hz`` of the peak over the remaining
            in-band power, in dB (``nan`` when undefined).
        at_band_edge: The in-band maximum is the first or last band bin.
        halfwidth_hz: Effective signal half-width used for the SNR.
        secondary_hz: Strongest other local peak that lies more than
            ``halfwidth_hz`` from the main peak and not within
            ``halfwidth_hz`` of half or twice its frequency, if any.
        secondary_power_ratio: That peak's power (bins shared with the main
            region excluded) over the main peak's power.
        joint_snr_db: SNR counting both peaks as signal.
    """

    frequency_hz: float | None
    snr_db: float
    at_band_edge: bool
    halfwidth_hz: float
    secondary_hz: float | None = None
    secondary_power_ratio: float = 0.0
    joint_snr_db: float = math.nan


@dataclass(frozen=True, eq=False)
class VitalSignEstimate:
    """Respiration and heart-rate estimates for one analysis window.

    Attributes:
        respiration: Respiration estimate (breaths per minute).
        heart: Heart-rate estimate (beats per minute).
        fs_analysis: Sampling rate the window was analysed at (Hz).
        segment_start_s: Start of the analysed clean segment, in seconds
            from the first sample of the input (or of the window).
        segment_end_s: End (exclusive) of the analysed segment, same origin.
        motion_fraction: Fraction of samples flagged as motion.
        missing_fraction: Fraction of samples lost (non-finite, or in a
            timestamp gap), including runs no longer than ``max_gap_s`` that
            were bridged by interpolation and still analysed.
        selected_streams: Indices of the streams used, in the order streams
            are formed: CSI amplitude streams (subcarrier-major, then
            antenna), then CSI phase streams; a radar input is stream 0.
        suppressed_harmonics_hz: Respiration harmonics removed from the
            heart band (empty when none were).
        respiration_waveform: Band-passed respiration waveform of the
            analysed segment (empty when none was formed).
        waveform_units: ``"m"`` for radar inputs, ``"a.u."`` for CSI.
    """

    respiration: RateEstimate
    heart: RateEstimate
    fs_analysis: float
    segment_start_s: float
    segment_end_s: float
    motion_fraction: float
    missing_fraction: float
    selected_streams: tuple[int, ...]
    suppressed_harmonics_hz: tuple[float, ...]
    respiration_waveform: FloatArray
    waveform_units: str


@dataclass(frozen=True, eq=False)
class VitalSignTrack:
    """Sliding-window rate series plus the full respiration waveform.

    Window ``i`` covers ``[window_start_s[i], window_end_s[i])`` on the same
    clock as ``sample_times_s``. Abstained windows hold ``nan`` rates and a
    non-``None`` reason.

    Attributes:
        fs_analysis: Sampling rate of the analysis grid (Hz).
        window_s: Window length (s).
        step_s: Window step (s).
        window_start_s: Window start times.
        window_end_s: Window end times (exclusive).
        window_center_s: Window centre times.
        respiration_bpm: Respiration rate per window (``nan`` = abstained).
        respiration_snr_db: Respiration in-band SNR per window.
        respiration_reasons: Abstention reason per window, or ``None``.
        heart_bpm: Heart rate per window (``nan`` = abstained).
        heart_snr_db: Heart in-band SNR per window.
        heart_reasons: Abstention reason per window, or ``None``.
        motion_fraction: Fraction of each window flagged as motion.
        sample_times_s: Time of every analysis sample.
        respiration_waveform: Band-passed respiration waveform over the
            whole recording; ``nan`` where samples are missing or motion.
        motion_mask: Samples flagged as motion (including the guard).
        missing_mask: Samples missing in the input.
        waveform_units: ``"m"`` for radar inputs, ``"a.u."`` for CSI.
    """

    fs_analysis: float
    window_s: float
    step_s: float
    window_start_s: FloatArray
    window_end_s: FloatArray
    window_center_s: FloatArray
    respiration_bpm: FloatArray
    respiration_snr_db: FloatArray
    respiration_reasons: tuple[AbstainReason | None, ...]
    heart_bpm: FloatArray
    heart_snr_db: FloatArray
    heart_reasons: tuple[AbstainReason | None, ...]
    motion_fraction: FloatArray
    sample_times_s: FloatArray
    respiration_waveform: FloatArray
    motion_mask: BoolArray
    missing_mask: BoolArray
    waveform_units: str


@dataclass(frozen=True, eq=False)
class WaveformExcerpt:
    """A stimulus-locked respiration waveform excerpt, or an abstention.

    Attributes:
        samples: Waveform samples, or ``None`` when abstaining.
        times_s: Sample times, or ``None`` when abstaining.
        fs: Sampling rate of the samples (Hz).
        start_s: Requested start time.
        end_s: Requested end time (exclusive).
        units: ``"m"`` for radar inputs, ``"a.u."`` for CSI.
        reason: Why the excerpt was withheld, or ``None``.
        detail: Human-readable explanation.
    """

    samples: FloatArray | None
    times_s: FloatArray | None
    fs: float
    start_s: float
    end_s: float
    units: str
    reason: AbstainReason | None
    detail: str


# ---------------------------------------------------------------------------
# Preprocessing primitives
# ---------------------------------------------------------------------------


def _real_array(x: ArrayLike, name: str) -> FloatArray:
    """Convert ``x`` to a float64 array, rejecting complex input."""
    if np.iscomplexobj(x):
        raise TypeError(f"{name} must be real; pass np.abs() or np.angle() of complex CSI")
    return np.asarray(x, dtype=np.float64)


def hampel_filter(
    x: ArrayLike, half_width: int = 3, n_sigmas: float = 3.0
) -> tuple[FloatArray, BoolArray]:
    """Replace outliers by the local median (Hampel identifier).

    A sample is an outlier when it differs from the median of the window
    ``[i - half_width, i + half_width]`` by more than
    ``n_sigmas * MAD_TO_SIGMA * MAD`` (Hampel 1974; Pearson 2002). Windows
    are reflected at the ends so that an end sample is judged against its
    neighbours. With ``n_sigmas = 0`` the filter is a running median.

    Args:
        x: Finite samples, shape ``(T,)`` or ``(T, K)``; filtered along
            axis 0.
        half_width: Window half-width in samples (0 returns a copy).
        n_sigmas: Threshold in robust standard deviations.

    Returns:
        The filtered array (same shape) and a mask of replaced samples.

    Raises:
        ValueError: On a bad shape, negative parameters or non-finite input.
    """
    arr = _real_array(x, "x")
    if arr.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {arr.shape}")
    if half_width < 0 or not n_sigmas >= 0.0:
        raise ValueError("half_width and n_sigmas must be non-negative")
    if not np.all(np.isfinite(arr)):
        raise ValueError("hampel_filter requires finite input")
    columns = arr.reshape(arr.shape[0], -1)
    out = columns.copy()
    flags = np.zeros(columns.shape, dtype=np.bool_)
    if half_width == 0 or columns.shape[0] < 2:
        return out.reshape(arr.shape), flags.reshape(arr.shape)
    width = 2 * half_width + 1
    for j in range(columns.shape[1]):
        col = columns[:, j]
        windows = sliding_window_view(np.pad(col, half_width, mode="reflect"), width)
        med = np.median(windows, axis=1)
        mad = np.median(np.abs(windows - med[:, None]), axis=1)
        bad = np.abs(col - med) > n_sigmas * MAD_TO_SIGMA * mad
        out[bad, j] = med[bad]
        flags[:, j] = bad
    return out.reshape(arr.shape), flags.reshape(arr.shape)


def sanitize_csi_phase(phase: ArrayLike, subcarrier_indices: ArrayLike | None = None) -> FloatArray:
    """Remove the per-packet linear phase term across subcarriers.

    The measured phase of subcarrier ``k`` contains ``2 pi k dt + beta``
    with a timing offset ``dt`` and phase offset ``beta`` that change from
    packet to packet (Sen et al. 2012, section 4.1). The phase is unwrapped
    across subcarriers and the least-squares line in (centred) subcarrier
    index is subtracted for every packet and antenna, which removes any
    per-packet ``a k + b`` term exactly.

    Args:
        phase: Phase in radians, shape ``(T, S)`` or ``(T, S, A)``.
        subcarrier_indices: Subcarrier indices, shape ``(S,)``, not all
            equal (default ``0 .. S-1``). Use the real indices for grouped
            subcarriers.

    Returns:
        Sanitised phase, same shape; rows with a non-finite value stay
        non-finite.

    Raises:
        ValueError: On a bad shape, fewer than three subcarriers, or
            degenerate indices.
    """
    arr = _real_array(phase, "phase")
    if arr.ndim not in (2, 3):
        raise ValueError(f"phase must have shape (T, S) or (T, S, A), got {arr.shape}")
    n_sub = arr.shape[1]
    if n_sub < 3:
        raise ValueError("phase sanitisation needs at least three subcarriers")
    if subcarrier_indices is None:
        index = np.arange(n_sub, dtype=np.float64)
    else:
        index = _real_array(subcarrier_indices, "subcarrier_indices")
        if index.shape != (n_sub,) or not np.all(np.isfinite(index)):
            raise ValueError("subcarrier_indices must be finite with shape (S,)")
    centred = index - index.mean()
    norm = float(np.sum(centred**2))
    if norm <= 0.0:
        raise ValueError("subcarrier_indices must not all be equal")
    unwrapped = np.unwrap(arr, axis=1)
    shape = [1] * arr.ndim
    shape[1] = n_sub
    kc = centred.reshape(shape)
    slope = np.sum(unwrapped * kc, axis=1, keepdims=True) / norm
    offset = np.mean(unwrapped, axis=1, keepdims=True)
    return np.asarray(unwrapped - slope * kc - offset, dtype=np.float64)


def _unwrap_time(x: FloatArray) -> FloatArray:
    """Unwrap along axis 0 using only rows that are entirely finite."""
    out = x.copy()
    rows = np.all(np.isfinite(out.reshape(out.shape[0], -1)), axis=1)
    if np.count_nonzero(rows) >= 2:
        out[rows] = np.unwrap(out[rows], axis=0)
    return out


def centre_wrapped_phase(phase: ArrayLike) -> FloatArray:
    """Wrap each stream of a phase series to ``(-pi, pi]`` about its circular mean.

    Unwrapping a noisy phase series along time lets each noise-induced
    2-pi cycle slip shift everything after it, which piles up low-frequency
    artefacts. A stationary subject moves the CSI phase of a static-dominated
    channel by much less than pi, so the series is instead re-wrapped about
    its circular mean, where no slip can accumulate. Rows holding a
    non-finite value are left non-finite and excluded from the mean.

    Args:
        phase: Phase in radians, shape ``(T, ...)``; axis 0 is time.

    Returns:
        Re-wrapped phase of the same shape.
    """
    arr = _real_array(phase, "phase")
    flat = arr.reshape(arr.shape[0], -1)
    rows = np.all(np.isfinite(flat), axis=1)
    out = np.full(flat.shape, np.nan)
    if np.any(rows):
        mean = np.angle(np.mean(np.exp(1j * flat[rows]), axis=0))
        out[rows] = np.angle(np.exp(1j * (flat[rows] - mean[None, :])))
    return out.reshape(arr.shape)


def antenna_phase_difference(phase: ArrayLike, reference_antenna: int = 0) -> FloatArray:
    """Phase difference between antennas of one receiver (PhaseBeat).

    Antennas of the same NIC share the oscillator and sampling clock, so the
    per-packet timing and phase offsets cancel in their difference (Wang,
    Yang and Mao 2017). Each difference is re-wrapped about its circular
    mean (``centre_wrapped_phase``) rather than unwrapped along time.

    Args:
        phase: Phase in radians, shape ``(T, S, A)`` with ``A >= 2``.
        reference_antenna: Antenna subtracted from all others.

    Returns:
        Phase differences, shape ``(T, S, A - 1)``.

    Raises:
        ValueError: On a bad shape or reference index.
    """
    arr = _real_array(phase, "phase")
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"phase must have shape (T, S, A) with A >= 2, got {arr.shape}")
    if not 0 <= reference_antenna < arr.shape[2]:
        raise ValueError("reference_antenna out of range")
    ref = arr[:, :, reference_antenna : reference_antenna + 1]
    others = np.delete(arr, reference_antenna, axis=2)
    return centre_wrapped_phase(others - ref)


def resample_uniform(
    timestamps_s: ArrayLike, values: ArrayLike, fs: float, max_gap_s: float = 0.25
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Linearly interpolate irregular packets onto a uniform grid.

    Rows of ``values`` containing a non-finite entry count as lost packets.
    Grid samples that fall in a gap between kept packets longer than
    ``max_gap_s`` are marked missing.

    Args:
        timestamps_s: Strictly increasing packet times, shape ``(T,)``.
        values: Samples, shape ``(T,)`` or ``(T, K)``.
        fs: Output sampling rate (Hz).
        max_gap_s: Longest gap bridged by interpolation (s).

    Returns:
        Grid times ``(N,)``, resampled values ``(N,)`` or ``(N, K)``, and
        the missing-sample mask ``(N,)``.

    Raises:
        ValueError: On mismatched shapes, unsorted times, or bad rates.
    """
    t = _real_array(timestamps_s, "timestamps_s")
    v = _real_array(values, "values")
    if t.ndim != 1 or t.shape[0] < 2 or v.shape[0] != t.shape[0] or v.ndim not in (1, 2):
        raise ValueError("timestamps_s must be (T,) with T >= 2 and values (T,) or (T, K)")
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0.0):
        raise ValueError("timestamps_s must be finite and strictly increasing")
    if not (math.isfinite(fs) and fs > 0.0 and max_gap_s > 0.0):
        raise ValueError("fs and max_gap_s must be positive")
    cols = v.reshape(v.shape[0], -1)
    keep = np.all(np.isfinite(cols), axis=1)
    n_out = math.floor((t[-1] - t[0]) * fs + 1e-9) + 1
    grid = t[0] + np.arange(n_out, dtype=np.float64) / fs
    out = np.full((n_out, cols.shape[1]), np.nan)
    missing = np.ones(n_out, dtype=np.bool_)
    if np.count_nonzero(keep) >= 2:
        tk = t[keep]
        for j in range(cols.shape[1]):
            out[:, j] = np.interp(grid, tk, cols[keep, j])
        right = np.clip(np.searchsorted(tk, grid, side="left"), 1, tk.shape[0] - 1)
        gap = tk[right] - tk[right - 1]
        exact = np.isclose(tk[right], grid, rtol=0.0, atol=1e-12)
        inside = (grid >= tk[0] - 1e-9) & (grid <= tk[-1] + 1e-9)
        missing = ~inside | ((gap > max_gap_s) & ~exact)
        out[missing] = np.nan
    shaped = out.reshape((n_out,) + v.shape[1:])
    return grid, shaped, missing


# ---------------------------------------------------------------------------
# Spectral primitives
# ---------------------------------------------------------------------------


def _as_columns(x: ArrayLike, name: str) -> FloatArray:
    arr = _real_array(x, name)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {arr.shape}")


def _check_band(band: tuple[float, float], fs: float) -> None:
    low, high = band
    if not (0.0 < low < high < fs / 2.0):
        raise ValueError(f"band {band} must satisfy 0 < low < high < fs/2 = {fs / 2.0}")


def _windowed_spectrum(x: FloatArray, fs: float) -> tuple[FloatArray, NDArray[np.complex128]]:
    """Hann-windowed DFT of each mean-removed column, without zero padding."""
    window = signal.get_window("hann", x.shape[0], fftbins=True)
    centred = x - x.mean(axis=0, keepdims=True)
    spectrum = np.asarray(np.fft.rfft(centred * window[:, None], axis=0), dtype=np.complex128)
    freqs = np.asarray(np.fft.rfftfreq(x.shape[0], d=1.0 / fs), dtype=np.float64)
    return freqs, spectrum


def _periodogram(x: FloatArray, fs: float, zero_pad_factor: int) -> tuple[FloatArray, FloatArray]:
    """Hann-windowed, zero-padded periodogram of each column."""
    n = x.shape[0]
    nfft = 1 << max(0, math.ceil(math.log2(max(1, n * zero_pad_factor))))
    window = signal.get_window("hann", n, fftbins=True)
    spectrum = np.fft.rfft(x * window[:, None], n=nfft, axis=0)
    power = np.asarray(np.abs(spectrum) ** 2, dtype=np.float64)
    freqs = np.asarray(np.fft.rfftfreq(nfft, d=1.0 / fs), dtype=np.float64)
    return freqs, power


def band_power_ratio(x: ArrayLike, fs: float, band: tuple[float, float]) -> FloatArray:
    """Fraction of each stream's (non-DC) power inside ``band``.

    Each stream is linearly detrended and its Hann-windowed periodogram
    computed; the ratio is the in-band power over the power of all non-zero
    frequency bins. A stream with no power scores 0.

    Args:
        x: Samples, shape ``(T,)`` or ``(T, K)``.
        fs: Sampling rate (Hz).
        band: ``(low, high)`` in Hz.

    Returns:
        Ratios in ``[0, 1]``, shape ``(K,)``.
    """
    cols = _as_columns(x, "x")
    _check_band(band, fs)
    detrended = signal.detrend(cols, axis=0, type="linear")
    freqs, power = _periodogram(np.asarray(detrended, dtype=np.float64), fs, 1)
    total = power[1:].sum(axis=0)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    inband = power[mask].sum(axis=0)
    ratio = np.zeros(cols.shape[1], dtype=np.float64)
    nz = total > 0.0
    ratio[nz] = inband[nz] / total[nz]
    return ratio


def first_principal_component(x: ArrayLike) -> FloatArray:
    """Scores of the first principal component of standardised columns.

    Columns are centred and scaled to unit variance (constant columns are
    dropped). The sign is fixed so that the largest-magnitude loading is
    positive; magnitudes are compared after rounding to 1e-9 and ties go to
    the lowest column, so standardised streams with equal loadings still
    give a deterministic sign.

    Args:
        x: Samples, shape ``(T, K)``.

    Returns:
        Component scores, shape ``(T,)`` (zeros when every column is
        constant).
    """
    cols = _as_columns(x, "x")
    centred = cols - cols.mean(axis=0, keepdims=True)
    std = centred.std(axis=0)
    keep = std > 0.0
    if not np.any(keep):
        return np.zeros(cols.shape[0], dtype=np.float64)
    z = centred[:, keep] / std[keep]
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    scores = u[:, 0] * s[0]
    if vt[0, int(np.argmax(np.round(np.abs(vt[0]), 9)))] < 0.0:
        scores = -scores
    return np.asarray(scores, dtype=np.float64)


def _mean_band_spectrum(
    cols: FloatArray, fs: float, band: tuple[float, float], zero_pad_factor: int
) -> tuple[FloatArray, FloatArray] | None:
    """In-band frequencies and mean periodogram of columns normalised to unit in-band power.

    Returns ``None`` when the segment is shorter than four samples, the band
    holds fewer than three bins, or no column has in-band power.
    """
    if cols.shape[0] < 4:
        return None
    freqs, power = _periodogram(cols, fs, zero_pad_factor)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f_band = freqs[mask]
    p_band = power[mask]
    col_power = p_band.sum(axis=0)
    live = col_power > 0.0
    if f_band.shape[0] < 3 or not np.any(live):
        return None
    return f_band, np.asarray(np.mean(p_band[:, live] / col_power[live], axis=1), dtype=np.float64)


def _maximum_on_upper_edge(
    x: FloatArray, fs: float, band: tuple[float, float], zero_pad_factor: int
) -> bool:
    """Whether the in-band maximum of ``dominant_frequency``'s spectrum is the top bin."""
    spectrum = _mean_band_spectrum(x, fs, band, zero_pad_factor)
    return spectrum is not None and int(np.argmax(spectrum[1])) == spectrum[1].shape[0] - 1


def effective_stream_count(
    x: ArrayLike,
    fs: float,
    band: tuple[float, float],
    *,
    exclude_hz: float | None = None,
    exclude_halfwidth_hz: float = 0.0,
) -> float:
    """Effective number of streams with independent in-band noise.

    The SNR null model of ``null_snr_threshold_db`` assumes the averaged
    streams carry independent noise. Noise common to the streams (shared
    oscillator or AGC fluctuations, interference seen by every subcarrier)
    makes their average no less variable than one stream, so the count of
    streams is replaced by the participation ratio of the eigenvalues
    ``lambda_i`` of the streams' in-band coherence matrix ``R``,
    ``(sum lambda_i)^2 / sum lambda_i^2 = K^2 / ||R||_F^2``: ``K`` for
    uncorrelated streams and 1 when all streams are proportional.

    ``R`` is formed from every second bin of the Hann-windowed DFT (no zero
    padding) inside ``band``, leaving out the bins within
    ``exclude_halfwidth_hz`` of ``exclude_hz``: the candidate signal is
    common to every stream by nature and would otherwise count against the
    noise it is tested against. The Hann window keeps the signal's leakage
    out of the remaining bins (sidelobes fall as the cube of the distance,
    Harris 1978), and Hann bins two apart are nearly uncorrelated for white
    noise, so the ``n`` bins used are close to independent. Each squared
    coherence ``r^2`` estimated from ``n`` bins is reduced by its
    finite-sample bias ``(1 - r^2)^2 / n``, which is ``1 / n`` for
    uncorrelated streams and vanishes for fully coherent ones.

    Args:
        x: Samples, shape ``(T,)`` or ``(T, K)`` (band-passed as analysed).
        fs: Sampling rate (Hz).
        band: ``(low, high)`` in Hz.
        exclude_hz: Centre of the signal region to leave out, or ``None``.
        exclude_halfwidth_hz: Half-width of that region (Hz).

    Returns:
        The participation ratio, clipped to ``[1, K]``; 1 when fewer than
        two streams have in-band power or fewer than two bins remain.
    """
    cols = _as_columns(x, "x")
    _check_band(band, fs)
    n_cols = cols.shape[1]
    if n_cols < 2 or cols.shape[0] < 4:
        return 1.0
    freqs, spectrum = _windowed_spectrum(cols, fs)
    keep = (freqs >= band[0]) & (freqs <= band[1])
    if exclude_hz is not None:
        keep &= np.abs(freqs - exclude_hz) > exclude_halfwidth_hz
    bins = spectrum[np.flatnonzero(keep)[::2]]
    power = np.sum(np.abs(bins) ** 2, axis=0)
    live = power > 0.0
    k = int(np.count_nonzero(live))
    n_bins = bins.shape[0]
    if k < 2 or n_bins < 2:
        return 1.0
    unit = bins[:, live] / np.sqrt(power[live])
    coherence = np.abs(unit.conj().T @ unit) ** 2
    off = coherence[~np.eye(k, dtype=np.bool_)]
    off = np.maximum(off - (1.0 - off) ** 2 / n_bins, 0.0)
    ratio = k * k / (k + float(np.sum(off)))
    return float(min(max(ratio, 1.0), float(n_cols)))


def dominant_frequency(
    x: ArrayLike,
    fs: float,
    band: tuple[float, float],
    *,
    signal_halfwidth_hz: float,
    zero_pad_factor: int = 8,
) -> SpectralPeak:
    """Locate the dominant in-band peak of the mean normalised periodogram.

    Each column's Hann-windowed, zero-padded periodogram is normalised to
    unit power inside ``band`` and the columns are averaged. The peak is
    refined by parabolic interpolation of the log spectrum (Smith and Serra
    1987). Signal power is the power within ``halfwidth`` of the peak, where
    ``halfwidth = max(signal_halfwidth_hz, 2 / duration)`` is never narrower
    than the Hann main lobe (Harris 1978); the SNR is that power over the
    rest of the band.

    Args:
        x: Samples, shape ``(T,)`` or ``(T, K)``; analysed as given (filter
            and detrend beforehand).
        fs: Sampling rate (Hz).
        band: ``(low, high)`` in Hz.
        signal_halfwidth_hz: Requested signal half-width (Hz).
        zero_pad_factor: Periodogram length as a multiple of ``T``.

    Returns:
        The peak description; ``frequency_hz`` is ``None`` when the band
        holds no power or its maximum is on a band edge.
    """
    cols = _as_columns(x, "x")
    _check_band(band, fs)
    n = cols.shape[0]
    halfwidth = max(float(signal_halfwidth_hz), 2.0 * fs / max(n, 1))
    spectrum = _mean_band_spectrum(cols, fs, band, zero_pad_factor)
    if spectrum is None:
        return SpectralPeak(None, math.nan, False, halfwidth)
    f_band, p = spectrum
    i = int(np.argmax(p))
    if i in (0, p.shape[0] - 1):
        return SpectralPeak(None, math.nan, True, halfwidth)
    alpha, beta, gamma = np.log(p[i - 1 : i + 2] + _EPS)
    denom = alpha - 2.0 * beta + gamma
    delta = 0.5 * (alpha - gamma) / denom if denom < 0.0 else 0.0
    df = float(f_band[1] - f_band[0])
    f_peak = float(f_band[i] + delta * df)
    total = float(p.sum())
    main_region = np.abs(f_band - f_peak) <= halfwidth
    main_power = float(p[main_region].sum())
    snr = _ratio_db(main_power, total - main_power)
    secondary_hz: float | None = None
    secondary_ratio = 0.0
    joint = math.nan
    peaks, _ = signal.find_peaks(p)
    for j in peaks:
        fj = float(f_band[j])
        resolvable = abs(fj - f_peak) > halfwidth
        harmonic = abs(fj - 2.0 * f_peak) <= halfwidth or abs(fj - 0.5 * f_peak) <= halfwidth
        if not resolvable or harmonic:
            continue
        region_j = np.abs(f_band - fj) <= halfwidth
        power_j = float(p[region_j & ~main_region].sum())
        ratio = power_j / main_power if main_power > 0.0 else 0.0
        if ratio > secondary_ratio:
            secondary_ratio = ratio
            secondary_hz = fj
            both = float(p[main_region | region_j].sum())
            joint = _ratio_db(both, total - both)
    return SpectralPeak(f_peak, snr, False, halfwidth, secondary_hz, secondary_ratio, joint)


def null_snr_threshold_db(
    n_streams: float,
    n_samples: int,
    fs: float,
    band: tuple[float, float],
    signal_halfwidth_hz: float,
    *,
    false_alarm: float = 0.01,
    dof_scale: float = 0.45,
    colour_margin_db: float = 2.6,
) -> float:
    """SNR threshold that noise alone exceeds with probability ``false_alarm``.

    The statistic of ``dominant_frequency`` compares the fraction ``s`` of
    in-band power inside the signal region with the rest. With ``M = B T``
    frequency cells in a band of width ``B`` over ``T`` seconds and
    ``m = 2 h T`` cells in the region, ``s`` for one region placement is
    modelled under Gaussian noise averaged over ``K`` streams as
    ``Beta(c m K, c (M - m) K)``, where ``c`` (``dof_scale``) absorbs the
    Hann-window correlation between cells. The peak is the best of about
    ``M / m`` placements, so the per-placement tail probability is
    ``false_alarm * m / M`` (Bonferroni). ``c`` was fitted by Monte Carlo
    to white Gaussian noise; ``colour_margin_db`` is added so that 1/f
    noise, whose in-band power is not flat, is covered too (see
    ``VitalSignConfig``).

    Args:
        n_streams: Number of averaged streams ``K``; may be fractional,
            e.g. the effective count of ``effective_stream_count`` for
            streams whose noise is correlated.
        n_samples: Segment length in samples.
        fs: Sampling rate (Hz).
        band: ``(low, high)`` in Hz.
        signal_halfwidth_hz: Requested signal half-width; widened to the
            Hann main lobe ``2 / T`` as in ``dominant_frequency``.
        false_alarm: Target false-acceptance probability.
        dof_scale: Degrees-of-freedom scale ``c``.
        colour_margin_db: Margin for non-white noise (dB).

    Returns:
        The threshold in dB; ``inf`` when the signal region covers the
        whole band, i.e. when the segment is too short for the test.

    Raises:
        ValueError: On non-positive sizes or a false-alarm outside (0, 1).
    """
    if n_streams < 1 or n_samples < 1 or not fs > 0.0 or not 0.0 < false_alarm < 1.0:
        raise ValueError("need n_streams, n_samples, fs > 0 and 0 < false_alarm < 1")
    duration = n_samples / fs
    halfwidth = max(float(signal_halfwidth_hz), 2.0 / duration)
    cells_band = (band[1] - band[0]) * duration
    cells_signal = 2.0 * halfwidth * duration
    if cells_signal >= cells_band:
        return math.inf
    a = dof_scale * cells_signal * n_streams
    b = dof_scale * (cells_band - cells_signal) * n_streams
    tail = false_alarm * cells_signal / cells_band
    s = float(special.betainccinv(a, b, tail))
    return 10.0 * math.log10(s / (1.0 - s)) + colour_margin_db


def _snr_threshold(
    fixed: float | None,
    n_streams: float,
    n_samples: int,
    fs: float,
    band: tuple[float, float],
    halfwidth_hz: float,
    cfg: VitalSignConfig,
) -> float:
    if fixed is not None:
        return fixed
    return null_snr_threshold_db(
        n_streams,
        n_samples,
        fs,
        band,
        halfwidth_hz,
        false_alarm=cfg.false_alarm,
        dof_scale=cfg.null_dof_scale,
        colour_margin_db=cfg.null_colour_margin_db,
    )


def _ratio_db(num: float, den: float) -> float:
    if den <= 0.0:
        return math.inf if num > 0.0 else math.nan
    if num <= 0.0:
        return -math.inf
    return 10.0 * math.log10(num / den)


def count_cycle_rate(
    waveform: ArrayLike, fs: float, max_rate_hz: float, prominence_fraction: float = 0.25
) -> float | None:
    """Rate from counting waveform peaks (per minute).

    Peaks must be at least ``1 / max_rate_hz`` apart (the minimum-distance
    rule of Liu et al. 2018, section IV-C) and have a prominence of at least
    ``prominence_fraction`` of the 5th-to-95th percentile range. The rate is
    ``60 * (n - 1) / (t_last - t_first)``.

    Args:
        waveform: Band-passed samples, shape ``(T,)``.
        fs: Sampling rate (Hz).
        max_rate_hz: Highest admissible cycle frequency (Hz).
        prominence_fraction: Minimum relative prominence.

    Returns:
        Cycles per minute, or ``None`` when fewer than three peaks are found
        or the waveform is not finite.
    """
    w = _real_array(waveform, "waveform")
    if w.ndim != 1 or not np.all(np.isfinite(w)) or w.shape[0] < 3:
        return None
    if not (fs > 0.0 and max_rate_hz > 0.0):
        raise ValueError("fs and max_rate_hz must be positive")
    spread = float(np.percentile(w, 95.0) - np.percentile(w, 5.0))
    if spread <= 0.0:
        return None
    distance = max(1, math.floor(fs / max_rate_hz))
    peaks, _ = signal.find_peaks(w, distance=distance, prominence=prominence_fraction * spread)
    if peaks.shape[0] < 3:
        return None
    return float(60.0 * (peaks.shape[0] - 1) * fs / (peaks[-1] - peaks[0]))


def respiration_harmonics(
    respiration_hz: float, band: tuple[float, float], max_order: int, margin_hz: float = 0.0
) -> tuple[float, ...]:
    """Harmonics ``k * f`` (``2 <= k <= max_order``) inside a widened band.

    Args:
        respiration_hz: Respiration fundamental (Hz).
        band: Heart band ``(low, high)`` in Hz.
        max_order: Highest harmonic order considered.
        margin_hz: Widening of the band on each side.

    Returns:
        The harmonic frequencies in increasing order.
    """
    if not (math.isfinite(respiration_hz) and respiration_hz > 0.0):
        return ()
    low, high = band[0] - margin_hz, band[1] + margin_hz
    return tuple(
        k * respiration_hz for k in range(2, max_order + 1) if low <= k * respiration_hz <= high
    )


def suppress_harmonics(
    x: ArrayLike, fs: float, harmonics_hz: tuple[float, ...], notch_width_hz: float
) -> FloatArray:
    """Apply a zero-phase cascade of IIR notches (Orfanidis 1996).

    Args:
        x: Samples, shape ``(T,)`` or ``(T, K)``; filtered along axis 0.
        fs: Sampling rate (Hz).
        harmonics_hz: Notch centre frequencies (Hz); those outside
            ``(0, fs/2)`` are skipped.
        notch_width_hz: -3 dB width of each notch (Hz).

    Returns:
        The filtered samples, same shape.
    """
    out = _real_array(x, "x").copy()
    for f0 in harmonics_hz:
        if 0.0 < f0 < fs / 2.0:
            b, a = signal.iirnotch(f0, f0 / notch_width_hz, fs=fs)
            out = np.asarray(signal.filtfilt(b, a, out, axis=0), dtype=np.float64)
    return out


def respiration_phase(waveform: ArrayLike, fs: float, respiration_hz: float) -> FloatArray:
    """Unwrapped instantaneous respiration phase (rad).

    The waveform is narrowed to ``[0.5, 1.5] * respiration_hz`` with a
    second-order zero-phase Butterworth band-pass, which keeps the
    fundamental and rejects its second harmonic, and the phase is the angle
    of its analytic signal (Hilbert transform).

    Args:
        waveform: Respiration waveform, shape ``(T,)``; ``T`` must exceed
            the filter's padding (more than 15 samples).
        fs: Sampling rate (Hz).
        respiration_hz: Approximate respiration fundamental (Hz).

    Returns:
        The unwrapped phase, shape ``(T,)``.
    """
    wave = _real_array(waveform, "waveform")
    narrow = (0.5 * respiration_hz, min(1.5 * respiration_hz, 0.95 * fs / 2.0))
    fundamental = _bandpass(wave[:, None], fs, narrow, 2)[:, 0]
    return np.asarray(np.unwrap(np.angle(signal.hilbert(fundamental))), dtype=np.float64)


def cancel_respiration_harmonics(
    x: ArrayLike,
    respiration_waveform: ArrayLike,
    fs: float,
    respiration_hz: float,
    orders: tuple[int, ...],
    band: tuple[float, float],
    *,
    poly_degree: int = 2,
    filter_order: int = 4,
) -> FloatArray:
    """Remove components phase-locked to respiration harmonics.

    The respiration phase ``phi(t)`` is the angle of the analytic signal of
    the waveform narrowed to ``[0.5, 1.5] * respiration_hz``. References
    ``P_j(t) cos(k phi)`` and ``P_j(t) sin(k phi)`` for every harmonic order
    ``k`` and Legendre polynomials ``P_j`` up to ``poly_degree`` (slow
    changes of breathing depth) are band-passed like ``x`` and fitted to it
    by least squares; the fit is subtracted. This is a batch least-squares
    form of the adaptive noise canceller with periodic references (Widrow et
    al. 1975); because the references follow the measured phase, they track
    breath-to-breath rate variability that a fixed notch would miss.
    Components not phase-locked to breathing, such as the heartbeat, are
    left in place except for their chance correlation with the references.

    Args:
        x: Heart-band samples, shape ``(T,)`` or ``(T, K)``.
        respiration_waveform: Respiration waveform, shape ``(T,)``.
        fs: Sampling rate (Hz).
        respiration_hz: Respiration fundamental (Hz).
        orders: Harmonic orders to cancel (each at least 2).
        band: Pass band applied to the references (that of ``x``).
        poly_degree: Highest Legendre degree of the amplitude modulation.
        filter_order: Butterworth prototype order of the band-passes.

    Returns:
        ``x`` with the fitted harmonic components removed (same shape).

    Raises:
        ValueError: On mismatched lengths or a non-positive fundamental.
    """
    data = _real_array(x, "x")
    cols = _as_columns(data, "x")
    wave = _real_array(respiration_waveform, "respiration_waveform")
    if wave.shape != (cols.shape[0],):
        raise ValueError("respiration_waveform must have shape (T,) matching x")
    if not (math.isfinite(respiration_hz) and respiration_hz > 0.0):
        raise ValueError("respiration_hz must be positive")
    if not orders or cols.shape[0] <= 3 * (2 * filter_order + 1):
        return data.copy()
    phase = respiration_phase(wave, fs, respiration_hz)
    t = np.linspace(-1.0, 1.0, cols.shape[0])
    poly = np.polynomial.legendre.legvander(t, poly_degree)
    refs = [poly * fn(k * phase)[:, None] for k in orders for fn in (np.cos, np.sin)]
    design = _bandpass(np.concatenate(refs, axis=1), fs, band, filter_order)
    coef, *_ = np.linalg.lstsq(design, cols, rcond=None)
    cleaned = cols - design @ coef
    return cleaned.reshape(data.shape)


def detect_motion(
    streams: ArrayLike,
    fs: float,
    *,
    frame_s: float = 1.0,
    threshold: float = 4.0,
    guard_s: float = 1.0,
    max_excursion: float | None = None,
) -> BoolArray:
    """Flag body-motion samples from short-time statistics.

    Each stream is linearly detrended, scaled by its median frame standard
    deviation and cut into frames of ``frame_s``. For every frame the
    coherent short-time energy is the largest eigenvalue of the frame's
    stream covariance (for one stream, its variance), computed for the
    samples and for their first differences; the frame score is the larger
    of the two square-rooted energies relative to their median over frames.
    Frames scoring above ``threshold`` are motion (short-time energy of the
    moving variance across subcarriers, Liu et al. 2018, section VI-A; the
    largest eigenvalue replaces their sum over subcarriers so that motion,
    which moves every subcarrier together, gains over independent noise).
    A stream that is constant in most frames flags every frame in which it
    varies. If ``max_excursion`` is given (displacement inputs in metres), a
    frame of stream 0 whose peak-to-peak value exceeds it is also motion.
    Flags are widened by ``guard_s`` on both sides.

    The test is relative to the recording's own median frame, so a subject
    moving during more than half of the recording is not detected by it;
    such recordings are caught only by the downstream SNR, band-edge,
    multi-source and peak-count checks.

    Args:
        streams: Finite samples, shape ``(T,)`` or ``(T, K)``.
        fs: Sampling rate (Hz).
        frame_s: Frame length (s).
        threshold: Relative frame-score threshold.
        guard_s: Guard interval (s).
        max_excursion: Absolute peak-to-peak limit per frame, or ``None``.

    Returns:
        Motion mask, shape ``(T,)``.
    """
    cols = _as_columns(streams, "streams")
    n = cols.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.bool_)
    if not np.all(np.isfinite(cols)):
        raise ValueError("detect_motion requires finite input")
    x = np.asarray(signal.detrend(cols, axis=0, type="linear"), dtype=np.float64)
    frame = max(2, round(frame_s * fs))
    n_frames = max(1, n // frame)
    starts = np.arange(n_frames) * frame
    ends = np.append(starts[1:], n)
    stds = np.empty((n_frames, cols.shape[1]))
    rms_diff = np.empty((n_frames, cols.shape[1]))
    excursion = np.empty(n_frames)
    for m, (lo, hi) in enumerate(zip(starts, ends)):
        seg = x[lo:hi]
        stds[m] = seg.std(axis=0)
        rms_diff[m] = np.sqrt(np.mean(np.diff(seg, axis=0) ** 2, axis=0))
        excursion[m] = float(np.ptp(seg[:, 0]))
    # detrending a constant leaves float residue; treat it as exactly flat
    tol = 1e-10 * (np.max(np.abs(cols), axis=0) + _EPS)
    stds = np.where(stds > tol, stds, 0.0)
    rms_diff = np.where(rms_diff > tol, rms_diff, 0.0)
    scale = np.median(stds, axis=0)
    live = scale > 0.0
    flagged = np.any(stds[:, ~live] > 0.0, axis=1)
    if np.any(live):
        diff_scale = np.median(rms_diff[:, live], axis=0)
        z = x[:, live] / scale[live]
        dz = np.diff(x[:, live], axis=0) / np.where(diff_scale > 0.0, diff_scale, 1.0)
        coherent = np.empty((n_frames, 2))
        for m, (lo, hi) in enumerate(zip(starts, ends)):
            seg = z[lo:hi] - z[lo:hi].mean(axis=0)
            dseg = dz[lo : max(lo + 1, hi - 1)]
            coherent[m, 0] = np.linalg.norm(seg, ord=2) / math.sqrt(seg.shape[0])
            coherent[m, 1] = np.linalg.norm(dseg, ord=2) / math.sqrt(dseg.shape[0])
        flagged |= np.max(_relative(coherent), axis=1) > threshold
    if max_excursion is not None:
        flagged |= excursion > max_excursion
    mask = np.zeros(n, dtype=np.bool_)
    for frame_index in np.flatnonzero(flagged):
        mask[starts[frame_index] : ends[frame_index]] = True
    return _dilate(mask, round(guard_s * fs))


def _relative(values: FloatArray) -> FloatArray:
    """Divide each column by its median; zero medians map >0 to inf, 0 to 0."""
    med = np.median(values, axis=0)
    out = np.zeros_like(values)
    pos = med > 0.0
    out[:, pos] = values[:, pos] / med[pos]
    zero_cols = ~pos
    out[:, zero_cols] = np.where(values[:, zero_cols] > 0.0, np.inf, 0.0)
    return out


def _dilate(mask: BoolArray, radius: int) -> BoolArray:
    if radius <= 0 or not np.any(mask):
        return mask.copy()
    kernel = np.ones(2 * radius + 1)
    return np.asarray(np.convolve(mask.astype(np.float64), kernel, mode="same") > 0.5)


def _longest_run(valid: BoolArray) -> tuple[int, int]:
    """Return ``(start, end)`` of the longest run of True (end exclusive)."""
    if valid.shape[0] == 0 or not np.any(valid):
        return 0, 0
    padded = np.concatenate(([False], valid, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]
    k = int(np.argmax(ends - starts))
    return int(starts[k]), int(ends[k])


def _long_runs(mask: BoolArray, max_len: int) -> BoolArray:
    """Keep only the True runs of ``mask`` longer than ``max_len`` samples."""
    if not np.any(mask):
        return mask.copy()
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    out = np.zeros_like(mask)
    for start, end in zip(edges[0::2], edges[1::2]):
        if end - start > max_len:
            out[start:end] = True
    return out


def _inpaint(x: FloatArray, invalid: BoolArray) -> FloatArray:
    """Linearly interpolate invalid rows from valid neighbours (per column)."""
    out = x.copy()
    good = ~invalid
    if not np.any(invalid) or np.count_nonzero(good) < 1:
        return out
    idx = np.arange(x.shape[0])
    for j in range(x.shape[1]):
        out[invalid, j] = np.interp(idx[invalid], idx[good], x[good, j])
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Prepared:
    streams: FloatArray
    stream_ids: IntArray
    missing: BoolArray  # lost samples in runs longer than max_gap_s
    lost: BoolArray  # every lost sample, including short bridged runs
    motion: BoolArray
    fs: float
    t0: float
    units: str


def _require_consent(consent_confirmed: bool | ConsentRecord) -> None:
    """Refuse unless consent is the literal ``True`` or a live record with the right scope.

    A record whose ``withdrawn_at`` is set never authorises sensing again,
    the same rule ``truth_decipher.require_consent`` applies to the CIT. The
    subject and time checks of that function need the caller's subject id and
    recording time, which these entry points do not take; callers that have
    them (the API, CLI and MCP surfaces) run ``require_consent`` first.

    Raises:
        ConsentRequiredError: The ``truth_decipher`` class, shared by every
            consent-gated entry point of the package.
    """
    if consent_confirmed is True:
        return
    if isinstance(consent_confirmed, ConsentRecord):
        if consent_confirmed.withdrawn_at is not None:
            raise ConsentRequiredError(
                "the subject has withdrawn consent to contactless vital-sign sensing"
            )
        if VITAL_SIGNS_CONSENT_SCOPE in consent_confirmed.scope:
            return
        raise ConsentRequiredError(
            "contactless vital-sign sensing requires the subject's informed consent; "
            f"the consent record's scope does not include {VITAL_SIGNS_CONSENT_SCOPE!r}"
        )
    raise ConsentRequiredError(
        "contactless vital-sign sensing requires the subject's informed consent; "
        "pass consent_confirmed=True (or a ConsentRecord whose scope includes "
        f"{VITAL_SIGNS_CONSENT_SCOPE!r}) only after it has been obtained"
    )


def _build_streams(
    csi_amplitude: ArrayLike | None,
    csi_phase: ArrayLike | None,
    subcarrier_indices: ArrayLike | None,
    radar_displacement_m: ArrayLike | None,
    radar_phase_rad: ArrayLike | None,
    wavelength_m: float | None,
) -> tuple[FloatArray, str]:
    csi_given = csi_amplitude is not None or csi_phase is not None
    n_given = int(csi_given) + int(radar_displacement_m is not None)
    n_given += int(radar_phase_rad is not None)
    if n_given != 1:
        raise ValueError(
            "pass exactly one modality: CSI (amplitude and/or phase), "
            "radar_displacement_m, or radar_phase_rad"
        )
    if radar_displacement_m is not None:
        disp = _real_array(radar_displacement_m, "radar_displacement_m")
        if disp.ndim != 1:
            raise ValueError(f"radar_displacement_m must have shape (T,), got {disp.shape}")
        return disp[:, None], "m"
    if radar_phase_rad is not None:
        if wavelength_m is None or not (math.isfinite(wavelength_m) and wavelength_m > 0.0):
            raise ValueError("radar_phase_rad needs a positive wavelength_m")
        phase = _real_array(radar_phase_rad, "radar_phase_rad")
        if phase.ndim != 1:
            raise ValueError(f"radar_phase_rad must have shape (T,), got {phase.shape}")
        disp = _unwrap_time(phase) * wavelength_m / (4.0 * math.pi)
        return disp[:, None], "m"
    parts: list[FloatArray] = []
    n_t: int | None = None
    if csi_amplitude is not None:
        amp = _real_array(csi_amplitude, "csi_amplitude")
        if amp.ndim not in (2, 3):
            raise ValueError(f"csi_amplitude must be (T, S) or (T, S, A), got {amp.shape}")
        n_t = amp.shape[0]
        parts.append(amp.reshape(n_t, -1))
    if csi_phase is not None:
        ph = _real_array(csi_phase, "csi_phase")
        if ph.ndim not in (2, 3):
            raise ValueError(f"csi_phase must be (T, S) or (T, S, A), got {ph.shape}")
        if n_t is not None and ph.shape[0] != n_t:
            raise ValueError("csi_amplitude and csi_phase must have the same length")
        if ph.ndim == 3 and ph.shape[2] >= 2:
            clean = antenna_phase_difference(ph)
        else:
            clean = centre_wrapped_phase(sanitize_csi_phase(ph, subcarrier_indices))
        parts.append(clean.reshape(ph.shape[0], -1))
    return np.asarray(np.concatenate(parts, axis=1), dtype=np.float64), "a.u."


def _decimate(
    streams: FloatArray, masks: tuple[BoolArray, ...], fs: float, target: float | None
) -> tuple[FloatArray, tuple[BoolArray, ...], float]:
    """Resample to ``target`` Hz; a mask is set wherever the FIR touched a set sample."""
    if target is None or fs <= target * (1.0 + 1e-9) or streams.shape[0] < 2:
        return streams, masks, fs
    ratio = Fraction(target / fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    if up == 0:
        up, down = 1, max(1, round(fs / target))
    out = np.asarray(signal.resample_poly(streams, up, down, axis=0), dtype=np.float64)
    half = 10.0 * max(up, down) / up  # default resample_poly FIR half-length, input samples
    n_in = streams.shape[0]
    centres = np.arange(out.shape[0]) * (down / up)
    lo = np.clip(np.floor(centres - half).astype(np.intp), 0, n_in)
    hi = np.clip(np.ceil(centres + half).astype(np.intp) + 1, 0, n_in)
    mapped = []
    for mask in masks:
        csum = np.concatenate(([0], np.cumsum(mask.astype(np.intp))))
        mapped.append(np.asarray((csum[hi] - csum[lo]) > 0))
    return out, tuple(mapped), fs * up / down


def _interpolation_rate(times: FloatArray, target: float) -> float:
    """Rate of the interpolation grid for packets at ``times`` and output rate ``target``.

    The smallest integer multiple of ``target`` that is at least the median
    packet rate (``target`` itself when packets are slower), so that the
    later decimation is by an exact integer factor. The grid is capped at
    four samples per packet on average, which only binds for bursty packet
    trains whose median spacing is far below their mean spacing.
    """
    steps = np.diff(times)
    if steps.shape[0] == 0 or not np.all(np.isfinite(steps)) or not np.all(steps > 0.0):
        return target  # resample_uniform reports the malformed timestamps
    median_step = float(np.median(steps))
    factor = max(1, math.ceil(1.0 / (median_step * target) - 1e-9))
    duration = float(times[-1] - times[0])
    cap = max(1, math.floor(4.0 * times.shape[0] / (duration * target)))
    return target * min(factor, cap)


def _prepare(
    *,
    fs: float,
    csi_amplitude: ArrayLike | None,
    csi_phase: ArrayLike | None,
    subcarrier_indices: ArrayLike | None,
    radar_displacement_m: ArrayLike | None,
    radar_phase_rad: ArrayLike | None,
    wavelength_m: float | None,
    timestamps_s: ArrayLike | None,
    cfg: VitalSignConfig,
) -> _Prepared:
    if isinstance(fs, bool) or not isinstance(fs, int | float | np.integer | np.floating):
        raise ValueError("fs must be a positive finite number")
    fs = float(fs)
    if not (math.isfinite(fs) and fs > 0.0):
        raise ValueError("fs must be a positive finite number")
    streams, units = _build_streams(
        csi_amplitude,
        csi_phase,
        subcarrier_indices,
        radar_displacement_m,
        radar_phase_rad,
        wavelength_m,
    )
    t0 = 0.0
    gap_missing: BoolArray | None = None
    target = cfg.analysis_fs
    if timestamps_s is not None:
        times = _real_array(timestamps_s, "timestamps_s")
        if times.shape != (streams.shape[0],):
            raise ValueError("timestamps_s must have shape (T,) matching the input")
        # fs is the uniform output rate (lowered to analysis_fs when that is
        # smaller). Interpolating straight onto it would fold every component
        # above fs / 2 into the analysis bands, so the grid runs at least at
        # the median packet rate, as an integer multiple of the output rate,
        # and the polyphase FIR of _decimate band-limits it on the way down.
        target = fs if cfg.analysis_fs is None else min(fs, cfg.analysis_fs)
        fs = _interpolation_rate(times, target)
        grid, streams, gap_missing = resample_uniform(times, streams, fs, cfg.max_gap_s)
        t0 = float(grid[0])
    finite = np.isfinite(streams)
    usable = np.any(finite, axis=0)
    stream_ids = np.flatnonzero(usable).astype(np.intp)
    streams = streams[:, usable]
    lost = ~np.all(np.isfinite(streams), axis=1)
    if gap_missing is not None:
        lost |= gap_missing
    # runs of lost samples no longer than max_gap_s are bridged by interpolation
    missing = _long_runs(lost, math.floor(cfg.max_gap_s * fs + 1e-9))
    if streams.shape[1] == 0 or np.count_nonzero(~lost) < 2:
        empty = np.zeros((streams.shape[0], streams.shape[1]))
        everything = np.ones(streams.shape[0], dtype=np.bool_)
        return _Prepared(
            empty, stream_ids, everything, everything.copy(), everything.copy(), fs, t0, units
        )
    streams = _inpaint(np.where(np.isfinite(streams), streams, 0.0), lost)
    streams, _ = hampel_filter(streams, cfg.hampel_half_width, cfg.hampel_n_sigmas)
    streams, (missing, lost), fs_a = _decimate(streams, (missing, lost), fs, target)
    for band in (cfg.respiration_band_hz, cfg.heart_band_hz):
        _check_band(band, fs_a)
    excursion = cfg.max_radar_excursion_m if units == "m" else None
    motion = detect_motion(
        streams,
        fs_a,
        frame_s=cfg.motion_frame_s,
        threshold=cfg.motion_threshold,
        guard_s=cfg.motion_guard_s,
        max_excursion=excursion,
    )
    return _Prepared(streams, stream_ids, missing, lost, motion & ~missing, fs_a, t0, units)


def _passes(snr_db: float, threshold_db: float) -> bool:
    """SNR test; an infinite threshold (segment too short to test) never passes."""
    return threshold_db != math.inf and snr_db >= threshold_db


def _abstain(reason: AbstainReason, detail: str) -> RateEstimate:
    return RateEstimate(None, reason, detail)


def _select(x: FloatArray, fs: float, cfg: VitalSignConfig) -> IntArray:
    if cfg.selection == "variance":
        scores = x.var(axis=0)
    else:
        scores = band_power_ratio(x, fs, cfg.respiration_band_hz)
    order = np.argsort(-scores, kind="stable")
    order = order[scores[order] > 0.0]
    return np.asarray(order[: cfg.n_streams], dtype=np.intp)


def _bandpass(x: FloatArray, fs: float, band: tuple[float, float], order: int) -> FloatArray:
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    return np.asarray(signal.sosfiltfilt(sos, x, axis=0), dtype=np.float64)


def _combine(x: FloatArray) -> FloatArray:
    if x.shape[1] == 1:
        return np.asarray(x[:, 0], dtype=np.float64)
    return first_principal_component(x)


def _respiration_estimate(
    peak: SpectralPeak,
    waveform: FloatArray,
    fs: float,
    threshold: float,
    joint_threshold: float,
    cfg: VitalSignConfig,
) -> RateEstimate:
    if peak.frequency_hz is None:
        return _abstain(
            AbstainReason.PEAK_AT_BAND_EDGE,
            "respiration-band maximum lies on the band edge (no interior peak)",
        )
    f_r = peak.frequency_hz
    multi = (
        peak.secondary_hz is not None
        and peak.secondary_power_ratio >= cfg.multi_source_power_ratio
        and (_passes(peak.snr_db, threshold) or _passes(peak.joint_snr_db, joint_threshold))
    )
    if multi:
        return RateEstimate(
            None,
            AbstainReason.MULTIPLE_SOURCES,
            f"second respiration-band peak at {peak.secondary_hz:.3f} Hz holds "
            f"{peak.secondary_power_ratio:.2f} of the main peak's power at {f_r:.3f} Hz",
            f_r,
            peak.snr_db,
            snr_threshold_db=threshold,
        )
    if not _passes(peak.snr_db, threshold):
        return RateEstimate(
            None,
            AbstainReason.LOW_SNR,
            f"respiration SNR {peak.snr_db:.1f} dB fails threshold {threshold:.1f} dB",
            f_r,
            peak.snr_db,
            snr_threshold_db=threshold,
        )
    spectral_bpm = 60.0 * f_r
    counted = count_cycle_rate(
        waveform, fs, cfg.respiration_band_hz[1], cfg.count_prominence_fraction
    )
    if counted is None or abs(counted - spectral_bpm) > cfg.count_tolerance_bpm:
        shown = "none" if counted is None else f"{counted:.1f}"
        return RateEstimate(
            None,
            AbstainReason.COUNT_DISAGREEMENT,
            f"spectral {spectral_bpm:.1f}/min vs peak count {shown}/min "
            f"(tolerance {cfg.count_tolerance_bpm:.1f})",
            f_r,
            peak.snr_db,
            counted,
            threshold,
        )
    return RateEstimate(spectral_bpm, None, "ok", f_r, peak.snr_db, counted, threshold)


def _heart_estimate(
    peak: SpectralPeak,
    threshold: float,
    respiration_hz: float | None,
    duration_s: float,
    cfg: VitalSignConfig,
) -> RateEstimate:
    if peak.frequency_hz is None:
        return _abstain(
            AbstainReason.PEAK_AT_BAND_EDGE,
            "heart-band maximum lies on the band edge (no interior peak)",
        )
    if respiration_hz is not None and cfg.harmonic_coincidence_fraction > 0.0:
        order = round(peak.frequency_hz / respiration_hz)
        precision = cfg.harmonic_coincidence_fraction + order * cfg.respiration_precision_bins
        tolerance = precision / duration_s
        if order >= 2 and abs(peak.frequency_hz - order * respiration_hz) <= tolerance:
            return RateEstimate(
                None,
                AbstainReason.HARMONIC_AMBIGUITY,
                f"heart peak {peak.frequency_hz:.3f} Hz coincides with respiration "
                f"harmonic {order} x {respiration_hz:.3f} Hz",
                peak.frequency_hz,
                peak.snr_db,
                snr_threshold_db=threshold,
            )
    if not _passes(peak.snr_db, threshold):
        return RateEstimate(
            None,
            AbstainReason.LOW_SNR,
            f"heart SNR {peak.snr_db:.1f} dB fails threshold {threshold:.1f} dB",
            peak.frequency_hz,
            peak.snr_db,
            snr_threshold_db=threshold,
        )
    return RateEstimate(
        60.0 * peak.frequency_hz,
        None,
        "ok",
        peak.frequency_hz,
        peak.snr_db,
        snr_threshold_db=threshold,
    )


def _harmonic_reference(
    x: FloatArray,
    fs: float,
    peak_r: SpectralPeak,
    waveform: FloatArray,
    cfg: VitalSignConfig,
) -> tuple[float | None, FloatArray, tuple[tuple[float, float], ...]]:
    """Respiration fundamental whose harmonics are handled in the heart band.

    Breathing faster than the respiration band (30 breaths/min and above
    with the default band) puts its second harmonic in the heart band (60 to
    75 beats/min for 30 to 37 breaths/min), often far stronger than the
    heartbeat, while the respiration-band spectrum then shows only the
    filter skirt: a maximum on either band edge, or a spurious interior
    bump. The fundamental is therefore also searched in a wider band, from
    the respiration band floor to half a notch width below the heart band.
    A peak there that lies above the respiration band and passes the
    null-model SNR test is the fundamental. Otherwise the respiration-band
    peak is used, as before.

    Where no fundamental is established, the interval it could lie in is
    returned instead: above the respiration band up to the heart band when
    the respiration maximum is on its upper edge, and between the top of the
    wide band and the heart band when the wide-band maximum is on its upper
    edge (breathing too fast to be resolved below the heart band). A heart
    peak at a harmonic of that interval is then withheld. A maximum on the
    lower edge (breathing slower than the band) adds the interval below the
    band floor, whose handled harmonics stay below the heart band with the
    default bands.

    Args:
        x: Detrended selected streams of the segment, shape ``(T, K)``.
        fs: Sampling rate (Hz).
        peak_r: Respiration-band peak of ``x``.
        waveform: Respiration waveform (first principal component).
        cfg: Estimator configuration.

    Returns:
        ``(fundamental_hz, reference_waveform, unresolved_hz)``: the
        fundamental (``None`` when none applies), the waveform its phase is
        taken from, and the frequency intervals ``(low, high)`` in which an
        unresolved respiration fundamental could lie (empty when there are
        none).
    """
    resp_low, resp_high = cfg.respiration_band_hz
    wide = (resp_low, cfg.heart_band_hz[0] - cfg.notch_width_hz / 2.0)
    ceiling = cfg.heart_band_hz[0] + cfg.notch_width_hz / 2.0
    upper_edge = peak_r.at_band_edge and _maximum_on_upper_edge(
        x, fs, cfg.respiration_band_hz, cfg.zero_pad_factor
    )
    unresolved: list[tuple[float, float]] = []
    if wide[1] > resp_high:
        xw = _bandpass(x, fs, wide, cfg.filter_order)
        peak_w = dominant_frequency(
            xw,
            fs,
            wide,
            signal_halfwidth_hz=cfg.respiration_signal_halfwidth_hz,
            zero_pad_factor=cfg.zero_pad_factor,
        )
        threshold = _snr_threshold(
            cfg.respiration_min_snr_db,
            effective_stream_count(
                xw,
                fs,
                wide,
                exclude_hz=peak_w.frequency_hz,
                exclude_halfwidth_hz=peak_w.halfwidth_hz,
            ),
            x.shape[0],
            fs,
            wide,
            cfg.respiration_signal_halfwidth_hz,
            cfg,
        )
        f_w = peak_w.frequency_hz
        if (
            f_w is not None
            and (f_w > resp_high or upper_edge)
            and _passes(peak_w.snr_db, threshold)
        ):
            return f_w, _combine(xw), ()
        if peak_w.at_band_edge and _maximum_on_upper_edge(xw, fs, wide, cfg.zero_pad_factor):
            unresolved.append((wide[1], ceiling))
    if peak_r.frequency_hz is not None:
        return peak_r.frequency_hz, waveform, tuple(unresolved)
    if upper_edge:
        unresolved.append((resp_high, ceiling))
    elif peak_r.at_band_edge:
        unresolved.append((0.0, resp_low))
    return None, waveform, tuple(unresolved)


def _unresolved_harmonic(
    f_hz: float,
    unresolved: tuple[tuple[float, float], ...],
    duration_s: float,
    cfg: VitalSignConfig,
) -> tuple[int, float, float] | None:
    """First ``(order, low, high)`` such that ``f_hz`` is a harmonic of ``[low, high]``.

    Orders run from 2 to ``max_harmonic_order``; the interval is widened by
    the tolerance of the harmonic-coincidence test.
    """
    for low, high in unresolved:
        for order in range(2, cfg.max_harmonic_order + 1):
            precision = cfg.harmonic_coincidence_fraction + order * cfg.respiration_precision_bins
            tolerance = precision / duration_s
            if order * low - tolerance <= f_hz <= order * high + tolerance:
                return order, low, high
    return None


def _screen_heart(
    heart: RateEstimate,
    respiration_hz: float | None,
    suppressed_hz: tuple[float, ...],
    unresolved: tuple[tuple[float, float], ...],
    duration_s: float,
    cfg: VitalSignConfig,
) -> RateEstimate:
    """Withhold a heart estimate that may be a harmonic the earlier tests cannot see.

    Three cases are withheld with ``HARMONIC_AMBIGUITY``: the peak is a
    harmonic of an unresolved respiration fundamental (see
    ``_harmonic_reference``); the peak lies within
    ``(cancellation_guard_bins + k * respiration_precision_bins) / T`` of a
    suppressed respiration harmonic ``k``, where suppression has removed or
    shifted the heartbeat and the residual peak is biased; or half its
    frequency coincides with a suppressed harmonic to within the tolerance
    of the harmonic-coincidence test, so the heartbeat's fundamental may
    have been removed together with that harmonic and the peak be the
    heartbeat's own second harmonic.

    Args:
        heart: The estimate after the SNR and coincidence tests.
        respiration_hz: Fundamental whose harmonics were suppressed.
        suppressed_hz: The suppressed harmonics (empty when none were).
        unresolved: Intervals of an unresolved respiration fundamental.
        duration_s: Segment duration (s).
        cfg: Estimator configuration.

    Returns:
        ``heart`` unchanged, or the abstention.
    """
    f_h = heart.peak_hz
    if f_h is None or heart.rate_bpm is None:
        return heart
    detail = ""
    hit = _unresolved_harmonic(f_h, unresolved, duration_s, cfg)
    if hit is not None:
        order, low, high = hit
        detail = (
            f"heart peak {f_h:.3f} Hz may be respiration harmonic {order} of an "
            f"unresolved fundamental in {low:.2f} to {high:.2f} Hz (respiration "
            "maximum on a band edge)"
        )
    elif respiration_hz is not None:
        for harmonic in suppressed_hz:
            order = round(harmonic / respiration_hz)
            spread = order * cfg.respiration_precision_bins
            guard = (cfg.cancellation_guard_bins + spread) / duration_s
            if cfg.cancellation_guard_bins > 0.0 and abs(f_h - harmonic) <= guard:
                detail = (
                    f"heart peak {f_h:.3f} Hz lies within {guard:.3f} Hz of suppressed "
                    f"respiration harmonic {order} x {respiration_hz:.3f} Hz, where "
                    "suppression biases the residual peak"
                )
                break
            precision = cfg.harmonic_coincidence_fraction + spread
            if abs(0.5 * f_h - harmonic) <= precision / duration_s:
                detail = (
                    f"heart peak {f_h:.3f} Hz may be the second harmonic of a heartbeat at "
                    f"{0.5 * f_h:.3f} Hz removed with respiration harmonic {order} x "
                    f"{respiration_hz:.3f} Hz"
                )
                break
    if not detail:
        return heart
    return RateEstimate(
        None,
        AbstainReason.HARMONIC_AMBIGUITY,
        detail,
        f_h,
        heart.snr_db,
        snr_threshold_db=heart.snr_threshold_db,
    )


def _analyse_segment(
    seg: FloatArray, fs: float, cfg: VitalSignConfig
) -> tuple[RateEstimate, RateEstimate, IntArray, tuple[float, ...], FloatArray]:
    """Estimate both rates on a clean, uniformly sampled segment."""
    empty_idx = np.zeros(0, dtype=np.intp)
    empty_wave = np.zeros(0, dtype=np.float64)
    sos_len = cfg.filter_order  # second-order sections per band-pass
    if seg.shape[0] <= 3 * (2 * sos_len + 1):
        out = _abstain(AbstainReason.INSUFFICIENT_DATA, "segment too short to filter")
        return out, out, empty_idx, (), empty_wave
    x = np.asarray(signal.detrend(seg, axis=0, type="linear"), dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(seg))))
    live = np.ptp(x, axis=0) > 1e-12 * scale
    if not np.any(live):
        out = _abstain(AbstainReason.FLAT_SIGNAL, "no stream varies over the segment")
        return out, out, empty_idx, (), empty_wave
    live_idx = np.flatnonzero(live)
    chosen = live_idx[_select(x[:, live], fs, cfg)]
    if chosen.shape[0] == 0:
        out = _abstain(AbstainReason.FLAT_SIGNAL, "no stream carries in-band power")
        return out, out, empty_idx, (), empty_wave
    xr = _bandpass(x[:, chosen], fs, cfg.respiration_band_hz, cfg.filter_order)
    waveform = _combine(xr)
    peak_r = dominant_frequency(
        xr,
        fs,
        cfg.respiration_band_hz,
        signal_halfwidth_hz=cfg.respiration_signal_halfwidth_hz,
        zero_pad_factor=cfg.zero_pad_factor,
    )
    n_seg = seg.shape[0]
    k_resp = effective_stream_count(
        xr,
        fs,
        cfg.respiration_band_hz,
        exclude_hz=peak_r.frequency_hz,
        exclude_halfwidth_hz=peak_r.halfwidth_hz,
    )
    thr_r = _snr_threshold(
        cfg.respiration_min_snr_db,
        k_resp,
        n_seg,
        fs,
        cfg.respiration_band_hz,
        cfg.respiration_signal_halfwidth_hz,
        cfg,
    )
    thr_joint = _snr_threshold(
        cfg.respiration_min_snr_db,
        k_resp,
        n_seg,
        fs,
        cfg.respiration_band_hz,
        2.0 * peak_r.halfwidth_hz,
        cfg,
    )
    resp = _respiration_estimate(peak_r, waveform, fs, thr_r, thr_joint, cfg)
    if resp.reason is AbstainReason.MULTIPLE_SOURCES:
        heart = _abstain(
            AbstainReason.MULTIPLE_SOURCES,
            "more than one periodic source in the respiration band; out of scope",
        )
        return resp, heart, chosen, (), waveform
    f_ref, ref_wave, unresolved = _harmonic_reference(x[:, chosen], fs, peak_r, waveform, cfg)
    if f_ref is not None and f_ref > cfg.respiration_band_hz[1] and not resp.abstained:
        # the in-band "peak" is filter skirt or a sidelobe, not the breathing cycle
        resp = RateEstimate(
            None,
            AbstainReason.PEAK_AT_BAND_EDGE,
            f"breathing fundamental {f_ref:.3f} Hz lies above the respiration band "
            f"{cfg.respiration_band_hz}; the in-band peak at {resp.peak_hz:.3f} Hz is not "
            "the breathing cycle",
            resp.peak_hz,
            resp.snr_db,
            resp.count_rate_bpm,
            resp.snr_threshold_db,
        )
    harmonics: tuple[float, ...] = ()
    if f_ref is not None:
        harmonics = respiration_harmonics(
            f_ref,
            cfg.heart_band_hz,
            cfg.max_harmonic_order,
            margin_hz=cfg.notch_width_hz / 2.0,
        )
    xh = _bandpass(x[:, chosen], fs, cfg.heart_band_hz, cfg.filter_order)
    if harmonics and f_ref is not None:
        if cfg.harmonic_suppression == "regression":
            orders = tuple(round(h / f_ref) for h in harmonics)
            xh = cancel_respiration_harmonics(
                xh,
                ref_wave,
                fs,
                f_ref,
                orders,
                cfg.heart_band_hz,
                poly_degree=min(cfg.harmonic_poly_degree, int(n_seg / fs // 15.0)),
                filter_order=cfg.filter_order,
            )
        elif cfg.harmonic_suppression == "notch":
            xh = suppress_harmonics(xh, fs, harmonics, cfg.notch_width_hz)
        else:
            harmonics = ()
    peak_h = dominant_frequency(
        xh,
        fs,
        cfg.heart_band_hz,
        signal_halfwidth_hz=cfg.heart_signal_halfwidth_hz,
        zero_pad_factor=cfg.zero_pad_factor,
    )
    thr_h = _snr_threshold(
        cfg.heart_min_snr_db,
        effective_stream_count(
            xh,
            fs,
            cfg.heart_band_hz,
            exclude_hz=peak_h.frequency_hz,
            exclude_halfwidth_hz=peak_h.halfwidth_hz,
        ),
        n_seg,
        fs,
        cfg.heart_band_hz,
        cfg.heart_signal_halfwidth_hz,
        cfg,
    )
    heart = _heart_estimate(peak_h, thr_h, f_ref, n_seg / fs, cfg)
    heart = _screen_heart(heart, f_ref, harmonics, unresolved, n_seg / fs, cfg)
    return resp, heart, chosen, harmonics, waveform


def _window_estimate(prep: _Prepared, i0: int, i1: int, cfg: VitalSignConfig) -> VitalSignEstimate:
    fs = prep.fs
    n = max(i1 - i0, 1)
    missing = prep.missing[i0:i1]
    motion = prep.motion[i0:i1]
    motion_fraction = float(np.count_nonzero(motion)) / n
    missing_fraction = float(np.count_nonzero(prep.lost[i0:i1])) / n
    empty_wave = np.zeros(0, dtype=np.float64)

    def withheld(reason: AbstainReason, detail: str) -> VitalSignEstimate:
        out = _abstain(reason, detail)
        return VitalSignEstimate(
            out,
            out,
            fs,
            0.0,
            0.0,
            motion_fraction,
            missing_fraction,
            (),
            (),
            empty_wave,
            prep.units,
        )

    if prep.streams.shape[1] == 0 or missing_fraction >= 1.0:
        return withheld(AbstainReason.INSUFFICIENT_DATA, "no finite samples in the input")
    if motion_fraction > cfg.max_motion_fraction:
        return withheld(
            AbstainReason.MOTION,
            f"{motion_fraction:.0%} of the window is body motion "
            f"(limit {cfg.max_motion_fraction:.0%}); subject not stationary",
        )
    start, end = _longest_run(~(missing | motion))
    clean_s = (end - start) / fs
    if clean_s < cfg.min_clean_s:
        reason = AbstainReason.MOTION if np.any(motion) else AbstainReason.INSUFFICIENT_DATA
        return withheld(reason, f"longest clean segment {clean_s:.1f} s < {cfg.min_clean_s:.1f} s")
    seg = prep.streams[i0 + start : i0 + end]
    resp, heart, chosen, harmonics, waveform = _analyse_segment(seg, fs, cfg)
    return VitalSignEstimate(
        resp,
        heart,
        fs,
        start / fs,
        end / fs,
        motion_fraction,
        missing_fraction,
        tuple(int(prep.stream_ids[c]) for c in chosen),
        harmonics,
        waveform,
        prep.units,
    )


def estimate_vital_signs(
    *,
    fs: float,
    consent_confirmed: bool | ConsentRecord,
    csi_amplitude: ArrayLike | None = None,
    csi_phase: ArrayLike | None = None,
    subcarrier_indices: ArrayLike | None = None,
    radar_displacement_m: ArrayLike | None = None,
    radar_phase_rad: ArrayLike | None = None,
    wavelength_m: float | None = None,
    timestamps_s: ArrayLike | None = None,
    config: VitalSignConfig | None = None,
) -> VitalSignEstimate:
    """Estimate respiration and heart rate over one recording window.

    Pass exactly one modality: CSI (``csi_amplitude`` and/or ``csi_phase``)
    or radar (``radar_displacement_m``, or ``radar_phase_rad`` with
    ``wavelength_m``). The whole input is treated as one window; the longest
    clean (motion-free, non-missing) segment is analysed.

    Args:
        fs: Sampling rate of the input (Hz). With ``timestamps_s`` it is the
            uniform rate the packets are resampled to: they are interpolated
            onto a grid at least as fast as the median packet rate and then
            anti-alias filtered down to ``fs`` (or to
            ``config.analysis_fs`` when that is lower).
        consent_confirmed: ``True``, or a ``truth_decipher.ConsentRecord``
            whose scope includes ``VITAL_SIGNS_CONSENT_SCOPE``: the subject
            has given informed consent to contactless sensing.
        csi_amplitude: CSI amplitude, shape ``(T, S)`` or ``(T, S, A)``.
        csi_phase: CSI phase in radians, shape ``(T, S)`` (sanitised by a
            linear fit) or ``(T, S, A)`` with ``A >= 2`` (antenna
            differences).
        subcarrier_indices: Subcarrier indices for phase sanitisation.
        radar_displacement_m: Radar chest displacement (m), shape ``(T,)``.
        radar_phase_rad: Radar CW/FMCW phase (rad), shape ``(T,)``.
        wavelength_m: Carrier wavelength (m) for ``radar_phase_rad``.
        timestamps_s: Packet times (s), shape ``(T,)``, for irregular input.
        config: Estimator configuration (defaults when ``None``).

    Returns:
        The estimates; abstentions carry ``rate_bpm=None`` and a reason.

    Raises:
        ConsentRequiredError: If ``consent_confirmed`` is neither ``True``
            nor a ``ConsentRecord`` with ``VITAL_SIGNS_CONSENT_SCOPE``.
        ValueError: On malformed inputs or an inadmissible configuration.
        TypeError: On complex-valued inputs.
    """
    _require_consent(consent_confirmed)
    cfg = config if config is not None else VitalSignConfig()
    prep = _prepare(
        fs=fs,
        csi_amplitude=csi_amplitude,
        csi_phase=csi_phase,
        subcarrier_indices=subcarrier_indices,
        radar_displacement_m=radar_displacement_m,
        radar_phase_rad=radar_phase_rad,
        wavelength_m=wavelength_m,
        timestamps_s=timestamps_s,
        cfg=cfg,
    )
    return _window_estimate(prep, 0, prep.streams.shape[0], cfg)


def _global_waveform(prep: _Prepared, cfg: VitalSignConfig) -> FloatArray:
    n = prep.streams.shape[0]
    invalid = prep.missing | prep.motion
    waveform = np.full(n, np.nan)
    if prep.streams.shape[1] == 0 or np.count_nonzero(~invalid) <= 3 * (2 * cfg.filter_order + 1):
        return waveform
    x = _inpaint(prep.streams, invalid)
    x = np.asarray(signal.detrend(x, axis=0, type="linear"), dtype=np.float64)
    chosen = _select(x, prep.fs, cfg)
    if chosen.shape[0] == 0:
        return waveform
    full = _combine(_bandpass(x[:, chosen], prep.fs, cfg.respiration_band_hz, cfg.filter_order))
    waveform[~invalid] = full[~invalid]
    return waveform


def track_vital_signs(
    *,
    fs: float,
    consent_confirmed: bool | ConsentRecord,
    csi_amplitude: ArrayLike | None = None,
    csi_phase: ArrayLike | None = None,
    subcarrier_indices: ArrayLike | None = None,
    radar_displacement_m: ArrayLike | None = None,
    radar_phase_rad: ArrayLike | None = None,
    wavelength_m: float | None = None,
    timestamps_s: ArrayLike | None = None,
    window_s: float = 30.0,
    step_s: float = 1.0,
    t0_s: float | None = None,
    config: VitalSignConfig | None = None,
) -> VitalSignTrack:
    """Track both rates in sliding windows and keep the respiration waveform.

    Motion is detected once over the whole recording; every window is then
    analysed like ``estimate_vital_signs`` on its longest clean segment.
    The respiration waveform is formed once over the whole recording (one
    stream selection and one principal component) so that excerpts taken at
    different times share a scale.

    Args:
        fs: Sampling rate of the input (Hz). With ``timestamps_s`` it is the
            uniform rate the packets are resampled to: they are interpolated
            onto a grid at least as fast as the median packet rate and then
            anti-alias filtered down to ``fs`` (or to
            ``config.analysis_fs`` when that is lower).
        consent_confirmed: ``True``, or a ``truth_decipher.ConsentRecord``
            whose scope includes ``VITAL_SIGNS_CONSENT_SCOPE``: the subject
            has given informed consent to contactless sensing.
        csi_amplitude: CSI amplitude, shape ``(T, S)`` or ``(T, S, A)``.
        csi_phase: CSI phase in radians, ``(T, S)`` or ``(T, S, A)``.
        subcarrier_indices: Subcarrier indices for phase sanitisation.
        radar_displacement_m: Radar chest displacement (m), shape ``(T,)``.
        radar_phase_rad: Radar CW/FMCW phase (rad), shape ``(T,)``.
        wavelength_m: Carrier wavelength (m) for ``radar_phase_rad``.
        timestamps_s: Packet times (s), shape ``(T,)``; the analysis clock
            then starts at ``timestamps_s[0]``.
        window_s: Window length (s); must be at least ``config.min_clean_s``.
        step_s: Window step (s).
        t0_s: Time of the first sample when no timestamps are given
            (default 0). Not allowed together with ``timestamps_s``.
        config: Estimator configuration (defaults when ``None``).

    Returns:
        The sliding-window track.

    Raises:
        ConsentRequiredError: If ``consent_confirmed`` is neither ``True``
            nor a ``ConsentRecord`` with ``VITAL_SIGNS_CONSENT_SCOPE``.
        ValueError: On malformed inputs, inadmissible window settings or a
            recording shorter than one window.
        TypeError: On complex-valued inputs.
    """
    _require_consent(consent_confirmed)
    cfg = config if config is not None else VitalSignConfig()
    if not (math.isfinite(window_s) and math.isfinite(step_s) and window_s > 0 and step_s > 0):
        raise ValueError("window_s and step_s must be positive and finite")
    if window_s < cfg.min_clean_s:
        raise ValueError("window_s must be at least config.min_clean_s")
    if timestamps_s is not None and t0_s is not None:
        raise ValueError("t0_s cannot be combined with timestamps_s")
    prep = _prepare(
        fs=fs,
        csi_amplitude=csi_amplitude,
        csi_phase=csi_phase,
        subcarrier_indices=subcarrier_indices,
        radar_displacement_m=radar_displacement_m,
        radar_phase_rad=radar_phase_rad,
        wavelength_m=wavelength_m,
        timestamps_s=timestamps_s,
        cfg=cfg,
    )
    origin = prep.t0 if timestamps_s is not None else (0.0 if t0_s is None else float(t0_s))
    n = prep.streams.shape[0]
    n_win = round(window_s * prep.fs)
    n_step = max(1, round(step_s * prep.fs))
    if n_win > n:
        raise ValueError(f"recording ({n / prep.fs:.1f} s) is shorter than one window")
    starts = np.arange(0, n - n_win + 1, n_step)
    estimates = [_window_estimate(prep, int(i), int(i) + n_win, cfg) for i in starts]
    start_s = origin + starts / prep.fs
    resp_bpm = np.array([_or_nan(e.respiration.rate_bpm) for e in estimates], dtype=np.float64)
    heart_bpm = np.array([_or_nan(e.heart.rate_bpm) for e in estimates], dtype=np.float64)
    return VitalSignTrack(
        fs_analysis=prep.fs,
        window_s=n_win / prep.fs,
        step_s=n_step / prep.fs,
        window_start_s=start_s,
        window_end_s=start_s + n_win / prep.fs,
        window_center_s=start_s + n_win / (2.0 * prep.fs),
        respiration_bpm=resp_bpm,
        respiration_snr_db=np.array([e.respiration.snr_db for e in estimates], dtype=np.float64),
        respiration_reasons=tuple(e.respiration.reason for e in estimates),
        heart_bpm=heart_bpm,
        heart_snr_db=np.array([e.heart.snr_db for e in estimates], dtype=np.float64),
        heart_reasons=tuple(e.heart.reason for e in estimates),
        motion_fraction=np.array([e.motion_fraction for e in estimates], dtype=np.float64),
        sample_times_s=origin + np.arange(n, dtype=np.float64) / prep.fs,
        respiration_waveform=_global_waveform(prep, cfg),
        motion_mask=prep.motion.copy(),
        missing_mask=prep.missing.copy(),
        waveform_units=prep.units,
    )


def _or_nan(value: float | None) -> float:
    return math.nan if value is None else value


# ---------------------------------------------------------------------------
# Stimulus-locked access for downstream consumers
# ---------------------------------------------------------------------------


def extract_waveform_excerpt(
    track: VitalSignTrack, start_s: float, end_s: float
) -> WaveformExcerpt:
    """Return the respiration waveform on ``[start_s, end_s)``, or abstain.

    The excerpt is withheld (``samples is None``) when the interval leaves
    the recording, holds fewer than two samples, or touches any motion or
    missing sample.

    Args:
        track: Output of ``track_vital_signs``.
        start_s: Interval start on the track's clock (s).
        end_s: Interval end, exclusive (s).

    Returns:
        The excerpt or an abstention with its reason.

    Raises:
        ValueError: If the bounds are not finite or ``end_s <= start_s``.
    """
    if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
        raise ValueError("need finite bounds with end_s > start_s")
    times = track.sample_times_s

    def withheld(reason: AbstainReason, detail: str) -> WaveformExcerpt:
        return WaveformExcerpt(
            None, None, track.fs_analysis, start_s, end_s, track.waveform_units, reason, detail
        )

    half_step = 0.5 / track.fs_analysis
    if times.shape[0] == 0 or start_s < times[0] - half_step or end_s > times[-1] + 2 * half_step:
        return withheld(AbstainReason.INSUFFICIENT_DATA, "interval leaves the recording")
    sel = (times >= start_s - 1e-9) & (times < end_s - 1e-9)
    if np.count_nonzero(sel) < 2:
        return withheld(AbstainReason.INSUFFICIENT_DATA, "fewer than two samples in interval")
    if np.any(track.motion_mask[sel]):
        return withheld(AbstainReason.MOTION, "interval overlaps body motion")
    samples = track.respiration_waveform[sel]
    if np.any(track.missing_mask[sel]) or not np.all(np.isfinite(samples)):
        return withheld(AbstainReason.INSUFFICIENT_DATA, "interval overlaps missing samples")
    return WaveformExcerpt(
        samples.copy(),
        times[sel].copy(),
        track.fs_analysis,
        start_s,
        end_s,
        track.waveform_units,
        None,
        "ok",
    )


def windows_within(track: VitalSignTrack, start_s: float, end_s: float) -> IntArray:
    """Indices of windows lying entirely inside ``[start_s, end_s]``.

    Args:
        track: Output of ``track_vital_signs``.
        start_s: Interval start (s).
        end_s: Interval end (s).

    Returns:
        Window indices in increasing time order (possibly empty).
    """
    tol = 1e-9
    inside = (track.window_start_s >= start_s - tol) & (track.window_end_s <= end_s + tol)
    return np.asarray(np.flatnonzero(inside), dtype=np.intp)


def rates_within(
    track: VitalSignTrack,
    start_s: float,
    end_s: float,
    which: Literal["respiration", "heart"] = "heart",
) -> tuple[FloatArray, FloatArray]:
    """Window centres and rates for windows entirely inside an interval.

    Abstained windows are dropped, so the result holds only estimates.

    Args:
        track: Output of ``track_vital_signs``.
        start_s: Interval start (s).
        end_s: Interval end (s).
        which: ``"respiration"`` or ``"heart"``.

    Returns:
        ``(centres_s, rates_per_minute)``, both possibly empty.

    Raises:
        ValueError: If ``which`` is not recognised.
    """
    if which == "heart":
        rates = track.heart_bpm
    elif which == "respiration":
        rates = track.respiration_bpm
    else:
        raise ValueError("which must be 'respiration' or 'heart'")
    idx = windows_within(track, start_s, end_s)
    idx = idx[np.isfinite(rates[idx])]
    return track.window_center_s[idx].copy(), rates[idx].copy()


def rate_series(
    track: VitalSignTrack, which: Literal["respiration", "heart"] = "heart"
) -> tuple[FloatArray, FloatArray]:
    """Finite ``(times, rates)`` of every non-abstained window, for downstream series.

    Times are window centres, strictly increasing; abstained windows are
    dropped rather than filled. Each value averages its whole window, so it
    cannot resolve changes faster than ``track.window_s``.

    Args:
        track: Output of ``track_vital_signs``.
        which: ``"respiration"`` or ``"heart"``.

    Returns:
        ``(window_centres_s, rates_per_minute)``, both possibly empty.

    Raises:
        ValueError: If ``which`` is not recognised.
    """
    if which not in ("respiration", "heart"):
        raise ValueError("which must be 'respiration' or 'heart'")
    return rates_within(track, -math.inf, math.inf, which)


def respiration_series(track: VitalSignTrack) -> tuple[FloatArray, FloatArray]:
    """Finite ``(times, samples)`` of the respiration waveform.

    Missing and motion samples are dropped, so consecutive returned samples
    can straddle a gap; check an interval with ``extract_waveform_excerpt``
    before computing anything that differences adjacent samples over it.

    Args:
        track: Output of ``track_vital_signs``.

    Returns:
        ``(sample_times_s, waveform)``, both possibly empty.
    """
    keep = np.isfinite(track.respiration_waveform) & ~track.motion_mask & ~track.missing_mask
    return track.sample_times_s[keep].copy(), track.respiration_waveform[keep].copy()

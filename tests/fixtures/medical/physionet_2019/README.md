# PhysioNet / Computing in Cardiology Challenge 2019 (two recorded ICU stays)

`p000001.psv` (54 hourly rows, never septic) and `p000011.psv` (34 hourly
rows, `SepsisLabel` = 1 from the 25th row on) are copied unchanged from training set A of
the PhysioNet/CinC Challenge 2019, "Early Prediction of Sepsis from Clinical
Data", version 1.0.0
(https://physionet.org/content/challenge-2019/1.0.0/training/training_setA/),
retrieved 2026-10-23. Licence: Creative Commons Attribution 4.0 International.

Cite: Reyna MA, Josef CS, Jeter R, et al. Early Prediction of Sepsis From
Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. Crit Care
Med 48(2):210-217 (2020); Goldberger AL, et al. PhysioBank, PhysioToolkit and
PhysioNet. Circulation 101(23):e215-e220 (2000).

| File | SHA-256 |
|---|---|
| `p000001.psv` | `63f11ce1bf106141297aa1df4afc25ad30dfb0332ba8ae25dc72669f92fbc680` |
| `p000011.psv` | `4303021a35d87a9b74db3590ad0061416a46b0fc2036627b7ae7bb1becbaae2e` |

Consumed by `tests/medical/test_sepsis_recorded.py`. `NaN` in the files means
"not measured that hour". `SepsisLabel` is not a screening reference: the
challenge sets it six hours before an onset that also requires suspected
infection (Reyna et al. 2020), which the files do not record.

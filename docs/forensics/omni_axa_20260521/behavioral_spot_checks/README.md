# Step 3(c) behavioral spot-check fallback (AST control-flow fingerprint)

Live imports were not attempted because Omni's full dependency tree (torch, sklearn, speech_recognition, pyod, etc.) is not installed in the audit container. Per the task rules, falling back to AST control-flow fingerprint and noting the downgrade.

| label | omni symbol | mercury symbol | omni CFG hash | mercury CFG hash | match? |
|---|---|---|---|---|---|
| truth_decipher.TruthDecipherFramework | `TruthDecipherFramework` | `TruthDecipherFramework` | `44c7295d9535068b` | `bff573c3f7f81cd7` | NO (implementation differs) |
| engine.OmniAnomalyEngine_vs_OmniMercuryEngine | `OmniAnomalyEngine` | `OmniMercuryEngine` | `2a029fa533445adb` | `6b2103c5bbe8bf99` | NO (implementation differs) |
| cli.detect | `detect` | `detect` | `dd09ed571dafdbbf` | `bf0bc139952d579a` | NO (implementation differs) |
| mercury_a_agent.MercuryAgent | `MercuryAgent` | `MercuryAgent` | `bb27e181d7f25f2c` | `6bbf82c8927cf306` | NO (implementation differs) |
| mercury_a_crews.AgentRole | `AgentRole` | `AgentRole` | `664165332cffcfc7` | `4f9ba1f19304f776` | NO (implementation differs) |
| mercury_a_learning.MercuryLearner_vs_PPOTrainer | `MercuryLearner` | `PPOTrainer` | `98c70a6c4c502328` | `b5a8a661188dcbf5` | NO (implementation differs) |
| comparison.PyODComparison | `PyODComparison` | `PyODComparison` | `1ab857653a4e51b1` | `df4fda07dda6938a` | NO (implementation differs) |
| detectors.StatisticalAnomalyDetector | `StatisticalAnomalyDetector` | `StatisticalAnomalyDetector` | `5f03bacd3c1c4b5d` | `NOT_FOUND` | NO (implementation differs) |

**Interpretation**: A *hash mismatch* between Omni and Mercury counterparts is the *expected* outcome when Mercury is a re-implementation or superset — it does NOT imply the Omni code should be ported. A hash *match* would prove byte-equivalent behavior at the AST level.
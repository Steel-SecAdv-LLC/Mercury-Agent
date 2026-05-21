# Symbol diff: mercury_a_crews

- **Omni**:    `src/omni_anomaly_engine/mercury_a_crews.py`  (size 20356 bytes)
- **Mercury**: `src/omni_mercury_engine/cognitive/multi_agent_coordination.py`  (size 42745 bytes)

## Public symbol counts
- Omni public symbols:    **10**
- Mercury public symbols: **15**
- Shared:                 **1**
- Only in Omni:           **9**   ← POTENTIAL_EXTRACTION_CANDIDATE
- Only in Mercury:        **14**

## ⚠ POTENTIAL_EXTRACTION_CANDIDATE (symbols in Omni only)
- `cls BaseCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:55
- `cls CrewCoordinator` @ src/omni_anomaly_engine/mercury_a_crews.py:518
- `cls CrewTask` @ src/omni_anomaly_engine/mercury_a_crews.py:44
- `cls EmergentCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:461
- `cls EnergyCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:298
- `cls InfrastructureCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:351
- `cls MedicalCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:105
- `cls SecurityCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:213
- `cls SpaceCrew` @ src/omni_anomaly_engine/mercury_a_crews.py:404

## Mercury-only (likely added enhancements)
- `cls AgentCapability` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:161
- `cls AgentCoordinator` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:815
- `cls AgentStatus` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:81
- `cls Coalition` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:237
- `cls ConsensusMethod` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:109
- `cls ConsensusProtocol` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:467
- `cls ConsensusResult` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:211
- `cls CoordinationStrategy` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:119
- `cls DetectionAgent` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:283
- `cls DetectionResult` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:185
- `cls Message` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:134
- `cls MessageType` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:91
- `cls MultiAgentDetectionSystem` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:1130
- `cls SimpleDetectionAgent` @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:389

## Shared symbols — AST signature diff
### `AgentRole` — = IDENTICAL
- Omni:    `cls AgentRole`  @ src/omni_anomaly_engine/mercury_a_crews.py:33
- Mercury: `cls AgentRole`  @ src/omni_mercury_engine/cognitive/multi_agent_coordination.py:68

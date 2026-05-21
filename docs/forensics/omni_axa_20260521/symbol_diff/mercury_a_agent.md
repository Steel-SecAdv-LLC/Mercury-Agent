# Symbol diff: mercury_a_agent

- **Omni**:    `src/omni_anomaly_engine/mercury_a_agent.py`  (size 32819 bytes)
- **Mercury**: `src/omni_mercury_engine/agentic/mercury_a_agent.py`  (size 35304 bytes)

## Public symbol counts
- Omni public symbols:    **8**
- Mercury public symbols: **12**
- Shared:                 **6**
- Only in Omni:           **2**   ← POTENTIAL_EXTRACTION_CANDIDATE
- Only in Mercury:        **6**

## ⚠ POTENTIAL_EXTRACTION_CANDIDATE (symbols in Omni only)
- `cls AgentState` @ src/omni_anomaly_engine/mercury_a_agent.py:71
- `def analyze_with_mercury(data: npt.NDArray[np.float64], domain: str='general', goal: Optional[str]=None, verbose: bool=False)` @ src/omni_anomaly_engine/mercury_a_agent.py:926

## Mercury-only (likely added enhancements)
- `cls AgentMode` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:53
- `cls MemoryEntry` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:88
- `cls PlanResult` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:129
- `cls ReasoningStep` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:117
- `cls TaskPriority` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:64
- `def create_mercury_agent(name: str='Mercury', autonomy_level: float=0.8, ethical_threshold: float=0.93)` @ src/omni_mercury_engine/agentic/mercury_a_agent.py:1028

## Shared symbols — AST signature diff
### `AgentMemory` — ≠ CHANGED
- Omni:    `cls AgentMemory`  @ src/omni_anomaly_engine/mercury_a_agent.py:112
- Mercury: `cls AgentMemory`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:140
  - methods only in Omni: ['recall', 'remember']
  - methods only in Mercury: ['get_statistics', 'retrieve_by_importance', 'retrieve_recent', 'search_semantic', 'store_episodic', 'store_long_term', 'store_semantic', 'store_short_term']
### `DomainType` — = IDENTICAL
- Omni:    `cls DomainType`  @ src/omni_anomaly_engine/mercury_a_agent.py:82
- Mercury: `cls DomainType`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:74
### `MercuryAgent` — ≠ CHANGED
- Omni:    `cls MercuryAgent`  @ src/omni_anomaly_engine/mercury_a_agent.py:603
- Mercury: `cls MercuryAgent`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:761
  - methods only in Omni: ['listen', 'speak']
  - methods only in Mercury: ['register_tool']
### `MercuryPlanner` — ≠ CHANGED
- Omni:    `cls MercuryPlanner`  @ src/omni_anomaly_engine/mercury_a_agent.py:144
- Mercury: `cls MercuryPlanner`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:468
  - methods only in Omni: ['reflect']
### `MercuryReasoner` — ≠ CHANGED
- Omni:    `cls MercuryReasoner`  @ src/omni_anomaly_engine/mercury_a_agent.py:453
- Mercury: `cls MercuryReasoner`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:329
  - methods only in Omni: ['build_correlation_graph']
  - methods only in Mercury: ['get_reasoning_trace']
### `Task` — = IDENTICAL
- Omni:    `cls Task`  @ src/omni_anomaly_engine/mercury_a_agent.py:96
- Mercury: `cls Task`  @ src/omni_mercury_engine/agentic/mercury_a_agent.py:101

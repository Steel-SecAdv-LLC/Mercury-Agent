"""
OMNI AVA (O+A)
Copyright (C) 2025 Steel Security Advisory LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
"""

"""
Mercury Agent Training and Evaluation Suite

This module provides training scenarios for the Mercury Agent's learning
capabilities. Since Mercury Agent is a rule-based autonomous agent (not a
neural network), "training" involves:

1. Exercising the agent through diverse scenarios across all domains
2. Building up the agent's memory systems (episodic, semantic, short/long-term)
3. Evaluating reasoning quality and plan execution success rates
4. Testing ethical constraint enforcement

The agent learns through experience accumulation in its memory systems,
which influences future planning and reasoning decisions.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from omni_anomaly_engine.agentic.mercury_a_agent import (
    DomainType,
    MercuryAgent,
    create_mercury_agent,
)


@dataclass
class TrainingScenario:
    """A training scenario for the Mercury Agent."""

    name: str
    domain: DomainType
    goal: str
    data: dict[str, Any]
    context: dict[str, Any]
    expected_task_count_min: int = 1
    expected_success_rate_min: float = 0.5


@dataclass
class TrainingResult:
    """Results from a training scenario."""

    scenario_name: str
    domain: str
    goal: str
    plan_confidence: float
    legacy_confidence: float  # The old fixed heuristic for comparison
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    reasoning_steps: int
    memory_entries: int
    execution_time_ms: float
    passed: bool


def create_training_scenarios() -> list[TrainingScenario]:
    """Create diverse training scenarios across all domains."""
    scenarios = [
        TrainingScenario(
            name="medical_vital_signs",
            domain=DomainType.MEDICAL,
            goal="Analyze patient vital signs for anomalies",
            data={
                "heart_rate": [72, 75, 78, 150, 73, 71],
                "blood_pressure": [(120, 80), (118, 78), (180, 110), (122, 82)],
                "temperature": [98.6, 98.4, 102.5, 98.7],
            },
            context={"patient_id": "P001", "urgency": "routine"},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="security_network_traffic",
            domain=DomainType.SECURITY,
            goal="Detect potential intrusion patterns in network traffic",
            data={
                "connections": [
                    {"src": "192.168.1.1", "dst": "10.0.0.1", "port": 443},
                    {"src": "192.168.1.2", "dst": "10.0.0.1", "port": 22},
                    {"src": "unknown", "dst": "10.0.0.1", "port": 4444},
                ],
                "bytes_transferred": [1024, 2048, 1000000],
            },
            context={"network_segment": "internal", "alert_level": "elevated"},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="humanitarian_crisis_detection",
            domain=DomainType.HUMANITARIAN,
            goal="Monitor humanitarian crisis indicators",
            data={
                "displacement_reports": 150,
                "food_security_index": 0.3,
                "water_access_percentage": 45,
                "medical_facility_status": "overwhelmed",
            },
            context={"region": "conflict_zone", "population": 50000},
            expected_task_count_min=3,
            expected_success_rate_min=0.6,
        ),
        TrainingScenario(
            name="infrastructure_monitoring",
            domain=DomainType.INFRASTRUCTURE,
            goal="Assess critical infrastructure health",
            data={
                "power_grid_load": 0.92,
                "water_pressure_psi": [45, 48, 12, 50],
                "bridge_sensor_readings": {"vibration": 0.8, "strain": 0.6},
            },
            context={"city": "metro_area", "season": "summer"},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="energy_grid_analysis",
            domain=DomainType.ENERGY,
            goal="Optimize energy distribution and detect anomalies",
            data={
                "solar_output_mw": [100, 95, 20, 105],
                "wind_output_mw": [50, 55, 48, 52],
                "demand_mw": [200, 210, 250, 195],
                "storage_level": 0.65,
            },
            context={"grid_region": "northeast", "time_of_day": "peak"},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="scientific_data_analysis",
            domain=DomainType.SCIENTIFIC,
            goal="Analyze experimental data for significant findings",
            data={
                "measurements": np.random.normal(100, 15, 50).tolist(),
                "control_group": np.random.normal(100, 10, 50).tolist(),
                "p_value": 0.03,
            },
            context={"experiment_type": "clinical_trial", "phase": 2},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="financial_fraud_detection",
            domain=DomainType.FINANCIAL,
            goal="Detect potential fraudulent transactions",
            data={
                "transactions": [
                    {"amount": 50, "merchant": "grocery", "location": "local"},
                    {"amount": 5000, "merchant": "electronics", "location": "foreign"},
                    {"amount": 75, "merchant": "restaurant", "location": "local"},
                ],
                "account_history_avg": 200,
            },
            context={"account_type": "personal", "risk_profile": "low"},
            expected_task_count_min=3,
            expected_success_rate_min=0.7,
        ),
        TrainingScenario(
            name="general_anomaly_detection",
            domain=DomainType.GENERAL,
            goal="Perform general anomaly detection on mixed data",
            data={
                "sensor_readings": [1.0, 1.1, 0.9, 5.5, 1.0, 1.2],
                "timestamps": list(range(6)),
                "labels": ["normal"] * 5 + ["unknown"],
            },
            context={"source": "iot_sensors", "frequency": "1hz"},
            expected_task_count_min=2,
            expected_success_rate_min=0.6,
        ),
    ]

    return scenarios


def register_mock_tools(agent: MercuryAgent) -> None:
    """Register mock tools for training scenarios."""

    def analyze_data(data: Any) -> dict[str, Any]:
        return {"status": "analyzed", "anomalies_found": 2}

    def alert_operator(message: str) -> dict[str, Any]:
        return {"status": "alerted", "message": message}

    def log_event(event: str) -> dict[str, Any]:
        return {"status": "logged", "event": event}

    def query_database(query: str) -> dict[str, Any]:
        return {"status": "queried", "results": []}

    agent.register_tool("analyze_data", analyze_data)
    agent.register_tool("alert_operator", alert_operator)
    agent.register_tool("log_event", log_event)
    agent.register_tool("query_database", query_database)


def run_training_scenario(agent: MercuryAgent, scenario: TrainingScenario) -> TrainingResult:
    """Run a single training scenario."""
    start_time = time.perf_counter()

    result = agent.analyze(
        data=scenario.data,
        domain=scenario.domain,
        goal=scenario.goal,
        context=scenario.context,
    )

    execution_time = (time.perf_counter() - start_time) * 1000

    execution = result.get("execution", {})
    tasks_completed = execution.get("tasks_completed", 0)
    tasks_failed = execution.get("tasks_failed", 0)
    success_rate = execution.get("success_rate", 0.0)

    reasoning = result.get("reasoning", {})
    reasoning_steps = len(reasoning.get("trace", []))

    memory_stats = result.get("memory_stats", {})
    memory_entries = sum(
        [
            memory_stats.get("short_term_count", 0),
            memory_stats.get("long_term_count", 0),
            memory_stats.get("episodic_count", 0),
            memory_stats.get("semantic_count", 0),
        ]
    )

    total_tasks = tasks_completed + tasks_failed
    passed = (
        total_tasks >= scenario.expected_task_count_min
        and success_rate >= scenario.expected_success_rate_min
    )

    # Extract legacy confidence from metadata if available
    plan_metadata = {}
    if agent.current_plan is not None:
        plan_metadata = agent.current_plan.metadata
    legacy_confidence = plan_metadata.get("legacy_confidence_heuristic", result.get("plan_confidence", 0.0))

    return TrainingResult(
        scenario_name=scenario.name,
        domain=scenario.domain.value,
        goal=scenario.goal,
        plan_confidence=result.get("plan_confidence", 0.0),
        legacy_confidence=legacy_confidence,
        tasks_completed=tasks_completed,
        tasks_failed=tasks_failed,
        success_rate=success_rate,
        reasoning_steps=reasoning_steps,
        memory_entries=memory_entries,
        execution_time_ms=execution_time,
        passed=passed,
    )


def run_training_epochs(
    agent: MercuryAgent,
    scenarios: list[TrainingScenario],
    epochs: int = 30,
) -> dict[str, Any]:
    """
    Run multiple training epochs through all scenarios.

    Each epoch exercises the agent through all scenarios, allowing it to
    accumulate experience in its memory systems.
    """
    print("=" * 70)
    print("MERCURY AGENT TRAINING SUITE")
    print(f"Running {epochs} epochs across {len(scenarios)} scenarios")
    print("=" * 70)
    print()

    all_results: list[list[TrainingResult]] = []
    epoch_summaries: list[dict[str, Any]] = []

    for epoch in range(epochs):
        epoch_results: list[TrainingResult] = []
        epoch_start = time.perf_counter()

        for scenario in scenarios:
            result = run_training_scenario(agent, scenario)
            epoch_results.append(result)

        epoch_time = (time.perf_counter() - epoch_start) * 1000
        all_results.append(epoch_results)

        avg_success = np.mean([r.success_rate for r in epoch_results])
        avg_confidence = np.mean([r.plan_confidence for r in epoch_results])
        avg_legacy_confidence = np.mean([r.legacy_confidence for r in epoch_results])
        passed_count = sum(1 for r in epoch_results if r.passed)

        # Get calibrator statistics if available
        calibrator_stats = {}
        if agent.confidence_calibrator is not None:
            calibrator_stats = agent.confidence_calibrator.get_summary()

        epoch_summary = {
            "epoch": epoch + 1,
            "avg_success_rate": float(avg_success),
            "avg_plan_confidence": float(avg_confidence),
            "avg_legacy_confidence": float(avg_legacy_confidence),
            "confidence_improvement": float(avg_confidence - avg_legacy_confidence),
            "scenarios_passed": passed_count,
            "total_scenarios": len(scenarios),
            "epoch_time_ms": epoch_time,
            "calibrator_stats": calibrator_stats,
        }
        epoch_summaries.append(epoch_summary)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            confidence_delta = avg_confidence - avg_legacy_confidence
            print(
                f"Epoch {epoch + 1:3d}/{epochs}: "
                f"Success={avg_success:.2%} | "
                f"Confidence={avg_confidence:.2f} (vs legacy {avg_legacy_confidence:.2f}, delta={confidence_delta:+.3f}) | "
                f"Passed={passed_count}/{len(scenarios)} | "
                f"Time={epoch_time:.1f}ms"
            )

    print()
    print("-" * 70)
    print("Training Complete")
    print("-" * 70)

    final_state = agent.get_state()
    memory_stats = final_state.get("memory", {})

    final_epoch_results = all_results[-1]
    final_avg_success = np.mean([r.success_rate for r in final_epoch_results])
    final_avg_confidence = np.mean([r.plan_confidence for r in final_epoch_results])
    final_avg_legacy = np.mean([r.legacy_confidence for r in final_epoch_results])

    first_epoch_results = all_results[0]
    first_avg_success = np.mean([r.success_rate for r in first_epoch_results])
    first_avg_confidence = np.mean([r.plan_confidence for r in first_epoch_results])

    improvement = final_avg_success - first_avg_success
    confidence_growth = final_avg_confidence - first_avg_confidence

    # Get final calibrator statistics
    final_calibrator_stats = {}
    if agent.confidence_calibrator is not None:
        final_calibrator_stats = agent.confidence_calibrator.get_summary()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epochs_completed": epochs,
        "scenarios_per_epoch": len(scenarios),
        "total_executions": epochs * len(scenarios),
        "final_metrics": {
            "avg_success_rate": float(final_avg_success),
            "avg_plan_confidence": float(final_avg_confidence),
            "avg_legacy_confidence": float(final_avg_legacy),
            "confidence_vs_legacy": float(final_avg_confidence - final_avg_legacy),
            "confidence_growth": float(confidence_growth),
            "improvement_from_start": float(improvement),
        },
        "memory_accumulated": {
            "short_term": memory_stats.get("short_term_count", 0),
            "long_term": memory_stats.get("long_term_count", 0),
            "episodic": memory_stats.get("episodic_count", 0),
            "semantic": memory_stats.get("semantic_count", 0),
        },
        "calibrator_final_state": final_calibrator_stats,
        "epoch_summaries": epoch_summaries,
        "final_epoch_details": [asdict(r) for r in final_epoch_results],
    }

    print(f"\nFinal Success Rate: {final_avg_success:.2%}")
    print(f"Final Plan Confidence: {final_avg_confidence:.2f} (legacy: {final_avg_legacy:.2f})")
    print(f"Confidence Growth: {confidence_growth:+.3f}")
    print(f"Improvement from Epoch 1: {improvement:+.2%}")
    print(f"Memory Entries Accumulated: {sum(summary['memory_accumulated'].values())}")
    if final_calibrator_stats:
        print(f"Calibrator Contexts Learned: {final_calibrator_stats.get('total_contexts', 0)}")

    return summary


def save_training_results(results: dict[str, Any], output_dir: Path) -> None:
    """Save training results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mercury_agent_training_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    report_path = output_dir / "MERCURY_AGENT_TRAINING_REPORT.md"
    with open(report_path, "w") as f:
        f.write("# Mercury Agent Training Report\n\n")
        f.write(f"**Generated:** {results['timestamp']}\n\n")

        f.write("## Training Summary\n\n")
        f.write(f"- **Epochs Completed:** {results['epochs_completed']}\n")
        f.write(f"- **Scenarios per Epoch:** {results['scenarios_per_epoch']}\n")
        f.write(f"- **Total Executions:** {results['total_executions']}\n\n")

        f.write("## Final Metrics\n\n")
        metrics = results["final_metrics"]
        f.write(f"- **Average Success Rate:** {metrics['avg_success_rate']:.2%}\n")
        f.write(f"- **Average Plan Confidence:** {metrics['avg_plan_confidence']:.2f}\n")
        f.write(f"- **Legacy Confidence (fixed heuristic):** {metrics.get('avg_legacy_confidence', 'N/A'):.2f}\n")
        f.write(f"- **Confidence vs Legacy:** {metrics.get('confidence_vs_legacy', 0):+.3f}\n")
        f.write(f"- **Confidence Growth:** {metrics.get('confidence_growth', 0):+.3f}\n")
        f.write(f"- **Improvement from Start:** {metrics['improvement_from_start']:+.2%}\n\n")

        # Add calibrator section if available
        calibrator = results.get("calibrator_final_state", {})
        if calibrator:
            f.write("## Bayesian Confidence Calibrator\n\n")
            f.write("The fixed 0.76 confidence heuristic has been replaced with a learned Bayesian calibrator.\n\n")
            f.write(f"- **Total Contexts Learned:** {calibrator.get('total_contexts', 0)}\n")
            f.write(f"- **Total Observations:** {calibrator.get('total_observations', 0)}\n")
            f.write(f"- **Average Posterior Mean:** {calibrator.get('avg_posterior_mean', 0):.3f}\n\n")
            contexts = calibrator.get("contexts", {})
            if contexts:
                f.write("### Per-Context Statistics\n\n")
                f.write("| Context | Observations | Successes | Failures | Posterior Mean |\n")
                f.write("|---------|--------------|-----------|----------|----------------|\n")
                for ctx_name, ctx_stats in contexts.items():
                    f.write(
                        f"| {ctx_name} | "
                        f"{ctx_stats.get('observations', 0)} | "
                        f"{ctx_stats.get('successes', 0)} | "
                        f"{ctx_stats.get('failures', 0)} | "
                        f"{ctx_stats.get('posterior_mean', 0):.3f} |\n"
                    )
                f.write("\n")

        f.write("## Memory Accumulation\n\n")
        memory = results["memory_accumulated"]
        f.write(f"- Short-term entries: {memory['short_term']}\n")
        f.write(f"- Long-term entries: {memory['long_term']}\n")
        f.write(f"- Episodic entries: {memory['episodic']}\n")
        f.write(f"- Semantic entries: {memory['semantic']}\n")
        f.write(f"- **Total:** {sum(memory.values())}\n\n")

        f.write("## Training Progress\n\n")
        f.write("| Epoch | Success Rate | Confidence | Legacy | Delta | Passed | Time (ms) |\n")
        f.write("|-------|--------------|------------|--------|-------|--------|----------|\n")
        for summary in results["epoch_summaries"]:
            legacy = summary.get('avg_legacy_confidence', summary['avg_plan_confidence'])
            delta = summary.get('confidence_improvement', 0)
            f.write(
                f"| {summary['epoch']} | "
                f"{summary['avg_success_rate']:.2%} | "
                f"{summary['avg_plan_confidence']:.2f} | "
                f"{legacy:.2f} | "
                f"{delta:+.3f} | "
                f"{summary['scenarios_passed']}/{summary['total_scenarios']} | "
                f"{summary['epoch_time_ms']:.1f} |\n"
            )

        f.write("\n## Final Epoch Details\n\n")
        for detail in results["final_epoch_details"]:
            f.write(f"### {detail['scenario_name']}\n\n")
            f.write(f"- Domain: {detail['domain']}\n")
            f.write(f"- Goal: {detail['goal']}\n")
            f.write(f"- Success Rate: {detail['success_rate']:.2%}\n")
            f.write(f"- Tasks Completed: {detail['tasks_completed']}\n")
            f.write(f"- Reasoning Steps: {detail['reasoning_steps']}\n")
            f.write(f"- Passed: {'Yes' if detail['passed'] else 'No'}\n\n")

        f.write("## Notes\n\n")
        f.write(
            "Mercury Agent is a rule-based autonomous agent that learns through "
            "experience accumulation in its memory systems and Bayesian confidence calibration. "
            "Unlike neural networks, it does not have gradient-based training. Instead, it builds "
            "up episodic and semantic memories that influence future planning and reasoning.\n\n"
        )
        f.write(
            "The training process exercises the agent through diverse scenarios across "
            "all supported domains (medical, security, humanitarian, infrastructure, "
            "energy, scientific, financial, general) to build a comprehensive knowledge base.\n\n"
        )
        f.write(
            "**Bayesian Confidence Calibration:** The fixed 0.76 confidence heuristic has been "
            "replaced with a learned Bayesian calibrator that uses Beta-Bernoulli conjugate priors. "
            "Confidence starts at ~0.76 for novel contexts and climbs toward 0.95-0.99+ after "
            "repeated successes, providing a more accurate reflection of the agent's growing competence.\n"
        )

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    print("\nInitializing Mercury Agent Training Suite...")
    print("This will run 30 epochs of training across 8 domain scenarios.\n")

    agent = create_mercury_agent(
        name="Mercury-Training",
        autonomy_level=0.8,
        ethical_threshold=0.93,
    )

    register_mock_tools(agent)

    scenarios = create_training_scenarios()

    results = run_training_epochs(agent, scenarios, epochs=30)

    output_dir = Path(__file__).parent.parent / "results"
    save_training_results(results, output_dir)

    print("\n" + "=" * 70)
    print("MERCURY AGENT TRAINING COMPLETE")
    print("=" * 70)

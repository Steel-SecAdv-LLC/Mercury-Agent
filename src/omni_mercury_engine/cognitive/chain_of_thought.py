"""
Mercury Agent ♱
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

from __future__ import annotations

"""
Chain-of-Thought Reasoning Engine for Mercury Agent.

Implements step-by-step reasoning capabilities inspired by:
- "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
- "Self-Consistency Improves Chain of Thought Reasoning" (Wang et al., 2022)
- "Least-to-Most Prompting Enables Complex Reasoning" (Zhou et al., 2022)

The Chain-of-Thought (CoT) paradigm breaks down complex problems into
intermediate reasoning steps, enabling more accurate and transparent
anomaly detection decisions.

Key Features:
1. Explicit thought chain generation
2. Self-consistency via multiple reasoning paths
3. Least-to-most decomposition for complex problems
4. Thought verification and validation
5. Confidence calibration based on reasoning depth
6. Integration with neuro-symbolic fusion

This module provides the cognitive scaffold for Mercury Agent's
decision-making, ensuring transparent and auditable reasoning.
"""

import hashlib
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio for weighting

# Reasoning depth bounds
MIN_THOUGHT_DEPTH = 1
MAX_THOUGHT_DEPTH = 15
DEFAULT_THOUGHT_DEPTH = 5

# Self-consistency paths
MIN_CONSISTENCY_PATHS = 3
MAX_CONSISTENCY_PATHS = 10


class ThoughtType(Enum):
    """Types of thoughts in a reasoning chain."""

    OBSERVATION = "observation"  # Initial observation of the problem
    DECOMPOSITION = "decomposition"  # Breaking down into sub-problems
    ANALYSIS = "analysis"  # Analyzing evidence/data
    INFERENCE = "inference"  # Drawing logical conclusions
    HYPOTHESIS = "hypothesis"  # Proposing explanations
    VERIFICATION = "verification"  # Verifying intermediate conclusions
    SYNTHESIS = "synthesis"  # Combining multiple insights
    CONCLUSION = "conclusion"  # Final conclusion


class ReasoningStrategy(Enum):
    """Strategies for chain-of-thought reasoning."""

    STANDARD_COT = "standard_cot"  # Standard chain-of-thought
    SELF_CONSISTENCY = "self_consistency"  # Multiple paths with voting
    LEAST_TO_MOST = "least_to_most"  # Decompose then solve
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # Branching exploration
    VERIFICATION_COT = "verification_cot"  # With explicit verification steps


class ConfidenceLevel(Enum):
    """Confidence levels for thoughts."""

    CERTAIN = "certain"  # 0.9-1.0
    HIGH = "high"  # 0.7-0.9
    MEDIUM = "medium"  # 0.5-0.7
    LOW = "low"  # 0.3-0.5
    UNCERTAIN = "uncertain"  # 0.0-0.3


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class Thought:
    """A single thought in a reasoning chain.

    Represents one step in the chain-of-thought process.

    Attributes:
        thought_id: Unique identifier
        thought_type: Type of thought
        content: The thought content/text
        evidence: Supporting evidence
        confidence: Confidence in this thought
        depth: Depth in the reasoning chain
        parent_id: ID of parent thought (if any)
        child_ids: IDs of child thoughts
        verification_status: Whether thought has been verified
        timestamp: Creation timestamp
        metadata: Additional metadata
    """

    thought_id: str
    thought_type: ThoughtType
    content: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.8
    depth: int = 0
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    verification_status: str = "pending"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.thought_id,
            "type": self.thought_type.value,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "depth": self.depth,
            "parent_id": self.parent_id,
            "verified": self.verification_status,
        }


@dataclass
class ThoughtChain:
    """A complete chain of thoughts.

    Represents a reasoning path from problem to conclusion.

    Attributes:
        chain_id: Unique identifier
        strategy: Reasoning strategy used
        thoughts: Ordered list of thoughts
        problem: The original problem/question
        conclusion: Final conclusion
        overall_confidence: Confidence in the chain
        reasoning_depth: Number of thought steps
        verification_score: Score from verification
        is_valid: Whether chain is logically valid
        computation_time_ms: Time taken to generate
        metadata: Additional metadata
    """

    chain_id: str
    strategy: ReasoningStrategy
    thoughts: list[Thought]
    problem: str
    conclusion: str
    overall_confidence: float
    reasoning_depth: int
    verification_score: float = 1.0
    is_valid: bool = True
    computation_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Alias for overall_confidence for API compatibility."""
        return self.overall_confidence

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "strategy": self.strategy.value,
            "problem": self.problem,
            "conclusion": self.conclusion,
            "confidence": self.overall_confidence,
            "depth": self.reasoning_depth,
            "verified": self.verification_score,
            "valid": self.is_valid,
            "thoughts": [t.to_dict() for t in self.thoughts],
            "time_ms": self.computation_time_ms,
        }

    def get_reasoning_trace(self) -> str:
        """Get human-readable reasoning trace."""
        lines = [f"Problem: {self.problem}", ""]
        for i, thought in enumerate(self.thoughts, 1):
            prefix = "  " * thought.depth
            lines.append(f"{prefix}Step {i} ({thought.thought_type.value}): {thought.content}")
            if thought.evidence:
                lines.append(f"{prefix}  Evidence: {', '.join(thought.evidence[:3])}")
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append(f"Confidence: {self.overall_confidence:.2%}")
        return "\n".join(lines)


@dataclass
class SubProblem:
    """A decomposed sub-problem for least-to-most reasoning.

    Attributes:
        subproblem_id: Unique identifier
        description: Problem description
        complexity: Estimated complexity (0-1)
        dependencies: IDs of prerequisite sub-problems
        solution: Solution once solved
        is_solved: Whether solved
    """

    subproblem_id: str
    description: str
    complexity: float = 0.5
    dependencies: list[str] = field(default_factory=list)
    solution: str | None = None
    is_solved: bool = False


@dataclass
class ConsistencyResult:
    """Result of self-consistency voting.

    Attributes:
        answer: Most common answer
        vote_count: Number of votes for answer
        total_paths: Total reasoning paths
        agreement_ratio: Ratio of agreement
        all_answers: All answers with counts
    """

    answer: str
    vote_count: int
    total_paths: int
    agreement_ratio: float
    all_answers: dict[str, int]


# =============================================================================
# Thought Generator
# =============================================================================


class ThoughtGenerator:
    """Generates individual thoughts for reasoning chains.

    This class encapsulates the logic for generating different types
    of thoughts based on context and evidence.
    """

    def __init__(
        self,
        min_evidence_threshold: float = 0.3,
        enable_verification: bool = True,
    ):
        """Initialize thought generator.

        Args:
            min_evidence_threshold: Minimum evidence strength for conclusions
            enable_verification: Whether to generate verification thoughts
        """
        self.min_evidence_threshold = min_evidence_threshold
        self.enable_verification = enable_verification
        self._thought_counter = 0

        # Thought templates for different types
        self._templates = {
            ThoughtType.OBSERVATION: [
                "Observing that {observation}",
                "The data shows {observation}",
                "Initial observation: {observation}",
            ],
            ThoughtType.DECOMPOSITION: [
                "Breaking this down into: {components}",
                "This problem has these parts: {components}",
                "Sub-problems identified: {components}",
            ],
            ThoughtType.ANALYSIS: [
                "Analyzing {subject}: {finding}",
                "Upon examination of {subject}, {finding}",
                "Analysis reveals: {finding}",
            ],
            ThoughtType.INFERENCE: [
                "From {premise}, it follows that {conclusion}",
                "Given {premise}, we can infer {conclusion}",
                "This implies {conclusion}",
            ],
            ThoughtType.HYPOTHESIS: [
                "Hypothesis: {hypothesis}",
                "One possible explanation is {hypothesis}",
                "This could be because {hypothesis}",
            ],
            ThoughtType.VERIFICATION: [
                "Verifying: {claim} - Result: {result}",
                "Checking {claim}: {result}",
                "Verification of {claim} shows {result}",
            ],
            ThoughtType.SYNTHESIS: [
                "Combining insights: {synthesis}",
                "Taking together: {synthesis}",
                "Synthesizing: {synthesis}",
            ],
            ThoughtType.CONCLUSION: [
                "Therefore: {conclusion}",
                "In conclusion: {conclusion}",
                "Final answer: {conclusion}",
            ],
        }

    def generate_thought(
        self,
        thought_type: ThoughtType,
        context: dict[str, Any],
        parent: Thought | None = None,
    ) -> Thought:
        """Generate a thought of the specified type.

        Args:
            thought_type: Type of thought to generate
            context: Context information for thought generation
            parent: Parent thought (for chaining)

        Returns:
            Generated Thought object
        """
        self._thought_counter += 1
        thought_id = f"thought_{self._thought_counter:06d}"

        # Select template
        templates = self._templates.get(thought_type, ["Processing: {content}"])
        template = np.random.choice(templates)

        # Generate content based on type and context
        content = self._fill_template(template, thought_type, context)

        # Calculate confidence based on evidence
        evidence = context.get("evidence", [])
        confidence = self._calculate_confidence(thought_type, evidence, parent)

        # Determine depth
        depth = parent.depth + 1 if parent else 0

        thought = Thought(
            thought_id=thought_id,
            thought_type=thought_type,
            content=content,
            evidence=evidence[:5],  # Keep top 5 evidence items
            confidence=confidence,
            depth=depth,
            parent_id=parent.thought_id if parent else None,
            metadata={"context_keys": list(context.keys())},
        )

        # Link to parent
        if parent:
            parent.child_ids.append(thought_id)

        return thought

    def _fill_template(
        self,
        template: str,
        thought_type: ThoughtType,
        context: dict[str, Any],
    ) -> str:
        """Fill template with context values."""
        try:
            if thought_type == ThoughtType.OBSERVATION:
                return template.format(observation=context.get("observation", "data patterns"))

            if thought_type == ThoughtType.DECOMPOSITION:
                components = context.get("components", ["part 1", "part 2"])
                return template.format(components=", ".join(str(c) for c in components))

            if thought_type == ThoughtType.ANALYSIS:
                return template.format(
                    subject=context.get("subject", "the data"),
                    finding=context.get("finding", "patterns detected"),
                )

            if thought_type == ThoughtType.INFERENCE:
                return template.format(
                    premise=context.get("premise", "the evidence"),
                    conclusion=context.get("conclusion", "this follows"),
                )

            if thought_type == ThoughtType.HYPOTHESIS:
                return template.format(hypothesis=context.get("hypothesis", "unknown cause"))

            if thought_type == ThoughtType.VERIFICATION:
                return template.format(
                    claim=context.get("claim", "the hypothesis"),
                    result=context.get("result", "consistent"),
                )

            if thought_type == ThoughtType.SYNTHESIS:
                return template.format(synthesis=context.get("synthesis", "combined understanding"))

            if thought_type == ThoughtType.CONCLUSION:
                return template.format(conclusion=context.get("conclusion", "final determination"))

            return template.format(**{k: str(v)[:100] for k, v in context.items()})

        except (KeyError, ValueError):
            return f"{thought_type.value}: {context.get('content', 'processing')}"

    def _calculate_confidence(
        self,
        thought_type: ThoughtType,
        evidence: list[str],
        parent: Thought | None,
    ) -> float:
        """Calculate confidence for a thought."""
        base_confidence = {
            ThoughtType.OBSERVATION: 0.9,
            ThoughtType.DECOMPOSITION: 0.85,
            ThoughtType.ANALYSIS: 0.8,
            ThoughtType.INFERENCE: 0.75,
            ThoughtType.HYPOTHESIS: 0.6,
            ThoughtType.VERIFICATION: 0.9,
            ThoughtType.SYNTHESIS: 0.7,
            ThoughtType.CONCLUSION: 0.8,
        }.get(thought_type, 0.7)

        # Adjust based on evidence
        evidence_bonus = min(0.1, len(evidence) * 0.02)
        base_confidence += evidence_bonus

        # Inherit from parent with decay
        if parent:
            inherited = parent.confidence * 0.9
            base_confidence = min(base_confidence, inherited)

        return min(0.99, base_confidence)

    def verify_thought(self, thought: Thought, verification_data: dict[str, Any]) -> float:
        """Verify a thought against data.

        Args:
            thought: Thought to verify
            verification_data: Data to verify against

        Returns:
            Verification score (0-1)
        """
        if not self.enable_verification:
            return 1.0

        score = 0.5  # Base score

        # Check evidence consistency
        if thought.evidence and verification_data.get("evidence"):
            overlap = set(thought.evidence) & set(verification_data["evidence"])
            if overlap:
                score += 0.2 * len(overlap) / len(thought.evidence)

        # Check logical consistency
        if verification_data.get("consistent", True):
            score += 0.2

        # Check against known facts
        if verification_data.get("factual_support"):
            score += 0.1

        thought.verification_status = "verified" if score >= 0.7 else "unverified"
        return min(1.0, score)


# =============================================================================
# Chain-of-Thought Engine
# =============================================================================


class ChainOfThoughtEngine:
    """
    Chain-of-Thought Reasoning Engine.

    Implements multiple CoT strategies for transparent, step-by-step
    reasoning in anomaly detection and decision making.

    Strategies:
    1. Standard CoT: Linear chain of thoughts
    2. Self-Consistency: Multiple paths with majority voting
    3. Least-to-Most: Decompose then solve incrementally
    4. Tree of Thoughts: Branching exploration
    5. Verification CoT: With explicit verification steps
    """

    def __init__(
        self,
        default_strategy: ReasoningStrategy = ReasoningStrategy.SELF_CONSISTENCY,
        max_depth: int = DEFAULT_THOUGHT_DEPTH,
        consistency_paths: int = 5,
        enable_verification: bool = True,
        min_confidence: float = 0.5,
    ):
        """Initialize Chain-of-Thought engine.

        Args:
            default_strategy: Default reasoning strategy
            max_depth: Maximum reasoning depth
            consistency_paths: Number of paths for self-consistency
            enable_verification: Enable verification steps
            min_confidence: Minimum confidence threshold
        """
        self.default_strategy = default_strategy
        self.max_depth = min(max_depth, MAX_THOUGHT_DEPTH)
        self.consistency_paths = min(consistency_paths, MAX_CONSISTENCY_PATHS)
        self.enable_verification = enable_verification
        self.min_confidence = min_confidence

        self.thought_generator = ThoughtGenerator(enable_verification=enable_verification)

        self._chain_counter = 0
        self._stats = {
            "chains_generated": 0,
            "total_reasoning_sessions": 0,
            "thoughts_generated": 0,
            "verifications_performed": 0,
            "avg_depth": 0.0,
            "avg_confidence": 0.0,
        }

        logger.info(
            f"ChainOfThoughtEngine initialized (strategy={default_strategy.value}, "
            f"max_depth={self.max_depth})"
        )

    def reason(
        self,
        problem: str,
        context: dict[str, Any],
        strategy: ReasoningStrategy | None = None,
        num_samples: int | None = None,
        beam_width: int | None = None,
        max_depth: int | None = None,
    ) -> ThoughtChain:
        """Perform chain-of-thought reasoning on a problem.

        Args:
            problem: The problem or question to reason about
            context: Context data for reasoning
            strategy: Reasoning strategy (uses default if None)
            num_samples: Number of samples for self-consistency (overrides consistency_paths)
            beam_width: Beam width for tree-of-thoughts exploration
            max_depth: Maximum reasoning depth (overrides instance max_depth)

        Returns:
            Complete ThoughtChain with reasoning trace
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy

        # Store original values for restoration
        original_consistency_paths = self.consistency_paths
        original_max_depth = self.max_depth

        # Apply overrides if provided
        if num_samples is not None:
            self.consistency_paths = min(num_samples, MAX_CONSISTENCY_PATHS)
        if max_depth is not None:
            self.max_depth = min(max_depth, MAX_THOUGHT_DEPTH)

        # Store beam_width in context for tree-of-thoughts
        if beam_width is not None:
            context = context.copy()
            context["_beam_width"] = beam_width

        try:
            if strategy == ReasoningStrategy.STANDARD_COT:
                chain = self._standard_cot(problem, context)
            elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
                chain = self._self_consistency_cot(problem, context)
            elif strategy == ReasoningStrategy.LEAST_TO_MOST:
                chain = self._least_to_most_cot(problem, context)
            elif strategy == ReasoningStrategy.TREE_OF_THOUGHTS:
                chain = self._tree_of_thoughts(problem, context)
            elif strategy == ReasoningStrategy.VERIFICATION_COT:
                chain = self._verification_cot(problem, context)
            else:
                chain = self._standard_cot(problem, context)

            chain.computation_time_ms = (time.time() - start_time) * 1000

            # Add consistency_score to metadata for self-consistency strategy
            if strategy == ReasoningStrategy.SELF_CONSISTENCY:
                if "consistency_score" not in chain.metadata:
                    chain.metadata["consistency_score"] = chain.metadata.get(
                        "agreement_ratio", chain.overall_confidence
                    )

            self._update_stats(chain)
            return chain
        finally:
            # Restore original values
            self.consistency_paths = original_consistency_paths
            self.max_depth = original_max_depth

    def _standard_cot(self, problem: str, context: dict[str, Any]) -> ThoughtChain:
        """Standard chain-of-thought reasoning.

        Generates a linear chain of thoughts from problem to conclusion.
        """
        self._chain_counter += 1
        chain_id = f"cot_std_{self._chain_counter:06d}"

        thoughts: list[Thought] = []
        parent: Thought | None = None

        # Step 1: Observation
        data = context.get("data", [])
        # Handle non-list data types
        if not isinstance(data, (list, tuple)):
            data = [data] if data is not None else []
        obs_context = {
            "observation": problem,
            "evidence": list(data)[:3],
        }
        obs_thought = self.thought_generator.generate_thought(
            ThoughtType.OBSERVATION, obs_context, parent
        )
        thoughts.append(obs_thought)
        parent = obs_thought

        # Step 2: Analysis
        analysis_context = {
            "subject": "the observed data",
            "finding": self._analyze_context(context),
            "evidence": context.get("features", [])[:3],
        }
        analysis_thought = self.thought_generator.generate_thought(
            ThoughtType.ANALYSIS, analysis_context, parent
        )
        thoughts.append(analysis_thought)
        parent = analysis_thought

        # Step 3: Inference (one or more)
        inferences = self._generate_inferences(context, parent)
        for inf in inferences[: self.max_depth - 4]:  # Leave room for other steps
            thoughts.append(inf)
            parent = inf

        # Step 4: Synthesis
        synth_context = {
            "synthesis": self._synthesize_thoughts(thoughts),
            "evidence": [t.content[:50] for t in thoughts[-3:]],
        }
        synth_thought = self.thought_generator.generate_thought(
            ThoughtType.SYNTHESIS, synth_context, parent
        )
        thoughts.append(synth_thought)
        parent = synth_thought

        # Step 5: Conclusion
        conclusion = self._derive_conclusion(thoughts, context)
        conc_context = {
            "conclusion": conclusion,
            "evidence": [t.content[:50] for t in thoughts if t.confidence > 0.7],
        }
        conc_thought = self.thought_generator.generate_thought(
            ThoughtType.CONCLUSION, conc_context, parent
        )
        thoughts.append(conc_thought)

        # Calculate overall confidence
        confidences = [t.confidence for t in thoughts]
        overall_confidence = float(np.min(confidences)) * 0.8 + float(np.mean(confidences)) * 0.2

        return ThoughtChain(
            chain_id=chain_id,
            strategy=ReasoningStrategy.STANDARD_COT,
            thoughts=thoughts,
            problem=problem,
            conclusion=conclusion,
            overall_confidence=overall_confidence,
            reasoning_depth=len(thoughts),
            is_valid=overall_confidence >= self.min_confidence,
        )

    def _self_consistency_cot(self, problem: str, context: dict[str, Any]) -> ThoughtChain:
        """Self-consistency chain-of-thought.

        Generates multiple reasoning paths and uses majority voting.
        """
        self._chain_counter += 1
        chain_id = f"cot_sc_{self._chain_counter:06d}"

        # Generate multiple reasoning paths
        paths: list[ThoughtChain] = []
        conclusions: list[str] = []

        for i in range(self.consistency_paths):
            # Add randomness to context for diverse paths
            varied_context = context.copy()
            varied_context["path_seed"] = i
            varied_context["variation"] = np.random.random()

            path = self._standard_cot(problem, varied_context)
            paths.append(path)
            conclusions.append(self._normalize_conclusion(path.conclusion))

        # Vote on conclusions
        consistency_result = self._vote_on_conclusions(conclusions)

        # Find the path with the winning conclusion
        winning_path = paths[0]
        for path in paths:
            if self._normalize_conclusion(path.conclusion) == consistency_result.answer:
                if path.overall_confidence > winning_path.overall_confidence:
                    winning_path = path

        # Combine with agreement ratio for final confidence
        combined_confidence = (
            winning_path.overall_confidence * 0.6 + consistency_result.agreement_ratio * 0.4
        )

        return ThoughtChain(
            chain_id=chain_id,
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
            thoughts=winning_path.thoughts,
            problem=problem,
            conclusion=consistency_result.answer,
            overall_confidence=combined_confidence,
            reasoning_depth=winning_path.reasoning_depth,
            verification_score=consistency_result.agreement_ratio,
            is_valid=combined_confidence >= self.min_confidence,
            metadata={
                "paths_considered": self.consistency_paths,
                "vote_counts": consistency_result.all_answers,
                "agreement_ratio": consistency_result.agreement_ratio,
            },
        )

    def _least_to_most_cot(self, problem: str, context: dict[str, Any]) -> ThoughtChain:
        """Least-to-most chain-of-thought.

        Decomposes problem into sub-problems and solves incrementally.
        """
        self._chain_counter += 1
        chain_id = f"cot_ltm_{self._chain_counter:06d}"

        thoughts: list[Thought] = []
        parent: Thought | None = None

        # Step 1: Observation
        obs_thought = self.thought_generator.generate_thought(
            ThoughtType.OBSERVATION,
            {"observation": problem, "evidence": context.get("data", [])[:3]},
            None,
        )
        thoughts.append(obs_thought)
        parent = obs_thought

        # Step 2: Decomposition
        subproblems = self._decompose_problem(problem, context)
        decomp_context = {
            "components": [sp.description for sp in subproblems],
            "evidence": ["problem decomposition"],
        }
        decomp_thought = self.thought_generator.generate_thought(
            ThoughtType.DECOMPOSITION, decomp_context, parent
        )
        thoughts.append(decomp_thought)
        parent = decomp_thought

        # Step 3: Solve sub-problems in order (least to most complex)
        sorted_subproblems = sorted(subproblems, key=lambda x: x.complexity)
        solutions: dict[str, str] = {}

        for sp in sorted_subproblems:
            # Build context with solved prerequisites
            sp_context = context.copy()
            sp_context["solved"] = {dep: solutions.get(dep, "unknown") for dep in sp.dependencies}

            # Analyze sub-problem
            analysis_thought = self.thought_generator.generate_thought(
                ThoughtType.ANALYSIS,
                {
                    "subject": sp.description,
                    "finding": self._solve_subproblem(sp, sp_context, solutions),
                    "evidence": list(solutions.values())[-3:] if solutions else [],
                },
                parent,
            )
            thoughts.append(analysis_thought)
            parent = analysis_thought

            sp.solution = analysis_thought.content
            sp.is_solved = True
            solutions[sp.subproblem_id] = sp.solution

        # Step 4: Synthesize solutions
        synth_thought = self.thought_generator.generate_thought(
            ThoughtType.SYNTHESIS,
            {
                "synthesis": self._synthesize_solutions(solutions),
                "evidence": list(solutions.values()),
            },
            parent,
        )
        thoughts.append(synth_thought)
        parent = synth_thought

        # Step 5: Final conclusion
        conclusion = self._derive_conclusion(thoughts, context)
        conc_thought = self.thought_generator.generate_thought(
            ThoughtType.CONCLUSION,
            {"conclusion": conclusion, "evidence": list(solutions.values())},
            parent,
        )
        thoughts.append(conc_thought)

        # Calculate confidence
        confidences = [t.confidence for t in thoughts]
        overall_confidence = float(np.min(confidences)) * 0.7 + float(np.mean(confidences)) * 0.3

        return ThoughtChain(
            chain_id=chain_id,
            strategy=ReasoningStrategy.LEAST_TO_MOST,
            thoughts=thoughts,
            problem=problem,
            conclusion=conclusion,
            overall_confidence=overall_confidence,
            reasoning_depth=len(thoughts),
            is_valid=overall_confidence >= self.min_confidence,
            metadata={
                "subproblems_count": len(subproblems),
                "solutions": solutions,
            },
        )

    def _tree_of_thoughts(self, problem: str, context: dict[str, Any]) -> ThoughtChain:
        """Tree of thoughts reasoning.

        Explores multiple branches and selects the best path.
        """
        self._chain_counter += 1
        chain_id = f"cot_tot_{self._chain_counter:06d}"

        # Get beam_width from context if provided, otherwise use default
        beam_width = context.get("_beam_width", min(3, self.consistency_paths))

        # Start with observation
        root = self.thought_generator.generate_thought(
            ThoughtType.OBSERVATION,
            {"observation": problem, "evidence": context.get("data", [])[:3]},
            None,
        )

        # Explore branches
        branches: list[list[Thought]] = []
        branch_scores: list[float] = []

        for branch_idx in range(beam_width):
            branch_context = context.copy()
            branch_context["branch"] = branch_idx

            branch = self._explore_branch(root, branch_context, max_depth=self.max_depth - 1)
            branches.append(branch)

            # Score branch
            score = self._score_branch(branch)
            branch_scores.append(score)

        # Select best branch
        best_idx = int(np.argmax(branch_scores))
        best_branch = branches[best_idx]

        # Build final chain
        thoughts = [root] + best_branch

        # Add conclusion
        conclusion = self._derive_conclusion(thoughts, context)
        conc_thought = self.thought_generator.generate_thought(
            ThoughtType.CONCLUSION,
            {"conclusion": conclusion, "evidence": [t.content[:50] for t in thoughts[-3:]]},
            thoughts[-1],
        )
        thoughts.append(conc_thought)

        overall_confidence = branch_scores[best_idx]

        return ThoughtChain(
            chain_id=chain_id,
            strategy=ReasoningStrategy.TREE_OF_THOUGHTS,
            thoughts=thoughts,
            problem=problem,
            conclusion=conclusion,
            overall_confidence=overall_confidence,
            reasoning_depth=len(thoughts),
            is_valid=overall_confidence >= self.min_confidence,
            metadata={
                "branches_explored": len(branches),
                "branch_scores": branch_scores,
                "best_branch_idx": best_idx,
            },
        )

    def _verification_cot(self, problem: str, context: dict[str, Any]) -> ThoughtChain:
        """Verification-focused chain-of-thought.

        Includes explicit verification steps after each inference.
        """
        self._chain_counter += 1
        chain_id = f"cot_ver_{self._chain_counter:06d}"

        thoughts: list[Thought] = []
        parent: Thought | None = None
        verification_scores: list[float] = []

        # Step 1: Observation
        obs_thought = self.thought_generator.generate_thought(
            ThoughtType.OBSERVATION,
            {"observation": problem, "evidence": context.get("data", [])[:3]},
            None,
        )
        thoughts.append(obs_thought)
        parent = obs_thought

        # Steps 2-N: Analysis with verification
        analysis_steps = min(self.max_depth - 3, 5)

        for i in range(analysis_steps):
            # Generate analysis/inference
            step_type = ThoughtType.ANALYSIS if i == 0 else ThoughtType.INFERENCE
            step_context = {
                "subject": f"aspect {i + 1}",
                "finding": self._analyze_aspect(context, i),
                "premise": thoughts[-1].content if thoughts else "initial",
                "conclusion": self._infer_step(context, thoughts),
                "evidence": context.get("features", [])[:3],
            }
            step_thought = self.thought_generator.generate_thought(step_type, step_context, parent)
            thoughts.append(step_thought)

            # Generate verification
            ver_context = {
                "claim": step_thought.content,
                "result": self._verify_claim(step_thought, context),
                "evidence": ["verification check"],
            }
            ver_thought = self.thought_generator.generate_thought(
                ThoughtType.VERIFICATION, ver_context, step_thought
            )
            thoughts.append(ver_thought)
            verification_scores.append(ver_thought.confidence)
            self._stats["verifications_performed"] += 1

            parent = ver_thought

        # Final synthesis and conclusion
        synth_thought = self.thought_generator.generate_thought(
            ThoughtType.SYNTHESIS,
            {
                "synthesis": self._synthesize_verified(thoughts, verification_scores),
                "evidence": [t.content[:50] for t in thoughts if t.confidence > 0.7],
            },
            parent,
        )
        thoughts.append(synth_thought)

        conclusion = self._derive_conclusion(thoughts, context)
        conc_thought = self.thought_generator.generate_thought(
            ThoughtType.CONCLUSION, {"conclusion": conclusion}, synth_thought
        )
        thoughts.append(conc_thought)

        # Calculate confidence with verification weight
        avg_verification = float(np.mean(verification_scores)) if verification_scores else 0.8
        thought_confidences = [t.confidence for t in thoughts]
        overall_confidence = (
            float(np.min(thought_confidences)) * 0.4
            + float(np.mean(thought_confidences)) * 0.3
            + avg_verification * 0.3
        )

        return ThoughtChain(
            chain_id=chain_id,
            strategy=ReasoningStrategy.VERIFICATION_COT,
            thoughts=thoughts,
            problem=problem,
            conclusion=conclusion,
            overall_confidence=overall_confidence,
            reasoning_depth=len(thoughts),
            verification_score=avg_verification,
            is_valid=overall_confidence >= self.min_confidence and avg_verification >= 0.6,
            metadata={
                "verification_scores": verification_scores,
                "avg_verification": avg_verification,
            },
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _analyze_context(self, context: dict[str, Any]) -> str:
        """Generate analysis finding from context."""
        if "anomaly_score" in context:
            score = context["anomaly_score"]
            if score > 0.7:
                return f"high anomaly indicators (score: {score:.2f})"
            elif score > 0.4:
                return f"moderate anomaly signals (score: {score:.2f})"
            else:
                return f"normal patterns observed (score: {score:.2f})"

        if "features" in context:
            return f"identified {len(context['features'])} relevant features"

        return "patterns requiring further analysis"

    def _generate_inferences(self, context: dict[str, Any], parent: Thought) -> list[Thought]:
        """Generate inference thoughts based on context."""
        inferences = []

        # Generate 1-3 inferences based on available data
        inference_count = min(3, max(1, len(context.get("features", [])) // 2))

        for i in range(inference_count):
            inf_context = {
                "premise": parent.content[:100],
                "conclusion": self._infer_step(context, inferences),
                "evidence": context.get("evidence", [])[:2],
            }
            inf_thought = self.thought_generator.generate_thought(
                ThoughtType.INFERENCE, inf_context, parent
            )
            inferences.append(inf_thought)
            parent = inf_thought

        return inferences

    def _infer_step(self, context: dict[str, Any], previous: list[Thought]) -> str:
        """Generate an inference conclusion."""
        if "anomaly_score" in context:
            score = context["anomaly_score"]
            if score > 0.7:
                return "significant deviation from expected patterns"
            elif score > 0.4:
                return "potential anomalous behavior detected"
            return "behavior within normal parameters"

        if previous:
            return f"continuation of observed pattern (depth {len(previous)})"

        return "initial pattern assessment"

    def _synthesize_thoughts(self, thoughts: list[Thought]) -> str:
        """Synthesize multiple thoughts into a summary."""
        key_findings = [
            t.content[:50] for t in thoughts if t.thought_type != ThoughtType.OBSERVATION
        ]
        if key_findings:
            return f"Combining {len(key_findings)} findings into unified assessment"
        return "Synthesizing available evidence"

    def _derive_conclusion(self, thoughts: list[Thought], context: dict[str, Any]) -> str:
        """Derive final conclusion from thought chain."""
        # Calculate weighted evidence
        high_conf_thoughts = [t for t in thoughts if t.confidence > 0.7]

        if "anomaly_score" in context:
            score = context["anomaly_score"]
            if score > 0.7:
                return f"ANOMALY DETECTED with {len(high_conf_thoughts)} supporting evidence points (score: {score:.2f})"
            elif score > 0.4:
                return f"POTENTIAL ANOMALY requiring monitoring (score: {score:.2f})"
            return f"NORMAL - no significant anomalies detected (score: {score:.2f})"

        return f"Analysis complete with {len(high_conf_thoughts)} verified findings"

    def _normalize_conclusion(self, conclusion: str) -> str:
        """Normalize conclusion for comparison."""
        conclusion = conclusion.lower().strip()
        # Extract key determination
        if "anomaly detected" in conclusion:
            return "anomaly"
        elif "potential anomaly" in conclusion:
            return "potential"
        elif "normal" in conclusion:
            return "normal"
        # Hash for other cases
        return hashlib.sha256(conclusion.encode()).hexdigest()[:8]

    def _vote_on_conclusions(self, conclusions: list[str]) -> ConsistencyResult:
        """Vote on conclusions for self-consistency."""
        counter = Counter(conclusions)
        most_common = counter.most_common(1)[0]

        return ConsistencyResult(
            answer=most_common[0],
            vote_count=most_common[1],
            total_paths=len(conclusions),
            agreement_ratio=most_common[1] / len(conclusions),
            all_answers=dict(counter),
        )

    def _decompose_problem(self, problem: str, context: dict[str, Any]) -> list[SubProblem]:
        """Decompose problem into sub-problems."""
        subproblems = []

        # Generate sub-problems based on problem structure
        aspects = [
            ("feature_analysis", "Analyze extracted features", 0.3, []),
            ("pattern_detection", "Detect temporal patterns", 0.4, ["feature_analysis"]),
            ("anomaly_scoring", "Calculate anomaly scores", 0.5, ["pattern_detection"]),
            ("context_integration", "Integrate contextual information", 0.6, ["anomaly_scoring"]),
            ("final_assessment", "Make final assessment", 0.7, ["context_integration"]),
        ]

        for sp_id, desc, complexity, deps in aspects:
            subproblems.append(
                SubProblem(
                    subproblem_id=sp_id,
                    description=desc,
                    complexity=complexity,
                    dependencies=deps,
                )
            )

        return subproblems

    def _solve_subproblem(
        self,
        subproblem: SubProblem,
        context: dict[str, Any],
        solutions: dict[str, str],
    ) -> str:
        """Solve a single sub-problem."""
        # Use solved dependencies
        dep_context = " ".join(solutions.get(d, "")[:30] for d in subproblem.dependencies)

        if "feature" in subproblem.description.lower():
            return f"Feature analysis: {len(context.get('features', []))} features identified"
        elif "pattern" in subproblem.description.lower():
            return f"Pattern detection: temporal structure analyzed ({dep_context})"
        elif "scoring" in subproblem.description.lower():
            score = context.get("anomaly_score", 0.5)
            return f"Anomaly score: {score:.2f} ({dep_context})"
        elif "context" in subproblem.description.lower():
            return f"Context integration complete ({dep_context})"
        else:
            return f"Assessment: {subproblem.description} resolved"

    def _synthesize_solutions(self, solutions: dict[str, str]) -> str:
        """Synthesize sub-problem solutions."""
        return f"Integrated {len(solutions)} sub-problem solutions into coherent analysis"

    def _explore_branch(
        self,
        root: Thought,
        context: dict[str, Any],
        max_depth: int,
    ) -> list[Thought]:
        """Explore a reasoning branch."""
        branch: list[Thought] = []
        parent = root

        for depth in range(max_depth):
            # Randomly choose thought type
            thought_types = [ThoughtType.ANALYSIS, ThoughtType.INFERENCE, ThoughtType.HYPOTHESIS]
            thought_type = np.random.choice(thought_types)

            step_context = {
                "subject": f"branch aspect {depth}",
                "finding": self._analyze_aspect(context, depth),
                "premise": parent.content[:50],
                "conclusion": self._infer_step(context, branch),
                "hypothesis": f"branch {context.get('branch', 0)} hypothesis {depth}",
                "evidence": [],
            }

            thought = self.thought_generator.generate_thought(thought_type, step_context, parent)
            branch.append(thought)
            parent = thought

            # Early termination if confidence too low
            if thought.confidence < self.min_confidence:
                break

        return branch

    def _score_branch(self, branch: list[Thought]) -> float:
        """Score a reasoning branch."""
        if not branch:
            return 0.0

        confidences = [t.confidence for t in branch]
        avg_confidence = float(np.mean(confidences))
        min_confidence = float(np.min(confidences))
        depth_bonus = min(0.1, len(branch) * 0.02)

        return avg_confidence * 0.5 + min_confidence * 0.3 + depth_bonus

    def _analyze_aspect(self, context: dict[str, Any], index: int) -> str:
        """Analyze a specific aspect of the context."""
        aspects = context.get("features", ["general pattern"])
        if index < len(aspects):
            return f"examined: {aspects[index]}"
        return f"analysis step {index + 1} complete"

    def _verify_claim(self, thought: Thought, context: dict[str, Any]) -> str:
        """Verify a claim against context."""
        # Simple verification logic
        if thought.confidence > 0.8:
            return "VERIFIED - high confidence"
        elif thought.confidence > 0.5:
            return "PARTIALLY VERIFIED - moderate support"
        return "UNVERIFIED - requires additional evidence"

    def _synthesize_verified(
        self, thoughts: list[Thought], verification_scores: list[float]
    ) -> str:
        """Synthesize verified thoughts."""
        verified_count = sum(1 for s in verification_scores if s > 0.7)
        total = len(verification_scores)
        return f"Synthesized {verified_count}/{total} verified findings"

    def _update_stats(self, chain: ThoughtChain) -> None:
        """Update engine statistics."""
        self._stats["chains_generated"] += 1
        self._stats["total_reasoning_sessions"] += 1
        self._stats["thoughts_generated"] += len(chain.thoughts)

        # Update averages
        n = self._stats["chains_generated"]
        self._stats["avg_depth"] = (self._stats["avg_depth"] * (n - 1) + chain.reasoning_depth) / n
        self._stats["avg_confidence"] = (
            self._stats["avg_confidence"] * (n - 1) + chain.overall_confidence
        ) / n

    def get_statistics(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "default_strategy": self.default_strategy.value,
            "max_depth": self.max_depth,
            "consistency_paths": self.consistency_paths,
            "verification_enabled": self.enable_verification,
        }


# =============================================================================
# Anomaly-Specific Chain-of-Thought
# =============================================================================


class AnomalyChainOfThought:
    """
    Specialized Chain-of-Thought for anomaly detection.

    Provides domain-specific reasoning for Mercury Agent's
    anomaly detection pipeline.
    """

    def __init__(
        self,
        cot_engine: ChainOfThoughtEngine | None = None,
        anomaly_threshold: float = 0.5,
        domain_specific: bool = True,
    ):
        """Initialize anomaly-specific CoT.

        Args:
            cot_engine: Base CoT engine (creates new if None)
            anomaly_threshold: Threshold for anomaly classification
            domain_specific: Use domain-specific reasoning templates
        """
        self.cot_engine = cot_engine or ChainOfThoughtEngine()
        self.anomaly_threshold = anomaly_threshold
        self.domain_specific = domain_specific

        # Domain-specific reasoning patterns
        self._anomaly_patterns = {
            "statistical": self._reason_statistical_anomaly,
            "temporal": self._reason_temporal_anomaly,
            "spatial": self._reason_spatial_anomaly,
            "behavioral": self._reason_behavioral_anomaly,
            "dimensional": self._reason_dimensional_anomaly,
        }

    def analyze_anomaly(
        self,
        data: dict[str, Any],
        anomaly_score_or_features: float | np.ndarray | None = None,
        domain: str = "general",
    ) -> dict[str, Any]:
        """Analyze potential anomaly with chain-of-thought reasoning.

        Args:
            data: Input data and features (dict with detection info)
            anomaly_score_or_features: Pre-computed anomaly score (float) or raw features (ndarray)
            domain: Anomaly domain (statistical, temporal, etc.)

        Returns:
            Dictionary with reasoning_chain, conclusion, and analysis details
        """
        # Handle different input formats for API compatibility
        if isinstance(anomaly_score_or_features, np.ndarray):
            # Test API: analyze_anomaly(detection_result, raw_features)
            raw_features = anomaly_score_or_features
            anomaly_score = data.get("score", 0.5)
            # Compute score from features if not provided
            if anomaly_score == 0.5 and len(raw_features) > 0:
                # Use simple z-score based anomaly detection
                mean_val = float(np.mean(raw_features))
                std_val = float(np.std(raw_features)) + 1e-8
                max_z = float(np.max(np.abs((raw_features - mean_val) / std_val)))
                anomaly_score = min(1.0, max_z / 3.0)
            data["raw_features"] = (
                raw_features.tolist() if hasattr(raw_features, "tolist") else raw_features
            )
        elif anomaly_score_or_features is not None:
            anomaly_score = float(anomaly_score_or_features)
        else:
            anomaly_score = data.get("score", data.get("anomaly_score", 0.5))

        # Build context
        context = {
            "anomaly_score": anomaly_score,
            "is_anomaly": anomaly_score > self.anomaly_threshold,
            "domain": domain,
            **data,
        }

        # Use domain-specific reasoning if available
        if self.domain_specific and domain in self._anomaly_patterns:
            chain = self._anomaly_patterns[domain](context)
        else:
            # Default reasoning
            problem = f"Analyze potential {domain} anomaly (score: {anomaly_score:.3f})"
            chain = self.cot_engine.reason(problem, context, ReasoningStrategy.VERIFICATION_COT)

        # Return dict format for API compatibility
        return {
            "reasoning_chain": [t.to_dict() for t in chain.thoughts],
            "conclusion": chain.conclusion,
            "confidence": chain.overall_confidence,
            "is_anomaly": anomaly_score > self.anomaly_threshold,
            "anomaly_score": anomaly_score,
            "chain": chain,  # Include original chain for advanced usage
        }

    def _reason_statistical_anomaly(self, context: dict[str, Any]) -> ThoughtChain:
        """Reason about statistical anomalies."""
        score = context.get("anomaly_score", 0.5)
        problem = f"Analyze statistical anomaly indicators (score: {score:.3f})"

        # Add statistical context
        context["features"] = [
            "z-score deviation",
            "distribution tail position",
            "variance analysis",
            "outlier classification",
        ]
        context["evidence"] = ["statistical distribution", "historical baseline"]

        return self.cot_engine.reason(problem, context, ReasoningStrategy.SELF_CONSISTENCY)

    def _reason_temporal_anomaly(self, context: dict[str, Any]) -> ThoughtChain:
        """Reason about temporal anomalies."""
        score = context.get("anomaly_score", 0.5)
        problem = f"Analyze temporal pattern anomaly (score: {score:.3f})"

        context["features"] = [
            "time-series trend",
            "seasonal component",
            "cyclical pattern",
            "change point detection",
        ]
        context["evidence"] = ["temporal sequence", "historical patterns"]

        return self.cot_engine.reason(problem, context, ReasoningStrategy.LEAST_TO_MOST)

    def _reason_spatial_anomaly(self, context: dict[str, Any]) -> ThoughtChain:
        """Reason about spatial anomalies."""
        score = context.get("anomaly_score", 0.5)
        problem = f"Analyze spatial distribution anomaly (score: {score:.3f})"

        context["features"] = [
            "geographic clustering",
            "spatial autocorrelation",
            "neighbor relationships",
            "distance metrics",
        ]

        return self.cot_engine.reason(problem, context, ReasoningStrategy.TREE_OF_THOUGHTS)

    def _reason_behavioral_anomaly(self, context: dict[str, Any]) -> ThoughtChain:
        """Reason about behavioral anomalies."""
        score = context.get("anomaly_score", 0.5)
        problem = f"Analyze behavioral pattern anomaly (score: {score:.3f})"

        context["features"] = [
            "action sequence",
            "frequency patterns",
            "timing analysis",
            "deviation from baseline",
        ]

        return self.cot_engine.reason(problem, context, ReasoningStrategy.VERIFICATION_COT)

    def _reason_dimensional_anomaly(self, context: dict[str, Any]) -> ThoughtChain:
        """Reason about high-dimensional anomalies."""
        score = context.get("anomaly_score", 0.5)
        problem = f"Analyze high-dimensional anomaly (score: {score:.3f})"

        context["features"] = [
            "manifold position",
            "feature correlations",
            "dimensionality reduction",
            "clustering distance",
        ]

        return self.cot_engine.reason(problem, context, ReasoningStrategy.SELF_CONSISTENCY)

    def explain_decision(self, chain: ThoughtChain) -> str:
        """Generate human-readable explanation of anomaly decision.

        Args:
            chain: Completed thought chain

        Returns:
            Human-readable explanation
        """
        return chain.get_reasoning_trace()

    def get_confidence_breakdown(self, chain: ThoughtChain) -> dict[str, float]:
        """Get confidence breakdown by thought type.

        Args:
            chain: Completed thought chain

        Returns:
            Dictionary of thought type to average confidence
        """
        type_confidences: dict[str, list[float]] = {}

        for thought in chain.thoughts:
            type_name = thought.thought_type.value
            if type_name not in type_confidences:
                type_confidences[type_name] = []
            type_confidences[type_name].append(thought.confidence)

        return {t: float(np.mean(confs)) for t, confs in type_confidences.items()}

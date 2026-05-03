"""
Pass Evaluator: Simulates optimization outcomes for pass sequences.

Evaluates the performance of different optimization pass sequences on
benchmark programs, producing metrics for execution time, code size,
and compilation overhead. Uses realistic models derived from published
research results.

References:
    - Wang & O'Boyle (2018): baseline speedups for ML-based optimization
    - PolyBench timing characteristics
    - CompilerGym reward functions
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.optimization.pass_manager import PassManager, PassSequence, OPTIMIZATION_LEVELS
from src.utils.logger import setup_logger

logger = setup_logger("PassEvaluator")


class PassEvaluator:
    """
    Evaluates optimization pass sequences on programs.

    Produces realistic performance metrics based on program characteristics
    and optimization strategy, using models derived from research literature.
    """

    def __init__(self, pass_manager: PassManager, seed: int = 42):
        """
        Initialize PassEvaluator.

        Args:
            pass_manager: PassManager instance for pass sequence info
            seed: Random seed for reproducibility
        """
        self.pass_manager = pass_manager
        self.rng = np.random.RandomState(seed)

        # Performance multipliers for different pass categories
        # Based on typical speedup ranges from Wang & O'Boyle (2018)
        self._speed_multipliers = {
            "aggressive_speed": (1.05, 1.45),
            "size_optimized": (0.90, 1.10),
            "balanced": (1.00, 1.25),
            "loop_intensive": (1.10, 1.60),
            "memory_optimized": (1.00, 1.30),
            "fast_compile": (0.95, 1.05),
            "interprocedural": (1.00, 1.20),
            "vectorization_focus": (1.15, 1.55),
            "embedded_iot": (0.95, 1.10),
            "hpc_scientific": (1.10, 1.50),
        }

        # Code size multipliers (< 1 means smaller code)
        self._size_multipliers = {
            "aggressive_speed": (1.05, 1.30),
            "size_optimized": (0.60, 0.85),
            "balanced": (0.90, 1.10),
            "loop_intensive": (1.10, 1.50),
            "memory_optimized": (0.90, 1.05),
            "fast_compile": (0.95, 1.00),
            "interprocedural": (0.85, 1.00),
            "vectorization_focus": (1.05, 1.25),
            "embedded_iot": (0.65, 0.80),
            "hpc_scientific": (1.10, 1.40),
        }

        # Compilation time multipliers (relative to O0)
        self._compile_multipliers = {
            "aggressive_speed": (3.0, 5.0),
            "size_optimized": (2.0, 3.5),
            "balanced": (2.5, 4.0),
            "loop_intensive": (3.5, 6.0),
            "memory_optimized": (2.0, 3.5),
            "fast_compile": (1.1, 1.5),
            "interprocedural": (4.0, 7.0),
            "vectorization_focus": (3.5, 5.5),
            "embedded_iot": (2.0, 3.0),
            "hpc_scientific": (4.0, 6.5),
        }

        # Security preservation multipliers (proxy for constant-time integrity)
        # Higher is better. Aggressive/vectorization passes hurt security.
        self._security_multipliers = {
            "aggressive_speed": (0.40, 0.60),
            "size_optimized": (0.80, 0.95),
            "balanced": (0.70, 0.85),
            "loop_intensive": (0.30, 0.50),
            "memory_optimized": (0.60, 0.80),
            "fast_compile": (0.90, 1.00),
            "interprocedural": (0.50, 0.70),
            "vectorization_focus": (0.20, 0.40),
            "embedded_iot": (0.85, 0.95),
            "hpc_scientific": (0.30, 0.55),
        }

    def evaluate(self, program: ProgramCharacteristics,
                 pass_sequence: PassSequence,
                 priority: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a pass sequence on a program.

        Args:
            program: Program characteristics
            pass_sequence: Optimization pass sequence
            priority: User priority vector [speed, size, compile_time, security]

        Returns:
            Dictionary with performance metrics
        """
        prog_seed = hash(program.name) % (2**31)
        rng = np.random.RandomState(prog_seed + pass_sequence.id)

        category = pass_sequence.category

        # Program-specific factors based on characteristics
        loop_factor = 1.0 + 0.1 * program.loop_depth
        mem_factor = program.num_load_inst / max(1, program.num_instructions)
        compute_factor = program.num_float_inst / max(1, program.num_instructions)

        # Execution time speedup (relative to -O3)
        speed_range = self._speed_multipliers[category]
        base_speedup = rng.uniform(*speed_range)

        # Adjust based on program characteristics
        if "loop" in category or "vectorization" in category:
            base_speedup *= (1 + 0.15 * (loop_factor - 1))
        if "memory" in category:
            base_speedup *= (1 + 0.2 * mem_factor)
        if "hpc" in category or "aggressive" in category:
            base_speedup *= (1 + 0.1 * compute_factor)

        # Code size ratio (vs -O0)
        size_range = self._size_multipliers[category]
        size_ratio = rng.uniform(*size_range)

        # Language-specific optimization characteristics
        if hasattr(program, "language"):
            if program.language == "cpp" and "interprocedural" in category:
                base_speedup *= 1.25  # C++ heavily benefits from inlining and devirtualization
                size_ratio *= 0.95
            elif program.language == "python" and "balanced" in category:
                base_speedup *= 1.30  # Python JIT benefits from jump-threading and simplifycfg
            elif program.language == "python" and ("vectorization" in category or "loop" in category):
                base_speedup *= 0.85  # Highly dynamic typing in Python makes vectorization less effective

        # Compilation time multiplier (vs -O0)
        compile_range = self._compile_multipliers[category]
        compile_time = rng.uniform(*compile_range)

        # Security preservation proxy
        security_range = self._security_multipliers[category]
        base_security = rng.uniform(*security_range)
        if program.num_branch_inst > program.num_instructions * 0.2:
            base_security *= 0.9

        # Compute composite score based on priority
        if priority is None:
            priority = np.array([0.25, 0.25, 0.25, 0.25])

        # Normalize metrics to [0, 1] range for scoring
        speed_score = min(1.0, base_speedup / 1.5)
        size_score = 1.0 - min(1.0, size_ratio / 1.5)
        compile_score = 1.0 - min(1.0, compile_time / 7.0)
        security_score = min(1.0, base_security)

        composite_score = (priority[0] * speed_score +
                          priority[1] * size_score +
                          priority[2] * compile_score +
                          priority[3] * security_score)

        return {
            "speedup": round(base_speedup, 4),
            "size_ratio": round(size_ratio, 4),
            "compile_time_ratio": round(compile_time, 4),
            "security": round(base_security, 4),
            "speed_score": round(speed_score, 4),
            "size_score": round(size_score, 4),
            "compile_score": round(compile_score, 4),
            "security_score": round(security_score, 4),
            "composite_score": round(composite_score, 4),
            "pass_sequence_id": pass_sequence.id,
            "category": category,
        }

    def evaluate_all_sequences(self, program: ProgramCharacteristics,
                                priority: np.ndarray) -> List[Dict[str, float]]:
        """
        Evaluate all pass sequences on a program with given priority.

        Args:
            program: Program characteristics
            priority: User priority vector

        Returns:
            List of evaluation results, one per pass sequence
        """
        results = []
        for seq in self.pass_manager.pass_sequences:
            result = self.evaluate(program, seq, priority)
            results.append(result)
        return results

    def find_optimal_sequence(self, program: ProgramCharacteristics,
                               priority: np.ndarray) -> Tuple[int, Dict[str, float]]:
        """
        Find the optimal pass sequence for a program with given priority.

        Args:
            program: Program characteristics
            priority: User priority vector

        Returns:
            Tuple of (best_class_id, best_result)
        """
        results = self.evaluate_all_sequences(program, priority)
        best_idx = max(range(len(results)), key=lambda i: results[i]['composite_score'])
        return best_idx, results[best_idx]

    def evaluate_standard_level(self, program: ProgramCharacteristics,
                                 level: str) -> Dict[str, float]:
        """
        Evaluate a standard optimization level (-O0, -O1, -O2, -O3).

        Args:
            program: Program characteristics
            level: Optimization level string

        Returns:
            Performance metrics
        """
        prog_seed = hash(program.name) % (2**31)
        rng = np.random.RandomState(prog_seed + hash(level) % 100)

        level_configs = {
            "O0": {"speedup": 1.0, "size": 1.0, "compile": 1.0},
            "O1": {"speedup": (1.05, 1.20), "size": (0.85, 0.95), "compile": (1.5, 2.5)},
            "O2": {"speedup": (1.15, 1.35), "size": (0.80, 1.05), "compile": (2.5, 4.0)},
            "O3": {"speedup": (1.20, 1.40), "size": (0.90, 1.20), "compile": (3.5, 5.5)},
        }

        config = level_configs.get(level, level_configs["O0"])

        if isinstance(config["speedup"], tuple):
            speedup = rng.uniform(*config["speedup"])
            size = rng.uniform(*config["size"])
            compile_time = rng.uniform(*config["compile"])
        else:
            speedup = config["speedup"]
            size = config["size"]
            compile_time = config["compile"]

        return {
            "speedup": round(speedup, 4),
            "size_ratio": round(size, 4),
            "compile_time_ratio": round(compile_time, 4),
            "level": level,
        }

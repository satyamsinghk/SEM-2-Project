"""
Standard Optimization Level Baselines.

Provides performance data for -O0, -O1, -O2, -O3 standard optimization
levels for comparison against our ML-based approach.
"""
import numpy as np
from typing import Dict, List

from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.optimization.pass_evaluator import PassEvaluator
from src.optimization.pass_manager import PassManager
from src.utils.logger import setup_logger

logger = setup_logger("StandardLevels")


class StandardOptimizationLevels:
    """
    Baseline: Standard compiler optimization levels (-O0 to -O3).

    This serves as the primary baseline for comparison.
    Our ML approach should match or exceed -O3 performance while
    being adaptable to different user priorities.
    """

    def __init__(self, pass_manager: PassManager, seed: int = 42):
        self.pass_evaluator = PassEvaluator(pass_manager, seed)
        self.levels = ["O0", "O1", "O2", "O3"]

    def evaluate_all_levels(self, programs: List[ProgramCharacteristics]) -> Dict[str, Dict]:
        """
        Evaluate all optimization levels on all programs.

        Returns:
            Dict mapping level name to aggregate metrics
        """
        results = {}

        for level in self.levels:
            speedups = []
            sizes = []
            compile_times = []

            for prog in programs:
                metrics = self.pass_evaluator.evaluate_standard_level(prog, level)
                speedups.append(metrics['speedup'])
                sizes.append(metrics['size_ratio'])
                compile_times.append(metrics['compile_time_ratio'])

            results[level] = {
                'avg_speedup': np.mean(speedups),
                'std_speedup': np.std(speedups),
                'min_speedup': np.min(speedups),
                'max_speedup': np.max(speedups),
                'avg_size_ratio': np.mean(sizes),
                'std_size_ratio': np.std(sizes),
                'avg_compile_time': np.mean(compile_times),
                'all_speedups': speedups,
                'all_sizes': sizes,
                'all_compile_times': compile_times,
            }

            logger.info(f"-{level}: avg_speedup={results[level]['avg_speedup']:.4f} "
                       f"± {results[level]['std_speedup']:.4f}")

        return results

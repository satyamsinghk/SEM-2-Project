"""
Iterative Compilation Baseline.

Implements random search over pass sequences to find the best one,
simulating the iterative compilation approach described in the base paper.
This baseline demonstrates the time overhead that our ML approach eliminates.

References:
    - Bodin et al. (1998): Iterative compilation in a non-linear space
    - Agakov et al. (2006): Using ML to focus iterative optimization
"""
import time
import numpy as np
from typing import Dict, List, Tuple

from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.optimization.pass_evaluator import PassEvaluator
from src.optimization.pass_manager import PassManager
from src.utils.logger import setup_logger

logger = setup_logger("IterativeCompilation")


class IterativeCompilation:
    """
    Baseline: Iterative Compilation via Random Search.

    Randomly samples pass sequences and evaluates them to find the best one.
    This is the most common baseline in compiler optimization research.

    Key comparison points:
        - Quality: Should find good solutions (upper bound)
        - Time: Extremely slow (100s-1000s of compilations per program)
    """

    def __init__(self, pass_manager: PassManager, seed: int = 42):
        self.pass_manager = pass_manager
        self.pass_evaluator = PassEvaluator(pass_manager, seed)
        self.rng = np.random.RandomState(seed)

    def search(self, program: ProgramCharacteristics,
               priority: np.ndarray,
               num_iterations: int = 100) -> Dict[str, any]:
        """
        Run iterative compilation for a program.

        Args:
            program: Program characteristics
            priority: User priority vector
            num_iterations: Number of random trials

        Returns:
            Best result and search statistics
        """
        start = time.time()

        best_score = -np.inf
        best_result = None
        all_scores = []

        for i in range(num_iterations):
            # Random pass sequence
            seq_id = self.rng.randint(0, self.pass_manager.num_classes)
            seq = self.pass_manager.get_sequence(seq_id)

            result = self.pass_evaluator.evaluate(program, seq, priority)
            score = result['composite_score']
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_result = result

        search_time = time.time() - start

        return {
            'best_result': best_result,
            'best_score': best_score,
            'num_iterations': num_iterations,
            'search_time': search_time,
            'all_scores': all_scores,
            'convergence': self._compute_convergence(all_scores),
        }

    def _compute_convergence(self, scores: List[float]) -> List[float]:
        """Compute running best score (convergence curve)."""
        convergence = []
        best = -np.inf
        for s in scores:
            best = max(best, s)
            convergence.append(best)
        return convergence

    def evaluate_multiple_budgets(self, programs: List[ProgramCharacteristics],
                                  priority: np.ndarray,
                                  budgets: List[int] = [10, 50, 100, 500, 1000]
                                  ) -> Dict[int, Dict]:
        """
        Evaluate iterative compilation at different search budgets.

        This shows how ML prediction compares vs different amounts
        of iterative search effort.
        """
        results = {}

        for budget in budgets:
            all_speedups = []
            all_times = []

            for prog in programs:
                search_result = self.search(prog, priority, budget)
                all_speedups.append(search_result['best_result']['speedup'])
                all_times.append(search_result['search_time'])

            results[budget] = {
                'avg_speedup': np.mean(all_speedups),
                'std_speedup': np.std(all_speedups),
                'avg_search_time': np.mean(all_times),
                'total_search_time': np.sum(all_times),
                'all_speedups': all_speedups,
            }

            logger.info(f"Budget={budget}: avg_speedup={results[budget]['avg_speedup']:.4f}, "
                       f"avg_time={results[budget]['avg_search_time']:.4f}s")

        return results

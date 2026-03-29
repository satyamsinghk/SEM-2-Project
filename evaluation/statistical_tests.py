"""
Statistical Tests for validating experimental results.
"""
import numpy as np
from typing import Dict, Tuple
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger("StatTests")


class StatisticalTests:
    """Statistical significance testing for method comparison."""

    @staticmethod
    def paired_t_test(method_a: np.ndarray, method_b: np.ndarray,
                       alpha: float = 0.05) -> Dict[str, float]:
        """Paired t-test between two methods."""
        t_stat, p_value = stats.ttest_rel(method_a, method_b)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
            'mean_diff': float(np.mean(method_a - method_b)),
        }

    @staticmethod
    def wilcoxon_test(method_a: np.ndarray, method_b: np.ndarray,
                       alpha: float = 0.05) -> Dict[str, float]:
        """Wilcoxon signed-rank test (non-parametric)."""
        diff = method_a - method_b
        non_zero = diff[diff != 0]
        if len(non_zero) < 3:
            return {'p_value': 1.0, 'significant': False, 'note': 'insufficient_data'}

        stat, p_value = stats.wilcoxon(non_zero)
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < alpha,
            'alpha': alpha,
        }

    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 1000,
                                       confidence: float = 0.95) -> Dict[str, float]:
        """Bootstrap confidence interval for mean."""
        rng = np.random.RandomState(42)
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            means.append(np.mean(sample))
        means = np.sort(means)
        lower = means[int((1 - confidence) / 2 * n_bootstrap)]
        upper = means[int((1 + confidence) / 2 * n_bootstrap)]
        return {
            'mean': float(np.mean(data)),
            'ci_lower': float(lower),
            'ci_upper': float(upper),
            'confidence': confidence,
        }

    @staticmethod
    def effect_size_cohens_d(method_a: np.ndarray, method_b: np.ndarray) -> float:
        """Cohen's d effect size."""
        diff = method_a - method_b
        return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-10))

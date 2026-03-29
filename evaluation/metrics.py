"""
Performance Metrics for Compiler Optimization Evaluation.

Computes all metrics defined in the zeroth review:
    1. Execution Time Speedup (relative to -O3)
    2. Code Size Reduction (%)
    3. Prediction Accuracy (F1-Score)
    4. Compilation Overhead
    5. Adaptability Score
"""
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

from src.utils.logger import setup_logger

logger = setup_logger("Metrics")


class PerformanceMetrics:
    """
    Comprehensive performance metrics calculator.

    Implements all metrics from the project proposal for rigorous evaluation.
    """

    @staticmethod
    def execution_time_speedup(our_speedups: np.ndarray,
                                baseline_speedups: np.ndarray) -> Dict[str, float]:
        """
        Compute execution time speedup relative to a baseline.

        Args:
            our_speedups: Speedup achieved by our method
            baseline_speedups: Speedup achieved by baseline

        Returns:
            Speedup statistics
        """
        relative = our_speedups / np.maximum(baseline_speedups, 1e-6)
        return {
            'mean_relative_speedup': float(np.mean(relative)),
            'median_relative_speedup': float(np.median(relative)),
            'std_relative_speedup': float(np.std(relative)),
            'min_relative_speedup': float(np.min(relative)),
            'max_relative_speedup': float(np.max(relative)),
            'pct_improved': float(np.mean(relative > 1.0) * 100),
            'mean_absolute_speedup': float(np.mean(our_speedups)),
        }

    @staticmethod
    def code_size_reduction(our_sizes: np.ndarray,
                            baseline_sizes: np.ndarray) -> Dict[str, float]:
        """Compute code size reduction percentage."""
        reduction = (1 - our_sizes / np.maximum(baseline_sizes, 1e-6)) * 100
        return {
            'mean_reduction_pct': float(np.mean(reduction)),
            'median_reduction_pct': float(np.median(reduction)),
            'std_reduction_pct': float(np.std(reduction)),
            'pct_smaller': float(np.mean(our_sizes < baseline_sizes) * 100),
        }

    @staticmethod
    def prediction_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        precision, recall, f1_per_class, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        per_class = {}
        for i in range(len(precision)):
            name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
            per_class[name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support[i]),
            }

        return {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'per_class_metrics': per_class,
        }

    @staticmethod
    def compilation_overhead(ml_prediction_time: float,
                              iterative_time: float,
                              num_programs: int) -> Dict[str, float]:
        """Compute compilation overhead comparison."""
        ml_per_prog = ml_prediction_time / max(1, num_programs)
        iter_per_prog = iterative_time / max(1, num_programs)
        speedup = iterative_time / max(ml_prediction_time, 1e-9)

        return {
            'ml_total_time': float(ml_prediction_time),
            'ml_per_program_ms': float(ml_per_prog * 1000),
            'iterative_total_time': float(iterative_time),
            'iterative_per_program_s': float(iter_per_prog),
            'time_speedup': float(speedup),
            'time_reduction_pct': float((1 - ml_prediction_time / max(iterative_time, 1e-9)) * 100),
        }

    @staticmethod
    def adaptability_score(predictions_by_priority: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Measure how well the model adapts to different priority settings.

        A good model should produce DIFFERENT predictions when priorities change.

        Args:
            predictions_by_priority: Dict mapping priority preset name to predictions

        Returns:
            Adaptability metrics
        """
        preset_names = list(predictions_by_priority.keys())
        predictions = list(predictions_by_priority.values())

        if len(predictions) < 2:
            return {'adaptability_score': 0.0}

        pairwise_diffs = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                diff_rate = np.mean(predictions[i] != predictions[j])
                pairwise_diffs.append(diff_rate)

        return {
            'adaptability_score': float(np.mean(pairwise_diffs)),
            'max_pairwise_diff': float(np.max(pairwise_diffs)),
            'min_pairwise_diff': float(np.min(pairwise_diffs)),
            'num_priority_settings': len(predictions),
            'pairwise_details': {
                f"{preset_names[i]}_vs_{preset_names[j]}": float(
                    np.mean(predictions[i] != predictions[j]))
                for i in range(len(predictions))
                for j in range(i + 1, len(predictions))
            },
        }

    @staticmethod
    def compute_all_metrics(y_true, y_pred, our_speedups, baseline_speedups,
                            our_sizes, baseline_sizes, ml_time, iter_time,
                            num_programs, predictions_by_priority=None,
                            class_names=None) -> Dict[str, Any]:
        """Compute all metrics in one call."""
        pm = PerformanceMetrics

        results = {
            'speedup': pm.execution_time_speedup(our_speedups, baseline_speedups),
            'code_size': pm.code_size_reduction(our_sizes, baseline_sizes),
            'prediction': pm.prediction_accuracy(y_true, y_pred, class_names),
            'overhead': pm.compilation_overhead(ml_time, iter_time, num_programs),
        }

        if predictions_by_priority:
            results['adaptability'] = pm.adaptability_score(predictions_by_priority)

        return results

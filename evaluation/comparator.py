"""
Method Comparator: Cross-method performance comparison.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from tabulate import tabulate

from src.utils.logger import setup_logger

logger = setup_logger("Comparator")


class MethodComparator:
    """Compares performance across all methods (ours, baselines, base paper)."""

    def __init__(self):
        self.results = {}

    def add_result(self, method_name: str, metrics: Dict[str, Any]):
        """Add evaluation results for a method."""
        self.results[method_name] = metrics
        logger.info(f"Added results for: {method_name}")

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate a comprehensive comparison table."""
        rows = []
        for method, metrics in self.results.items():
            row = {'Method': method}

            # Extract key metrics
            if 'speedup' in metrics:
                row['Avg Speedup'] = metrics['speedup'].get('mean_absolute_speedup', '-')
                row['Speedup vs O3'] = metrics['speedup'].get('mean_relative_speedup', '-')
                row['% Improved'] = metrics['speedup'].get('pct_improved', '-')

            if 'code_size' in metrics:
                row['Size Reduction %'] = metrics['code_size'].get('mean_reduction_pct', '-')

            if 'prediction' in metrics:
                row['Accuracy'] = metrics['prediction'].get('accuracy', '-')
                row['F1 (macro)'] = metrics['prediction'].get('f1_macro', '-')
                row['F1 (weighted)'] = metrics['prediction'].get('f1_weighted', '-')
                row['Precision'] = metrics['prediction'].get('precision_macro', '-')
                row['Recall'] = metrics['prediction'].get('recall_macro', '-')

            if 'overhead' in metrics:
                row['Pred Time (ms)'] = metrics['overhead'].get('ml_per_program_ms', '-')
                row['Time Speedup'] = metrics['overhead'].get('time_speedup', '-')

            if 'adaptability' in metrics:
                row['Adaptability'] = metrics['adaptability'].get('adaptability_score', '-')

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def print_comparison(self):
        """Print formatted comparison table."""
        df = self.generate_comparison_table()
        print("\n" + "=" * 100)
        print("COMPREHENSIVE METHOD COMPARISON")
        print("=" * 100)
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.4f', showindex=False))
        print("=" * 100 + "\n")

    def get_best_method(self, metric: str = 'f1_macro') -> str:
        """Find the best performing method for a given metric."""
        best_method = None
        best_value = -np.inf

        for method, metrics in self.results.items():
            value = None
            if metric in metrics:
                value = metrics[metric]
            elif 'prediction' in metrics and metric in metrics['prediction']:
                value = metrics['prediction'][metric]
            elif 'speedup' in metrics and metric in metrics['speedup']:
                value = metrics['speedup'][metric]

            if value is not None and isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best_method = method

        return best_method

    def save_comparison(self, filepath: str):
        """Save comparison table to CSV."""
        df = self.generate_comparison_table()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved comparison to {filepath}")

"""
Priority Sensitivity Analysis Visualization.

Shows how the model's predictions change with different user priorities.
This is a key visualization for demonstrating the novel contribution.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from src.utils.logger import setup_logger

logger = setup_logger("PrioritySensitivity")


class PrioritySensitivityPlotter:
    """Visualizes the effect of priority vectors on optimization decisions."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_priority_sensitivity(self, predictions_by_priority: Dict[str, np.ndarray],
                                    class_names: List[str]):
        """Show how predictions change across different priority settings."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        preset_names = list(predictions_by_priority.keys())

        for idx, (preset, predictions) in enumerate(list(predictions_by_priority.items())[:4]):
            ax = axes[idx // 2][idx % 2]
            unique, counts = np.unique(predictions, return_counts=True)
            names = [class_names[u] if u < len(class_names) else f"class_{u}" for u in unique]

            colors = sns.color_palette("Set3", len(unique))
            wedges, texts, autotexts = ax.pie(
                counts, labels=None, autopct='%1.1f%%',
                colors=colors, startangle=90, pctdistance=0.85
            )
            ax.set_title(f'Priority: {preset}', fontweight='bold')
            ax.legend(names, loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=8)

        plt.suptitle('Optimization Strategy Distribution by Priority Setting', fontsize=14)
        plt.tight_layout()
        path = os.path.join(self.output_dir, "priority_sensitivity.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_adaptability_heatmap(self, diff_matrix: Dict[str, float],
                                   preset_names: List[str]):
        """Pairwise difference heatmap between priority settings."""
        n = len(preset_names)
        matrix = np.zeros((n, n))

        for key, value in diff_matrix.items():
            parts = key.split('_vs_')
            if len(parts) == 2:
                try:
                    i = preset_names.index(parts[0])
                    j = preset_names.index(parts[1])
                    matrix[i][j] = value
                    matrix[j][i] = value
                except ValueError:
                    pass

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=preset_names, yticklabels=preset_names, ax=ax)
        ax.set_title('Prediction Difference Rate Between Priority Settings')

        plt.tight_layout()
        path = os.path.join(self.output_dir, "adaptability_heatmap.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_priority_sweep(self, sweep_results: List[Dict],
                             class_names: List[str]):
        """Plot how class distribution changes as speed weight varies."""
        fig, ax = plt.subplots(figsize=(12, 6))

        speed_weights = [r['speed_weight'] for r in sweep_results]
        class_counts = {}

        for r in sweep_results:
            for cls, count in r['distribution'].items():
                if cls not in class_counts:
                    class_counts[cls] = []
                class_counts[cls].append(count)

        colors = sns.color_palette("Set2", len(class_counts))
        bottom = np.zeros(len(speed_weights))

        for i, (cls, counts) in enumerate(class_counts.items()):
            cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{cls}"
            ax.bar(range(len(speed_weights)), counts, bottom=bottom,
                   label=cls_name, color=colors[i % len(colors)], alpha=0.8)
            bottom += np.array(counts)

        ax.set_xlabel('Speed Weight')
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Predicted Strategy Distribution vs Speed Priority')
        ax.set_xticks(range(len(speed_weights)))
        ax.set_xticklabels([f'{w:.1f}' for w in speed_weights], fontsize=9)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "priority_sweep.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

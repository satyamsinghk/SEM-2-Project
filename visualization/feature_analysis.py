"""
Feature Analysis Visualization.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
from src.utils.logger import setup_logger

logger = setup_logger("FeatureAnalysis")


class FeatureAnalysisPlotter:
    """Visualizes feature importance and correlations."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_feature_importance(self, importances: np.ndarray,
                                 feature_names: List[str],
                                 top_n: int = 20,
                                 model_name: str = "Random Forest"):
        """Plot top-N feature importances as horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(10, 8))

        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_values = importances[indices]

        colors = sns.color_palette("viridis", top_n)
        ax.barh(range(top_n), top_values[::-1], color=colors,
                edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top-{top_n} Feature Importances ({model_name})')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_feature_correlation_matrix(self, features: np.ndarray,
                                         feature_names: List[str],
                                         top_n: int = 25):
        """Plot feature correlation heatmap."""
        fig, ax = plt.subplots(figsize=(14, 12))

        # Use only top-N features for readability
        if len(feature_names) > top_n:
            # Select features with highest variance
            variances = np.var(features, axis=0)
            indices = np.argsort(variances)[::-1][:top_n]
            features = features[:, indices]
            feature_names = [feature_names[i] for i in indices]

        corr = np.corrcoef(features.T)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    xticklabels=feature_names, yticklabels=feature_names)

        ax.set_title('Feature Correlation Matrix (Top Variance Features)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_correlation_matrix.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_feature_group_importance(self, importances: np.ndarray,
                                       groups: Dict[str, List[int]]):
        """Plot aggregate importance by feature category."""
        fig, ax = plt.subplots(figsize=(10, 6))

        group_importances = {}
        for name, indices in groups.items():
            group_importances[name] = np.sum(importances[indices])

        names = list(group_importances.keys())
        values = list(group_importances.values())
        colors = sns.color_palette("Set2", len(names))

        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Aggregate Importance')
        ax.set_title('Feature Importance by Category')
        ax.set_xticklabels(names, rotation=25, ha='right')
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "feature_group_importance.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

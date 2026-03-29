"""
Performance Plots for results visualization.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("PerformancePlots")

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


class PerformancePlotter:
    """Creates publication-quality performance comparison plots."""

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_speedup_comparison(self, results: Dict[str, Dict],
                                 title: str = "Execution Time Speedup Comparison"):
        """Bar chart comparing speedups across methods."""
        fig, ax = plt.subplots(figsize=(12, 6))

        methods = list(results.keys())
        speedups = [results[m].get('mean_absolute_speedup',
                    results[m].get('avg_speedup', 1.0)) for m in methods]
        stds = [results[m].get('std_speedup', 0) for m in methods]

        colors = sns.color_palette("husl", len(methods))
        bars = ax.bar(range(len(methods)), speedups, yerr=stds,
                      color=colors, edgecolor='black', linewidth=0.5,
                      capsize=5, alpha=0.85)

        ax.set_xlabel("Method")
        ax.set_ylabel("Average Speedup (relative to -O0)")
        ax.set_title(title)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=25, ha='right')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='-O0 baseline')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "speedup_comparison.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_model_comparison_radar(self, model_results: Dict[str, Dict],
                                     metrics: List[str] = None):
        """Radar/spider chart comparing models across multiple metrics."""
        if metrics is None:
            metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = sns.color_palette("Set2", len(model_results))

        for i, (model_name, results) in enumerate(model_results.items()):
            values = [results.get(m, 0) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Multi-Metric Model Comparison", pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        path = os.path.join(self.output_dir, "model_radar_comparison.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                               model_name: str = "Model"):
        """Plot confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {model_name}')

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_training_curves(self, history: Dict[str, List],
                              model_name: str = "DNN"):
        """Plot training/validation loss curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        if 'train_losses' in history:
            axes[0].plot(history['train_losses'], label='Training Loss', linewidth=2)
        if 'val_losses' in history:
            axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Training Curves')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy curve
        if 'val_accuracies' in history:
            axes[1].plot(history['val_accuracies'], label='Validation Accuracy',
                        linewidth=2, color='green')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title(f'{model_name} - Validation Accuracy')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"training_curves_{model_name.lower().replace(' ', '_')}.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_compilation_overhead(self, ml_time_ms: float, iterative_times: Dict[int, float]):
        """Plot compilation time comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        budgets = sorted(iterative_times.keys())
        iter_times = [iterative_times[b] * 1000 for b in budgets]  # Convert to ms

        ax.bar(['ML Prediction'], [ml_time_ms], color='#2ecc71', edgecolor='black',
               linewidth=0.5, alpha=0.85, label='Our Approach')

        ax.bar([f'Iter-{b}' for b in budgets], iter_times,
               color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.7,
               label='Iterative Compilation')

        ax.set_ylabel('Time per Program (ms)')
        ax.set_title('Compilation Overhead: ML Prediction vs Iterative Compilation')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        ax.text(0, ml_time_ms * 1.3, f'{ml_time_ms:.2f}ms',
                ha='center', fontweight='bold', fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "compilation_overhead.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

    def plot_per_benchmark_speedup(self, benchmark_names: List[str],
                                     our_speedups: List[float],
                                     o3_speedups: List[float]):
        """Per-benchmark speedup comparison."""
        fig, ax = plt.subplots(figsize=(14, 7))

        x = np.arange(len(benchmark_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, o3_speedups, width, label='-O3',
                       color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width / 2, our_speedups, width, label='Our Approach',
                       color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Speedup')
        ax.set_title('Per-Benchmark Speedup: Our Approach vs -O3')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmark_names, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        path = os.path.join(self.output_dir, "per_benchmark_speedup.png")
        plt.savefig(path)
        plt.close()
        logger.info(f"Saved: {path}")
        return path

"""
Run Experiment: End-to-End Pipeline.

Executes the complete experiment:
1. Generate dataset
2. Train all models (RF, XGBoost, DNN, Ensemble)
3. Evaluate against all baselines
4. Compute all metrics
5. Generate all visualizations
6. Produce comparison tables

Usage:
    python scripts/run_experiment.py
"""
import os
import sys
import time
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import generate_dataset, create_polybench_programs
from src.models.random_forest_model import RandomForestOptimizer
from src.models.xgboost_model import XGBoostOptimizer
from src.models.dnn_model import DNNOptimizer
from src.models.ensemble_model import EnsembleOptimizer
from src.baselines.standard_levels import StandardOptimizationLevels
from src.baselines.iterative_compilation import IterativeCompilation
from src.baselines.base_paper_approach import BasePaperApproach
from src.optimization.pass_manager import PassManager
from src.optimization.pass_evaluator import PassEvaluator
from src.priority_vector.metric_priority import MetricPriorityVector, PRIORITY_PRESETS
from src.feature_extraction.ir_feature_extractor import IRFeatureExtractor
from evaluation.metrics import PerformanceMetrics
from evaluation.comparator import MethodComparator
from evaluation.statistical_tests import StatisticalTests
from visualization.performance_plots import PerformancePlotter
from visualization.feature_analysis import FeatureAnalysisPlotter
from visualization.priority_sensitivity import PrioritySensitivityPlotter
from src.utils.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("Experiment")


def run_experiment(config_path: str = "config/config.yaml"):
    """Run the complete experiment pipeline."""
    experiment_start = time.time()

    logger.info("=" * 70)
    logger.info("  PREDICTIVE & PREFERENCE-AWARE COMPILER OPTIMIZATION EXPERIMENT")
    logger.info("  M.Tech Project - NIT Tiruchirappalli")
    logger.info("=" * 70)

    # ================================================================
    # PHASE 1: Dataset Generation
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATASET GENERATION")
    logger.info("=" * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = generate_dataset(config_path)
    logger.info(f"Dataset: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    # ================================================================
    # PHASE 2: Model Training
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 50)

    # 2a. Random Forest
    logger.info("\n--- Random Forest ---")
    rf_model = RandomForestOptimizer(num_classes=10)
    rf_history = rf_model.train(X_train, y_train, X_val, y_val, tune_hyperparams=False)

    # 2b. XGBoost
    logger.info("\n--- XGBoost ---")
    xgb_model = XGBoostOptimizer(num_classes=10)
    xgb_history = xgb_model.train(X_train, y_train, X_val, y_val)

    # 2c. DNN
    logger.info("\n--- Deep Neural Network ---")
    dnn_model = DNNOptimizer(num_classes=10, input_dim=X_train.shape[1])
    dnn_history = dnn_model.train(X_train, y_train, X_val, y_val)

    # 2d. Ensemble
    logger.info("\n--- Ensemble (RF + XGBoost + DNN) ---")
    ensemble_model = EnsembleOptimizer(num_classes=10, input_dim=X_train.shape[1])
    ensemble_history = ensemble_model.train(X_train, y_train, X_val, y_val)

    # 2e. Base Paper Approaches
    logger.info("\n--- Base Paper (Decision Tree) ---")
    base_dt = BasePaperApproach(num_classes=10, method="decision_tree")
    base_dt.train(X_train, y_train, X_val, y_val)

    logger.info("\n--- Base Paper (KNN) ---")
    base_knn = BasePaperApproach(num_classes=10, method="knn")
    base_knn.train(X_train, y_train, X_val, y_val)

    # ================================================================
    # PHASE 3: Evaluation
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 50)

    pass_manager = PassManager(num_classes=10)
    pass_evaluator = PassEvaluator(pass_manager)
    class_names = pass_manager.get_class_names()

    # 3a. Evaluate all models
    models = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'DNN': dnn_model,
        'Ensemble': ensemble_model,
        'Base Paper (DT)': base_dt,
        'Base Paper (KNN)': base_knn,
    }

    model_results = {}
    for name, model in models.items():
        logger.info(f"\nEvaluating: {name}")
        result = model.evaluate(X_test, y_test)
        model_results[name] = result

    # 3b. Standard optimization level baselines
    logger.info("\n--- Standard Level Baselines ---")
    programs = create_polybench_programs()
    std_baselines = StandardOptimizationLevels(pass_manager)
    std_results = std_baselines.evaluate_all_levels(programs)

    # 3c. Iterative compilation baseline
    logger.info("\n--- Iterative Compilation Baseline ---")
    priority_system = MetricPriorityVector()
    balanced_priority = priority_system.get_preset("balanced")
    iterative = IterativeCompilation(pass_manager)
    iter_results = iterative.evaluate_multiple_budgets(
        programs, balanced_priority, budgets=[10, 50, 100, 500]
    )

    # 3d. Compute speedups for our best model (Ensemble)
    logger.info("\n--- Computing Speedups ---")
    feature_extractor = IRFeatureExtractor(normalize=True)
    raw_features = feature_extractor.extract_batch(programs)

    our_speedups = []
    o3_speedups = []
    our_sizes = []
    o3_sizes = []

    for i, prog in enumerate(programs):
        # Our prediction
        augmented = np.concatenate([raw_features[i], balanced_priority])
        pred_class = ensemble_model.predict(augmented.reshape(1, -1))[0]
        pred_seq = pass_manager.get_sequence(pred_class)
        our_result = pass_evaluator.evaluate(prog, pred_seq, balanced_priority)
        our_speedups.append(our_result['speedup'])
        our_sizes.append(our_result['size_ratio'])

        # O3 baseline
        o3_result = pass_evaluator.evaluate_standard_level(prog, "O3")
        o3_speedups.append(o3_result['speedup'])
        o3_sizes.append(o3_result['size_ratio'])

    our_speedups = np.array(our_speedups)
    o3_speedups = np.array(o3_speedups)
    our_sizes = np.array(our_sizes)
    o3_sizes = np.array(o3_sizes)

    # 3e. Adaptability Score
    logger.info("\n--- Adaptability Analysis ---")
    predictions_by_priority = {}
    for preset_name in ['speed_first', 'size_first', 'balanced', 'energy_efficient']:
        priority = priority_system.get_preset(preset_name)
        augmented_features = priority_system.augment_features(raw_features, priority)
        preds = ensemble_model.predict(augmented_features)
        predictions_by_priority[preset_name] = preds

    adaptability = PerformanceMetrics.adaptability_score(predictions_by_priority)
    logger.info(f"Adaptability Score: {adaptability['adaptability_score']:.4f}")

    # 3f. Compilation overhead
    pred_start = time.time()
    _ = ensemble_model.predict(X_test)
    ml_prediction_time = time.time() - pred_start

    overhead = PerformanceMetrics.compilation_overhead(
        ml_prediction_time,
        sum(iter_results[b]['total_search_time'] for b in iter_results) / len(iter_results),
        len(X_test)
    )

    # ================================================================
    # PHASE 4: Comprehensive Results
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 4: COMPREHENSIVE RESULTS")
    logger.info("=" * 50)

    # Method comparator
    comparator = MethodComparator()

    for name, result in model_results.items():
        metrics = {
            'prediction': {
                'accuracy': result['accuracy'],
                'f1_macro': result['f1_macro'],
                'f1_weighted': result['f1_weighted'],
                'precision_macro': result['precision_macro'],
                'recall_macro': result['recall_macro'],
            },
            'overhead': {
                'ml_per_program_ms': result['prediction_time_per_sample'] * 1000,
            },
        }
        comparator.add_result(name, metrics)

    # Add speedup info for our best model
    speedup_metrics = PerformanceMetrics.execution_time_speedup(our_speedups, o3_speedups)
    comparator.add_result('Our Approach (Ensemble)', {
        'prediction': model_results['Ensemble'],
        'speedup': speedup_metrics,
        'code_size': PerformanceMetrics.code_size_reduction(our_sizes, o3_sizes),
        'overhead': overhead,
        'adaptability': adaptability,
    })

    # Add standard level results
    for level, data in std_results.items():
        comparator.add_result(f'-{level}', {
            'speedup': {'mean_absolute_speedup': data['avg_speedup'],
                       'std_speedup': data['std_speedup']},
        })

    # Print comparison
    comparator.print_comparison()
    comparator.save_comparison("results/tables/method_comparison.csv")

    # ================================================================
    # PHASE 5: Visualization
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 5: GENERATING VISUALIZATIONS")
    logger.info("=" * 50)

    perf_plotter = PerformancePlotter()
    feat_plotter = FeatureAnalysisPlotter()
    pri_plotter = PrioritySensitivityPlotter()

    # 5a. Model comparison radar
    radar_data = {}
    for name in ['Random Forest', 'XGBoost', 'DNN', 'Ensemble']:
        radar_data[name] = model_results[name]
    perf_plotter.plot_model_comparison_radar(radar_data)

    # 5b. Speedup comparison bar chart
    speedup_data = {
        '-O1': std_results.get('O1', {}),
        '-O2': std_results.get('O2', {}),
        '-O3': std_results.get('O3', {}),
        'Iter-100': iter_results.get(100, {}),
        'Our Approach': {
            'mean_absolute_speedup': float(np.mean(our_speedups)),
            'std_speedup': float(np.std(our_speedups)),
        },
    }
    perf_plotter.plot_speedup_comparison(speedup_data)

    # 5c. Confusion matrices
    for name in ['Random Forest', 'Ensemble']:
        if 'confusion_matrix' in model_results[name]:
            perf_plotter.plot_confusion_matrix(
                model_results[name]['confusion_matrix'],
                class_names, name
            )

    # 5d. DNN training curves
    if 'train_losses' in dnn_history:
        perf_plotter.plot_training_curves(dnn_history, "DNN")

    # 5e. Feature importance (from RF)
    try:
        importances = rf_model.get_feature_importances()
        feat_names = feature_extractor.get_all_feature_names() + \
                     ['priority_speed', 'priority_size', 'priority_compile', 'priority_security']
        feat_plotter.plot_feature_importance(importances, feat_names, top_n=20)

        groups = feature_extractor.get_feature_importance_groups()
        groups['Priority_Vector'] = [91, 92, 93, 94]
        feat_plotter.plot_feature_group_importance(importances, groups)
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")

    # 5f. Feature correlation
    feat_plotter.plot_feature_correlation_matrix(
        X_train, feat_names if 'feat_names' in dir() else
        [f"f{i}" for i in range(X_train.shape[1])]
    )

    # 5g. Priority sensitivity
    pri_plotter.plot_priority_sensitivity(predictions_by_priority, class_names)

    # 5h. Priority adaptability heatmap
    if 'pairwise_details' in adaptability:
        preset_names_list = list(predictions_by_priority.keys())
        pri_plotter.plot_adaptability_heatmap(
            adaptability['pairwise_details'], preset_names_list
        )

    # 5i. Per-benchmark speedup
    benchmark_names = [p.name for p in programs]
    perf_plotter.plot_per_benchmark_speedup(benchmark_names, our_speedups.tolist(), o3_speedups.tolist())

    # 5j. Compilation overhead
    perf_plotter.plot_compilation_overhead(
        overhead['ml_per_program_ms'],
        {b: iter_results[b]['avg_search_time'] for b in iter_results}
    )

    # ================================================================
    # PHASE 6: Statistical Tests
    # ================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 6: STATISTICAL SIGNIFICANCE TESTING")
    logger.info("=" * 50)

    stat_tests = StatisticalTests()

    ttest = stat_tests.paired_t_test(our_speedups, o3_speedups)
    logger.info(f"Paired t-test (Our vs O3): t={ttest['t_statistic']:.4f}, "
                f"p={ttest['p_value']:.6f}, significant={ttest['significant']}")

    wilcoxon_result = stat_tests.wilcoxon_test(our_speedups, o3_speedups)
    logger.info(f"Wilcoxon test: p={wilcoxon_result['p_value']:.6f}, "
                f"significant={wilcoxon_result['significant']}")

    bootstrap = stat_tests.bootstrap_confidence_interval(our_speedups - o3_speedups)
    logger.info(f"Bootstrap 95% CI for speedup difference: "
                f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}]")

    effect_size = stat_tests.effect_size_cohens_d(our_speedups, o3_speedups)
    logger.info(f"Cohen's d effect size: {effect_size:.4f}")

    # ================================================================
    # Save Summary
    # ================================================================
    total_time = time.time() - experiment_start

    summary = {
        "experiment_time_seconds": round(total_time, 2),
        "dataset": {
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
            "feature_dim": X_train.shape[1],
            "num_classes": 10,
        },
        "model_accuracy": {
            name: round(result['accuracy'], 4)
            for name, result in model_results.items()
        },
        "model_f1_macro": {
            name: round(result['f1_macro'], 4)
            for name, result in model_results.items()
        },
        "best_model": max(model_results, key=lambda k: model_results[k]['f1_macro']),
        "speedup_vs_O3": {
            "mean": round(float(np.mean(our_speedups / o3_speedups)), 4),
            "our_mean": round(float(np.mean(our_speedups)), 4),
            "o3_mean": round(float(np.mean(o3_speedups)), 4),
        },
        "adaptability_score": adaptability['adaptability_score'],
        "prediction_time_ms": overhead['ml_per_program_ms'],
        "statistical_tests": {
            "ttest_p_value": ttest['p_value'],
            "ttest_significant": ttest['significant'],
            "cohens_d": effect_size,
        },
    }

    summary_path = "results/experiment_summary.json"
    os.makedirs("results", exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    logger.info("\n" + "=" * 70)
    logger.info("  EXPERIMENT COMPLETE")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Best model: {summary['best_model']}")
    logger.info(f"  Best F1-macro: {max(summary['model_f1_macro'].values()):.4f}")
    logger.info(f"  Speedup vs O3: {summary['speedup_vs_O3']['mean']:.4f}x")
    logger.info(f"  Adaptability: {summary['adaptability_score']:.4f}")
    logger.info(f"  Results saved to: results/")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    summary = run_experiment()
    print("\n\nFINAL SUMMARY:")
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))

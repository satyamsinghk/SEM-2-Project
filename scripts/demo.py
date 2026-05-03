"""
Interactive Demo Script for Predictive Compiler Optimization.

Usage:
    python scripts/demo.py [--file path/to/sample.c] [--priority speed_first|size_first|balanced]
"""
import os
import sys
import argparse
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_dataset import generate_dataset
from src.models.xgboost_model import XGBoostOptimizer
from src.models.dnn_model import DNNOptimizer
from src.models.ensemble_model import EnsembleOptimizer
from src.feature_extraction.ir_feature_extractor import IRFeatureExtractor
from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.optimization.pass_manager import PassManager
from src.optimization.pass_evaluator import PassEvaluator
from src.priority_vector.metric_priority import MetricPriorityVector
from src.utils.logger import setup_logger

logger = setup_logger("Demo")

SAMPLE_C_CODE = """
#include <stdio.h>
#include <stdlib.h>

#define N 500

int main() {
    // Basic matrix multiplication heavily dependent on optimizations
    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    
    // Initialize matrices
    for(int i = 0; i < N*N; i++) {
        A[i] = (double)(i % 100) / 100.0;
        B[i] = (double)(i % 50) / 50.0;
        C[i] = 0.0;
    }
    
    // Matrix mult
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0.0;
            for(int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    
    // Prevent dead code elimination
    double final_sum = 0.0;
    for(int i = 0; i < N*N; i++) {
        final_sum += C[i];
    }
    printf("Result: %f\\n", final_sum);
    
    free(A); free(B); free(C);
    return 0;
}
"""

def create_sample_file(path="data/raw/sample.c"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(SAMPLE_C_CODE)
    return path

def run_demo(file_path=None, priority_preset="speed_first", config_path="config/config.yaml"):
    logger.info("="*60)
    logger.info("  ML PREDICTIVE COMPILER OPTIMIZATION DEMO")
    logger.info("="*60)
    
    if not file_path or not os.path.exists(file_path):
        logger.info(f"Using default inline Matrix Multiplication sample.")
        file_path = create_sample_file()
    else:
        logger.info(f"Using provided source file: {file_path}")
        
    logger.info(f"Selected Priority Profile: '{priority_preset.upper()}'")

    # 1. Quick Model Bootstrapping
    logger.info("\n[1/4] Bootstrapping Training Data & Models (Fast cache load)...")
    X_train, X_val, X_test, y_train, y_val, y_test = generate_dataset(config_path)
    
    logger.info("Training best models (DNN & ExtraTrees) on cached dataset...")
    # Train robust fallbacks as the primary demo engine natively avoiding segmentation faults
    model = EnsembleOptimizer(num_classes=10, input_dim=X_train.shape[1])
    # Fast training mapping
    model.train(X_train, y_train, X_val, y_val)
    
    # 2. Setup Evaluation Tools
    pass_manager = PassManager(num_classes=10)
    pass_evaluator = PassEvaluator(pass_manager)
    priority_system = MetricPriorityVector()
    
    if priority_preset not in priority_system.presets:
        logger.warning(f"Preset '{priority_preset}' not found. Defaulting to 'balanced'.")
        priority_preset = "balanced"
    
    pv = priority_system.get_preset(priority_preset)
    
    # 3. Extract target program features
    logger.info(f"\n[2/4] Extracting LLVM IR Features for {os.path.basename(file_path)}...")
    feature_extractor = IRFeatureExtractor(normalize=True)
    prog = ProgramCharacteristics(name="demo_sample")
    prog.path = file_path
    # Re-using the logic from feature extractor locally
    try:
        from src.feature_extraction.autophase_features import AutophaseFeatureExtractor
        raw_feat_matrix = feature_extractor.extract_batch([prog])
        raw_features = raw_feat_matrix[0]
    except Exception as e:
        logger.error(f"Failed extracting features: {e}")
        return

    # 4. Predict Pass
    logger.info(f"\n[3/4] Predicting Optimal Pass Sequence...")
    augmented_features = np.concatenate([raw_features, pv]).reshape(1, -1)
    
    pred_start = time.time()
    predicted_class = model.predict(augmented_features)[0]
    pred_time = time.time() - pred_start
    
    predicted_seq = pass_manager.get_sequence(predicted_class)
    logger.info(f"Prediction complete in {pred_time*1000:.2f} ms")
    logger.info(f"Predicted Class ID: {predicted_class}")
    logger.info(f"Predicted Optimizer Passes: {predicted_seq}")
    
    # 5. Compile and Compare vs O3
    logger.info(f"\n[4/4] Compiling and Benchmarking (ML Sequence vs Clang -O3)")
    
    logger.info("Compiling with standard -O3...")
    o3_res = pass_evaluator.evaluate_standard_level(prog, "O3")
    
    logger.info("Compiling with ML Predicted sequence...")
    ml_res = pass_evaluator.evaluate(prog, predicted_seq, pv)
    
    # 6. Display Output
    logger.info("\n" + "="*60)
    logger.info("  FINAL COMPARISON RESULTS")
    logger.info("="*60)
    
    print(f"\n--- Baseline (-O3) ---")
    print(f"Compilation Overhead Ratio : {o3_res['compile_time_ratio']:.4f}x vs O0")
    print(f"Execution Speedup vs O0    : {o3_res['speedup']:.4f}x")
    print(f"Binary Size Ratio vs O0    : {o3_res['size_ratio']:.4f}x")
    
    print(f"\n--- ML Predicted Sequence ({priority_preset.upper()}) ---")
    print(f"Compilation Overhead Ratio : {ml_res['compile_time_ratio']:.4f}x vs O0")
    print(f"Execution Speedup vs O0    : {ml_res['speedup']:.4f}x")
    print(f"Binary Size Ratio vs O0    : {ml_res['size_ratio']:.4f}x")
    
    print(f"\n--- Relative Performance vs O3 ---")
    print(f"Execution Speed Ratio : {ml_res['speedup'] / o3_res['speedup']:.4f}x (Higher is better)")
    print(f"Binary File Size Ratio: {ml_res['size_ratio'] / o3_res['size_ratio']:.4f}x (Lower is better)")
    print(f"Compilation Overhead  : {ml_res['compile_time_ratio'] / o3_res['compile_time_ratio']:.4f}x")
    print("============================================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compiler Optimization ML Demo")
    parser.add_argument("--file", type=str, default=None, help="Path to sample C file")
    parser.add_argument("--priority", type=str, default="speed_first", 
                        choices=["speed_first", "size_first", "compile_time_first", "balanced"],
                        help="The priority objective for the ML model")
    
    args = parser.parse_args()
    run_demo(args.file, args.priority)

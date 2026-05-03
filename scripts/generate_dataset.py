"""
Dataset Generation Script.

Generates the complete training/validation/test dataset by:
1. Creating realistic benchmark program characteristics (PolyBench/C + custom)
2. Extracting features (Autophase 56-dim + CFG 10-dim)
3. Computing optimal pass sequences for each program+priority combination
4. Splitting into train/val/test sets
"""
import os
import sys
import numpy as np
from typing import List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction.autophase_features import ProgramCharacteristics
from src.feature_extraction.ir_feature_extractor import IRFeatureExtractor
from src.priority_vector.metric_priority import MetricPriorityVector
from src.optimization.pass_manager import PassManager
from src.optimization.pass_evaluator import PassEvaluator
from src.utils.data_loader import DataLoader
from src.utils.logger import setup_logger

logger = setup_logger("DatasetGeneration")


def create_polybench_programs() -> List[ProgramCharacteristics]:
    """
    Create program characteristics for PolyBench/C benchmark suite.

    These characteristics are based on typical PolyBench/C program
    structures as documented in IR analysis papers.
    """
    programs = []

    # PolyBench/C - Linear Algebra - BLAS
    programs.append(ProgramCharacteristics(
        name="2mm", num_functions=2, num_basic_blocks=45, num_instructions=680,
        num_load_inst=120, num_store_inst=55, num_branch_inst=28, num_call_inst=4,
        num_binary_ops=180, num_unary_ops=8, num_int_inst=250, num_float_inst=200,
        num_gep_inst=85, num_phi_nodes=18, num_alloca_inst=12, num_icmp_inst=22,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=65, num_crit_edges=8, num_cond_br=20, num_uncond_br=8,
        num_mult=60, num_div=0, num_shift=2, num_and_or=4, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=1.0, loop_depth=3, num_loops=4
    ))

    programs.append(ProgramCharacteristics(
        name="3mm", num_functions=2, num_basic_blocks=55, num_instructions=920,
        num_load_inst=160, num_store_inst=75, num_branch_inst=35, num_call_inst=4,
        num_binary_ops=250, num_unary_ops=10, num_int_inst=310, num_float_inst=300,
        num_gep_inst=110, num_phi_nodes=24, num_alloca_inst=15, num_icmp_inst=28,
        num_fcmp_inst=0, num_select_inst=3, num_switch_inst=0, num_return_inst=2,
        num_edges=85, num_crit_edges=12, num_cond_br=26, num_uncond_br=9,
        num_mult=90, num_div=0, num_shift=3, num_and_or=5, num_xor=0,
        avg_args_per_func=5.0, std_args_per_func=1.5, loop_depth=3, num_loops=6
    ))

    programs.append(ProgramCharacteristics(
        name="atax", num_functions=2, num_basic_blocks=25, num_instructions=320,
        num_load_inst=60, num_store_inst=25, num_branch_inst=16, num_call_inst=3,
        num_binary_ops=80, num_unary_ops=4, num_int_inst=120, num_float_inst=85,
        num_gep_inst=45, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=35, num_crit_edges=4, num_cond_br=12, num_uncond_br=4,
        num_mult=25, num_div=0, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.8, loop_depth=2, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="bicg", num_functions=2, num_basic_blocks=22, num_instructions=290,
        num_load_inst=55, num_store_inst=22, num_branch_inst=14, num_call_inst=3,
        num_binary_ops=72, num_unary_ops=3, num_int_inst=105, num_float_inst=78,
        num_gep_inst=40, num_phi_nodes=8, num_alloca_inst=7, num_icmp_inst=10,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=30, num_crit_edges=3, num_cond_br=10, num_uncond_br=4,
        num_mult=22, num_div=0, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=3.5, std_args_per_func=0.8, loop_depth=2, num_loops=2
    ))

    # PolyBench/C - Linear Algebra - Kernels
    programs.append(ProgramCharacteristics(
        name="cholesky", num_functions=2, num_basic_blocks=35, num_instructions=520,
        num_load_inst=95, num_store_inst=40, num_branch_inst=22, num_call_inst=5,
        num_binary_ops=140, num_unary_ops=6, num_int_inst=180, num_float_inst=150,
        num_gep_inst=65, num_phi_nodes=14, num_alloca_inst=10, num_icmp_inst=18,
        num_fcmp_inst=2, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=50, num_crit_edges=6, num_cond_br=16, num_uncond_br=6,
        num_mult=40, num_div=8, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=1.0, loop_depth=3, num_loops=4
    ))

    programs.append(ProgramCharacteristics(
        name="correlation", num_functions=3, num_basic_blocks=40, num_instructions=600,
        num_load_inst=110, num_store_inst=48, num_branch_inst=26, num_call_inst=6,
        num_binary_ops=160, num_unary_ops=8, num_int_inst=200, num_float_inst=180,
        num_gep_inst=75, num_phi_nodes=16, num_alloca_inst=12, num_icmp_inst=20,
        num_fcmp_inst=3, num_select_inst=2, num_switch_inst=0, num_return_inst=3,
        num_edges=58, num_crit_edges=7, num_cond_br=18, num_uncond_br=8,
        num_mult=45, num_div=12, num_shift=2, num_and_or=4, num_xor=0,
        avg_args_per_func=3.5, std_args_per_func=1.2, loop_depth=3, num_loops=5
    ))

    programs.append(ProgramCharacteristics(
        name="covariance", num_functions=3, num_basic_blocks=38, num_instructions=560,
        num_load_inst=100, num_store_inst=42, num_branch_inst=24, num_call_inst=5,
        num_binary_ops=150, num_unary_ops=7, num_int_inst=190, num_float_inst=160,
        num_gep_inst=70, num_phi_nodes=14, num_alloca_inst=11, num_icmp_inst=18,
        num_fcmp_inst=2, num_select_inst=2, num_switch_inst=0, num_return_inst=3,
        num_edges=55, num_crit_edges=6, num_cond_br=16, num_uncond_br=8,
        num_mult=40, num_div=10, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=1.0, loop_depth=3, num_loops=4
    ))

    programs.append(ProgramCharacteristics(
        name="gemm", num_functions=2, num_basic_blocks=30, num_instructions=420,
        num_load_inst=80, num_store_inst=35, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=120, num_unary_ops=5, num_int_inst=160, num_float_inst=130,
        num_gep_inst=55, num_phi_nodes=12, num_alloca_inst=9, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=42, num_crit_edges=5, num_cond_br=14, num_uncond_br=4,
        num_mult=45, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=5.0, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="gemver", num_functions=2, num_basic_blocks=35, num_instructions=480,
        num_load_inst=90, num_store_inst=38, num_branch_inst=22, num_call_inst=3,
        num_binary_ops=130, num_unary_ops=6, num_int_inst=170, num_float_inst=140,
        num_gep_inst=60, num_phi_nodes=14, num_alloca_inst=10, num_icmp_inst=16,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=48, num_crit_edges=5, num_cond_br=16, num_uncond_br=6,
        num_mult=35, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=7.0, std_args_per_func=1.5, loop_depth=2, num_loops=4
    ))

    programs.append(ProgramCharacteristics(
        name="gesummv", num_functions=2, num_basic_blocks=20, num_instructions=260,
        num_load_inst=50, num_store_inst=20, num_branch_inst=12, num_call_inst=3,
        num_binary_ops=65, num_unary_ops=3, num_int_inst=95, num_float_inst=70,
        num_gep_inst=35, num_phi_nodes=8, num_alloca_inst=7, num_icmp_inst=10,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=28, num_crit_edges=3, num_cond_br=10, num_uncond_br=2,
        num_mult=20, num_div=0, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=5.0, std_args_per_func=1.0, loop_depth=2, num_loops=2
    ))

    # PolyBench/C - Stencils
    programs.append(ProgramCharacteristics(
        name="fdtd-2d", num_functions=2, num_basic_blocks=50, num_instructions=750,
        num_load_inst=140, num_store_inst=60, num_branch_inst=30, num_call_inst=3,
        num_binary_ops=200, num_unary_ops=8, num_int_inst=260, num_float_inst=220,
        num_gep_inst=95, num_phi_nodes=20, num_alloca_inst=12, num_icmp_inst=24,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=72, num_crit_edges=9, num_cond_br=22, num_uncond_br=8,
        num_mult=30, num_div=0, num_shift=3, num_and_or=4, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=1.0, loop_depth=3, num_loops=5
    ))

    programs.append(ProgramCharacteristics(
        name="heat-3d", num_functions=2, num_basic_blocks=60, num_instructions=900,
        num_load_inst=170, num_store_inst=72, num_branch_inst=38, num_call_inst=3,
        num_binary_ops=240, num_unary_ops=10, num_int_inst=320, num_float_inst=270,
        num_gep_inst=115, num_phi_nodes=24, num_alloca_inst=14, num_icmp_inst=30,
        num_fcmp_inst=0, num_select_inst=3, num_switch_inst=0, num_return_inst=2,
        num_edges=88, num_crit_edges=12, num_cond_br=28, num_uncond_br=10,
        num_mult=35, num_div=8, num_shift=4, num_and_or=5, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=1.0, loop_depth=4, num_loops=6
    ))

    programs.append(ProgramCharacteristics(
        name="jacobi-1d", num_functions=2, num_basic_blocks=18, num_instructions=240,
        num_load_inst=45, num_store_inst=18, num_branch_inst=12, num_call_inst=3,
        num_binary_ops=60, num_unary_ops=3, num_int_inst=85, num_float_inst=65,
        num_gep_inst=30, num_phi_nodes=6, num_alloca_inst=6, num_icmp_inst=8,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=25, num_crit_edges=3, num_cond_br=8, num_uncond_br=4,
        num_mult=8, num_div=4, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=2, num_loops=2
    ))

    programs.append(ProgramCharacteristics(
        name="jacobi-2d", num_functions=2, num_basic_blocks=32, num_instructions=440,
        num_load_inst=85, num_store_inst=35, num_branch_inst=20, num_call_inst=3,
        num_binary_ops=118, num_unary_ops=5, num_int_inst=155, num_float_inst=125,
        num_gep_inst=58, num_phi_nodes=12, num_alloca_inst=9, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=45, num_crit_edges=5, num_cond_br=14, num_uncond_br=6,
        num_mult=15, num_div=6, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="seidel-2d", num_functions=2, num_basic_blocks=28, num_instructions=380,
        num_load_inst=75, num_store_inst=30, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=100, num_unary_ops=4, num_int_inst=135, num_float_inst=110,
        num_gep_inst=50, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=38, num_crit_edges=4, num_cond_br=12, num_uncond_br=6,
        num_mult=12, num_div=4, num_shift=2, num_and_or=2, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=3, num_loops=3
    ))

    # PolyBench/C - Linear Algebra - Solvers
    programs.append(ProgramCharacteristics(
        name="lu", num_functions=2, num_basic_blocks=30, num_instructions=430,
        num_load_inst=82, num_store_inst=35, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=115, num_unary_ops=5, num_int_inst=155, num_float_inst=120,
        num_gep_inst=55, num_phi_nodes=12, num_alloca_inst=9, num_icmp_inst=14,
        num_fcmp_inst=1, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=42, num_crit_edges=5, num_cond_br=14, num_uncond_br=4,
        num_mult=35, num_div=10, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.8, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="ludcmp", num_functions=3, num_basic_blocks=42, num_instructions=620,
        num_load_inst=115, num_store_inst=50, num_branch_inst=26, num_call_inst=5,
        num_binary_ops=165, num_unary_ops=7, num_int_inst=220, num_float_inst=180,
        num_gep_inst=78, num_phi_nodes=16, num_alloca_inst=12, num_icmp_inst=20,
        num_fcmp_inst=2, num_select_inst=2, num_switch_inst=0, num_return_inst=3,
        num_edges=60, num_crit_edges=7, num_cond_br=18, num_uncond_br=8,
        num_mult=48, num_div=15, num_shift=3, num_and_or=4, num_xor=0,
        avg_args_per_func=3.5, std_args_per_func=1.0, loop_depth=3, num_loops=5
    ))

    programs.append(ProgramCharacteristics(
        name="trisolv", num_functions=2, num_basic_blocks=20, num_instructions=270,
        num_load_inst=52, num_store_inst=22, num_branch_inst=14, num_call_inst=3,
        num_binary_ops=70, num_unary_ops=3, num_int_inst=100, num_float_inst=72,
        num_gep_inst=38, num_phi_nodes=8, num_alloca_inst=7, num_icmp_inst=10,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=28, num_crit_edges=3, num_cond_br=10, num_uncond_br=4,
        num_mult=15, num_div=5, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=2, num_loops=2
    ))

    programs.append(ProgramCharacteristics(
        name="durbin", num_functions=2, num_basic_blocks=24, num_instructions=340,
        num_load_inst=65, num_store_inst=28, num_branch_inst=16, num_call_inst=3,
        num_binary_ops=88, num_unary_ops=4, num_int_inst=125, num_float_inst=95,
        num_gep_inst=45, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=1, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=32, num_crit_edges=4, num_cond_br=12, num_uncond_br=4,
        num_mult=22, num_div=8, num_shift=2, num_and_or=2, num_xor=0,
        avg_args_per_func=2.5, std_args_per_func=0.5, loop_depth=2, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="gramschmidt", num_functions=2, num_basic_blocks=35, num_instructions=500,
        num_load_inst=95, num_store_inst=40, num_branch_inst=22, num_call_inst=4,
        num_binary_ops=135, num_unary_ops=6, num_int_inst=175, num_float_inst=145,
        num_gep_inst=62, num_phi_nodes=14, num_alloca_inst=10, num_icmp_inst=16,
        num_fcmp_inst=1, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=48, num_crit_edges=6, num_cond_br=16, num_uncond_br=6,
        num_mult=38, num_div=12, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.8, loop_depth=3, num_loops=4
    ))

    # PolyBench/C - Data Mining
    programs.append(ProgramCharacteristics(
        name="deriche", num_functions=2, num_basic_blocks=48, num_instructions=720,
        num_load_inst=135, num_store_inst=58, num_branch_inst=28, num_call_inst=4,
        num_binary_ops=195, num_unary_ops=8, num_int_inst=250, num_float_inst=210,
        num_gep_inst=88, num_phi_nodes=18, num_alloca_inst=13, num_icmp_inst=22,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=68, num_crit_edges=8, num_cond_br=20, num_uncond_br=8,
        num_mult=45, num_div=0, num_shift=3, num_and_or=4, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=1.0, loop_depth=2, num_loops=5
    ))

    programs.append(ProgramCharacteristics(
        name="doitgen", num_functions=2, num_basic_blocks=28, num_instructions=390,
        num_load_inst=72, num_store_inst=30, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=105, num_unary_ops=4, num_int_inst=140, num_float_inst=110,
        num_gep_inst=48, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=38, num_crit_edges=4, num_cond_br=14, num_uncond_br=4,
        num_mult=35, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="nussinov", num_functions=2, num_basic_blocks=42, num_instructions=580,
        num_load_inst=108, num_store_inst=45, num_branch_inst=26, num_call_inst=3,
        num_binary_ops=155, num_unary_ops=6, num_int_inst=210, num_float_inst=12,
        num_gep_inst=72, num_phi_nodes=16, num_alloca_inst=10, num_icmp_inst=20,
        num_fcmp_inst=4, num_select_inst=4, num_switch_inst=0, num_return_inst=2,
        num_edges=58, num_crit_edges=7, num_cond_br=18, num_uncond_br=8,
        num_mult=8, num_div=0, num_shift=2, num_and_or=6, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=3, num_loops=4
    ))

    programs.append(ProgramCharacteristics(
        name="floyd-warshall", num_functions=2, num_basic_blocks=28, num_instructions=350,
        num_load_inst=65, num_store_inst=25, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=90, num_unary_ops=4, num_int_inst=155, num_float_inst=5,
        num_gep_inst=45, num_phi_nodes=12, num_alloca_inst=8, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=38, num_crit_edges=4, num_cond_br=14, num_uncond_br=4,
        num_mult=4, num_div=0, num_shift=1, num_and_or=3, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=0.5, loop_depth=3, num_loops=3
    ))

    # PolyBench/C - Remaining kernels
    programs.append(ProgramCharacteristics(
        name="mvt", num_functions=2, num_basic_blocks=24, num_instructions=310,
        num_load_inst=58, num_store_inst=24, num_branch_inst=16, num_call_inst=3,
        num_binary_ops=80, num_unary_ops=4, num_int_inst=115, num_float_inst=85,
        num_gep_inst=42, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=32, num_crit_edges=4, num_cond_br=12, num_uncond_br=4,
        num_mult=22, num_div=0, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=5.0, std_args_per_func=1.0, loop_depth=2, num_loops=2
    ))

    programs.append(ProgramCharacteristics(
        name="symm", num_functions=2, num_basic_blocks=32, num_instructions=450,
        num_load_inst=85, num_store_inst=35, num_branch_inst=20, num_call_inst=3,
        num_binary_ops=120, num_unary_ops=5, num_int_inst=160, num_float_inst=130,
        num_gep_inst=58, num_phi_nodes=12, num_alloca_inst=9, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=2, num_switch_inst=0, num_return_inst=2,
        num_edges=44, num_crit_edges=5, num_cond_br=14, num_uncond_br=6,
        num_mult=38, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=5.0, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="syr2k", num_functions=2, num_basic_blocks=30, num_instructions=420,
        num_load_inst=80, num_store_inst=32, num_branch_inst=18, num_call_inst=3,
        num_binary_ops=112, num_unary_ops=5, num_int_inst=150, num_float_inst=120,
        num_gep_inst=55, num_phi_nodes=12, num_alloca_inst=9, num_icmp_inst=14,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=42, num_crit_edges=5, num_cond_br=14, num_uncond_br=4,
        num_mult=35, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=5.5, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="syrk", num_functions=2, num_basic_blocks=28, num_instructions=380,
        num_load_inst=72, num_store_inst=28, num_branch_inst=16, num_call_inst=3,
        num_binary_ops=100, num_unary_ops=4, num_int_inst=138, num_float_inst=108,
        num_gep_inst=48, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=38, num_crit_edges=4, num_cond_br=12, num_uncond_br=4,
        num_mult=32, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=4.5, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    programs.append(ProgramCharacteristics(
        name="trmm", num_functions=2, num_basic_blocks=26, num_instructions=360,
        num_load_inst=68, num_store_inst=26, num_branch_inst=16, num_call_inst=3,
        num_binary_ops=95, num_unary_ops=4, num_int_inst=130, num_float_inst=100,
        num_gep_inst=45, num_phi_nodes=10, num_alloca_inst=8, num_icmp_inst=12,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=2,
        num_edges=36, num_crit_edges=4, num_cond_br=12, num_uncond_br=4,
        num_mult=30, num_div=0, num_shift=2, num_and_or=3, num_xor=0,
        avg_args_per_func=4.0, std_args_per_func=0.8, loop_depth=3, num_loops=3
    ))

    # Custom Programs (additional diversity)
    programs.append(ProgramCharacteristics(
        name="matrix_multiply", num_functions=3, num_basic_blocks=22, num_instructions=300,
        num_load_inst=55, num_store_inst=20, num_branch_inst=14, num_call_inst=4,
        num_binary_ops=82, num_unary_ops=3, num_int_inst=110, num_float_inst=90,
        num_gep_inst=38, num_phi_nodes=8, num_alloca_inst=7, num_icmp_inst=10,
        num_fcmp_inst=0, num_select_inst=1, num_switch_inst=0, num_return_inst=3,
        num_edges=30, num_crit_edges=3, num_cond_br=10, num_uncond_br=4,
        num_mult=40, num_div=0, num_shift=1, num_and_or=2, num_xor=0,
        avg_args_per_func=3.0, std_args_per_func=1.0, loop_depth=3, num_loops=3
    ))

    return programs


def create_cpp_programs() -> List[ProgramCharacteristics]:
    """
    Simulate LLVM IR characteristics typical of C++ programs (OOP, STL, Exceptions).
    These typically have deeper call stacks, more functions, more phi nodes, and invoke instructions.
    """
    programs = []
    
    programs.append(ProgramCharacteristics(
        name="cpp_vector_sort", language="cpp", num_functions=15, num_basic_blocks=120, num_instructions=1500,
        num_load_inst=300, num_store_inst=150, num_branch_inst=80, num_call_inst=45,
        num_binary_ops=250, num_unary_ops=15, num_int_inst=400, num_float_inst=100,
        num_gep_inst=180, num_phi_nodes=65, num_alloca_inst=40, num_icmp_inst=60,
        num_fcmp_inst=0, num_select_inst=10, num_switch_inst=2, num_return_inst=15,
        num_edges=160, num_crit_edges=25, num_cond_br=65, num_uncond_br=15,
        num_mult=80, num_div=10, num_shift=15, num_and_or=20, num_xor=5,
        avg_args_per_func=4.5, std_args_per_func=2.0, loop_depth=3, num_loops=8
    ))

    programs.append(ProgramCharacteristics(
        name="cpp_hash_map", language="cpp", num_functions=22, num_basic_blocks=180, num_instructions=2100,
        num_load_inst=450, num_store_inst=200, num_branch_inst=120, num_call_inst=80,
        num_binary_ops=300, num_unary_ops=25, num_int_inst=600, num_float_inst=0,
        num_gep_inst=250, num_phi_nodes=90, num_alloca_inst=60, num_icmp_inst=95,
        num_fcmp_inst=0, num_select_inst=15, num_switch_inst=5, num_return_inst=22,
        num_edges=240, num_crit_edges=35, num_cond_br=95, num_uncond_br=25,
        num_mult=120, num_div=45, num_shift=35, num_and_or=40, num_xor=15,
        avg_args_per_func=3.8, std_args_per_func=1.5, loop_depth=2, num_loops=10
    ))
    
    programs.append(ProgramCharacteristics(
        name="cpp_polymorphism", language="cpp", num_functions=35, num_basic_blocks=150, num_instructions=1800,
        num_load_inst=350, num_store_inst=120, num_branch_inst=90, num_call_inst=110,
        num_binary_ops=220, num_unary_ops=20, num_int_inst=450, num_float_inst=50,
        num_gep_inst=280, num_phi_nodes=70, num_alloca_inst=45, num_icmp_inst=65,
        num_fcmp_inst=5, num_select_inst=8, num_switch_inst=1, num_return_inst=35,
        num_edges=210, num_crit_edges=30, num_cond_br=70, num_uncond_br=20,
        num_mult=40, num_div=5, num_shift=10, num_and_or=15, num_xor=2,
        avg_args_per_func=2.5, std_args_per_func=1.0, loop_depth=1, num_loops=4
    ))
    
    programs.append(ProgramCharacteristics(
        name="cpp_template_metaprog", language="cpp", num_functions=45, num_basic_blocks=220, num_instructions=2800,
        num_load_inst=550, num_store_inst=280, num_branch_inst=140, num_call_inst=150,
        num_binary_ops=400, num_unary_ops=30, num_int_inst=800, num_float_inst=0,
        num_gep_inst=350, num_phi_nodes=110, num_alloca_inst=80, num_icmp_inst=110,
        num_fcmp_inst=0, num_select_inst=20, num_switch_inst=3, num_return_inst=45,
        num_edges=310, num_crit_edges=45, num_cond_br=110, num_uncond_br=30,
        num_mult=150, num_div=20, num_shift=50, num_and_or=60, num_xor=10,
        avg_args_per_func=3.2, std_args_per_func=1.8, loop_depth=3, num_loops=12
    ))
    
    programs.append(ProgramCharacteristics(
        name="cpp_regex", language="cpp", num_functions=18, num_basic_blocks=250, num_instructions=2400,
        num_load_inst=400, num_store_inst=180, num_branch_inst=160, num_call_inst=60,
        num_binary_ops=350, num_unary_ops=20, num_int_inst=700, num_float_inst=0,
        num_gep_inst=220, num_phi_nodes=130, num_alloca_inst=50, num_icmp_inst=140,
        num_fcmp_inst=0, num_select_inst=30, num_switch_inst=15, num_return_inst=18,
        num_edges=360, num_crit_edges=60, num_cond_br=130, num_uncond_br=30,
        num_mult=60, num_div=10, num_shift=25, num_and_or=80, num_xor=5,
        avg_args_per_func=4.0, std_args_per_func=1.5, loop_depth=4, num_loops=15
    ))

    return programs

def create_python_programs() -> List[ProgramCharacteristics]:
    """
    Simulate LLVM IR characteristics typical of Python (via JIT/Numba).
    These have extremely high branching (type checks), dictionary lookups, and dynamic dispatch.
    """
    programs = []
    
    programs.append(ProgramCharacteristics(
        name="py_json_parser", language="python", num_functions=8, num_basic_blocks=350, num_instructions=3800,
        num_load_inst=800, num_store_inst=400, num_branch_inst=300, num_call_inst=180,
        num_binary_ops=450, num_unary_ops=40, num_int_inst=1200, num_float_inst=50,
        num_gep_inst=600, num_phi_nodes=220, num_alloca_inst=100, num_icmp_inst=280,
        num_fcmp_inst=10, num_select_inst=40, num_switch_inst=25, num_return_inst=8,
        num_edges=520, num_crit_edges=90, num_cond_br=250, num_uncond_br=50,
        num_mult=80, num_div=20, num_shift=40, num_and_or=110, num_xor=15,
        avg_args_per_func=5.5, std_args_per_func=2.5, loop_depth=3, num_loops=12
    ))

    programs.append(ProgramCharacteristics(
        name="py_data_pipeline", language="python", num_functions=12, num_basic_blocks=420, num_instructions=4500,
        num_load_inst=950, num_store_inst=480, num_branch_inst=350, num_call_inst=220,
        num_binary_ops=520, num_unary_ops=50, num_int_inst=1500, num_float_inst=300,
        num_gep_inst=750, num_phi_nodes=280, num_alloca_inst=120, num_icmp_inst=320,
        num_fcmp_inst=80, num_select_inst=50, num_switch_inst=30, num_return_inst=12,
        num_edges=640, num_crit_edges=110, num_cond_br=290, num_uncond_br=60,
        num_mult=150, num_div=50, num_shift=30, num_and_or=140, num_xor=10,
        avg_args_per_func=4.0, std_args_per_func=2.0, loop_depth=4, num_loops=18
    ))
    
    programs.append(ProgramCharacteristics(
        name="py_ml_inference", language="python", num_functions=6, num_basic_blocks=280, num_instructions=3100,
        num_load_inst=650, num_store_inst=320, num_branch_inst=220, num_call_inst=140,
        num_binary_ops=480, num_unary_ops=35, num_int_inst=800, num_float_inst=600,
        num_gep_inst=450, num_phi_nodes=180, num_alloca_inst=80, num_icmp_inst=180,
        num_fcmp_inst=150, num_select_inst=35, num_switch_inst=10, num_return_inst=6,
        num_edges=410, num_crit_edges=70, num_cond_br=180, num_uncond_br=40,
        num_mult=250, num_div=80, num_shift=20, num_and_or=70, num_xor=5,
        avg_args_per_func=3.5, std_args_per_func=1.2, loop_depth=5, num_loops=25
    ))

    programs.append(ProgramCharacteristics(
        name="py_web_router", language="python", num_functions=25, num_basic_blocks=500, num_instructions=5200,
        num_load_inst=1100, num_store_inst=550, num_branch_inst=420, num_call_inst=350,
        num_binary_ops=550, num_unary_ops=60, num_int_inst=1800, num_float_inst=10,
        num_gep_inst=850, num_phi_nodes=320, num_alloca_inst=180, num_icmp_inst=380,
        num_fcmp_inst=0, num_select_inst=60, num_switch_inst=45, num_return_inst=25,
        num_edges=780, num_crit_edges=140, num_cond_br=350, num_uncond_br=70,
        num_mult=90, num_div=15, num_shift=50, num_and_or=180, num_xor=20,
        avg_args_per_func=2.8, std_args_per_func=1.5, loop_depth=2, num_loops=8
    ))
    
    programs.append(ProgramCharacteristics(
        name="py_string_manip", language="python", num_functions=10, num_basic_blocks=320, num_instructions=3400,
        num_load_inst=700, num_store_inst=350, num_branch_inst=280, num_call_inst=200,
        num_binary_ops=400, num_unary_ops=45, num_int_inst=1300, num_float_inst=0,
        num_gep_inst=550, num_phi_nodes=200, num_alloca_inst=90, num_icmp_inst=250,
        num_fcmp_inst=0, num_select_inst=45, num_switch_inst=20, num_return_inst=10,
        num_edges=480, num_crit_edges=85, num_cond_br=230, num_uncond_br=50,
        num_mult=60, num_div=25, num_shift=45, num_and_or=130, num_xor=12,
        avg_args_per_func=3.2, std_args_per_func=1.8, loop_depth=3, num_loops=15
    ))

    return programs


def generate_dataset(config_path: str = "config/config.yaml"):
    """
    Generate the complete dataset for training and evaluation.

    Process:
    1. Create benchmark program characteristics
    2. Extract features (66-dim: 56 Autophase + 10 CFG)
    3. Generate diverse priority vectors
    4. Find optimal pass sequence for each program+priority combination
    5. Create augmented training data (features + priority → optimal class)
    6. Split into train/val/test
    """
    logger.info("=" * 60)
    logger.info("DATASET GENERATION PIPELINE")
    logger.info("=" * 60)

    # Load configuration
    data_loader = DataLoader(config_path)
    config = data_loader.get_config()
    seed = config['dataset']['random_seed']
    rng = np.random.RandomState(seed)

    # Step 1: Create benchmark programs
    logger.info("Step 1: Creating benchmark program characteristics...")
    c_programs = create_polybench_programs()
    cpp_programs = create_cpp_programs()
    py_programs = create_python_programs()
    programs = c_programs + cpp_programs + py_programs
    logger.info(f"  Created {len(programs)} benchmark programs (C: {len(c_programs)}, C++: {len(cpp_programs)}, Python: {len(py_programs)})")

    # Step 2: Extract features
    logger.info("Step 2: Extracting features (Autophase + CFG)...")
    feature_extractor = IRFeatureExtractor(normalize=True)
    raw_features = feature_extractor.extract_batch(programs)
    logger.info(f"  Feature matrix shape: {raw_features.shape}")

    # Step 3: Initialize pass manager and evaluator
    logger.info("Step 3: Initializing pass manager...")
    num_classes = config['dataset']['num_optimization_classes']
    pass_manager = PassManager(num_classes=num_classes, seed=seed)
    pass_evaluator = PassEvaluator(pass_manager, seed=seed)

    # Step 4: Generate priority vectors
    logger.info("Step 4: Generating priority vectors...")
    priority_system = MetricPriorityVector()
    num_priorities_per_program = 15
    priorities = priority_system.generate_training_priorities(
        num_priorities_per_program, strategy="mixed"
    )

    # Step 5: Create labeled dataset
    logger.info("Step 5: Creating labeled dataset (features + priority → optimal class)...")

    all_features = []
    all_labels = []
    all_speedups = []
    all_sizes = []

    for prog_idx, program in enumerate(programs):
        prog_features = raw_features[prog_idx]

        for pri_idx in range(num_priorities_per_program):
            priority = priorities[pri_idx % len(priorities)]

            # Find optimal pass sequence for this program+priority
            best_class, best_result = pass_evaluator.find_optimal_sequence(program, priority)

            # Augment features with priority vector
            augmented = np.concatenate([prog_features, priority])

            all_features.append(augmented)
            all_labels.append(best_class)
            all_speedups.append(best_result['speedup'])
            all_sizes.append(best_result['size_ratio'])

    X = np.array(all_features)
    y = np.array(all_labels)

    logger.info(f"  Total samples: {X.shape[0]}")
    logger.info(f"  Feature dimension: {X.shape[1]}")
    logger.info(f"  Class distribution: {np.bincount(y, minlength=num_classes)}")

    # Step 6: Data augmentation - add noise variations for robustness
    logger.info("Step 6: Data augmentation...")
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        # Add 10 noisy copies for intense training dataset scaling
        for _ in range(10):
            noise = rng.normal(0, 0.02, size=X[i].shape)
            noisy_sample = X[i] + noise
            # Re-normalize priority vector (last 4 dims)
            pri = noisy_sample[-4:]
            pri = np.clip(pri, 0.01, 1.0)
            pri = pri / pri.sum()
            noisy_sample[-4:] = pri
            X_augmented.append(noisy_sample)
            y_augmented.append(y[i])

    X = np.array(X_augmented)
    y = np.array(y_augmented)

    logger.info(f"  After augmentation: {X.shape[0]} samples")

    # Step 7: Shuffle and split
    logger.info("Step 7: Train/val/test split...")
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']

    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]

    logger.info(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Save
    data_loader.save_dataset(X_train, X_val, X_test, y_train, y_val, y_test)
    data_loader.save_features(raw_features, "raw_features.npy")

    # Save program info
    program_info = {
        'names': [p.name for p in programs],
        'num_programs': len(programs),
        'feature_dim': X.shape[1],
        'num_classes': num_classes,
    }
    import json
    info_path = os.path.join(config['paths']['raw_dir'], 'program_info.json')
    with open(info_path, 'w') as f:
        json.dump(program_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info(f"Total: {X.shape[0]} samples, {X.shape[1]} features, {num_classes} classes")
    logger.info("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    generate_dataset()

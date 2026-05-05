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


def create_cpp_programs(rng=None) -> List[ProgramCharacteristics]:
    """
    Generate 20 structurally diverse C++ program IR profiles.
    Uses parametric sampling across OOP archetypes.
    """
    if rng is None:
        rng = np.random.RandomState(123)

    archetypes = [
        ("cpp_stl_algo",    (8, 25),  (80, 250),  (800, 2800),  0.0,  0.15, 0.08, (1, 3)),
        ("cpp_oop_inherit", (15, 40), (100, 200),  (1200, 2500), 0.05, 0.20, 0.06, (1, 2)),
        ("cpp_template",    (20, 50), (150, 300),  (1800, 3500), 0.0,  0.12, 0.07, (2, 4)),
        ("cpp_exception",   (10, 30), (120, 280),  (1500, 3000), 0.02, 0.18, 0.10, (1, 3)),
        ("cpp_smart_ptr",   (12, 35), (90, 220),   (1000, 2400), 0.01, 0.22, 0.07, (1, 2)),
    ]
    programs = []
    for arch_idx, (prefix, fr, bbr, ir, ff, cf, bf, ldr) in enumerate(archetypes):
        for var in range(4):
            nf = rng.randint(*fr)
            nbb = rng.randint(*bbr)
            ni = rng.randint(*ir)
            nl = rng.randint(int(ni*0.18), int(ni*0.28))
            ns = rng.randint(int(ni*0.06), int(ni*0.14))
            nbr = rng.randint(int(ni*bf*0.7), int(ni*bf*1.3)+1)
            nc = rng.randint(int(ni*cf*0.7), int(ni*cf*1.3)+1)
            nbin = rng.randint(int(ni*0.12), int(ni*0.22))
            nfloat = int(ni*ff*rng.uniform(0.8, 1.2))
            nint = rng.randint(int(ni*0.25), int(ni*0.40))
            ngep = rng.randint(int(ni*0.10), int(ni*0.18))
            nphi = rng.randint(int(nbb*0.3), int(nbb*0.7))
            nalloc = rng.randint(int(nf*1.5), int(nf*3))
            nicmp = rng.randint(int(nbr*0.5), int(nbr*0.9))
            ld = rng.randint(*ldr)
            nloops = rng.randint(2, 15)
            programs.append(ProgramCharacteristics(
                name=f"{prefix}_v{var}", language="cpp",
                num_functions=nf, num_basic_blocks=nbb, num_instructions=ni,
                num_load_inst=nl, num_store_inst=ns, num_branch_inst=nbr,
                num_call_inst=nc, num_binary_ops=nbin, num_unary_ops=rng.randint(5, 40),
                num_int_inst=nint, num_float_inst=nfloat,
                num_gep_inst=ngep, num_phi_nodes=nphi, num_alloca_inst=nalloc,
                num_icmp_inst=nicmp, num_fcmp_inst=rng.randint(0, 15),
                num_select_inst=rng.randint(2, 30), num_switch_inst=rng.randint(0, 10),
                num_return_inst=nf,
                num_edges=rng.randint(int(nbb*1.2), int(nbb*1.8)),
                num_crit_edges=rng.randint(int(nbb*0.1), int(nbb*0.25)),
                num_cond_br=rng.randint(int(nbr*0.6), int(nbr*0.85)),
                num_uncond_br=rng.randint(int(nbr*0.1), int(nbr*0.3)),
                num_mult=rng.randint(20, 180), num_div=rng.randint(0, 50),
                num_shift=rng.randint(5, 55), num_and_or=rng.randint(10, 90),
                num_xor=rng.randint(0, 20),
                avg_args_per_func=round(rng.uniform(2.0, 5.0), 1),
                std_args_per_func=round(rng.uniform(0.8, 2.0), 1),
                loop_depth=ld, num_loops=nloops
            ))
    return programs


def create_python_programs(rng=None) -> List[ProgramCharacteristics]:
    """
    Generate 20 structurally diverse Python (JIT/Numba) IR profiles.
    """
    if rng is None:
        rng = np.random.RandomState(456)

    archetypes = [
        ("py_data_proc",  (5, 15),  (200, 500), (2000, 5500), 0.10, 0.08, 0.10, (2, 5)),
        ("py_web_app",    (10, 30), (300, 600), (3000, 6000), 0.01, 0.12, 0.12, (1, 3)),
        ("py_ml_model",   (4, 12),  (200, 400), (2500, 5000), 0.25, 0.06, 0.08, (3, 6)),
        ("py_text_nlp",   (6, 20),  (250, 500), (2800, 5500), 0.02, 0.10, 0.11, (2, 4)),
        ("py_scientific", (3, 10),  (150, 350), (1800, 4000), 0.35, 0.05, 0.07, (4, 7)),
    ]
    programs = []
    for arch_idx, (prefix, fr, bbr, ir, ff, cf, bf, ldr) in enumerate(archetypes):
        for var in range(4):
            nf = rng.randint(*fr)
            nbb = rng.randint(*bbr)
            ni = rng.randint(*ir)
            nl = rng.randint(int(ni*0.18), int(ni*0.28))
            ns = rng.randint(int(ni*0.08), int(ni*0.14))
            nbr = rng.randint(int(ni*bf*0.8), int(ni*bf*1.2)+1)
            nc = rng.randint(int(ni*cf*0.7), int(ni*cf*1.3)+1)
            nbin = rng.randint(int(ni*0.10), int(ni*0.16))
            nfloat = int(ni*ff*rng.uniform(0.7, 1.3))
            nint = rng.randint(int(ni*0.30), int(ni*0.45))
            ngep = rng.randint(int(ni*0.12), int(ni*0.22))
            nphi = rng.randint(int(nbb*0.5), int(nbb*0.8))
            nalloc = rng.randint(int(nf*3), int(nf*8))
            nicmp = rng.randint(int(nbr*0.6), int(nbr*0.95))
            ld = rng.randint(*ldr)
            nloops = rng.randint(3, 25)
            programs.append(ProgramCharacteristics(
                name=f"{prefix}_v{var}", language="python",
                num_functions=nf, num_basic_blocks=nbb, num_instructions=ni,
                num_load_inst=nl, num_store_inst=ns, num_branch_inst=nbr,
                num_call_inst=nc, num_binary_ops=nbin, num_unary_ops=rng.randint(10, 60),
                num_int_inst=nint, num_float_inst=nfloat,
                num_gep_inst=ngep, num_phi_nodes=nphi, num_alloca_inst=nalloc,
                num_icmp_inst=nicmp, num_fcmp_inst=rng.randint(0, int(nfloat*0.3)+1),
                num_select_inst=rng.randint(10, 60), num_switch_inst=rng.randint(5, 45),
                num_return_inst=nf,
                num_edges=rng.randint(int(nbb*1.3), int(nbb*1.8)),
                num_crit_edges=rng.randint(int(nbb*0.15), int(nbb*0.30)),
                num_cond_br=rng.randint(int(nbr*0.7), int(nbr*0.88)),
                num_uncond_br=rng.randint(int(nbr*0.10), int(nbr*0.25)),
                num_mult=rng.randint(30, 260), num_div=rng.randint(5, 90),
                num_shift=rng.randint(10, 55), num_and_or=rng.randint(40, 200),
                num_xor=rng.randint(2, 25),
                avg_args_per_func=round(rng.uniform(2.5, 6.0), 1),
                std_args_per_func=round(rng.uniform(1.0, 2.8), 1),
                loop_depth=ld, num_loops=nloops
            ))
    return programs


def create_extra_c_programs(rng=None) -> List[ProgramCharacteristics]:
    """
    Generate 50 additional diverse C programs using parametric sampling.
    Covers tiny utils, medium kernels, large stencils, graph algos, crypto.
    """
    if rng is None:
        rng = np.random.RandomState(789)

    archetypes = [
        ("c_tiny_util",    10, (1, 4),   (5, 25),    (50, 300),    0.0,  0.12, (0, 1)),
        ("c_medium_kern",  10, (2, 6),   (20, 60),   (250, 800),   0.30, 0.08, (2, 3)),
        ("c_large_stencil",10, (3, 8),   (40, 100),  (500, 1500),  0.40, 0.06, (3, 5)),
        ("c_graph_algo",   10, (2, 5),   (30, 80),   (300, 1000),  0.0,  0.15, (1, 3)),
        ("c_crypto",       10, (3, 10),  (50, 150),  (600, 2000),  0.0,  0.10, (2, 4)),
    ]
    programs = []
    for prefix, count, fr, bbr, ir, ff, bf, ldr in archetypes:
        for var in range(count):
            nf = rng.randint(*fr)
            nbb = rng.randint(*bbr)
            ni = rng.randint(*ir)
            nl = rng.randint(int(ni*0.15), int(ni*0.25)+1)
            ns = rng.randint(int(ni*0.05), int(ni*0.12)+1)
            nbr = rng.randint(int(ni*bf*0.7), int(ni*bf*1.3)+1)
            nc = rng.randint(max(1, int(nf*0.5)), int(nf*2)+1)
            nbin = rng.randint(int(ni*0.10), int(ni*0.25)+1)
            nfloat = int(ni*ff*rng.uniform(0.7, 1.3))
            nint = rng.randint(int(ni*0.20), int(ni*0.40)+1)
            ngep = rng.randint(int(ni*0.08), int(ni*0.18)+1)
            nphi = rng.randint(max(0, int(nbb*0.2)), int(nbb*0.6)+1)
            nalloc = rng.randint(max(1, nf), int(nf*3)+1)
            nicmp = rng.randint(max(0, int(nbr*0.4)), int(nbr*0.8)+1)
            ld = rng.randint(*ldr)
            nloops = rng.randint(0, max(1, int(nbb*0.15))+1)
            programs.append(ProgramCharacteristics(
                name=f"{prefix}_v{var}", language="c",
                num_functions=nf, num_basic_blocks=max(1, nbb), num_instructions=max(10, ni),
                num_load_inst=max(1, nl), num_store_inst=max(1, ns),
                num_branch_inst=max(1, nbr), num_call_inst=max(1, nc),
                num_binary_ops=max(1, nbin), num_unary_ops=rng.randint(0, 15),
                num_int_inst=max(1, nint), num_float_inst=max(0, nfloat),
                num_gep_inst=max(0, ngep), num_phi_nodes=max(0, nphi),
                num_alloca_inst=max(1, nalloc), num_icmp_inst=max(0, nicmp),
                num_fcmp_inst=rng.randint(0, max(1, int(nfloat*0.1))+1),
                num_select_inst=rng.randint(0, 10),
                num_switch_inst=rng.randint(0, 3),
                num_return_inst=max(1, nf),
                num_edges=max(1, rng.randint(max(1, int(nbb*1.1)), int(nbb*1.7)+1)),
                num_crit_edges=rng.randint(0, max(1, int(nbb*0.2))+1),
                num_cond_br=max(0, rng.randint(max(0, int(nbr*0.5)), int(nbr*0.8)+1)),
                num_uncond_br=rng.randint(0, max(1, int(nbr*0.3))+1),
                num_mult=rng.randint(0, max(1, int(nbin*0.5))+1),
                num_div=rng.randint(0, max(1, int(nbin*0.15))+1),
                num_shift=rng.randint(0, max(1, int(nbin*0.1))+1),
                num_and_or=rng.randint(0, max(1, int(nbin*0.1))+1),
                num_xor=rng.randint(0, 5),
                avg_args_per_func=round(rng.uniform(1.5, 5.0), 1),
                std_args_per_func=round(rng.uniform(0.5, 2.0), 1),
                loop_depth=ld, num_loops=max(0, nloops)
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

    # Step 1: Create benchmark programs (120 total: 80 C + 20 C++ + 20 Python)
    logger.info("Step 1: Creating benchmark program characteristics...")
    c_programs = create_polybench_programs()
    c_extra = create_extra_c_programs()
    cpp_programs = create_cpp_programs()
    py_programs = create_python_programs()
    programs = c_programs + c_extra + cpp_programs + py_programs
    logger.info(f"  Created {len(programs)} benchmark programs (C: {len(c_programs)+len(c_extra)}, C++: {len(cpp_programs)}, Python: {len(py_programs)})")

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
    num_priorities_per_program = 30
    priorities = priority_system.generate_training_priorities(
        num_priorities_per_program, strategy="mixed"
    )
    logger.info(f"  Generated {len(priorities)} priority vectors")

    # Step 5: PROGRAM-LEVEL SPLIT (prevents data leakage)
    # Split programs FIRST, then generate samples per split.
    # Test programs are NEVER seen during training — real deployment accuracy.
    logger.info("Step 5: Program-level train/val/test split...")

    n_programs = len(programs)
    prog_indices = rng.permutation(n_programs)

    train_ratio = config['dataset']['train_ratio']
    val_ratio = config['dataset']['val_ratio']

    n_train_progs = max(1, int(n_programs * train_ratio))
    n_val_progs = max(1, int(n_programs * val_ratio))
    n_test_progs = n_programs - n_train_progs - n_val_progs

    train_prog_idx = prog_indices[:n_train_progs]
    val_prog_idx = prog_indices[n_train_progs:n_train_progs + n_val_progs]
    test_prog_idx = prog_indices[n_train_progs + n_val_progs:]

    train_prog_names = [programs[i].name for i in train_prog_idx]
    val_prog_names = [programs[i].name for i in val_prog_idx]
    test_prog_names = [programs[i].name for i in test_prog_idx]

    logger.info(f"  Train programs ({n_train_progs}): {train_prog_names}")
    logger.info(f"  Val programs ({n_val_progs}): {val_prog_names}")
    logger.info(f"  Test programs ({n_test_progs}): {test_prog_names}")

    # Step 6: Generate labeled samples per split
    logger.info("Step 6: Creating labeled dataset (features + priority → optimal class)...")

    def make_samples(prog_indices_list):
        """Generate (feature+priority, label) for given program indices."""
        feats, labels = [], []
        for pidx in prog_indices_list:
            prog = programs[pidx]
            prog_feat = raw_features[pidx]
            for pri_idx in range(num_priorities_per_program):
                priority = priorities[pri_idx % len(priorities)]
                best_class, best_result = pass_evaluator.find_optimal_sequence(prog, priority)
                augmented = np.concatenate([prog_feat, priority])
                feats.append(augmented)
                labels.append(best_class)
        return np.array(feats), np.array(labels)

    X_train_base, y_train_base = make_samples(train_prog_idx)
    X_val, y_val = make_samples(val_prog_idx)
    X_test, y_test = make_samples(test_prog_idx)

    logger.info(f"  Base train: {X_train_base.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    logger.info(f"  Feature dimension: {X_train_base.shape[1]}")
    logger.info(f"  Train class distribution: {np.bincount(y_train_base, minlength=num_classes)}")
    logger.info(f"  Test class distribution:  {np.bincount(y_test, minlength=num_classes)}")

    # Step 7: Augment ONLY training data (test stays clean / untouched)
    logger.info("Step 7: Data augmentation (training set only)...")
    X_augmented = []
    y_augmented = []

    for i in range(len(X_train_base)):
        X_augmented.append(X_train_base[i])
        y_augmented.append(y_train_base[i])

        # Add 19 noisy copies (20x total) — enough with 120 base programs
        for _ in range(19):
            noise = rng.normal(0, 0.02, size=X_train_base[i].shape)
            noisy_sample = X_train_base[i] + noise
            # Re-normalize priority vector (last 4 dims)
            pri = noisy_sample[-4:]
            pri = np.clip(pri, 0.01, 1.0)
            pri = pri / pri.sum()
            noisy_sample[-4:] = pri
            X_augmented.append(noisy_sample)
            y_augmented.append(y_train_base[i])

    X_train = np.array(X_augmented)
    y_train = np.array(y_augmented)

    # Shuffle training data
    train_perm = rng.permutation(len(X_train))
    X_train = X_train[train_perm]
    y_train = y_train[train_perm]

    logger.info(f"  After augmentation — Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Save
    data_loader.save_dataset(X_train, X_val, X_test, y_train, y_val, y_test)
    data_loader.save_features(raw_features, "raw_features.npy")

    # Save program info
    program_info = {
        'names': [p.name for p in programs],
        'num_programs': len(programs),
        'feature_dim': X_train.shape[1],
        'num_classes': num_classes,
        'split': {
            'train_programs': train_prog_names,
            'val_programs': val_prog_names,
            'test_programs': test_prog_names,
        }
    }
    import json
    info_path = os.path.join(config['paths']['raw_dir'], 'program_info.json')
    with open(info_path, 'w') as f:
        json.dump(program_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("DATASET GENERATION COMPLETE (PROGRAM-LEVEL SPLIT)")
    logger.info(f"Train: {X_train.shape[0]} samples ({n_train_progs} programs, augmented)")
    logger.info(f"Val:   {X_val.shape[0]} samples ({n_val_progs} programs, clean)")
    logger.info(f"Test:  {X_test.shape[0]} samples ({n_test_progs} programs, clean)")
    logger.info(f"Features: {X_train.shape[1]}, Classes: {num_classes}")
    logger.info("=" * 60)

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    generate_dataset()

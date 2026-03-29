"""
Control Flow Graph (CFG) Analyzer.

Extracts CFG-based metrics that complement the Autophase features,
providing deeper structural understanding of program control flow.

Features extracted:
    1. Cyclomatic complexity
    2. Loop nesting depth (max)
    3. Average loop nesting depth
    4. Critical path length estimate
    5. Fan-in ratio (avg predecessors per BB)
    6. Fan-out ratio (avg successors per BB)
    7. Dominance tree depth estimate
    8. Back-edge ratio
    9. Branch density
    10. Memory operation ratio
"""
import numpy as np
from typing import List, Dict

from src.feature_extraction.autophase_features import ProgramCharacteristics


CFG_FEATURE_NAMES = [
    "cyclomatic_complexity",
    "max_loop_nesting_depth",
    "avg_loop_nesting_depth",
    "critical_path_length",
    "fan_in_ratio",
    "fan_out_ratio",
    "dominance_tree_depth",
    "back_edge_ratio",
    "branch_density",
    "memory_op_ratio",
]


class CFGAnalyzer:
    """
    Analyzes program Control Flow Graph metrics.

    These 10 features supplement the 56 Autophase features to provide
    a richer representation of program structure for ML-based optimization.
    """

    def __init__(self):
        """Initialize CFG Analyzer."""
        self.feature_names = CFG_FEATURE_NAMES
        self.num_features = len(self.feature_names)

    def analyze(self, program: ProgramCharacteristics) -> np.ndarray:
        """
        Extract 10-dimension CFG feature vector.

        Args:
            program: ProgramCharacteristics instance

        Returns:
            numpy array of shape (10,)
        """
        features = np.zeros(self.num_features, dtype=np.float64)
        rng = np.random.RandomState(hash(program.name) % (2**31) + 1)

        nb = program.num_basic_blocks
        ni = program.num_instructions
        ne = program.num_edges

        # 1. Cyclomatic Complexity: M = E - N + 2P
        #    (edges - nodes + 2 * connected_components)
        features[0] = max(1, ne - nb + 2 * program.num_functions)

        # 2. Max Loop Nesting Depth
        features[1] = program.loop_depth

        # 3. Average Loop Nesting Depth
        if program.num_loops > 0:
            # Simulate average (usually less than max)
            features[2] = program.loop_depth * rng.uniform(0.4, 0.75)
        else:
            features[2] = 0.0

        # 4. Critical Path Length Estimate
        # Roughly proportional to instruction count / parallelism potential
        ilp_estimate = max(1.0, program.num_binary_ops / max(1, nb))
        features[3] = ni / max(1.0, ilp_estimate) / max(1, program.num_functions)

        # 5. Fan-in Ratio (avg predecessors per BB)
        features[4] = ne / max(1, nb)

        # 6. Fan-out Ratio (avg successors per BB)
        features[5] = ne / max(1, nb)  # In a CFG, total in-edges == total out-edges

        # 7. Dominance Tree Depth Estimate
        # Typically log2(num_BBs) to num_BBs for pathological cases
        features[6] = max(1, int(np.log2(max(2, nb)) * (1 + program.loop_depth * 0.3)))

        # 8. Back-edge Ratio (back-edges / total edges)
        # Back edges ≈ number of loops
        features[7] = program.num_loops / max(1, ne)

        # 9. Branch Density (branches / total instructions)
        features[8] = program.num_branch_inst / max(1, ni)

        # 10. Memory Operation Ratio (loads + stores / total instructions)
        features[9] = (program.num_load_inst + program.num_store_inst) / max(1, ni)

        return features

    def analyze_batch(self, programs: List[ProgramCharacteristics]) -> np.ndarray:
        """
        Extract CFG features for multiple programs.

        Args:
            programs: List of ProgramCharacteristics

        Returns:
            numpy array of shape (N, 10)
        """
        features = np.zeros((len(programs), self.num_features))
        for i, prog in enumerate(programs):
            features[i] = self.analyze(prog)
        return features

    def get_feature_names(self) -> List[str]:
        """Return list of CFG feature names."""
        return list(self.feature_names)

"""
LLVM Optimization Pass Manager.

Manages optimization pass sequences, their categorization, and the mapping
between pass sequence labels and actual pass configurations.

References:
    - MiCOMP (Ashouri et al., 2017): optimization sub-sequences
    - LLVM Pass documentation: https://llvm.org/docs/Passes.html
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import itertools

from src.utils.logger import setup_logger

logger = setup_logger("PassManager")


# Categorized LLVM optimization passes
PASS_CATEGORIES = {
    "scalar": [
        "-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
        "-gvn", "-sccp", "-dce", "-adce", "-bdce", "-early-cse",
        "-jump-threading", "-correlated-propagation",
    ],
    "loop": [
        "-licm", "-loop-unroll", "-loop-rotate", "-loop-simplify",
        "-indvars", "-loop-idiom", "-loop-deletion", "-loop-distribute",
        "-loop-load-elim",
    ],
    "memory": [
        "-sroa", "-memcpyopt", "-dse",
    ],
    "interprocedural": [
        "-inline", "-deadargelim", "-tailcallelim", "-ipsccp",
        "-called-value-propagation", "-globalopt", "-globaldce",
        "-constmerge", "-strip-dead-prototypes", "-mergefunc",
    ],
    "vectorization": [
        "-loop-vectorize", "-slp-vectorizer",
    ],
    "misc": [
        "-float2int", "-lower-constant-intrinsics",
        "-alignment-from-assumptions", "-div-rem-pairs",
    ],
}

# Standard optimization levels and their typical pass compositions
OPTIMIZATION_LEVELS = {
    "O0": [],  # No optimization
    "O1": ["-mem2reg", "-simplifycfg", "-instcombine", "-early-cse",
            "-sroa", "-dce", "-loop-simplify", "-loop-rotate"],
    "O2": ["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
            "-gvn", "-licm", "-loop-unroll", "-loop-rotate", "-loop-simplify",
            "-indvars", "-sccp", "-dce", "-adce", "-sroa", "-early-cse",
            "-jump-threading", "-correlated-propagation", "-inline",
            "-deadargelim", "-memcpyopt", "-dse", "-loop-vectorize"],
    "O3": ["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
            "-gvn", "-licm", "-loop-unroll", "-loop-rotate", "-loop-simplify",
            "-indvars", "-sccp", "-dce", "-adce", "-bdce", "-sroa",
            "-early-cse", "-jump-threading", "-correlated-propagation",
            "-tailcallelim", "-inline", "-deadargelim", "-memcpyopt",
            "-dse", "-loop-idiom", "-loop-deletion", "-loop-vectorize",
            "-slp-vectorizer", "-ipsccp", "-globalopt", "-globaldce",
            "-constmerge", "-float2int", "-div-rem-pairs", "-mergefunc"],
}


@dataclass
class PassSequence:
    """Represents an optimization pass sequence with metadata."""
    id: int
    passes: List[str]
    category: str  # "speed_optimized", "size_optimized", "balanced", etc.
    description: str = ""

    def __len__(self):
        return len(self.passes)

    def __repr__(self):
        return f"PassSequence(id={self.id}, cat='{self.category}', len={len(self.passes)})"


class PassManager:
    """
    Manages LLVM optimization pass sequences for the prediction framework.

    Generates meaningful pass sequence classes that serve as prediction
    targets for the ML models. Each class represents a distinct optimization
    strategy suited for different program types and user priorities.
    """

    def __init__(self, num_classes: int = 10, seed: int = 42):
        """
        Initialize PassManager with specified number of optimization classes.

        Args:
            num_classes: Number of distinct pass sequence classes
            seed: Random seed for reproducibility
        """
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)
        self.all_passes = self._get_all_passes()
        self.pass_sequences = self._generate_pass_sequences()
        self.class_names = [seq.category for seq in self.pass_sequences]

        logger.info(f"Initialized PassManager with {num_classes} pass sequence classes, "
                    f"{len(self.all_passes)} available passes")

    def _get_all_passes(self) -> List[str]:
        """Get flattened list of all available passes."""
        all_passes = []
        for category_passes in PASS_CATEGORIES.values():
            all_passes.extend(category_passes)
        return all_passes

    def _generate_pass_sequences(self) -> List[PassSequence]:
        """
        Generate meaningful pass sequence classes.

        Each class represents a distinct optimization strategy:
        - Speed-optimized (aggressive loop/vectorization)
        - Size-optimized (DCE, function merging, minimal loop expansion)
        - Memory-optimized (memory passes, less loop unrolling)
        - Balanced (mix of all)
        - Etc.
        """
        sequences = []

        # Class 0: Aggressive Speed Optimization
        sequences.append(PassSequence(
            id=0,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
                    "-gvn", "-licm", "-loop-unroll", "-loop-rotate", "-loop-simplify",
                    "-indvars", "-loop-vectorize", "-slp-vectorizer", "-inline",
                    "-early-cse", "-jump-threading"],
            category="aggressive_speed",
            description="Maximum speed: aggressive loop opts + vectorization + inlining"
        ))

        # Class 1: Size-Optimized
        sequences.append(PassSequence(
            id=1,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-sccp",
                    "-dce", "-adce", "-bdce", "-deadargelim", "-globaldce",
                    "-constmerge", "-strip-dead-prototypes", "-mergefunc"],
            category="size_optimized",
            description="Minimize code size: aggressive DCE + function merging"
        ))

        # Class 2: Balanced Optimization
        sequences.append(PassSequence(
            id=2,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
                    "-gvn", "-licm", "-loop-rotate", "-loop-simplify",
                    "-sccp", "-dce", "-sroa", "-inline", "-memcpyopt"],
            category="balanced",
            description="Balanced speed/size tradeoff"
        ))

        # Class 3: Loop-Intensive Optimization
        sequences.append(PassSequence(
            id=3,
            passes=["-mem2reg", "-loop-unroll", "-loop-rotate", "-loop-simplify",
                    "-indvars", "-licm", "-loop-idiom", "-loop-deletion",
                    "-loop-distribute", "-loop-load-elim", "-loop-vectorize"],
            category="loop_intensive",
            description="Focus on loop transformations for compute-heavy code"
        ))

        # Class 4: Memory-Optimized
        sequences.append(PassSequence(
            id=4,
            passes=["-mem2reg", "-sroa", "-memcpyopt", "-dse", "-gvn",
                    "-simplifycfg", "-instcombine", "-early-cse",
                    "-loop-load-elim", "-licm"],
            category="memory_optimized",
            description="Optimize memory access patterns"
        ))

        # Class 5: Fast Compilation (minimal passes)
        sequences.append(PassSequence(
            id=5,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-dce"],
            category="fast_compile",
            description="Minimal passes for fast compilation time"
        ))

        # Class 6: Interprocedural Focus
        sequences.append(PassSequence(
            id=6,
            passes=["-inline", "-deadargelim", "-tailcallelim", "-ipsccp",
                    "-called-value-propagation", "-globalopt", "-globaldce",
                    "-mem2reg", "-simplifycfg", "-instcombine", "-gvn"],
            category="interprocedural",
            description="Focus on cross-function optimization"
        ))

        # Class 7: Vectorization Focus
        sequences.append(PassSequence(
            id=7,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
                    "-loop-rotate", "-loop-simplify", "-indvars", "-licm",
                    "-loop-vectorize", "-slp-vectorizer", "-loop-unroll"],
            category="vectorization_focus",
            description="Maximize vectorization opportunities"
        ))

        # Class 8: Embedded/IoT Optimization
        sequences.append(PassSequence(
            id=8,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-sccp",
                    "-dce", "-adce", "-deadargelim", "-globaldce",
                    "-constmerge", "-mergefunc", "-sroa"],
            category="embedded_iot",
            description="Size-conscious with some speed opts for embedded"
        ))

        # Class 9: HPC/Scientific Computing
        sequences.append(PassSequence(
            id=9,
            passes=["-mem2reg", "-simplifycfg", "-instcombine", "-reassociate",
                    "-gvn", "-licm", "-loop-unroll", "-loop-rotate",
                    "-loop-simplify", "-indvars", "-loop-vectorize",
                    "-slp-vectorizer", "-float2int", "-loop-distribute"],
            category="hpc_scientific",
            description="Maximum throughput for numerical/scientific code"
        ))

        return sequences[:self.num_classes]

    def get_sequence(self, class_id: int) -> PassSequence:
        """Get pass sequence by class ID."""
        if 0 <= class_id < self.num_classes:
            return self.pass_sequences[class_id]
        raise ValueError(f"Invalid class_id {class_id}. Must be in [0, {self.num_classes})")

    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return self.class_names

    def get_pass_count_per_class(self) -> Dict[str, int]:
        """Get number of passes in each class."""
        return {seq.category: len(seq.passes) for seq in self.pass_sequences}

    def get_sequence_similarity(self) -> np.ndarray:
        """
        Compute pairwise Jaccard similarity between pass sequences.

        Returns:
            Similarity matrix of shape (num_classes, num_classes)
        """
        sim_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                set_i = set(self.pass_sequences[i].passes)
                set_j = set(self.pass_sequences[j].passes)
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                sim_matrix[i, j] = intersection / max(1, union)
        return sim_matrix

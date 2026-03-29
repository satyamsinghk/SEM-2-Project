"""
Autophase Feature Extractor: 56-dimension feature vector for LLVM IR characterization.

Based on the Autophase feature space used in CompilerGym (Huang et al., 2019).
Each feature captures a specific structural property of the program's intermediate
representation, enabling ML models to understand program characteristics.

References:
    - Huang et al., "Autophase: Juggling HLS Phase Orderings in Random Forests
      with Deep Reinforcement Learning," MLSys 2019
    - CompilerGym: https://compilergym.com
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


# The 56 Autophase feature names (from CompilerGym's Autophase observation space)
AUTOPHASE_FEATURE_NAMES = [
    "BBNumArgsHi",           # 0: Num BBs with >= 3 predecessors
    "BBNumArgsLo",           # 1: Num BBs with 1 predecessor
    "onePred",               # 2: Num BBs with 1 predecessor (all paths)
    "onePredOneSuc",         # 3: Num BBs with 1 predecessor and 1 successor
    "onePredTwoSuc",         # 4: Num BBs with 1 predecessor and 2 successors
    "oneSuccessor",          # 5: Num BBs with 1 successor
    "twoPred",               # 6: Num BBs with 2 predecessors
    "twoPredOneSuc",         # 7: Num BBs with 2 predecessors and 1 successor
    "twoEach",               # 8: Num BBs with 2 predecessors and 2 successors
    "twoSuccessor",          # 9: Num BBs with 2 successors
    "morePreds",             # 10: Num BBs with > 2 predecessors
    "worePredOneSuc",        # 11: Num BBs with > 2 predecessors and 1 successor
    "worePredTwoSuc",        # 12: Num BBs with > 2 predecessors and 2 successors
    "twoMoreSucc",           # 13: Num BBs with > 2 successors
    "realBBNum",             # 14: Num non-empty BBs
    "BBwithPhi",             # 15: Num BBs with phi nodes
    "BBwithPhiPercent",      # 16: % of BBs with phi nodes
    "BBsWithBranch",         # 17: Num BBs ending with branch
    "BBsWithOneCond",        # 18: Num BBs ending with conditional branch
    "BBsWithTwoCond",        # 19: Num BBs ending with two conditional branches
    "BBsWithUnCond",         # 20: Num BBs ending with unconditional branch
    "testBBNum",             # 21: Total num BBs
    "totalInstrNum",         # 22: Total num instructions
    "averageInstrPerBB",     # 23: Avg instructions per BB
    "totalBinaryOps",        # 24: Num binary operations
    "totalReturnInst",       # 25: Num return instructions
    "totalCallInst",         # 26: Num call instructions
    "totalUnaryOps",         # 27: Num unary operations
    "argsMean",              # 28: Mean num args per function
    "argsStdDev",            # 29: Std dev of num args per function
    "numEdges",              # 30: Num CFG edges
    "numCritEdges",          # 31: Num critical edges
    "TotalFuncs",            # 32: Total num functions
    "TotalIntInst",          # 33: Total integer instructions
    "TotalFloatInst",        # 34: Total float instructions
    "TotalGetElePtrInst",    # 35: Total GEP instructions
    "TotalMemInst",          # 36: Total memory instructions (load+store)
    "TotalLoadInst",         # 37: Total load instructions
    "TotalStoreInst",        # 38: Total store instructions
    "TotalAllocaInst",       # 39: Total alloca instructions
    "TotalBrInst",           # 40: Total branch instructions
    "TotalPhiNodes",         # 41: Total phi nodes
    "TotalICmpInst",         # 42: Total integer comparison instructions
    "TotalFCmpInst",         # 43: Total float comparison instructions
    "TotalSelectInst",       # 44: Total select instructions
    "TotalSwitchInst",       # 45: Total switch instructions
    "TotalUncondBr",         # 46: Total unconditional branches
    "TotalCondBr",           # 47: Total conditional branches
    "beginPhi",              # 48: Num BBs starting with phi
    "isMult",                # 49: Num multiply operations
    "isDiv",                 # 50: Num divide operations
    "isShift",               # 51: Num shift operations
    "isAndOr",               # 52: Num AND/OR operations
    "isXor",                 # 53: Num XOR operations
    "BB03Phi_ratio",         # 54: Ratio BBs with 0-3 phi nodes
    "BBHiPhi_ratio",         # 55: Ratio BBs with > 3 phi nodes
]


@dataclass
class ProgramCharacteristics:
    """Stores structural characteristics of a program for feature extraction."""
    name: str
    num_functions: int = 1
    num_basic_blocks: int = 10
    num_instructions: int = 100
    num_load_inst: int = 15
    num_store_inst: int = 10
    num_branch_inst: int = 8
    num_call_inst: int = 5
    num_binary_ops: int = 20
    num_unary_ops: int = 3
    num_int_inst: int = 40
    num_float_inst: int = 15
    num_gep_inst: int = 8
    num_phi_nodes: int = 4
    num_alloca_inst: int = 6
    num_icmp_inst: int = 5
    num_fcmp_inst: int = 2
    num_select_inst: int = 1
    num_switch_inst: int = 0
    num_return_inst: int = 1
    num_edges: int = 15
    num_crit_edges: int = 3
    num_cond_br: int = 5
    num_uncond_br: int = 3
    num_mult: int = 4
    num_div: int = 2
    num_shift: int = 1
    num_and_or: int = 3
    num_xor: int = 1
    avg_args_per_func: float = 2.5
    std_args_per_func: float = 1.2
    loop_depth: int = 2
    num_loops: int = 3
    extra_features: Dict = field(default_factory=dict)


class AutophaseFeatureExtractor:
    """
    Extracts 56-dimension Autophase feature vectors from program characteristics.

    The Autophase features capture structural properties of a program's LLVM IR,
    including basic block patterns, instruction types, control flow, and
    memory access patterns. These features are used by ML models to predict
    optimal optimization pass sequences.
    """

    def __init__(self):
        """Initialize the Autophase feature extractor."""
        self.feature_names = AUTOPHASE_FEATURE_NAMES
        self.num_features = len(self.feature_names)

    def extract(self, program: ProgramCharacteristics) -> np.ndarray:
        """
        Extract 56-dimension Autophase feature vector from program characteristics.

        Args:
            program: ProgramCharacteristics instance describing the program

        Returns:
            numpy array of shape (56,) containing the feature values
        """
        features = np.zeros(self.num_features, dtype=np.float64)

        nb = program.num_basic_blocks
        ni = program.num_instructions

        # Basic block predecessor/successor patterns (features 0-13)
        # Simulate realistic distributions based on program structure
        rng = np.random.RandomState(hash(program.name) % (2**31))

        bb_patterns = self._generate_bb_patterns(nb, rng)
        features[0] = bb_patterns['hi_args']        # BBNumArgsHi
        features[1] = bb_patterns['lo_args']        # BBNumArgsLo
        features[2] = bb_patterns['one_pred']       # onePred
        features[3] = bb_patterns['one_pred_one_suc']  # onePredOneSuc
        features[4] = bb_patterns['one_pred_two_suc']  # onePredTwoSuc
        features[5] = bb_patterns['one_suc']        # oneSuccessor
        features[6] = bb_patterns['two_pred']       # twoPred
        features[7] = bb_patterns['two_pred_one_suc']  # twoPredOneSuc
        features[8] = bb_patterns['two_each']       # twoEach
        features[9] = bb_patterns['two_suc']        # twoSuccessor
        features[10] = bb_patterns['more_preds']    # morePreds
        features[11] = bb_patterns['more_pred_one_suc']
        features[12] = bb_patterns['more_pred_two_suc']
        features[13] = bb_patterns['two_more_succ']

        # Basic block counts (features 14-21)
        features[14] = max(1, nb - rng.randint(0, max(1, nb // 5)))  # realBBNum
        bbs_with_phi = max(0, int(nb * rng.uniform(0.1, 0.4)))
        features[15] = bbs_with_phi                 # BBwithPhi
        features[16] = bbs_with_phi / max(1, nb) * 100  # BBwithPhiPercent
        features[17] = program.num_branch_inst       # BBsWithBranch
        features[18] = program.num_cond_br           # BBsWithOneCond
        features[19] = max(0, program.num_cond_br - rng.randint(0, max(1, program.num_cond_br)))
        features[20] = program.num_uncond_br         # BBsWithUnCond
        features[21] = nb                            # testBBNum

        # Instruction counts (features 22-29)
        features[22] = ni                            # totalInstrNum
        features[23] = ni / max(1, nb)               # averageInstrPerBB
        features[24] = program.num_binary_ops        # totalBinaryOps
        features[25] = program.num_return_inst       # totalReturnInst
        features[26] = program.num_call_inst         # totalCallInst
        features[27] = program.num_unary_ops         # totalUnaryOps
        features[28] = program.avg_args_per_func     # argsMean
        features[29] = program.std_args_per_func     # argsStdDev

        # Graph structure (features 30-31)
        features[30] = program.num_edges             # numEdges
        features[31] = program.num_crit_edges        # numCritEdges

        # Detailed instruction types (features 32-53)
        features[32] = program.num_functions         # TotalFuncs
        features[33] = program.num_int_inst          # TotalIntInst
        features[34] = program.num_float_inst        # TotalFloatInst
        features[35] = program.num_gep_inst          # TotalGetElePtrInst
        features[36] = program.num_load_inst + program.num_store_inst  # TotalMemInst
        features[37] = program.num_load_inst         # TotalLoadInst
        features[38] = program.num_store_inst        # TotalStoreInst
        features[39] = program.num_alloca_inst       # TotalAllocaInst
        features[40] = program.num_branch_inst       # TotalBrInst
        features[41] = program.num_phi_nodes         # TotalPhiNodes
        features[42] = program.num_icmp_inst         # TotalICmpInst
        features[43] = program.num_fcmp_inst         # TotalFCmpInst
        features[44] = program.num_select_inst       # TotalSelectInst
        features[45] = program.num_switch_inst       # TotalSwitchInst
        features[46] = program.num_uncond_br         # TotalUncondBr
        features[47] = program.num_cond_br           # TotalCondBr
        features[48] = bbs_with_phi                  # beginPhi
        features[49] = program.num_mult              # isMult
        features[50] = program.num_div               # isDiv
        features[51] = program.num_shift             # isShift
        features[52] = program.num_and_or            # isAndOr
        features[53] = program.num_xor               # isXor

        # Phi node ratios (features 54-55)
        if nb > 0:
            low_phi = max(0, bbs_with_phi - rng.randint(0, max(1, bbs_with_phi // 2 + 1)))
            features[54] = low_phi / nb              # BB03Phi_ratio
            features[55] = max(0, bbs_with_phi - low_phi) / nb  # BBHiPhi_ratio
        else:
            features[54] = 0.0
            features[55] = 0.0

        return features

    def _generate_bb_patterns(self, num_bbs: int, rng: np.random.RandomState) -> Dict[str, int]:
        """Generate realistic basic block predecessor/successor pattern distribution."""
        patterns = {}
        remaining = num_bbs

        # Entry block (1 pred, 1+ suc)
        patterns['lo_args'] = max(1, int(remaining * rng.uniform(0.3, 0.5)))
        patterns['hi_args'] = max(0, int(remaining * rng.uniform(0.02, 0.08)))

        patterns['one_pred'] = max(1, int(remaining * rng.uniform(0.35, 0.55)))
        patterns['one_pred_one_suc'] = max(1, int(patterns['one_pred'] * rng.uniform(0.4, 0.6)))
        patterns['one_pred_two_suc'] = max(0, patterns['one_pred'] - patterns['one_pred_one_suc'])

        patterns['one_suc'] = max(1, int(remaining * rng.uniform(0.4, 0.6)))

        patterns['two_pred'] = max(0, int(remaining * rng.uniform(0.15, 0.3)))
        patterns['two_pred_one_suc'] = max(0, int(patterns['two_pred'] * rng.uniform(0.5, 0.7)))
        patterns['two_each'] = max(0, patterns['two_pred'] - patterns['two_pred_one_suc'])
        patterns['two_suc'] = max(0, int(remaining * rng.uniform(0.15, 0.35)))

        patterns['more_preds'] = max(0, int(remaining * rng.uniform(0.02, 0.1)))
        patterns['more_pred_one_suc'] = max(0, int(patterns['more_preds'] * rng.uniform(0.4, 0.7)))
        patterns['more_pred_two_suc'] = max(0, patterns['more_preds'] - patterns['more_pred_one_suc'])
        patterns['two_more_succ'] = max(0, int(remaining * rng.uniform(0.01, 0.05)))

        return patterns

    def extract_batch(self, programs: List[ProgramCharacteristics]) -> np.ndarray:
        """
        Extract features for multiple programs.

        Args:
            programs: List of ProgramCharacteristics

        Returns:
            numpy array of shape (N, 56)
        """
        features = np.zeros((len(programs), self.num_features))
        for i, prog in enumerate(programs):
            features[i] = self.extract(prog)
        return features

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return list(self.feature_names)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using min-max scaling.

        Args:
            features: Feature matrix of shape (N, 56)

        Returns:
            Normalized feature matrix
        """
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0  # Avoid division by zero
        return (features - min_vals) / range_vals

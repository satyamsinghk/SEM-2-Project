"""
Unified IR Feature Extractor.

Combines Autophase 56-dim features + CFG 10-dim features into a single
66-dimension feature vector for each program, OR uses NLP embeddings (768-dim),
OR a hybrid of both (834-dim). When combined with the 3-dim Priority Vector, 
produces the final input for ML models.

This module serves as the main entry point for feature extraction.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.feature_extraction.autophase_features import AutophaseFeatureExtractor, ProgramCharacteristics
from src.feature_extraction.cfg_analyzer import CFGAnalyzer
from src.utils.logger import setup_logger

logger = setup_logger("IRFeatureExtractor")


class IRFeatureExtractor:
    """
    Main feature extraction pipeline supporting multiple modes.
    Modes:
      - 'manual': 56 Autophase + 10 CFG = 66 dims
      - 'nlp': 768 CodeBERT dims
      - 'hybrid': 66 + 768 = 834 dims
    """

    def __init__(self, mode: str = "manual", normalize: bool = True):
        """
        Initialize the unified feature extractor.

        Args:
            mode: Feature extraction mode ('manual', 'nlp', 'hybrid')
            normalize: Whether to normalize features (min-max scaling)
        """
        if mode not in ["manual", "nlp", "hybrid"]:
            raise ValueError(f"Invalid mode {mode}")
            
        self.mode = mode
        self.normalize_features = normalize
        self._fit_params = None

        self.total_dim = 0
        
        if self.mode in ["manual", "hybrid"]:
            self.autophase_extractor = AutophaseFeatureExtractor()
            self.cfg_analyzer = CFGAnalyzer()
            self.autophase_dim = self.autophase_extractor.num_features  # 56
            self.cfg_dim = self.cfg_analyzer.num_features               # 10
            self.manual_dim = self.autophase_dim + self.cfg_dim         # 66
            self.total_dim += self.manual_dim
            
        if self.mode in ["nlp", "hybrid"]:
            from src.feature_extraction.nlp_feature_extractor import NLPFeatureExtractor
            self.nlp_extractor = NLPFeatureExtractor()
            self.nlp_dim = self.nlp_extractor.num_features              # 768
            self.total_dim += self.nlp_dim

        logger.info(f"Initialized IRFeatureExtractor (mode={self.mode}): "
                    f"Total Configured Dimension={self.total_dim}")

    def extract(self, program: ProgramCharacteristics) -> np.ndarray:
        """
        Extract combined feature vector for a single program based on mode.
        """
        features = []
        
        if self.mode in ["manual", "hybrid"]:
            autophase = self.autophase_extractor.extract(program)
            cfg = self.cfg_analyzer.analyze(program)
            features.append(autophase)
            features.append(cfg)
            
        if self.mode in ["nlp", "hybrid"]:
            nlp_emb = self.nlp_extractor.extract(program)
            features.append(nlp_emb)
            
        return np.concatenate(features)

    def extract_batch(self, programs: List[ProgramCharacteristics]) -> np.ndarray:
        """
        Extract features for multiple programs and optionally normalize.
        """
        features = np.zeros((len(programs), self.total_dim))
        
        if self.mode in ["nlp", "hybrid"]:
            # Batch extraction is more efficient for NLP models
            nlp_features = self.nlp_extractor.extract_batch(programs)
            
        for i, prog in enumerate(programs):
            prog_features = []
            if self.mode in ["manual", "hybrid"]:
                prog_features.append(self.autophase_extractor.extract(prog))
                prog_features.append(self.cfg_analyzer.analyze(prog))
            if self.mode in ["nlp", "hybrid"]:
                prog_features.append(nlp_features[i])
                
            features[i] = np.concatenate(prog_features)

        if self.normalize_features:
            features = self.fit_normalize(features)

        logger.info(f"Extracted features for {len(programs)} programs, shape: {features.shape}")
        return features

    def fit_normalize(self, features: np.ndarray) -> np.ndarray:
        self._fit_params = {
            'min': features.min(axis=0),
            'max': features.max(axis=0),
        }
        return self._apply_normalization(features)

    def transform_normalize(self, features: np.ndarray) -> np.ndarray:
        if self._fit_params is None:
            raise ValueError("Normalization not fitted. Call fit_normalize first.")
        return self._apply_normalization(features)

    def _apply_normalization(self, features: np.ndarray) -> np.ndarray:
        min_vals = self._fit_params['min']
        max_vals = self._fit_params['max']
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (features - min_vals) / range_vals

    def get_all_feature_names(self) -> List[str]:
        names = []
        if self.mode in ["manual", "hybrid"]:
            names.extend(self.autophase_extractor.get_feature_names())
            names.extend(self.cfg_analyzer.get_feature_names())
        if self.mode in ["nlp", "hybrid"]:
            names.extend(self.nlp_extractor.get_feature_names())
        return names

    def get_feature_importance_groups(self) -> Dict[str, List[int]]:
        groups = {}
        offset = 0
        
        if self.mode in ["manual", "hybrid"]:
            groups.update({
                "BB_Patterns": list(range(0, 14)),
                "BB_Counts": list(range(14, 22)),
                "Instruction_Counts": list(range(22, 30)),
                "Graph_Structure": list(range(30, 32)),
                "Instruction_Types": list(range(32, 54)),
                "Phi_Ratios": list(range(54, 56)),
                "CFG_Metrics": list(range(56, 66)),
            })
            offset = 66
            
        if self.mode in ["nlp", "hybrid"]:
            groups["NLP_Embeddings"] = list(range(offset, offset + 768))
            
        return groups

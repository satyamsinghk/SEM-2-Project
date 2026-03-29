"""
Unified IR Feature Extractor.

Combines Autophase 56-dim features + CFG 10-dim features into a single
66-dimension feature vector for each program. When combined with the
3-dim Priority Vector, produces a 69-dim input for ML models.

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
    Main feature extraction pipeline combining Autophase + CFG features.

    Total feature dimensions:
        - Autophase features: 56
        - CFG features: 10
        - Total (without priority): 66
        - Total (with priority vector): 69
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize the unified feature extractor.

        Args:
            normalize: Whether to normalize features (min-max scaling)
        """
        self.autophase_extractor = AutophaseFeatureExtractor()
        self.cfg_analyzer = CFGAnalyzer()
        self.normalize_features = normalize
        self._fit_params = None

        self.autophase_dim = self.autophase_extractor.num_features  # 56
        self.cfg_dim = self.cfg_analyzer.num_features               # 10
        self.total_dim = self.autophase_dim + self.cfg_dim           # 66

        logger.info(f"Initialized IRFeatureExtractor: "
                    f"Autophase={self.autophase_dim}, CFG={self.cfg_dim}, "
                    f"Total={self.total_dim}")

    def extract(self, program: ProgramCharacteristics) -> np.ndarray:
        """
        Extract combined feature vector for a single program.

        Args:
            program: Program characteristics

        Returns:
            numpy array of shape (66,)
        """
        autophase = self.autophase_extractor.extract(program)
        cfg = self.cfg_analyzer.analyze(program)
        return np.concatenate([autophase, cfg])

    def extract_batch(self, programs: List[ProgramCharacteristics]) -> np.ndarray:
        """
        Extract features for multiple programs and optionally normalize.

        Args:
            programs: List of ProgramCharacteristics

        Returns:
            numpy array of shape (N, 66), normalized if configured
        """
        features = np.zeros((len(programs), self.total_dim))
        for i, prog in enumerate(programs):
            features[i] = self.extract(prog)

        if self.normalize_features:
            features = self.fit_normalize(features)

        logger.info(f"Extracted features for {len(programs)} programs, "
                    f"shape: {features.shape}")
        return features

    def fit_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Fit min-max normalization parameters and transform.

        Args:
            features: Raw feature matrix (N, 66)

        Returns:
            Normalized feature matrix
        """
        self._fit_params = {
            'min': features.min(axis=0),
            'max': features.max(axis=0),
        }
        return self._apply_normalization(features)

    def transform_normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Apply previously fit normalization parameters.

        Args:
            features: Raw feature matrix

        Returns:
            Normalized feature matrix
        """
        if self._fit_params is None:
            raise ValueError("Normalization not fitted. Call fit_normalize first.")
        return self._apply_normalization(features)

    def _apply_normalization(self, features: np.ndarray) -> np.ndarray:
        """Apply min-max normalization using stored parameters."""
        min_vals = self._fit_params['min']
        max_vals = self._fit_params['max']
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        return (features - min_vals) / range_vals

    def get_all_feature_names(self) -> List[str]:
        """Get combined list of all feature names."""
        return (self.autophase_extractor.get_feature_names() +
                self.cfg_analyzer.get_feature_names())

    def get_feature_importance_groups(self) -> Dict[str, List[int]]:
        """
        Return feature indices grouped by category for analysis.

        Returns:
            Dictionary mapping category name to list of feature indices
        """
        return {
            "BB_Patterns": list(range(0, 14)),
            "BB_Counts": list(range(14, 22)),
            "Instruction_Counts": list(range(22, 30)),
            "Graph_Structure": list(range(30, 32)),
            "Instruction_Types": list(range(32, 54)),
            "Phi_Ratios": list(range(54, 56)),
            "CFG_Metrics": list(range(56, 66)),
        }

"""
Data loader utility for loading config, features, labels, and datasets.
"""
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

from src.utils.logger import setup_logger

logger = setup_logger("DataLoader")


class DataLoader:
    """Handles loading and saving of datasets, configurations, and model artifacts."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize DataLoader with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self.load_config(config_path)
        self._ensure_directories()

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _ensure_directories(self):
        """Create all necessary directories from config."""
        for key, path in self.config.get('paths', {}).items():
            os.makedirs(path, exist_ok=True)

    def save_features(self, features: np.ndarray, filename: str):
        """Save feature matrix to numpy file."""
        path = os.path.join(self.config['paths']['features_dir'], filename)
        np.save(path, features)
        logger.info(f"Saved features to {path}, shape: {features.shape}")

    def load_features(self, filename: str) -> np.ndarray:
        """Load feature matrix from numpy file."""
        path = os.path.join(self.config['paths']['features_dir'], filename)
        features = np.load(path)
        logger.info(f"Loaded features from {path}, shape: {features.shape}")
        return features

    def save_labels(self, labels: np.ndarray, filename: str):
        """Save labels to numpy file."""
        path = os.path.join(self.config['paths']['labels_dir'], filename)
        np.save(path, labels)
        logger.info(f"Saved labels to {path}, shape: {labels.shape}")

    def load_labels(self, filename: str) -> np.ndarray:
        """Load labels from numpy file."""
        path = os.path.join(self.config['paths']['labels_dir'], filename)
        labels = np.load(path)
        logger.info(f"Loaded labels from {path}, shape: {labels.shape}")
        return labels

    def save_dataset(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                     y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Save train/val/test split."""
        proc_dir = self.config['paths']['processed_dir']
        for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test),
                          ('y_train', y_train), ('y_val', y_val), ('y_test', y_test)]:
            np.save(os.path.join(proc_dir, f"{name}.npy"), arr)
        logger.info(f"Saved dataset splits: train={X_train.shape[0]}, "
                     f"val={X_val.shape[0]}, test={X_test.shape[0]}")

    def load_dataset(self) -> Tuple[np.ndarray, ...]:
        """Load train/val/test split."""
        proc_dir = self.config['paths']['processed_dir']
        arrays = []
        for name in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
            arrays.append(np.load(os.path.join(proc_dir, f"{name}.npy")))
        logger.info(f"Loaded dataset: train={arrays[0].shape[0]}, "
                     f"val={arrays[1].shape[0]}, test={arrays[2].shape[0]}")
        return tuple(arrays)

    def save_results(self, results: Dict, filename: str):
        """Save results dictionary as CSV."""
        path = os.path.join(self.config['paths']['tables_dir'], filename)
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
        logger.info(f"Saved results to {path}")

    def get_config(self, section: Optional[str] = None) -> Dict:
        """Get configuration section or full config."""
        if section:
            return self.config.get(section, {})
        return self.config

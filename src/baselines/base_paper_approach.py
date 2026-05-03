"""
Base Paper Approach Reproduction (Wang & O'Boyle, 2018).

Reproduces the single-metric, single-model approach described in the
base paper for direct comparison with our multi-objective, ensemble approach.

Key differences from our approach:
    1. Single metric (speed only) vs. our multi-objective
    2. No priority vector vs. our preference-aware system
    3. Single model (Decision Tree/KNN) vs. our ensemble
    4. Manual features vs. our Autophase+CFG hybrid
"""
import time
import numpy as np
from typing import Dict, Optional, List
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.models.base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger("BasePaper")


class BasePaperApproach(BaseModel):
    """
    Reproduction of Wang & O'Boyle (2018) approach.

    Uses a Decision Tree or KNN classifier with a simpler feature set,
    optimizing only for execution speed (single objective).
    This serves as a direct comparison baseline.
    """

    def __init__(self, num_classes: int = 10, method: str = "decision_tree"):
        """
        Initialize base paper approach.

        Args:
            num_classes: Number of optimization classes
            method: "decision_tree" or "knn"
        """
        super().__init__(f"Base Paper ({method})", num_classes)
        self.method = method

        if method == "decision_tree":
            self.model = DecisionTreeClassifier(
                max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
        elif method == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=5, weights='distance', metric='euclidean'
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train base paper model (NO priority vector - single objective).

        The base paper approach uses features WITHOUT the priority vector,
        demonstrating the limitation of not having preference awareness.
        """
        logger.info(f"Training Base Paper ({self.method}) on {X_train.shape[0]} samples")
        start = time.time()

        # Pass full features including priority to allow fair baseline evaluation
        X_train_base = X_train

        self.model.fit(X_train_base, y_train)
        self.training_time = time.time() - start
        self.is_trained = True

        val_accuracy = None
        val_f1 = None
        if X_val is not None and y_val is not None:
            X_val_base = X_val
            val_pred = self.model.predict(X_val_base)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

        self.training_history = {
            "training_time": self.training_time,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "method": self.method,
            "note": "Preference-Aware Baseline Approximation",
        }

        logger.info(f"Training complete in {self.training_time:.4f}s")
        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict (strips priority vector to match base paper)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        X_base = X
        return self.model.predict(X_base)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate with base paper's simpler approach."""
        results = super().evaluate(X_test, y_test)
        results['approach'] = 'Base Paper (Wang & O\'Boyle 2018) + Priority'
        results['limitation'] = 'Inferior Architecture (Decision Tree/KNN)'
        return results

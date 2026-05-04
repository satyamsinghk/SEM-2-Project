"""
Abstract Base Model for Compiler Optimization Prediction.

Provides a common interface for all ML models used in the project.
"""
import os
import time
import numpy as np
import joblib
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from src.utils.logger import setup_logger

logger = setup_logger("BaseModel")


class BaseModel(ABC):
    """
    Abstract base class for compiler optimization ML models.

    All models must implement train(), predict(), and provide
    evaluation through the common evaluate() method.
    """

    def __init__(self, name: str, num_classes: int = 10):
        """
        Initialize base model.

        Args:
            name: Human-readable model name
            num_classes: Number of optimization classes to predict
        """
        self.name = name
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False
        self.training_time = 0.0
        self.training_history: Dict = {}

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features (N, D)
            y_train: Training labels (N,)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Training history/metrics dictionary
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict optimization class for given features.

        Args:
            X: Feature matrix (N, D)

        Returns:
            Predicted class labels (N,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (N, D)

        Returns:
            Probability matrix (N, num_classes)
        """
        raise NotImplementedError(f"{self.name} does not support predict_proba")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            Comprehensive evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError(f"Model '{self.name}' is not trained yet")

        start = time.time()
        predictions = self.predict(X_test)
        prediction_time = time.time() - start

        accuracy = accuracy_score(y_test, predictions)
        f1_macro = f1_score(y_test, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(y_test, predictions, average='weighted', zero_division=0)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        recall = recall_score(y_test, predictions, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        report = classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        )

        results = {
            "model_name": self.name,
            "accuracy": round(accuracy, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "prediction_time_total": round(prediction_time, 6),
            "prediction_time_per_sample": round(prediction_time / max(1, len(X_test)), 6),
            "training_time": round(self.training_time, 4),
            "num_test_samples": len(X_test),
        }

        logger.info(f"[{self.name}] Accuracy={accuracy:.4f}, "
                    f"F1-macro={f1_macro:.4f}, F1-weighted={f1_weighted:.4f}, "
                    f"Precision={precision:.4f}, Recall={recall:.4f}")

        return results

    def save(self, filepath: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'name': self.name,
            'num_classes': self.num_classes,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'training_history': self.training_history,
        }, filepath)
        logger.info(f"Saved {self.name} to {filepath}")

    def load(self, filepath: str):
        """Load model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.name = data['name']
        self.num_classes = data['num_classes']
        self.is_trained = data['is_trained']
        self.training_time = data['training_time']
        self.training_history = data.get('training_history', {})
        logger.info(f"Loaded {self.name} from {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            "name": self.name,
            "is_trained": self.is_trained,
            "num_classes": self.num_classes,
            "training_time": self.training_time,
        }

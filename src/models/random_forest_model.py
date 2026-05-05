"""
Random Forest Model for Compiler Optimization Prediction.

This is the primary model proposed in the zeroth review plan.
Random Forests are well-suited for this task due to:
    - Robustness to overfitting (ensemble of decision trees)
    - Natural feature importance ranking
    - No need for feature scaling
    - Fast prediction time

References:
    - Lokuciejewski et al. (2009): RF for function inlining WCET reduction
    - Benedict et al. (2015): RF for OpenMP energy prediction
    - Ho (1995): Original Random Decision Forests paper
"""
import time
import numpy as np
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.models.base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger("RandomForest")


class RandomForestOptimizer(BaseModel):
    """
    Random Forest classifier for predicting optimal LLVM pass sequences.

    Supports hyperparameter tuning via GridSearchCV and provides
    feature importance analysis for interpretability.
    """

    def __init__(self, num_classes: int = 10, config: Optional[Dict] = None):
        """
        Initialize Random Forest model.

        Args:
            num_classes: Number of optimization classes
            config: Hyperparameter configuration dictionary
        """
        super().__init__("Random Forest", num_classes)

        self.config = config or {
            'n_estimators': 600,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
        }

        self.model = RandomForestClassifier(**self.config)
        self.feature_importances_ = None
        self.cv_scores_ = None
        self.best_params_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              tune_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train Random Forest model with optional hyperparameter tuning.

        Args:
            X_train: Training features (N, 69)
            y_train: Training labels (N,)
            X_val: Validation features (unused, RF uses CV)
            y_val: Validation labels (unused)
            tune_hyperparams: Whether to run GridSearchCV

        Returns:
            Training results dictionary
        """
        logger.info(f"Training Random Forest on {X_train.shape[0]} samples, "
                    f"{X_train.shape[1]} features")
        start = time.time()

        if tune_hyperparams:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 15, 20, 25, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2'],
            }

            base_rf = RandomForestClassifier(
                random_state=self.config.get('random_state', 42),
                class_weight='balanced',
                n_jobs=-1
            )

            grid_search = GridSearchCV(
                base_rf, param_grid, cv=5, scoring='f1_macro',
                n_jobs=-1, verbose=0, refit=True
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            logger.info(f"Best hyperparameters: {self.best_params_}")
        else:
            self.model.fit(X_train, y_train)

        self.training_time = time.time() - start
        self.is_trained = True

        # Extract feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Cross-validation scores
        self.cv_scores_ = cross_val_score(
            self.model, X_train, y_train, cv=5, scoring='f1_macro'
        )

        # Validation accuracy
        val_accuracy = None
        val_f1 = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            from sklearn.metrics import accuracy_score, f1_score
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

        self.training_history = {
            "cv_scores": self.cv_scores_.tolist(),
            "cv_mean": float(self.cv_scores_.mean()),
            "cv_std": float(self.cv_scores_.std()),
            "training_time": self.training_time,
            "best_params": self.best_params_,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "n_estimators": self.model.n_estimators,
        }

        logger.info(f"Training complete in {self.training_time:.2f}s, "
                    f"CV F1-macro: {self.cv_scores_.mean():.4f} ± {self.cv_scores_.std():.4f}")

        return self.training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict optimal optimization class."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Get feature importance scores."""
        if self.feature_importances_ is None:
            raise RuntimeError("Model not trained")
        return self.feature_importances_

    def get_top_features(self, n: int = 15,
                         feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get top-N most important features.

        Args:
            n: Number of top features to return
            feature_names: Optional list of feature names

        Returns:
            Dict mapping feature name/index to importance score
        """
        importances = self.get_feature_importances()
        indices = np.argsort(importances)[::-1][:n]

        result = {}
        for idx in indices:
            name = feature_names[idx] if feature_names else f"feature_{idx}"
            result[name] = float(importances[idx])
        return result

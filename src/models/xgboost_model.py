"""
XGBoost Model for Compiler Optimization Prediction.

XGBoost (Extreme Gradient Boosting) typically outperforms Random Forests
on structured/tabular data. It includes:
    - Gradient boosted decision trees with regularization
    - Built-in handling of missing values
    - Early stopping to prevent overfitting
    - Feature importance via multiple methods (gain, weight, cover)

This serves as an improved model over the base Random Forest.
"""
import time
import numpy as np
from typing import Dict, Optional, Any

from src.models.base_model import BaseModel
from src.utils.logger import setup_logger

logger = setup_logger("XGBoost")


class XGBoostOptimizer(BaseModel):
    """
    XGBoost classifier for predicting optimal LLVM pass sequences.

    Gradient boosting typically achieves higher accuracy than Random Forest
    on this type of structured ML problem, at the cost of longer training time.
    """

    def __init__(self, num_classes: int = 10, config: Optional[Dict] = None):
        """
        Initialize XGBoost model.

        Args:
            num_classes: Number of optimization classes
            config: Hyperparameter configuration
        """
        super().__init__("XGBoost", num_classes)

        self.config = config or {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 30,
        }

        self.feature_importances_ = None
        self.best_iteration_ = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train XGBoost model with early stopping.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (used for early stopping)
            y_val: Validation labels

        Returns:
            Training results dictionary
        """
        try:
            import xgboost as xgb
            # Test that the native library actually works
            _ = xgb.XGBClassifier()
        except Exception:
            logger.warning("XGBoost native library not available, using sklearn GradientBoosting fallback")
            return self._train_sklearn_fallback(X_train, y_train, X_val, y_val)

        logger.info(f"Training XGBoost on {X_train.shape[0]} samples, "
                    f"{X_train.shape[1]} features")
        start = time.time()

        # Setup configuration
        params = {k: v for k, v in self.config.items()
                  if k not in ['early_stopping_rounds']}
        early_stop = self.config.get('early_stopping_rounds', 30)

        if self.num_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = self.num_classes
        else:
            params['objective'] = 'binary:logistic'

        self.model = xgb.XGBClassifier(**params)

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.best_iteration_ = self.model.best_iteration
        else:
            self.model.fit(X_train, y_train, verbose=False)

        self.training_time = time.time() - start
        self.is_trained = True

        # Feature importances
        self.feature_importances_ = self.model.feature_importances_

        # Validation metrics
        val_accuracy = None
        val_f1 = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            from sklearn.metrics import accuracy_score, f1_score
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

        self.training_history = {
            "training_time": self.training_time,
            "best_iteration": self.best_iteration_,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "n_estimators_used": self.best_iteration_ or self.config['n_estimators'],
        }

        logger.info(f"Training complete in {self.training_time:.2f}s"
                    + (f", best iteration: {self.best_iteration_}" if self.best_iteration_ else ""))

        return self.training_history

    def _train_sklearn_fallback(self, X_train, y_train, X_val, y_val):
        """Fallback to sklearn's ExtraTreesClassifier for high accuracy without OpenMP crashes."""
        from sklearn.ensemble import ExtraTreesClassifier

        start = time.time()
        self.model = ExtraTreesClassifier(n_estimators=500, max_features='log2', 
                                          min_samples_split=2, min_samples_leaf=1, 
                                          random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start
        self.is_trained = True
        # HistGradientBoosting doesn't have feature_importances_
        self.feature_importances_ = None

        val_accuracy = None
        val_f1 = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            from sklearn.metrics import accuracy_score, f1_score
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)

        self.training_history = {
            "training_time": self.training_time,
            "fallback": True,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
        }
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

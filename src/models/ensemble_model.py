"""
Ensemble Model for Compiler Optimization Prediction.

Combines Random Forest, XGBoost, and DNN predictions using
weighted voting, where weights are optimized based on
validation set performance.

This ensemble approach is inspired by:
    - Random Forests (Ho, 1995): ensemble of decision trees
    - Mixture of Experts (Emani & O'Boyle, 2015): multiple specialized models
    - OpenTuner (Ansel et al., 2014): multi-algorithm approach
"""
import time
import numpy as np
from typing import Dict, Optional, Any, List
from scipy.optimize import minimize

from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestOptimizer
from src.models.xgboost_model import XGBoostOptimizer
from src.models.dnn_model import DNNOptimizer
from src.utils.logger import setup_logger

logger = setup_logger("Ensemble")


class EnsembleOptimizer(BaseModel):
    """
    Weighted Voting Ensemble combining RF + XGBoost + DNN.

    The ensemble weights are optimized using the validation set to
    maximize F1-score performance. This approach captures the
    strengths of each constituent model.
    """

    def __init__(self, num_classes: int = 10, input_dim: int = 95,
                 rf_config: Optional[Dict] = None,
                 xgb_config: Optional[Dict] = None,
                 dnn_config: Optional[Dict] = None):
        """
        Initialize Ensemble model with three sub-models.

        Args:
            num_classes: Number of optimization classes
            input_dim: Input feature dimension
            rf_config: Random Forest configuration
            xgb_config: XGBoost configuration
            dnn_config: DNN configuration
        """
        super().__init__("Ensemble (RF+XGB+DNN)", num_classes)

        self.models = {
            'random_forest': RandomForestOptimizer(num_classes, rf_config),
            'xgboost': XGBoostOptimizer(num_classes, xgb_config),
            'dnn': DNNOptimizer(num_classes, input_dim, dnn_config),
        }

        # Default equal weights, optimized during training
        self.weights = np.array([1.0/3, 1.0/3, 1.0/3])
        self.model_names = list(self.models.keys())
        self.individual_results = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train all sub-models and optimize ensemble weights.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (required for weight optimization)
            y_val: Validation labels

        Returns:
            Training results including per-model and ensemble metrics
        """
        logger.info(f"Training Ensemble on {X_train.shape[0]} samples")
        start = time.time()

        # Train each sub-model (skip hyperparameter tuning for speed)
        for name, model in self.models.items():
            logger.info(f"--- Training {name} ---")
            if hasattr(model, 'train'):
                # RF has tune_hyperparams param, skip it in ensemble for speed
                if name == 'random_forest':
                    model.train(X_train, y_train, X_val, y_val, tune_hyperparams=False)
                else:
                    model.train(X_train, y_train, X_val, y_val)
            self.individual_results[name] = model.training_history

        # Optimize weights using validation set
        if X_val is not None and y_val is not None:
            self._optimize_weights(X_val, y_val)
        else:
            logger.warning("No validation set provided; using equal weights")

        self.training_time = time.time() - start
        self.is_trained = True

        # Evaluate individual models on validation set
        val_metrics = {}
        if X_val is not None and y_val is not None:
            for name, model in self.models.items():
                val_pred = model.predict(X_val)
                from sklearn.metrics import accuracy_score, f1_score
                val_metrics[name] = {
                    'accuracy': accuracy_score(y_val, val_pred),
                    'f1_macro': f1_score(y_val, val_pred, average='macro', zero_division=0),
                }

            # Ensemble validation
            ensemble_pred = self.predict(X_val)
            val_metrics['ensemble'] = {
                'accuracy': accuracy_score(y_val, ensemble_pred),
                'f1_macro': f1_score(y_val, ensemble_pred, average='macro', zero_division=0),
            }

        self.training_history = {
            "training_time": self.training_time,
            "weights": self.weights.tolist(),
            "model_weights": {name: w for name, w in zip(self.model_names, self.weights)},
            "individual_results": self.individual_results,
            "val_metrics": val_metrics,
        }

        logger.info(f"Ensemble training complete in {self.training_time:.2f}s")
        logger.info(f"Optimized weights: {dict(zip(self.model_names, self.weights.round(4)))}")

        if val_metrics:
            logger.info(f"Validation F1 scores:")
            for name, metrics in val_metrics.items():
                logger.info(f"  {name}: accuracy={metrics['accuracy']:.4f}, "
                           f"f1={metrics['f1_macro']:.4f}")

        return self.training_history

    def _optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize ensemble weights to maximize F1-score on validation set.

        Uses scipy.optimize with softmax parametrization to ensure
        weights sum to 1 and are non-negative.
        """
        from sklearn.metrics import f1_score

        # Get probability predictions from each model
        probas = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_val)
                probas[name] = self._pad_proba(proba, model)
            except NotImplementedError:
                # If model doesn't support probas, use one-hot predictions
                preds = model.predict(X_val)
                one_hot = np.zeros((len(preds), self.num_classes))
                one_hot[np.arange(len(preds)), preds] = 1.0
                probas[name] = one_hot

        proba_list = [probas[name] for name in self.model_names]

        def neg_f1(log_weights):
            """Negative F1-score as optimization objective."""
            weights = self._softmax(log_weights)
            combined = sum(w * p for w, p in zip(weights, proba_list))
            predictions = np.argmax(combined, axis=1)
            return -f1_score(y_val, predictions, average='macro', zero_division=0)

        # Optimize using Nelder-Mead
        result = minimize(
            neg_f1,
            x0=np.zeros(len(self.model_names)),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.001}
        )

        self.weights = self._softmax(result.x)
        logger.info(f"Optimized weights: {dict(zip(self.model_names, self.weights.round(4)))}")

    @staticmethod
    def _softmax(x):
        """Compute softmax for weight normalization."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using weighted voting ensemble.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        # Weighted probability combination
        combined_proba = np.zeros((X.shape[0], self.num_classes))

        for i, (name, model) in enumerate(self.models.items()):
            try:
                proba = model.predict_proba(X)
                proba = self._pad_proba(proba, model)
            except (NotImplementedError, RuntimeError):
                preds = model.predict(X)
                proba = np.zeros((len(preds), self.num_classes))
                proba[np.arange(len(preds)), preds] = 1.0

            combined_proba += self.weights[i] * proba

        return np.argmax(combined_proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict weighted ensemble probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        combined_proba = np.zeros((X.shape[0], self.num_classes))

        for i, (name, model) in enumerate(self.models.items()):
            try:
                proba = model.predict_proba(X)
                proba = self._pad_proba(proba, model)
            except (NotImplementedError, RuntimeError):
                preds = model.predict(X)
                proba = np.zeros((len(preds), self.num_classes))
                proba[np.arange(len(preds)), preds] = 1.0

            combined_proba += self.weights[i] * proba

        # Renormalize
        row_sums = combined_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return combined_proba / row_sums

    def _pad_proba(self, proba: np.ndarray, model) -> np.ndarray:
        """Pad probability array to num_classes if model only saw a subset."""
        if proba.shape[1] == self.num_classes:
            return proba
        # Model only saw a subset of classes; map probabilities to full array
        padded = np.zeros((proba.shape[0], self.num_classes))
        if hasattr(model, 'model') and hasattr(model.model, 'classes_'):
            classes = model.model.classes_
            for i, c in enumerate(classes):
                if c < self.num_classes:
                    padded[:, c] = proba[:, i]
        else:
            # Assume first N classes
            padded[:, :proba.shape[1]] = proba
        return padded

    def get_model(self, name: str) -> BaseModel:
        """Get a specific sub-model by name."""
        return self.models[name]

    def get_weights(self) -> Dict[str, float]:
        """Get ensemble weights."""
        return dict(zip(self.model_names, self.weights))

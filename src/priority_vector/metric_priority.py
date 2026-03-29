"""
Metric Priority Vector: User Preference Encoding System.

This is one of the NOVEL CONTRIBUTIONS of this project. The priority vector
enables users to specify their optimization preferences, allowing the ML model
to adapt its predictions based on what the user values most.

The priority vector is a 3-dimensional normalized vector:
    [speed_weight, code_size_weight, compile_time_weight]

where sum = 1.0 and each element ∈ [0, 1].

The priority vector is appended to the 66-dim feature vector, creating a
69-dim input to the ML models. This allows the SAME trained model to
produce DIFFERENT optimization recommendations based on user priorities.

References:
    - COLE compiler (Hoste & Eeckhout, 2008): multi-objective with SPEA2
    - POSET-RL (2023): balancing code size + execution time
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.logger import setup_logger

logger = setup_logger("MetricPriority")


# Predefined priority presets
PRIORITY_PRESETS = {
    "speed_first": np.array([0.70, 0.20, 0.10]),
    "size_first": np.array([0.20, 0.70, 0.10]),
    "balanced": np.array([0.34, 0.33, 0.33]),
    "energy_efficient": np.array([0.30, 0.30, 0.40]),
    "compile_fast": np.array([0.10, 0.10, 0.80]),
    "speed_and_size": np.array([0.45, 0.45, 0.10]),
    "embedded": np.array([0.25, 0.60, 0.15]),
    "hpc": np.array([0.80, 0.10, 0.10]),
}

METRIC_NAMES = ["execution_speed", "code_size", "compilation_time"]


@dataclass
class PriorityProfile:
    """Represents a named priority configuration."""
    name: str
    speed_weight: float
    size_weight: float
    compile_time_weight: float
    description: str = ""

    @property
    def vector(self) -> np.ndarray:
        return np.array([self.speed_weight, self.size_weight, self.compile_time_weight])

    def validate(self) -> bool:
        v = self.vector
        return np.all(v >= 0) and np.all(v <= 1) and np.isclose(v.sum(), 1.0, atol=0.01)


class MetricPriorityVector:
    """
    Manages the Metric Priority Vector system for preference-aware optimization.

    This system allows users to express their optimization preferences as a
    normalized 3D vector, which is then used as additional input features
    for the ML models, enabling the same model to produce different
    optimization strategies based on user needs.
    """

    def __init__(self):
        """Initialize with predefined presets."""
        self.presets = PRIORITY_PRESETS.copy()
        self.metric_names = METRIC_NAMES
        self.dim = len(METRIC_NAMES)
        logger.info(f"Initialized MetricPriorityVector with {len(self.presets)} presets")

    def create_vector(self, speed: float, size: float, compile_time: float) -> np.ndarray:
        """
        Create a normalized priority vector from raw weights.

        Args:
            speed: Weight for execution speed optimization
            size: Weight for code size reduction
            compile_time: Weight for compilation time minimization

        Returns:
            Normalized priority vector of shape (3,)

        Raises:
            ValueError: If all weights are zero or negative
        """
        raw = np.array([speed, size, compile_time], dtype=np.float64)

        if np.any(raw < 0):
            raise ValueError("Priority weights must be non-negative")

        total = raw.sum()
        if total <= 0:
            raise ValueError("At least one priority weight must be positive")

        normalized = raw / total
        logger.debug(f"Created priority vector: speed={normalized[0]:.3f}, "
                     f"size={normalized[1]:.3f}, compile_time={normalized[2]:.3f}")
        return normalized

    def get_preset(self, name: str) -> np.ndarray:
        """
        Get a predefined priority preset.

        Args:
            name: Preset name (e.g., "speed_first", "balanced")

        Returns:
            Priority vector of shape (3,)
        """
        if name not in self.presets:
            available = ", ".join(self.presets.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        return self.presets[name].copy()

    def generate_training_priorities(self, num_samples: int,
                                     strategy: str = "mixed") -> np.ndarray:
        """
        Generate diverse priority vectors for training data augmentation.

        Args:
            num_samples: Number of priority vectors to generate
            strategy: Generation strategy - "mixed", "uniform", "preset_focused"

        Returns:
            numpy array of shape (num_samples, 3)
        """
        rng = np.random.RandomState(42)
        priorities = np.zeros((num_samples, self.dim))

        if strategy == "uniform":
            # Uniform sampling from simplex
            for i in range(num_samples):
                raw = rng.exponential(1.0, size=self.dim)
                priorities[i] = raw / raw.sum()

        elif strategy == "preset_focused":
            # Sample near predefined presets with small noise
            preset_list = list(self.presets.values())
            for i in range(num_samples):
                base = preset_list[i % len(preset_list)].copy()
                noise = rng.normal(0, 0.05, size=self.dim)
                perturbed = np.clip(base + noise, 0.01, 1.0)
                priorities[i] = perturbed / perturbed.sum()

        elif strategy == "mixed":
            # 50% uniform + 30% preset-focused + 20% extreme
            n_uniform = int(num_samples * 0.5)
            n_preset = int(num_samples * 0.3)
            n_extreme = num_samples - n_uniform - n_preset

            # Uniform
            for i in range(n_uniform):
                raw = rng.exponential(1.0, size=self.dim)
                priorities[i] = raw / raw.sum()

            # Preset-focused
            preset_list = list(self.presets.values())
            for i in range(n_uniform, n_uniform + n_preset):
                base = preset_list[i % len(preset_list)].copy()
                noise = rng.normal(0, 0.08, size=self.dim)
                perturbed = np.clip(base + noise, 0.01, 1.0)
                priorities[i] = perturbed / perturbed.sum()

            # Extreme (one dimension dominates)
            for i in range(n_uniform + n_preset, num_samples):
                dominant = rng.randint(0, self.dim)
                vec = np.full(self.dim, 0.05)
                vec[dominant] = 0.90
                priorities[i] = vec / vec.sum()

        logger.info(f"Generated {num_samples} training priority vectors "
                    f"using '{strategy}' strategy")
        return priorities

    def augment_features(self, features: np.ndarray,
                         priority: np.ndarray) -> np.ndarray:
        """
        Append priority vector to feature vectors.

        Args:
            features: Feature matrix of shape (N, 66) or (66,)
            priority: Priority vector of shape (3,)

        Returns:
            Augmented features of shape (N, 69) or (69,)
        """
        if features.ndim == 1:
            return np.concatenate([features, priority])
        else:
            # Broadcast priority to all samples
            priority_matrix = np.tile(priority, (features.shape[0], 1))
            return np.concatenate([features, priority_matrix], axis=1)

    def augment_features_batch(self, features: np.ndarray,
                                priorities: np.ndarray) -> np.ndarray:
        """
        Append different priority vectors to each feature vector.

        Args:
            features: Feature matrix of shape (N, 66)
            priorities: Priority matrix of shape (N, 3)

        Returns:
            Augmented features of shape (N, 69)
        """
        assert features.shape[0] == priorities.shape[0], \
            f"Shape mismatch: features={features.shape[0]}, priorities={priorities.shape[0]}"
        return np.concatenate([features, priorities], axis=1)

    def get_all_presets(self) -> Dict[str, np.ndarray]:
        """Return all available presets."""
        return {name: vec.copy() for name, vec in self.presets.items()}

    def compute_adaptability_score(self, predictions_by_priority: Dict[str, np.ndarray]) -> float:
        """
        Compute how much the model's predictions change across different priorities.

        A high adaptability score means the model is responsive to priority changes.
        A low score means the model ignores the priority input.

        Args:
            predictions_by_priority: Dict mapping preset name to predicted pass sequences

        Returns:
            Adaptability score ∈ [0, 1], where 1 = maximally adaptive
        """
        if len(predictions_by_priority) < 2:
            return 0.0

        predictions = list(predictions_by_priority.values())
        total_pairs = 0
        total_diff = 0

        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Compute fraction of different predictions
                diff = np.mean(predictions[i] != predictions[j])
                total_diff += diff
                total_pairs += 1

        return total_diff / max(1, total_pairs)

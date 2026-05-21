"""
Training package: single-horizon and multi-horizon training, plus optimisation.
"""

from .train import plot_training_history, train_model
from .train_multi_horizon import (
    MultiHorizonVolatilityModel,
    evaluate_multi_horizon_performance,
    train_multi_horizon_model,
)

__all__ = [
    "train_model",
    "plot_training_history",
    "MultiHorizonVolatilityModel",
    "train_multi_horizon_model",
    "evaluate_multi_horizon_performance",
]

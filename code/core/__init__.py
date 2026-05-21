"""
Core package: model architecture and shared utility functions.
"""

from .model import (
    AttentionLayer,
    build_lstm_attention_model,
    compile_model,
    create_callbacks,
    get_attention_weights,
    pinball_loss,
)
from .utils import (
    calculate_mae,
    calculate_qlike,
    calculate_r2,
    calculate_rmse,
    calculate_var,
    christoffersen_test,
    create_sequences,
    diebold_mariano_test,
    kupiec_test,
    load_and_prepare_data,
    train_val_test_split,
)

__all__ = [
    "AttentionLayer",
    "build_lstm_attention_model",
    "compile_model",
    "create_callbacks",
    "get_attention_weights",
    "pinball_loss",
    "calculate_mae",
    "calculate_qlike",
    "calculate_r2",
    "calculate_rmse",
    "calculate_var",
    "christoffersen_test",
    "create_sequences",
    "diebold_mariano_test",
    "kupiec_test",
    "load_and_prepare_data",
    "train_val_test_split",
]

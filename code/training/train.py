"""
Training Script for LSTM-Attention Model
"""

import os
import pickle

import numpy as np
import tensorflow as tf
from core.model import build_lstm_attention_model, compile_model, create_callbacks
from core.utils import load_and_prepare_data, train_val_test_split

# Set seeds for reproducibility
np.random.seed(123)
tf.random.set_seed(456)


def train_model(data_dict, epochs=100, batch_size=64, save_path="../models"):
    """
    Train LSTM-Attention model.

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing train, val, test data
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
    save_path : str
        Path to save trained model

    Returns:
    --------
    model : keras.Model
        Trained model
    history : keras.callbacks.History
        Training history
    """
    # Extract data
    X_train = data_dict["train"]["X"]
    y_train = data_dict["train"]["y"]
    X_val = data_dict["val"]["X"]
    y_val = data_dict["val"]["y"]

    print("\n" + "=" * 70)
    print("TRAINING LSTM-ATTENTION MODEL")
    print("=" * 70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input shape: {X_train.shape}")
    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {epochs}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (time_steps, features)
    model = build_lstm_attention_model(input_shape)
    model = compile_model(model)

    # Create callbacks
    callbacks = create_callbacks(patience=15, lr_patience=5)

    # Prepare targets for both outputs
    y_train_dict = {
        "volatility": y_train,
        "var": y_train,  # Using same target, but different loss functions
    }

    y_val_dict = {"volatility": y_val, "var": y_val}

    print("\nStarting training...")
    print("-" * 70)

    # Train model
    history = model.fit(
        X_train,
        y_train_dict,
        validation_data=(X_val, y_val_dict),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTraining completed!")

    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "lstm_attention_model.keras")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save training history
    history_path = os.path.join(save_path, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"Training history saved to: {history_path}")

    return model, history


def plot_training_history(history, save_path="../docs/figures"):
    """
    Plot and save training history.

    Parameters:
    -----------
    history : keras.callbacks.History or dict
        Training history
    save_path : str
        Path to save figure
    """
    import matplotlib.pyplot as plt

    # Extract history dict
    if hasattr(history, "history"):
        history_dict = history.history
    else:
        history_dict = history

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot total loss
    axes[0].plot(history_dict["loss"], label="Training Loss", linewidth=2)
    axes[0].plot(history_dict["val_loss"], label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot volatility loss: key name varies across TF versions
    vol_loss_key = next(
        (k for k in ("volatility_loss", "output_1_loss") if k in history_dict),
        None,
    )
    val_vol_loss_key = next(
        (k for k in ("val_volatility_loss", "val_output_1_loss") if k in history_dict),
        None,
    )

    if vol_loss_key and val_vol_loss_key:
        axes[1].plot(history_dict[vol_loss_key], label="Training Vol Loss", linewidth=2)
        axes[1].plot(
            history_dict[val_vol_loss_key], label="Validation Vol Loss", linewidth=2
        )
        axes[1].set_ylabel("Volatility Loss (MSE)", fontsize=12)
        axes[1].set_title("Volatility Prediction Loss", fontsize=14, fontweight="bold")
    else:
        # Fallback: repeat the total loss if per-output keys are unavailable
        axes[1].plot(history_dict["loss"], label="Training Loss", linewidth=2)
        axes[1].plot(history_dict["val_loss"], label="Validation Loss", linewidth=2)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title(
            "Training Loss (fallback view)", fontsize=14, fontweight="bold"
        )

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "training_validation_loss.png")
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to: {save_file}")

    plt.close()


if __name__ == "__main__":
    # Load and prepare data
    print("Loading data...")
    df, feature_cols = load_and_prepare_data("./data/synthetic_data.csv")

    # Split data
    data_dict = train_val_test_split(
        df,
        feature_cols,
        target_col="realized_volatility",
        train_end="2022-12-31",
        val_end="2023-06-30",
    )

    # Train model
    model, history = train_model(
        data_dict, epochs=100, batch_size=64, save_path="../models"
    )

    # Plot training history
    plot_training_history(history, save_path="../docs/figures")

    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 70)

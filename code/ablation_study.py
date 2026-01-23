"""
Ablation Study: Component-wise Performance Analysis
Compares LSTM-only vs LSTM+Attention vs LSTM+Attention+SHAP
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import time

from model import AttentionLayer, pinball_loss, create_callbacks
from utils import calculate_rmse, calculate_mae, calculate_r2
import shap

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)


class LSTMOnlyModel:
    """Baseline LSTM model without attention or SHAP"""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        lstm_units: List[int] = [128, 64],
        dense_units: int = 32,
        dropout: float = 0.2,
    ) -> Model:
        """Build LSTM-only model"""

        inputs = layers.Input(shape=input_shape, name="input")

        # LSTM layers
        x = layers.LSTM(
            lstm_units[0], return_sequences=True, dropout=dropout, name="lstm_1"
        )(inputs)

        x = layers.LSTM(
            lstm_units[1],
            return_sequences=False,  # Don't return sequences
            dropout=dropout,
            name="lstm_2",
        )(x)

        # Dense layer
        x = layers.Dense(dense_units, activation="relu", name="dense")(x)
        x = layers.Dropout(dropout)(x)

        # Output
        volatility_output = layers.Dense(1, activation="linear", name="volatility")(x)
        var_output = layers.Dense(1, activation="linear", name="var")(x)

        model = Model(
            inputs=inputs, outputs=[volatility_output, var_output], name="LSTM_Only"
        )

        return model


class LSTMAttentionModel:
    """LSTM + Attention model without SHAP"""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        lstm_units: List[int] = [128, 64],
        attention_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
    ) -> Model:
        """Build LSTM+Attention model"""

        inputs = layers.Input(shape=input_shape, name="input")

        # LSTM layers
        x = layers.LSTM(
            lstm_units[0], return_sequences=True, dropout=dropout, name="lstm_1"
        )(inputs)

        x = layers.LSTM(
            lstm_units[1],
            return_sequences=True,  # Return sequences for attention
            dropout=dropout,
            name="lstm_2",
        )(x)

        # Attention layer
        context_vector, attention_weights = AttentionLayer(
            units=attention_units, name="attention"
        )(x)

        # Dense layer
        x = layers.Dense(dense_units, activation="relu", name="dense")(context_vector)
        x = layers.Dropout(dropout)(x)

        # Output
        volatility_output = layers.Dense(1, activation="linear", name="volatility")(x)
        var_output = layers.Dense(1, activation="linear", name="var")(x)

        model = Model(
            inputs=inputs,
            outputs=[volatility_output, var_output],
            name="LSTM_Attention",
        )

        return model


class LSTMAttentionSHAPModel:
    """Full model: LSTM + Attention + SHAP interpretability"""

    @staticmethod
    def build(
        input_shape: Tuple[int, int],
        lstm_units: List[int] = [128, 64],
        attention_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
    ) -> Model:
        """Build complete LSTM+Attention+SHAP model (same as LSTM+Attention structurally)"""

        # Architecture is same as LSTM+Attention
        # SHAP is applied post-training for interpretability
        return LSTMAttentionModel.build(
            input_shape, lstm_units, attention_units, dense_units, dropout
        )


class AblationStudy:
    """Conducts ablation study across model variants"""

    def __init__(self, data_dict: Dict):
        self.data_dict = data_dict
        self.results = {}

    def compile_model(self, model: Model, learning_rate: float = 1e-3) -> Model:
        """Compile model with standard configuration"""

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        losses = {
            "volatility": "mse",
            "var": lambda y_true, y_pred: pinball_loss(y_true, y_pred, tau=0.01),
        }

        loss_weights = {"volatility": 1.0, "var": 0.5}

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics={"volatility": ["mae", "mse"], "var": ["mae"]},
        )

        return model

    def train_and_evaluate(
        self, model: Model, model_name: str, epochs: int = 100, batch_size: int = 64
    ) -> Dict:
        """
        Train and evaluate a single model variant

        Returns:
        --------
        results : dict
            Performance metrics and training time
        """

        print(f"\n{'='*80}")
        print(f"Training {model_name}")
        print(f"{'='*80}")

        X_train = self.data_dict["train"]["X"]
        y_train = self.data_dict["train"]["y"]
        X_val = self.data_dict["val"]["X"]
        y_val = self.data_dict["val"]["y"]
        X_test = self.data_dict["test"]["X"]
        y_test = self.data_dict["test"]["y"]

        # Prepare targets
        y_train_dict = {"volatility": y_train, "var": y_train}
        y_val_dict = {"volatility": y_val, "var": y_val}

        # Compile model
        model = self.compile_model(model)

        print(f"Model parameters: {model.count_params():,}")

        # Train
        callbacks = create_callbacks(patience=15, lr_patience=5)

        start_time = time.time()

        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,  # Suppress output for cleaner ablation logs
        )

        training_time = time.time() - start_time

        # Evaluate
        predictions = model.predict(X_test, verbose=0)
        y_pred = predictions[0].flatten()

        # Calculate metrics
        rmse = calculate_rmse(y_test, y_pred)
        mae = calculate_mae(y_test, y_pred)
        r2 = calculate_r2(y_test, y_pred)

        # Calculate SHAP values if full model
        shap_time = 0
        shap_feature_importance = None

        if "SHAP" in model_name:
            print("\nCalculating SHAP values for interpretability...")
            shap_start = time.time()

            # Use a subset for SHAP calculation
            background = X_train[:100]
            test_samples = X_test[:50]

            # Create explainer
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(test_samples)

            # Calculate feature importance (mean absolute SHAP values)
            if isinstance(shap_values, list):
                shap_values_array = shap_values[0]  # Volatility head
            else:
                shap_values_array = shap_values

            # Average across time steps and samples
            shap_feature_importance = np.mean(
                np.abs(shap_values_array).reshape(len(test_samples), -1), axis=0
            )

            shap_time = time.time() - shap_start

        results = {
            "model_name": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "training_time": training_time,
            "shap_time": shap_time,
            "total_params": model.count_params(),
            "epochs_trained": len(history.history["loss"]),
            "final_train_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
            "shap_feature_importance": shap_feature_importance,
            "history": history.history,
        }

        print(f"\nResults:")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE:  {mae:.6f}")
        print(f"  R²:   {r2:.6f}")
        print(f"  Training time: {training_time:.2f}s")
        if shap_time > 0:
            print(f"  SHAP calculation time: {shap_time:.2f}s")

        return results

    def run_full_ablation(
        self, epochs: int = 100, batch_size: int = 64
    ) -> pd.DataFrame:
        """
        Run complete ablation study across all variants

        Returns:
        --------
        comparison_df : pd.DataFrame
            Comparison table of all variants
        """

        print("\n" + "=" * 80)
        print("ABLATION STUDY: Component-wise Performance Analysis")
        print("=" * 80)
        print("\nVariants:")
        print("  1. LSTM-only (baseline)")
        print("  2. LSTM + Attention")
        print("  3. LSTM + Attention + SHAP (full model)")

        input_shape = (
            self.data_dict["train"]["X"].shape[1],
            self.data_dict["train"]["X"].shape[2],
        )

        # Variant 1: LSTM-only
        model_lstm = LSTMOnlyModel.build(input_shape)
        results_lstm = self.train_and_evaluate(
            model_lstm, "LSTM-only", epochs, batch_size
        )
        self.results["LSTM-only"] = results_lstm

        # Variant 2: LSTM + Attention
        model_attention = LSTMAttentionModel.build(input_shape)
        results_attention = self.train_and_evaluate(
            model_attention, "LSTM+Attention", epochs, batch_size
        )
        self.results["LSTM+Attention"] = results_attention

        # Variant 3: LSTM + Attention + SHAP
        model_full = LSTMAttentionSHAPModel.build(input_shape)
        results_full = self.train_and_evaluate(
            model_full, "LSTM+Attention+SHAP", epochs, batch_size
        )
        self.results["LSTM+Attention+SHAP"] = results_full

        # Create comparison table
        comparison_df = self._create_comparison_table()

        return comparison_df

    def _create_comparison_table(self) -> pd.DataFrame:
        """Create formatted comparison table"""

        comparison_data = []

        baseline_rmse = self.results["LSTM-only"]["rmse"]

        for variant_name, results in self.results.items():
            # Calculate improvement over baseline
            rmse_improvement = ((baseline_rmse - results["rmse"]) / baseline_rmse) * 100

            comparison_data.append(
                {
                    "Model Variant": variant_name,
                    "RMSE (×10⁻²)": results["rmse"] * 100,
                    "MAE (×10⁻²)": results["mae"] * 100,
                    "R² Score": results["r2"],
                    "Improvement (%)": rmse_improvement,
                    "Parameters": results["total_params"],
                    "Training Time (s)": results["training_time"],
                    "Epochs": results["epochs_trained"],
                }
            )

        comparison_df = pd.DataFrame(comparison_data)

        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)
        print(comparison_df.to_string(index=False))

        return comparison_df

    def plot_ablation_results(self, save_path: str = "../figures"):
        """Generate visualization of ablation study results"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        variants = list(self.results.keys())
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Performance comparison
        ax = axes[0, 0]
        rmse_values = [self.results[v]["rmse"] * 100 for v in variants]
        bars = ax.bar(
            variants,
            rmse_values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("RMSE (×10⁻²)", fontsize=12, fontweight="bold")
        ax.set_title("Performance Comparison", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # R² scores
        ax = axes[0, 1]
        r2_values = [self.results[v]["r2"] for v in variants]
        bars = ax.bar(
            variants,
            r2_values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("R² Score", fontsize=12, fontweight="bold")
        ax.set_title("Goodness of Fit", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Training time comparison
        ax = axes[1, 0]
        training_times = [self.results[v]["training_time"] for v in variants]
        bars = ax.bar(
            variants,
            training_times,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.set_ylabel("Training Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_title("Training Efficiency", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")

        # Training curves
        ax = axes[1, 1]
        for i, variant in enumerate(variants):
            history = self.results[variant]["history"]
            ax.plot(history["val_loss"], label=variant, linewidth=2, color=colors[i])

        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
        ax.set_title("Training Convergence", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, "ablation_study_results.png")
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"\nAblation study plot saved to: {save_file}")

        plt.close()


if __name__ == "__main__":
    print("Ablation study module loaded.")
    print("Use this module to compare model variants.")

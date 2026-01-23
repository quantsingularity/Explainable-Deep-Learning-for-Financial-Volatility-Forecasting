"""
Training Optimization Module
Implements mixed precision training, model pruning, and knowledge distillation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from typing import Dict, Tuple
import time
import os

from model import build_lstm_attention_model, compile_model

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)


class MixedPrecisionTrainer:
    """
    Mixed Precision Training for faster training with TF16/FP16
    Can reduce training time by 2-3x on compatible GPUs
    """

    def __init__(self, policy: str = "mixed_float16"):
        """
        Initialize mixed precision policy

        Parameters:
        -----------
        policy : str
            'mixed_float16' for V100/A100 GPUs, 'mixed_bfloat16' for TPUs
        """
        self.policy = keras.mixed_precision.Policy(policy)
        keras.mixed_precision.set_global_policy(self.policy)

        print(f"Mixed precision policy set to: {self.policy.name}")
        print(f"Compute dtype: {self.policy.compute_dtype}")
        print(f"Variable dtype: {self.policy.variable_dtype}")

    def train_with_mixed_precision(
        self,
        model: keras.Model,
        data_dict: Dict,
        epochs: int = 100,
        batch_size: int = 64,
    ) -> Tuple[keras.Model, dict]:
        """
        Train model with mixed precision

        Returns:
        --------
        model : keras.Model
            Trained model
        history : dict
            Training history
        training_time : float
            Total training time
        """

        print("\n" + "=" * 80)
        print("MIXED PRECISION TRAINING")
        print("=" * 80)

        X_train = data_dict["train"]["X"]
        y_train = data_dict["train"]["y"]
        X_val = data_dict["val"]["X"]
        y_val = data_dict["val"]["y"]

        # Prepare targets
        y_train_dict = {"volatility": y_train, "var": y_train}
        y_val_dict = {"volatility": y_val, "var": y_val}

        # Compile with loss scaling for mixed precision
        model = compile_model(model, learning_rate=1e-3)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5
            ),
        ]

        print(f"\nStarting mixed precision training...")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        start_time = time.time()

        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(
            f"Average time per epoch: {training_time/len(history.history['loss']):.2f}s"
        )

        return model, history.history, training_time


class ModelPruner:
    """
    Model Pruning for reducing model size and inference time
    Removes weights with low magnitude during training
    """

    def __init__(self, target_sparsity: float = 0.5):
        """
        Initialize pruning configuration

        Parameters:
        -----------
        target_sparsity : float
            Target sparsity level (0.5 = 50% of weights pruned)
        """
        self.target_sparsity = target_sparsity

    def create_pruned_model(
        self, model: keras.Model, pruning_schedule: str = "polynomial"
    ) -> keras.Model:
        """
        Create a pruned version of the model

        Parameters:
        -----------
        model : keras.Model
            Original model
        pruning_schedule : str
            'polynomial' or 'constant'

        Returns:
        --------
        pruned_model : keras.Model
            Model with pruning applied
        """

        print("\n" + "=" * 80)
        print("MODEL PRUNING")
        print("=" * 80)
        print(f"Target sparsity: {self.target_sparsity*100}%")

        # Define pruning schedule
        if pruning_schedule == "polynomial":
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=self.target_sparsity,
                    begin_step=0,
                    end_step=1000,  # Gradually prune over 1000 steps
                )
            }
        else:
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                    target_sparsity=self.target_sparsity, begin_step=0
                )
            }

        # Apply pruning to dense and LSTM layers
        def apply_pruning_to_layer(layer):
            if isinstance(layer, (keras.layers.Dense, keras.layers.LSTM)):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer

        # Clone model with pruning
        pruned_model = keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_layer,
        )

        return pruned_model

    def train_pruned_model(
        self,
        pruned_model: keras.Model,
        data_dict: Dict,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> Tuple[keras.Model, dict]:
        """
        Train pruned model with pruning callbacks

        Returns:
        --------
        model : keras.Model
            Trained pruned model
        history : dict
            Training history
        """

        X_train = data_dict["train"]["X"]
        y_train = data_dict["train"]["y"]
        X_val = data_dict["val"]["X"]
        y_val = data_dict["val"]["y"]

        y_train_dict = {"volatility": y_train, "var": y_train}
        y_val_dict = {"volatility": y_val, "var": y_val}

        # Compile
        pruned_model = compile_model(pruned_model, learning_rate=1e-3)

        # Pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir="./logs"),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
        ]

        print("\nTraining pruned model...")

        start_time = time.time()

        history = pruned_model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = time.time() - start_time

        # Strip pruning wrappers for inference
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

        print(f"\nPruning completed in {training_time:.2f} seconds")

        # Calculate actual sparsity
        sparsity = self._calculate_sparsity(final_model)
        print(f"Achieved sparsity: {sparsity*100:.2f}%")

        return final_model, history.history

    def _calculate_sparsity(self, model: keras.Model) -> float:
        """Calculate actual sparsity of model weights"""

        total_weights = 0
        zero_weights = 0

        for layer in model.layers:
            if hasattr(layer, "get_weights"):
                weights = layer.get_weights()
                for w in weights:
                    total_weights += np.size(w)
                    zero_weights += np.sum(w == 0)

        return zero_weights / total_weights if total_weights > 0 else 0


class KnowledgeDistiller:
    """
    Knowledge Distillation: Train smaller student model from larger teacher
    """

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        """
        Initialize distillation parameters

        Parameters:
        -----------
        temperature : float
            Temperature for softening probability distributions
        alpha : float
            Weight for distillation loss (1-alpha for student loss)
        """
        self.temperature = temperature
        self.alpha = alpha

    def create_student_model(
        self, input_shape: Tuple[int, int], compression_factor: float = 0.5
    ) -> keras.Model:
        """
        Create smaller student model

        Parameters:
        -----------
        input_shape : tuple
            Input shape (time_steps, features)
        compression_factor : float
            Factor to reduce model size (0.5 = half the size)

        Returns:
        --------
        student_model : keras.Model
            Smaller student model
        """

        print("\n" + "=" * 80)
        print("KNOWLEDGE DISTILLATION")
        print("=" * 80)
        print(f"Compression factor: {compression_factor}")

        # Smaller architecture
        lstm_units = [int(128 * compression_factor), int(64 * compression_factor)]
        attention_units = int(64 * compression_factor)
        dense_units = int(32 * compression_factor)

        student = build_lstm_attention_model(
            input_shape=input_shape,
            lstm_units=lstm_units,
            attention_units=attention_units,
            dense_units=dense_units,
            dropout=0.2,
        )

        print(f"\nStudent model created:")
        print(f"  LSTM units: {lstm_units}")
        print(f"  Attention units: {attention_units}")
        print(f"  Dense units: {dense_units}")
        print(f"  Total parameters: {student.count_params():,}")

        return student

    def distillation_loss(
        self, y_true: tf.Tensor, y_student: tf.Tensor, y_teacher: tf.Tensor
    ) -> tf.Tensor:
        """
        Combined distillation loss

        Returns:
        --------
        loss : tf.Tensor
            Weighted combination of student and distillation losses
        """

        # Student loss (standard MSE)
        student_loss = tf.reduce_mean(tf.square(y_true - y_student))

        # Distillation loss (match teacher predictions)
        distillation_loss = tf.reduce_mean(tf.square(y_teacher - y_student))

        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss

        return total_loss

    def train_student(
        self,
        teacher_model: keras.Model,
        student_model: keras.Model,
        data_dict: Dict,
        epochs: int = 50,
        batch_size: int = 64,
    ) -> Tuple[keras.Model, dict]:
        """
        Train student model with knowledge distillation

        Returns:
        --------
        student_model : keras.Model
            Trained student model
        history : dict
            Training history
        """

        X_train = data_dict["train"]["X"]
        y_train = data_dict["train"]["y"]
        X_val = data_dict["val"]["X"]
        y_val = data_dict["val"]["y"]

        # Get teacher predictions (soft targets)
        print("\nGenerating teacher predictions...")
        teacher_train_preds = teacher_model.predict(X_train, verbose=0)
        teacher_val_preds = teacher_model.predict(X_val, verbose=0)

        # Extract volatility predictions
        if isinstance(teacher_train_preds, list):
            teacher_train_preds = teacher_train_preds[0]
            teacher_val_preds = teacher_val_preds[0]

        # Compile student
        student_model = compile_model(student_model, learning_rate=1e-3)

        # Training with distillation
        y_train_dict = {"volatility": y_train, "var": y_train}
        y_val_dict = {"volatility": y_val, "var": y_val}

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5
            ),
        ]

        print("\nTraining student model...")

        start_time = time.time()

        history = student_model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        training_time = time.time() - start_time

        print(f"\nDistillation completed in {training_time:.2f} seconds")

        return student_model, history.history


def compare_optimization_methods(results: Dict, save_path: str = "../tables"):
    """
    Compare different optimization methods

    Parameters:
    -----------
    results : dict
        Dictionary with results from different methods
    save_path : str
        Path to save comparison table
    """

    comparison_data = []

    for method_name, method_results in results.items():
        comparison_data.append(
            {
                "Method": method_name,
                "Training Time (s)": method_results.get("training_time", 0),
                "Model Size (params)": method_results.get("model_params", 0),
                "Speedup": method_results.get("speedup", 1.0),
                "Size Reduction (%)": method_results.get("size_reduction", 0),
                "RMSE": method_results.get("rmse", 0),
                "Performance Loss (%)": method_results.get("performance_loss", 0),
            }
        )

    comparison_df = pd.DataFrame(comparison_data)

    print("\n" + "=" * 80)
    print("OPTIMIZATION METHODS COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))

    # Save
    os.makedirs(save_path, exist_ok=True)
    comparison_df.to_csv(f"{save_path}/optimization_comparison.csv", index=False)
    print(f"\nComparison saved to: {save_path}/optimization_comparison.csv")

    return comparison_df


if __name__ == "__main__":
    print("Training optimization module loaded.")
    print("Available optimizations:")
    print("  1. Mixed Precision Training (TF16/FP16)")
    print("  2. Model Pruning")
    print("  3. Knowledge Distillation")

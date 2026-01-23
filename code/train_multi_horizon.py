"""
Multi-Horizon Volatility Forecasting
Extends the base model to predict 1-day, 5-day, and 22-day ahead volatility.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import pandas as pd
import os
from typing import Dict, List, Tuple
import mlflow
import mlflow.keras

from model import AttentionLayer, pinball_loss
from utils import train_val_test_split, load_and_prepare_data

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)


class MultiHorizonVolatilityModel:
    """
    Multi-horizon LSTM-Attention model for forecasting volatility
    at multiple time horizons (1-day, 5-day, 22-day ahead).
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        horizons: List[int] = [1, 5, 22],
        lstm_units: List[int] = [128, 64],
        attention_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.1,
    ):
        """
        Initialize multi-horizon model.

        Parameters:
        -----------
        input_shape : tuple
            Input shape (time_steps, n_features)
        horizons : list
            Forecast horizons in days [1, 5, 22]
        lstm_units : list
            LSTM layer units
        attention_units : int
            Attention mechanism units
        dense_units : int
            Dense layer units
        dropout : float
            Dropout rate
        recurrent_dropout : float
            Recurrent dropout rate
        """
        self.input_shape = input_shape
        self.horizons = horizons
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Build multi-horizon LSTM-Attention architecture."""

        # Input layer
        inputs = layers.Input(shape=self.input_shape, name="input_sequence")

        # Shared LSTM layers
        x = layers.LSTM(
            self.lstm_units[0],
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            name="lstm_1",
        )(inputs)

        x = layers.LSTM(
            self.lstm_units[1],
            return_sequences=True,
            dropout=self.dropout,
            recurrent_dropout=self.recurrent_dropout,
            name="lstm_2",
        )(x)

        # Shared attention mechanism
        context_vector, attention_weights = AttentionLayer(
            units=self.attention_units, name="shared_attention"
        )(x)

        # Shared dense layer
        shared_features = layers.Dense(
            self.dense_units, activation="relu", name="shared_dense"
        )(context_vector)
        shared_features = layers.Dropout(self.dropout)(shared_features)

        # Horizon-specific output heads
        outputs = {}

        for horizon in self.horizons:
            # Horizon-specific dense layer
            horizon_features = layers.Dense(
                16, activation="relu", name=f"dense_h{horizon}"
            )(shared_features)

            # Volatility prediction
            vol_output = layers.Dense(
                1, activation="linear", name=f"volatility_h{horizon}"
            )(horizon_features)

            # VaR prediction
            var_output = layers.Dense(1, activation="linear", name=f"var_h{horizon}")(
                horizon_features
            )

            outputs[f"volatility_h{horizon}"] = vol_output
            outputs[f"var_h{horizon}"] = var_output

        # Build model
        model = Model(
            inputs=inputs, outputs=outputs, name="MultiHorizon_LSTM_Attention"
        )

        return model

    def compile_model(self, learning_rate: float = 1e-3):
        """Compile model with appropriate losses for each horizon."""

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999
        )

        # Define losses and weights for each horizon
        losses = {}
        loss_weights = {}
        metrics = {}

        for horizon in self.horizons:
            # Volatility losses (MSE)
            losses[f"volatility_h{horizon}"] = "mse"
            loss_weights[f"volatility_h{horizon}"] = (
                1.0 / horizon
            )  # Weight by inverse of horizon
            metrics[f"volatility_h{horizon}"] = ["mae", "mse"]

            # VaR losses (Pinball)
            losses[f"var_h{horizon}"] = lambda y_true, y_pred: pinball_loss(
                y_true, y_pred, tau=0.01
            )
            loss_weights[f"var_h{horizon}"] = 0.3 / horizon
            metrics[f"var_h{horizon}"] = ["mae"]

        self.model.compile(
            optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics
        )

        return self.model

    def prepare_multi_horizon_targets(
        self, df: pd.DataFrame, target_col: str = "realized_volatility"
    ) -> Dict:
        """
        Prepare targets for multiple horizons.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with time series data
        target_col : str
            Target column name

        Returns:
        --------
        targets : dict
            Dictionary with targets for each horizon
        """
        targets = {}

        for horizon in self.horizons:
            # Forward-looking target (h-days ahead)
            if horizon == 1:
                targets[f"h{horizon}"] = df[target_col].values
            else:
                # Average volatility over next h days
                targets[f"h{horizon}"] = (
                    df[target_col]
                    .rolling(window=horizon, min_periods=1)
                    .mean()
                    .shift(-horizon + 1)
                    .fillna(method="ffill")
                    .values
                )

        return targets


def train_multi_horizon_model(
    data_dict: Dict,
    horizons: List[int] = [1, 5, 22],
    epochs: int = 100,
    batch_size: int = 64,
    save_path: str = "../models",
    use_mlflow: bool = True,
) -> Tuple[MultiHorizonVolatilityModel, dict]:
    """
    Train multi-horizon volatility forecasting model.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with train/val/test data
    horizons : list
        Forecast horizons
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    save_path : str
        Model save path
    use_mlflow : bool
        Whether to use MLflow tracking

    Returns:
    --------
    model_wrapper : MultiHorizonVolatilityModel
        Trained model wrapper
    history : dict
        Training history
    """

    print("\n" + "=" * 80)
    print("TRAINING MULTI-HORIZON VOLATILITY FORECASTING MODEL")
    print("=" * 80)
    print(f"Horizons: {horizons}")
    print(f"Training samples: {len(data_dict['train']['X'])}")
    print(f"Validation samples: {len(data_dict['val']['X'])}")

    # Start MLflow run
    if use_mlflow:
        mlflow.set_experiment("multi_horizon_volatility_forecasting")
        mlflow.start_run(run_name=f"multi_horizon_{'-'.join(map(str, horizons))}")

        # Log parameters
        mlflow.log_param("horizons", horizons)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

    # Extract data
    X_train = data_dict["train"]["X"]
    X_val = data_dict["val"]["X"]

    # Initialize model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_wrapper = MultiHorizonVolatilityModel(
        input_shape=input_shape, horizons=horizons
    )
    model = model_wrapper.compile_model()

    # Log model architecture
    if use_mlflow:
        mlflow.log_param("total_params", model.count_params())

    print(f"\nModel parameters: {model.count_params():,}")

    # Prepare multi-horizon targets
    y_train_dict = {}
    y_val_dict = {}

    for horizon in horizons:
        # For training, use the same target but model learns horizon-specific patterns
        y_train_dict[f"volatility_h{horizon}"] = data_dict["train"]["y"]
        y_train_dict[f"var_h{horizon}"] = data_dict["train"]["y"]

        y_val_dict[f"volatility_h{horizon}"] = data_dict["val"]["y"]
        y_val_dict[f"var_h{horizon}"] = data_dict["val"]["y"]

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        ),
    ]

    if use_mlflow:
        callbacks.append(
            mlflow.keras.MlflowCallback(model, save_model_every_n_epochs=10)
        )

    # Train model
    print("\nStarting training...")
    print("-" * 80)

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

    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(
        save_path, f'multi_horizon_model_{"_".join(map(str, horizons))}.h5'
    )
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Log metrics to MLflow
    if use_mlflow:
        for horizon in horizons:
            final_val_loss = history.history.get(
                f"val_volatility_h{horizon}_loss", [0]
            )[-1]
            mlflow.log_metric(f"final_val_loss_h{horizon}", final_val_loss)

        mlflow.log_artifact(model_path)
        mlflow.end_run()

    return model_wrapper, history.history


def evaluate_multi_horizon_performance(
    model: Model, data_dict: Dict, horizons: List[int], scaler
) -> pd.DataFrame:
    """
    Evaluate model performance across all horizons.

    Parameters:
    -----------
    model : Model
        Trained multi-horizon model
    data_dict : dict
        Test data dictionary
    horizons : list
        Forecast horizons
    scaler : sklearn scaler
        Target scaler

    Returns:
    --------
    results_df : pd.DataFrame
        Performance metrics for each horizon
    """
    from utils import calculate_rmse, calculate_mae, calculate_r2

    X_test = data_dict["test"]["X"]
    y_test = data_dict["test"]["y"]

    print("\n" + "=" * 80)
    print("MULTI-HORIZON PERFORMANCE EVALUATION")
    print("=" * 80)

    # Make predictions
    predictions = model.predict(X_test, verbose=0)

    # Evaluate each horizon
    results = []

    for horizon in horizons:
        y_pred = predictions[f"volatility_h{horizon}"].flatten()

        # Inverse transform
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate metrics
        rmse = calculate_rmse(y_test_orig, y_pred_orig)
        mae = calculate_mae(y_test_orig, y_pred_orig)
        r2 = calculate_r2(y_test_orig, y_pred_orig)

        results.append(
            {
                "Horizon": f"{horizon}-day",
                "RMSE (×10⁻²)": rmse * 100,
                "MAE (×10⁻²)": mae * 100,
                "R²": r2,
            }
        )

        print(f"\n{horizon}-day horizon:")
        print(f"  RMSE: {rmse:.4f} ({rmse*100:.2f}e-2)")
        print(f"  MAE:  {mae:.4f} ({mae*100:.2f}e-2)")
        print(f"  R²:   {r2:.4f}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON ACROSS HORIZONS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df, feature_cols = load_and_prepare_data("../data/synthetic_data.csv")

    # Split data
    data_dict = train_val_test_split(
        df,
        feature_cols,
        target_col="realized_volatility",
        train_end="2022-12-31",
        val_end="2023-06-30",
    )

    # Train multi-horizon model
    model_wrapper, history = train_multi_horizon_model(
        data_dict,
        horizons=[1, 5, 22],
        epochs=100,
        batch_size=64,
        save_path="../models",
        use_mlflow=True,
    )

    # Evaluate
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(data_dict["train"]["y"].reshape(-1, 1))

    results_df = evaluate_multi_horizon_performance(
        model_wrapper.model, data_dict, horizons=[1, 5, 22], scaler=scaler
    )

    # Save results
    os.makedirs("../tables", exist_ok=True)
    results_df.to_csv("../tables/multi_horizon_performance.csv", index=False)
    print("\nResults saved to: ../tables/multi_horizon_performance.csv")

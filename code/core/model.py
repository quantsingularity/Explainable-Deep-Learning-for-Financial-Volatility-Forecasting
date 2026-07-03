"""
LSTM-Attention Model Architecture for Volatility Forecasting
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers

# Version-agnostic access to register_keras_serializable:
# - TF 2.14 / Keras 2: tf.keras.saving.register_keras_serializable
# - TF 2.16+ / Keras 3: keras.saving.register_keras_serializable
#   (the tf.keras lazy proxy no longer exposes .saving)
if hasattr(keras, "saving") and hasattr(keras.saving, "register_keras_serializable"):
    register_keras_serializable = keras.saving.register_keras_serializable
elif hasattr(keras, "utils") and hasattr(keras.utils, "register_keras_serializable"):
    register_keras_serializable = keras.utils.register_keras_serializable
else:  # very old fallback: no-op decorator

    def register_keras_serializable(package="Custom", name=None):
        def _decorator(obj):
            return obj

        return _decorator


# Set random seeds for reproducibility
tf.random.set_seed(456)
np.random.seed(123)


@register_keras_serializable(package="VolatilityForecasting")
class AttentionLayer(layers.Layer):
    """
    Bahdanau-style Attention mechanism for sequence models.
    Computes attention weights and context vector from LSTM outputs.
    """

    def __init__(self, units=64, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Initialize attention weights."""
        # W_a: weight matrix for alignment model
        self.W_a = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # U_a: weight matrix for hidden state
        self.U_a = self.add_weight(
            name="attention_u",
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        # v_a: weight vector for computing attention scores
        self.v_a = self.add_weight(
            name="attention_v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b_a = self.add_weight(
            name="attention_bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Compute attention weights and context vector.

        Parameters:
        -----------
        inputs : tensor
            LSTM outputs (batch_size, time_steps, lstm_units)

        Returns:
        --------
        context_vector : tensor
            Weighted sum of inputs (batch_size, lstm_units)
        attention_weights : tensor
            Attention weights (batch_size, time_steps, 1)
        """
        score = tf.nn.tanh(
            tf.tensordot(inputs, self.W_a, axes=1)
            + tf.tensordot(inputs, self.U_a, axes=1)
            + self.b_a
        )
        attention_scores = tf.tensordot(score, self.v_a, axes=1)

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Compute context vector as weighted sum
        context_vector = tf.reduce_sum(attention_weights * inputs, axis=1)

        return context_vector, attention_weights

    def get_config(self):
        """Return configuration for serialization."""
        config = super(AttentionLayer, self).get_config()
        config.update({"units": self.units})
        return config


def build_lstm_attention_model(
    input_shape,
    lstm_units=None,
    attention_units=64,
    dense_units=32,
    dropout=0.2,
    recurrent_dropout=0.1,
):
    """
    Build LSTM-Attention model for volatility and VaR forecasting.

    Parameters:
    -----------
    input_shape : tuple
        Input shape (time_steps, n_features)
    lstm_units : list
        Number of units in each LSTM layer (default: [128, 64])
    attention_units : int
        Number of units in attention mechanism
    dense_units : int
        Number of units in dense layer
    dropout : float
        Dropout rate for LSTM and Dense layers
    recurrent_dropout : float
        Recurrent dropout rate for LSTM

    Returns:
    --------
    model : keras.Model
        Compiled LSTM-Attention model
    """
    if lstm_units is None:
        lstm_units = [128, 64]
    # Input layer
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    # First LSTM layer (return sequences for stacking)
    x = layers.LSTM(
        lstm_units[0],
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="lstm_1",
    )(inputs)

    # Second LSTM layer (return sequences for attention)
    x = layers.LSTM(
        lstm_units[1],
        return_sequences=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="lstm_2",
    )(x)

    # Attention layer
    context_vector, attention_weights = AttentionLayer(
        units=attention_units, name="attention"
    )(x)

    # Dense layer with dropout
    x = layers.Dense(dense_units, activation="relu", name="dense_1")(context_vector)
    x = layers.Dropout(dropout, name="dropout")(x)

    # Output heads
    # Volatility forecasting head
    volatility_output = layers.Dense(1, activation="linear", name="volatility")(x)

    # VaR forecasting head (using same features)
    var_output = layers.Dense(1, activation="linear", name="var")(x)

    # Build model with two outputs
    model = Model(
        inputs=inputs,
        outputs=[volatility_output, var_output],
        name="LSTM_Attention_SHAP",
    )

    return model


@register_keras_serializable(package="VolatilityForecasting")
def pinball_loss(y_true, y_pred, tau=0.01):
    """
    Pinball loss (quantile loss) for VaR prediction.
    Registered as a Keras-serializable function so it is preserved
    when a model is saved and reloaded in .keras format.

    Parameters:
    -----------
    y_true : tensor
        True values
    y_pred : tensor
        Predicted values
    tau : float
        Quantile level (0.01 for 99% VaR)

    Returns:
    --------
    loss : tensor
        Pinball loss value
    """
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(tau * error, (tau - 1) * error))


def compile_model(model, learning_rate=1e-3):
    """
    Compile model with appropriate loss functions and optimizer.

    Parameters:
    -----------
    model : keras.Model
        Model to compile
    learning_rate : float
        Learning rate for Adam optimizer

    Returns:
    --------
    model : keras.Model
        Compiled model
    """
    # Define optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=0.9, beta_2=0.999
    )

    # Define losses: pinball_loss is @register_keras_serializable so it
    # round-trips correctly through .keras save/load without custom_objects.
    losses = {
        "volatility": "mse",
        "var": pinball_loss,  # tau defaults to 0.01 (99 % VaR)
    }

    # Loss weights (prioritize volatility prediction)
    loss_weights = {"volatility": 1.0, "var": 0.5}

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={"volatility": ["mae", "mse"], "var": ["mae"]},
    )

    return model


def create_callbacks(patience=15, lr_patience=5, min_delta=0.0001):
    """
    Create training callbacks for early stopping and learning rate reduction.

    Parameters:
    -----------
    patience : int
        Patience for early stopping
    lr_patience : int
        Patience for learning rate reduction
    min_delta : float
        Minimum change to qualify as improvement

    Returns:
    --------
    callbacks : list
        List of Keras callbacks
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=lr_patience,
        min_delta=min_delta,
        verbose=1,
    )

    return [early_stopping, reduce_lr]


def get_attention_weights(model, X):
    """
    Extract attention weights from trained model.

    Parameters:
    -----------
    model : keras.Model
        Trained LSTM-Attention model
    X : np.array
        Input sequences

    Returns:
    --------
    attention_weights : np.array
        Attention weights for each sequence
    """
    # Create intermediate model to extract attention weights
    attention_layer = model.get_layer("attention")

    # Get LSTM output (input to attention layer)
    lstm_output_model = Model(
        inputs=model.input, outputs=model.get_layer("lstm_2").output
    )

    lstm_outputs = lstm_output_model.predict(X, verbose=0)

    # Compute attention weights
    _, attention_weights = attention_layer(lstm_outputs)

    return attention_weights.numpy()


if __name__ == "__main__":
    print("Building LSTM-Attention model...")

    # Example: Build model
    input_shape = (30, 12)  # 30 time steps, 12 features
    model = build_lstm_attention_model(input_shape)
    model = compile_model(model)

    # Print model summary
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE")
    print("=" * 70)
    model.summary()

    print(f"\nTotal trainable parameters: {model.count_params():,}")
    print("\nModel built successfully!")

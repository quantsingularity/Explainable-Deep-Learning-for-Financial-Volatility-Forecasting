"""
Smoke Tests for Volatility Forecasting Pipeline
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import tensorflow as tf
from core.model import build_lstm_attention_model, compile_model
from core.utils import calculate_rmse, create_sequences
from data_processing.data_generator import generate_synthetic_dataset

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)


def test_data_generation():
    """Test synthetic data generation."""
    print("\n[TEST 1/5] Data Generation...")

    df = generate_synthetic_dataset(n_days=100, start_date="2020-01-01")

    assert len(df) > 0, "Data generation failed: empty dataframe"
    assert "returns" in df.columns, "Missing 'returns' column"
    assert "realized_volatility" in df.columns, "Missing 'realized_volatility' column"
    assert "gpr_index" in df.columns, "Missing 'gpr_index' column"

    print("  ✓ Data generation successful")
    print(f"  Generated {len(df)} samples with {len(df.columns)} features")

    return df


def test_sequence_creation():
    """Test sequence creation for LSTM."""
    print("\n[TEST 2/5] Sequence Creation...")

    # Create dummy data
    n_samples = 100
    n_features = 12
    lookback = 30

    data = np.random.randn(n_samples, n_features)
    target = np.random.randn(n_samples)

    X, y = create_sequences(data, target, lookback)

    assert (
        X.shape[0] == n_samples - lookback
    ), f"Wrong number of sequences: {X.shape[0]}"
    assert X.shape[1] == lookback, f"Wrong lookback: {X.shape[1]}"
    assert X.shape[2] == n_features, f"Wrong features: {X.shape[2]}"
    assert len(y) == n_samples - lookback, f"Wrong target length: {len(y)}"

    print("  ✓ Sequence creation successful")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")

    return X, y


def test_model_architecture():
    """Test model building and compilation."""
    print("\n[TEST 3/5] Model Architecture...")

    input_shape = (30, 12)  # (time_steps, features)

    model = build_lstm_attention_model(input_shape)
    model = compile_model(model)

    assert model is not None, "Model build failed"
    assert len(model.layers) > 0, "Model has no layers"
    assert model.count_params() > 0, "Model has no parameters"

    # Check output shapes
    dummy_input = np.random.randn(8, 30, 12)
    outputs = model.predict(dummy_input, verbose=0)

    assert len(outputs) == 2, f"Wrong number of outputs: {len(outputs)}"
    assert outputs[0].shape == (
        8,
        1,
    ), f"Wrong volatility output shape: {outputs[0].shape}"
    assert outputs[1].shape == (8, 1), f"Wrong VaR output shape: {outputs[1].shape}"

    print("  ✓ Model architecture valid")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Outputs: {[o.shape for o in outputs]}")

    return model


def test_training_step():
    """Test single training step."""
    print("\n[TEST 4/5] Training Step...")

    # Create small dataset
    X_train = np.random.randn(64, 30, 12)
    y_train = np.random.randn(64, 1)

    # Build model
    model = build_lstm_attention_model((30, 12))
    model = compile_model(model)

    # Train for 1 epoch
    history = model.fit(
        X_train,
        {"volatility": y_train, "var": y_train},
        epochs=1,
        batch_size=32,
        verbose=0,
    )

    assert "loss" in history.history, "Training failed: no loss recorded"
    assert len(history.history["loss"]) == 1, "Wrong number of epochs"

    print("  ✓ Training step successful")
    print(f"  Final loss: {history.history['loss'][0]:.4f}")

    return model


def test_evaluation_metrics():
    """Test evaluation metrics calculation."""
    print("\n[TEST 5/5] Evaluation Metrics...")

    y_true = np.array([0.01, 0.02, 0.015, 0.025, 0.018])
    y_pred = np.array([0.012, 0.019, 0.016, 0.024, 0.017])

    rmse = calculate_rmse(y_true, y_pred)

    assert rmse > 0, "RMSE calculation failed"
    assert rmse < 1, f"RMSE suspiciously large: {rmse}"

    print("  ✓ Evaluation metrics working")
    print(f"  RMSE: {rmse:.4f}")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 70)
    print("SMOKE TESTS FOR LSTM-ATTENTION-SHAP PIPELINE")
    print("=" * 70)

    try:
        test_data_generation()
        test_sequence_creation()
        test_model_architecture()
        test_training_step()
        test_evaluation_metrics()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)

        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False

    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

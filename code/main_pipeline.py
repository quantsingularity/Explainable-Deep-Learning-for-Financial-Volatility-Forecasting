"""
Main Pipeline: Complete Workflow for Model Training, Evaluation, and Explanation
"""

import os

import numpy as np
import tensorflow as tf

# Set seeds
np.random.seed(123)
tf.random.set_seed(456)

from core import console as ui
from core.model import get_attention_weights
from core.utils import load_and_prepare_data, train_val_test_split

# Import custom modules
from data_processing.data_generator import generate_synthetic_dataset
from evaluation.eval import (
    compare_with_baselines,
    evaluate_var_backtest,
    evaluate_volatility_forecast,
    generate_results_table,
    plot_var_backtest,
)
from explainability.explain import run_full_explainability_pipeline
from training.train import plot_training_history, train_model


def main():
    """Execute complete pipeline."""

    ui.banner(
        "LSTM-Attention-SHAP Pipeline",
        "Explainable Deep Learning for Financial Volatility Forecasting",
        {"stages": "data, training, evaluation, explainability, summary"},
    )

    # ==================================================================
    # STEP 1: DATA PREPARATION
    # ==================================================================
    ui.section("1/5", "DATA PREPARATION")

    data_path = "./data/synthetic_data.csv"

    # Check if data exists, otherwise generate
    if not os.path.exists(data_path):
        print("Data not found. Generating synthetic dataset...")
        df = generate_synthetic_dataset(n_days=1827, start_date="2018-01-01")
        df.to_csv(data_path, index=False)

    # Load and split data
    df, feature_cols = load_and_prepare_data(data_path)
    data_dict = train_val_test_split(
        df,
        feature_cols,
        target_col="realized_volatility",
        train_end="2022-12-31",
        val_end="2023-06-30",
    )

    print(f"\nFeatures used ({len(feature_cols)}): {feature_cols[:5]}...")

    # ==================================================================
    # STEP 2: MODEL TRAINING
    # ==================================================================
    ui.section("2/5", "MODEL TRAINING")

    model_path = "../models/lstm_attention_model.keras"

    # Check if model exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        from core.model import AttentionLayer, pinball_loss

        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "AttentionLayer": AttentionLayer,
                "pinball_loss": pinball_loss,
            },
        )
    else:
        print("Training new model...")
        model, history = train_model(
            data_dict, epochs=100, batch_size=64, save_path="../models"
        )
        plot_training_history(history, save_path="../docs/figures")

    # ==================================================================
    # STEP 3: MODEL EVALUATION
    # ==================================================================
    ui.section("3/5", "MODEL EVALUATION")

    # Volatility forecasting evaluation
    results, predictions_dict = evaluate_volatility_forecast(
        model, data_dict, data_dict["scalers"]["target"]
    )

    # VaR backtesting
    var_results = evaluate_var_backtest(predictions_dict, alpha=0.01)

    # Compare with baselines
    comparison_df = compare_with_baselines(predictions_dict)

    # Generate plots
    plot_var_backtest(var_results, predictions_dict, save_path="../docs/figures")

    # Save tables
    generate_results_table(comparison_df, var_results, save_path="../docs/tables")

    # ==================================================================
    # STEP 4: EXPLAINABILITY ANALYSIS
    # ==================================================================
    ui.section("4/5", "EXPLAINABILITY ANALYSIS")

    # Extract attention weights
    print("Extracting attention weights...")
    attention_weights = get_attention_weights(model, data_dict["test"]["X"])

    # Run SHAP analysis
    explain_results = run_full_explainability_pipeline(
        model,
        data_dict["train"]["X"],
        data_dict["test"]["X"],
        data_dict["feature_names"],
        attention_weights=attention_weights,
    )

    # ==================================================================
    # STEP 5: SUMMARY
    # ==================================================================
    ui.section("5/5", "PIPELINE SUMMARY")

    ui.step_done("data prepared and split chronologically")
    ui.step_done(f"model trained (RMSE: {results['RMSE']:.4f})")
    ui.step_done(
        f"VaR backtest complete (violation rate: "
        f"{var_results['violation_rate'] * 100:.2f}%)"
    )
    ui.step_done("SHAP analysis complete")
    ui.step_done(
        f"top SHAP driver: " f"{explain_results['importance_df'].iloc[0]['Feature']}"
    )

    ui.summary_panel(
        "PIPELINE COMPLETE",
        {
            "Test RMSE": f"{results['RMSE']:.4f}",
            "VaR violation rate": f"{var_results['violation_rate'] * 100:.2f}%",
            "Top SHAP driver": explain_results["importance_df"].iloc[0]["Feature"],
            "Model": "../models/lstm_attention_model.keras",
            "Figures": "../docs/figures/*.png",
            "Tables": "../docs/tables/*.csv",
        },
    )

    return {
        "model": model,
        "data": data_dict,
        "results": results,
        "var_results": var_results,
        "explain_results": explain_results,
    }


if __name__ == "__main__":
    pipeline_results = main()
    print("\nAll tasks completed. Ready for paper generation.")

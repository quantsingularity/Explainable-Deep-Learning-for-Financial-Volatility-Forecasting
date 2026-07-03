#!/usr/bin/env bash
# =============================================================================
# Run script for Explainable Deep Learning for Financial Volatility
# Forecasting (LSTM-Attention-SHAP)
#
# Usage:
#   bash scripts/run.sh <command> [options]
#
# Commands:
#   test        Run the unit and regression test suite
#   data        Generate the synthetic dataset (code/data/synthetic_data.csv)
#   train       Run the complete pipeline (train, evaluate, VaR, SHAP)
#   baselines   Fit GARCH / EGARCH / HAR-RV baseline comparison
#   backtest    Run the volatility trading strategy backtests
#   figures     Regenerate publication figures
#   api         Launch the FastAPI inference server
#   demo        Quick smoke run: 2-epoch training + SHAP (~3 min, CPU)
#   all         test -> train
#
# The full multi-step pipeline with progress banners is also available as
# scripts/run_all.sh (run from the project root).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CODE_DIR="$ROOT_DIR/code"

export PYTHONPATH="$CODE_DIR:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

PYTHON="${PYTHON:-python3}"

usage() {
    awk 'NR>1 && /^# ={10,}/{c++; if(c==2) exit} NR>1{sub(/^# ?/,""); print}' "$0"
}

cmd="${1:-}"
shift || true

case "$cmd" in
    test)
        echo "[run.sh] Running test suite..."
        cd "$CODE_DIR"
        "$PYTHON" -m pytest tests/ -v "$@"
        ;;

    data)
        echo "[run.sh] Generating synthetic dataset..."
        cd "$CODE_DIR"
        "$PYTHON" -m data_processing.data_generator "$@"
        ;;

    train)
        echo "[run.sh] Running complete pipeline..."
        mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/docs/figures" "$ROOT_DIR/docs/tables"
        cd "$CODE_DIR"
        "$PYTHON" main_pipeline.py "$@"
        ;;

    baselines)
        echo "[run.sh] Fitting baseline models..."
        cd "$CODE_DIR"
        "$PYTHON" -m evaluation.baseline_models "$@"
        ;;

    backtest)
        echo "[run.sh] Running trading strategy backtests..."
        cd "$CODE_DIR"
        "$PYTHON" -m evaluation.trading_backtest "$@"
        ;;

    figures)
        echo "[run.sh] Generating paper figures..."
        mkdir -p "$ROOT_DIR/docs/figures"
        cd "$CODE_DIR"
        "$PYTHON" -m visualization.generate_paper_figures "$@"
        ;;

    api)
        echo "[run.sh] Launching FastAPI inference server..."
        cd "$CODE_DIR"
        "$PYTHON" -m uvicorn serving.api_server:app --host 0.0.0.0 --port 8000 "$@"
        ;;

    demo)
        echo "[run.sh] Demo: 2-epoch training + evaluation + SHAP..."
        mkdir -p "$ROOT_DIR/models"
        cd "$CODE_DIR"
        "$PYTHON" - <<'PY'
import os
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
np.random.seed(123)
tf.random.set_seed(456)
os.makedirs("../models", exist_ok=True)

from core.utils import load_and_prepare_data, train_val_test_split
from data_processing.data_generator import generate_synthetic_dataset
from training.train import train_model
from evaluation.eval import evaluate_volatility_forecast, evaluate_var_backtest
from core.model import get_attention_weights
from explainability.explain import run_full_explainability_pipeline

data_path = "./data/synthetic_data.csv"
if not os.path.exists(data_path):
    df = generate_synthetic_dataset(n_days=1827, start_date="2018-01-01")
    df.to_csv(data_path, index=False)

df, feature_cols = load_and_prepare_data(data_path)
data = train_val_test_split(
    df, feature_cols, target_col="realized_volatility",
    train_end="2022-12-31", val_end="2023-06-30",
)

model, _ = train_model(data, epochs=2, batch_size=64, save_path="../models")

results, preds = evaluate_volatility_forecast(
    model, data, data["scalers"]["target"]
)
var_results = evaluate_var_backtest(preds, alpha=0.01)

att = get_attention_weights(model, data["test"]["X"])
ex = run_full_explainability_pipeline(
    model, data["train"]["X"], data["test"]["X"],
    data["feature_names"], attention_weights=att,
)

from core import console as ui

ui.summary_panel(
    "DEMO COMPLETE",
    {
        "RMSE": f"{results['RMSE']:.4f}",
        "VaR violation rate": f"{var_results['violation_rate']*100:.2f}%",
        "Top SHAP driver": ex["importance_df"].iloc[0]["Feature"],
    },
    footer="2-epoch smoke run; use run.sh train for the full pipeline",
)
PY
        ;;

    all)
        bash "$0" test
        bash "$0" train
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "Unknown command: $cmd" >&2
        usage
        exit 1
        ;;
esac

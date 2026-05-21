# Explainable Deep Learning for Financial Volatility Forecasting

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18667024.svg)](https://doi.org/10.5281/zenodo.18667024)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](infrastructure/requirements.txt)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red.svg)](infrastructure/requirements.txt)
[![MLOps](https://img.shields.io/badge/MLOps-MLflow%2FDocker-blue.svg)](docker-compose.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade and production-ready framework for financial volatility forecasting using a hybrid
**LSTM-Attention architecture** jointly optimised for volatility prediction and Value-at-Risk
estimation. Integrates **SHAP explainability** for regulatory transparency and supports
multi-horizon forecasting across equities, commodities, and currencies.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Results](#results)
- [Explainability](#explainability)
- [MLOps and Production](#mlops-and-production)
- [License](#license)

---

## Project Structure

```
.
├── .github/
│   ├── readme/                       # GitHub-rendered README assets
│   └── workflows/
│       └── cicd.yml                  # CI formatting checks (black, ruff, prettier)
├── code/
│   ├── core/                         # Model architecture and shared utilities
│   │   ├── __init__.py
│   │   ├── model.py                  # LSTM-Attention-SHAP model definition
│   │   └── utils.py                  # Metrics, VaR tests, data helpers
│   ├── data_processing/              # Data ingestion
│   │   ├── __init__.py
│   │   ├── data_generator.py         # Synthetic dataset generation
│   │   └── download_real_data.py     # yfinance / pandas-datareader fetcher
│   ├── training/                     # Model training
│   │   ├── __init__.py
│   │   ├── train.py                  # Single-horizon training loop
│   │   ├── train_multi_horizon.py    # Multi-horizon (1d / 5d / 22d) training
│   │   └── training_optimization.py  # Mixed precision, pruning, distillation
│   ├── evaluation/                   # Metrics, ablation, baselines, backtest
│   │   ├── __init__.py
│   │   ├── eval.py                   # Evaluation metrics and VaR backtesting
│   │   ├── ablation_study.py         # Component ablation experiments
│   │   ├── baseline_models.py        # GARCH / HAR-RV / XGBoost baselines
│   │   └── trading_backtest.py       # Vol-arb and trend-following strategies
│   ├── explainability/               # XAI
│   │   ├── __init__.py
│   │   └── explain.py                # SHAP + attention explainability
│   ├── serving/                      # API and real-time inference
│   │   ├── __init__.py
│   │   ├── api_server.py             # FastAPI inference server
│   │   └── streaming_inference.py    # Real-time streaming pipeline
│   ├── visualization/                # Research outputs
│   │   ├── __init__.py
│   │   ├── generate_paper_figures.py # Publication-quality figure generation
│   │   └── visualize_architecture.py # Model architecture diagram
│   ├── data/                         # Input datasets
│   │   └── synthetic_data.csv
│   ├── tests/                        # Test suite
│   │   ├── __init__.py
│   │   ├── test_smoke.py             # Core functionality smoke tests
│   │   └── test_integration.py       # End-to-end integration tests
│   ├── main_pipeline.py              # Orchestration entry point
│   └── requirements.txt              # All dependencies
├── docs/
│   ├── USER_GUIDE.md                 # Detailed usage and configuration guide
│   ├── figures/                      # Pre-generated paper figures (PNG)
│   └── tables/                       # Pre-generated result tables (CSV)
├── scripts/
│   ├── run_all.sh                    # One-command full pipeline execution
│   ├── setup.sh                      # Production environment setup
│   └── lint.sh                       # Ruff / flake8 / mypy / pylint runner
├── Dockerfile                        # Multi-stage image (base / training / api / dev)
├── docker-compose.yml                # Full stack: MLflow, PostgreSQL, Redis, Grafana
├── .gitignore
├── LICENSE
└── README.md
```

---

## Overview

| Category             | Feature               | Description                                                                         | Key Metric                    |
| :------------------- | :-------------------- | :---------------------------------------------------------------------------------- | :---------------------------- |
| **Core Model**       | LSTM-Attention Hybrid | LSTM for long-term dependencies + Bahdanau Attention for dynamic temporal weighting | +**17.3%** RMSE vs LSTM-only  |
| **Interpretability** | SHAP-based XAI        | Global and local feature attribution for regulatory compliance                      | Basel III / SR 11-7 aligned   |
| **Optimisation**     | Multi-Objective Loss  | Joint MSE for point forecasts + Pinball Loss for VaR estimation                     | VaR violation rate: **1.05%** |
| **MLOps**            | MLflow Integration    | Experiment tracking, model versioning, production registry                          | Full experiment lineage       |
| **Forecasting**      | Multi-Horizon         | Simultaneous 1-day, 5-day, and 22-day volatility forecasts                          | 3x forecast coverage          |
| **Validation**       | Backtest Engine       | Vol Arbitrage and Trend Following with realistic cost and slippage                  | Annual return: **18.5%**      |

---

## Architecture

### LSTM-Attention-SHAP Model

| Component               | Responsibility                                        | Configuration                                           |
| :---------------------- | :---------------------------------------------------- | :------------------------------------------------------ |
| **Input Layer**         | 30-day lookback window of 15-dimensional features     | `lookback_window = 30`, `primary_ticker = SPY`          |
| **LSTM Layers**         | Sequential patterns and long-term memory              | `lstm_units = 128, 64`, `recurrent_dropout = 0.1`       |
| **Attention Mechanism** | Dynamic weighting of each day in the lookback window  | `attention_units = 64`                                  |
| **Output Heads**        | Dual-head for Volatility (MSE) and VaR (Pinball Loss) | `volatility_activation = linear`, `var_quantile = 0.01` |

---

## Quick Start

### Installation

```bash
git clone https://github.com/quantsingularity/Explainable-Deep-Learning-for-Financial-Volatility-Forecasting
cd Explainable-Deep-Learning-for-Financial-Volatility-Forecasting

python3 -m venv venv && source venv/bin/activate
pip install -r code/requirements.txt
```

### Automated Research Pipeline

```bash
# Runs: data download → training → evaluation → SHAP analysis → figure generation
bash scripts/run_all.sh

ls docs/figures/   # model_architecture.png, shap_importance_bar.png, ...
ls docs/tables/    # table1_model_comparison.csv, table2_var_backtesting.csv
```

### Docker Deployment

| Command                                    | Profile      | Purpose                      | URL                          |
| :----------------------------------------- | :----------- | :--------------------------- | :--------------------------- |
| `docker-compose up -d postgres mlflow`     | default      | MLflow tracking + PostgreSQL | `http://localhost:5000`      |
| `docker-compose --profile training-gpu up` | training-gpu | GPU-accelerated training     | -                            |
| `docker-compose --profile api up`          | api          | FastAPI inference server     | `http://localhost:8000/docs` |
| `docker-compose --profile monitoring up`   | monitoring   | Prometheus + Grafana         | `http://localhost:3000`      |

---

## Configuration

All parameters are managed via `config.ini`.

### Model Hyperparameters

| Parameter                | Section      | Value   | Description                       |
| :----------------------- | :----------- | :------ | :-------------------------------- |
| `lookback_window`        | `[DATA]`     | 30      | Historical days per forecast      |
| `lstm_units`             | `[MODEL]`    | 128, 64 | Two-layer LSTM dimensions         |
| `attention_units`        | `[MODEL]`    | 64      | Attention mechanism dimension     |
| `var_quantile`           | `[MODEL]`    | 0.01    | 99% VaR estimation quantile       |
| `volatility_loss_weight` | `[TRAINING]` | 0.7     | MSE weight in joint loss          |
| `var_loss_weight`        | `[TRAINING]` | 0.3     | Pinball Loss weight in joint loss |

### Feature Engineering

The model uses a 15-dimensional feature vector across four categories.

| Category   | Features                           | Source                       |
| :--------- | :--------------------------------- | :--------------------------- |
| Price      | Returns, High-Low Spread           | `code/data_generator.py`     |
| Volume     | Normalised Volume                  | `code/data_generator.py`     |
| Volatility | Realised Volatility (RV), VIX      | `code/data_generator.py`     |
| Lagged     | RV Lag 1, RV Lag 5, Returns Lag 22 | `config.ini` (`lag_periods`) |

---

## Results

### Benchmark Comparison (Test Period: 2023-2024)

| Metric               | GARCH(1,1) | HAR-RV | XGBoost | LSTM-Attention-SHAP |
| :------------------- | :--------- | :----- | :------ | :------------------ |
| RMSE (x10-2)         | 2.10       | 1.92   | 1.85    | **1.50**            |
| MAE (x10-2)          | 1.60       | 1.45   | 1.38    | **1.12**            |
| R2 Score             | 0.45       | 0.54   | 0.58    | **0.72**            |
| Improvement vs GARCH | baseline   | +8.6%  | +11.9%  | **+28.6%**          |

### VaR Backtesting

| Test                        | Statistic | P-Value | Result                        |
| :-------------------------- | :-------- | :------ | :---------------------------- |
| VaR Violation Rate          | **1.05%** | -       | Target 1.00% acceptable       |
| Kupiec POF Test             | 0.12      | 0.72    | Pass - correct coverage       |
| Christoffersen Independence | 0.88      | 0.45    | Pass - independent violations |

### Trading Strategy Validation

| Strategy                 | Annual Return | Sharpe   | Max Drawdown |
| :----------------------- | :------------ | :------- | :----------- |
| **Volatility Arbitrage** | **18.5%**     | **1.42** | -12.3%       |
| Trend Following          | 14.2%         | 1.18     | -15.8%       |
| Mean Reversion           | 12.8%         | 1.35     | -10.5%       |
| Buy and Hold (Benchmark) | 8.5%          | 0.65     | -22.1%       |

---

## Explainability

The SHAP module provides dual-layer interpretability for regulatory and risk reporting.

| Layer        | Method            | Output                                        | Purpose                            |
| :----------- | :---------------- | :-------------------------------------------- | :--------------------------------- |
| **Temporal** | Attention Weights | Heatmap of 30-day window importance           | Explains when the model is looking |
| **Feature**  | SHAP Values       | Bar and Beeswarm plots of feature attribution | Explains what drives each forecast |

### SHAP Feature Importance

| Feature                   | Mean Absolute SHAP | Interpretation                                    |
| :------------------------ | :----------------- | :------------------------------------------------ |
| Realised Volatility Lag 1 | 0.45               | Yesterday's volatility is the strongest predictor |
| VIX Index                 | 0.32               | Market-wide fear and risk sentiment               |
| RV Lag 22 (1-Month)       | 0.18               | Long-term volatility clustering                   |
| High-Low Spread           | 0.11               | Intra-day price movement indicator                |

---

## MLOps and Production

### Training Optimisation

| Technique              | Purpose                                   | Impact                                     |
| :--------------------- | :---------------------------------------- | :----------------------------------------- |
| Mixed Precision (FP16) | Reduces memory, accelerates GPU training  | 2-3x speedup                               |
| Model Pruning          | Removes unnecessary weights               | 50% smaller, less than 5% performance loss |
| Knowledge Distillation | Smaller student model from larger teacher | 90%+ retention at 50% size                 |

### Infrastructure Setup

```bash
# Full production environment setup (interactive)
bash scripts/setup.sh

# Lint the codebase
bash scripts/lint.sh
```

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for detailed configuration, API reference, and
deployment instructions.

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

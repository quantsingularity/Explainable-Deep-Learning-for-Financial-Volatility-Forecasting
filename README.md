# Explainable Deep Learning for Financial Volatility Forecasting

## 🎯 Project Overview

This repository presents a **fully implemented, research-grade deep learning framework** for financial volatility forecasting, as detailed in the paper _"Explainable Deep Learning for Financial Volatility Forecasting: An LSTM-Attention-SHAP Framework with Comprehensive Validation"_. The system integrates a hybrid **LSTM-Attention architecture** with **SHAP-based interpretability** to provide high-accuracy forecasts alongside regulatory-grade transparency.

### Key Features

| Feature                                  | Description                                                                                                                     |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **Hybrid LSTM-Attention Architecture**   | Combines long short-term memory networks with Bahdanau attention to capture complex temporal dependencies in market volatility. |
| **Dual-Layer Interpretability**          | Integrates attention mechanisms for temporal relevance and SHAP (SHapley Additive exPlanations) for feature-level attribution.  |
| **Multi-Objective Optimization**         | Jointly optimizes for volatility forecasting (MSE) and Value-at-Risk (VaR) estimation (Pinball Loss).                           |
| **Comprehensive Statistical Validation** | Includes Diebold-Mariano, Kupiec POF, and Christoffersen Independence tests for rigorous performance benchmarking.              |
| **Multi-Asset Support**                  | Pre-configured for Equities (SPY, QQQ), Commodities (GLD, USO), and Currencies (UUP) via Yahoo Finance integration.             |
| **Automated Research Pipeline**          | End-to-end script for data generation, model training, evaluation, and publication-quality figure generation.                   |

## 📊 Key Results (Benchmark Performance)

The LSTM-Attention-SHAP framework significantly outperforms traditional econometric models and standard machine learning baselines.

| Metric                   | GARCH(1,1) | HAR-RV | XGBoost | **LSTM-Attention-SHAP** |
| :----------------------- | :--------- | :----- | :------ | :---------------------- |
| **RMSE (×10⁻²)**         | 2.10       | 1.92   | 1.85    | **1.50**                |
| **MAE (×10⁻²)**          | 1.60       | 1.45   | 1.38    | **1.12**                |
| **R² Score**             | 0.45       | 0.54   | 0.58    | **0.72**                |
| **VaR Violation Rate**   | 1.80%      | 1.50%  | 1.35%   | **1.05%**               |
| **Improvement vs GARCH** | Baseline   | +8.6%  | +11.9%  | **+28.6%**              |

## 🚀 Quick Start (30 minutes)

The project is designed for rapid deployment and reproducibility.

### 1. Installation

```bash
# Clone repository
git clone https://github.com/quantsingularity/Explainable-Deep-Learning-for-Financial-Volatility-Forecasting
cd Explainable-Deep-Learning-for-Financial-Volatility-Forecasting

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

The automated execution script handles the entire research workflow.

```bash
# Run end-to-end pipeline (Data -> Train -> Eval -> Explain)
bash run_all.sh

# View generated figures and tables
ls figures/
ls tables/
```

## 📁 Repository Structure

The repository is structured to separate core implementation, data artifacts, and research outputs.

```
Explainable-Deep-Learning-for-Financial-Volatility-Forecasting/
├── README.md                          # This file
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── run_all.sh                         # Automated execution script
│
├── code/                              # Main implementation
│   ├── main_pipeline.py               # Complete end-to-end workflow
│   ├── model.py                       # LSTM-Attention architecture
│   ├── train.py                       # Training procedures
│   ├── eval.py                        # Evaluation metrics & backtesting
│   ├── explain.py                     # SHAP explainability analysis
│   ├── data_generator.py              # Synthetic data generation
│   ├── baseline_models.py             # GARCH, EGARCH, HAR-RV baselines
│   └── generate_paper_figures.py      # Publication-quality figure generation
│
├── data/                              # Data artifacts
│   └── synthetic_data.csv             # Generated synthetic dataset
│
├── figures/                           # Publication-ready visualizations
│   ├── model_architecture.png         # System design
│   ├── shap_importance_bar.png        # Feature attribution
│   └── var_backtesting_plot.png       # Risk validation
│
└── tables/                            # Experimental outputs (CSV)
    ├── table1_model_comparison.csv    # Performance benchmarks
    └── table2_var_backtesting.csv     # Statistical test results
```

## 🏗️ Architecture

The system employs a sophisticated deep learning architecture designed for both predictive power and transparency.

### Model Hierarchy & Responsibilities

| Component               | Responsibility                                                                                            | Implementation Location  |
| :---------------------- | :-------------------------------------------------------------------------------------------------------- | :----------------------- |
| **Data Generator**      | Creates statistically realistic synthetic market data with jumps and volatility clustering.               | `code/data_generator.py` |
| **LSTM Layers**         | Captures long-range temporal dependencies in the 15-dimensional feature vector.                           | `code/model.py`          |
| **Attention Mechanism** | Learns to weight specific historical days based on their relevance to future volatility.                  | `code/model.py`          |
| **SHAP Explainer**      | Decomposes model predictions into individual feature contributions for global and local interpretability. | `code/explain.py`        |
| **Backtesting Engine**  | Validates Value-at-Risk (VaR) forecasts using Kupiec and Christoffersen statistical tests.                | `code/eval.py`           |

### Key Design Principles

| Principle                    | Explanation                                                                                                 |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------- |
| **Temporal Awareness**       | Uses a 30-day lookback window to capture short-term and long-term volatility dynamics.                      |
| **Multi-Objective Learning** | Jointly optimizes for point forecasts and tail risk, improving overall model robustness.                    |
| **Regulatory Compliance**    | Provides "why" behind every forecast, meeting the transparency requirements of modern financial regulation. |
| **Reproducibility**          | All randomness is controlled via fixed seeds (123/456) across NumPy and TensorFlow.                         |
| **Extensibility**            | Modular design allows for easy integration of new features (e.g., sentiment) or alternative architectures.  |

## 🧪 Evaluation Framework

The framework includes a rigorous evaluation suite to ensure the validity of the forecasting results.

### Forecasting Metrics

- **RMSE/MAE**: Standard error metrics for point forecast accuracy.
- **QLIKE**: A robust loss function specifically designed for volatility forecasting.
- **Diebold-Mariano**: Statistical test to confirm if model improvements are significant.

### Risk Validation

- **Kupiec POF Test**: Validates if the number of VaR violations matches the target level (e.g., 1%).
- **Christoffersen Test**: Ensures that VaR violations are independent and not clustered in time.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

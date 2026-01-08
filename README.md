# Explainable Deep Learning for Financial Volatility Forecasting

## An LSTM-Attention-SHAP Framework with Comprehensive Validation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.14](https://img.shields.io/badge/TensorFlow-2.14-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📚 Overview

This repository contains the complete implementation of the research paper **"Explainable Deep Learning for Financial Volatility Forecasting: An LSTM-Attention-SHAP Framework with Comprehensive Validation"** by Abrar Ahmed (December 31, 2025).

### Key Contributions

1. **Novel Architecture**: Hybrid LSTM-Attention model achieving 30% improvement in RMSE over GARCH(1,1) baseline
2. **Dual-Layer Interpretability**: Integration of attention mechanisms and SHAP for regulatory-grade explainability
3. **Comprehensive Validation**: Rigorous statistical testing including Diebold-Mariano, Kupiec, and Christoffersen tests
4. **Economic Insights**: Quantitative evidence that GPR (Geopolitical Risk) index is the dominant driver of extreme volatility

### Performance Highlights

| Model                   | RMSE (×10⁻²) | MAE (×10⁻²) | R²       | VaR Violation Rate |
| ----------------------- | ------------ | ----------- | -------- | ------------------ |
| GARCH(1,1)              | 2.10         | 1.60        | 0.45     | 1.80%              |
| HAR-RV                  | 1.92         | 1.45        | 0.54     | 1.50%              |
| XGBoost                 | 1.85         | 1.38        | 0.58     | 1.35%              |
| **LSTM-Attention-SHAP** | **1.50**     | **1.12**    | **0.72** | **1.05%**          |

---

## 🏗️ Project Structure

```
├── code/
│   ├── main_pipeline.py           # Complete end-to-end pipeline
│   ├── model.py                   # LSTM-Attention architecture
│   ├── train.py                   # Training procedures
│   ├── eval.py                    # Evaluation metrics & backtesting
│   ├── explain.py                 # SHAP explainability analysis
│   ├── utils.py                   # Utility functions
│   ├── data_generator.py          # Synthetic data generation
│   ├── download_real_data.py      # Real market data downloader
│   ├── baseline_models.py         # GARCH, EGARCH, HAR-RV implementations
│   ├── generate_paper_figures.py  # Publication-quality figure generation
│   └── visualize_architecture.py  # Model architecture visualization
│
├── data/                          # Data directory
│   ├── synthetic_data.csv         # Generated synthetic dataset
│   └── real_data_SPY.csv          # Real S&P 500 data (if downloaded)
│
├── models/                        # Saved models
│   ├── lstm_attention_model.h5    # Trained model weights
│   └── training_history.pkl       # Training history
│
├── figures/                       # Generated figures
│   ├── figure1_model_architecture.png
│   ├── figure2_training_loss.png
│   ├── figure3_forecast_comparison.png
│   ├── figure4_shap_importance.png
│   ├── figure5_shap_beeswarm.png
│   ├── figure6_attention_heatmap.png
│   ├── figure7_var_backtesting.png
│   └── figure8_model_comparison.png
│
├── tables/                        # Generated tables (CSV format)
│   ├── table1_model_comparison.csv
│   ├── table2_var_backtesting.csv
│   └── shap_feature_importance.csv
│
├── tests/                         # Unit tests
│   └── test_smoke.py              # Smoke tests for all modules
│
├── requirements.txt               # Python dependencies
├── run_all.sh                     # Automated execution script
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Installation

**Prerequisites:**

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

**Clone and Install:**

```bash
# Clone the repository
git clone https://github.com/abrar2030/Explainable-Deep-Learning-for-Financial-Volatility-Forecasting.git
cd Explainable-Deep-Learning-for-Financial-Volatility-Forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Complete Pipeline

**Option A: Automated Execution (Recommended)**

```bash
bash run_all.sh
```

This script will:

1. Generate/download data
2. Train the LSTM-Attention model
3. Evaluate on test set
4. Run SHAP explainability analysis
5. Generate all figures and tables

**Option B: Step-by-Step Execution**

```bash
# Navigate to code directory
cd code

# Step 1: Generate synthetic data (or download real data)
python data_generator.py

# Or download real market data:
# python download_real_data.py

# Step 2: Run complete pipeline
python main_pipeline.py

# Step 3: Generate all paper figures
python generate_paper_figures.py
```

### 3. Expected Runtime

- **Synthetic Data Generation**: ~2 minutes
- **Model Training**: ~15-30 minutes (CPU) / ~5 minutes (GPU)
- **Evaluation & SHAP Analysis**: ~10 minutes
- **Figure Generation**: ~1 minute

**Total**: ~30-45 minutes for complete pipeline

---

## 📊 Using Real Market Data

To use actual financial data instead of synthetic data:

```python
# In code/download_real_data.py
from download_real_data import build_complete_dataset

# Download S&P 500 data
df_spy = build_complete_dataset(
    ticker='SPY',
    start_date='2018-01-01',
    end_date='2024-12-31'
)

# Save for pipeline
df_spy.to_csv('../data/real_data_SPY.csv', index=False)
```

**Supported Assets:**

- **Equities**: SPY (S&P 500), QQQ (NASDAQ-100)
- **Commodities**: GLD (Gold), USO (Oil)
- **Currencies**: UUP (US Dollar Index)

Data is automatically fetched from Yahoo Finance with the `yfinance` library.

---

## 🧪 Model Architecture

### LSTM-Attention Network

```
Input (30 days × 15 features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Bahdanau Attention Mechanism (64 units)
    ↓
Dense Layer (32 units, ReLU) + Dropout (0.2)
    ↓
    ├─→ Volatility Output (Linear)
    └─→ VaR Output (Pinball Loss, τ=0.01)
```

**Total Parameters**: ~147,000

### Training Configuration

- **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
- **Loss Functions**:
  - Volatility: Mean Squared Error (MSE)
  - VaR: Pinball Loss (τ=0.01)
- **Loss Weights**: λ_vol=0.7, λ_VaR=0.3
- **Batch Size**: 64
- **Max Epochs**: 100
- **Early Stopping**: Patience=15 epochs
- **Learning Rate Decay**: ReduceLROnPlateau (factor=0.5, patience=5)

---

## 📈 Evaluation Metrics

### Volatility Forecasting

1. **RMSE** (Root Mean Squared Error): √(Σ(ŷᵢ - yᵢ)²/n)
2. **MAE** (Mean Absolute Error): Σ|ŷᵢ - yᵢ|/n
3. **QLIKE**: Σ(yᵢ/ŷᵢ - log(yᵢ/ŷᵢ) - 1)/n
4. **R²**: 1 - SS_res/SS_tot

### VaR Backtesting

1. **Kupiec POF Test**: Tests if violation rate equals expected rate
2. **Christoffersen Independence Test**: Tests for clustering of violations
3. **Violation Rate**: Actual % of VaR breaches vs. 1% target

### Statistical Validation

- **Diebold-Mariano Test**: Compares forecast accuracy between models
- **Model Confidence Set (MCS)**: Identifies best-performing model set

---

## 🔍 Explainability Analysis

### SHAP (SHapley Additive exPlanations)

SHAP provides feature-level interpretability by computing each feature's contribution to predictions:

```python
from explain import run_full_explainability_pipeline

# Run SHAP analysis
results = run_full_explainability_pipeline(
    model=trained_model,
    X_train=train_data,
    X_test=test_data,
    feature_names=feature_list
)

# Top features
print(results['importance_df'].head())
```

**Key Findings:**

1. **GPR Index** (Geopolitical Risk): Mean |SHAP| = 0.45
2. **VIX**: Mean |SHAP| = 0.38
3. **Realized Volatility (t-1)**: Mean |SHAP| = 0.32

### Attention Mechanism

Temporal interpretability showing which historical days influence predictions:

```python
from model import get_attention_weights

# Extract attention weights
attention_weights = get_attention_weights(model, X_test)

# Visualize
import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(attention_weights[:50].T, cmap='YlOrRd')
plt.xlabel('Sample')
plt.ylabel('Days Back')
plt.title('Attention Weights Heatmap')
plt.show()
```

---

## 🔬 Baseline Models

The repository includes implementations of traditional econometric models for comparison:

### 1. GARCH(1,1)

```python
from baseline_models import GARCHModel

garch = GARCHModel(p=1, q=1)
garch.fit(returns)
forecasts = garch.rolling_forecast(returns, train_size=1000, test_size=200)
```

### 2. EGARCH(1,1)

```python
from baseline_models import EGARCHModel

egarch = EGARCHModel(p=1, o=1, q=1)
egarch.fit(returns)
forecasts = egarch.rolling_forecast(returns, train_size=1000, test_size=200)
```

### 3. HAR-RV (Heterogeneous Autoregressive Realized Volatility)

```python
from baseline_models import HARRVModel

har = HARRVModel()
har.fit(realized_volatility)
forecast = har.forecast(rv_history[-22:])
```

### Comparison Workflow

```python
from baseline_models import compare_baseline_models

results, actual = compare_baseline_models(
    returns=returns_series,
    rv_actual=rv_series,
    train_ratio=0.7
)

# Results contains forecasts from all baseline models
print(results.keys())  # ['Historical', 'GARCH', 'EGARCH', 'HAR-RV']
```

---

## 📊 Generating Figures

All publication-quality figures can be regenerated:

```bash
cd code
python generate_paper_figures.py
```

This generates:

- **Figure 1**: Model architecture diagram
- **Figure 2**: Training/validation loss curves
- **Figure 3**: Forecast vs. actual volatility
- **Figure 4**: SHAP feature importance (bar plot)
- **Figure 5**: SHAP beeswarm plot
- **Figure 6**: Attention weights heatmap
- **Figure 7**: VaR backtesting results
- **Figure 8**: Model comparison (RMSE)

All figures are saved as high-resolution PNG (300 DPI) in `figures/` directory.

---

## 🧪 Testing

Run unit tests to verify installation:

```bash
cd tests
python test_smoke.py
```

Expected output:

```
test_data_generation ... OK
test_model_build ... OK
test_shap_analysis ... OK
test_baseline_models ... OK

----------------------------------------------------------------------
Ran 4 tests in 12.34s

OK
```

---

## 📚 Features

### 15-Dimensional Feature Vector

1. **Price-Based**:
   - Log returns
   - High-low spread

2. **Volume**:
   - Normalized trading volume (30-day z-score)

3. **Volatility Proxies**:
   - Realized volatility (30-day)
   - VIX (implied volatility)

4. **External Risk Factors**:
   - Geopolitical Risk (GPR) index

5. **Temporal Lags**:
   - RV lags: t-1, t-5, t-22
   - Return lags: t-1, t-5, t-22

All features are standardized using rolling z-scores (252-day window).

---

## 🔧 Customization

### Modify Model Architecture

Edit `code/model.py`:

```python
# Change LSTM units
model = build_lstm_attention_model(
    input_shape=(30, 15),
    lstm_units=[256, 128],      # Increase capacity
    attention_units=128,
    dense_units=64,
    dropout=0.3
)
```

### Adjust Training Parameters

Edit `code/main_pipeline.py`:

```python
model, history = train_model(
    data_dict,
    epochs=150,              # More epochs
    batch_size=32,           # Smaller batches
    save_path='../models'
)
```

### Use Different Assets

Edit `code/download_real_data.py`:

```python
# Download Bitcoin or other crypto
import yfinance as yf
btc = yf.download('BTC-USD', start='2018-01-01', end='2024-12-31')
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

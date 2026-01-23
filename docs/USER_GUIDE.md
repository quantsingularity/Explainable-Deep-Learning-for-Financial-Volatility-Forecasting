# Comprehensive User Guide

## Table of Contents

1. [Installation Guide](#installation-guide)
2. [Docker Deployment](#docker-deployment)
3. [Local Development](#local-development)
4. [Model Training](#model-training)
5. [API Usage](#api-usage)
6. [Trading Backtests](#trading-backtests)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Installation Guide

### System Requirements

#### Minimum Requirements

- CPU: 4 cores
- RAM: 8GB
- Storage: 20GB free space
- OS: Linux, macOS, or Windows with WSL2

#### Recommended Requirements

- CPU: 8+ cores or NVIDIA GPU (V100/A100)
- RAM: 16GB+
- Storage: 50GB SSD
- OS: Ubuntu 20.04+ or similar

### Quick Installation

```bash
# Clone repository
git clone <repository-url>
cd volatility-forecasting-enhanced

# Run automated setup
./setup.sh

# Follow the prompts to configure your environment
```

---

## Docker Deployment

### Starting Services

#### 1. Core Infrastructure (MLflow + PostgreSQL)

```bash
# Start tracking server
docker-compose up -d postgres mlflow

# Check status
docker-compose ps

# View logs
docker-compose logs -f mlflow
```

#### 2. Training Service

```bash
# CPU training
docker-compose --profile training-cpu up

# GPU training (requires NVIDIA Docker)
docker-compose --profile training-gpu up

# Background training
docker-compose --profile training-cpu up -d
```

#### 3. API Server

```bash
# Start inference API
docker-compose --profile api up -d

# Test endpoint
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

#### 4. Monitoring Stack

```bash
# Start Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Login: admin / admin
```

### Service Management

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Restart a specific service
docker-compose restart mlflow

# View resource usage
docker stats

# Clean up unused images
docker system prune -a
```

---

## Local Development

### Python Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-prod.txt
```

### Development Workflow

```bash
# 1. Generate/load data
python code/data_generator.py

# 2. Train baseline model
python code/train.py

# 3. Run ablation study
python code/ablation_study.py

# 4. Train multi-horizon model
python code/train_multi_horizon.py

# 5. Run trading backtest
python code/trading_backtest.py

# 6. Start API server locally
python code/api_server.py
```

### Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Or with Docker
docker-compose --profile development up
# Access at http://localhost:8888
```

---

## Model Training

### Single Horizon Training

```python
from code.train import train_model
from code.utils import load_and_prepare_data, train_val_test_split

# Load data
df, feature_cols = load_and_prepare_data("data/synthetic_data.csv")

# Split data
data_dict = train_val_test_split(
    df,
    feature_cols,
    target_col='realized_volatility',
    train_end='2022-12-31',
    val_end='2023-06-30'
)

# Train
model, history = train_model(
    data_dict,
    epochs=100,
    batch_size=64,
    save_path="./models"
)
```

### Multi-Horizon Training

```python
from code.train_multi_horizon import train_multi_horizon_model

# Train for multiple horizons
model_wrapper, history = train_multi_horizon_model(
    data_dict,
    horizons=[1, 5, 22],  # 1-day, 5-day, 22-day
    epochs=100,
    batch_size=64,
    use_mlflow=True
)
```

### With MLflow Tracking

```python
import mlflow

# Set experiment
mlflow.set_experiment("volatility_forecasting")

# Start run
with mlflow.start_run(run_name="experiment_1"):
    # Log parameters
    mlflow.log_param("epochs", 100)
    mlflow.log_param("batch_size", 64)

    # Train model
    model, history = train_model(data_dict)

    # Log metrics
    mlflow.log_metric("final_val_loss", history.history['val_loss'][-1])

    # Log model
    mlflow.keras.log_model(model, "model")
```

### Training Optimization

#### Mixed Precision Training

```python
from code.training_optimization import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(policy='mixed_float16')
model, history, time = trainer.train_with_mixed_precision(
    model, data_dict, epochs=100
)
print(f"Training time: {time:.2f}s")
```

#### Model Pruning

```python
from code.training_optimization import ModelPruner

pruner = ModelPruner(target_sparsity=0.5)
pruned_model = pruner.create_pruned_model(model)
final_model, history = pruner.train_pruned_model(
    pruned_model, data_dict
)
```

---

## API Usage

### Python Client

```python
import requests
import numpy as np

# API endpoint
API_URL = "http://localhost:8000"

# Prepare request
features = np.random.randn(30, 12).tolist()  # 30 timesteps, 12 features

request_data = {
    "features": features,
    "horizon": 1,
    "asset_id": "SPY",
    "return_var": True
}

# Make prediction
response = requests.post(f"{API_URL}/predict", json=request_data)
prediction = response.json()

print(f"Predicted volatility: {prediction['volatility_forecast']:.4f}")
print(f"VaR (99%): {prediction['var_forecast']:.4f}")
print(f"Inference time: {prediction['inference_time_ms']:.2f}ms")
```

### Batch Predictions

```python
batch_request = {
    "requests": [
        {"features": features1, "horizon": 1, "asset_id": "SPY"},
        {"features": features2, "horizon": 1, "asset_id": "QQQ"},
        {"features": features3, "horizon": 5, "asset_id": "AAPL"}
    ]
}

response = requests.post(f"{API_URL}/predict/batch", json=batch_request)
results = response.json()

for pred in results['predictions']:
    print(f"{pred['asset_id']}: {pred['volatility_forecast']:.4f}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.01, 0.02, ...]],
    "horizon": 1,
    "asset_id": "SPY"
  }'

# Model info
curl http://localhost:8000/model/info

# Metrics
curl http://localhost:8000/metrics
```

---

## Trading Backtests

### Volatility Arbitrage Strategy

```python
from code.trading_backtest import (
    VolatilityArbitrageStrategy,
    BacktestEngine,
    BacktestConfig
)

# Configure
config = BacktestConfig(
    initial_capital=1_000_000,
    transaction_cost_bps=5.0,
    slippage_bps=2.0,
    max_position_size=0.2
)

# Initialize strategy
strategy = VolatilityArbitrageStrategy(config, threshold=0.15)

# Run backtest
engine = BacktestEngine(config)
results = engine.run_backtest(
    strategy,
    returns,
    volatility_forecasts,
    realized_volatility,
    dates
)

# Print metrics
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
```

### Comparing Multiple Strategies

```python
from code.trading_backtest import compare_strategies

strategies = {
    'Vol Arb': VolatilityArbitrageStrategy(config),
    'Trend': TrendFollowingVolStrategy(config),
    'Mean Rev': MeanReversionVolStrategy(config)
}

results = {}
for name, strat in strategies.items():
    results[name] = engine.run_backtest(strat, returns, forecasts, rv, dates)

comparison_df = compare_strategies(results)
```

---

## Performance Optimization

### Benchmarking

```python
import time

# Baseline training
start = time.time()
model, history = train_model(data_dict, epochs=100)
baseline_time = time.time() - start

# Mixed precision training
trainer = MixedPrecisionTrainer()
start = time.time()
model_fp16, history = trainer.train_with_mixed_precision(model, data_dict)
fp16_time = time.time() - start

print(f"Speedup: {baseline_time / fp16_time:.2f}x")
```

### Inference Optimization

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save
with open('model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

# Benchmark inference
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='model_optimized.tflite')
interpreter.allocate_tensors()

# Measure latency
latencies = []
for _ in range(1000):
    start = time.time()
    interpreter.invoke()
    latencies.append((time.time() - start) * 1000)

print(f"P50 latency: {np.percentile(latencies, 50):.2f}ms")
print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
```

---

## Troubleshooting

### Common Issues

#### 1. Docker build fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

#### 2. GPU not detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Update docker-compose.yml to specify GPU
```

#### 3. MLflow connection error

```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres mlflow
```

#### 4. API server not responding

```bash
# Check if model is loaded
curl http://localhost:8000/health

# Check logs
docker-compose logs api-server

# Verify model path
ls -la models/lstm_attention_model.h5
```

#### 5. Out of memory during training

```python
# Reduce batch size
train_model(data_dict, batch_size=32)  # Instead of 64

# Enable memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Performance Issues

#### Slow training

```bash
# Check GPU utilization
nvidia-smi -l 1

# Enable mixed precision
trainer = MixedPrecisionTrainer()

# Use smaller model
lstm_units = [64, 32]  # Instead of [128, 64]
```

#### High API latency

```bash
# Check Redis connection
docker-compose ps redis

# Enable model caching
# Use batch predictions for multiple assets

# Monitor metrics
curl http://localhost:8000/metrics | grep latency
```

### Debugging

```python
# Enable TensorFlow debugging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model summary
model.summary()

# Validate input shapes
print(f"X_train shape: {X_train.shape}")
print(f"Expected: (samples, 30, 12)")
```

### Getting Help

1. **Check logs**: `docker-compose logs -f [service]`
2. **GitHub Issues**: Report bugs and request features
3. **Documentation**: Refer to README and code comments
4. **MLflow UI**: Check experiment tracking at http://localhost:5000

---

## Best Practices

### Model Training

- Always use MLflow for experiment tracking
- Save checkpoints during long training runs
- Monitor validation loss to prevent overfitting
- Use early stopping with patience=15

### Production Deployment

- Use Docker for consistent environments
- Enable monitoring (Prometheus + Grafana)
- Implement caching (Redis) for frequently accessed predictions
- Set up health checks and alerts

### Code Quality

- Run tests before deployment: `pytest tests/`
- Use type hints for better code documentation
- Follow PEP 8 style guide
- Document API changes

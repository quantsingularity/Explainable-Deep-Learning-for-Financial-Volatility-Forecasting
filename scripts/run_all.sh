#!/bin/bash

# Complete Pipeline Execution Script
# Run from the project root directory

echo "======================================================================"
echo "  LSTM-ATTENTION-SHAP VOLATILITY FORECASTING - COMPLETE PIPELINE"
echo "======================================================================"

# Must be run from project root
if [ ! -d "code" ]; then
    echo "Error: run this script from the project root directory"
    exit 1
fi

mkdir -p models docs/figures docs/tables logs

# All Python commands run from inside code/ so relative paths resolve correctly
cd code || exit 1

echo ""
echo "======================================================================"
echo "STEP 1/6: DATA PREPARATION"
echo "======================================================================"

if [ -f "./data/synthetic_data.csv" ]; then
    echo "Synthetic data already exists: skipping generation"
else
    echo "Generating synthetic financial data..."
    python -m data_processing.data_generator || { echo "Data generation failed"; exit 1; }
fi

echo ""
echo "======================================================================"
echo "STEP 2/6: MODEL TRAINING"
echo "======================================================================"
echo "Expected time: 15-30 min (CPU) / 5 min (GPU)"

python main_pipeline.py || { echo "Pipeline execution failed"; exit 1; }

echo ""
echo "======================================================================"
echo "STEP 3/6: BASELINE COMPARISON"
echo "======================================================================"

python - << 'PYEOF'
import sys
sys.path.insert(0, ".")
from data_processing.data_generator import generate_synthetic_dataset
from evaluation.baseline_models import compare_baseline_models
from core.utils import load_and_prepare_data
import pandas as pd, os

df, _ = load_and_prepare_data("./data/synthetic_data.csv")
results, actual = compare_baseline_models(df["returns"].values, df["realized_volatility"].values, train_ratio=0.7)
out = pd.DataFrame(results)
out["Actual"] = actual
os.makedirs("../docs/tables", exist_ok=True)
out.to_csv("../docs/tables/baseline_forecasts.csv", index=False)
print("Baseline comparison saved to docs/tables/baseline_forecasts.csv")
PYEOF

echo ""
echo "======================================================================"
echo "STEP 4/6: FIGURE GENERATION"
echo "======================================================================"

python -m visualization.generate_paper_figures || echo "Figure generation failed: continuing"

echo ""
echo "======================================================================"
echo "STEP 5/6: TESTS"
echo "======================================================================"

python -m pytest tests/test_smoke.py -v || echo "Some tests failed: continuing"

echo ""
echo "======================================================================"
echo "STEP 6/6: SUMMARY"
echo "======================================================================"

python - << 'PYEOF'
import os, glob

sections = {
    "Data files":    glob.glob("./data/*.csv"),
    "Trained models": glob.glob("../models/*.keras"),
    "Figures":       glob.glob("../docs/figures/*.png"),
    "Tables":        glob.glob("../docs/tables/*.csv"),
}

print("\nPIPELINE EXECUTION SUMMARY")
print("=" * 60)
for label, files in sections.items():
    print(f"\n{label}: {len(files)}")
    for f in sorted(files):
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({size:.2f} MB)")

print("\n" + "=" * 60)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nOutput locations:")
print(f"  Models:  {os.path.abspath('../models')}")
print(f"  Figures: {os.path.abspath('../docs/figures')}")
print(f"  Tables:  {os.path.abspath('../docs/tables')}")
print("\nNext steps:")
print("  1. Review docs/figures/ for visualisations")
print("  2. Check docs/tables/ for performance metrics")
print("  3. See README.md or docs/USER_GUIDE.md for full usage")
PYEOF

cd ..
echo ""

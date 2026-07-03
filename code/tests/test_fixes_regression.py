"""
Regression tests for fixes applied during the completeness review.

Covers:
1. Keras-3-compatible serialization round-trip (.keras format) for the
   custom AttentionLayer and pinball_loss (version-agnostic registration).
2. SHAP GradientExplainer on the multi-output model (volatility-head
   wrapper fix).
3. Backtest portfolio-value length convention and strategy-comparison
   plot alignment (off-by-one fix).
4. Offline behaviour of the data_processing package (lazy yfinance).
"""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from core.model import build_lstm_attention_model, compile_model, pinball_loss


class TestSerializationRoundTrip:
    """Custom layer and loss must round-trip through .keras save/load."""

    def test_save_load_without_custom_objects(self, tmp_path):
        model = build_lstm_attention_model((10, 4), lstm_units=[8, 4])
        model = compile_model(model)

        path = str(tmp_path / "roundtrip.keras")
        model.save(path)

        # The registration decorator should make custom_objects unnecessary,
        # but passing them must also work (as main_pipeline.py does).
        loaded = tf.keras.models.load_model(path, compile=False)

        X = np.random.randn(4, 10, 4).astype(np.float32)
        out_orig = model.predict(X, verbose=0)
        out_load = loaded.predict(X, verbose=0)

        o1 = out_orig[0] if isinstance(out_orig, list) else out_orig["volatility"]
        o2 = out_load[0] if isinstance(out_load, list) else out_load["volatility"]
        np.testing.assert_allclose(o1, o2, atol=1e-5)

    def test_pinball_loss_properties(self):
        y_true = tf.constant([[0.0], [0.0]])
        # For tau=0.01 (a low quantile such as 1% VaR), over-prediction is
        # penalised with weight (1 - tau) = 0.99 while under-prediction
        # costs only tau = 0.01, pushing predictions toward the lower tail.
        under = pinball_loss(y_true, tf.constant([[-1.0], [-1.0]]), tau=0.01)
        over = pinball_loss(y_true, tf.constant([[1.0], [1.0]]), tau=0.01)
        assert float(under) < float(over)
        assert abs(float(under) - 0.01) < 1e-6
        assert abs(float(over) - 0.99) < 1e-6
        # Perfect prediction: zero loss.
        zero = pinball_loss(y_true, y_true)
        assert abs(float(zero)) < 1e-8


class TestShapMultiOutput:
    """GradientExplainer must work on the two-headed model."""

    def test_shap_values_shape(self):
        from explainability.explain import compute_shap_values

        model = build_lstm_attention_model((8, 3), lstm_units=[6, 4])
        model = compile_model(model)

        background = np.random.randn(10, 8, 3).astype(np.float32)
        X_test = np.random.randn(12, 8, 3).astype(np.float32)
        feature_names = ["f1", "f2", "f3"]

        shap_values, explainer = compute_shap_values(
            model, X_test, background, feature_names
        )

        vol_shap = np.array(shap_values[0])
        assert vol_shap.shape == (12, 8, 3), f"got {vol_shap.shape}"
        assert np.isfinite(vol_shap).all()


class TestBacktestConventions:
    """Portfolio values must be initial capital plus one value per period."""

    def _run(self, n=100):
        from evaluation.trading_backtest import (
            BacktestConfig,
            BacktestEngine,
            VolatilityArbitrageStrategy,
        )

        rng = np.random.default_rng(0)
        cfg = BacktestConfig()
        engine = BacktestEngine(cfg)
        strategy = VolatilityArbitrageStrategy(cfg)
        returns = rng.normal(0, 0.01, n)
        fc = np.abs(rng.normal(0, 0.01, n)) + 0.01
        rv = np.abs(rng.normal(0, 0.01, n)) + 0.01
        dates = pd.date_range("2023-01-01", periods=n)
        return engine.run_backtest(strategy, returns, fc, rv, dates), n

    def test_portfolio_value_length(self):
        results, n = self._run()
        assert len(results["portfolio_values"]) == n + 1
        assert results["portfolio_values"][0] > 0  # initial capital

    def test_compare_strategies_plot_alignment(self, tmp_path):
        """The comparison plot must not raise a shape-mismatch error."""
        from evaluation.trading_backtest import compare_strategies

        results, _ = self._run()
        df = compare_strategies(
            {"VolArb": results}, save_path=str(tmp_path / "figures")
        )
        assert df is None or len(df) >= 1
        # A comparison CSV should have been written
        tables_dir = str(tmp_path / "tables")
        assert os.path.exists(os.path.join(tables_dir, "strategy_comparison.csv"))


class TestOfflineDataProcessing:
    """data_processing must import and fail clearly without yfinance."""

    def test_package_imports(self):
        import data_processing  # noqa: F401

    def test_clear_error_when_yfinance_missing(self, monkeypatch):
        import data_processing.download_real_data as drd

        monkeypatch.setattr(drd, "yf", None)
        with pytest.raises(ImportError, match="yfinance is required"):
            drd.download_price_data(["SPY"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

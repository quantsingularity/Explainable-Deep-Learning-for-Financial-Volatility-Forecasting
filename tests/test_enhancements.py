"""
Comprehensive Test Suite for Production Enhancements
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

# Import modules to test
from code.train_multi_horizon import MultiHorizonVolatilityModel
from code.trading_backtest import (
    VolatilityArbitrageStrategy,
    BacktestEngine,
    BacktestConfig,
)
from code.ablation_study import LSTMOnlyModel, LSTMAttentionModel
from code.streaming_inference import StreamingDataBuffer, MarketDataPoint


class TestMultiHorizonForecasting:
    """Tests for multi-horizon forecasting module"""

    def test_model_creation(self):
        """Test multi-horizon model creation"""
        input_shape = (30, 12)
        horizons = [1, 5, 22]

        model_wrapper = MultiHorizonVolatilityModel(
            input_shape=input_shape, horizons=horizons
        )

        assert model_wrapper.model is not None
        assert len(model_wrapper.horizons) == 3

        # Check output heads
        output_names = [layer.name for layer in model_wrapper.model.outputs]
        for horizon in horizons:
            assert f"volatility_h{horizon}" in output_names
            assert f"var_h{horizon}" in output_names

    def test_model_compilation(self):
        """Test model compilation with appropriate losses"""
        input_shape = (30, 12)
        model_wrapper = MultiHorizonVolatilityModel(input_shape)
        model = model_wrapper.compile_model()

        assert model.optimizer is not None
        assert len(model.loss) > 0

    def test_prediction_shape(self):
        """Test prediction output shapes"""
        input_shape = (30, 12)
        horizons = [1, 5, 22]

        model_wrapper = MultiHorizonVolatilityModel(
            input_shape=input_shape, horizons=horizons
        )
        model_wrapper.compile_model()

        # Test prediction
        test_input = np.random.randn(1, 30, 12).astype(np.float32)
        predictions = model_wrapper.model.predict(test_input, verbose=0)

        assert isinstance(predictions, dict)
        assert len(predictions) == 6  # 3 horizons x 2 outputs each


class TestTradingBacktest:
    """Tests for trading backtest module"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        n_samples = 252  # One year of daily data
        dates = pd.date_range("2023-01-01", periods=n_samples, freq="D")

        returns = np.random.randn(n_samples) * 0.02
        volatility_forecasts = np.abs(np.random.randn(n_samples) * 0.02)
        realized_volatility = np.abs(np.random.randn(n_samples) * 0.02)

        return dates, returns, volatility_forecasts, realized_volatility

    def test_backtest_config(self):
        """Test backtest configuration"""
        config = BacktestConfig(
            initial_capital=1_000_000, transaction_cost_bps=5.0, slippage_bps=2.0
        )

        assert config.initial_capital == 1_000_000
        assert config.transaction_cost_bps == 5.0

    def test_vol_arbitrage_strategy(self, sample_data):
        """Test volatility arbitrage strategy"""
        dates, returns, vol_forecasts, realized_vol = sample_data

        config = BacktestConfig()
        strategy = VolatilityArbitrageStrategy(config, threshold=0.15)

        signals = strategy.generate_signals(vol_forecasts, realized_vol)

        assert len(signals) == len(returns)
        assert np.all(signals >= -1.0) and np.all(signals <= 1.0)

    def test_backtest_execution(self, sample_data):
        """Test backtest execution"""
        dates, returns, vol_forecasts, realized_vol = sample_data

        config = BacktestConfig(initial_capital=1_000_000)
        strategy = VolatilityArbitrageStrategy(config)
        engine = BacktestEngine(config)

        results = engine.run_backtest(
            strategy, returns, vol_forecasts, realized_vol, dates
        )

        assert "metrics" in results
        assert "portfolio_values" in results
        assert len(results["portfolio_values"]) == len(returns) + 1

        # Check metrics
        metrics = results["metrics"]
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

    def test_transaction_costs(self):
        """Test transaction cost calculation"""
        config = BacktestConfig(transaction_cost_bps=5.0, slippage_bps=2.0)
        strategy = VolatilityArbitrageStrategy(config)

        trade_value = 100_000
        cost = strategy.apply_transaction_costs(trade_value)

        expected_cost = trade_value * 0.0007  # 7 bps total
        assert abs(cost - expected_cost) < 0.01


class TestAblationStudy:
    """Tests for ablation study module"""

    def test_lstm_only_model(self):
        """Test LSTM-only model creation"""
        input_shape = (30, 12)
        model = LSTMOnlyModel.build(input_shape)

        assert model is not None
        assert model.name == "LSTM_Only"

        # Check that attention layer is not present
        layer_names = [layer.name for layer in model.layers]
        assert not any("attention" in name for name in layer_names)

    def test_lstm_attention_model(self):
        """Test LSTM+Attention model creation"""
        input_shape = (30, 12)
        model = LSTMAttentionModel.build(input_shape)

        assert model is not None
        assert model.name == "LSTM_Attention"

        # Check that attention layer is present
        layer_names = [layer.name for layer in model.layers]
        assert any("attention" in name for name in layer_names)

    def test_model_parameter_comparison(self):
        """Test that model variants have different parameter counts"""
        input_shape = (30, 12)

        model_lstm = LSTMOnlyModel.build(input_shape)
        model_attention = LSTMAttentionModel.build(input_shape)

        params_lstm = model_lstm.count_params()
        params_attention = model_attention.count_params()

        # Attention model should have more parameters
        assert params_attention > params_lstm


class TestStreamingInference:
    """Tests for streaming inference module"""

    def test_buffer_creation(self):
        """Test streaming buffer creation"""
        buffer = StreamingDataBuffer(lookback_window=30, n_features=12)

        assert buffer.lookback_window == 30
        assert buffer.n_features == 12
        assert len(buffer.buffers) == 0

    def test_data_addition(self):
        """Test adding data to buffer"""
        buffer = StreamingDataBuffer(lookback_window=30, n_features=12)

        symbol = "SPY"
        features = np.random.randn(12)

        buffer.add_data_point(symbol, features)

        assert symbol in buffer.buffers
        assert len(buffer.buffers[symbol]) == 1

    def test_buffer_ready_state(self):
        """Test buffer ready state"""
        buffer = StreamingDataBuffer(lookback_window=5, n_features=12)

        symbol = "SPY"

        # Add data points one by one
        for i in range(4):
            features = np.random.randn(12)
            buffer.add_data_point(symbol, features)
            assert not buffer.is_ready(symbol)

        # Add final point
        features = np.random.randn(12)
        buffer.add_data_point(symbol, features)
        assert buffer.is_ready(symbol)

    def test_get_input_sequence(self):
        """Test getting input sequence"""
        buffer = StreamingDataBuffer(lookback_window=5, n_features=12)

        symbol = "SPY"

        # Fill buffer
        for i in range(5):
            features = np.random.randn(12)
            buffer.add_data_point(symbol, features)

        sequence = buffer.get_input_sequence(symbol)

        assert sequence is not None
        assert sequence.shape == (1, 5, 12)

    def test_market_data_point(self):
        """Test MarketDataPoint dataclass"""
        data_point = MarketDataPoint(
            timestamp=datetime.now().isoformat(),
            symbol="SPY",
            price=450.0,
            volume=1000000,
            high=452.0,
            low=448.0,
            open=449.0,
            close=451.0,
        )

        assert data_point.symbol == "SPY"
        assert data_point.price == 450.0


class TestIntegration:
    """Integration tests for end-to-end workflows"""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for integration testing"""
        n_samples = 1000
        n_features = 12
        lookback = 30

        # Generate synthetic features
        features = np.random.randn(n_samples, n_features)

        # Generate synthetic targets
        targets = np.abs(np.random.randn(n_samples)) * 0.02

        # Create sequences
        X = []
        y = []

        for i in range(lookback, n_samples):
            X.append(features[i - lookback : i])
            y.append(targets[i])

        X = np.array(X)
        y = np.array(y)

        # Split
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        data_dict = {
            "train": {"X": X[:train_size], "y": y[:train_size]},
            "val": {
                "X": X[train_size : train_size + val_size],
                "y": y[train_size : train_size + val_size],
            },
            "test": {"X": X[train_size + val_size :], "y": y[train_size + val_size :]},
        }

        return data_dict

    def test_end_to_end_training(self, sample_dataset):
        """Test complete training pipeline"""
        input_shape = (30, 12)

        # Create model
        model_wrapper = MultiHorizonVolatilityModel(
            input_shape=input_shape, horizons=[1, 5]
        )
        model = model_wrapper.compile_model()

        # Prepare data
        X_train = sample_dataset["train"]["X"]
        y_train = sample_dataset["train"]["y"]

        y_train_dict = {
            "volatility_h1": y_train,
            "var_h1": y_train,
            "volatility_h5": y_train,
            "var_h5": y_train,
        }

        # Train for a few epochs
        history = model.fit(X_train, y_train_dict, epochs=2, batch_size=32, verbose=0)

        assert len(history.history["loss"]) == 2

        # Make predictions
        X_test = sample_dataset["test"]["X"]
        predictions = model.predict(X_test, verbose=0)

        assert isinstance(predictions, dict)
        assert "volatility_h1" in predictions


def test_imports():
    """Test that all modules can be imported"""
    try:
        pass

        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

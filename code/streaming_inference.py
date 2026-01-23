"""
Real-Time Streaming Inference Pipeline
Processes real-time market data with <100ms latency target
"""

import asyncio
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque
import logging
import time
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketDataPoint:
    """Single market data point"""

    timestamp: str
    symbol: str
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float


@dataclass
class VolatilityPrediction:
    """Volatility prediction output"""

    timestamp: str
    symbol: str
    predicted_volatility: float
    var_99: Optional[float]
    confidence_lower: float
    confidence_upper: float
    inference_time_ms: float
    model_version: str


class StreamingDataBuffer:
    """
    Maintains rolling window of market data for real-time inference
    """

    def __init__(
        self, lookback_window: int = 30, n_features: int = 12, max_symbols: int = 100
    ):
        """
        Initialize streaming buffer

        Parameters:
        -----------
        lookback_window : int
            Number of time steps to maintain
        n_features : int
            Number of features per time step
        max_symbols : int
            Maximum number of symbols to track
        """
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.max_symbols = max_symbols

        # Buffer for each symbol: deque of feature vectors
        self.buffers: Dict[str, deque] = {}

        # Feature statistics for normalization
        self.feature_stats: Dict[str, Dict] = {}

        logger.info(
            f"StreamingDataBuffer initialized (window={lookback_window}, features={n_features})"
        )

    def add_data_point(self, symbol: str, features: np.ndarray):
        """
        Add new data point to buffer

        Parameters:
        -----------
        symbol : str
            Asset symbol
        features : np.ndarray
            Feature vector (n_features,)
        """
        if symbol not in self.buffers:
            if len(self.buffers) >= self.max_symbols:
                logger.warning(
                    f"Max symbols ({self.max_symbols}) reached. Ignoring {symbol}"
                )
                return

            self.buffers[symbol] = deque(maxlen=self.lookback_window)
            self.feature_stats[symbol] = {
                "mean": np.zeros(self.n_features),
                "std": np.ones(self.n_features),
                "count": 0,
            }

        # Update statistics (online mean/std calculation)
        stats = self.feature_stats[symbol]
        stats["count"] += 1
        delta = features - stats["mean"]
        stats["mean"] += delta / stats["count"]
        delta2 = features - stats["mean"]

        if stats["count"] > 1:
            stats["std"] = np.sqrt(
                ((stats["count"] - 1) * stats["std"] ** 2 + delta * delta2)
                / stats["count"]
            )

        # Normalize and add to buffer
        normalized_features = (features - stats["mean"]) / (stats["std"] + 1e-8)
        self.buffers[symbol].append(normalized_features)

    def get_input_sequence(self, symbol: str) -> Optional[np.ndarray]:
        """
        Get input sequence for model inference

        Returns:
        --------
        sequence : np.ndarray or None
            Array of shape (1, lookback_window, n_features) or None if not ready
        """
        if symbol not in self.buffers:
            return None

        buffer = self.buffers[symbol]

        if len(buffer) < self.lookback_window:
            logger.debug(
                f"Buffer for {symbol} not full yet ({len(buffer)}/{self.lookback_window})"
            )
            return None

        # Convert to array and reshape
        sequence = np.array(buffer)  # Shape: (lookback_window, n_features)
        sequence = sequence.reshape(1, self.lookback_window, self.n_features)

        return sequence

    def is_ready(self, symbol: str) -> bool:
        """Check if buffer is ready for inference"""
        return (
            symbol in self.buffers and len(self.buffers[symbol]) >= self.lookback_window
        )


class StreamingInferencePipeline:
    """
    High-performance streaming inference pipeline
    Target latency: <100ms per prediction
    """

    def __init__(
        self,
        model_path: str,
        lookback_window: int = 30,
        n_features: int = 12,
        batch_size: int = 32,
    ):
        """
        Initialize streaming pipeline

        Parameters:
        -----------
        model_path : str
            Path to trained model
        lookback_window : int
            Lookback window size
        n_features : int
            Number of features
        batch_size : int
            Batch size for inference
        """
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.batch_size = batch_size

        # Load model
        self.model = self._load_model()

        # Initialize buffer
        self.buffer = StreamingDataBuffer(lookback_window, n_features)

        # Prediction callbacks
        self.callbacks: List[Callable] = []

        # Performance metrics
        self.latency_history = deque(maxlen=1000)
        self.prediction_count = 0

        # Batching for efficiency
        self.pending_predictions: Dict[str, asyncio.Future] = {}
        self.batch_lock = asyncio.Lock()

        logger.info("StreamingInferencePipeline initialized")
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Target latency: <100ms")

    def _load_model(self) -> tf.keras.Model:
        """Load trained model with custom objects"""
        from code.model import AttentionLayer

        custom_objects = {"AttentionLayer": AttentionLayer}

        logger.info(f"Loading model from {self.model_path}...")
        model = tf.keras.models.load_model(
            self.model_path, custom_objects=custom_objects
        )

        # Warmup
        dummy_input = np.random.randn(1, self.lookback_window, self.n_features).astype(
            np.float32
        )
        _ = model.predict(dummy_input, verbose=0)

        logger.info("Model loaded and warmed up")

        return model

    def register_callback(self, callback: Callable[[VolatilityPrediction], None]):
        """Register callback to receive predictions"""
        self.callbacks.append(callback)

    async def process_market_data(self, data_point: MarketDataPoint):
        """
        Process incoming market data point

        Parameters:
        -----------
        data_point : MarketDataPoint
            Incoming market data
        """
        # Extract features from market data
        features = self._extract_features(data_point)

        # Add to buffer
        self.buffer.add_data_point(data_point.symbol, features)

        # Check if ready for prediction
        if self.buffer.is_ready(data_point.symbol):
            # Trigger prediction
            prediction = await self.predict(data_point.symbol)

            if prediction:
                # Execute callbacks
                for callback in self.callbacks:
                    try:
                        callback(prediction)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

    def _extract_features(self, data_point: MarketDataPoint) -> np.ndarray:
        """
        Extract features from market data point

        Returns:
        --------
        features : np.ndarray
            Feature vector (n_features,)
        """
        # Simplified feature extraction
        # In production, this would include more sophisticated features

        returns = (
            (data_point.close - data_point.open) / data_point.open
            if data_point.open > 0
            else 0
        )
        high_low_spread = (
            (data_point.high - data_point.low) / data_point.close
            if data_point.close > 0
            else 0
        )
        volume_norm = data_point.volume / 1e6  # Normalize volume

        # Create feature vector (pad to n_features)
        features = np.array(
            [
                returns,
                high_low_spread,
                volume_norm,
                data_point.close / 100,  # Normalized price
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,  # Placeholder for additional features
            ]
        )[: self.n_features]

        return features

    async def predict(self, symbol: str) -> Optional[VolatilityPrediction]:
        """
        Make volatility prediction for symbol

        Returns:
        --------
        prediction : VolatilityPrediction or None
            Prediction result or None if not ready
        """
        # Get input sequence
        sequence = self.buffer.get_input_sequence(symbol)

        if sequence is None:
            return None

        # Measure latency
        start_time = time.time()

        # Inference
        predictions = self.model.predict(sequence, verbose=0)

        # Extract predictions
        if isinstance(predictions, list):
            volatility_pred = float(predictions[0][0][0])
            var_pred = float(predictions[1][0][0])
        else:
            volatility_pred = float(predictions[0][0])
            var_pred = None

        inference_time = (time.time() - start_time) * 1000

        # Track latency
        self.latency_history.append(inference_time)
        self.prediction_count += 1

        # Log if latency exceeds target
        if inference_time > 100:
            logger.warning(
                f"Latency exceeded target: {inference_time:.2f}ms for {symbol}"
            )

        # Create prediction object
        prediction = VolatilityPrediction(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            predicted_volatility=volatility_pred,
            var_99=var_pred,
            confidence_lower=volatility_pred * 0.85,
            confidence_upper=volatility_pred * 1.15,
            inference_time_ms=inference_time,
            model_version="1.0.0",
        )

        return prediction

    async def predict_batch(
        self, symbols: List[str]
    ) -> List[Optional[VolatilityPrediction]]:
        """
        Batch prediction for multiple symbols (more efficient)

        Returns:
        --------
        predictions : list
            List of predictions (same order as symbols)
        """
        # Collect sequences
        sequences = []
        valid_symbols = []

        for symbol in symbols:
            seq = self.buffer.get_input_sequence(symbol)
            if seq is not None:
                sequences.append(seq)
                valid_symbols.append(symbol)

        if not sequences:
            return [None] * len(symbols)

        # Batch inference
        start_time = time.time()

        batch_input = np.vstack(sequences)
        predictions_batch = self.model.predict(batch_input, verbose=0)

        inference_time = (time.time() - start_time) * 1000

        logger.info(
            f"Batch inference: {len(sequences)} symbols in {inference_time:.2f}ms"
        )

        # Create prediction objects
        predictions = []

        for i, symbol in enumerate(valid_symbols):
            if isinstance(predictions_batch, list):
                vol_pred = float(predictions_batch[0][i][0])
                var_pred = float(predictions_batch[1][i][0])
            else:
                vol_pred = float(predictions_batch[i][0])
                var_pred = None

            prediction = VolatilityPrediction(
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                predicted_volatility=vol_pred,
                var_99=var_pred,
                confidence_lower=vol_pred * 0.85,
                confidence_upper=vol_pred * 1.15,
                inference_time_ms=inference_time / len(sequences),
                model_version="1.0.0",
            )

            predictions.append(prediction)

        # Map back to original symbol order
        result = []
        pred_idx = 0
        for symbol in symbols:
            if symbol in valid_symbols:
                result.append(predictions[pred_idx])
                pred_idx += 1
            else:
                result.append(None)

        return result

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.latency_history:
            return {
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "total_predictions": 0,
            }

        latencies = np.array(self.latency_history)

        return {
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": np.max(latencies),
            "total_predictions": self.prediction_count,
        }


# Example usage and testing
async def example_streaming_workflow():
    """Example workflow for streaming inference"""

    # Initialize pipeline
    pipeline = StreamingInferencePipeline(
        model_path="../models/lstm_attention_model.h5",
        lookback_window=30,
        n_features=12,
    )

    # Register callback
    def on_prediction(prediction: VolatilityPrediction):
        print(
            f"[{prediction.timestamp}] {prediction.symbol}: "
            f"Vol={prediction.predicted_volatility:.4f} "
            f"(latency={prediction.inference_time_ms:.2f}ms)"
        )

    pipeline.register_callback(on_prediction)

    # Simulate streaming data
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]

    for i in range(100):
        for symbol in symbols:
            # Simulate market data
            data_point = MarketDataPoint(
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                price=100.0 + np.random.randn() * 2,
                volume=1000000 + np.random.randint(-100000, 100000),
                high=101.0 + np.random.randn(),
                low=99.0 + np.random.randn(),
                open=100.0 + np.random.randn(),
                close=100.0 + np.random.randn(),
            )

            await pipeline.process_market_data(data_point)

        # Small delay to simulate real-time data feed
        await asyncio.sleep(0.1)

    # Print performance stats
    stats = pipeline.get_performance_stats()
    print("\nPerformance Statistics:")
    print(f"  Average latency: {stats['avg_latency_ms']:.2f}ms")
    print(f"  P95 latency: {stats['p95_latency_ms']:.2f}ms")
    print(f"  P99 latency: {stats['p99_latency_ms']:.2f}ms")
    print(f"  Total predictions: {stats['total_predictions']}")


if __name__ == "__main__":
    print("Streaming inference pipeline loaded.")
    print("Run example: asyncio.run(example_streaming_workflow())")

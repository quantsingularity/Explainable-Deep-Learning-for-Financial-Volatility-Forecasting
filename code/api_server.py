"""
FastAPI Inference Server for Real-Time Volatility Forecasting
Provides REST API for model serving with <100ms latency target
"""

import os
import time
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import logging

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Volatility Forecasting API",
    description="Production-grade API for multi-horizon volatility forecasting",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"]
)
MODEL_INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds", "Model inference time"
)
ACTIVE_REQUESTS = Gauge("api_active_requests", "Number of active requests")

# Global variables
model = None
redis_client = None
model_loaded = False


class VolatilityRequest(BaseModel):
    """Request schema for volatility prediction"""

    features: List[List[float]] = Field(
        ...,
        description="Input features as 2D array (time_steps x n_features)",
        example=[
            [
                0.01,
                0.02,
                0.015,
                0.008,
                0.012,
                0.005,
                0.003,
                0.001,
                0.002,
                0.004,
                0.006,
                0.003,
            ]
        ],
    )
    horizon: Optional[int] = Field(
        1, description="Forecast horizon (1, 5, or 22 days)", ge=1, le=22
    )
    asset_id: Optional[str] = Field("SPY", description="Asset identifier for caching")
    return_var: Optional[bool] = Field(
        True, description="Whether to return VaR predictions"
    )


class VolatilityResponse(BaseModel):
    """Response schema for volatility prediction"""

    volatility_forecast: float = Field(..., description="Predicted volatility")
    var_forecast: Optional[float] = Field(None, description="Value-at-Risk forecast")
    horizon: int = Field(..., description="Forecast horizon in days")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="95% confidence interval"
    )
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")
    asset_id: str = Field(..., description="Asset identifier")


class BatchVolatilityRequest(BaseModel):
    """Batch prediction request"""

    requests: List[VolatilityRequest] = Field(
        ..., description="List of volatility prediction requests"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    timestamp: str
    uptime_seconds: float


# Startup time for uptime calculation
START_TIME = time.time()


def load_model(model_path: str = "/app/models/lstm_attention_model.h5"):
    """Load the trained model"""
    global model, model_loaded

    try:
        logger.info(f"Loading model from {model_path}")

        # Custom objects for loading
        from code.model import AttentionLayer

        custom_objects = {"AttentionLayer": AttentionLayer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

        # Warmup inference
        dummy_input = np.random.randn(1, 30, 12).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)

        model_loaded = True
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        model_loaded = False


def init_redis():
    """Initialize Redis connection"""
    global redis_client

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    try:
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed: {str(e)}. Caching disabled.")
        redis_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    model_path = os.getenv("MODEL_PATH", "/app/models/lstm_attention_model.h5")
    load_model(model_path)
    init_redis()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=time.time() - START_TIME,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


def get_cache_key(request: VolatilityRequest) -> str:
    """Generate cache key for request"""
    features_hash = hash(tuple(map(tuple, request.features)))
    return f"volatility:{request.asset_id}:{request.horizon}:{features_hash}"


async def get_cached_prediction(cache_key: str) -> Optional[Dict]:
    """Get prediction from cache"""
    if redis_client is None:
        return None

    try:
        cached = redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {str(e)}")

    return None


async def cache_prediction(cache_key: str, prediction: Dict, ttl: int = 300):
    """Cache prediction with TTL"""
    if redis_client is None:
        return

    try:
        redis_client.setex(cache_key, ttl, json.dumps(prediction))
    except Exception as e:
        logger.warning(f"Cache write error: {str(e)}")


@app.post("/predict", response_model=VolatilityResponse)
async def predict_volatility(request: VolatilityRequest):
    """
    Single volatility prediction endpoint

    Target latency: <100ms
    """

    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # Validate model is loaded
        if not model_loaded or model is None:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Check cache
        cache_key = get_cache_key(request)
        cached_result = await get_cached_prediction(cache_key)

        if cached_result:
            logger.info("Returning cached prediction")
            REQUEST_COUNT.labels(endpoint="/predict", status="cache_hit").inc()
            return VolatilityResponse(**cached_result)

        # Prepare input
        features_array = np.array(request.features, dtype=np.float32)

        # Validate shape
        if len(features_array.shape) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Features must be 2D array, got shape {features_array.shape}",
            )

        # Reshape to (batch_size, time_steps, features)
        if features_array.shape[0] == 1:
            # Already in correct format
            model_input = features_array.reshape(1, *features_array.shape)
        else:
            model_input = features_array.reshape(1, *features_array.shape)

        # Inference
        inference_start = time.time()
        predictions = model.predict(model_input, verbose=0)
        inference_time = (time.time() - inference_start) * 1000

        MODEL_INFERENCE_LATENCY.observe(inference_time / 1000)

        # Extract predictions
        if isinstance(predictions, list):
            volatility_pred = float(predictions[0][0][0])
            var_pred = float(predictions[1][0][0]) if request.return_var else None
        else:
            volatility_pred = float(predictions[0][0])
            var_pred = None

        # Calculate confidence interval (simplified)
        ci_lower = volatility_pred * 0.85
        ci_upper = volatility_pred * 1.15

        # Prepare response
        response_data = {
            "volatility_forecast": volatility_pred,
            "var_forecast": var_pred,
            "horizon": request.horizon,
            "confidence_interval": {"lower": ci_lower, "upper": ci_upper},
            "inference_time_ms": inference_time,
            "timestamp": datetime.utcnow().isoformat(),
            "asset_id": request.asset_id,
        }

        # Cache result
        await cache_prediction(cache_key, response_data)

        # Metrics
        total_time = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint="/predict").observe(total_time / 1000)
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()

        logger.info(
            f"Prediction completed in {total_time:.2f}ms (inference: {inference_time:.2f}ms)"
        )

        return VolatilityResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.post("/predict/batch")
async def predict_batch(batch_request: BatchVolatilityRequest):
    """
    Batch prediction endpoint for multiple assets
    """

    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        if not model_loaded or model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Process predictions concurrently
        tasks = [predict_volatility(req) for req in batch_request.requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        successful_results = []
        errors = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"index": i, "error": str(result)})
            else:
                successful_results.append(result)

        total_time = (time.time() - start_time) * 1000

        return {
            "predictions": successful_results,
            "errors": errors,
            "total_time_ms": total_time,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/model/info")
async def model_info():
    """Get model information"""

    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model.name,
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum([tf.size(w).numpy() for w in model.trainable_weights])
        ),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    uvicorn.run(
        "api_server:app", host="0.0.0.0", port=port, workers=4, log_level="info"
    )

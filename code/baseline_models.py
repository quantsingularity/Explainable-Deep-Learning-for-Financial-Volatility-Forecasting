"""
Baseline Models Implementation
GARCH, EGARCH, HAR-RV, and other benchmark models for comparison.
"""

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """GARCH(1,1) volatility forecasting model."""
    
    def __init__(self, p=1, q=1):
        """
        Initialize GARCH model.
        
        Parameters:
        -----------
        p : int
            GARCH order (lags of variance)
        q : int
            ARCH order (lags of squared residuals)
        """
        self.p = p
        self.q = q
        self.model = None
        self.results = None
    
    def fit(self, returns, verbose=False):
        """
        Fit GARCH model to returns.
        
        Parameters:
        -----------
        returns : np.array or pd.Series
            Return series
        verbose : bool
            Print fitting information
        """
        # Scale returns to percentage
        returns_scaled = returns * 100
        
        # Build and fit model
        self.model = arch_model(
            returns_scaled,
            vol='Garch',
            p=self.p,
            q=self.q,
            dist='normal'
        )
        
        self.results = self.model.fit(disp='off' if not verbose else 'final')
        
        if verbose:
            print(self.results.summary())
    
    def forecast(self, horizon=1):
        """
        Forecast volatility.
        
        Parameters:
        -----------
        horizon : int
            Forecast horizon
        
        Returns:
        --------
        forecast : np.array
            Volatility forecasts
        """
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get forecasts
        forecasts = self.results.forecast(horizon=horizon)
        
        # Extract conditional variance and convert to volatility
        variance_forecast = forecasts.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back to decimal
        
        return volatility_forecast
    
    def rolling_forecast(self, returns, train_size, test_size):
        """
        Perform rolling window forecast.
        
        Parameters:
        -----------
        returns : np.array
            Full return series
        train_size : int
            Initial training window size
        test_size : int
            Number of out-of-sample forecasts
        
        Returns:
        --------
        forecasts : np.array
            Rolling volatility forecasts
        """
        forecasts = np.zeros(test_size)
        
        for i in range(test_size):
            # Get training window
            train_data = returns[i:train_size + i]
            
            # Fit model
            self.fit(train_data, verbose=False)
            
            # Forecast
            forecasts[i] = self.forecast(horizon=1)[0]
        
        return forecasts


class EGARCHModel:
    """EGARCH(1,1) volatility forecasting model with leverage effects."""
    
    def __init__(self, p=1, o=1, q=1):
        """
        Initialize EGARCH model.
        
        Parameters:
        -----------
        p : int
            GARCH order
        o : int
            Asymmetry order
        q : int
            ARCH order
        """
        self.p = p
        self.o = o
        self.q = q
        self.model = None
        self.results = None
    
    def fit(self, returns, verbose=False):
        """Fit EGARCH model to returns."""
        returns_scaled = returns * 100
        
        self.model = arch_model(
            returns_scaled,
            vol='EGARCH',
            p=self.p,
            o=self.o,
            q=self.q,
            dist='normal'
        )
        
        self.results = self.model.fit(disp='off' if not verbose else 'final')
        
        if verbose:
            print(self.results.summary())
    
    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecasts = self.results.forecast(horizon=horizon)
        variance_forecast = forecasts.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast) / 100
        
        return volatility_forecast
    
    def rolling_forecast(self, returns, train_size, test_size):
        """Perform rolling window forecast."""
        forecasts = np.zeros(test_size)
        
        for i in range(test_size):
            train_data = returns[i:train_size + i]
            
            try:
                self.fit(train_data, verbose=False)
                forecasts[i] = self.forecast(horizon=1)[0]
            except:
                # If fitting fails, use last known volatility
                forecasts[i] = np.std(train_data)
        
        return forecasts


class HARRVModel:
    """
    Heterogeneous Autoregressive Realized Volatility (HAR-RV) model.
    
    RV_t = β_0 + β_d * RV_{t-1} + β_w * RV_{t-5:t-1} + β_m * RV_{t-22:t-1} + ε_t
    """
    
    def __init__(self):
        """Initialize HAR-RV model."""
        self.coefficients = None
    
    def _create_features(self, rv_series):
        """
        Create HAR features (daily, weekly, monthly components).
        
        Parameters:
        -----------
        rv_series : np.array or pd.Series
            Realized volatility series
        
        Returns:
        --------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        """
        rv = np.array(rv_series)
        n = len(rv)
        
        # Initialize feature matrix
        X = []
        y = []
        
        # Need at least 22 days of history
        for t in range(22, n):
            features = [
                1.0,  # Intercept
                rv[t-1],  # Daily component
                np.mean(rv[t-5:t]),  # Weekly component (5-day average)
                np.mean(rv[t-22:t])  # Monthly component (22-day average)
            ]
            X.append(features)
            y.append(rv[t])
        
        return np.array(X), np.array(y)
    
    def fit(self, rv_series, verbose=False):
        """
        Fit HAR-RV model using OLS.
        
        Parameters:
        -----------
        rv_series : np.array or pd.Series
            Realized volatility series
        verbose : bool
            Print fitting information
        """
        X, y = self._create_features(rv_series)
        
        # OLS estimation: β = (X'X)^{-1} X'y
        self.coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        
        if verbose:
            print("\nHAR-RV Model Coefficients:")
            print(f"  Intercept: {self.coefficients[0]:.6f}")
            print(f"  Daily (RV_t-1): {self.coefficients[1]:.6f}")
            print(f"  Weekly (RV_t-5:t): {self.coefficients[2]:.6f}")
            print(f"  Monthly (RV_t-22:t): {self.coefficients[3]:.6f}")
            
            # Calculate R²
            predictions = X @ self.coefficients
            r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
            print(f"  In-sample R²: {r2:.4f}")
    
    def forecast(self, rv_history):
        """
        Forecast next-period volatility.
        
        Parameters:
        -----------
        rv_history : np.array
            Historical realized volatility (at least 22 periods)
        
        Returns:
        --------
        forecast : float
            One-step-ahead volatility forecast
        """
        if self.coefficients is None:
            raise ValueError("Model must be fitted before forecasting")
        
        if len(rv_history) < 22:
            raise ValueError("Need at least 22 periods of history")
        
        # Create features for forecast
        features = np.array([
            1.0,
            rv_history[-1],
            np.mean(rv_history[-5:]),
            np.mean(rv_history[-22:])
        ])
        
        forecast = features @ self.coefficients
        
        return forecast
    
    def rolling_forecast(self, rv_series, train_size, test_size):
        """
        Perform rolling window forecast.
        
        Parameters:
        -----------
        rv_series : np.array
            Full realized volatility series
        train_size : int
            Initial training window size
        test_size : int
            Number of out-of-sample forecasts
        
        Returns:
        --------
        forecasts : np.array
            Rolling forecasts
        """
        forecasts = np.zeros(test_size)
        
        for i in range(test_size):
            # Get training window
            train_data = rv_series[i:train_size + i]
            
            # Fit model
            self.fit(train_data, verbose=False)
            
            # Forecast
            rv_history = train_data[-22:]
            forecasts[i] = self.forecast(rv_history)
        
        return forecasts


class HistoricalVolatility:
    """Simple historical volatility baseline (rolling standard deviation)."""
    
    def __init__(self, window=30):
        """
        Initialize historical volatility model.
        
        Parameters:
        -----------
        window : int
            Rolling window size
        """
        self.window = window
    
    def forecast(self, returns):
        """
        Forecast volatility as rolling standard deviation.
        
        Parameters:
        -----------
        returns : np.array
            Return series
        
        Returns:
        --------
        forecast : float
            Volatility forecast
        """
        if len(returns) < self.window:
            return np.std(returns)
        
        return np.std(returns[-self.window:])
    
    def rolling_forecast(self, returns, train_size, test_size):
        """Perform rolling window forecast."""
        forecasts = np.zeros(test_size)
        
        for i in range(test_size):
            train_data = returns[i:train_size + i]
            forecasts[i] = self.forecast(train_data)
        
        return forecasts


def compare_baseline_models(returns, rv_actual, train_ratio=0.7):
    """
    Compare all baseline models on the same dataset.
    
    Parameters:
    -----------
    returns : np.array
        Return series
    rv_actual : np.array
        Actual realized volatility
    train_ratio : float
        Proportion of data for training
    
    Returns:
    --------
    results : dict
        Comparison results
    """
    n = len(returns)
    train_size = int(n * train_ratio)
    test_size = n - train_size
    
    print(f"\n{'='*70}")
    print("BASELINE MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"Training size: {train_size}")
    print(f"Test size: {test_size}")
    
    results = {}
    
    # Historical Volatility
    print("\n[1/4] Historical Volatility...")
    hist_vol = HistoricalVolatility(window=30)
    hist_forecasts = hist_vol.rolling_forecast(returns, train_size, test_size)
    results['Historical'] = hist_forecasts
    
    # GARCH(1,1)
    print("[2/4] GARCH(1,1)...")
    garch = GARCHModel(p=1, q=1)
    garch_forecasts = garch.rolling_forecast(returns, train_size, test_size)
    results['GARCH'] = garch_forecasts
    
    # EGARCH(1,1)
    print("[3/4] EGARCH(1,1)...")
    egarch = EGARCHModel(p=1, o=1, q=1)
    egarch_forecasts = egarch.rolling_forecast(returns, train_size, test_size)
    results['EGARCH'] = egarch_forecasts
    
    # HAR-RV
    print("[4/4] HAR-RV...")
    har = HARRVModel()
    har_forecasts = har.rolling_forecast(rv_actual, train_size, test_size)
    results['HAR-RV'] = har_forecasts
    
    print("\n✓ All baseline models completed")
    
    return results, rv_actual[train_size:]


if __name__ == "__main__":
    # Example usage
    print("Baseline models module loaded.")
    print("\nAvailable models:")
    print("  - GARCHModel: GARCH(1,1) volatility model")
    print("  - EGARCHModel: EGARCH with leverage effects")
    print("  - HARRVModel: Heterogeneous Autoregressive RV")
    print("  - HistoricalVolatility: Simple rolling volatility")
    print("\nUse compare_baseline_models() for comprehensive comparison.")

"""
Synthetic Data Generator for Volatility Forecasting
Generates realistic financial time series data for demonstration purposes.
"""

import numpy as np
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(123)


def generate_garch_returns(n_days=1827, omega=0.00001, alpha=0.1, beta=0.85):
    """
    Generate returns following a GARCH(1,1) process.

    Parameters:
    -----------
    n_days : int
        Number of trading days to generate
    omega, alpha, beta : float
        GARCH parameters

    Returns:
    --------
    returns : np.array
        Daily log returns
    volatility : np.array
        Daily conditional volatility
    """
    returns = np.zeros(n_days)
    volatility = np.zeros(n_days)

    # Initialize
    volatility[0] = np.sqrt(omega / (1 - alpha - beta))
    returns[0] = volatility[0] * np.random.randn()

    for t in range(1, n_days):
        # GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        volatility[t] = np.sqrt(
            omega + alpha * returns[t - 1] ** 2 + beta * volatility[t - 1] ** 2
        )

        # Add regime shifts for realism
        if t == 500:  # Simulated crisis
            volatility[t] *= 2.5
        elif t == 1400:  # Another volatility spike
            volatility[t] *= 1.8

        returns[t] = volatility[t] * np.random.randn()

    return returns, volatility


def generate_geopolitical_risk_index(n_days=1827):
    """Generate synthetic GPR index with trend and spikes."""
    base_level = 100
    trend = np.linspace(0, 20, n_days)
    noise = np.random.randn(n_days) * 10

    # Add crisis spikes
    spikes = np.zeros(n_days)
    spikes[500:550] = 80  # Crisis 1
    spikes[1400:1450] = 60  # Crisis 2

    gpr = base_level + trend + noise + spikes
    return np.clip(gpr, 50, 300)


def generate_vix_index(volatility, base_vix=15):
    """Generate VIX-like implied volatility index."""
    # VIX roughly scales with realized vol but is forward-looking
    vix = base_vix + volatility * 500 + np.random.randn(len(volatility)) * 2
    return np.clip(vix, 10, 80)


def calculate_high_low_spread(returns, volatility):
    """Simulate intraday high-low spread."""
    # High-low spread correlates with volatility
    base_spread = volatility * 1.5
    noise = np.abs(np.random.randn(len(returns)) * volatility * 0.3)
    return base_spread + noise


def generate_volume(n_days=1827, base_volume=1e9):
    """Generate normalized trading volume."""
    # Volume increases during volatile periods
    trend = np.random.randn(n_days) * 0.2
    spikes = np.zeros(n_days)
    spikes[500:550] = 1.5
    spikes[1400:1450] = 1.2

    volume = base_volume * (1 + trend + spikes)
    return volume


def generate_synthetic_dataset(n_days=1827, start_date="2018-01-01"):
    """
    Generate complete synthetic dataset for volatility forecasting.

    Parameters:
    -----------
    n_days : int
        Number of trading days
    start_date : str
        Starting date for the dataset

    Returns:
    --------
    df : pd.DataFrame
        Complete dataset with all features
    """
    print(f"Generating {n_days} days of synthetic financial data...")

    # Generate base price and returns using GARCH
    returns, realized_vol = generate_garch_returns(n_days)

    # Generate price from returns (start at 100)
    price = 100 * np.exp(np.cumsum(returns))

    # Calculate high and low prices
    hl_spread = calculate_high_low_spread(returns, realized_vol)
    high = price * (1 + hl_spread / 2)
    low = price * (1 - hl_spread / 2)

    # Generate auxiliary features
    gpr = generate_geopolitical_risk_index(n_days)
    vix = generate_vix_index(realized_vol)
    volume = generate_volume(n_days)

    # Create date range (business days only)
    dates = pd.date_range(start=start_date, periods=n_days, freq="B")

    # Build DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": high,
            "low": low,
            "close": price,
            "volume": volume,
            "returns": returns,
            "realized_volatility": realized_vol,
            "high_low_spread": hl_spread,
            "vix": vix,
            "gpr_index": gpr,
        }
    )

    # Add lagged features
    for lag in [1, 5, 22]:
        df[f"rv_lag{lag}"] = df["realized_volatility"].shift(lag)
        df[f"returns_lag{lag}"] = df["returns"].shift(lag)

    # Normalize volume (rolling 30-day window)
    df["volume_normalized"] = (df["volume"] - df["volume"].rolling(30).mean()) / df[
        "volume"
    ].rolling(30).std()

    # Drop NaN rows from lagging
    df = df.dropna().reset_index(drop=True)

    print(f"Generated dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


def save_dataset(df, filename="synthetic_data.csv"):
    """Save generated dataset to CSV."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    # Generate synthetic dataset
    df = generate_synthetic_dataset(n_days=1827, start_date="2018-01-01")

    save_dataset(df, "./data/synthetic_data.csv")

    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(df[["returns", "realized_volatility", "gpr_index", "vix"]].describe())

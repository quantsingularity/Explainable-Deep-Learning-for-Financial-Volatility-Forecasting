"""
Utility Functions for Data Loading, Preprocessing, and Evaluation
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(123)


def load_and_prepare_data(filepath, feature_cols=None):
    """
    Load dataset and prepare features.

    Parameters:
    -----------
    filepath : str
        Path to CSV file
    feature_cols : list
        List of feature column names to use

    Returns:
    --------
    df : pd.DataFrame
        Loaded and prepared dataframe
    """
    df = pd.read_csv(filepath)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Default feature set
    if feature_cols is None:
        feature_cols = [
            "returns",
            "high_low_spread",
            "volume_normalized",
            "realized_volatility",
            "vix",
            "gpr_index",
            "rv_lag1",
            "rv_lag5",
            "rv_lag22",
            "returns_lag1",
            "returns_lag5",
            "returns_lag22",
        ]

    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]

    return df, available_features


def create_sequences(data, target, lookback=30):
    """
    Create sequences for LSTM training.

    Parameters:
    -----------
    data : np.array
        Feature matrix (n_samples, n_features)
    target : np.array
        Target values (n_samples,)
    lookback : int
        Number of time steps to look back

    Returns:
    --------
    X : np.array
        Sequences (n_sequences, lookback, n_features)
    y : np.array
        Targets (n_sequences,)
    """
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i])
        y.append(target[i])

    return np.array(X), np.array(y)


def train_val_test_split(
    df,
    feature_cols,
    target_col="realized_volatility",
    train_end="2022-12-31",
    val_end="2023-06-30",
):
    """
    Split data into train, validation, and test sets chronologically.

    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    feature_cols : list
        Feature column names
    target_col : str
        Target column name
    train_end : str
        End date for training set
    val_end : str
        End date for validation set

    Returns:
    --------
    train_data, val_data, test_data : dict
        Dictionaries containing X, y, dates, and scalers
    """
    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Split by date
    train_df = df[df["date"] <= train_end].copy()
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test_df = df[df["date"] > val_end].copy()

    print(
        f"Train set: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})"
    )
    print(
        f"Val set: {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})"
    )
    print(
        f"Test set: {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})"
    )

    # Fit scaler on training data only
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.reshape(-1, 1)

    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values.reshape(-1, 1)

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.reshape(-1, 1)

    # Fit and transform
    X_train_scaled = feature_scaler.fit_transform(X_train)
    y_train_scaled = target_scaler.fit_transform(y_train)

    X_val_scaled = feature_scaler.transform(X_val)
    y_val_scaled = target_scaler.transform(y_val)

    X_test_scaled = feature_scaler.transform(X_test)
    y_test_scaled = target_scaler.transform(y_test)

    # Create sequences
    lookback = 30
    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled, y_train_scaled.flatten(), lookback
    )
    X_val_seq, y_val_seq = create_sequences(
        X_val_scaled, y_val_scaled.flatten(), lookback
    )
    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled, y_test_scaled.flatten(), lookback
    )

    print("\nSequence shapes:")
    print(f"X_train: {X_train_seq.shape}, y_train: {y_train_seq.shape}")
    print(f"X_val: {X_val_seq.shape}, y_val: {y_val_seq.shape}")
    print(f"X_test: {X_test_seq.shape}, y_test: {y_test_seq.shape}")

    return {
        "train": {
            "X": X_train_seq,
            "y": y_train_seq,
            "dates": train_df["date"].values[lookback:],
        },
        "val": {
            "X": X_val_seq,
            "y": y_val_seq,
            "dates": val_df["date"].values[lookback:],
        },
        "test": {
            "X": X_test_seq,
            "y": y_test_seq,
            "dates": test_df["date"].values[lookback:],
        },
        "scalers": {"feature": feature_scaler, "target": target_scaler},
        "feature_names": feature_cols,
    }


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def calculate_qlike(y_true, y_pred, epsilon=1e-10):
    """
    Calculate QLIKE loss.
    QLIKE penalizes under-predictions more heavily.
    """
    y_pred = np.maximum(y_pred, epsilon)  # Avoid division by zero
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


def calculate_r2(y_true, y_pred):
    """Calculate R-squared coefficient."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def diebold_mariano_test(errors1, errors2):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Parameters:
    -----------
    errors1, errors2 : np.array
        Forecast errors from two models

    Returns:
    --------
    dm_stat : float
        DM test statistic
    p_value : float
        Two-tailed p-value
    """
    d = errors1**2 - errors2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    n = len(d)

    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    return dm_stat, p_value


def calculate_var(returns, alpha=0.01):
    """
    Calculate Value at Risk (VaR) at specified confidence level.

    Parameters:
    -----------
    returns : np.array
        Return series
    alpha : float
        Significance level (e.g., 0.01 for 99% VaR)

    Returns:
    --------
    var : float
        VaR threshold
    """
    return np.quantile(returns, alpha)


def kupiec_test(violations, n_observations, alpha=0.01):
    """
    Kupiec's Proportion of Failures (POF) test.

    Parameters:
    -----------
    violations : int
        Number of VaR violations
    n_observations : int
        Total number of observations
    alpha : float
        VaR significance level

    Returns:
    --------
    lr_stat : float
        Likelihood ratio statistic
    p_value : float
        P-value
    """
    violation_rate = violations / n_observations

    if violations == 0:
        lr_stat = -2 * (n_observations * np.log(1 - alpha))
    elif violations == n_observations:
        lr_stat = -2 * (n_observations * np.log(alpha))
    else:
        lr_stat = -2 * (
            violations * np.log(alpha / violation_rate)
            + (n_observations - violations) * np.log((1 - alpha) / (1 - violation_rate))
        )

    # LR follows chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return lr_stat, p_value


def christoffersen_test(violations_binary):
    """
    Christoffersen's Independence test for VaR violations.

    Parameters:
    -----------
    violations_binary : np.array
        Binary array (1 = violation, 0 = no violation)

    Returns:
    --------
    lr_stat : float
        Likelihood ratio statistic
    p_value : float
        P-value
    """
    n00 = np.sum((violations_binary[:-1] == 0) & (violations_binary[1:] == 0))
    n01 = np.sum((violations_binary[:-1] == 0) & (violations_binary[1:] == 1))
    n10 = np.sum((violations_binary[:-1] == 1) & (violations_binary[1:] == 0))
    n11 = np.sum((violations_binary[:-1] == 1) & (violations_binary[1:] == 1))

    # Transition probabilities
    if (n00 + n01) == 0:
        pi_01 = 0
    else:
        pi_01 = n01 / (n00 + n01)

    if (n10 + n11) == 0:
        pi_11 = 0
    else:
        pi_11 = n11 / (n10 + n11)

    # n-1 pairs (transitions), not the full n-length array.
    n_transitions = len(violations_binary) - 1
    pi = (n01 + n11) / n_transitions if n_transitions > 0 else 0

    # Likelihood ratio
    # Guard every log argument to avoid log(0) = -inf and log(1-1) = log(0) = -inf
    if pi_01 <= 0 or pi_01 >= 1 or pi_11 <= 0 or pi_11 >= 1 or pi <= 0 or pi >= 1:
        lr_stat = 0
    else:
        lr_stat = -2 * (
            (n00 + n10) * np.log(1 - pi)
            + (n01 + n11) * np.log(pi)
            - n00 * np.log(1 - pi_01)
            - n01 * np.log(pi_01)
            - n10 * np.log(1 - pi_11)
            - n11 * np.log(pi_11)
        )

    # LR follows chi-squared with 1 df
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return lr_stat, p_value


if __name__ == "__main__":
    print("Utility functions loaded successfully.")
    print("Available functions:")
    print("  - load_and_prepare_data")
    print("  - create_sequences")
    print("  - train_val_test_split")
    print("  - calculate_rmse, calculate_mae, calculate_qlike, calculate_r2")
    print("  - diebold_mariano_test")
    print("  - calculate_var, kupiec_test, christoffersen_test")

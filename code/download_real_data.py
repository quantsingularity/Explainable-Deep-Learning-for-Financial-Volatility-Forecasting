"""
Real Financial Data Downloader
Downloads actual market data from Yahoo Finance and external sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

def download_price_data(tickers, start_date='2018-01-01', end_date='2024-12-31'):
    """
    Download historical price data for multiple assets.
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for data download
    end_date : str
        End date for data download
    
    Returns:
    --------
    data_dict : dict
        Dictionary containing dataframes for each ticker
    """
    print(f"\n{'='*70}")
    print("DOWNLOADING REAL MARKET DATA")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Assets: {', '.join(tickers)}")
    
    data_dict = {}
    
    for ticker in tickers:
        print(f"\nDownloading {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data_dict[ticker] = data
                print(f"  ✓ Downloaded {len(data)} trading days")
            else:
                print(f"  ✗ No data available for {ticker}")
        except Exception as e:
            print(f"  ✗ Error downloading {ticker}: {str(e)}")
    
    return data_dict


def calculate_realized_volatility(df, window=30):
    """
    Calculate realized volatility from price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with OHLC columns
    window : int
        Rolling window for volatility calculation
    
    Returns:
    --------
    rv : pd.Series
        Realized volatility series
    """
    # Calculate log returns
    returns = np.log(df['Close'] / df['Close'].shift(1))
    
    # Calculate realized volatility (rolling std)
    rv = returns.rolling(window=window).std()
    
    # Annualize volatility (252 trading days)
    rv = rv * np.sqrt(252)
    
    return rv


def calculate_high_low_volatility(df):
    """
    Calculate Parkinson's high-low volatility estimator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with High and Low columns
    
    Returns:
    --------
    hl_vol : pd.Series
        High-low volatility series
    """
    hl_ratio = np.log(df['High'] / df['Low'])
    hl_vol = hl_ratio / (2 * np.sqrt(np.log(2)))
    
    # Annualize
    hl_vol = hl_vol * np.sqrt(252)
    
    return hl_vol


def download_vix_data(start_date='2018-01-01', end_date='2024-12-31'):
    """
    Download VIX (Volatility Index) data.
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns:
    --------
    vix_df : pd.DataFrame
        VIX data
    """
    print("\nDownloading VIX data...")
    try:
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if not vix.empty:
            print(f"  ✓ Downloaded VIX data: {len(vix)} days")
            return vix['Close']
        else:
            print("  ✗ No VIX data available")
            return None
    except Exception as e:
        print(f"  ✗ Error downloading VIX: {str(e)}")
        return None


def generate_synthetic_gpr(dates):
    """
    Generate synthetic Geopolitical Risk (GPR) index.
    Note: Real GPR data requires specialized data sources.
    This creates a realistic proxy with crisis events.
    
    Parameters:
    -----------
    dates : pd.DatetimeIndex
        Date index
    
    Returns:
    --------
    gpr : pd.Series
        Synthetic GPR index
    """
    print("\nGenerating synthetic GPR index...")
    print("  Note: Real GPR data requires Federal Reserve Board access")
    
    np.random.seed(42)
    n = len(dates)
    
    # Base level with trend
    base = 100
    trend = np.linspace(0, 20, n)
    noise = np.random.randn(n) * 8
    
    # Add major geopolitical events
    gpr_values = base + trend + noise
    
    for i, date in enumerate(dates):
        # COVID-19 pandemic (early 2020)
        if '2020-03-01' <= str(date) <= '2020-06-01':
            gpr_values[i] += 60
        
        # Russia-Ukraine conflict (Feb 2022 onwards)
        if date >= pd.Timestamp('2022-02-24'):
            gpr_values[i] += 50
        
        # Middle East tensions (2023-2024)
        if '2023-10-01' <= str(date) <= '2024-03-01':
            gpr_values[i] += 30
    
    gpr = pd.Series(np.clip(gpr_values, 50, 300), index=dates, name='gpr_index')
    print(f"  ✓ Generated GPR index: {len(gpr)} days")
    
    return gpr


def build_complete_dataset(ticker='SPY', start_date='2018-01-01', end_date='2024-12-31'):
    """
    Build complete dataset with all features for a single asset.
    
    Parameters:
    -----------
    ticker : str
        Asset ticker symbol
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns:
    --------
    df : pd.DataFrame
        Complete dataset with all features
    """
    print(f"\n{'='*70}")
    print(f"BUILDING COMPLETE DATASET FOR {ticker}")
    print(f"{'='*70}")
    
    # Download price data
    price_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if price_data.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    
    # Reset index to have date as column
    df = price_data.reset_index()
    df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    
    print(f"\nBase data: {len(df)} trading days")
    
    # Calculate returns
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate realized volatility
    df['realized_volatility'] = df['returns'].rolling(window=30).std() * np.sqrt(252)
    
    # Calculate high-low spread (normalized)
    df['high_low_spread'] = np.log(df['high'] / df['low'])
    
    # Normalize volume
    df['volume_normalized'] = (df['volume'] - df['volume'].rolling(30).mean()) / df['volume'].rolling(30).std()
    
    # Download VIX
    vix = download_vix_data(start_date, end_date)
    if vix is not None:
        vix_df = pd.DataFrame({'date': vix.index, 'vix': vix.values})
        df = df.merge(vix_df, on='date', how='left')
        df['vix'] = df['vix'].fillna(method='ffill')
    else:
        # Use implied volatility proxy
        df['vix'] = df['realized_volatility'] * 100 * 1.2
    
    # Generate GPR index
    gpr = generate_synthetic_gpr(df['date'])
    df['gpr_index'] = gpr.values
    
    # Create lagged features
    for lag in [1, 5, 22]:
        df[f'rv_lag{lag}'] = df['realized_volatility'].shift(lag)
        df[f'returns_lag{lag}'] = df['returns'].shift(lag)
    
    # Drop NaN rows
    df = df.dropna().reset_index(drop=True)
    
    print(f"\nFinal dataset: {len(df)} samples with {len(df.columns)} features")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(df[['returns', 'realized_volatility', 'vix', 'gpr_index']].describe())
    
    return df


def download_multiple_assets(start_date='2018-01-01', end_date='2024-12-31'):
    """
    Download and process data for multiple asset classes.
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str
        End date
    
    Returns:
    --------
    datasets : dict
        Dictionary of dataframes for each asset
    """
    assets = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ-100 ETF',
        'GLD': 'Gold ETF',
        'USO': 'Oil ETF',
        'UUP': 'US Dollar ETF'
    }
    
    print(f"\n{'='*70}")
    print("DOWNLOADING MULTI-ASSET DATA")
    print(f"{'='*70}")
    print(f"Assets: {list(assets.keys())}")
    
    datasets = {}
    
    for ticker, name in assets.items():
        print(f"\n\nProcessing {name} ({ticker})...")
        print("-" * 70)
        try:
            df = build_complete_dataset(ticker, start_date, end_date)
            df['asset'] = ticker
            df['asset_name'] = name
            datasets[ticker] = df
            print(f"✓ {name} dataset ready")
        except Exception as e:
            print(f"✗ Error processing {ticker}: {str(e)}")
    
    return datasets


if __name__ == "__main__":
    # Build dataset for SPY (S&P 500)
    df_spy = build_complete_dataset(
        ticker='SPY',
        start_date='2018-01-01',
        end_date='2024-12-31'
    )
    
    # Save to CSV
    output_path = '../data/real_data_SPY.csv'
    df_spy.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Dataset saved to: {output_path}")
    print(f"{'='*70}")
    
    # Optionally download multi-asset data
    # datasets = download_multiple_assets()
    # for ticker, df in datasets.items():
    #     df.to_csv(f'../data/real_data_{ticker}.csv', index=False)

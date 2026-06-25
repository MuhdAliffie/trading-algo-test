"""
Data loading module - downloads and cleans OHLCV data.
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def create_yfinance_session():
    """Create a yfinance session that works behind local SSL interception."""
    try:
        import curl_cffi.requests as requests

        return requests.Session(impersonate="chrome", verify=False)
    except Exception as exc:
        logger.warning(f"Could not create curl_cffi session; using yfinance default session: {exc}")
        return None


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download OHLCV data for a ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Adj Close, Volume
    """
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
            threads=False,
            session=create_yfinance_session(),
        )
        
        if df.empty:
            logger.warning(f"No data downloaded for {ticker}")
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to make Date a column
        df.reset_index(inplace=True)

        if 'Adj Close' not in df.columns and 'Close' in df.columns:
            df['Adj Close'] = df['Close']

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"{ticker}: Missing required columns: {missing_columns}")
            return None

        df = df[required_columns]
        
        # Ensure correct data types
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Clean missing values
        df = clean_missing_values(df)
        
        logger.info(f"{ticker}: Downloaded {len(df)} rows from {start_date} to {end_date}")
        return df
    
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        return None


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean missing values in OHLCV data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Cleaned DataFrame
    """
    # Remove rows with NaN in key columns
    df = df.dropna(subset=['Close', 'Volume'])
    
    # Forward fill other columns
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Adj Close'] = df['Adj Close'].fillna(df['Close'])
    
    # Ensure High >= Close >= Low
    df['High'] = df[['High', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Close']].min(axis=1)
    
    return df.reset_index(drop=True)


def validate_data(df: pd.DataFrame, ticker: str, min_points: int = 252) -> bool:
    """
    Validate if data meets minimum requirements.
    
    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker (for logging)
        min_points: Minimum number of data points required
    
    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"{ticker}: No data available")
        return False
    
    if len(df) < min_points:
        logger.warning(f"{ticker}: Insufficient data ({len(df)} < {min_points} required)")
        return False
    
    # Check for too many NaNs
    if df.isnull().sum().sum() > len(df) * 0.1:  # Allow max 10% NaN
        logger.warning(f"{ticker}: Too many missing values")
        return False
    
    return True


def get_stock_data(ticker: str, start_date: str, end_date: str, min_points: int = 252) -> pd.DataFrame:
    """
    Get validated stock data.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        min_points: Minimum number of data points required
    
    Returns:
        DataFrame with OHLCV data or None if validation fails
    """
    df = download_data(ticker, start_date, end_date)
    
    if validate_data(df, ticker, min_points):
        return df
    else:
        return None


def add_returns(df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Add daily returns column to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        use_adjusted: Use Adj Close for returns if True, else Close
    
    Returns:
        DataFrame with added 'Returns' column
    """
    price_col = 'Adj Close' if use_adjusted else 'Close'
    df['Returns'] = df[price_col].pct_change()
    return df


def add_log_returns(df: pd.DataFrame, use_adjusted: bool = True) -> pd.DataFrame:
    """
    Add log returns column to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        use_adjusted: Use Adj Close for returns if True, else Close
    
    Returns:
        DataFrame with added 'Log_Returns' column
    """
    price_col = 'Adj Close' if use_adjusted else 'Close'
    df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
    return df

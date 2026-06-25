"""
Configuration settings for the trading strategy backtesting system.
"""

from datetime import datetime, timedelta

# Backtest Period
BACKTEST_START_DATE = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")  # 5 years ago
BACKTEST_END_DATE = datetime.now().strftime("%Y-%m-%d")

# Market Regions
REGIONS = {
    "Asia": {
        "universe_file": "data/universe/asia_top100.csv",
        "description": "Asia top 100 stocks by market cap"
    },
    "Europe": {
        "universe_file": "data/universe/europe_top100.csv",
        "description": "Europe top 100 stocks by market cap"
    },
    "US": {
        "universe_file": "data/universe/us_top100.csv",
        "description": "US top 100 stocks by market cap"
    },
    "Malaysia": {
        "universe_file": "data/universe/malaysia_top100.csv",
        "description": "Malaysia top 100 Bursa Malaysia stocks"
    }
}

# Strategy List
STRATEGIES = [
    "RSI",
    "ADX_DI",
    "ATR_Breakout",
    "SMA_Crossover",
    "EMA_Crossover",
    "MACD",
    "Bollinger_Mean_Reversion",
    "Stochastic",
    "CCI",
    "MFI",
    "OBV",
    "VWAP",
    "Supertrend",
    "Parabolic_SAR",
    "Williams_R",
    "ROC_Momentum",
    "Donchian_Channel",
    "Keltner_Channel",
]

# Backtesting Parameters
INITIAL_CAPITAL = 10000  # Per stock
COMMISSION = 0.001  # 0.1% per trade
MAX_POSITIONS = 1  # One position at a time
USE_ADJUSTED_CLOSE = True

# Data Parameters
DATA_SOURCE = "yfinance"  # Can extend to other sources
OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
MIN_DATA_POINTS = 252  # At least 1 year of trading days

# Technical Indicators - Volume Strategy
VOLUME_MA_WINDOW = 20
VOLUME_SPIKE_THRESHOLD = 1.5

# Technical Indicators - MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Technical Indicators - LSTM-Inspired Bollinger Band Mean Reversion
BB_WINDOW = 20
BB_STD = 2
LSTM_FEATURES = ["bb_position", "z_score", "volatility", "momentum", "ma_distance"]

# Technical Indicators - Ichimoku Cloud
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
CHIKOU_PERIOD = 26
SENKOU_DISPLACEMENT = 26

# Technical Indicators - Supertrend
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

# Technical Indicators - Additional analysis indicators
ADX_PERIOD = 14
ATR_PERIOD = 14
CCI_PERIOD = 20
FIBONACCI_WINDOW = 60
PSAR_STEP = 0.02
PSAR_MAX_STEP = 0.2
RSI_PERIOD = 14
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
SMA_FAST = 50
SMA_SLOW = 200
EMA_FAST = 12
EMA_SLOW = 26
MFI_PERIOD = 14
OBV_SMA_WINDOW = 20
VWAP_VOLUME_SMA_WINDOW = 20
WILLIAMS_R_PERIOD = 14
ROC_PERIOD = 12
DONCHIAN_HIGH_WINDOW = 20
DONCHIAN_LOW_WINDOW = 10
KELTNER_EMA_PERIOD = 20
KELTNER_ATR_PERIOD = 10
KELTNER_MULTIPLIER = 2.0
ATR_BREAKOUT_HIGH_WINDOW = 20
ATR_BREAKOUT_MULTIPLIER = 0.5
ATR_TRAILING_STOP_MULTIPLIER = 2.0

# Parallel Processing
MAX_WORKERS = 4  # Number of parallel workers
ENABLE_PROGRESS_BAR = True

# Output Paths
OUTPUT_DIR = "results"
CSV_OUTPUT_DIR = f"{OUTPUT_DIR}/combined"
CHART_OUTPUT_DIR = f"{OUTPUT_DIR}/charts"
LOGS_DIR = f"{OUTPUT_DIR}/logs"

# Reporting Parameters
TOP_BOTTOM_N = 10  # Top/bottom N stocks to display
TOP_CHART_STOCKS = 5  # Top/bottom N stocks to chart
ENABLE_INDIVIDUAL_CHARTS = False  # Set to True to generate individual stock charts

# Risk Management
STOP_LOSS_PERCENT = None  # Set to a % value to enable, e.g., 0.05 for 5%
TAKE_PROFIT_PERCENT = None  # Set to a % value to enable

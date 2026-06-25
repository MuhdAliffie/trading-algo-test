"""
Technical indicators and strategy signals.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# VOLUME STRATEGY
# ============================================================================

def calculate_volume_strategy(df: pd.DataFrame, ma_window: int = 20, 
                             spike_threshold: float = 1.5) -> pd.DataFrame:
    """
    Volume-based strategy.
    
    Logic:
    - Buy: Volume significantly above MA and price closes higher
    - Sell: Volume spike confirms weakness or momentum reverses
    
    Args:
        df: DataFrame with OHLCV data
        ma_window: Window for volume moving average
        spike_threshold: Volume spike multiplier (1.5 = 50% above MA)
    
    Returns:
        DataFrame with added volume signals
    """
    df = df.copy()
    
    # Calculate volume moving average
    df['Volume_MA'] = df['Volume'].rolling(window=ma_window).mean()
    
    # Volume ratio
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Price change
    df['Price_Change'] = df['Close'].pct_change()
    
    # Generate signals
    df['Volume_Signal'] = 0  # 0 = HOLD, 1 = BUY, -1 = SELL
    
    # Buy condition: High volume + positive price change
    buy_condition = (df['Volume_Ratio'] > spike_threshold) & (df['Price_Change'] > 0)
    df.loc[buy_condition, 'Volume_Signal'] = 1
    
    # Sell condition: Drop in volume or negative price movement
    sell_condition = (df['Volume_Ratio'] < 1.0) & (df['Price_Change'] < -0.01)
    df.loc[sell_condition, 'Volume_Signal'] = -1
    
    return df


# ============================================================================
# MACD STRATEGY
# ============================================================================

def calculate_ema(data: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=span, adjust=False).mean()


def calculate_macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                           signal: int = 9) -> pd.DataFrame:
    """
    MACD strategy.
    
    Logic:
    - Buy: MACD crosses above signal line
    - Sell: MACD crosses below signal line
    
    Args:
        df: DataFrame with OHLCV data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal EMA period
    
    Returns:
        DataFrame with added MACD signals
    """
    df = df.copy()
    
    # Calculate EMAs
    ema_fast = calculate_ema(df['Close'], fast)
    ema_slow = calculate_ema(df['Close'], slow)
    
    # MACD line and signal line
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = calculate_ema(df['MACD'], signal)
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Generate signals (avoid look-ahead bias by using previous day's values)
    df['MACD_Prev'] = df['MACD'].shift(1)
    df['MACD_Signal_Prev'] = df['MACD_Signal'].shift(1)
    
    df['MACD_Signal_Output'] = 0  # 0 = HOLD, 1 = BUY, -1 = SELL
    
    # Buy: MACD crosses above signal line
    buy_condition = (df['MACD_Prev'] < df['MACD_Signal_Prev']) & \
                   (df['MACD'] > df['MACD_Signal'])
    df.loc[buy_condition, 'MACD_Signal_Output'] = 1
    
    # Sell: MACD crosses below signal line
    sell_condition = (df['MACD_Prev'] > df['MACD_Signal_Prev']) & \
                    (df['MACD'] < df['MACD_Signal'])
    df.loc[sell_condition, 'MACD_Signal_Output'] = -1
    
    return df


# ============================================================================
# LSTM-INSPIRED BOLLINGER BAND MEAN REVERSION STRATEGY
# ============================================================================

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, 
                             num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    df['BB_MA'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_MA'] + (df['BB_Std'] * num_std)
    df['BB_Lower'] = df['BB_MA'] - (df['BB_Std'] * num_std)
    df['Bollinger_Middle'] = df['BB_MA']
    df['Bollinger_Upper'] = df['BB_Upper']
    df['Bollinger_Lower'] = df['BB_Lower']
    return df


def calculate_lstm_bollinger_strategy(df: pd.DataFrame, window: int = 20, 
                                     num_std: float = 2.0) -> pd.DataFrame:
    """
    LSTM-Inspired Bollinger Band Mean Reversion Strategy.
    
    This strategy uses historical features to create an "LSTM-inspired" score:
    - Bollinger Band position (Z-score)
    - Price volatility
    - Short-term momentum
    - Distance from MA
    
    Logic:
    - Buy: Price near/below lower BB + high mean reversion score
    - Sell: Price reaches middle/upper BB or score weakens
    
    Args:
        df: DataFrame with OHLCV data
        window: Bollinger Band window
        num_std: Bollinger Band standard deviations
    
    Returns:
        DataFrame with added LSTM-inspired signals
    """
    df = df.copy()
    
    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df, window, num_std)
    
    # Feature 1: Bollinger Band position (Z-score)
    # 0 = lower band, 1 = upper band
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Position'] = df['BB_Position'].clip(0, 1)  # Clip to [0, 1]
    
    # Feature 2: Price Z-score
    df['Price_Z_Score'] = (df['Close'] - df['BB_MA']) / (df['BB_Std'] + 1e-6)
    
    # Feature 3: Rolling volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df['Volatility'] = df['Volatility'].fillna(df['Volatility'].mean())
    
    # Feature 4: Short-term momentum (10-day ROC)
    df['Momentum'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
    df['Momentum'] = df['Momentum'].fillna(0)
    
    # Feature 5: Distance from MA (normalized)
    df['MA_Distance'] = (df['Close'] - df['BB_MA']) / (df['BB_MA'] + 1e-6)
    
    # Combine features into mean reversion score
    # Lower score = more likely to revert up (good for buying)
    # Higher score = stock is extended up (good for selling)
    df['MR_Score'] = (
        (1 - df['BB_Position']) * 0.4 +  # Prefer lower BB position
        (-df['Price_Z_Score']) * 0.2 +   # Prefer negative Z-score (below MA)
        (1 - (df['Volatility'] / (df['Volatility'].max() + 1e-6))) * 0.2 +  # Normalize volatility
        ((df['Momentum'] + 1) / 2) * 0.1 +  # Include momentum
        ((df['MA_Distance'] + 1) / 2) * 0.1   # Normalize MA distance
    )
    
    # Generate signals
    df['LSTM_BB_Signal'] = 0  # 0 = HOLD, 1 = BUY, -1 = SELL
    
    # Buy: Near lower BB with good mean reversion score
    buy_condition = (df['Close'] < df['BB_MA']) & (df['MR_Score'] > 0.6)
    df.loc[buy_condition, 'LSTM_BB_Signal'] = 1
    
    # Sell: Reached upper band or score weakens
    sell_condition = (df['Close'] > df['BB_MA']) & (df['MR_Score'] < 0.4)
    df.loc[sell_condition, 'LSTM_BB_Signal'] = -1
    
    return df


# ============================================================================
# ICHIMOKU CLOUD STRATEGY
# ============================================================================

def calculate_ichimoku_strategy(df: pd.DataFrame, 
                               tenkan_period: int = 9,
                               kijun_period: int = 26,
                               senkou_b_period: int = 52,
                               chikou_period: int = 26,
                               senkou_displacement: int = 26) -> pd.DataFrame:
    """
    Ichimoku Cloud strategy.
    
    Components:
    - Tenkan-sen: 9-period high-low average
    - Kijun-sen: 26-period high-low average
    - Senkou Span A: Average of Tenkan and Kijun, shifted forward 26 periods
    - Senkou Span B: 52-period high-low average, shifted forward 26 periods
    - Chikou Span: Close shifted back 26 periods
    
    Logic:
    - Buy: Price above cloud + Tenkan crosses above Kijun
    - Sell: Price below cloud OR Tenkan crosses below Kijun
    
    Args:
        df: DataFrame with OHLCV data
        tenkan_period: Tenkan-sen period
        kijun_period: Kijun-sen period
        senkou_b_period: Senkou Span B period
        chikou_period: Chikou span period
        senkou_displacement: Forward displacement for Senkou spans
    
    Returns:
        DataFrame with added Ichimoku signals
    """
    df = df.copy()
    
    # Tenkan-sen: 9-period high-low average
    high_9 = df['High'].rolling(window=tenkan_period).max()
    low_9 = df['Low'].rolling(window=tenkan_period).min()
    df['Tenkan'] = (high_9 + low_9) / 2
    
    # Kijun-sen: 26-period high-low average
    high_26 = df['High'].rolling(window=kijun_period).max()
    low_26 = df['Low'].rolling(window=kijun_period).min()
    df['Kijun'] = (high_26 + low_26) / 2
    
    # Senkou Span A: Average of Tenkan and Kijun, shifted forward
    senkou_a = (df['Tenkan'] + df['Kijun']) / 2
    df['Senkou_A'] = senkou_a.shift(senkou_displacement)
    
    # Senkou Span B: 52-period high-low average, shifted forward
    high_52 = df['High'].rolling(window=senkou_b_period).max()
    low_52 = df['Low'].rolling(window=senkou_b_period).min()
    senkou_b = (high_52 + low_52) / 2
    df['Senkou_B'] = senkou_b.shift(senkou_displacement)
    
    # Chikou Span: Close shifted back
    df['Chikou'] = df['Close'].shift(-chikou_period)
    
    # Define cloud
    df['Cloud_Top'] = df[['Senkou_A', 'Senkou_B']].max(axis=1)
    df['Cloud_Bottom'] = df[['Senkou_A', 'Senkou_B']].min(axis=1)
    
    # Generate signals (use previous day to avoid look-ahead bias)
    df['Tenkan_Prev'] = df['Tenkan'].shift(1)
    df['Kijun_Prev'] = df['Kijun'].shift(1)
    
    df['Ichimoku_Signal'] = 0  # 0 = HOLD, 1 = BUY, -1 = SELL
    
    # Buy condition: Price above cloud AND Tenkan crosses above Kijun
    price_above_cloud = df['Close'] > df['Cloud_Top']
    tenkan_cross_above = (df['Tenkan_Prev'] < df['Kijun_Prev']) & \
                        (df['Tenkan'] > df['Kijun'])
    buy_condition = price_above_cloud & tenkan_cross_above
    df.loc[buy_condition, 'Ichimoku_Signal'] = 1
    
    # Sell condition: Price below cloud OR Tenkan crosses below Kijun
    price_below_cloud = df['Close'] < df['Cloud_Bottom']
    tenkan_cross_below = (df['Tenkan_Prev'] > df['Kijun_Prev']) & \
                        (df['Tenkan'] < df['Kijun'])
    sell_condition = price_below_cloud | tenkan_cross_below
    df.loc[sell_condition, 'Ichimoku_Signal'] = -1
    
    return df


# ============================================================================
# SUPERTREND STRATEGY
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate Average True Range."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# ============================================================================
# ADDITIONAL TECHNICAL INDICATORS
# ============================================================================

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index with +DI and -DI."""
    df = df.copy()

    high = df['High']
    low = df['Low']
    close = df['Close']

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index
    )

    true_range = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    true_range_sum = true_range.rolling(window=period).sum()
    plus_di = 100 * plus_dm.rolling(window=period).sum() / true_range_sum.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window=period).sum() / true_range_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)

    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['ADX'] = dx.rolling(window=period).mean()
    return df


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Calculate Commodity Channel Index."""
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    typical_price_ma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda values: np.mean(np.abs(values - values.mean())),
        raw=True
    )

    df['CCI'] = (typical_price - typical_price_ma) / (0.015 * mean_deviation.replace(0, np.nan))
    return df


def calculate_fibonacci_retracement_levels(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Calculate rolling Fibonacci retracement levels from recent high/low range."""
    df = df.copy()
    rolling_high = df['High'].rolling(window=window).max()
    rolling_low = df['Low'].rolling(window=window).min()
    price_range = rolling_high - rolling_low

    df['Fib_0'] = rolling_high
    df['Fib_23_6'] = rolling_high - (price_range * 0.236)
    df['Fib_38_2'] = rolling_high - (price_range * 0.382)
    df['Fib_50'] = rolling_high - (price_range * 0.5)
    df['Fib_61_8'] = rolling_high - (price_range * 0.618)
    df['Fib_78_6'] = rolling_high - (price_range * 0.786)
    df['Fib_100'] = rolling_low
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate On-Balance Volume."""
    df = df.copy()
    direction = np.sign(df['Close'].diff()).fillna(0)
    df['OBV'] = (direction * df['Volume']).cumsum()
    return df


def calculate_parabolic_sar(df: pd.DataFrame,
                            step: float = 0.02,
                            max_step: float = 0.2) -> pd.DataFrame:
    """Calculate Parabolic SAR and trend direction."""
    df = df.copy()
    if df.empty:
        df['Parabolic_SAR'] = np.nan
        df['PSAR_Trend'] = np.nan
        return df

    high = df['High'].reset_index(drop=True)
    low = df['Low'].reset_index(drop=True)
    psar = pd.Series(index=range(len(df)), dtype='float64')
    trend = pd.Series(index=range(len(df)), dtype='int64')

    is_uptrend = True
    acceleration = step
    extreme_point = high.iloc[0]
    psar.iloc[0] = low.iloc[0]
    trend.iloc[0] = 1

    for i in range(1, len(df)):
        prior_psar = psar.iloc[i - 1]

        if is_uptrend:
            current_psar = prior_psar + acceleration * (extreme_point - prior_psar)
            if i >= 2:
                current_psar = min(current_psar, low.iloc[i - 1], low.iloc[i - 2])
            else:
                current_psar = min(current_psar, low.iloc[i - 1])

            if low.iloc[i] < current_psar:
                is_uptrend = False
                current_psar = extreme_point
                extreme_point = low.iloc[i]
                acceleration = step
            else:
                if high.iloc[i] > extreme_point:
                    extreme_point = high.iloc[i]
                    acceleration = min(acceleration + step, max_step)
        else:
            current_psar = prior_psar + acceleration * (extreme_point - prior_psar)
            if i >= 2:
                current_psar = max(current_psar, high.iloc[i - 1], high.iloc[i - 2])
            else:
                current_psar = max(current_psar, high.iloc[i - 1])

            if high.iloc[i] > current_psar:
                is_uptrend = True
                current_psar = extreme_point
                extreme_point = high.iloc[i]
                acceleration = step
            else:
                if low.iloc[i] < extreme_point:
                    extreme_point = low.iloc[i]
                    acceleration = min(acceleration + step, max_step)

        psar.iloc[i] = current_psar
        trend.iloc[i] = 1 if is_uptrend else -1

    df['Parabolic_SAR'] = psar.to_numpy()
    df['PSAR_Trend'] = trend.to_numpy()
    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Relative Strength Index."""
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    df['RSI'] = rsi
    return df


def calculate_stochastic_oscillator(df: pd.DataFrame,
                                    k_period: int = 14,
                                    d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator %K and %D."""
    df = df.copy()
    lowest_low = df['Low'].rolling(window=k_period).min()
    highest_high = df['High'].rolling(window=k_period).max()
    high_low_range = (highest_high - lowest_low).replace(0, np.nan)

    df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / high_low_range
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
    return df


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Money Flow Index."""
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    price_change = typical_price.diff()

    positive_flow = raw_money_flow.where(price_change > 0, 0.0)
    negative_flow = raw_money_flow.where(price_change < 0, 0.0).abs()

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    mfi = mfi.mask((negative_sum == 0) & (positive_sum > 0), 100)
    mfi = mfi.mask((positive_sum == 0) & (negative_sum > 0), 0)
    mfi = mfi.mask((positive_sum == 0) & (negative_sum == 0), 50)

    df['MFI'] = mfi
    return df


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative VWAP over the available daily data."""
    df = df.copy()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cumulative_volume = df['Volume'].cumsum().replace(0, np.nan)
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / cumulative_volume
    return df


def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Williams %R."""
    df = df.copy()
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    price_range = (highest_high - lowest_low).replace(0, np.nan)
    df['Williams_R'] = -100 * (highest_high - df['Close']) / price_range
    return df


def calculate_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    """Calculate Rate of Change momentum."""
    df = df.copy()
    df['ROC'] = ((df['Close'] / df['Close'].shift(period)) - 1) * 100
    return df


def calculate_sma_ema_columns(df: pd.DataFrame,
                              sma_fast: int = 50,
                              sma_slow: int = 200,
                              ema_fast: int = 12,
                              ema_slow: int = 26) -> pd.DataFrame:
    """Calculate moving average columns used by crossover strategies."""
    df = df.copy()
    df['SMA_50'] = df['Close'].rolling(window=sma_fast).mean()
    df['SMA_200'] = df['Close'].rolling(window=sma_slow).mean()
    df['EMA_12'] = calculate_ema(df['Close'], ema_fast)
    df['EMA_26'] = calculate_ema(df['Close'], ema_slow)
    return df


def calculate_channel_columns(df: pd.DataFrame,
                              donchian_high_window: int = 20,
                              donchian_low_window: int = 10,
                              keltner_ema_period: int = 20,
                              keltner_atr_period: int = 10,
                              keltner_multiplier: float = 2.0) -> pd.DataFrame:
    """Calculate Donchian and Keltner channel columns."""
    df = df.copy()
    df['Donchian_Prev_20_High'] = df['High'].shift(1).rolling(window=donchian_high_window).max()
    df['Donchian_Prev_10_Low'] = df['Low'].shift(1).rolling(window=donchian_low_window).min()
    df['Keltner_Middle'] = calculate_ema(df['Close'], keltner_ema_period)
    df['Keltner_ATR'] = calculate_atr(df, keltner_atr_period)
    df['Keltner_Upper'] = df['Keltner_Middle'] + (keltner_multiplier * df['Keltner_ATR'])
    df['Keltner_Lower'] = df['Keltner_Middle'] - (keltner_multiplier * df['Keltner_ATR'])
    return df


def calculate_technical_indicators(df: pd.DataFrame, strategy_params: dict = None) -> pd.DataFrame:
    """Calculate reusable technical indicator columns for analysis and future strategies."""
    if strategy_params is None:
        strategy_params = {}

    df = df.copy()
    df = calculate_bollinger_bands(
        df,
        window=strategy_params.get('bb_window', 20),
        num_std=strategy_params.get('bb_std', 2.0)
    )
    df['ATR'] = calculate_atr(df, strategy_params.get('atr_period', 14))
    df = calculate_adx(df, period=strategy_params.get('adx_period', 14))
    df = calculate_cci(df, period=strategy_params.get('cci_period', 20))
    df = calculate_fibonacci_retracement_levels(
        df,
        window=strategy_params.get('fibonacci_window', 60)
    )
    df = calculate_obv(df)
    df = calculate_parabolic_sar(
        df,
        step=strategy_params.get('psar_step', 0.02),
        max_step=strategy_params.get('psar_max_step', 0.2)
    )
    df = calculate_rsi(df, period=strategy_params.get('rsi_period', 14))
    df = calculate_stochastic_oscillator(
        df,
        k_period=strategy_params.get('stochastic_k_period', 14),
        d_period=strategy_params.get('stochastic_d_period', 3)
    )
    df = calculate_mfi(df, period=strategy_params.get('mfi_period', 14))
    df = calculate_vwap(df)
    df = calculate_williams_r(df, period=strategy_params.get('williams_r_period', 14))
    df = calculate_roc(df, period=strategy_params.get('roc_period', 12))
    df = calculate_sma_ema_columns(
        df,
        sma_fast=strategy_params.get('sma_fast', 50),
        sma_slow=strategy_params.get('sma_slow', 200),
        ema_fast=strategy_params.get('ema_fast', 12),
        ema_slow=strategy_params.get('ema_slow', 26)
    )
    df['OBV_SMA_20'] = df['OBV'].rolling(window=strategy_params.get('obv_sma_window', 20)).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(window=strategy_params.get('vwap_volume_sma_window', 20)).mean()
    df = calculate_channel_columns(
        df,
        donchian_high_window=strategy_params.get('donchian_high_window', 20),
        donchian_low_window=strategy_params.get('donchian_low_window', 10),
        keltner_ema_period=strategy_params.get('keltner_ema_period', 20),
        keltner_atr_period=strategy_params.get('keltner_atr_period', 10),
        keltner_multiplier=strategy_params.get('keltner_multiplier', 2.0)
    )
    return df


def calculate_supertrend_strategy(df: pd.DataFrame,
                                  atr_period: int = 10,
                                  multiplier: float = 3.0) -> pd.DataFrame:
    """
    Supertrend strategy.

    Logic:
    - Buy: Supertrend flips from downtrend to uptrend
    - Sell: Supertrend flips from uptrend to downtrend

    Args:
        df: DataFrame with OHLCV data
        atr_period: ATR lookback period
        multiplier: ATR multiplier for upper/lower bands

    Returns:
        DataFrame with added Supertrend signals
    """
    df = df.copy()

    df['Supertrend_ATR'] = calculate_atr(df, atr_period)
    hl2 = (df['High'] + df['Low']) / 2
    df['Supertrend_Basic_Upper'] = hl2 + (multiplier * df['Supertrend_ATR'])
    df['Supertrend_Basic_Lower'] = hl2 - (multiplier * df['Supertrend_ATR'])

    final_upper = pd.Series(index=df.index, dtype='float64')
    final_lower = pd.Series(index=df.index, dtype='float64')
    trend = pd.Series(0, index=df.index, dtype='int64')
    supertrend = pd.Series(index=df.index, dtype='float64')

    for i in range(len(df)):
        if pd.isna(df.iloc[i]['Supertrend_ATR']):
            continue

        basic_upper = df.iloc[i]['Supertrend_Basic_Upper']
        basic_lower = df.iloc[i]['Supertrend_Basic_Lower']

        if i == 0 or pd.isna(final_upper.iloc[i - 1]):
            final_upper.iloc[i] = basic_upper
            final_lower.iloc[i] = basic_lower
            trend.iloc[i] = 1 if df.iloc[i]['Close'] >= basic_lower else -1
        else:
            prev_close = df.iloc[i - 1]['Close']
            prev_final_upper = final_upper.iloc[i - 1]
            prev_final_lower = final_lower.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]

            final_upper.iloc[i] = (
                basic_upper
                if basic_upper < prev_final_upper or prev_close > prev_final_upper
                else prev_final_upper
            )
            final_lower.iloc[i] = (
                basic_lower
                if basic_lower > prev_final_lower or prev_close < prev_final_lower
                else prev_final_lower
            )

            if prev_trend == -1 and df.iloc[i]['Close'] > final_upper.iloc[i]:
                trend.iloc[i] = 1
            elif prev_trend == 1 and df.iloc[i]['Close'] < final_lower.iloc[i]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = prev_trend

        supertrend.iloc[i] = final_lower.iloc[i] if trend.iloc[i] == 1 else final_upper.iloc[i]

    df['Supertrend_Upper'] = final_upper
    df['Supertrend_Lower'] = final_lower
    df['Supertrend'] = supertrend
    df['Supertrend_Direction'] = trend
    df['Supertrend_Direction_Prev'] = df['Supertrend_Direction'].shift(1)

    df['Supertrend_Signal'] = 0
    buy_condition = (df['Supertrend_Direction_Prev'] == -1) & (df['Supertrend_Direction'] == 1)
    sell_condition = (df['Supertrend_Direction_Prev'] == 1) & (df['Supertrend_Direction'] == -1)
    df.loc[buy_condition, 'Supertrend_Signal'] = 1
    df.loc[sell_condition, 'Supertrend_Signal'] = -1

    return df


# ============================================================================
# INDICATOR STRATEGIES FROM strategy.txt
# ============================================================================

def crossed_above(series: pd.Series, threshold) -> pd.Series:
    """True when a series crosses above another series or scalar threshold."""
    if isinstance(threshold, pd.Series):
        return (series.shift(1) <= threshold.shift(1)) & (series > threshold)
    return (series.shift(1) <= threshold) & (series > threshold)


def crossed_below(series: pd.Series, threshold) -> pd.Series:
    """True when a series crosses below another series or scalar threshold."""
    if isinstance(threshold, pd.Series):
        return (series.shift(1) >= threshold.shift(1)) & (series < threshold)
    return (series.shift(1) >= threshold) & (series < threshold)


def calculate_rsi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """RSI strategy: oversold recovery entry, overbought or momentum loss exit."""
    df = df.copy()
    df['RSI_Signal'] = 0
    df.loc[crossed_above(df['RSI'], 30), 'RSI_Signal'] = 1
    df.loc[crossed_above(df['RSI'], 70) | crossed_below(df['RSI'], 50), 'RSI_Signal'] = -1
    return df


def calculate_adx_di_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """ADX-DI strategy: directional index crossover confirmed by trend strength."""
    df = df.copy()
    df['ADX_DI_Signal'] = 0
    df.loc[crossed_above(df['Plus_DI'], df['Minus_DI']) & (df['ADX'] > 25), 'ADX_DI_Signal'] = 1
    df.loc[crossed_above(df['Minus_DI'], df['Plus_DI']) | (df['ADX'] < 20), 'ADX_DI_Signal'] = -1
    return df


def calculate_atr_breakout_strategy(df: pd.DataFrame,
                                    high_window: int = 20,
                                    breakout_multiplier: float = 0.5,
                                    trailing_stop_multiplier: float = 2.0) -> pd.DataFrame:
    """ATR breakout strategy with stateful trailing stop."""
    df = df.copy()
    previous_high = df['High'].shift(1).rolling(window=high_window).max()
    breakout_level = previous_high + (breakout_multiplier * df['ATR'])

    signal = pd.Series(0, index=df.index, dtype='int64')
    in_position = False
    highest_close_since_entry = np.nan

    for i in range(len(df)):
        close = df.iloc[i]['Close']
        atr = df.iloc[i]['ATR']
        level = breakout_level.iloc[i]

        if pd.isna(close) or pd.isna(atr):
            continue

        if not in_position:
            if not pd.isna(level) and close > level:
                signal.iloc[i] = 1
                in_position = True
                highest_close_since_entry = close
        else:
            highest_close_since_entry = max(highest_close_since_entry, close)
            trailing_stop = highest_close_since_entry - (trailing_stop_multiplier * atr)
            if close < trailing_stop:
                signal.iloc[i] = -1
                in_position = False
                highest_close_since_entry = np.nan

    df['ATR_Breakout_Level'] = breakout_level
    df['ATR_Breakout_Signal'] = signal
    return df


def calculate_sma_crossover_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """SMA crossover strategy using SMA(50) and SMA(200)."""
    df = df.copy()
    df['SMA_Crossover_Signal'] = 0
    df.loc[crossed_above(df['SMA_50'], df['SMA_200']), 'SMA_Crossover_Signal'] = 1
    df.loc[crossed_below(df['SMA_50'], df['SMA_200']), 'SMA_Crossover_Signal'] = -1
    return df


def calculate_ema_crossover_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """EMA crossover strategy using EMA(12) and EMA(26)."""
    df = df.copy()
    df['EMA_Crossover_Signal'] = 0
    df.loc[crossed_above(df['EMA_12'], df['EMA_26']), 'EMA_Crossover_Signal'] = 1
    df.loc[crossed_below(df['EMA_12'], df['EMA_26']), 'EMA_Crossover_Signal'] = -1
    return df


def calculate_bollinger_mean_reversion_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Bollinger Band mean reversion strategy from strategy.txt."""
    df = df.copy()
    df['Bollinger_Mean_Reversion_Signal'] = 0
    buy_condition = (df['Close'].shift(1) < df['Bollinger_Lower'].shift(1)) & (df['Close'] > df['Bollinger_Lower'])
    sell_condition = (df['Close'] >= df['Bollinger_Middle']) | (df['Close'] >= df['Bollinger_Upper'])
    df.loc[buy_condition, 'Bollinger_Mean_Reversion_Signal'] = 1
    df.loc[sell_condition, 'Bollinger_Mean_Reversion_Signal'] = -1
    return df


def calculate_stochastic_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Stochastic oscillator strategy."""
    df = df.copy()
    df['Stochastic_Signal'] = 0
    buy_condition = crossed_above(df['Stoch_K'], df['Stoch_D']) & (df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)
    sell_condition = crossed_below(df['Stoch_K'], df['Stoch_D']) & (df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)
    df.loc[buy_condition, 'Stochastic_Signal'] = 1
    df.loc[sell_condition, 'Stochastic_Signal'] = -1
    return df


def calculate_cci_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """CCI strategy."""
    df = df.copy()
    df['CCI_Signal'] = 0
    df.loc[crossed_above(df['CCI'], -100), 'CCI_Signal'] = 1
    df.loc[crossed_below(df['CCI'], 100) | crossed_below(df['CCI'], 0), 'CCI_Signal'] = -1
    return df


def calculate_mfi_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """MFI strategy."""
    df = df.copy()
    df['MFI_Signal'] = 0
    df.loc[crossed_above(df['MFI'], 20), 'MFI_Signal'] = 1
    df.loc[crossed_below(df['MFI'], 80), 'MFI_Signal'] = -1
    return df


def calculate_obv_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """OBV strategy using OBV and OBV SMA(20)."""
    df = df.copy()
    df['OBV_Signal'] = 0
    df.loc[crossed_above(df['OBV'], df['OBV_SMA_20']), 'OBV_Signal'] = 1
    df.loc[crossed_below(df['OBV'], df['OBV_SMA_20']), 'OBV_Signal'] = -1
    return df


def calculate_vwap_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """VWAP strategy with volume confirmation for entries."""
    df = df.copy()
    df['VWAP_Signal'] = 0
    buy_condition = crossed_above(df['Close'], df['VWAP']) & (df['Volume'] > df['Volume_SMA_20'])
    sell_condition = crossed_below(df['Close'], df['VWAP'])
    df.loc[buy_condition, 'VWAP_Signal'] = 1
    df.loc[sell_condition, 'VWAP_Signal'] = -1
    return df


def calculate_parabolic_sar_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Parabolic SAR strategy."""
    df = df.copy()
    df['Parabolic_SAR_Signal'] = 0
    df.loc[crossed_above(df['Close'], df['Parabolic_SAR']), 'Parabolic_SAR_Signal'] = 1
    df.loc[crossed_below(df['Close'], df['Parabolic_SAR']), 'Parabolic_SAR_Signal'] = -1
    return df


def calculate_williams_r_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Williams %R strategy."""
    df = df.copy()
    df['Williams_R_Signal'] = 0
    df.loc[crossed_above(df['Williams_R'], -80), 'Williams_R_Signal'] = 1
    df.loc[crossed_below(df['Williams_R'], -20), 'Williams_R_Signal'] = -1
    return df


def calculate_roc_momentum_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """ROC momentum strategy."""
    df = df.copy()
    df['ROC_Momentum_Signal'] = 0
    df.loc[crossed_above(df['ROC'], 0), 'ROC_Momentum_Signal'] = 1
    df.loc[crossed_below(df['ROC'], 0), 'ROC_Momentum_Signal'] = -1
    return df


def calculate_donchian_channel_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Donchian channel breakout strategy."""
    df = df.copy()
    df['Donchian_Channel_Signal'] = 0
    df.loc[df['Close'] > df['Donchian_Prev_20_High'], 'Donchian_Channel_Signal'] = 1
    df.loc[df['Close'] < df['Donchian_Prev_10_Low'], 'Donchian_Channel_Signal'] = -1
    return df


def calculate_keltner_channel_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Keltner channel breakout strategy."""
    df = df.copy()
    df['Keltner_Channel_Signal'] = 0
    df.loc[crossed_above(df['Close'], df['Keltner_Upper']), 'Keltner_Channel_Signal'] = 1
    df.loc[df['Close'] < df['Keltner_Middle'], 'Keltner_Channel_Signal'] = -1
    return df


# ============================================================================
# UNIFIED STRATEGY CALCULATION
# ============================================================================

def calculate_all_strategies(df: pd.DataFrame, strategy_params: dict = None) -> pd.DataFrame:
    """
    Calculate all strategy signals.
    
    Args:
        df: DataFrame with OHLCV data
        strategy_params: Dict with strategy parameters (optional)
    
    Returns:
        DataFrame with all strategy signals
    """
    if strategy_params is None:
        strategy_params = {}
    
    df = calculate_technical_indicators(df, strategy_params)

    # Existing strategies kept available for compatibility and individual charts.
    df = calculate_volume_strategy(df, 
                                  ma_window=strategy_params.get('volume_ma_window', 20),
                                  spike_threshold=strategy_params.get('volume_spike_threshold', 1.5))
    
    df = calculate_macd_strategy(df,
                                fast=strategy_params.get('macd_fast', 12),
                                slow=strategy_params.get('macd_slow', 26),
                                signal=strategy_params.get('macd_signal', 9))
    
    df = calculate_lstm_bollinger_strategy(df,
                                          window=strategy_params.get('bb_window', 20),
                                          num_std=strategy_params.get('bb_std', 2.0))
    
    df = calculate_ichimoku_strategy(df,
                                    tenkan_period=strategy_params.get('tenkan_period', 9),
                                    kijun_period=strategy_params.get('kijun_period', 26),
                                    senkou_b_period=strategy_params.get('senkou_b_period', 52),
                                    chikou_period=strategy_params.get('chikou_period', 26),
                                    senkou_displacement=strategy_params.get('senkou_displacement', 26))

    df = calculate_supertrend_strategy(df,
                                      atr_period=strategy_params.get('supertrend_atr_period', 10),
                                      multiplier=strategy_params.get('supertrend_multiplier', 3.0))

    # Backtestable strategies requested in strategy.txt.
    df = calculate_rsi_strategy(df)
    df = calculate_adx_di_strategy(df)
    df = calculate_atr_breakout_strategy(
        df,
        high_window=strategy_params.get('atr_breakout_high_window', 20),
        breakout_multiplier=strategy_params.get('atr_breakout_multiplier', 0.5),
        trailing_stop_multiplier=strategy_params.get('atr_trailing_stop_multiplier', 2.0)
    )
    df = calculate_sma_crossover_strategy(df)
    df = calculate_ema_crossover_strategy(df)
    df = calculate_bollinger_mean_reversion_strategy(df)
    df = calculate_stochastic_strategy(df)
    df = calculate_cci_strategy(df)
    df = calculate_mfi_strategy(df)
    df = calculate_obv_strategy(df)
    df = calculate_vwap_strategy(df)
    df = calculate_parabolic_sar_strategy(df)
    df = calculate_williams_r_strategy(df)
    df = calculate_roc_momentum_strategy(df)
    df = calculate_donchian_channel_strategy(df)
    df = calculate_keltner_channel_strategy(df)
    
    return df

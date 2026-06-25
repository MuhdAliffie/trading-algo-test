"""
Backtesting engine - simulates trades and calculates performance.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_date, entry_price, strategy_name, ticker):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = None
        self.exit_price = None
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.shares = 0
        self.return_pct = 0
        self.days_held = 0
        self.exit_reason = None
    
    def close(self, exit_date, exit_price, exit_reason="Signal"):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.return_pct = (exit_price - self.entry_price) / self.entry_price
        self.days_held = (exit_date - self.entry_date).days
        self.exit_reason = exit_reason
    
    def to_dict(self):
        """Convert trade to dictionary."""
        return {
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'return_pct': self.return_pct,
            'days_held': self.days_held,
            'exit_reason': self.exit_reason,
            'strategy': self.strategy_name,
            'ticker': self.ticker
        }


class BacktestEngine:
    """Backtesting engine for a single stock and strategy."""
    
    def __init__(self, ticker: str, strategy_name: str, initial_capital: float = 10000,
                 commission: float = 0.001, use_adjusted: bool = True):
        """
        Initialize backtesting engine.
        
        Args:
            ticker: Stock ticker
            strategy_name: Name of strategy to backtest
            initial_capital: Initial capital per stock
            commission: Commission as decimal (0.001 = 0.1%)
            use_adjusted: Use adjusted close for calculations
        """
        self.ticker = ticker
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital
        self.commission = commission
        self.use_adjusted = use_adjusted
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = 0
        self.current_position = None  # Current open trade
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        
    def get_signal_column(self) -> str:
        """Get the signal column name for this strategy."""
        signal_map = {
            'Volume': 'Volume_Signal',
            'MACD': 'MACD_Signal_Output',
            'LSTM_Bollinger': 'LSTM_BB_Signal',
            'Ichimoku': 'Ichimoku_Signal',
            'Supertrend': 'Supertrend_Signal',
            'RSI': 'RSI_Signal',
            'ADX_DI': 'ADX_DI_Signal',
            'ATR_Breakout': 'ATR_Breakout_Signal',
            'SMA_Crossover': 'SMA_Crossover_Signal',
            'EMA_Crossover': 'EMA_Crossover_Signal',
            'Bollinger_Mean_Reversion': 'Bollinger_Mean_Reversion_Signal',
            'Stochastic': 'Stochastic_Signal',
            'CCI': 'CCI_Signal',
            'MFI': 'MFI_Signal',
            'OBV': 'OBV_Signal',
            'VWAP': 'VWAP_Signal',
            'Parabolic_SAR': 'Parabolic_SAR_Signal',
            'Williams_R': 'Williams_R_Signal',
            'ROC_Momentum': 'ROC_Momentum_Signal',
            'Donchian_Channel': 'Donchian_Channel_Signal',
            'Keltner_Channel': 'Keltner_Channel_Signal'
        }
        return signal_map.get(self.strategy_name, 'Signal')
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run backtest on the dataframe.
        
        Args:
            df: DataFrame with OHLCV data and signals
        
        Returns:
            Dictionary with backtest results
        """
        mark_price_col = 'Adj Close' if self.use_adjusted else 'Close'
        execution_price_col = 'Open'
        signal_col = self.get_signal_column()
        
        # Initialize tracking
        self.cash = self.initial_capital
        self.shares = 0
        self.current_position = None
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.signals = []
        
        # Iterate through each day
        for idx, row in df.iterrows():
            date = row['Date']
            mark_price = row[mark_price_col]
            execution_price = row[execution_price_col]
            prior_signal = df.iloc[idx - 1].get(signal_col, 0) if idx > 0 else 0

            if idx > 0 and not pd.isna(execution_price) and execution_price > 0:
                # Signals are generated after market close and executed at the next trading day's open.
                if prior_signal == 1 and self.shares == 0:  # BUY signal and no position
                    commission_amount = self.cash * self.commission
                    available_cash = self.cash - commission_amount
                    self.shares = available_cash / execution_price
                    self.cash = 0
                    self.current_position = Trade(date, execution_price, self.strategy_name, self.ticker)
                    logger.debug(f"{self.ticker} {self.strategy_name}: BUY on {date} at {execution_price:.2f}")

                elif prior_signal == -1 and self.shares > 0:  # SELL signal and holding position
                    proceeds = self.shares * execution_price
                    commission_amount = proceeds * self.commission
                    self.cash = proceeds - commission_amount

                    self.current_position.close(date, execution_price, exit_reason="Signal")
                    self.trades.append(self.current_position)
                    self.current_position = None
                    self.shares = 0
                    logger.debug(f"{self.ticker} {self.strategy_name}: SELL on {date} at {execution_price:.2f}")

            # Store close-of-day portfolio value after any next-open execution.
            portfolio_value = self.cash + (self.shares * mark_price)
            self.portfolio_values.append(portfolio_value)
            self.dates.append(date)
            self.signals.append(prior_signal)
        
        # Calculate final portfolio value
        final_portfolio_value = self.cash + (self.shares * df.iloc[-1][mark_price_col])
        
        # Calculate metrics
        results = self._calculate_metrics(df, mark_price_col)
        
        return results
    
    def _calculate_metrics(self, df: pd.DataFrame, price_col: str = 'Close') -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            df: Original dataframe with OHLCV data
            price_col: Price column to use
        
        Returns:
            Dictionary with metrics
        """
        # Calculate portfolio values
        final_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        total_return = (final_portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate buy-and-hold return
        start_price = df.iloc[0][price_col]
        end_price = df.iloc[-1][price_col]
        buy_hold_return = (end_price - start_price) / start_price
        
        # Calculate dates
        start_date = df.iloc[0]['Date']
        end_date = df.iloc[-1]['Date']
        years = (end_date - start_date).days / 365.25
        
        # Annualized return
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = total_return
        
        # Calculate Sharpe ratio
        if len(df) > 1:
            returns = df[price_col].pct_change()
            daily_std = returns.std()
            sharpe_ratio = (annualized_return - 0.02) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown()
        
        # Trade statistics
        num_trades = len(self.trades)
        
        if num_trades > 0:
            trade_returns = [t.return_pct for t in self.trades]
            avg_trade_return = np.mean(trade_returns)
            best_trade = np.max(trade_returns)
            worst_trade = np.min(trade_returns)
            win_rate = sum(1 for r in trade_returns if r > 0) / num_trades
            
            # Profit factor
            wins = sum(r for r in trade_returns if r > 0)
            losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = wins / losses if losses > 0 else wins if wins > 0 else 0
        else:
            avg_trade_return = 0
            best_trade = 0
            worst_trade = 0
            win_rate = 0
            profit_factor = 0
        
        # Exposure time (% of time in position)
        if len(self.trades) > 0:
            total_days_held = sum(t.days_held for t in self.trades)
            total_days = (end_date - start_date).days
            exposure_time = total_days_held / total_days if total_days > 0 else 0
        else:
            exposure_time = 0
        
        # Latest signal
        signal_col = self.get_signal_column()
        latest_signal = 0  # HOLD
        latest_signal_date = end_date
        if self.shares > 0 and self.current_position is not None:
            latest_signal = 1
            latest_signal_date = self.current_position.entry_date
        
        if not df.empty:
            if latest_signal != 1:
                for idx in range(len(df) - 1, -1, -1):
                    sig = df.iloc[idx].get(signal_col, 0)
                    if sig != 0:
                        latest_signal = sig
                        latest_signal_date = df.iloc[idx]['Date']
                        break
        
        signal_map = {1: 'BUY', 0: 'HOLD', -1: 'SELL'}
        
        return {
            'ticker': self.ticker,
            'strategy': self.strategy_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'avg_trade_return': avg_trade_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'exposure_time': exposure_time,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'latest_signal': signal_map.get(latest_signal, 'HOLD'),
            'latest_signal_date': latest_signal_date if 'latest_signal_date' in locals() else end_date,
            'final_price': end_price,
            'final_portfolio_value': final_portfolio_value,
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_values': self.portfolio_values,
            'dates': self.dates,
            'initial_capital': self.initial_capital
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.portfolio_values:
            return 0
        
        running_max = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return max_drawdown if max_drawdown < 0 else 0
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_dicts = [t.to_dict() for t in self.trades]
        return pd.DataFrame(trade_dicts)

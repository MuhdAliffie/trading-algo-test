"""
Visualization module - generates charts and graphs.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def ensure_output_dir(path: str):
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def normalize_plot_date(date):
    """Normalize date-like values so cumulative return lines aggregate by day."""
    parsed_date = pd.to_datetime(date, errors='coerce')
    if pd.isna(parsed_date):
        return date
    return parsed_date.normalize().to_pydatetime()


def slugify(value: str) -> str:
    """Create a filename-safe label."""
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def plot_equity_curve_by_strategy(results_df: pd.DataFrame, raw_results: List[Dict], region: str, 
                                 output_dir: str = "results/charts") -> str:
    """
    Plot cumulative returns over time by strategy for a region.
    
    Args:
        results_df: Results dataframe
        region: Region name
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    strategies = results_df['strategy'].unique()
    
    for strategy in strategies:
        strategy_results = [r for r in raw_results if r.get('region') == region and r.get('strategy') == strategy]
        cumulative_returns_by_date = {}

        for result in strategy_results:
            if 'portfolio_values' in result and 'dates' in result:
                initial_capital = result.get('initial_capital', 10000)
                for date, value in zip(result['dates'], result['portfolio_values']):
                    plot_date = normalize_plot_date(date)
                    cumulative_return = (value - initial_capital) / initial_capital
                    cumulative_returns_by_date.setdefault(plot_date, []).append(cumulative_return)

        if cumulative_returns_by_date:
            sorted_dates = sorted(cumulative_returns_by_date.keys())
            avg_returns = [np.mean(cumulative_returns_by_date[date]) for date in sorted_dates]
            ax.plot(sorted_dates, avg_returns, linewidth=2, label=strategy)
    
    ax.set_title(f'{region} - Cumulative Returns by Strategy')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{region.lower()}_equity_curve_by_strategy.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_average_return_by_strategy_region(results_df: pd.DataFrame,
                                          output_dir: str = "results/charts") -> str:
    """
    Plot average cumulative return by strategy and region.
    
    Args:
        results_df: Results dataframe
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Aggregate by region and strategy
    pivot_data = results_df.pivot_table(
        values='total_return',
        index='strategy',
        columns='region',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='line', marker='o', ax=ax, linewidth=2)
    ax.set_title('Average Cumulative Return by Strategy and Region')
    ax.set_ylabel('Average Cumulative Return')
    ax.set_xlabel('Strategy')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(title='Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'average_return_by_strategy_region.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_sharpe_ratio_comparison(results_df: pd.DataFrame,
                                output_dir: str = "results/charts") -> str:
    """
    Plot Sharpe ratio comparison by strategy and region.
    
    Args:
        results_df: Results dataframe
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_data = results_df.pivot_table(
        values='sharpe_ratio',
        index='strategy',
        columns='region',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', ax=ax)
    ax.set_title('Average Sharpe Ratio by Strategy and Region')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_xlabel('Strategy')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(title='Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'sharpe_ratio_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_max_drawdown_comparison(results_df: pd.DataFrame,
                                output_dir: str = "results/charts") -> str:
    """
    Plot maximum drawdown comparison by strategy and region.
    
    Args:
        results_df: Results dataframe
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_data = results_df.pivot_table(
        values='max_drawdown',
        index='strategy',
        columns='region',
        aggfunc='mean'
    )
    
    pivot_data.plot(kind='bar', ax=ax)
    ax.set_title('Average Maximum Drawdown by Strategy and Region')
    ax.set_ylabel('Maximum Drawdown')
    ax.set_xlabel('Strategy')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(title='Region')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'max_drawdown_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_signal_distribution(results_df: pd.DataFrame, region: str,
                            output_dir: str = "results/charts") -> str:
    """
    Plot signal distribution by region.
    
    Args:
        results_df: Results dataframe
        region: Region name
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    region_data = results_df[results_df['region'] == region]
    signal_counts = region_data['latest_signal'].value_counts()
    
    colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
    color_list = [colors.get(sig, 'blue') for sig in signal_counts.index]
    
    signal_counts.plot(kind='bar', ax=ax, color=color_list)
    ax.set_title(f'{region} - Latest Signal Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Signal')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{region.lower()}_signal_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_top_stocks(results_df: pd.DataFrame, region: str, n_top: int = 10,
                   output_dir: str = "results/charts") -> str:
    """
    Plot top N performing stocks in a region.
    
    Args:
        results_df: Results dataframe
        region: Region name
        n_top: Number of top stocks to display
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    region_data = results_df[results_df['region'] == region]
    top_stocks = region_data.nlargest(n_top, 'total_return')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_stocks['ticker'] + ' - ' + top_stocks['strategy'], top_stocks['total_return'])
    ax.set_title(f'{region} - Top {n_top} Performing Stocks')
    ax.set_xlabel('Total Return')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{region.lower()}_top_{n_top}_stocks.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_worst_stocks(results_df: pd.DataFrame, region: str, n_worst: int = 10,
                     output_dir: str = "results/charts") -> str:
    """
    Plot worst N performing stocks in a region.
    
    Args:
        results_df: Results dataframe
        region: Region name
        n_worst: Number of worst stocks to display
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    region_data = results_df[results_df['region'] == region]
    worst_stocks = region_data.nsmallest(n_worst, 'total_return')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(worst_stocks['ticker'] + ' - ' + worst_stocks['strategy'], worst_stocks['total_return'])
    ax.set_title(f'{region} - Worst {n_worst} Performing Stocks')
    ax.set_xlabel('Total Return')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{region.lower()}_worst_{n_worst}_stocks.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_strategy_vs_buyhold(results_df: pd.DataFrame, region: str,
                            output_dir: str = "results/charts") -> str:
    """
    Plot strategy returns vs buy-and-hold comparison.
    
    Args:
        results_df: Results dataframe
        region: Region name
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    region_data = results_df[results_df['region'] == region]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(region_data))
    width = 0.35
    
    ax.bar(x - width/2, region_data['total_return'], width, label='Strategy Return', alpha=0.8)
    ax.bar(x + width/2, region_data['buy_hold_return'], width, label='Buy-and-Hold Return', alpha=0.8)
    
    ax.set_title(f'{region} - Strategy Return vs Buy-and-Hold')
    ax.set_ylabel('Return')
    ax.set_xlabel('Stock-Strategy')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    plt.xticks([])
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'{region.lower()}_strategy_vs_buyhold.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_stock_price_with_signals(df: pd.DataFrame, ticker: str, strategy: str,
                                 output_dir: str = "results/charts") -> str:
    """
    Plot stock price with buy/sell signals.
    
    Args:
        df: DataFrame with price and signal data
        ticker: Stock ticker
        strategy: Strategy name
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    signal_col_map = {
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
    
    signal_col = signal_col_map.get(strategy, 'Signal')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot price
    ax.plot(df['Date'], df['Close'], label='Close Price', color='black', linewidth=1.5)
    
    # Plot buy signals
    buy_signals = df[df[signal_col] == 1]
    if not buy_signals.empty:
        ax.scatter(buy_signals['Date'], buy_signals['Close'], 
                  color='green', marker='^', s=100, label='BUY', zorder=5)
    
    # Plot sell signals
    sell_signals = df[df[signal_col] == -1]
    if not sell_signals.empty:
        ax.scatter(sell_signals['Date'], sell_signals['Close'],
                  color='red', marker='v', s=100, label='SELL', zorder=5)
    
    ax.set_title(f'{ticker} - {strategy} Strategy')
    ax.set_ylabel('Price')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    filename = f'{ticker}_{strategy}_signals.png'.replace(' ', '_')
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_cumulative_returns_by_strategy(results_df: pd.DataFrame, raw_results: List[Dict],
                                       output_dir: str = "results/charts") -> str:
    """
    Plot cumulative returns over time by strategy.
    
    Args:
        results_df: Aggregated results dataframe
        raw_results: Raw results with time series data
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    strategies = results_df['strategy'].unique()
    
    for strategy in strategies:
        cumulative_returns_by_date = {}
        
        # Collect all portfolio values for this strategy
        strategy_results = [r for r in raw_results 
                           if r.get('strategy') == strategy and 'portfolio_values' in r]
        
        for result in strategy_results:
            if 'portfolio_values' in result and 'dates' in result:
                dates = result['dates']
                values = result['portfolio_values']
                initial_capital = result.get('initial_capital', 10000)
                
                for date, value in zip(dates, values):
                    plot_date = normalize_plot_date(date)
                    ret = (value - initial_capital) / initial_capital
                    if plot_date not in cumulative_returns_by_date:
                        cumulative_returns_by_date[plot_date] = []
                    cumulative_returns_by_date[plot_date].append(ret)
        
        if cumulative_returns_by_date:
            sorted_dates = sorted(cumulative_returns_by_date.keys())
            avg_returns = [np.mean(cumulative_returns_by_date[d]) for d in sorted_dates]
            ax.plot(sorted_dates, avg_returns, marker='', linewidth=2, label=strategy)
    
    ax.set_title('Cumulative Returns Over Time by Strategy', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'cumulative_returns_by_strategy.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_cumulative_returns_by_region(results_df: pd.DataFrame, raw_results: List[Dict],
                                      output_dir: str = "results/charts") -> str:
    """
    Plot cumulative returns over time by region.
    
    Args:
        results_df: Aggregated results dataframe
        raw_results: Raw results with time series data
        output_dir: Output directory
    
    Returns:
        Path to saved figure
    """
    ensure_output_dir(output_dir)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    regions = results_df['region'].unique()
    
    for region in regions:
        cumulative_returns_by_date = {}
        
        # Collect all portfolio values for this region
        region_results = [r for r in raw_results 
                         if r.get('region') == region and 'portfolio_values' in r]
        
        for result in region_results:
            if 'portfolio_values' in result and 'dates' in result:
                dates = result['dates']
                values = result['portfolio_values']
                initial_capital = result.get('initial_capital', 10000)
                
                for date, value in zip(dates, values):
                    plot_date = normalize_plot_date(date)
                    ret = (value - initial_capital) / initial_capital
                    if plot_date not in cumulative_returns_by_date:
                        cumulative_returns_by_date[plot_date] = []
                    cumulative_returns_by_date[plot_date].append(ret)
        
        if cumulative_returns_by_date:
            sorted_dates = sorted(cumulative_returns_by_date.keys())
            avg_returns = [np.mean(cumulative_returns_by_date[d]) for d in sorted_dates]
            ax.plot(sorted_dates, avg_returns, marker='', linewidth=2, label=region)
    
    ax.set_title('Cumulative Returns Over Time by Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'cumulative_returns_by_region.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved chart: {filepath}")
    return filepath


def plot_trade_timeline_by_region_strategy(trade_history_df: pd.DataFrame,
                                           output_dir: str = "results/charts") -> List[str]:
    """
    Plot buy/sell event timelines and ROI after each sell for every region and strategy.

    Args:
        trade_history_df: Flattened trade history dataframe
        output_dir: Output directory

    Returns:
        Paths to saved figures
    """
    if trade_history_df is None or trade_history_df.empty:
        logger.warning("No trade history available for timeline charts")
        return []

    ensure_output_dir(output_dir)
    data = trade_history_df.copy()
    data['buy_date'] = pd.to_datetime(data['buy_date'], errors='coerce')
    data['sell_date'] = pd.to_datetime(data['sell_date'], errors='coerce')
    data['roi_after_sell'] = pd.to_numeric(data['roi_after_sell'], errors='coerce')
    data = data.dropna(subset=['buy_date', 'sell_date', 'roi_after_sell'])

    region_slugs = {slugify(region) for region in data['region'].dropna().unique()}
    for filename in os.listdir(output_dir):
        if not filename.startswith('trade_timeline_') or not filename.endswith('.png'):
            continue
        if any(filename.startswith(f'trade_timeline_{region_slug}_') for region_slug in region_slugs):
            os.remove(os.path.join(output_dir, filename))

    chart_paths = []
    for region in sorted(data['region'].dropna().unique()):
        region_data = data[data['region'] == region]
        for strategy in sorted(region_data['strategy'].dropna().unique()):
            subset = region_data[region_data['strategy'] == strategy].copy()
            if subset.empty:
                continue

            fig, (event_ax, roi_ax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(14, 8),
                sharex=True,
                gridspec_kw={'height_ratios': [1, 1.4]}
            )

            buy_counts = subset.set_index('buy_date').resample('ME').size()
            sell_counts = subset.set_index('sell_date').resample('ME').size()
            event_counts = pd.concat(
                [buy_counts.rename('Buy'), sell_counts.rename('Sell')],
                axis=1
            ).fillna(0)

            event_ax.plot(event_counts.index, event_counts['Buy'], color='green', linewidth=2, label='Buy count')
            event_ax.plot(event_counts.index, event_counts['Sell'], color='red', linewidth=2, label='Sell count')
            event_ax.set_title(f'{region} - {strategy} Buy/Sell Timeline')
            event_ax.set_ylabel('Monthly Trades')
            event_ax.legend(loc='upper left')
            event_ax.grid(True, alpha=0.3)

            colors = np.where(subset['roi_after_sell'] >= 0, 'green', 'red')
            roi_ax.scatter(
                subset['sell_date'],
                subset['roi_after_sell'],
                c=colors,
                alpha=0.35,
                s=22,
                label='Sell ROI'
            )

            monthly_roi = subset.set_index('sell_date')['roi_after_sell'].resample('ME').mean()
            roi_ax.plot(
                monthly_roi.index,
                monthly_roi.values,
                color='black',
                linewidth=2,
                label='Monthly average sell ROI'
            )
            roi_ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            roi_ax.set_title('ROI After Each Sell')
            roi_ax.set_ylabel('ROI After Sell')
            roi_ax.set_xlabel('Date')
            roi_ax.legend(loc='upper left')
            roi_ax.grid(True, alpha=0.3)

            fig.autofmt_xdate()
            plt.tight_layout()

            filepath = os.path.join(
                output_dir,
                f'trade_timeline_{slugify(region)}_{slugify(strategy)}.png'
            )
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved chart: {filepath}")
            chart_paths.append(filepath)

    return chart_paths


def generate_all_charts(results_df: pd.DataFrame, raw_results: List[Dict],
                       output_dir: str = "results/charts",
                       trade_history_df: pd.DataFrame = None):
    """
    Generate all required charts.
    
    Args:
        results_df: Aggregated results dataframe
        raw_results: Raw results with individual stock data
        output_dir: Output directory
    """
    ensure_output_dir(output_dir)
    
    logger.info("Generating charts...")
    
    regions = results_df['region'].unique()
    
    for region in regions:
        plot_equity_curve_by_strategy(results_df, raw_results, region, output_dir)
        plot_signal_distribution(results_df, region, output_dir)
        plot_top_stocks(results_df, region, 10, output_dir)
        plot_worst_stocks(results_df, region, 10, output_dir)
        plot_strategy_vs_buyhold(results_df, region, output_dir)
    
    # Overall charts
    plot_average_return_by_strategy_region(results_df, output_dir)
    plot_sharpe_ratio_comparison(results_df, output_dir)
    plot_max_drawdown_comparison(results_df, output_dir)
    
    # New line graphs showing money movement
    try:
        plot_cumulative_returns_by_strategy(results_df, raw_results, output_dir)
        plot_cumulative_returns_by_region(results_df, raw_results, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate cumulative returns graphs: {str(e)}")

    try:
        plot_trade_timeline_by_region_strategy(trade_history_df, output_dir)
    except Exception as e:
        logger.warning(f"Could not generate trade timeline graphs: {str(e)}")
    
    logger.info("Chart generation complete")

# Trading Strategy Backtesting System

A comprehensive Python-based backtesting system for comparing four technical analysis strategies across international stock markets.

## Overview

This system backtests the following trading strategies across the top 100 stocks in Asia, Europe, and the US:

1. **Volume-based Strategy** - Uses volume patterns and spikes
2. **MACD Strategy** - Employs MACD crossovers  
3. **LSTM-Inspired Bollinger Band Mean Reversion** - Combines Bollinger Bands with multi-feature scoring
4. **Ichimoku Cloud Strategy** - Uses Ichimoku cloud components

## Features

- ✅ Parallel backtesting for speed (4+ stocks simultaneously)
- ✅ Automatic data download from Yahoo Finance
- ✅ Comprehensive performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- ✅ Beautiful visualizations and charts
- ✅ Detailed Markdown report with tables and graphs
- ✅ Error handling and logging
- ✅ Configurable parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone or extract this repository:
```bash
cd trading
```

2. Create a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete backtesting pipeline:

```bash
python main.py
```

This will:
1. Create the stock universes (top 100 for each region)
2. Download OHLCV data from Yahoo Finance
3. Calculate all technical indicators
4. Run backtests for all stocks and strategies in parallel
5. Generate performance visualizations
6. Create a comprehensive Markdown report

## Output Files

After running, you'll find:

```
results/
├── report.md                          # Main report (open in any markdown viewer)
├── combined/
│   ├── all_strategy_results.csv      # All backtest results
│   ├── latest_signals.csv            # Current BUY/SELL/HOLD signals
│   ├── asia_errors.csv               # Asia errors (if any)
│   ├── europe_errors.csv             # Europe errors (if any)
│   ├── us_errors.csv                 # US errors (if any)
│   └── ...
├── charts/
│   ├── asia_equity_curve_by_strategy.png
│   ├── europe_equity_curve_by_strategy.png
│   ├── us_equity_curve_by_strategy.png
│   ├── average_return_by_strategy_region.png
│   ├── sharpe_ratio_comparison.png
│   ├── max_drawdown_comparison.png
│   └── ... (more charts)
└── logs/
    └── backtest_YYYYMMDD_HHMMSS.log
```

## Configuration

Edit `src/config.py` to customize:

- **Backtest Period**: Change `BACKTEST_START_DATE` and `BACKTEST_END_DATE`
- **Strategy Parameters**: Adjust MACD periods, Bollinger Band window, etc.
- **Backtesting Rules**: Commission, initial capital, etc.
- **Parallel Processing**: Number of workers
- **Output**: Enable/disable individual stock charts, etc.

### Key Configuration Options

```python
# Backtest period (default: 5 years)
BACKTEST_START_DATE = "2019-06-25"
BACKTEST_END_DATE = "2024-06-25"

# Capital and commission
INITIAL_CAPITAL = 10000  # Per stock
COMMISSION = 0.001      # 0.1% per trade

# Strategy parameters
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Parallel processing
MAX_WORKERS = 4

# Output options
ENABLE_INDIVIDUAL_CHARTS = False  # Set True for 300+ charts
```

## Strategy Rules

### Volume Strategy
- **Buy**: Volume > 150% of 20-day MA AND price closes higher
- **Sell**: Volume drops below MA OR price declines >1%

### MACD Strategy  
- **Buy**: MACD crosses above Signal line (12-26-9)
- **Sell**: MACD crosses below Signal line

### LSTM-Inspired Bollinger Band Mean Reversion
- **Components**: BB position, Z-score, volatility, momentum, MA distance
- **Buy**: Price below 20-period MA AND mean reversion score > 0.6
- **Sell**: Price above MA AND mean reversion score < 0.4

### Ichimoku Cloud Strategy
- **Components**: Tenkan, Kijun, Senkou Spans A & B, Chikou
- **Buy**: Price above cloud AND Tenkan crosses above Kijun
- **Sell**: Price below cloud OR Tenkan crosses below Kijun

## Data Source

- **Provider**: Yahoo Finance (via yfinance library)
- **Data Type**: Daily OHLCV (Open, High, Low, Close, Volume)
- **Adjusted Close**: Used for return calculations by default

## Backtesting Assumptions

- ✓ Long-only strategy (no short selling)
- ✓ One position at a time
- ✓ No leverage
- ✓ Commission: 0.1% per trade
- ✓ No slippage or market impact
- ✓ Signals generated after market close
- ✓ Trades executed at next day's open price
- ✓ No stop-loss or take-profit

## Performance Metrics

For each stock and strategy, the system calculates:

| Metric | Description |
|--------|-------------|
| Total Return | Cumulative return from start to end |
| Annualized Return | Return normalized to per-year |
| Sharpe Ratio | Risk-adjusted return (vs 2% risk-free rate) |
| Max Drawdown | Largest peak-to-trough decline |
| Win Rate | % of trades with positive returns |
| Num Trades | Total number of trades executed |
| Avg Trade Return | Average return per trade |
| Profit Factor | Gross profit / Gross loss |
| Excess Return | Strategy return - Buy-and-hold return |

## Interpreting Results

### Report Structure

1. **Objective** - Strategy overview and disclaimer
2. **Data and Universe** - Source, date range, stocks tested
3. **Strategy Rules** - Buy/sell logic for each strategy
4. **Overall Results** - Aggregated metrics by region and strategy
5. **Latest Signals** - Current BUY/SELL/HOLD for each stock-strategy
6. **Best/Worst Performers** - Top and bottom 10 stocks
7. **Visualizations** - Charts comparing strategies
8. **Conclusion** - Summary and limitations

### Key Insight: Buy-and-Hold Comparison

The `excess_return` column shows whether each strategy beat buy-and-hold:
- **Positive**: Strategy outperformed buy-and-hold
- **Negative**: Buy-and-hold was better
- Compare aggregate averages to see which strategies work best

## Troubleshooting

### "No data available for ticker XXX"
- The ticker may not be available on Yahoo Finance
- The stock may have been delisted
- Solution: Edit `data/universe/*.csv` to replace with valid tickers

### "Insufficient data"
- Stock has fewer than 252 trading days of history
- More recent IPOs often have this issue
- Solution: Set `MIN_DATA_POINTS` lower in config, or ignore these stocks

### ImportError for pandas, numpy, etc.
- Dependencies not installed
- Solution: Run `pip install -r requirements.txt`

### Charts not generating
- Matplotlib backend issue
- Solution: This is non-critical; the backtest still completes
- Check `results/logs/` for detailed error messages

## Limitations & Disclaimers

⚠️ **Important**: This is for research and educational purposes only.

- Past performance does not guarantee future results
- Backtests use historical data; real-world execution may differ
- Assumes perfect execution without slippage
- Results vary significantly by stock and parameter selection
- Technical analysis results are highly sensitive to tuning
- No financial advice provided; not a trading system

## Development

### Project Structure

```
trading/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── universe.py           # Market universe loading
│   ├── data_loader.py        # OHLCV data download
│   ├── indicators.py         # Technical indicators
│   ├── backtest.py           # Backtesting engine
│   ├── parallel_runner.py    # Parallel execution
│   ├── visualization.py      # Chart generation
│   └── reporting.py          # Report generation
├── data/
│   └── universe/             # Stock universe CSVs
│       ├── asia_top100.csv
│       ├── europe_top100.csv
│       └── us_top100.csv
└── results/
    ├── report.md             # Final report
    ├── combined/             # CSV results
    ├── charts/               # PNG visualizations
    └── logs/                 # Execution logs
```

### Adding a New Strategy

1. Create indicator function in `src/indicators.py`:
```python
def calculate_my_strategy(df, **params):
    df = df.copy()
    # Calculate indicator...
    df['My_Strategy_Signal'] = ...  # 1=BUY, -1=SELL, 0=HOLD
    return df
```

2. Add to `calculate_all_strategies()` in same file

3. Add signal column mapping in `src/backtest.py`

4. Update `src/config.py` STRATEGIES list

## Performance Tips

- **Speed**: Increase `MAX_WORKERS` (up to your CPU cores)
- **Memory**: Disable individual stock charts: `ENABLE_INDIVIDUAL_CHARTS = False`
- **Accuracy**: Increase backtest period but expect longer runtime

## Version History

- **v1.0.0** (2024-06-25) - Initial release
  - 4 strategies (Volume, MACD, LSTM-Bollinger, Ichimoku)
  - 3 regions (Asia, Europe, US)
  - Parallel backtesting
  - Comprehensive reporting

## License

This project is provided as-is for research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `results/logs/`
3. Verify configuration in `src/config.py`
4. Check that all dependencies are installed: `pip install -r requirements.txt`

---

**Enjoy your backtesting! Remember: Technical analysis is not financial advice.** 📊

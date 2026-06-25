"""
Reporting module - generates Markdown report with results.
"""

import os
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
import re

from src import config

logger = logging.getLogger(__name__)


def slugify(value: str) -> str:
    """Create a filename-safe label."""
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def create_report(results_df: pd.DataFrame, universe_info: Dict, 
                 skipped_df: pd.DataFrame, errors_df: pd.DataFrame,
                 output_file: str = "results/report.md",
                 trade_history_df: pd.DataFrame = None):
    """
    Create comprehensive Markdown report.
    
    Args:
        results_df: Results dataframe
        universe_info: Information about stock universes
        skipped_df: DataFrame with skipped tickers
        errors_df: DataFrame with errors
        output_file: Output file path
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    results_df = results_df.copy()
    for date_col in ['start_date', 'end_date', 'latest_signal_date']:
        if date_col in results_df.columns:
            results_df[date_col] = pd.to_datetime(results_df[date_col], errors='coerce')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("# Trading Tools Performance Report\n\n")
        f.write(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Objective
        f.write("## 1. Objective\n\n")
        strategy_names = sorted(results_df['strategy'].dropna().unique())
        region_names = sorted(results_df['region'].dropna().unique())
        f.write(
            f"This study compares {len(strategy_names)} technical analysis strategies "
            f"across {', '.join(region_names)} stock markets:\n\n"
        )
        for strategy in strategy_names:
            f.write(f"- **{strategy}**\n")
        f.write("\n")
        f.write("Each strategy is backtested on the top 100 stocks by market capitalization in each region.\n\n")
        f.write("**Disclaimer:** This analysis is for research and educational purposes only, not financial advice.\n\n")
        
        # 2. Data and Universe
        f.write("## 2. Data and Universe\n\n")
        
        total_stocks = results_df['ticker'].nunique()
        total_results = len(results_df)
        
        f.write(f"- **Data Source**: Yahoo Finance (yfinance)\n")
        f.write(f"- **Data Frequency**: Daily OHLCV\n")
        start_date = pd.to_datetime(results_df['start_date'], errors='coerce').min()
        end_date = pd.to_datetime(results_df['end_date'], errors='coerce').max()
        f.write(f"- **Date Range**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")
        f.write(f"- **Total Stocks Tested**: {total_stocks}\n")
        f.write(f"- **Total Results (Stocks x Strategies)**: {total_results}\n\n")
        
        f.write("### Stocks by Region:\n\n")
        for region in sorted(results_df['region'].dropna().unique()):
            info = universe_info.get(region, {})
            region_stocks = results_df[results_df['region'] == region]['ticker'].nunique()
            f.write(f"- **{region}**: {region_stocks} stocks ({info.get('description', 'N/A')})\n")
        f.write("\n")
        
        if not skipped_df.empty:
            f.write(f"### Skipped Stocks: {len(skipped_df)}\n\n")
            f.write("The following stocks were skipped due to insufficient data:\n\n")
            for _, row in skipped_df.head(20).iterrows():
                f.write(f"- {row['ticker']}: {row['reason']}\n")
            if len(skipped_df) > 20:
                f.write(f"- ... and {len(skipped_df) - 20} more\n")
            f.write("\n")
        
        if not errors_df.empty:
            f.write(f"### Errors: {len(errors_df)}\n\n")
            f.write("The following stocks encountered errors during backtesting:\n\n")
            for _, row in errors_df.head(10).iterrows():
                f.write(f"- {row['ticker']}: {row['error']}\n")
            if len(errors_df) > 10:
                f.write(f"- ... and {len(errors_df) - 10} more\n")
            f.write("\n")
        
        # 3. Strategy Rules
        f.write("## 3. Strategy Rules\n\n")
        f.write("The active strategy rules are defined in `strategy.txt` and implemented as signal columns in `src/indicators.py`. ")
        f.write("Signals are generated after market close and executed at the next trading day's open. ")
        f.write("All strategies are long-only and use one position at a time.\n\n")
        f.write("### Active Strategy List\n\n")
        for strategy in config.STRATEGIES:
            f.write(f"- **{strategy}**\n")
        f.write("\n")
        
        # 4. Overall Results
        f.write("## 4. Overall Results\n\n")
        
        # Aggregate metrics by strategy and region
        summary = results_df.groupby(['region', 'strategy']).agg({
            'total_return': ['mean', 'median'],
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'max_drawdown': 'mean',
            'win_rate': 'mean',
            'num_trades': 'mean',
            'buy_hold_return': 'mean',
            'excess_return': 'mean'
        }).round(4)
        
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        # 5. Latest Signals
        f.write("## 5. Latest Buy/Sell/Hold Indications\n\n")
        
        for region in sorted(results_df['region'].unique()):
            f.write(f"### {region} Latest Signals\n\n")
            
            region_data = results_df[results_df['region'] == region].copy()
            region_data = region_data.sort_values('latest_signal_date', ascending=False)
            
            # Filter to most recent signals
            signal_table = region_data[[
                'ticker', 'strategy', 'latest_signal', 'latest_signal_date',
                'final_price', 'total_return', 'buy_hold_return', 'excess_return'
            ]].head(50)
            
            signal_table.columns = [
                'Ticker', 'Strategy', 'Signal', 'Signal Date',
                'Last Price', 'Strategy Return', 'B&H Return', 'Excess Return'
            ]
            
            f.write(signal_table.to_markdown(index=False))
            f.write("\n\n")
        
        # 6. Best Performing Stocks
        f.write("## 6. Best Performing Stocks\n\n")
        
        for region in sorted(results_df['region'].unique()):
            f.write(f"### {region} Top 10 Stocks\n\n")
            
            region_data = results_df[results_df['region'] == region].copy()
            top_stocks = region_data.nlargest(10, 'total_return')[[
                'ticker', 'strategy', 'total_return', 'annualized_return',
                'sharpe_ratio', 'max_drawdown', 'num_trades', 'latest_signal'
            ]]
            
            top_stocks.columns = [
                'Ticker', 'Strategy', 'Total Return', 'Annualized Return',
                'Sharpe Ratio', 'Max Drawdown', 'Num Trades', 'Latest Signal'
            ]
            
            f.write(top_stocks.to_markdown(index=False))
            f.write("\n\n")
        
        # 7. Worst Performing Stocks
        f.write("## 7. Worst Performing Stocks\n\n")
        
        for region in sorted(results_df['region'].unique()):
            f.write(f"### {region} Worst 10 Stocks\n\n")
            
            region_data = results_df[results_df['region'] == region].copy()
            worst_stocks = region_data.nsmallest(10, 'total_return')[[
                'ticker', 'strategy', 'total_return', 'annualized_return',
                'sharpe_ratio', 'max_drawdown', 'num_trades', 'latest_signal'
            ]]
            
            worst_stocks.columns = [
                'Ticker', 'Strategy', 'Total Return', 'Annualized Return',
                'Sharpe Ratio', 'Max Drawdown', 'Num Trades', 'Latest Signal'
            ]
            
            f.write(worst_stocks.to_markdown(index=False))
            f.write("\n\n")
        
        # 8. Graphs
        f.write("## 8. Visualizations\n\n")
        
        f.write("### Equity Curve Analysis - Money Movement Over Time\n\n")
        f.write("![Cumulative returns by strategy](charts/cumulative_returns_by_strategy.png)\n\n")
        f.write("![Cumulative returns by region](charts/cumulative_returns_by_region.png)\n\n")
        
        f.write("### Regional Performance\n\n")
        for region in sorted(results_df['region'].unique()):
            f.write(f"![{region} equity curve](charts/{region.lower()}_equity_curve_by_strategy.png)\n\n")
        
        f.write("### Strategy Comparison\n\n")
        f.write("![Average cumulative return by strategy and region](charts/average_return_by_strategy_region.png)\n\n")
        f.write("![Sharpe ratio comparison](charts/sharpe_ratio_comparison.png)\n\n")
        f.write("![Maximum drawdown comparison](charts/max_drawdown_comparison.png)\n\n")
        
        f.write("### Performance Analysis\n\n")
        for region in sorted(results_df['region'].unique()):
            f.write(f"![{region} signal distribution](charts/{region.lower()}_signal_distribution.png)\n\n")
            f.write(f"![{region} top performers](charts/{region.lower()}_top_10_stocks.png)\n\n")
            f.write(f"![{region} worst performers](charts/{region.lower()}_worst_10_stocks.png)\n\n")
        
        # 9. Trade Timeline and Sell ROI Analysis
        f.write("## 9. Trade Timeline and Sell ROI Analysis\n\n")
        f.write("ROI after each sell is calculated as the net return from the buy price to the sell price after the configured commission on both sides of the trade. ")
        f.write("The full trade-level dataset is saved at `results/combined/trade_history.csv`.\n\n")

        if trade_history_df is not None and not trade_history_df.empty:
            trades = trade_history_df.copy()
            for date_col in ['buy_date', 'sell_date']:
                if date_col in trades.columns:
                    trades[date_col] = pd.to_datetime(trades[date_col], errors='coerce')
            trades['roi_after_sell'] = pd.to_numeric(trades['roi_after_sell'], errors='coerce')
            trades['holding_period_days'] = pd.to_numeric(trades['holding_period_days'], errors='coerce')
            trades = trades.dropna(subset=['roi_after_sell'])

            sell_summary = trades.groupby(['region', 'strategy']).agg(
                Sell_Count=('roi_after_sell', 'count'),
                Avg_ROI_After_Sell=('roi_after_sell', 'mean'),
                Median_ROI_After_Sell=('roi_after_sell', 'median'),
                Win_Rate_After_Sell=('roi_after_sell', lambda values: (values > 0).mean()),
                Best_Sell_ROI=('roi_after_sell', 'max'),
                Worst_Sell_ROI=('roi_after_sell', 'min'),
                Avg_Holding_Days=('holding_period_days', 'mean')
            ).reset_index()

            display_summary = sell_summary.copy()
            percent_cols = [
                'Avg_ROI_After_Sell',
                'Median_ROI_After_Sell',
                'Win_Rate_After_Sell',
                'Best_Sell_ROI',
                'Worst_Sell_ROI'
            ]
            for col in percent_cols:
                display_summary[col] = display_summary[col].map(lambda value: f"{value:.2%}")
            display_summary['Avg_Holding_Days'] = display_summary['Avg_Holding_Days'].map(lambda value: f"{value:.1f}")
            display_summary.columns = [
                'Region', 'Strategy', 'Sell Count', 'Avg ROI After Sell',
                'Median ROI After Sell', 'Win Rate After Sell', 'Best Sell ROI',
                'Worst Sell ROI', 'Avg Holding Days'
            ]

            f.write("### Sell ROI Summary by Region and Strategy\n\n")
            f.write(display_summary.to_markdown(index=False))
            f.write("\n\n")

            f.write("### Trade Timelines\n\n")
            chart_dir = os.path.join(os.path.dirname(output_file), 'charts')
            for region in sorted(trades['region'].dropna().unique()):
                f.write(f"#### {region}\n\n")
                for strategy in sorted(trades[trades['region'] == region]['strategy'].dropna().unique()):
                    filename = f"trade_timeline_{slugify(region)}_{slugify(strategy)}.png"
                    if os.path.exists(os.path.join(chart_dir, filename)):
                        f.write(f"![{region} {strategy} trade timeline](charts/{filename})\n\n")

            recent_sells = trades.sort_values('sell_date', ascending=False).head(30)[[
                'region', 'ticker', 'strategy', 'buy_date', 'buy_price',
                'sell_date', 'sell_price', 'roi_after_sell', 'holding_period_days', 'exit_reason'
            ]].copy()
            recent_sells['roi_after_sell'] = recent_sells['roi_after_sell'].map(lambda value: f"{value:.2%}")
            recent_sells.columns = [
                'Region', 'Ticker', 'Strategy', 'Buy Date', 'Buy Price',
                'Sell Date', 'Sell Price', 'ROI After Sell', 'Holding Days', 'Exit Reason'
            ]

            f.write("### Most Recent Sell Outcomes\n\n")
            f.write(recent_sells.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No completed trades were available for sell ROI analysis.\n\n")

        # 10. Conclusion
        f.write("## 10. Conclusion\n\n")
        
        # Calculate best strategy overall
        best_by_return = results_df.loc[results_df['total_return'].idxmax()]
        best_by_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        
        avg_by_strategy = results_df.groupby('strategy')['total_return'].mean().sort_values(ascending=False)
        
        f.write("### Key Findings:\n\n")
        f.write(f"- **Best Strategy Overall (by total return)**: {best_by_return['strategy']} ")
        f.write(f"({best_by_return['total_return']:.2%}) on {best_by_return['ticker']}\n")
        f.write(f"- **Best Risk-Adjusted Strategy (by Sharpe)**: {best_by_sharpe['strategy']} ")
        f.write(f"(Sharpe: {best_by_sharpe['sharpe_ratio']:.2f}) on {best_by_sharpe['ticker']}\n\n")
        
        f.write("### Performance by Strategy:\n\n")
        for strategy in avg_by_strategy.index:
            avg_return = avg_by_strategy[strategy]
            f.write(f"- **{strategy}**: Average return {avg_return:.2%}\n")
        f.write("\n")
        
        for region in sorted(results_df['region'].unique()):
            region_best = results_df[results_df['region'] == region].loc[
                results_df[results_df['region'] == region]['total_return'].idxmax()
            ]
            f.write(f"- **Best in {region}**: {region_best['strategy']} ({region_best['total_return']:.2%})\n")
        f.write("\n")
        
        # Buy-and-hold comparison
        strategy_avg_return = results_df['total_return'].mean()
        buyhold_avg_return = results_df['buy_hold_return'].mean()
        
        f.write(f"### Strategies vs Buy-and-Hold:\n\n")
        f.write(f"- **Average Strategy Return**: {strategy_avg_return:.2%}\n")
        f.write(f"- **Average Buy-and-Hold Return**: {buyhold_avg_return:.2%}\n")
        f.write(f"- **Average Excess Return**: {(strategy_avg_return - buyhold_avg_return):.2%}\n\n")
        
        if strategy_avg_return > buyhold_avg_return:
            f.write("[YES] **Technical strategies outperformed buy-and-hold on average**\n\n")
        else:
            f.write("[NO] **Buy-and-hold outperformed technical strategies on average**\n\n")
        
        f.write("### Limitations and Considerations:\n\n")
        f.write("- Analysis based on historical data; past performance does not guarantee future results\n")
        f.write("- Backtests assume perfect execution without slippage or market impact\n")
        f.write("- Transaction costs limited to commissions (0.1%); real-world costs may be higher\n")
        f.write("- Some stocks had insufficient data and were excluded from analysis\n")
        f.write("- Signals generated after market close; trades executed at next day open\n")
        f.write("- No stop-loss or take-profit implemented in this analysis\n")
        f.write("- Results vary significantly by stock and may not be representative\n")
        f.write("- Technical analysis results are highly sensitive to parameter selection\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    logger.info(f"Report saved to: {output_file}")
    return output_file

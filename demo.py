"""
Demo script - Generate sample results to demonstrate the reporting system.
Creates synthetic backtest results and generates the full report.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.reporting import create_report
from src.visualization import plot_equity_curve_by_strategy, plot_average_return_by_strategy_region
from src.visualization import plot_sharpe_ratio_comparison, plot_max_drawdown_comparison
from src.visualization import plot_cumulative_returns_by_strategy, plot_cumulative_returns_by_region, generate_all_charts
from src import config

def generate_sample_results():
    """Generate sample backtest results."""
    print("Generating sample backtest results...")
    
    regions = ["Asia", "Europe", "US"]
    strategies = config.STRATEGIES
    
    # Sample tickers for each region
    tickers = {
        "Asia": ["0700.HK", "9999.T", "000001.SS"],
        "Europe": ["SAP", "ASML.AS", "SAF.PA"],
        "US": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    }
    
    results = []
    raw_results = []
    
    for region in regions:
        for ticker in tickers[region]:
            for strategy in strategies:
                # Generate realistic but random metrics
                total_return = np.random.normal(0.15, 0.35)  # Average 15% return
                buy_hold = np.random.normal(0.12, 0.25)  # Average 12% buy-hold
                
                # Generate time series data for equity curve
                start_date = datetime.now() - timedelta(days=365*5)
                end_date = datetime.now()
                num_days = (end_date - start_date).days
                dates = [start_date + timedelta(days=i) for i in range(num_days) if i % 2 == 0]  # Every other day
                
                # Generate portfolio values with trend
                initial_capital = 10000
                trend = np.linspace(0, total_return, len(dates))
                noise = np.random.normal(0, 0.02, len(dates))
                portfolio_values = initial_capital * (1 + trend + noise)
                
                result = {
                    'ticker': ticker,
                    'strategy': strategy,
                    'region': region,
                    'total_return': total_return,
                    'annualized_return': total_return / 5,  # 5-year backtest
                    'sharpe_ratio': np.random.normal(0.8, 0.6),
                    'max_drawdown': np.random.normal(-0.25, 0.15),
                    'num_trades': np.random.randint(5, 50),
                    'avg_trade_return': np.random.normal(0.02, 0.05),
                    'best_trade': np.random.uniform(0.05, 0.30),
                    'worst_trade': np.random.uniform(-0.20, -0.02),
                    'win_rate': np.random.uniform(0.35, 0.65),
                    'profit_factor': np.random.uniform(1.0, 3.0),
                    'exposure_time': np.random.uniform(0.3, 0.9),
                    'buy_hold_return': buy_hold,
                    'excess_return': total_return - buy_hold,
                    'latest_signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'latest_signal_date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                    'final_price': np.random.uniform(100, 500),
                    'final_portfolio_value': initial_capital * (1 + total_return),
                    'start_date': start_date,
                    'end_date': end_date,
                    'portfolio_values': portfolio_values.tolist(),
                    'dates': dates,
                    'initial_capital': initial_capital
                }
                results.append(result)
                raw_results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(config.CSV_OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(config.CSV_OUTPUT_DIR, 'all_strategy_results.csv'), index=False)
    
    print(f"Generated {len(results_df)} sample results")
    return results_df, raw_results

def main():
    """Generate demo report with sample data."""
    print("=" * 80)
    print("TRADING STRATEGY BACKTESTING SYSTEM - DEMO")
    print("=" * 80)
    print("\nNote: This uses synthetic data to demonstrate the reporting system.")
    print("In production, this uses real data downloaded from Yahoo Finance.\n")
    
    # Generate sample results
    results_df, raw_results = generate_sample_results()
    
    # Create dummy error/skipped dataframes
    skipped_df = pd.DataFrame({
        'ticker': ['PLACEHOLDER_1', 'PLACEHOLDER_2'],
        'reason': ['Insufficient data', 'No data source']
    })
    
    errors_df = pd.DataFrame()
    
    # Universe info
    universe_info = {
        "Asia": {
            'count': 3,
            'description': 'Asia top 100 stocks by market cap'
        },
        "Europe": {
            'count': 3,
            'description': 'Europe top 100 stocks by market cap'
        },
        "US": {
            'count': 5,
            'description': 'US top 100 stocks by market cap'
        }
    }
    
    # Create report
    print("Generating visualizations...")
    os.makedirs(config.CHART_OUTPUT_DIR, exist_ok=True)
    
    try:
        # Use the master function to generate all charts
        generate_all_charts(results_df, raw_results, config.CHART_OUTPUT_DIR)
        print("Visualizations generated successfully")
    except Exception as e:
        print(f"Warning: Could not generate visualizations: {e}")
    
    # Create report
    print("Creating Markdown report...")
    report_file = os.path.join(config.OUTPUT_DIR, 'report.md')
    create_report(results_df, universe_info, skipped_df, errors_df, report_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Total results: {len(results_df)}")
    print(f"  Regions: {', '.join(results_df['region'].unique())}")
    print(f"  Strategies: {', '.join(results_df['strategy'].unique())}")
    
    print(f"\nOutput Files:")
    print(f"  📄 Report: {report_file}")
    print(f"  📊 Results CSV: {os.path.join(config.CSV_OUTPUT_DIR, 'all_strategy_results.csv')}")
    if os.path.exists(config.CHART_OUTPUT_DIR):
        charts = len([f for f in os.listdir(config.CHART_OUTPUT_DIR) if f.endswith('.png')])
        print(f"  📈 Charts: {charts} PNG files in {config.CHART_OUTPUT_DIR}")
    
    print(f"\nPerformance Summary by Strategy:")
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        avg_return = strategy_data['total_return'].mean()
        avg_sharpe = strategy_data['sharpe_ratio'].mean()
        print(f"  {strategy}:")
        print(f"    - Avg Return: {avg_return:+.2%}")
        print(f"    - Avg Sharpe: {avg_sharpe:.2f}")
    
    print(f"\n✅ The system is ready for production use with real market data!")
    print(f"   To run with real data, ensure network/SSL is configured and run: python main.py")

if __name__ == "__main__":
    main()

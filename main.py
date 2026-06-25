"""
Main entry point for the trading strategy backtesting system.
Run this script to execute the complete backtesting pipeline.
"""

import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.universe import ensure_universe_files_exist, get_all_universes
from src.parallel_runner import ParallelBacktestRunner
from src.visualization import (
    generate_all_charts,
    plot_average_return_by_strategy_region,
    plot_equity_curve_by_strategy,
    plot_max_drawdown_comparison,
    plot_sharpe_ratio_comparison,
    plot_signal_distribution,
    plot_strategy_vs_buyhold,
    plot_top_stocks,
    plot_trade_timeline_by_region_strategy,
    plot_worst_stocks,
)
from src.reporting import create_report

# Configure logging
log_dir = config.LOGS_DIR
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run trading strategy backtests.")
    parser.add_argument(
        "--regions",
        nargs="+",
        help="Optional region names to run, e.g. --regions Malaysia or --regions Asia US",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge selected-region results into existing combined CSVs instead of replacing all outputs.",
    )
    return parser.parse_args()


def normalize_regions(regions):
    """Validate and normalize region names from CLI input."""
    available = {region.lower(): region for region in config.REGIONS}
    if not regions:
        return list(config.REGIONS.keys())

    normalized = []
    for item in regions:
        for part in str(item).split(","):
            key = part.strip().lower()
            if not key:
                continue
            if key not in available:
                raise ValueError(f"Unknown region '{part}'. Available regions: {', '.join(config.REGIONS)}")
            region = available[key]
            if region not in normalized:
                normalized.append(region)
    return normalized


def merge_region_rows(existing_path, new_df, selected_regions):
    """Replace selected regions in an existing combined CSV with newly generated rows."""
    if not os.path.exists(existing_path) or new_df.empty or "region" not in new_df.columns:
        return new_df

    existing_df = pd.read_csv(existing_path)
    if existing_df.empty or "region" not in existing_df.columns:
        return new_df

    existing_df = existing_df[~existing_df["region"].isin(selected_regions)]
    return pd.concat([existing_df, new_df], ignore_index=True, sort=False)


def generate_merge_charts(run_results_df, run_raw_results, merged_results_df, merged_trade_history_df):
    """Generate selected-region charts and refresh comparison/timeline charts after a merge."""
    os.makedirs(config.CHART_OUTPUT_DIR, exist_ok=True)

    for region in run_results_df['region'].unique():
        plot_equity_curve_by_strategy(run_results_df, run_raw_results, region, config.CHART_OUTPUT_DIR)
        plot_signal_distribution(run_results_df, region, config.CHART_OUTPUT_DIR)
        plot_top_stocks(run_results_df, region, 10, config.CHART_OUTPUT_DIR)
        plot_worst_stocks(run_results_df, region, 10, config.CHART_OUTPUT_DIR)
        plot_strategy_vs_buyhold(run_results_df, region, config.CHART_OUTPUT_DIR)

    plot_average_return_by_strategy_region(merged_results_df, config.CHART_OUTPUT_DIR)
    plot_sharpe_ratio_comparison(merged_results_df, config.CHART_OUTPUT_DIR)
    plot_max_drawdown_comparison(merged_results_df, config.CHART_OUTPUT_DIR)
    plot_trade_timeline_by_region_strategy(merged_trade_history_df, config.CHART_OUTPUT_DIR)


def main(regions=None, merge_existing=False):
    """Main execution pipeline."""
    selected_regions = normalize_regions(regions)
    
    logger.info("=" * 80)
    logger.info("TRADING STRATEGY BACKTESTING SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Backtest period: {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}")
    logger.info(f"Regions: {', '.join(selected_regions)}")
    logger.info(f"Merge existing outputs: {merge_existing}")
    logger.info(f"Strategies: {', '.join(config.STRATEGIES)}")
    logger.info(f"Max workers: {config.MAX_WORKERS}")
    
    # Step 1: Ensure universe files exist
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: UNIVERSE CREATION")
    logger.info("=" * 80)
    ensure_universe_files_exist(regions=selected_regions)
    universes = get_all_universes(selected_regions)
    
    for region, df in universes.items():
        logger.info(f"{region}: {len(df)} stocks loaded")
    
    # Step 2: Run parallel backtests
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: PARALLEL BACKTESTING")
    logger.info("=" * 80)
    
    all_results = []
    all_trades = []
    universe_info = {}
    total_stocks = 0
    
    strategy_params = {
        'volume_ma_window': config.VOLUME_MA_WINDOW,
        'volume_spike_threshold': config.VOLUME_SPIKE_THRESHOLD,
        'macd_fast': config.MACD_FAST,
        'macd_slow': config.MACD_SLOW,
        'macd_signal': config.MACD_SIGNAL,
        'bb_window': config.BB_WINDOW,
        'bb_std': config.BB_STD,
        'tenkan_period': config.TENKAN_PERIOD,
        'kijun_period': config.KIJUN_PERIOD,
        'senkou_b_period': config.SENKOU_B_PERIOD,
        'chikou_period': config.CHIKOU_PERIOD,
        'senkou_displacement': config.SENKOU_DISPLACEMENT,
        'supertrend_atr_period': config.SUPERTREND_ATR_PERIOD,
        'supertrend_multiplier': config.SUPERTREND_MULTIPLIER,
        'adx_period': config.ADX_PERIOD,
        'atr_period': config.ATR_PERIOD,
        'cci_period': config.CCI_PERIOD,
        'fibonacci_window': config.FIBONACCI_WINDOW,
        'psar_step': config.PSAR_STEP,
        'psar_max_step': config.PSAR_MAX_STEP,
        'rsi_period': config.RSI_PERIOD,
        'stochastic_k_period': config.STOCHASTIC_K_PERIOD,
        'stochastic_d_period': config.STOCHASTIC_D_PERIOD,
        'sma_fast': config.SMA_FAST,
        'sma_slow': config.SMA_SLOW,
        'ema_fast': config.EMA_FAST,
        'ema_slow': config.EMA_SLOW,
        'mfi_period': config.MFI_PERIOD,
        'obv_sma_window': config.OBV_SMA_WINDOW,
        'vwap_volume_sma_window': config.VWAP_VOLUME_SMA_WINDOW,
        'williams_r_period': config.WILLIAMS_R_PERIOD,
        'roc_period': config.ROC_PERIOD,
        'donchian_high_window': config.DONCHIAN_HIGH_WINDOW,
        'donchian_low_window': config.DONCHIAN_LOW_WINDOW,
        'keltner_ema_period': config.KELTNER_EMA_PERIOD,
        'keltner_atr_period': config.KELTNER_ATR_PERIOD,
        'keltner_multiplier': config.KELTNER_MULTIPLIER,
        'atr_breakout_high_window': config.ATR_BREAKOUT_HIGH_WINDOW,
        'atr_breakout_multiplier': config.ATR_BREAKOUT_MULTIPLIER,
        'atr_trailing_stop_multiplier': config.ATR_TRAILING_STOP_MULTIPLIER,
    }
    
    for region, universe_df in universes.items():
        logger.info(f"\nProcessing {region}...")
        
        # Get tickers
        tickers = universe_df['ticker'].tolist()
        total_stocks += len(tickers)
        
        # Store universe info
        universe_info[region] = {
            'count': len(tickers),
            'description': config.REGIONS[region]['description']
        }
        
        # Run backtests
        runner = ParallelBacktestRunner(
            max_workers=config.MAX_WORKERS,
            enable_progress=config.ENABLE_PROGRESS_BAR
        )
        
        results = runner.run_parallel_backtests(
            tickers,
            config.BACKTEST_START_DATE,
            config.BACKTEST_END_DATE,
            strategy_params,
            region
        )
        
        company_lookup = universe_df.set_index('ticker')['company_name'].to_dict()

        # Add region info to results and flatten trade histories for timeline analysis
        for stock_result in runner.results:
            for strategy_result in stock_result['results']:
                result_dict = strategy_result.copy()
                trade_history = result_dict.pop('trade_history', pd.DataFrame())
                result_dict['region'] = region
                all_results.append(result_dict)

                if isinstance(trade_history, pd.DataFrame) and not trade_history.empty:
                    trade_history = trade_history.copy()
                    trade_history['region'] = region
                    trade_history['company_name'] = trade_history['ticker'].map(company_lookup).fillna('')
                    trade_history['company'] = trade_history['company_name']
                    trade_history['gross_return_pct'] = trade_history['return_pct']
                    trade_history['roi_after_sell'] = (
                        ((1 - config.COMMISSION) ** 2)
                        * (trade_history['exit_price'] / trade_history['entry_price'])
                        - 1
                    )
                    trade_history['roi_pct'] = trade_history['roi_after_sell'] * 100
                    trade_history['trade_return'] = trade_history['exit_price'] - trade_history['entry_price']
                    all_trades.append(trade_history)
        
        # Log summary
        logger.info(f"{region} Summary:")
        logger.info(f"  - Successful: {len(results)}")
        logger.info(f"  - Errors: {len(runner.errors)}")
        logger.info(f"  - Skipped: {len(runner.skipped)}")
        
        # Save error and skipped logs, removing stale files after clean reruns.
        error_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_errors.csv')
        skipped_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_skipped.csv')
        os.makedirs(config.CSV_OUTPUT_DIR, exist_ok=True)

        if runner.errors:
            error_df = runner.get_error_summary()
            error_df.to_csv(error_file, index=False)
            logger.info(f"  - Saved errors to: {error_file}")
        elif os.path.exists(error_file):
            os.remove(error_file)
            logger.info(f"  - Removed stale errors file: {error_file}")
        
        if runner.skipped:
            skipped_df = runner.get_skipped_summary()
            skipped_df.to_csv(skipped_file, index=False)
            logger.info(f"  - Saved skipped to: {skipped_file}")
        elif os.path.exists(skipped_file):
            os.remove(skipped_file)
            logger.info(f"  - Removed stale skipped file: {skipped_file}")
    
    # Step 3: Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: AGGREGATING RESULTS")
    logger.info("=" * 80)
    
    if not all_results:
        logger.error("No results to process!")
        return
    
    run_results_df = pd.DataFrame(all_results)
    logger.info(f"Run results: {len(run_results_df)}")
    logger.info(f"Total stocks tested: {total_stocks}")
    logger.info(f"Unique tickers in run: {run_results_df['ticker'].nunique()}")
    
    run_trade_history_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not run_trade_history_df.empty:
        run_trade_history_df = run_trade_history_df.rename(columns={
            'entry_date': 'buy_date',
            'entry_price': 'buy_price',
            'exit_date': 'sell_date',
            'exit_price': 'sell_price',
            'days_held': 'holding_period'
        })
        run_trade_history_df['holding_period_days'] = run_trade_history_df['holding_period']
        trade_columns = [
            'region', 'ticker', 'company', 'company_name', 'strategy',
            'buy_date', 'buy_price', 'sell_date', 'sell_price',
            'holding_period', 'holding_period_days', 'roi_pct',
            'trade_return', 'gross_return_pct', 'roi_after_sell', 'exit_reason'
        ]
        run_trade_history_df = run_trade_history_df[trade_columns].sort_values(
            ['region', 'strategy', 'sell_date', 'ticker']
        )

    # Save combined results CSV
    os.makedirs(config.CSV_OUTPUT_DIR, exist_ok=True)
    results_csv = os.path.join(config.CSV_OUTPUT_DIR, 'all_strategy_results.csv')
    trade_history_csv = os.path.join(config.CSV_OUTPUT_DIR, 'trade_history.csv')
    signals_csv = os.path.join(config.CSV_OUTPUT_DIR, 'latest_signals.csv')

    results_df = run_results_df.drop(columns=['portfolio_values', 'dates'], errors='ignore')
    trade_history_df = run_trade_history_df

    if merge_existing:
        logger.info(f"Merging selected regions into existing combined outputs: {', '.join(selected_regions)}")
        results_df = merge_region_rows(results_csv, results_df, selected_regions)
        trade_history_df = merge_region_rows(trade_history_csv, trade_history_df, selected_regions)

    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved combined results to: {results_csv}")

    trade_history_df.to_csv(trade_history_csv, index=False)
    logger.info(f"Saved trade history to: {trade_history_csv}")
    
    # Save latest signals
    signals_df = results_df[[
        'ticker', 'strategy', 'region', 'latest_signal', 'latest_signal_date',
        'final_price', 'total_return', 'buy_hold_return'
    ]].drop_duplicates()
    signals_df.to_csv(signals_csv, index=False)
    logger.info(f"Saved latest signals to: {signals_csv}")
    
    # Step 4: Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    os.makedirs(config.CHART_OUTPUT_DIR, exist_ok=True)
    try:
        if merge_existing:
            generate_merge_charts(run_results_df, all_results, results_df, trade_history_df)
        else:
            generate_all_charts(run_results_df, all_results, config.CHART_OUTPUT_DIR, trade_history_df)
        logger.info("Visualizations generated successfully")
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
    
    # Step 5: Generate report
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: GENERATING REPORT")
    logger.info("=" * 80)
    
    # Collect errors and skipped for report
    all_errors = []
    all_skipped = []
    
    report_regions = sorted(results_df['region'].dropna().unique()) if merge_existing else list(universes.keys())
    report_universe_info = {
        region: {
            'count': int(results_df[results_df['region'] == region]['ticker'].nunique()),
            'description': config.REGIONS.get(region, {}).get('description', 'N/A')
        }
        for region in report_regions
    }

    for region in report_regions:
        error_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_errors.csv')
        if os.path.exists(error_file):
            all_errors.append(pd.read_csv(error_file))
        
        skipped_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_skipped.csv')
        if os.path.exists(skipped_file):
            all_skipped.append(pd.read_csv(skipped_file))
    
    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    
    report_file = os.path.join(config.OUTPUT_DIR, 'report.md')
    create_report(results_df, report_universe_info, skipped_df, errors_df, report_file, trade_history_df=trade_history_df)
    logger.info(f"Report saved to: {report_file}")
    
    # Step 6: Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"\nResults by Region:")
    for region in results_df['region'].unique():
        region_data = results_df[results_df['region'] == region]
        logger.info(f"  {region}:")
        logger.info(f"    - Stocks: {region_data['ticker'].nunique()}")
        logger.info(f"    - Avg Return: {region_data['total_return'].mean():.2%}")
        logger.info(f"    - Avg Sharpe: {region_data['sharpe_ratio'].mean():.2f}")
    
    logger.info(f"\nResults by Strategy:")
    for strategy in results_df['strategy'].unique():
        strategy_data = results_df[results_df['strategy'] == strategy]
        logger.info(f"  {strategy}:")
        logger.info(f"    - Stocks: {strategy_data['ticker'].nunique()}")
        logger.info(f"    - Avg Return: {strategy_data['total_return'].mean():.2%}")
        logger.info(f"    - Avg Sharpe: {strategy_data['sharpe_ratio'].mean():.2f}")
    
    logger.info(f"\nOutput Files:")
    logger.info(f"  - Report: {report_file}")
    logger.info(f"  - Results CSV: {results_csv}")
    logger.info(f"  - Charts directory: {config.CHART_OUTPUT_DIR}")
    logger.info(f"  - Log file: {log_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"BACKTEST COMPLETE - {datetime.now()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        args = parse_args()
        main(regions=args.regions, merge_existing=args.merge_existing)
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        sys.exit(1)

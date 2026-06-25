"""
Main entry point for the trading strategy backtesting system.
Run this script to execute the complete backtesting pipeline.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.universe import ensure_universe_files_exist, get_all_universes
from src.parallel_runner import ParallelBacktestRunner
from src.visualization import generate_all_charts
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


def main():
    """Main execution pipeline."""
    
    logger.info("=" * 80)
    logger.info("TRADING STRATEGY BACKTESTING SYSTEM")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Backtest period: {config.BACKTEST_START_DATE} to {config.BACKTEST_END_DATE}")
    logger.info(f"Strategies: {', '.join(config.STRATEGIES)}")
    logger.info(f"Max workers: {config.MAX_WORKERS}")
    
    # Step 1: Ensure universe files exist
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: UNIVERSE CREATION")
    logger.info("=" * 80)
    ensure_universe_files_exist()
    universes = get_all_universes()
    
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
        
        # Save error and skipped logs
        if runner.errors:
            error_df = runner.get_error_summary()
            error_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_errors.csv')
            os.makedirs(os.path.dirname(error_file), exist_ok=True)
            error_df.to_csv(error_file, index=False)
            logger.info(f"  - Saved errors to: {error_file}")
        
        if runner.skipped:
            skipped_df = runner.get_skipped_summary()
            skipped_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_skipped.csv')
            os.makedirs(os.path.dirname(skipped_file), exist_ok=True)
            skipped_df.to_csv(skipped_file, index=False)
            logger.info(f"  - Saved skipped to: {skipped_file}")
    
    # Step 3: Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: AGGREGATING RESULTS")
    logger.info("=" * 80)
    
    if not all_results:
        logger.error("No results to process!")
        return
    
    results_df = pd.DataFrame(all_results)
    logger.info(f"Total results: {len(results_df)}")
    logger.info(f"Total stocks tested: {total_stocks}")
    logger.info(f"Unique tickers: {results_df['ticker'].nunique()}")
    
    trade_history_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if not trade_history_df.empty:
        trade_history_df = trade_history_df.rename(columns={
            'entry_date': 'buy_date',
            'entry_price': 'buy_price',
            'exit_date': 'sell_date',
            'exit_price': 'sell_price',
            'days_held': 'holding_period'
        })
        trade_history_df['holding_period_days'] = trade_history_df['holding_period']
        trade_columns = [
            'region', 'ticker', 'company', 'company_name', 'strategy',
            'buy_date', 'buy_price', 'sell_date', 'sell_price',
            'holding_period', 'holding_period_days', 'roi_pct',
            'trade_return', 'gross_return_pct', 'roi_after_sell', 'exit_reason'
        ]
        trade_history_df = trade_history_df[trade_columns].sort_values(
            ['region', 'strategy', 'sell_date', 'ticker']
        )

    # Save combined results CSV
    os.makedirs(config.CSV_OUTPUT_DIR, exist_ok=True)
    results_csv = os.path.join(config.CSV_OUTPUT_DIR, 'all_strategy_results.csv')
    results_df.drop(columns=['portfolio_values', 'dates'], errors='ignore').to_csv(results_csv, index=False)
    logger.info(f"Saved combined results to: {results_csv}")

    trade_history_csv = os.path.join(config.CSV_OUTPUT_DIR, 'trade_history.csv')
    trade_history_df.to_csv(trade_history_csv, index=False)
    logger.info(f"Saved trade history to: {trade_history_csv}")
    
    # Save latest signals
    signals_csv = os.path.join(config.CSV_OUTPUT_DIR, 'latest_signals.csv')
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
        generate_all_charts(results_df, all_results, config.CHART_OUTPUT_DIR, trade_history_df)
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
    
    for region in universes.keys():
        error_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_errors.csv')
        if os.path.exists(error_file):
            all_errors.append(pd.read_csv(error_file))
        
        skipped_file = os.path.join(config.CSV_OUTPUT_DIR, f'{region.lower()}_skipped.csv')
        if os.path.exists(skipped_file):
            all_skipped.append(pd.read_csv(skipped_file))
    
    errors_df = pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()
    skipped_df = pd.concat(all_skipped, ignore_index=True) if all_skipped else pd.DataFrame()
    
    report_file = os.path.join(config.OUTPUT_DIR, 'report.md')
    create_report(results_df, universe_info, skipped_df, errors_df, report_file, trade_history_df=trade_history_df)
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
        main()
    except Exception as e:
        logger.error(f"FATAL ERROR: {str(e)}", exc_info=True)
        sys.exit(1)

"""
Parallel executor - runs backtests concurrently for efficiency.
"""

import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict
import traceback

from src.data_loader import get_stock_data
from src.indicators import calculate_all_strategies
from src.backtest import BacktestEngine
from src import config

logger = logging.getLogger(__name__)


class ParallelBacktestRunner:
    """Runs backtests for multiple stocks and strategies in parallel."""
    
    def __init__(self, max_workers: int = 4, enable_progress: bool = True):
        """
        Initialize the parallel runner.
        
        Args:
            max_workers: Number of parallel workers
            enable_progress: Whether to show progress bar
        """
        self.max_workers = max_workers
        self.enable_progress = enable_progress
        self.results = []
        self.errors = []
        self.skipped = []
    
    def run_backtest_for_stock(self, ticker: str, start_date: str, end_date: str, 
                              strategy_params: Dict = None, region: str = None) -> Dict:
        """
        Run backtest for a single stock across all strategies.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            strategy_params: Strategy parameters
            region: Region for this stock
        
        Returns:
            Dictionary with results for all strategies
        """
        try:
            # Load data
            df = get_stock_data(ticker, start_date, end_date, min_points=config.MIN_DATA_POINTS)
            
            if df is None:
                self.skipped.append({
                    'ticker': ticker,
                    'reason': 'Insufficient data'
                })
                return None
            
            # Calculate indicators
            df = calculate_all_strategies(df, strategy_params)
            
            # Run backtest for each strategy
            results = []
            for strategy in config.STRATEGIES:
                engine = BacktestEngine(
                    ticker=ticker,
                    strategy_name=strategy,
                    initial_capital=config.INITIAL_CAPITAL,
                    commission=config.COMMISSION,
                    use_adjusted=config.USE_ADJUSTED_CLOSE
                )
                
                result = engine.run_backtest(df)
                result['trade_history'] = engine.get_trade_history()
                if region:
                    result['region'] = region
                results.append(result)
            
            return {
                'ticker': ticker,
                'results': results,
                'data': df
            }
        
        except Exception as e:
            error_msg = f"{ticker}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            self.errors.append({
                'ticker': ticker,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return None
    
    def run_parallel_backtests(self, tickers: List[str], start_date: str, end_date: str,
                              strategy_params: Dict = None, region: str = None) -> List[Dict]:
        """
        Run backtests for multiple stocks in parallel.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            strategy_params: Strategy parameters
            region: Region for these stocks
        
        Returns:
            List of results dictionaries
        """
        self.results = []
        self.errors = []
        self.skipped = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.run_backtest_for_stock,
                    ticker,
                    start_date,
                    end_date,
                    strategy_params,
                    region
                ): ticker for ticker in tickers
            }
            
            # Use tqdm for progress bar if enabled
            iterator = as_completed(futures)
            if self.enable_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Running backtests")
            
            for future in iterator:
                try:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                except Exception as e:
                    ticker = futures[future]
                    error_msg = f"{ticker}: {str(e)}"
                    logger.error(error_msg)
                    self.errors.append({
                        'ticker': ticker,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
        
        return self.results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to a single DataFrame."""
        all_results = []
        
        for stock_result in self.results:
            for strategy_result in stock_result['results']:
                all_results.append(strategy_result)
        
        return pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    def get_error_summary(self) -> pd.DataFrame:
        """Get error summary as DataFrame."""
        if not self.errors:
            return pd.DataFrame()
        
        return pd.DataFrame(self.errors)
    
    def get_skipped_summary(self) -> pd.DataFrame:
        """Get skipped stocks summary as DataFrame."""
        if not self.skipped:
            return pd.DataFrame()
        
        return pd.DataFrame(self.skipped)

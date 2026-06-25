"""
Test script - Run backtest with reliable stocks for demonstration.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override universe with reliable stocks
RELIABLE_STOCKS = {
    "Asia": [
        {"ticker": "9999.T", "company_name": "SoftBank", "region": "Asia", "exchange": "TSE", "sector": "Tech"},
    ],
    "Europe": [
        {"ticker": "SAP", "company_name": "SAP", "region": "Europe", "exchange": "XETRA", "sector": "Software"},
    ],
    "US": [
        {"ticker": "AAPL", "company_name": "Apple", "region": "US", "exchange": "NASDAQ", "sector": "Tech"},
        {"ticker": "MSFT", "company_name": "Microsoft", "region": "US", "exchange": "NASDAQ", "sector": "Tech"},
        {"ticker": "GOOGL", "company_name": "Alphabet", "region": "US", "exchange": "NASDAQ", "sector": "Tech"},
        {"ticker": "AMZN", "company_name": "Amazon", "region": "US", "exchange": "NASDAQ", "sector": "Consumer"},
        {"ticker": "NVDA", "company_name": "NVIDIA", "region": "US", "exchange": "NASDAQ", "sector": "Semiconductors"},
    ]
}

def create_test_universes():
    """Create test universe CSVs."""
    os.makedirs("data/universe", exist_ok=True)
    
    for region, stocks in RELIABLE_STOCKS.items():
        csv_path = f"data/universe/{region.lower()}_top100.csv"
        df = pd.DataFrame(stocks)
        df.to_csv(csv_path, index=False)
        print(f"Created test universe: {csv_path} ({len(stocks)} stocks)")

if __name__ == "__main__":
    print("Creating test stock universes...")
    create_test_universes()
    print("\nRunning backtest with test stocks...")
    print("(Using smaller dataset for faster demonstration)\n")
    
    # Now import and run main
    from main import main
    main()

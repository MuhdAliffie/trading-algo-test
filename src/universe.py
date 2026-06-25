"""
Market universe builder - fetches or loads 100-stock regional universes.
"""

import csv
import logging
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

MIN_UNIVERSE_SIZE = 100
REQUIRED_COLUMNS = ["ticker", "company_name", "region", "exchange", "sector", "market_cap"]

SOURCE_CONFIGS = {
    "US": {
        "url": "https://en.wikipedia.org/wiki/S%26P_100",
        "description": "S&P 100 large-cap constituents used as the US top-100 proxy",
        "exchange": "NYSE/NASDAQ",
        "symbol_header": "symbol",
        "name_header": "name",
        "sector_header": "sector",
    },
    "Europe": {
        "url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "description": "FTSE 100 constituents used as the Europe top-100 proxy",
        "exchange": "LSE",
        "symbol_header": "ticker",
        "name_header": "company",
        "sector_header": "sector",
    },
    "Asia": {
        "url": "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225",
        "description": "Nikkei 225 constituents used as the Asia top-100 proxy",
        "exchange": "TSE",
        "symbol_header": "code",
        "name_header": "company name",
        "prefer_fallback_order": True,
    },
    "Malaysia": {
        "url": "https://companiesmarketcap.com/malaysia/largest-companies-in-malaysia-by-market-cap/",
        "description": "Largest Bursa Malaysia listed companies by market capitalization",
        "exchange": "Bursa Malaysia",
        "source_type": "companiesmarketcap",
        "regex": r"([A-Za-z0-9]+(?:SS)?\.KL)",
        "prefer_fallback_order": True,
    },
}

YFINANCE_NAME_CACHE = {}

FALLBACK_SYMBOLS = {
    "US": [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "AMAT", "AMD", "AMGN",
        "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY",
        "BRK-B", "C", "CAT", "CHTR", "CI", "CL", "CMCSA", "CME", "COF", "COP",
        "COST", "CRM", "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "ELV",
        "EMR", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD",
        "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC", "KO", "LIN",
        "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "MDT", "META", "MMM", "MO",
        "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP",
        "PFE", "PG", "PLD", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO",
        "SPG", "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS",
        "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
    ],
    "Europe": [
        "III.L", "ADM.L", "AAF.L", "ALW.L", "AAL.L", "ANTO.L", "ABF.L", "AZN.L", "AUTO.L", "AV.L",
        "BAB.L", "BA.L", "BARC.L", "BTRW.L", "BEZ.L", "BKG.L", "BP.L", "BATS.L", "BLND.L", "BT-A.L",
        "BNZL.L", "CNA.L", "CCH.L", "CPG.L", "CRDA.L", "DCC.L", "DGE.L", "DPLM.L", "EDV.L", "ENT.L",
        "EXPN.L", "FCIT.L", "FRAS.L", "FRES.L", "GLEN.L", "GSK.L", "HLN.L", "HLMA.L", "HIK.L", "HSBA.L",
        "IMI.L", "IMB.L", "INF.L", "IHG.L", "ICP.L", "ITRK.L", "JD.L", "KGF.L", "LAND.L", "LGEN.L",
        "LLOY.L", "LMP.L", "LSEG.L", "MNG.L", "MKS.L", "MNDI.L", "NG.L", "NWG.L", "NXT.L", "PSON.L",
        "PSH.L", "PCT.L", "PHNX.L", "PRU.L", "RKT.L", "REL.L", "RTO.L", "RMV.L", "RR.L", "RIO.L",
        "SGE.L", "SBRY.L", "SDR.L", "SMT.L", "SGRO.L", "SVT.L", "SHEL.L", "SN.L", "SMIN.L", "SMDS.L",
        "SPX.L", "SSE.L", "STJ.L", "STAN.L", "TAY.L", "TW.L", "TSCO.L", "ULVR.L", "UU.L", "VOD.L",
        "WEIR.L", "WPP.L", "WTB.L", "WISE.L", "WKL.L", "AHT.L", "BME.L", "CCL.L", "IAG.L", "ITV.L",
        "SXS.L", "UTG.L", "HL.L", "EZJ.L", "OCDO.L",
    ],
    "Asia": [
        "7203.T", "6758.T", "9984.T", "8306.T", "6861.T", "6501.T", "6098.T", "4063.T", "8035.T", "9432.T",
        "7974.T", "9983.T", "4519.T", "8058.T", "4568.T", "6367.T", "4502.T", "6902.T", "8001.T", "8766.T",
        "8316.T", "7267.T", "9433.T", "6954.T", "6981.T", "9613.T", "7751.T", "3382.T", "7741.T", "8411.T",
        "7269.T", "6702.T", "6503.T", "4503.T", "8053.T", "9020.T", "5108.T", "8801.T", "9022.T", "9202.T",
        "4901.T", "6762.T", "6752.T", "6301.T", "7201.T", "7202.T", "7733.T", "4543.T", "1925.T", "2502.T",
        "2503.T", "2914.T", "2802.T", "2801.T", "2269.T", "2282.T", "2871.T", "3861.T", "3402.T", "3407.T",
        "4005.T", "4042.T", "4043.T", "4183.T", "4188.T", "4208.T", "4452.T", "4631.T", "4902.T", "5019.T",
        "5020.T", "5101.T", "5201.T", "5202.T", "5214.T", "5233.T", "5301.T", "5332.T", "5333.T", "5401.T",
        "5406.T", "5411.T", "5631.T", "5703.T", "5706.T", "5711.T", "5713.T", "5801.T", "5802.T", "5803.T",
        "5901.T", "6113.T", "6302.T", "6305.T", "6326.T", "6361.T", "6471.T", "6472.T", "6473.T", "6479.T",
        "6504.T", "6506.T", "6645.T", "6674.T", "6701.T", "6724.T", "6753.T", "6770.T", "6841.T", "6857.T",
    ],
    "Malaysia": [
        "1155", "1295", "5347", "1023", "5225", "8869", "5819", "5285", "6742", "3816",
        "5211", "6033", "6947", "5183", "5326", "4863", "1961", "6012", "5398", "2445",
        "4677", "4707", "1082", "5246", "6888", "5235SS", "5296", "4197", "7084", "4065",
        "4715", "7277", "3182", "3255", "7113", "5258", "5168", "7153", "3026", "5983",
        "5099", "7106", "5274", "9997", "1818", "1015", "1066", "2488", "5288", "5176",
        "5125", "5227", "5014", "5878", "0166", "0097", "5292", "5286", "7204", "5302",
        "3689", "2836", "1899", "3336", "9679", "5209", "5253", "5228", "3034", "1562",
        "5008", "5072", "5916", "9822", "7034", "7052", "5139", "5148", "5186", "5199",
        "5250", "5257", "5318", "5138", "5172", "5248", "5020", "5069", "5106", "5147",
        "5200", "5210", "5272", "5284", "5291", "5301", "7054", "7078", "7164", "7202",
        "7247", "7293", "8583", "8664", "8877", "9296", "9326", "9687", "9695", "9895",
        "2089", "2291", "3867", "4006", "4502", "5077", "5109", "5681", "5789", "6599",
    ],
}


def _normalize_header(value: str) -> str:
    value = re.sub(r"\[[^\]]+\]", "", value or "")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def _normalize_symbol(symbol: str, region: str) -> str:
    symbol = str(symbol).strip()
    symbol = re.sub(r"\s+", "", symbol)

    if region == "US":
        return symbol.replace(".", "-")
    if region == "Europe":
        symbol = symbol.replace(".", "-")
        return symbol if symbol.endswith(".L") else f"{symbol}.L"
    if region == "Asia":
        symbol = symbol.replace(".T", "")
        return f"{symbol}.T"
    if region == "Malaysia":
        symbol = symbol.replace(".KL", "")
        return f"{symbol}.KL"

    return symbol


def _row(ticker: str, company_name: str, region: str, exchange: str, sector: str = "") -> Dict:
    return {
        "ticker": _normalize_symbol(ticker, region),
        "company_name": company_name or ticker,
        "region": region,
        "exchange": exchange,
        "sector": sector or "Unknown",
        "market_cap": "",
    }


def _needs_company_name_resolution(row: Dict) -> bool:
    name = str(row.get("company_name", "")).strip()
    ticker = str(row.get("ticker", "")).strip()
    base_ticker = re.sub(r"\.(T|KL|L)$", "", ticker, flags=re.IGNORECASE)
    return not name or name.upper() in {ticker.upper(), base_ticker.upper()}


def _unique_first_100(rows: List[Dict], limit: int = MIN_UNIVERSE_SIZE) -> List[Dict]:
    seen = set()
    unique_rows = []
    for row in rows:
        ticker = row["ticker"]
        if not ticker or ticker in seen or "PLACEHOLDER" in ticker.upper():
            continue
        seen.add(ticker)
        unique_rows.append(row)
        if limit and len(unique_rows) >= limit:
            break
    return unique_rows


def _fetch_html(url: str) -> str:
    try:
        import curl_cffi.requests as requests

        session = requests.Session(impersonate="chrome", verify=False)
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.warning(f"Could not fetch universe source {url}: {exc}")
        return ""


def _extract_table_rows(region: str, html: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup is unavailable; using static universe fallback")
        return []

    config = SOURCE_CONFIGS[region]
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        if not header_cells:
            continue

        headers = [_normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells.find_all(["th", "td"])]
        if not headers:
            continue

        symbol_idx = _find_header_index(headers, config["symbol_header"])
        name_idx = _find_header_index(headers, config["name_header"])
        sector_idx = _find_header_index(headers, config.get("sector_header", "sector"))
        if symbol_idx is None or name_idx is None:
            continue

        for tr in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if len(cells) <= max(symbol_idx, name_idx):
                continue
            sector = cells[sector_idx] if sector_idx is not None and len(cells) > sector_idx else ""
            rows.append(_row(cells[symbol_idx], cells[name_idx], region, config["exchange"], sector))

    return _unique_first_100(rows, limit=None)


def _extract_companies_market_cap_rows(region: str, html: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("BeautifulSoup is unavailable; using static universe fallback")
        return []

    config = SOURCE_CONFIGS[region]
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for table in soup.find_all("table"):
        header_cells = table.find("tr")
        if not header_cells:
            continue

        headers = [_normalize_header(cell.get_text(" ", strip=True)) for cell in header_cells.find_all(["th", "td"])]
        name_idx = _find_header_index(headers, "name")
        if name_idx is None:
            continue

        for tr in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            if len(cells) <= name_idx:
                continue

            name_text = cells[name_idx].strip()
            match = re.search(config["regex"], name_text)
            if not match:
                continue

            code = match.group(1)
            company_name = re.sub(rf"\s*{re.escape(code)}\s*$", "", name_text).strip()
            rows.append(_row(code, company_name, region, config["exchange"]))

    return _unique_first_100(rows, limit=None)


def _find_header_index(headers: List[str], target: str):
    target = _normalize_header(target)
    for idx, header in enumerate(headers):
        if header == target or target in header:
            return idx
    return None


def _extract_regex_rows(region: str, html: str) -> List[Dict]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    config = SOURCE_CONFIGS[region]
    lines = BeautifulSoup(html, "html.parser").get_text("\n", strip=True).splitlines()
    rows = []
    for line in lines:
        match = re.search(config["regex"], line)
        if not match:
            continue
        groups = match.groups()
        if len(groups) >= 2:
            name, code = groups[:2]
        else:
            code = groups[0]
            name = line.replace(code, "")
        name = re.sub(r"\s+", " ", name).strip(" -*")
        rows.append(_row(code, name, region, config["exchange"]))
    return _unique_first_100(rows, limit=None)


def _fallback_rows(region: str, limit: int = MIN_UNIVERSE_SIZE) -> List[Dict]:
    exchange = SOURCE_CONFIGS[region]["exchange"]
    rows = [_row(symbol, symbol, region, exchange) for symbol in FALLBACK_SYMBOLS[region]]
    return _unique_first_100(rows, limit=limit)


def _merge_with_fallback_order(region: str, source_rows: List[Dict]) -> List[Dict]:
    """Keep the curated ticker order while using source rows for company names."""
    source_by_ticker = {row["ticker"]: row for row in source_rows}
    rows = []

    for row in _fallback_rows(region, limit=None):
        source_row = source_by_ticker.get(row["ticker"])
        if source_row and not _needs_company_name_resolution(source_row):
            row = {
                **row,
                "company_name": source_row["company_name"],
                "sector": source_row.get("sector") or row["sector"],
                "market_cap": source_row.get("market_cap") or row["market_cap"],
            }
        rows.append(row)

    return _unique_first_100(rows + source_rows)


def _lookup_yfinance_name(ticker: str) -> str:
    if ticker in YFINANCE_NAME_CACHE:
        return YFINANCE_NAME_CACHE[ticker]

    name = ""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).get_info()
        for key in ["longName", "shortName", "displayName"]:
            value = str(info.get(key, "")).strip()
            if value:
                name = value
                break
    except Exception as exc:
        logger.debug(f"Could not resolve company name for {ticker}: {exc}")

    YFINANCE_NAME_CACHE[ticker] = name
    return name


def _resolve_missing_company_names(rows: List[Dict]) -> List[Dict]:
    missing = [row["ticker"] for row in rows if _needs_company_name_resolution(row)]
    if not missing:
        return rows

    logger.info(f"Resolving company names for {len(missing)} fallback tickers")
    resolved = {}
    with ThreadPoolExecutor(max_workers=min(8, len(missing))) as executor:
        future_by_ticker = {executor.submit(_lookup_yfinance_name, ticker): ticker for ticker in missing}
        for future in as_completed(future_by_ticker):
            ticker = future_by_ticker[future]
            try:
                name = future.result()
            except Exception:
                name = ""
            if name:
                resolved[ticker] = name

    for row in rows:
        if _needs_company_name_resolution(row) and row["ticker"] in resolved:
            row["company_name"] = resolved[row["ticker"]]

    return rows


def build_universe(region: str) -> pd.DataFrame:
    """Build a 100-stock universe from live source tables, with static fallback."""
    config = SOURCE_CONFIGS[region]
    html = _fetch_html(config["url"])
    source_rows = []

    if html:
        if config.get("source_type") == "companiesmarketcap":
            source_rows = _extract_companies_market_cap_rows(region, html)
        elif "regex" in config:
            source_rows = _extract_regex_rows(region, html)
        else:
            source_rows = _extract_table_rows(region, html)

    if config.get("prefer_fallback_order"):
        rows = _merge_with_fallback_order(region, source_rows)
    else:
        rows = source_rows

    if len(rows) < MIN_UNIVERSE_SIZE:
        logger.warning(
            f"{region}: source yielded {len(rows)} usable tickers; supplementing with static {config['description']}"
        )
        rows = _unique_first_100(rows + _fallback_rows(region, limit=None))

    if len(rows) < MIN_UNIVERSE_SIZE:
        raise ValueError(f"{region}: universe fallback only has {len(rows)} tickers")

    rows = _resolve_missing_company_names(rows[:MIN_UNIVERSE_SIZE])
    if config.get("prefer_fallback_order"):
        unresolved = [row["ticker"] for row in rows if _needs_company_name_resolution(row)]
        if unresolved:
            unresolved_tickers = set(unresolved)
            logger.warning(
                f"{region}: replacing unresolved/stale tickers with named constituents: {', '.join(unresolved)}"
            )
            rows = _unique_first_100(
                [row for row in rows if not _needs_company_name_resolution(row)]
                + [row for row in source_rows if row["ticker"] not in unresolved_tickers]
                + [row for row in _fallback_rows(region, limit=None) if row["ticker"] not in unresolved_tickers]
            )
            rows = _resolve_missing_company_names(rows[:MIN_UNIVERSE_SIZE])

    return pd.DataFrame(rows[:MIN_UNIVERSE_SIZE], columns=REQUIRED_COLUMNS)


def _read_existing_universe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning(f"Could not read {csv_path}; rebuilding it: {exc}")
        return pd.DataFrame()

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[REQUIRED_COLUMNS]


def _has_enough_real_tickers(df: pd.DataFrame) -> bool:
    if df.empty or "ticker" not in df.columns:
        return False

    real_tickers = df[~df["ticker"].astype(str).str.contains("PLACEHOLDER", case=False, na=False)]
    return real_tickers["ticker"].nunique() >= MIN_UNIVERSE_SIZE


def ensure_universe_files_exist(force: bool = False, regions: List[str] = None):
    """Create or repair universe CSV files so each region has 100 real tickers."""
    regions = regions or list(SOURCE_CONFIGS.keys())
    for region in regions:
        if region not in SOURCE_CONFIGS:
            raise ValueError(f"Unknown region: {region}")
        csv_path = f"data/universe/{region.lower()}_top100.csv"
        existing_df = _read_existing_universe(csv_path)

        if not force and _has_enough_real_tickers(existing_df):
            logger.info(f"Universe file is valid: {csv_path} ({len(existing_df)} rows)")
            continue

        logger.info(f"Rebuilding universe file: {csv_path}")
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        universe_df = build_universe(region)
        universe_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL)
        logger.info(f"Wrote {len(universe_df)} stocks to {csv_path}")


def load_universe(region: str) -> pd.DataFrame:
    """
    Load the stock universe for a given region.

    Args:
        region: Region name such as "Asia", "Europe", "US", or "Malaysia"

    Returns:
        DataFrame with columns: ticker, company_name, region, exchange, sector, market_cap
    """
    csv_path = f"data/universe/{region.lower()}_top100.csv"
    df = _read_existing_universe(csv_path)

    if not _has_enough_real_tickers(df):
        ensure_universe_files_exist(force=True, regions=[region])
        df = _read_existing_universe(csv_path)

    df = df[~df["ticker"].astype(str).str.contains("PLACEHOLDER", case=False, na=False)]
    df = df.drop_duplicates(subset=["ticker"]).head(MIN_UNIVERSE_SIZE).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} stocks for {region}")
    return df


def get_all_universes(regions: List[str] = None) -> dict:
    """
    Load all universes.

    Returns:
        Dictionary with region names as keys and DataFrames as values
    """
    regions = regions or list(SOURCE_CONFIGS.keys())
    ensure_universe_files_exist(regions=regions)
    return {region: load_universe(region) for region in regions}

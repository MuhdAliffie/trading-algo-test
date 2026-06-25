"""
Dash dashboard for trading strategy timelines.

Run with:
    python dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate


RESULTS_DIR = Path("results") / "combined"
TRADE_HISTORY_PATH = RESULTS_DIR / "trade_history.csv"
LATEST_SIGNALS_PATH = RESULTS_DIR / "latest_signals.csv"
ALL_RESULTS_PATH = RESULTS_DIR / "all_strategy_results.csv"
ERROR_LOG_PATH = RESULTS_DIR / "error_log.csv"

DEFAULT_INITIAL_CAPITAL = 10000.0

STRATEGY_LABELS = {
    "Volume": "Volume",
    "MACD": "MACD",
    "LSTM_Bollinger": "LSTM-Inspired Bollinger Band Mean Reversion",
    "Ichimoku": "Ichimoku Cloud",
    "Supertrend": "Supertrend",
    "RSI": "RSI",
    "ADX_DI": "ADX-DI",
    "ATR_Breakout": "ATR Breakout",
    "SMA_Crossover": "SMA Crossover",
    "EMA_Crossover": "EMA Crossover",
    "Bollinger_Mean_Reversion": "Bollinger Band Mean Reversion",
    "Stochastic": "Stochastic Oscillator",
    "CCI": "Commodity Channel Index",
    "MFI": "Money Flow Index",
    "OBV": "On-Balance Volume",
    "VWAP": "VWAP",
    "Parabolic_SAR": "Parabolic SAR",
    "Williams_R": "Williams %R",
    "ROC_Momentum": "ROC Momentum",
    "Donchian_Channel": "Donchian Channel",
    "Keltner_Channel": "Keltner Channel",
}

STATUS_OPTIONS = [
    {"label": "All trades", "value": "all"},
    {"label": "Only completed trades", "value": "completed"},
    {"label": "Only profitable trades", "value": "profitable"},
    {"label": "Only losing trades", "value": "losing"},
    {"label": "Only active BUY signals", "value": "active_buy"},
]


def load_trade_history(path: Path = TRADE_HISTORY_PATH) -> pd.DataFrame:
    """Load trade history from the generated results folder."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_latest_signals(path: Path = LATEST_SIGNALS_PATH) -> pd.DataFrame:
    """Load latest signal data if it exists."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "latest_signal_date" in df.columns:
        df["latest_signal_date"] = pd.to_datetime(df["latest_signal_date"], errors="coerce")
    return df


def load_optional_csv(path: Path) -> pd.DataFrame:
    """Load an optional CSV if it exists."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_decimal_return(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().abs().max() > 2:
        return values / 100
    return values


def clean_trade_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade-history columns and infer values where possible."""
    if df.empty:
        return df

    df = df.copy()
    df = df.rename(columns={
        "entry_date": "buy_date",
        "entry_price": "buy_price",
        "exit_date": "sell_date",
        "exit_price": "sell_price",
        "days_held": "holding_period_days",
        "return_pct": "gross_return_pct",
    })

    if "company_name" not in df.columns and "company" in df.columns:
        df["company_name"] = df["company"]

    for column in ["region", "ticker", "company_name", "strategy", "exit_reason"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    for column in ["buy_date", "sell_date"]:
        if column not in df.columns:
            df[column] = pd.NaT
        df[column] = pd.to_datetime(df[column], errors="coerce")

    numeric_columns = [
        "buy_price",
        "sell_price",
        "gross_return_pct",
        "roi_after_sell",
        "roi_pct",
        "trade_return",
        "holding_period_days",
        "portfolio_value",
        "cumulative_return",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "buy_price" not in df.columns:
        df["buy_price"] = np.nan
    if "sell_price" not in df.columns:
        df["sell_price"] = np.nan

    if "trade_return" not in df.columns or df["trade_return"].isna().all():
        df["trade_return"] = df["sell_price"] - df["buy_price"]

    if "holding_period_days" not in df.columns or df["holding_period_days"].isna().all():
        if "holding_period" in df.columns:
            df["holding_period_days"] = pd.to_numeric(df["holding_period"], errors="coerce")
        else:
            df["holding_period_days"] = (df["sell_date"] - df["buy_date"]).dt.days

    if "roi_after_sell" in df.columns and not df["roi_after_sell"].isna().all():
        df["roi_decimal"] = _as_decimal_return(df["roi_after_sell"])
    elif "roi_pct" in df.columns and not df["roi_pct"].isna().all():
        df["roi_decimal"] = _as_decimal_return(df["roi_pct"])
    elif "gross_return_pct" in df.columns and not df["gross_return_pct"].isna().all():
        df["roi_decimal"] = _as_decimal_return(df["gross_return_pct"])
    else:
        df["roi_decimal"] = (df["sell_price"] - df["buy_price"]) / df["buy_price"]

    df["roi_pct"] = df["roi_decimal"] * 100
    df["trade_status"] = np.where(df["sell_date"].notna(), "Completed", "Open")

    return df.sort_values(["region", "strategy", "sell_date", "ticker"], na_position="last")


def _active_buy_keys(signals_df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if signals_df.empty or "latest_signal" not in signals_df.columns:
        return set()

    buy_signals = signals_df[signals_df["latest_signal"].astype(str).str.upper() == "BUY"]
    required = {"region", "ticker", "strategy"}
    if not required.issubset(buy_signals.columns):
        return set()

    return set(zip(
        buy_signals["region"].astype(str),
        buy_signals["ticker"].astype(str),
        buy_signals["strategy"].astype(str),
    ))


def apply_filters(
    df: pd.DataFrame,
    selected_regions: list[str] | None,
    selected_strategies: list[str] | None,
    selected_tickers: list[str] | None,
    start_date: str | None,
    end_date: str | None,
    trade_status: str,
    active_buy_keys: set[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters consistently to all views."""
    if df.empty:
        return df

    filtered = df.copy()

    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    if selected_strategies:
        filtered = filtered[filtered["strategy"].isin(selected_strategies)]
    else:
        return filtered.iloc[0:0]

    if selected_tickers:
        filtered = filtered[filtered["ticker"].isin(selected_tickers)]

    if start_date and end_date:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[
            filtered["buy_date"].between(start_ts, end_ts)
            | filtered["sell_date"].between(start_ts, end_ts)
        ]

    if trade_status == "completed":
        filtered = filtered[filtered["sell_date"].notna()]
    elif trade_status == "profitable":
        filtered = filtered[filtered["roi_decimal"] > 0]
    elif trade_status == "losing":
        filtered = filtered[filtered["roi_decimal"] < 0]
    elif trade_status == "active_buy":
        if not active_buy_keys:
            return filtered.iloc[0:0]
        keys = list(zip(filtered["region"], filtered["ticker"], filtered["strategy"]))
        filtered = filtered[[key in active_buy_keys for key in keys]]

    return filtered


def calculate_summary_metrics(df: pd.DataFrame) -> dict:
    """Calculate headline metrics for filtered completed trades."""
    completed = df[df["sell_date"].notna()].copy()
    if completed.empty:
        return {
            "total_trades": 0,
            "average_roi": np.nan,
            "total_roi": np.nan,
            "win_rate": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
            "average_holding_duration": np.nan,
        }

    return {
        "total_trades": int(len(completed)),
        "average_roi": completed["roi_pct"].mean(),
        "total_roi": completed["roi_pct"].sum(),
        "win_rate": (completed["roi_decimal"] > 0).mean() * 100,
        "best_trade": completed["roi_pct"].max(),
        "worst_trade": completed["roi_pct"].min(),
        "average_holding_duration": completed["holding_period_days"].mean(),
    }


def _blank_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def plot_money_movement(df: pd.DataFrame, initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> go.Figure:
    """
    Plot estimated capital by strategy.

    Assumption: trade_history.csv is trade-level, not a daily portfolio ledger.
    If portfolio_value is missing, this compounds the average ROI of trades that
    closed on each date, starting from DEFAULT_INITIAL_CAPITAL per strategy.
    """
    completed = df[df["sell_date"].notna()].copy()
    fig = _blank_figure("Money Movement by Strategy")

    if completed.empty:
        return fig

    if "portfolio_value" in completed.columns and completed["portfolio_value"].notna().any():
        for strategy, strategy_df in completed.groupby("strategy"):
            path = strategy_df.dropna(subset=["sell_date", "portfolio_value"]).sort_values("sell_date")
            fig.add_trace(go.Scattergl(
                x=path["sell_date"],
                y=path["portfolio_value"],
                mode="lines",
                name=STRATEGY_LABELS.get(strategy, strategy),
            ))
    else:
        for strategy, strategy_df in completed.groupby("strategy"):
            daily_roi = (
                strategy_df
                .dropna(subset=["sell_date", "roi_decimal"])
                .groupby("sell_date")["roi_decimal"]
                .mean()
                .sort_index()
            )
            if daily_roi.empty:
                continue

            capital = initial_capital * (1 + daily_roi).cumprod()
            fig.add_trace(go.Scattergl(
                x=capital.index,
                y=capital.values,
                mode="lines",
                name=STRATEGY_LABELS.get(strategy, strategy),
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Estimated capital=%{y:,.2f}<extra></extra>",
            ))

    fig.update_layout(
        xaxis_title="Sell date",
        yaxis_title="Estimated capital",
        hovermode="x unified",
        legend_title="Strategy",
    )
    fig.update_yaxes(tickprefix="$", separatethousands=True)
    return fig


def plot_trade_timeline(df: pd.DataFrame, chart_limit: int) -> go.Figure:
    """Plot BUY and SELL events for filtered trades."""
    fig = _blank_figure("Buy/Sell Timeline")
    if df.empty:
        return fig

    chart_df = df.sort_values("sell_date").tail(chart_limit).copy()

    buy_events = chart_df[
        ["region", "ticker", "company_name", "strategy", "buy_date", "buy_price", "roi_pct"]
    ].rename(columns={"buy_date": "date", "buy_price": "price"})
    buy_events["event"] = "BUY"

    sell_events = chart_df[
        ["region", "ticker", "company_name", "strategy", "sell_date", "sell_price", "roi_pct"]
    ].rename(columns={"sell_date": "date", "sell_price": "price"})
    sell_events["event"] = "SELL"

    events = pd.concat([buy_events, sell_events], ignore_index=True).dropna(subset=["date"])
    if events.empty:
        return fig

    events["asset"] = events["strategy"] + " | " + events["ticker"]
    events["marker_color"] = np.where(events["event"] == "BUY", "#1f77b4", "#d62728")
    events["marker_symbol"] = np.where(events["event"] == "BUY", "triangle-up", "triangle-down")

    for event_name, event_df in events.groupby("event"):
        fig.add_trace(go.Scattergl(
            x=event_df["date"],
            y=event_df["asset"],
            mode="markers",
            name=event_name,
            marker={
                "size": 8,
                "opacity": 0.78,
                "color": "#1f77b4" if event_name == "BUY" else "#d62728",
                "symbol": "triangle-up" if event_name == "BUY" else "triangle-down",
            },
            customdata=np.stack([
                event_df["region"],
                event_df["ticker"],
                event_df["company_name"],
                event_df["strategy"],
                event_df["price"],
                event_df["roi_pct"],
            ], axis=-1),
            hovertemplate=(
                "Date=%{x|%Y-%m-%d}<br>"
                "Region=%{customdata[0]}<br>"
                "Ticker=%{customdata[1]}<br>"
                "Company=%{customdata[2]}<br>"
                "Strategy=%{customdata[3]}<br>"
                "Price=%{customdata[4]:,.4f}<br>"
                "ROI=%{customdata[5]:.2f}%<extra></extra>"
            ),
        ))

    if len(df) > len(chart_df):
        fig.update_layout(title=f"Buy/Sell Timeline, latest {len(chart_df):,} trades")
    fig.update_layout(xaxis_title="Date", yaxis_title="Strategy and ticker", legend_title="Event")
    return fig


def plot_roi_bar_chart(df: pd.DataFrame, chart_limit: int) -> go.Figure:
    """Plot ROI percentage after each completed sell."""
    completed = df[df["sell_date"].notna()].copy()
    fig = _blank_figure("ROI After Each Sell")

    if completed.empty:
        return fig

    completed = completed.sort_values("sell_date").tail(chart_limit).reset_index(drop=True)
    completed["trade_number"] = completed.index + 1
    colors = np.where(completed["roi_pct"] >= 0, "#2ca02c", "#d62728")

    fig.add_trace(go.Bar(
        x=completed["trade_number"],
        y=completed["roi_pct"],
        marker_color=colors,
        customdata=np.stack([
            completed["region"],
            completed["ticker"],
            completed["company_name"],
            completed["strategy"],
            completed["sell_date"].dt.strftime("%Y-%m-%d"),
            completed["holding_period_days"],
        ], axis=-1),
        hovertemplate=(
            "Trade=%{x}<br>"
            "ROI=%{y:.2f}%<br>"
            "Region=%{customdata[0]}<br>"
            "Ticker=%{customdata[1]}<br>"
            "Company=%{customdata[2]}<br>"
            "Strategy=%{customdata[3]}<br>"
            "Sell date=%{customdata[4]}<br>"
            "Holding days=%{customdata[5]:.0f}<extra></extra>"
        ),
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="#444")
    fig.update_layout(
        title=f"ROI After Each Sell, latest {len(completed):,} completed trades",
        xaxis_title="Completed trade number",
        yaxis_title="ROI after sell (%)",
        showlegend=False,
    )
    return fig


def render_trade_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready completed-trade table."""
    completed = df[df["sell_date"].notna()].copy()
    if completed.empty:
        return completed

    table = pd.DataFrame({
        "Region": completed["region"],
        "Ticker": completed["ticker"],
        "Company name": completed["company_name"],
        "Strategy": completed["strategy"].map(lambda value: STRATEGY_LABELS.get(value, value)),
        "Buy date": completed["buy_date"].dt.strftime("%Y-%m-%d"),
        "Buy price": completed["buy_price"].round(4),
        "Sell date": completed["sell_date"].dt.strftime("%Y-%m-%d"),
        "Sell price": completed["sell_price"].round(4),
        "Holding duration in days": completed["holding_period_days"].round(0),
        "Trade return": completed["trade_return"].round(4),
        "ROI percentage": completed["roi_pct"].round(2),
        "Exit reason": completed["exit_reason"],
    })
    return table.sort_values(["Sell date", "Strategy", "Ticker"], ascending=[False, True, True])


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}%"


def format_number(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}{suffix}"


def metric_card(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ],
        className="metric-card",
    )


def missing_data_layout() -> html.Div:
    return html.Div(
        [
            html.H1("Trading Strategy Timeline Dashboard"),
            html.Div(
                [
                    html.H3("Missing trade history"),
                    html.P("Run `python main.py` first so results/combined/trade_history.csv exists."),
                ],
                className="notice",
            ),
        ],
        className="page",
    )


RAW_TRADES = load_trade_history()
TRADES = clean_trade_data(RAW_TRADES)
LATEST_SIGNALS = load_latest_signals()
ALL_RESULTS = load_optional_csv(ALL_RESULTS_PATH)
ERROR_LOG = load_optional_csv(ERROR_LOG_PATH)
ACTIVE_BUY_KEYS = _active_buy_keys(LATEST_SIGNALS)

REGIONS = sorted(TRADES["region"].dropna().unique()) if not TRADES.empty else []
STRATEGIES = sorted(TRADES["strategy"].dropna().unique()) if not TRADES.empty else []
TICKERS = sorted(TRADES["ticker"].dropna().unique()) if not TRADES.empty else []

MIN_DATE = pd.concat([TRADES["buy_date"], TRADES["sell_date"]]).dropna().min() if not TRADES.empty else None
MAX_DATE = pd.concat([TRADES["buy_date"], TRADES["sell_date"]]).dropna().max() if not TRADES.empty else None


app = Dash(__name__, title="Trading Strategy Timeline Dashboard")
server = app.server

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #171717; }
            .page { padding: 24px; }
            .shell { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 18px; align-items: start; overflow: visible; }
            .sidebar { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; position: sticky; top: 18px; z-index: 5000; overflow: visible; }
            .main { min-width: 0; position: relative; z-index: 1; }
            .panel { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 14px; margin-bottom: 16px; position: relative; z-index: 1; }
            .metrics { display: grid; grid-template-columns: repeat(7, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
            .metric-card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 12px; min-height: 72px; }
            .metric-label { color: #666; font-size: 12px; margin-bottom: 8px; }
            .metric-value { font-weight: 700; font-size: 20px; }
            .control-label { font-weight: 700; margin: 14px 0 6px; display: block; }
            .note, .notice { color: #555; background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin: 12px 0 16px; }
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td,
            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th { font-family: Arial, sans-serif; }
            .DateRangePicker, .DateRangePickerInput, .DateInput { position: relative; z-index: 6000; }
            .DateRangePicker_picker, .SingleDatePicker_picker, .DayPicker {
                z-index: 99999 !important;
            }
            .DateRangePicker_picker {
                top: 44px !important;
                left: 0 !important;
            }
            .CalendarMonthGrid, .CalendarMonth, .DayPicker_transitionContainer {
                z-index: 99999 !important;
            }
            @media (max-width: 1000px) {
                .shell { grid-template-columns: 1fr; }
                .sidebar { position: static; }
                .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if TRADES.empty:
    app.layout = missing_data_layout()
else:
    app.layout = html.Div(
        [
            html.H1("Trading Strategy Timeline Dashboard"),
            html.Div(
                "Strategy filters come from results/combined/trade_history.csv. "
                "After rerunning main.py, this dashboard will show the 18 backtestable "
                "strategies defined in strategy.txt.",
                className="note",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Filters"),
                            html.Label("Region", className="control-label"),
                            dcc.Dropdown(
                                id="region-filter",
                                options=[{"label": region, "value": region} for region in REGIONS],
                                value=REGIONS,
                                multi=True,
                                placeholder="All regions",
                            ),
                            html.Label("Strategies", className="control-label"),
                            dcc.Checklist(
                                id="strategy-filter",
                                options=[
                                    {"label": STRATEGY_LABELS.get(strategy, strategy), "value": strategy}
                                    for strategy in STRATEGIES
                                ],
                                value=STRATEGIES,
                                inputStyle={"marginRight": "8px"},
                                labelStyle={"display": "block", "marginBottom": "8px"},
                            ),
                            html.Label("Ticker", className="control-label"),
                            dcc.Dropdown(
                                id="ticker-filter",
                                options=[{"label": ticker, "value": ticker} for ticker in TICKERS],
                                value=[],
                                multi=True,
                                placeholder="All tickers",
                            ),
                            html.Label("Date range", className="control-label"),
                            dcc.DatePickerRange(
                                id="date-filter",
                                min_date_allowed=MIN_DATE.date(),
                                max_date_allowed=MAX_DATE.date(),
                                start_date=MIN_DATE.date(),
                                end_date=MAX_DATE.date(),
                                display_format="YYYY-MM-DD",
                            ),
                            html.Label("Trade status", className="control-label"),
                            dcc.Dropdown(
                                id="status-filter",
                                options=STATUS_OPTIONS,
                                value="all",
                                clearable=False,
                            ),
                            html.Label("Max trades drawn", className="control-label"),
                            dcc.Slider(
                                id="chart-limit",
                                min=500,
                                max=10000,
                                step=500,
                                value=3000,
                                marks={500: "500", 3000: "3k", 10000: "10k"},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Button("Download filtered table", id="download-button", n_clicks=0),
                            dcc.Download(id="download-data"),
                        ],
                        className="sidebar",
                    ),
                    html.Div(
                        [
                            html.Div(id="empty-message"),
                            html.Div(id="summary-metrics", className="metrics"),
                            html.Div(dcc.Graph(id="money-movement"), className="panel"),
                            html.Div(dcc.Graph(id="trade-timeline"), className="panel"),
                            html.Div(dcc.Graph(id="roi-bar-chart"), className="panel"),
                            html.Div(
                                [
                                    html.H3("Duration Held Table"),
                                    dash_table.DataTable(
                                        id="trade-table",
                                        page_size=25,
                                        sort_action="native",
                                        filter_action="native",
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "fontFamily": "Arial, sans-serif",
                                            "fontSize": 13,
                                            "padding": "8px",
                                            "minWidth": "90px",
                                            "maxWidth": "220px",
                                            "whiteSpace": "normal",
                                        },
                                        style_header={"fontWeight": "bold", "backgroundColor": "#f1f3f5"},
                                    ),
                                ],
                                className="panel",
                            ),
                        ],
                        className="main",
                    ),
                ],
                className="shell",
            ),
        ],
        className="page",
    )


@app.callback(
    Output("ticker-filter", "options"),
    Output("ticker-filter", "value"),
    Input("region-filter", "value"),
    State("ticker-filter", "value"),
)
def update_ticker_options(selected_regions, selected_tickers):
    if TRADES.empty:
        return [], []

    selected_regions = selected_regions or REGIONS
    tickers = sorted(TRADES[TRADES["region"].isin(selected_regions)]["ticker"].dropna().unique())
    selected_tickers = selected_tickers or []
    selected_tickers = [ticker for ticker in selected_tickers if ticker in tickers]
    return [{"label": ticker, "value": ticker} for ticker in tickers], selected_tickers


@app.callback(
    Output("summary-metrics", "children"),
    Output("money-movement", "figure"),
    Output("trade-timeline", "figure"),
    Output("roi-bar-chart", "figure"),
    Output("trade-table", "data"),
    Output("trade-table", "columns"),
    Output("empty-message", "children"),
    Input("region-filter", "value"),
    Input("strategy-filter", "value"),
    Input("ticker-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("status-filter", "value"),
    Input("chart-limit", "value"),
)
def update_dashboard(
    selected_regions,
    selected_strategies,
    selected_tickers,
    start_date,
    end_date,
    trade_status,
    chart_limit,
):
    filtered = apply_filters(
        TRADES,
        selected_regions=selected_regions,
        selected_strategies=selected_strategies,
        selected_tickers=selected_tickers,
        start_date=start_date,
        end_date=end_date,
        trade_status=trade_status,
        active_buy_keys=ACTIVE_BUY_KEYS,
    )

    if filtered.empty:
        empty_message = html.Div("No trades match the selected filters.", className="notice")
        blank = _blank_figure("No matching data")
        return [], blank, blank, blank, [], [], empty_message

    metrics = calculate_summary_metrics(filtered)
    metric_children = [
        metric_card("Total trades", f"{metrics['total_trades']:,}"),
        metric_card("Average ROI", format_percent(metrics["average_roi"])),
        metric_card("Total ROI", format_percent(metrics["total_roi"])),
        metric_card("Win rate", format_percent(metrics["win_rate"])),
        metric_card("Best trade", format_percent(metrics["best_trade"])),
        metric_card("Worst trade", format_percent(metrics["worst_trade"])),
        metric_card("Avg holding", format_number(metrics["average_holding_duration"], " days")),
    ]

    trade_table = render_trade_table(filtered)
    columns = [{"name": column, "id": column} for column in trade_table.columns]

    return (
        metric_children,
        plot_money_movement(filtered),
        plot_trade_timeline(filtered, int(chart_limit or 3000)),
        plot_roi_bar_chart(filtered, int(chart_limit or 3000)),
        trade_table.to_dict("records"),
        columns,
        "",
    )


@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("region-filter", "value"),
    State("strategy-filter", "value"),
    State("ticker-filter", "value"),
    State("date-filter", "start_date"),
    State("date-filter", "end_date"),
    State("status-filter", "value"),
    prevent_initial_call=True,
)
def download_filtered_table(
    n_clicks,
    selected_regions,
    selected_strategies,
    selected_tickers,
    start_date,
    end_date,
    trade_status,
):
    if not n_clicks:
        raise PreventUpdate

    filtered = apply_filters(
        TRADES,
        selected_regions=selected_regions,
        selected_strategies=selected_strategies,
        selected_tickers=selected_tickers,
        start_date=start_date,
        end_date=end_date,
        trade_status=trade_status,
        active_buy_keys=ACTIVE_BUY_KEYS,
    )
    table = render_trade_table(filtered)
    return dcc.send_data_frame(table.to_csv, "filtered_trade_history.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

"""
US Stock Data Prediction Example for Kronos

This script demonstrates how to use Kronos with US stock market data.
US stock data (e.g., from yfinance) uses daily OHLCV format without an
'amount' column — Kronos handles this automatically.

Requirements:
    pip install yfinance

Usage:
    cd examples
    python prediction_us_stock_example.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../")
from model import Kronos, KronosTokenizer, KronosPredictor


# ─────────────────────────────────────────────
# Step 1: Fetch US stock data via yfinance
# ─────────────────────────────────────────────
def fetch_us_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance and format it for Kronos.

    Args:
        ticker: Stock symbol, e.g. "AAPL", "MSFT", "NVDA"
        start:  Start date string, e.g. "2022-01-01"
        end:    End date string,   e.g. "2024-12-31"

    Returns:
        DataFrame with columns: timestamps, open, high, low, close, volume
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install it with: pip install yfinance")

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker and date range.")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.x)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "timestamps": raw.index.tz_localize(None) if raw.index.tz is not None else raw.index,
        "open":   raw["Open"].values,
        "high":   raw["High"].values,
        "low":    raw["Low"].values,
        "close":  raw["Close"].values,
        "volume": raw["Volume"].values,
    })
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Step 2: Visualization helper
# ─────────────────────────────────────────────
def plot_prediction(ticker: str, kline_df: pd.DataFrame, pred_df: pd.DataFrame, lookback: int):
    historical = kline_df.iloc[:lookback]
    ground_truth = kline_df.iloc[lookback:]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(historical["timestamps"], historical["close"],
            label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(ground_truth["timestamps"], ground_truth["close"],
            label="Ground Truth", color="green", linewidth=1.5, linestyle="--")
    ax.plot(pred_df.index, pred_df["close"],
            label="Kronos Prediction", color="red", linewidth=1.5)
    ax.set_title(f"{ticker} — Kronos Daily Close Price Forecast", fontsize=14)
    ax.set_ylabel("Close Price (USD)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL"       # US stock ticker
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"
    LOOKBACK = 400        # Number of historical bars as context (≤ 512 for Kronos-small/base)
    PRED_LEN = 30         # Forecast horizon (trading days)

    # 1. Load Kronos model and tokenizer from Hugging Face Hub
    print("Loading Kronos model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    # 2. Instantiate predictor (auto-selects GPU/MPS/CPU)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print(f"Using device: {predictor.device}")

    # 3. Fetch US stock data
    print(f"Fetching {TICKER} daily data from {START_DATE} to {END_DATE}...")
    df = fetch_us_stock_data(TICKER, START_DATE, END_DATE)
    print(f"  Total bars: {len(df)}")

    if len(df) < LOOKBACK + PRED_LEN:
        raise ValueError(
            f"Not enough data: need at least {LOOKBACK + PRED_LEN} bars, "
            f"got {len(df)}. Adjust date range or reduce LOOKBACK/PRED_LEN."
        )

    # 4. Prepare input windows
    #    Kronos needs: x_df (history), x_timestamp (history timestamps),
    #                  y_timestamp (future timestamps to predict)
    x_df = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume"]]
    x_timestamp = df.loc[:LOOKBACK - 1, "timestamps"]
    y_timestamp = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]

    print(f"  Context window : {x_df.shape[0]} bars  "
          f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")
    print(f"  Forecast horizon: {PRED_LEN} bars  "
          f"({y_timestamp.iloc[0].date()} → {y_timestamp.iloc[-1].date()})")

    # NOTE: 'amount' column is not present in US market data.
    # KronosPredictor automatically computes amount = volume * mean(price)
    # when 'amount' is absent, so no manual workaround is needed.

    # 5. Generate forecast
    print("Running Kronos inference...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,       # sampling temperature
        top_p=0.9,   # nucleus sampling
        sample_count=1,
        verbose=True,
    )

    # 6. Display results
    print("\nForecast (first 5 rows):")
    print(pred_df[["open", "high", "low", "close", "volume"]].head())

    # 7. Plot
    kline_df = df.loc[:LOOKBACK + PRED_LEN - 1].copy()
    plot_prediction(TICKER, kline_df, pred_df, LOOKBACK)

"""
Kronos — US Stock Data Demo
============================
Two modes, one command:

  # Quick (offline, no downloads, ~10s)
  python run_us_stock_demo.py --mode quick

  # Full  (downloads real model + live data, requires internet)
  python run_us_stock_demo.py --mode full --ticker AAPL

Quick mode  : synthetic GBM data + tiny random-weight model.
              Proves the entire pipeline works with no internet access.

Full mode   : fetches real OHLCV from Yahoo Finance via yfinance,
              loads Kronos-small + Kronos-Tokenizer-base from HuggingFace
              (~500 MB first run, cached afterwards).

Run from the project root:
    python run_us_stock_demo.py [--mode quick|full] [--ticker AAPL]
                                [--lookback 400] [--pred-len 30]
                                [--no-plot] [--save-plot]
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from model import Kronos, KronosTokenizer, KronosPredictor


# ──────────────────────────────────────────────────────────────────────────────
# Tiny model config  (quick mode only)
# Same architecture as Kronos-small; tiny dims so it runs in seconds on CPU
# ──────────────────────────────────────────────────────────────────────────────
_TINY_TOKENIZER = dict(
    d_in=6, d_model=32, n_heads=2, ff_dim=64,
    n_enc_layers=2, n_dec_layers=2,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=8, s2_bits=8,
    beta=1.0, gamma0=0.0, gamma=0.1, zeta=1.0, group_size=4,
)
_TINY_MODEL = dict(
    s1_bits=8, s2_bits=8, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    token_dropout_p=0.0, learn_te=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic US stock data  (Geometric Brownian Motion)
# ──────────────────────────────────────────────────────────────────────────────
def make_synthetic_us_stock(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """
    Simulate daily OHLCV that looks like a US stock (~$150 base price).
    Intentionally omits the 'amount' column to mirror real US market data feeds.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu, sigma = 0.10, 0.20
    log_ret = rng.normal((mu - 0.5 * sigma ** 2) * dt,
                         sigma * math.sqrt(dt), n_bars)
    close = 150.0 * np.exp(np.cumsum(log_ret))
    spread = close * rng.uniform(0.005, 0.025, n_bars)
    high   = close + spread * rng.uniform(0.3, 0.7, n_bars)
    low    = close - spread * rng.uniform(0.3, 0.7, n_bars)
    open_  = low + (high - low) * rng.uniform(0.2, 0.8, n_bars)
    volume = rng.integers(1_000_000, 50_000_000, n_bars).astype(float)

    return pd.DataFrame({
        "timestamps": pd.bdate_range("2022-01-03", periods=n_bars),
        "open":   open_, "high": high, "low": low,
        "close":  close, "volume": volume,
        # no 'amount' — KronosPredictor fills it automatically
    })


# ──────────────────────────────────────────────────────────────────────────────
# Real US stock data via yfinance  (full mode)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit(
            "\n[ERROR] yfinance not installed.\n"
            "  Run:  pip install yfinance\n"
            "  Or use --mode quick to skip the download.\n"
        )

    print(f"  Downloading {ticker} from Yahoo Finance ({start} → {end}) …")
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        sys.exit(f"\n[ERROR] No data for '{ticker}'. Check the ticker symbol and date range.")

    # yfinance ≥ 0.2 sometimes returns MultiIndex columns
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame({
        "timestamps": pd.to_datetime(
            raw.index.tz_localize(None) if raw.index.tz else raw.index
        ),
        "open":   raw["Open"].values,
        "high":   raw["High"].values,
        "low":    raw["Low"].values,
        "close":  raw["Close"].values,
        "volume": raw["Volume"].values,
    }).dropna().reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Build predictor
# ──────────────────────────────────────────────────────────────────────────────
def build_predictor_quick() -> KronosPredictor:
    print("  Building tiny random-weight model …")
    tokenizer = KronosTokenizer(**_TINY_TOKENIZER).eval()
    model     = Kronos(**_TINY_MODEL).eval()
    return KronosPredictor(model, tokenizer, max_context=512)


def build_predictor_full() -> KronosPredictor:
    print("  Loading NeoQuasar/Kronos-Tokenizer-base from HuggingFace …")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    print("  Loading NeoQuasar/Kronos-small from HuggingFace …")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    return KronosPredictor(model, tokenizer, max_context=512)


# ──────────────────────────────────────────────────────────────────────────────
# Visualize
# ──────────────────────────────────────────────────────────────────────────────
def visualize(df: pd.DataFrame, pred_df: pd.DataFrame,
              lookback: int, ticker: str,
              show: bool, save_path: str | None):
    hist  = df.iloc[:lookback]
    truth = df.iloc[lookback:lookback + len(pred_df)]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)

    # ── Close price ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(hist["timestamps"],  hist["close"],
            label="Historical", color="steelblue", linewidth=1.5)
    ax.plot(truth["timestamps"], truth["close"],
            label="Ground Truth", color="green", linewidth=1.5, linestyle="--")
    ax.plot(pred_df.index,       pred_df["close"],
            label="Kronos Prediction", color="crimson", linewidth=1.8)
    ax.set_title(f"Kronos Forecast — {ticker} (Daily)", fontsize=13)
    ax.set_ylabel("Close Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Volume ───────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(truth["timestamps"], truth["volume"],
           label="Ground Truth Volume", color="green", alpha=0.5, width=0.8)
    ax.bar(pred_df.index,       pred_df["volume"],
           label="Predicted Volume", color="crimson", alpha=0.6, width=0.8)
    ax.set_ylabel("Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"\n  Plot saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Kronos US stock demo — quick (offline) or full (live data + real model)"
    )
    p.add_argument("--mode",      choices=["quick", "full"], default="quick",
                   help="quick: synthetic data + tiny model  |  full: yfinance + HuggingFace model")
    p.add_argument("--ticker",    default="AAPL",
                   help="US stock ticker (full mode only, default: AAPL)")
    p.add_argument("--start",     default="2022-01-01",
                   help="Start date for historical data (full mode, default: 2022-01-01)")
    p.add_argument("--end",       default="2024-12-31",
                   help="End date for historical data (full mode, default: 2024-12-31)")
    p.add_argument("--lookback",  type=int, default=400,
                   help="Historical bars used as model context (≤512, default: 400)")
    p.add_argument("--pred-len",  type=int, default=30,
                   help="Bars to forecast (default: 30)")
    p.add_argument("--no-plot",   action="store_true",
                   help="Skip matplotlib window (useful on headless servers)")
    p.add_argument("--save-plot", action="store_true",
                   help="Save plot as PNG next to this script")
    return p.parse_args()


def main():
    args = parse_args()
    lookback = args.lookback
    pred_len = args.pred_len

    print("\n" + "=" * 62)
    print(f"  Kronos US Stock Demo  |  mode={args.mode}")
    print("=" * 62)

    # ── 1. Build predictor ────────────────────────────────────────────────────
    print("\n[1/4] Loading model …")
    if args.mode == "quick":
        predictor = build_predictor_quick()
    else:
        predictor = build_predictor_full()
    print(f"  Device: {predictor.device}")

    # ── 2. Get data ───────────────────────────────────────────────────────────
    print("\n[2/4] Preparing data …")
    if args.mode == "quick":
        print(f"  Generating synthetic US stock data (GBM, {lookback + pred_len + 10} bars) …")
        df = make_synthetic_us_stock(lookback + pred_len + 10)
        ticker = "SYNTHETIC"
    else:
        df = fetch_yfinance(args.ticker, args.start, args.end)
        ticker = args.ticker.upper()

    need = lookback + pred_len
    if len(df) < need:
        sys.exit(
            f"\n[ERROR] Need at least {need} bars, got {len(df)}.\n"
            f"  Reduce --lookback or --pred-len, or widen the date range."
        )

    x_df        = df.iloc[:lookback][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_timestamp = df.iloc[:lookback]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[lookback:lookback + pred_len]["timestamps"].reset_index(drop=True)

    print(f"  Ticker      : {ticker}")
    print(f"  Total bars  : {len(df)}")
    print(f"  Context     : {lookback} bars  "
          f"({x_timestamp.iloc[0].date()} → {x_timestamp.iloc[-1].date()})")
    print(f"  Forecast    : {pred_len} bars  "
          f"({y_timestamp.iloc[0].date()} → {y_timestamp.iloc[-1].date()})")
    print(f"  Columns     : {list(x_df.columns)}  (no 'amount' — auto-filled)")

    # ── 3. Predict ────────────────────────────────────────────────────────────
    print("\n[3/4] Running Kronos inference …")
    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1,
            verbose=True,
        )

    # ── 4. Results ────────────────────────────────────────────────────────────
    print("\n[4/4] Results")
    print("-" * 62)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(pred_df[["open", "high", "low", "close", "volume"]].to_string())
    print("-" * 62)
    print(f"  Output shape : {pred_df.shape}")
    print(f"  Close range  : [{pred_df['close'].min():.2f}, {pred_df['close'].max():.2f}]")
    print(f"  NaN values   : {pred_df.isnull().values.any()}")
    print("\n  Pipeline completed successfully.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    show_plot = not args.no_plot
    save_path = None
    if args.save_plot:
        save_path = str(Path(__file__).parent / f"kronos_{ticker.lower()}_forecast.png")

    if show_plot or save_path:
        try:
            if not show_plot:
                matplotlib.use("Agg")   # headless backend for save-only
            visualize(df, pred_df, lookback, ticker, show=show_plot, save_path=save_path)
        except Exception as e:
            print(f"\n  [Note] Could not display plot: {e}")
            print("  Tip: use --no-plot --save-plot to save the image instead.")

    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()

"""
US Stock Data Pipeline Test — No HuggingFace Download Required

This test verifies that the full Kronos prediction pipeline works correctly
with US-style stock market data (daily OHLCV, no 'amount' column).

It uses:
  - Synthetic daily OHLCV data that mimics real US stock prices (GBM simulation)
  - A tiny Kronos model/tokenizer with random weights (same architecture, minimal dims)

No internet access or large model downloads are needed.

Run from the project root:
    python -m pytest tests/test_us_stock_pipeline.py -v
or directly:
    python tests/test_us_stock_pipeline.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import Kronos, KronosTokenizer, KronosPredictor


# ─────────────────────────────────────────────────────────
# Tiny model config — matches real architecture, tiny dims
# ─────────────────────────────────────────────────────────
TINY_TOKENIZER_CFG = dict(
    d_in=6,           # open, high, low, close, volume, amount
    d_model=32,
    n_heads=2,
    ff_dim=64,
    n_enc_layers=2,
    n_dec_layers=2,
    ffn_dropout_p=0.0,
    attn_dropout_p=0.0,
    resid_dropout_p=0.0,
    s1_bits=8,        # vocab_size = 2^8 = 256
    s2_bits=8,
    beta=1.0,
    gamma0=0.0,
    gamma=0.1,
    zeta=1.0,
    group_size=4,
)

TINY_MODEL_CFG = dict(
    s1_bits=8,
    s2_bits=8,
    n_layers=2,
    d_model=32,
    n_heads=2,
    ff_dim=64,
    ffn_dropout_p=0.0,
    attn_dropout_p=0.0,
    resid_dropout_p=0.0,
    token_dropout_p=0.0,
    learn_te=False,
)

LOOKBACK = 60     # 60 trading days of history as context
PRED_LEN = 10     # predict next 10 trading days


# ─────────────────────────────────────────────────────────
# Synthetic US stock data generator (Geometric Brownian Motion)
# ─────────────────────────────────────────────────────────
def make_us_stock_df(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic daily OHLCV data that mimics a US stock.

    Key difference from Chinese A-share examples:
      - No 'amount' column (KronosPredictor must handle this automatically)
      - Timestamps are business days (no intraday minutes)
      - Prices in USD range (100–300)
    """
    rng = np.random.default_rng(seed)

    # Simulate close prices via GBM
    dt = 1 / 252
    mu, sigma = 0.10, 0.20
    log_returns = rng.normal((mu - 0.5 * sigma ** 2) * dt, sigma * math.sqrt(dt), n_bars)
    close = 150.0 * np.exp(np.cumsum(log_returns))

    # Build OHLCV from close
    daily_range = close * rng.uniform(0.005, 0.025, n_bars)
    high  = close + daily_range * rng.uniform(0.3, 0.7, n_bars)
    low   = close - daily_range * rng.uniform(0.3, 0.7, n_bars)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, n_bars)
    volume = rng.integers(1_000_000, 50_000_000, n_bars).astype(float)

    dates = pd.bdate_range(start="2023-01-03", periods=n_bars)  # US business days

    df = pd.DataFrame({
        "timestamps": dates,
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
        # NOTE: intentionally NO 'amount' column — this is typical for US market data
    })
    return df


# ─────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────
def build_predictor(device="cpu") -> KronosPredictor:
    tokenizer = KronosTokenizer(**TINY_TOKENIZER_CFG).eval()
    model = Kronos(**TINY_MODEL_CFG).eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=512)


def test_us_stock_no_amount_column():
    """
    Verify KronosPredictor accepts US stock data that has no 'amount' column.
    The predictor should silently fill it in (volume * mean_price).
    """
    predictor = build_predictor()
    df = make_us_stock_df(n_bars=LOOKBACK + PRED_LEN + 5)

    x_df        = df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_timestamp = df.iloc[:LOOKBACK]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]["timestamps"].reset_index(drop=True)

    assert "amount" not in x_df.columns, "Test setup error: 'amount' must be absent"

    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    assert pred_df.shape == (PRED_LEN, 6), (
        f"Expected ({PRED_LEN}, 6) output, got {pred_df.shape}"
    )
    assert list(pred_df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert not pred_df.isnull().values.any(), "Prediction contains NaN values"
    assert (pred_df["close"] > 0).all(), "Predicted close prices should be positive"

    print(f"\n[PASS] test_us_stock_no_amount_column")
    print(f"  Input : {x_df.shape[0]} bars, columns = {list(x_df.columns)}")
    print(f"  Output: {pred_df.shape[0]} bars, columns = {list(pred_df.columns)}")
    print(f"  Predicted close range: [{pred_df['close'].min():.2f}, {pred_df['close'].max():.2f}]")


def test_us_stock_daily_timestamps():
    """
    Verify that daily business-day timestamps (no intraday minutes) are handled
    correctly. The time-feature extractor uses weekday/day/month, all of which
    are meaningful for daily bars.
    """
    predictor = build_predictor()
    df = make_us_stock_df(n_bars=LOOKBACK + PRED_LEN + 5)

    x_df        = df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    x_timestamp = df.iloc[:LOOKBACK]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]["timestamps"].reset_index(drop=True)

    # Timestamps should be date-only (no time component)
    assert x_timestamp.dt.hour.sum() == 0, "Expected date-only timestamps"

    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    assert pred_df.shape[0] == PRED_LEN
    print(f"\n[PASS] test_us_stock_daily_timestamps")
    print(f"  Timestamp range: {x_timestamp.iloc[0].date()} → {y_timestamp.iloc[-1].date()}")


def test_us_stock_ohlcv_only():
    """
    Verify the pipeline works with only OHLCV (no volume either), i.e., the
    minimal required columns: open, high, low, close.
    """
    predictor = build_predictor()
    df = make_us_stock_df(n_bars=LOOKBACK + PRED_LEN + 5)

    x_df        = df.iloc[:LOOKBACK][["open", "high", "low", "close"]].reset_index(drop=True)
    x_timestamp = df.iloc[:LOOKBACK]["timestamps"].reset_index(drop=True)
    y_timestamp = df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]["timestamps"].reset_index(drop=True)

    with torch.no_grad():
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=PRED_LEN,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    assert pred_df.shape == (PRED_LEN, 6)
    assert not pred_df.isnull().values.any()
    print(f"\n[PASS] test_us_stock_ohlcv_only")
    print(f"  Input columns: {list(x_df.columns)} (no volume/amount)")
    print(f"  Output columns: {list(pred_df.columns)}")


def test_us_stock_batch_prediction():
    """
    Verify predict_batch works with multiple US stock series simultaneously.
    All series have the same lookback and pred_len (required by predict_batch).
    """
    predictor = build_predictor()
    n_stocks = 3

    dfs, x_timestamps, y_timestamps = [], [], []
    for i in range(n_stocks):
        df = make_us_stock_df(n_bars=LOOKBACK + PRED_LEN + 5, seed=i)
        dfs.append(df.iloc[:LOOKBACK][["open", "high", "low", "close", "volume"]].reset_index(drop=True))
        x_timestamps.append(df.iloc[:LOOKBACK]["timestamps"].reset_index(drop=True))
        y_timestamps.append(df.iloc[LOOKBACK:LOOKBACK + PRED_LEN]["timestamps"].reset_index(drop=True))

    with torch.no_grad():
        pred_dfs = predictor.predict_batch(
            df_list=dfs,
            x_timestamp_list=x_timestamps,
            y_timestamp_list=y_timestamps,
            pred_len=PRED_LEN,
            T=1.0,
            top_k=1,
            top_p=1.0,
            sample_count=1,
            verbose=False,
        )

    assert len(pred_dfs) == n_stocks, f"Expected {n_stocks} results, got {len(pred_dfs)}"
    for i, pred_df in enumerate(pred_dfs):
        assert pred_df.shape == (PRED_LEN, 6), f"Stock {i}: wrong output shape {pred_df.shape}"
        assert not pred_df.isnull().values.any(), f"Stock {i}: NaN in predictions"

    print(f"\n[PASS] test_us_stock_batch_prediction")
    print(f"  Predicted {n_stocks} US stocks simultaneously, each {PRED_LEN} bars forward")


# ─────────────────────────────────────────────────────────
# Run directly (without pytest)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Kronos US Stock Data Pipeline Test")
    print("=" * 60)
    print(f"  Tiny model: d_model={TINY_MODEL_CFG['d_model']}, "
          f"n_layers={TINY_MODEL_CFG['n_layers']}, "
          f"s1_bits={TINY_MODEL_CFG['s1_bits']}, s2_bits={TINY_MODEL_CFG['s2_bits']}")
    print(f"  Lookback={LOOKBACK} bars, pred_len={PRED_LEN} bars, device=cpu\n")

    tests = [
        test_us_stock_no_amount_column,
        test_us_stock_daily_timestamps,
        test_us_stock_ohlcv_only,
        test_us_stock_batch_prediction,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Result: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("All tests PASSED — US stock data pipeline is fully functional.")
    print("=" * 60)
    sys.exit(0 if passed == len(tests) else 1)

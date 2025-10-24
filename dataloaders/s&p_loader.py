#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# --- Config ---
OUTPUT_PATH = Path("data/sp500_ohlcv_returns.csv")
TICKER = "^GSPC"   # Yahoo Finance symbol for S&P 500 index
START = "2023-01-01"
END = None  # None = today

# --- Fetch data ---
print(f"Downloading {TICKER} from Yahoo Finance...")
spx = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)

# Check the structure of the data
print(f"Data shape: {spx.shape}")
print(f"Columns: {spx.columns.tolist()}")

# If we have multi-level columns, flatten them
if isinstance(spx.columns, pd.MultiIndex):
    spx.columns = spx.columns.get_level_values(0)

# --- Compute additional metrics ---
spx["Daily_Return"] = spx["Close"].pct_change()  # Use "Close" instead of "Adj Close"
spx["Log_Return"] = np.log(spx["Close"] / spx["Close"].shift(1))
spx["High_Low_Range"] = (spx["High"] - spx["Low"]) / spx["Close"]
spx["Open_Close_Range"] = abs(spx["Open"] - spx["Close"]) / spx["Close"]

# Volume analysis
spx["Volume_MA_20"] = spx["Volume"].rolling(window=20).mean()
spx["Volume_Ratio"] = spx["Volume"] / spx["Volume_MA_20"]

# Moving averages
spx["Price_MA_20"] = spx["Close"].rolling(window=20).mean()
spx["Price_MA_50"] = spx["Close"].rolling(window=50).mean()

# Volatility
spx["Volatility_20"] = spx["Daily_Return"].rolling(20).std()

# Keep all OHLCV columns plus computed metrics
columns_to_keep = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Log_Return", "High_Low_Range", "Open_Close_Range",
    "Volume_MA_20", "Volume_Ratio", "Price_MA_20", "Price_MA_50", "Volatility_20"
]

# Filter to available columns
available_columns = [col for col in columns_to_keep if col in spx.columns]
daily = spx[available_columns].dropna().copy()
daily.index.name = "Date (UTC)"

# --- Save ---
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
daily.to_csv(OUTPUT_PATH)
print(f"✅ Saved OHLCV + returns to: {OUTPUT_PATH.resolve()}")

# --- Preview ---
print("\n📊 Data Preview:")
print(daily.tail())

print(f"\n📈 Data Summary:")
print(f"Records: {len(daily)}")
print(f"Date range: {daily.index.min()} to {daily.index.max()}")
print(f"Average daily return: {daily['Daily_Return'].mean():.4f}")
print(f"Volatility (std): {daily['Daily_Return'].std():.4f}")
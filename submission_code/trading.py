"""
trading.py

Utility functions for:


This module is generated dynamically inside the notebook
to comply with the "no external .py files" requirement.
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

def split(df, frac=0.8):
    df = df
    split = int(len(df) * frac)
    train_idx = df.index[:split]
    test_idx  = df.index[split:]
    return train_idx, test_idx

def sharpe(series):
    s = series.dropna()
    return (np.sqrt(252) * s.mean() / (s.std() + 1e-12)) if len(s) else np.nan

# Backtest helper
def backtest_slice(df, idx, sig_series_or_name, ret_col='intraday_ret'):
    s = df.loc[idx, sig_series_or_name] if isinstance(sig_series_or_name, str) else sig_series_or_name.loc[idx]
    r = s.astype(float) * df.loc[idx, ret_col]
    cum = (1 + r.fillna(0)).cumprod()
    return r, cum



"""
trading.py

Utility functions for:


This module is generated dynamically inside the notebook
to comply with the "no external .py files" requirement.
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

"""
trading.py

Small utility helpers used by the notebooks:

- `split`: Create train/test index splits by fraction, preserving order.
- `sharpe`: Compute annualized Sharpe ratio from a return series.
- `backtest_slice`: Turn a trading signal into strategy returns and a
  cumulative return curve over a given index slice.

Notes
-----
This module can be generated dynamically inside a notebook to comply with
"no external .py files" constraints. The functions are intentionally minimal
and avoid side effects so they can be reused across experiments.
"""

def split(df, frac=0.8):
    """Return train/test index splits by fraction, preserving order.

    Parameters
    - df: DataFrame whose index defines the timeline to split.
    - frac: Fraction of samples allocated to the train split (0..1).

    Returns
    - (train_idx, test_idx): Tuple of index slices referencing ``df``.

    Notes
    - No shuffling is performed; suitable for time series where order matters.
    """
    df = df
    split = int(len(df) * frac)
    train_idx = df.index[:split]
    test_idx  = df.index[split:]
    return train_idx, test_idx

def sharpe(series):
    """Annualized Sharpe ratio (daily sampling assumed).

    Parameters
    - series: Return series (e.g., daily strategy returns).

    Returns
    - float Sharpe ratio scaled by ``sqrt(252)``. Returns ``NaN`` if empty.

    Notes
    - Adds a small epsilon in the denominator to avoid division by zero.
    - If your sampling frequency differs, change the scaling factor.
    """
    s = series.dropna()
    return (np.sqrt(252) * s.mean() / (s.std() + 1e-12)) if len(s) else np.nan

# Backtest helper
def backtest_slice(df, idx, sig_series_or_name, ret_col='intraday_ret'):
    """Compute strategy and cumulative returns over a slice of ``df``.

    Parameters
    - df: DataFrame containing at least the return column ``ret_col``.
    - idx: Index or index-like selector defining the evaluation slice.
    - sig_series_or_name: Either the name of a column in ``df`` or an
      external Series aligned to ``df.index`` that encodes the trading
      signal (e.g., -1/0/1 or fractional exposure).
    - ret_col: Name of the per-period return column in ``df``.

    Returns
    - r: Series of per-period strategy returns on ``idx``.
    - cum: Series of cumulative returns (starting at 1.0) on ``idx``.

    Important
    - Ensure your signal does not leak future information. If the signal
      is computed from same-day data, consider shifting it by one period
      before backtesting to avoid look-ahead bias.
    - Missing signal values are propagated into ``r``; for the cumulative
      curve they are treated as zero return for that period.
    Example
    -------
    # Avoid look-ahead by acting next period
    sig_safe = df['signal_markov_oos'].shift(1)
    r, cum = backtest_slice(df, idx=test_idx, sig_series_or_name=sig_safe)
    """
    # Resolve the signal series from a column name or an external Series
    if isinstance(sig_series_or_name, str):
        signal = df.loc[idx, sig_series_or_name]
    else:
        signal = sig_series_or_name.loc[idx]

    # Underlying per-period returns from the dataframe
    asset_returns = df.loc[idx, ret_col]

    # Per-period strategy returns: position × underlying returns
    strategy_returns = signal.astype(float) * asset_returns

    # Cumulative return curve (1.0 baseline). NaNs treated as 0 return.
    cumulative = (1 + strategy_returns.fillna(0)).cumprod()

    return strategy_returns, cumulative


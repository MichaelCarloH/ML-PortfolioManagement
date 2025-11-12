
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional
import pandas as pd
import numpy as np

def plot_price_and_cumrets(
    df,
    idx,
    price_col='Close',
    trade_specs=None,
    curve_specs=None,
    title_prefix='TRAIN',
    thresh=None,
    price_kwargs=None,
    grid_alpha=0.2,
    figsize=(14, 6),
):
    if price_kwargs is None:
        price_kwargs = {
            'label': price_col,
            'color': 'tab:blue',
            'linewidth': 1.5,
            'alpha': 0.9
        }

    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(2, 1, 1)

    price = df.loc[idx, price_col]
    ax1.plot(price.index, price, **price_kwargs)

    if trade_specs:
        for spec in trade_specs:
            col_or_series = spec.get('col') or spec.get('series')
            if isinstance(col_or_series, str):
                s = df.loc[idx, col_or_series]
            else:
                s = col_or_series.loc[idx]

            long_val = spec.get('long_val', 1)
            short_val = spec.get('short_val', -1)

            long_idx = s.index[s == long_val]
            short_idx = s.index[s == short_val]

            long_kwargs = {
                'marker': 'o',
                'facecolors': 'none',
                'edgecolors': 'tab:blue',
                's': 64,
                'linewidths': 1.5
            }
            long_kwargs.update(spec.get('long_kwargs', {}))

            short_kwargs = {
                'marker': 'X',
                'color': 'tab:orange',
                's': 56
            }
            short_kwargs.update(spec.get('short_kwargs', {}))

            long_label = spec.get('long_label', 'Long')
            short_label = spec.get('short_label', 'Short')

            ax1.scatter(long_idx, df.loc[long_idx, price_col], label=long_label, **long_kwargs)
            ax1.scatter(short_idx, df.loc[short_idx, price_col], label=short_label, **short_kwargs)

    ttl = f'{title_prefix}: Price with trades'
    if thresh is not None:
        ttl += f' (thresh={thresh})'
    ax1.set_title(ttl)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=grid_alpha)

    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    if curve_specs:
        for spec in curve_specs:
            series = spec['series']
            label = spec.get('label')
            plot_kwargs = spec.get('plot_kwargs', {})
            ax2.plot(series.index, series, label=label, **plot_kwargs)

    ax2.legend()
    ax2.grid(alpha=grid_alpha)
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_candles_and_trades(
    data: pd.DataFrame,
    price_cols: dict = None,
    signal_col: Optional[str] = None,
    thresh: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (14, 6),
    body_width: float = 0.6,
    color_up: str = "#2ca02c",
    color_down: str = "#d62728",
    wick_color: str = "black",
    wick_width: float = 0.8,
    wick_alpha: float = 0.6,
    show_trades: bool = True,
) -> tuple[plt.Figure, plt.Axes, np.ndarray]:
    """
    Generalized plotting function for candles + optional trade signals.

    Parameters
    ----------
    data : pd.DataFrame
        Must include at least 'Open', 'Close' columns. Optionally 'High', 'Low' for wicks.
        If `signal_col` is provided, its values (+1/-1) will be used to plot trade markers.
    price_cols : dict, optional
        Custom column mapping, e.g. {'Open': 'O', 'High': 'H', 'Low': 'L', 'Close': 'C'}.
    signal_col : str, optional
        Column name with trade signals (+1 for long, -1 for short).
    thresh : float, optional
        Threshold value to display in title.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. Creates new figure if None.
    figsize : tuple
        Figure size if ax is None.
    show_trades : bool
        Whether to plot trade markers (requires `signal_col`).
    """
    # Setup
    if price_cols is None:
        price_cols = {"Open": "Open", "High": "High", "Low": "Low", "Close": "Close"}

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    o = data[price_cols["Open"]].values
    c = data[price_cols["Close"]].values
    has_hl = {"High", "Low"}.issubset(price_cols) and all(
        col in data.columns for col in (price_cols["High"], price_cols["Low"])
    )
    if has_hl:
        h = data[price_cols["High"]].values
        l = data[price_cols["Low"]].values

    x = np.arange(len(data))
    colors = np.where(c >= o, color_up, color_down)

    # Wicks
    if has_hl:
        for xi, hi, lo in zip(x, h, l):
            ax.vlines(xi, lo, hi, color=wick_color, linewidth=wick_width, alpha=wick_alpha)

    # Bodies
    prange = float(data[price_cols["Close"]].max() - data[price_cols["Close"]].min())
    min_body = prange * 0.002 if prange > 0 else 1e-6
    for xi, oi, ci, col in zip(x, o, c, colors):
        lower = min(oi, ci)
        height = max(abs(ci - oi), min_body)
        rect = Rectangle(
            (xi - body_width / 2, lower),
            body_width,
            height,
            facecolor=col,
            edgecolor=col,
            linewidth=0.8,
            alpha=0.85,
        )
        ax.add_patch(rect)

    # Plot trades if requested
    if show_trades and signal_col in data.columns:
        sig = data[signal_col]
        pnl = sig * data.get("intraday_ret", 0)  # assumes intraday_ret exists
        mk_long = data[sig == 1]
        mk_short = data[sig == -1]

        pos_map = {idx: i for i, idx in enumerate(data.index)}

        # Colors based on profit/loss
        long_colors = [color_up if pnl.loc[i] > 0 else color_down for i in mk_long.index]
        short_colors = [color_up if pnl.loc[i] > 0 else color_down for i in mk_short.index]

        # Marker positions
        if has_hl:
            pad = (data[price_cols["High"]] - data[price_cols["Low"]]).median() * 0.05
            y_long = data.loc[mk_long.index, price_cols["High"]] + pad
            y_short = data.loc[mk_short.index, price_cols["Low"]] - pad
        else:
            pad = data[price_cols["Close"]].median() * 0.003
            y_long = data.loc[mk_long.index, price_cols["Close"]] * (1 + 0.003)
            y_short = data.loc[mk_short.index, price_cols["Close"]] * (1 - 0.003)

        ax.scatter(
            [pos_map[i] for i in mk_long.index],
            y_long,
            marker="o",
            facecolors="none",
            edgecolors=long_colors,
            s=64,
            linewidths=1.5,
            label="Long (profit=green, loss=red)",
            zorder=3,
        )
        ax.scatter(
            [pos_map[i] for i in mk_short.index],
            y_short,
            marker="X",
            c=short_colors,
            s=56,
            label="Short (profit=green, loss=red)",
            zorder=3,
        )

    # Cosmetics
    ax.set_xlim(-1, len(data))
    ticks = x[::max(1, len(x) // 10)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([pd.to_datetime(d).date() for d in data.index[ticks]])
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)

    if title:
        if thresh is not None:
            title = f"{title} (THRESH={thresh})"
        ax.set_title(title)
    if show_trades and signal_col in data.columns:
        ax.legend(loc="upper left")

    return fig, ax, x

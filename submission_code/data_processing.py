"""
data_processing.py

Utility functions for:
- Loading and processing Fed-related CSV files into a single ML-ready DataFrame
- Optionally downloading Fed CSV files from GitHub into ./data/fed_csv
- Loading and processing S&P 500 OHLCV data

This module is generated dynamically inside the notebook
to comply with the "no external .py files" requirement.
"""

import os
import glob
import re
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Local directory where Fed CSV files are stored
DATA_DIR = "data/fed_csv"

# GitHub repo configuration (optional, for reproducibility)
REPO_OWNER = "MichaelCarloH"
REPO_NAME = "ML-PortfolioManagement"
FOLDER_PATH = "data/fed_csv"  # path inside the repo (no leading/trailing slash)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def ensure_data_dir(path: str = DATA_DIR) -> None:
    """
    Ensure that the local data directory exists.
    """
    os.makedirs(path, exist_ok=True)


def fetch_csv_from_github_folder(
    repo_owner: str = REPO_OWNER,
    repo_name: str = REPO_NAME,
    folder_path: str = FOLDER_PATH,
    local_dir: str = DATA_DIR,
    verbose: bool = False,
) -> bool:
    """
    Try to fetch all .csv files from a GitHub folder via the GitHub API
    and save them into `local_dir`.

    Returns
    -------
    bool
        True if at least one CSV was downloaded, False otherwise.
    """
    ensure_data_dir(local_dir)

    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"
    response = requests.get(api_url)

    if response.status_code != 200:
        _log(
            f"[INFO] Could not access GitHub folder ({response.status_code}). "
            "Proceeding without remote download.",
            verbose,
        )
        return False

    items = response.json()
    csv_files = [item for item in items if item["name"].endswith(".csv")]

    if not csv_files:
        _log("[INFO] No CSV files found in the specified GitHub folder.", verbose)
        return False

    downloaded_any = False
    for item in csv_files:
        download_url = item["download_url"]
        filename = item["name"]
        local_path = os.path.join(local_dir, filename)

        file_response = requests.get(download_url)
        if file_response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(file_response.content)
            downloaded_any = True
            _log(f"Downloaded: {filename}", verbose)
        else:
            _log(
                f"[WARN] Failed to download {filename}: {file_response.status_code}",
                verbose,
            )

    return downloaded_any


# ---------------------------------------------------------------------
# Fed CSV processing
# ---------------------------------------------------------------------

def load_all_csvs(
    local_dir: str = DATA_DIR,
    pattern: str = "*.csv",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load, merge, and process all Fed CSVs into an ML-ready, forward-filled dataframe.

    Steps:
    - Reads all CSVs in `local_dir` matching `pattern`.
    - Extracts date and outcome columns.
    - Merges all files into one unified dataframe.
    - Converts to ML-wide format with clean feature names.
    - Saves:
        - data/fed_events_merged.csv
        - data/fed_events_ml_ready.csv
        - data/fed_events_ml_ready_ffill.csv
    - Returns:
        - Forward-filled ML-ready DataFrame (fed_events_ml_ready_ffill.csv).
    """

    def log(msg: str):
        if verbose:
            print(msg)

    ensure_data_dir(local_dir)

    # --- Step 1: find CSV files ---
    fed_csv_path = Path(local_dir)
    csv_files = list(fed_csv_path.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {fed_csv_path} matching {pattern}")
    log(f"Found {len(csv_files)} CSV files")

    # --- Step 2: process each file ---
    all_dataframes = []
    for csv_file in csv_files:
        title = csv_file.stem
        log(f"Processing: {title}")

        try:
            df = pd.read_csv(csv_file)

            # Identify date column
            date_col = next(
                (c for c in df.columns if "date" in c.lower() and "utc" in c.lower()),
                None,
            )
            if date_col is None:
                log(f"  ⚠ No date column found, skipping {title}")
                continue

            # Identify outcome columns
            outcome_cols = [
                c for c in df.columns
                if c != date_col and "timestamp" not in c.lower()
            ]
            if not outcome_cols:
                log(f"  ⚠ No outcome columns found, skipping {title}")
                continue

            # Convert data types
            df[date_col] = pd.to_datetime(
                df[date_col],
                format="%m-%d-%Y %H:%M",
                errors="coerce",
            )
            for c in outcome_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            # Merge same-date rows: first non-null per column
            merged_df = df.groupby(date_col).agg({
                c: (lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else None)
                for c in outcome_cols
            }).reset_index()

            merged_df["Title"] = title
            merged_df = merged_df.rename(columns={date_col: "Date"})
            merged_df = merged_df[["Date", "Title"] + outcome_cols]
            all_dataframes.append(merged_df)
            log(f"  ✓ {len(merged_df)} unique dates")

        except Exception as e:
            log(f"  ✗ Error processing {title}: {e}")

    if not all_dataframes:
        raise RuntimeError("No valid CSVs processed.")

    # --- Step 3: merge all dataframes ---
    log("Merging all dataframes...")
    all_outcome_cols = sorted(list({
        c
        for df in all_dataframes
        for c in df.columns
        if c not in ["Date", "Title"]
    }))

    final_dfs = []
    for df in all_dataframes:
        for c in all_outcome_cols:
            if c not in df.columns:
                df[c] = None
        final_dfs.append(df[["Date", "Title"] + all_outcome_cols])

    final_df = (
        pd.concat(final_dfs, ignore_index=True)
        .sort_values(["Date", "Title"])
        .reset_index(drop=True)
    )

    os.makedirs("data", exist_ok=True)
    merged_path = Path("data/fed_events_merged.csv")
    final_df.to_csv(merged_path, index=False)
    log(f"Saved merged dataframe to: {merged_path}")

    # --- Step 4: long format ---
    id_vars = ["Date", "Title"]
    value_cols = [c for c in final_df.columns if c not in id_vars]

    df_long = pd.melt(
        final_df,
        id_vars=id_vars,
        value_vars=value_cols,
        var_name="Outcome",
        value_name="Probability",
    ).dropna(subset=["Probability"])

    # --- Step 5: clean feature names ---
    def sanitize_feature_name(title: str, outcome: str) -> str:
        feature = f"{title}_{outcome}"
        feature = re.sub(r"[^a-zA-Z0-9_\s]", "", feature)
        feature = re.sub(r"\s+", "_", feature)
        feature = re.sub(r"_+", "_", feature).strip("_")
        return feature

    df_long["Feature"] = df_long.apply(
        lambda r: sanitize_feature_name(r["Title"], r["Outcome"]),
        axis=1,
    )

    # --- Step 6: pivot to ML-wide ---
    df_ml = df_long.pivot_table(
        index="Date",
        columns="Feature",
        values="Probability",
        aggfunc="first",
    ).reset_index()

    cols = ["Date"] + sorted([c for c in df_ml.columns if c != "Date"])
    df_ml = df_ml[cols]

    ml_ready_path = Path("data/fed_events_ml_ready.csv")
    df_ml.to_csv(ml_ready_path, index=False)
    log(f"Saved ML-ready dataframe to: {ml_ready_path}")

    # --- Step 7: forward-fill version ---
    df_ml_ffill = df_ml.sort_values("Date").copy()
    df_ml_ffill.iloc[:, 1:] = df_ml_ffill.iloc[:, 1:].ffill().fillna(0)

    ffill_path = Path("data/fed_events_ml_ready_ffill.csv")
    df_ml_ffill.to_csv(ffill_path, index=False)
    log(f"Saved forward-filled dataframe to: {ffill_path}")

    # --- Step 8: summary (optional) ---
    log("=== FED DATA SUMMARY ===")
    log(f"Rows (dates): {len(df_ml_ffill)}")
    log(f"Features: {len(df_ml_ffill.columns) - 1}")
    log(
        f"Date range: {df_ml_ffill['Date'].min()} → {df_ml_ffill['Date'].max()}"
    )

    return df_ml_ffill


def get_fed_data(verbose: bool = False) -> pd.DataFrame:
    """
    High-level convenience function for the notebook.

    - Tries to load processed data from local ./data/fed_csv.
    - If no raw CSVs exist, optionally tries to download them from GitHub.
    - Returns the final forward-filled ML-ready dataframe.
    """
    ensure_data_dir(DATA_DIR)

    # 1) Check for existing local CSVs
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    # 2) If none, try GitHub (non-fatal if it fails)
    if not csv_files:
        _log(
            "No local Fed CSV files found in ./data/fed_csv. "
            "Attempting to download from GitHub...",
            verbose,
        )
        ok = fetch_csv_from_github_folder(verbose=verbose)
        if not ok:
            raise FileNotFoundError(
                "No local Fed CSVs and GitHub download failed. "
                "Ensure ./data/fed_csv contains the required files "
                "in the submitted project."
            )

    # 3) Build and return processed dataframe
    return load_all_csvs(local_dir=DATA_DIR, verbose=verbose)


# ---------------------------------------------------------------------
# S&P 500 loader
# ---------------------------------------------------------------------

def load_sp500_data(
    ticker: str = "^GSPC",
    start: str = "2023-01-01",
    end: str = None,
    output_path: str = "data/sp500_ohlcv_returns.csv",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Download, process, and save S&P 500 OHLCV and derived metrics.

    Returns a dataframe indexed by Date (UTC) with:
    - OHLCV
    - Daily_Return, Log_Return
    - High/Low and Open/Close ranges
    - Volume_MA_20, Volume_Ratio
    - Price_MA_20, Price_MA_50
    - Volatility_20
    """

    def log(msg: str):
        if verbose:
            print(msg)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Downloading {ticker} from Yahoo Finance...")
    spx = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if spx.empty:
        raise RuntimeError(f"No data returned for {ticker} from Yahoo Finance.")

    # Flatten multi-index columns if present
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Metrics
    spx["Daily_Return"] = spx["Close"].pct_change()
    spx["Log_Return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["High_Low_Range"] = (spx["High"] - spx["Low"]) / spx["Close"]
    spx["Open_Close_Range"] = (spx["Open"] - spx["Close"]).abs() / spx["Close"]
    spx["Volume_MA_20"] = spx["Volume"].rolling(20).mean()
    spx["Volume_Ratio"] = spx["Volume"] / spx["Volume_MA_20"]
    spx["Price_MA_20"] = spx["Close"].rolling(20).mean()
    spx["Price_MA_50"] = spx["Close"].rolling(50).mean()
    spx["Volatility_20"] = spx["Daily_Return"].rolling(20).std()

    cols_to_keep = ["Daily_Return", "Open", "Close"]
    # cols_to_keep = [
    #     "Open", "High", "Low", "Close", "Volume",
    #     "Daily_Return", "Log_Return",
    #     "High_Low_Range", "Open_Close_Range",
    #     "Volume_MA_20", "Volume_Ratio",
    #     "Price_MA_20", "Price_MA_50", "Volatility_20",
    # ]
    available = [c for c in cols_to_keep if c in spx.columns]

    daily = spx[available].dropna().copy()
    daily.index.name = "Date (UTC)"

    daily.to_csv(output_path)
    log(f"Saved S&P 500 data to: {output_path.resolve()}")

    if verbose:
        log(f"Rows: {len(daily)}")
        log(f"Date range: {daily.index.min()} → {daily.index.max()}")

    return daily

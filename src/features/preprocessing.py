import pandas as pd
import numpy as np
import re

DATETIME_PATTERNS = [r'date', r'time', r'timestamp', r'period', r'ds']
TARGET_PATTERNS = [r'target', r'value', r'y', r'consumption', r'load', r'count', r'amount']

def load_and_prepare(filepath, datetime_col, target_col):
    """
    Load a CSV, parse datetime, set index, sort, and fill missing values.
    """
    df = pd.read_csv(filepath)
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)
    df = df.sort_index()
    df[target_col] = df[target_col].fillna(method='ffill')
    return df

def add_time_features(df):
    """
    Add year, month, day, weekday, lag, and rolling features.
    """
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['lag1'] = df.iloc[:,0].shift(1)
    df['rolling3'] = df.iloc[:,0].rolling(window=3).mean()
    df['rolling12'] = df.iloc[:,0].rolling(window=12).mean()
    return df

def detect_datetime_column(df):
    # Try to find a column with common datetime patterns
    for col in df.columns:
        for pat in DATETIME_PATTERNS:
            if re.search(pat, col, re.IGNORECASE):
                return col
    # Fallback: try to parse each column as datetime and pick the one with most valid values
    best_col = None
    max_valid = 0
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors='coerce')
        valid = parsed.notna().sum()
        if valid > max_valid and valid > 0:
            best_col = col
            max_valid = valid
    # If index looks like datetime, use it
    if best_col is None and pd.api.types.is_datetime64_any_dtype(df.index):
        return df.index.name or 'index'
    return best_col

def detect_target_column(df, datetime_col):
    # Exclude datetime column, pick by common target patterns
    for col in df.columns:
        if col == datetime_col:
            continue
        for pat in TARGET_PATTERNS:
            if re.search(pat, col, re.IGNORECASE):
                return col
    # Fallback: pick the first numeric column
    for col in df.columns:
        if col == datetime_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            return col
    # Fallback: pick the first non-datetime column
    for col in df.columns:
        if col != datetime_col:
            return col
    return None

def load_and_prepare_auto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly detect datetime and target columns, standardize names, parse types, and set index.
    """
    if df.shape[1] < 2 and not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("Dataset must have at least two columns (datetime, value).")
    df = df.copy()
    datetime_col = detect_datetime_column(df)
    if not datetime_col:
        raise ValueError("Could not detect a datetime column. Please ensure your file has a date, time, or timestamp column.")
    target_col = detect_target_column(df, datetime_col)
    if not target_col:
        raise ValueError("Could not detect a target column. Please ensure your file has a numeric value column (e.g., target, value, y, consumption, load, count, amount).")
    # Standardize column names
    if datetime_col in df.columns:
        df = df.rename(columns={datetime_col: 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df = df.set_index('datetime')
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        df.index.name = 'datetime'
    else:
        raise ValueError("Could not parse a valid datetime column or index.")
    df = df.rename(columns={target_col: 'consumption'})
    df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
    df = df.dropna(subset=['consumption'])
    df = df.sort_index()
    return df

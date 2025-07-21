import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def profile_data(df):
    # Ensure datetime is the index
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    # Data types and columns
    dtypes = df.dtypes.apply(str).to_dict()
    columns = df.columns.tolist()
    # Missing and duplicate values
    missing = df.isnull().sum().to_dict()
    duplicate_rows = int(df.duplicated().sum())
    # Index checks
    monotonic_increasing = df.index.is_monotonic_increasing
    freq = None
    is_regular = False
    if isinstance(df.index, pd.DatetimeIndex):
        freq = pd.infer_freq(df.index)
        is_regular = freq is not None
    else:
        freq = 'Not a DatetimeIndex'
    return {
        'dtypes': dtypes,
        'columns': columns,
        'missing': missing,
        'duplicate_rows': duplicate_rows,
        'monotonic_increasing': monotonic_increasing,
        'datetime_index_freq': freq,
        'datetime_index_is_regular': is_regular
    }

def calendar_heatmap(df):
    # Calendar heatmap for time coverage (monthly)
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    df['year'] = df.index.year
    df['month'] = df.index.month
    pivot = df.pivot_table(index='year', columns='month', values='consumption', aggfunc='count')
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(pivot, cmap='Blues', cbar_kws={'label': 'Observations'}, ax=ax)
    ax.set_title('Calendar Heatmap: Data Coverage by Month')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    df.drop(['year','month'], axis=1, inplace=True)
    return img_base64

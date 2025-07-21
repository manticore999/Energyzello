import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
import seaborn as sns

def summary_stats(df):
    stats = df.describe().to_dict()
    skew = df['consumption'].skew()
    kurt = df['consumption'].kurt()
    return {'describe': stats, 'skew': skew, 'kurt': kurt}

def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(14,4))
    ax.plot(df.index, df['consumption'])
    ax.set_title('Full Time Series')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_stl(df):
    stl = STL(df['consumption'], period=12)
    res = stl.fit()
    fig = res.plot()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_acf_pacf(df):
    fig1 = plt.figure(figsize=(8,3))
    plot_acf(df['consumption'].dropna(), lags=24, ax=fig1.gca())
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig1)
    buf1.seek(0)
    acf_img = base64.b64encode(buf1.read()).decode('utf-8')
    fig2 = plt.figure(figsize=(8,3))
    plot_pacf(df['consumption'].dropna(), lags=24, ax=fig2.gca())
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    buf2.seek(0)
    pacf_img = base64.b64encode(buf2.read()).decode('utf-8')
    return {'acf': acf_img, 'pacf': pacf_img}

def plot_hist_box(df):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    df['consumption'].hist(bins=30, ax=ax[0])
    ax[0].set_title('Histogram of Consumption')
    df['consumption'].plot.box(ax=ax[1])
    ax[1].set_title('Boxplot of Consumption')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_missing_heatmap(df):
    fig, ax = plt.subplots(figsize=(14,2))
    sns.heatmap(df['consumption'].isnull().to_frame().T, cbar=False, ax=ax)
    ax.set_title('Missing Value Heatmap')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def stationarity_tests(df):
    adf_result = adfuller(df['consumption'].dropna())
    adf_stat = adf_result[0]
    adf_p = adf_result[1]
    kpss_result = kpss(df['consumption'].dropna(), regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_p = kpss_result[1]
    return {'adf_stat': adf_stat, 'adf_p': adf_p, 'kpss_stat': kpss_stat, 'kpss_p': kpss_p}

def ljung_box_test(df):
    lb_stat, lb_p = acorr_ljungbox(df['consumption'].dropna(), lags=[12], return_df=False)
    return {'lb_stat': float(lb_stat[0]), 'lb_p': float(lb_p[0])}

def feature_engineering(df):
    # Ensure datetime is the index and sorted
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').set_index('datetime')
    # Fill missing values (forward fill)
    df['consumption'] = df['consumption'].fillna(method='ffill')
    # Time-based features
    if 'year' not in df.columns:
        df['year'] = df.index.year
    if 'month' not in df.columns:
        df['month'] = df.index.month
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    # Lag features
    df['consumption_lag1'] = df['consumption'].shift(1)
    # Rolling features
    df['consumption_rolling3'] = df['consumption'].rolling(window=3).mean()
    df['consumption_rolling12'] = df['consumption'].rolling(window=12).mean()
    return df

def outlier_and_stationarity_checks(df):
    summary = {}
    # Duplicate timestamps
    summary['duplicate_timestamps'] = int(df.index.duplicated().sum())
    # Constant columns
    summary['constant_columns'] = [col for col in df.columns if df[col].nunique() == 1]
    # Monotonicity
    summary['monotonic_increasing'] = df.index.is_monotonic_increasing
    summary['monotonic_decreasing'] = df.index.is_monotonic_decreasing
    # Outlier detection (IQR)
    Q1 = df['consumption'].quantile(0.25)
    Q3 = df['consumption'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df['consumption'] < (Q1 - 1.5 * IQR)) | (df['consumption'] > (Q3 + 1.5 * IQR))
    summary['outlier_count'] = int(outlier_mask.sum())
    # ADF test
    adf_result = adfuller(df['consumption'].dropna())
    summary['adf_statistic'] = adf_result[0]
    summary['adf_pvalue'] = adf_result[1]
    # Outlier plot
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df['consumption'], label='consumption')
    ax.scatter(df.index[outlier_mask], df['consumption'][outlier_mask], color='red', label='outliers')
    ax.legend()
    ax.set_title('Consumption with Outliers Highlighted')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    outlier_img = base64.b64encode(buf.read()).decode('utf-8')
    return summary, outlier_img

def model_recommendation(df):
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss
    recommendation = []
    stl = STL(df['consumption'], period=12)
    res = stl.fit()
    seasonality_strength = res.seasonal.std() / df['consumption'].std()
    adf_pvalue = adfuller(df['consumption'].dropna())[1]
    kpss_result = kpss(df['consumption'].dropna(), regression='c', nlags='auto')
    kpss_stat = kpss_result[0]
    kpss_p = kpss_result[1]
    if seasonality_strength > 0.3:
        recommendation.append('Prophet or SARIMA (strong seasonality detected)')
    if adf_pvalue > 0.05 or kpss_p < 0.05:
        recommendation.append('Apply differencing or use models that handle non-stationarity (Prophet, SARIMA)')
    if len(df) < 50:
        recommendation.append('Dataset is short: use ARIMA or Exponential Smoothing')
    elif len(df) > 200:
        recommendation.append('Dataset is long: LSTM/Deep Learning models are possible')
    return recommendation

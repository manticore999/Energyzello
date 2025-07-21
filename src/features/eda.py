import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def generate_summary(df):
    # Only summarize the target column (assumed standardized as 'consumption')
    summary_stats = df['consumption'].describe().to_dict()
    # Ensure all keys are strings for JSON serialization
    summary_stats = {str(k): v for k, v in summary_stats.items()}
    missing = {'consumption': int(df['consumption'].isnull().sum())}
    return {'summary': summary_stats, 'missing': missing}

def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    # Ensure datetime is a column for plotting
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df.plot(x='datetime', y='consumption', ax=ax)
    ax.set_title('Consumption Over Time')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_seasonality(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    if 'month' in df.columns:
        sns.boxplot(x='month', y='consumption', data=df, ax=ax)
        ax.set_title('Monthly Seasonality')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4))
    df['consumption'].hist(bins=30, ax=ax)
    ax.set_title('Histogram of Consumption')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 2))
    df['consumption'].plot.box(ax=ax)
    ax.set_title('Boxplot of Consumption')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_calendar_heatmap(df):
    df = df.copy()
    # Ensure 'datetime' is a column for grouping
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    daily = df.groupby('date')['consumption'].mean().reset_index()
    daily['date'] = pd.to_datetime(daily['date'])
    pivot = daily.pivot_table(index=daily['date'].dt.month, columns=daily['date'].dt.day, values='consumption')
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Avg Consumption'})
    ax.set_title('Calendar Heatmap of Daily Average Consumption')
    ax.set_xlabel('Day of Month')
    ax.set_ylabel('Month')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_acf_pacf(df, lags=40):
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # Ensure 'datetime' is a column for plotting if needed
    if 'datetime' not in df.columns:
        df = df.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df['consumption'].dropna(), lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation (ACF)')
    plot_pacf(df['consumption'].dropna(), lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title('Partial Autocorrelation (PACF)')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def adf_test(df):
    result = adfuller(df['consumption'].dropna())
    output = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'used_lag': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }
    return output

def recommend_model(df):
    # Simple logic: if strong seasonality, recommend SARIMA; else, ARIMA
    if 'month' in df.columns:
        return 'SARIMA'
    return 'ARIMA'

def run_eda(df):
    summary = generate_summary(df)
    ts_plot = plot_time_series(df)
    season_plot = plot_seasonality(df)
    model = recommend_model(df)
    return {
        'summary': summary,
        'time_series_plot': ts_plot,
        'seasonality_plot': season_plot,
        'recommended_model': model
    }

def basic_eda(df):
    summary = generate_summary(df)
    ts_plot = plot_time_series(df)
    hist_plot = plot_histogram(df)
    box_plot = plot_boxplot(df)
    return {
        'summary': summary,
        'time_series_plot': ts_plot,
        'histogram_plot': hist_plot,
        'boxplot': box_plot
    }

def advanced_eda(df):
    calendar_plot = plot_calendar_heatmap(df)
    acf_pacf_plot = plot_acf_pacf(df)
    adf_result = adf_test(df)
    return {
        'calendar_heatmap': calendar_plot,
        'acf_pacf_plot': acf_pacf_plot,
        'adf_test': adf_result
    }

def visual_eda(df):
    calendar_plot = plot_calendar_heatmap(df)
    acf_pacf_plot = plot_acf_pacf(df)
    return {
        'calendar_heatmap': calendar_plot,
        'acf_pacf_plot': acf_pacf_plot
    }

def plot_feature_distribution(df, feature, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[feature].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def date_feature_distributions(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.weekday
    # Optionally: df['hour'] = df['datetime'].dt.hour if data is hourly
    plots = {
        'year': plot_feature_distribution(df, 'year', 'Distribution by Year', 'Year'),
        'month': plot_feature_distribution(df, 'month', 'Distribution by Month', 'Month'),
        'day': plot_feature_distribution(df, 'day', 'Distribution by Day of Month', 'Day'),
        'weekday': plot_feature_distribution(df, 'weekday', 'Distribution by Weekday (0=Mon)', 'Weekday'),
    }
    return plots

def detect_outliers_iqr(df):
    # IQR method for outlier detection
    q1 = df['consumption'].quantile(0.25)
    q3 = df['consumption'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = (df['consumption'] < lower) | (df['consumption'] > upper)
    return outliers

def plot_outliers(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    outliers = detect_outliers_iqr(df)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['datetime'], df['consumption'], label='Consumption', color='blue')
    ax.scatter(df['datetime'][outliers], df['consumption'][outliers], color='red', label='Outliers', zorder=5)
    ax.set_title('Time Series with Outliers Highlighted')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Consumption')
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def add_advanced_features(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df = df.copy()
    df['rolling_mean_7'] = df['consumption'].rolling(window=7, min_periods=1).mean()
    df['lag_1'] = df['consumption'].shift(1)
    return df

def plot_rolling_mean(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['datetime'], df['consumption'], label='Original', alpha=0.5)
    ax.plot(df['datetime'], df['rolling_mean_7'], label='7-day Rolling Mean', color='orange')
    ax.set_title('Original Series and 7-day Rolling Mean')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Consumption')
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_lag_feature(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['lag_1'], df['consumption'], alpha=0.5)
    ax.set_title('Lag-1 vs. Consumption')
    ax.set_xlabel('Lag-1 Consumption')
    ax.set_ylabel('Current Consumption')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def outliers_and_features_eda(df):
    df_feat = add_advanced_features(df)
    outlier_plot = plot_outliers(df_feat)
    rolling_plot = plot_rolling_mean(df_feat)
    lag_plot = plot_lag_feature(df_feat)
    return {
        'outlier_plot': outlier_plot,
        'rolling_mean_plot': rolling_plot,
        'lag_feature_plot': lag_plot
    }

def plot_missing_data(df):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    # Plot missing values over time
    missing = df['consumption'].isnull()
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(df['datetime'], missing, marker='|', linestyle='None', color='red', label='Missing')
    ax.set_title('Missing Data Over Time')
    ax.set_xlabel('Datetime')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Present', 'Missing'])
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def missing_data_analysis(df):
    n_missing = df['consumption'].isnull().sum()
    if n_missing == 0:
        return {'has_missing': False}
    missing_plot = plot_missing_data(df)
    return {
        'has_missing': True,
        'n_missing': int(n_missing),
        'missing_plot': missing_plot
    }

def plot_decomposition(df, model='additive', freq=None):
    # Ensure datetime is a column
    if 'datetime' not in df.columns:
        df = df.reset_index()
    df = df.set_index('datetime')
    # Infer frequency if not provided
    if not freq:
        freq = pd.infer_freq(df.index)
    result = seasonal_decompose(df['consumption'], model=model, period=7 if freq is None else None)
    plots = {}
    for comp in ['trend', 'seasonal', 'resid']:
        fig, ax = plt.subplots(figsize=(12, 3))
        getattr(result, comp).plot(ax=ax)
        ax.set_title(f'{comp.capitalize()} Component')
        ax.set_xlabel('Datetime')
        ax.set_ylabel(comp.capitalize())
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plots[comp] = img_base64
    return plots

def decomposition_eda(df):
    try:
        plots = plot_decomposition(df)
        return {'success': True, 'plots': plots}
    except Exception as e:
        return {'success': False, 'error': str(e)}

import streamlit as st
import pandas as pd
import requests
import time
from io import BytesIO
from eda_section import show_eda_section
from advanced_viz_section import show_advanced_viz_section
from features_section import show_features_section
from missing_section import show_missing_section
from decomposition_section import show_decomposition_section
from model_history_section import show_model_history_section
from forecast_section import show_forecast_section

st.set_page_config(layout="wide")

# Then manually control the width
st.markdown(
    """
    <style>
    /* Constrain the main content width */
    .main > div {
        max-width: 1100px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("EnergyZello : a no-code energy forecasting tool")

for key, default in [
    ('uploaded_file', None),
    ('eda', None),
    ('vis', None),
    ('date_feats', None),
    ('out_feats', None),
    ('missing', None),
    ('decomp', None),
    ('show_tests', False),
    ('tests', None),
    ('model_history', []),
    ('data_analyzed', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- File Upload & Dataset Preview ---
uploaded_file = st.file_uploader("Upload your time series CSV")
if uploaded_file:
    st.session_state['uploaded_file'] = uploaded_file.getvalue()
    df = pd.read_csv(uploaded_file)
    st.markdown('---')
    st.subheader('Select Columns')
    st.markdown("""
### Dataset Preview

Below is a preview of the first few rows of your uploaded dataset. This helps you verify that your data was loaded correctly and that the datetime and target columns are properly recognized.

- **Datetime column:** Should contain time information (e.g., timestamps, dates). This will be used to order and analyze your time series.
- **Target column:** Should contain the values you want to forecast (e.g., energy consumption, sales, temperature).

Take a moment to check for missing values, unexpected formats, or outliers. If something looks off, you can re-upload your data or select different columns.
""")
    st.write(df.head())
    datetime_col = st.selectbox('Select the datetime column:', df.columns, key='datetime_col')
    target_col = st.selectbox('Select the target column:', [col for col in df.columns if col != datetime_col], key='target_col')
    if st.button("Analyze Data"):
        if not datetime_col or not target_col:
            st.warning("Please select both a datetime column and a target column.")
        else:
            files = {"file": st.session_state['uploaded_file']}
            upload_response = requests.post(f"http://localhost:8000/upload/?datetime_col={datetime_col}&target_col={target_col}", files=files)
            if upload_response.status_code == 200:
                st.success("File uploaded and processed successfully.")
                eda_response = requests.post("http://localhost:8000/eda/")
                if eda_response.status_code == 200:
                    st.session_state['eda'] = eda_response.json()
                vis_response = requests.post("http://localhost:8000/visual_eda/")
                if vis_response.status_code == 200:
                    st.session_state['vis'] = vis_response.json()
                date_feat_response = requests.post("http://localhost:8000/date_features/")
                if date_feat_response.status_code == 200:
                    st.session_state['date_feats'] = date_feat_response.json()
                out_feat_response = requests.post("http://localhost:8000/outliers_features/")
                if out_feat_response.status_code == 200:
                    st.session_state['out_feats'] = out_feat_response.json()
                missing_response = requests.post("http://localhost:8000/missing_data/")
                if missing_response.status_code == 200:
                    st.session_state['missing'] = missing_response.json()
                decomp_response = requests.post("http://localhost:8000/decomposition/")
                if decomp_response.status_code == 200:
                    st.session_state['decomp'] = decomp_response
                st.session_state['data_analyzed'] = True
            else:
                st.error(f"Upload failed: {upload_response.text}")

# --- EDA & Visualizations ---
if st.session_state['data_analyzed']:
    if st.session_state['eda']:
        st.markdown("### Exploratory Data Analysis (EDA)")
        st.markdown("""
**What is EDA?**

Exploratory Data Analysis (EDA) is like looking at a map before a road trip. It helps you spot patterns, outliers, and surprises in your data before you start modeling.
""")
        
        show_eda_section(st.session_state['eda'])
        st.markdown("---")
    if st.session_state['vis']:
        st.markdown("### Advanced Visualizations")
        st.markdown("""
**Autocorrelation Plot:**
Shows how much your data repeats at different time lags. Peaks at regular intervals mean strong seasonality (like weekly or yearly cycles).

**Heatmap:**
Visualizes patterns across time (e.g., by hour, day, or month). Helps you spot when values are unusually high or low.
""")
        show_advanced_viz_section(st.session_state['vis'])
        st.markdown("---")
    if st.session_state['date_feats'] and st.session_state['out_feats']:
        st.markdown("### Feature Analysis")
        st.markdown("""
**What are features?**

Features are extra pieces of information we create from your data to help models make better predictions. For example, knowing the day of the week or the value from last month can help the model spot patterns.

**Date Features:**
Show how your data changes by day, week, month, or season. Useful for capturing regular cycles.

**Lag Features:**
Show what happened in the past (e.g., value 7 days ago). Models use these to learn from history.

**Outlier Features:**
Highlight unusual values that might need special attention or cleaning.
""")
        show_features_section(st.session_state['date_feats'], st.session_state['out_feats'])
        st.markdown("---")
    if st.session_state['missing']:
        st.markdown("### Missing Data Overview")
        st.markdown("""
**Why care about missing data?**

Missing values can confuse models and lead to bad predictions. This section shows where your data is missing, so you can decide how to handle it (fill, drop, or flag).
""")
        show_missing_section(st.session_state['missing'])
        st.markdown("---")
    if st.session_state['decomp']:
        st.markdown("### Time Series Decomposition")
        st.markdown("""
**What is decomposition?**

Decomposition breaks your data into three parts:
- **Trend:** The long-term direction (up, down, or flat).
- **Seasonality:** Regular repeating patterns (like weekends or holidays).
- **Residuals:** What’s left after removing trend and seasonality (random noise).

This helps you see if your data has strong trends or cycles, and if your model should account for them.
""")
        show_decomposition_section(st.session_state['decomp'])
        st.markdown("---")

# --- Statistical Tests & Model Recommendation ---
if st.session_state['uploaded_file'] is not None and st.session_state['eda'] is not None:
    st.markdown("---")
    st.markdown("## Next Step: Automated Data Understanding & Model Recommendation")
    run_tests = st.button("Run Statistical Tests & Recommend Model")
    if run_tests:
        st.session_state['show_tests'] = True
        progress = st.progress(0, text="Running statistical tests and analyzing your data...")
        for percent in range(1, 101, 10):
            time.sleep(0.05)
            progress.progress(percent, text=f"Running statistical tests... {percent}%")
        tests_response = requests.post("http://localhost:8000/stat_tests/")
        progress.empty()
        if tests_response.status_code == 200:
            st.session_state['tests'] = tests_response.json()
        else:
            st.session_state['tests'] = None

if st.session_state.get('show_tests') and st.session_state.get('tests'):
    tests = st.session_state['tests']
    st.markdown("## Statistical Tests & Model Recommendation")
    st.markdown("### Stationarity Test (ADF)")
    st.markdown(f"- **ADF Statistic:** {tests['adf']['stat']:.3f}")
    st.markdown(f"- **p-value:** {tests['adf']['pvalue']:.3g}")
    st.markdown("""
**What is stationarity?**

Imagine you’re tracking your daily steps. If your average steps and how much they vary stay about the same over time, your data is *stationary*. If you start walking more every month, or your routine changes with the seasons, your data is *not* stationary.

**Why does this matter?**

Many forecasting models work best when the data’s “behavior” doesn’t change over time. If your data is not stationary, predictions can become unreliable.

**How do we check? (ADF Test)**

The Augmented Dickey-Fuller (ADF) test checks if your data is stationary.  
- If the **p-value** is below 0.05, your data is likely stationary (good for most models).
- If it’s above 0.05, your data may have trends or changing patterns.

**What should I do if my data is not stationary?**

- Try removing trends (subtracting the previous value, called “differencing”).
- Remove seasonality (subtracting the value from last year/month).
- Sometimes, a simple transformation (like log or square root) helps.
""")
    st.markdown("### Seasonality Test (Autocorrelation)")
    st.markdown(f"- **Max autocorrelation (lag 1-24):** {tests['seasonality']['max_acf']:.3f}")
    st.markdown("""
**What is seasonality?**

Think of electricity use: it’s higher in the evening, or in summer. Seasonality means your data has regular, repeating patterns—like weekends, holidays, or weather cycles.

**Why does this matter?**

If your data has strong seasonality, you need models that can “see” and use these patterns. Otherwise, forecasts will miss important ups and downs.

**How do we check? (Autocorrelation)**

Autocorrelation measures how much your data “repeats” at regular intervals (lags).  
- High autocorrelation at certain lags means strong seasonality (e.g., every 7 days for weekly cycles).

**What should I do if my data is seasonal?**

- Use models that handle seasonality (SARIMA, XGBoost, LightGBM, CatBoost).
- Add features for time of year, day of week, holidays, etc.
""")
    st.markdown("### Model Recommendation")
    st.markdown("""
**How do we pick a model?**

We look at your data’s patterns:
- If it changes a lot over time (not stationary), or has strong cycles (seasonality), we recommend models that can handle those.
- If it’s stable and regular, simpler models may work well.

**Why does this matter?**

Choosing the right model saves time and gives better forecasts.  
- For beginners: Start with the recommended model.
- For experts: Try several and compare results.

**Tip:**  
No model is perfect! Try a few, compare their results, and see which works best for your data and goals.
""")
    st.markdown("---")
    st.markdown("## Model Selection & Training")
    adf_p = tests['adf']['pvalue']
    max_acf = tests['seasonality']['max_acf']
    if adf_p > 0.05 and max_acf > 0.5:
        recommended_models = ["SARIMA", "LightGBM", "CatBoost", "XGBoost"]
        rec_text = "Your data is non-stationary and shows strong seasonality. SARIMA, LightGBM, CatBoost, or XGBoost are well-suited for such series."
    elif adf_p > 0.05:
        recommended_models = ["ARIMA", "LightGBM", "CatBoost", "RandomForest"]
        rec_text = "Your data is non-stationary but does not show strong seasonality. ARIMA, LightGBM, CatBoost, or RandomForest are good starting points."
    elif max_acf > 0.5:
        recommended_models = ["SARIMA", "LightGBM", "CatBoost", "XGBoost"]
        rec_text = "Your data is stationary but shows strong seasonality. SARIMA, LightGBM, CatBoost, or XGBoost are recommended."
    else:
        recommended_models = ["ARIMA", "LightGBM", "CatBoost", "RandomForest"]
        rec_text = "Your data is stationary and does not show strong seasonality. ARIMA, LightGBM, CatBoost, or RandomForest are good choices."
    st.markdown(f"**Recommended Models:** :star: {', '.join(recommended_models)} :star:")
    ml_models = ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    stat_models = ["ARIMA", "SARIMA"]
    rec_ml = ', '.join([m for m in recommended_models if m in ml_models])
    rec_stat = ', '.join([m for m in recommended_models if m in stat_models])
    st.markdown(f"<span style='font-size:1.2em; font-weight:bold; color:#2b7bba;'>Recommended ML Models:</span> <span style='color:#228B22;'>{rec_ml}</span>", unsafe_allow_html=True)
    if rec_stat:
        st.markdown(f"<span style='font-size:1.1em; font-weight:bold; color:#b8860b;'>You may also try statistical models:</span> <span style='color:#8B0000;'>{rec_stat}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size:1em; color:#444;'>{rec_text}</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='font-size:0.95em; color:#555;'>{tests['explanation']}</span>", unsafe_allow_html=True)
    st.info("Note: While this app focuses on machine learning models, you can also experiment with statistical models (ARIMA, SARIMA) in your own environment for comparison.")

# --- Model Training Section ---
if st.session_state['tests']:
    st.markdown("---")
    st.markdown("## Model Selection & Training")
    recommended = st.session_state['tests']['recommendation']
    adf_p = st.session_state['tests']['adf']['pvalue']
    max_acf = st.session_state['tests']['seasonality']['max_acf']
    if adf_p > 0.05 and max_acf > 0.5:
        recommended_models = ["SARIMA", "LightGBM", "CatBoost", "XGBoost"]
        rec_text = "Your data is non-stationary and shows strong seasonality. SARIMA, LightGBM, CatBoost, or XGBoost are well-suited for such series."
    elif adf_p > 0.05:
        recommended_models = ["ARIMA", "LightGBM", "CatBoost", "RandomForest"]
        rec_text = "Your data is non-stationary but does not show strong seasonality. ARIMA, LightGBM, CatBoost, or RandomForest are good starting points."
    elif max_acf > 0.5:
        recommended_models = ["SARIMA", "LightGBM", "CatBoost", "XGBoost"]
        rec_text = "Your data is stationary but shows strong seasonality. SARIMA, LightGBM, CatBoost, or XGBoost are recommended."
    else:
        recommended_models = ["ARIMA", "LightGBM", "CatBoost", "RandomForest"]
        rec_text = "Your data is stationary and does not show strong seasonality. ARIMA, LightGBM, CatBoost, or RandomForest are good choices."
    st.markdown(f"**Recommended Models:** :star: {', '.join(recommended_models)} :star:")
    st.markdown(rec_text)
    st.markdown(st.session_state['tests']['explanation'])
    st.info("Prophet is not available in this app due to compatibility issues. If Prophet is recommended for your data, you can try it in your own environment. [Prophet official docs](https://facebook.github.io/prophet/)")
    # Remove Prophet from selectable models
    all_models = [recommended] + ["ARIMA", "SARIMA", "Naive", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
    seen = set()
    model_options = [x for x in all_models if not (x in seen or seen.add(x))]
    model_choice = st.selectbox(
        "Select a model to train:",
        model_options,
        help="The recommended model is listed first, but you can try others. Prophet is not available in this app due to compatibility issues."
    )
    st.markdown(f"**About {model_choice}:**")
    if model_choice == "ARIMA":
        st.markdown("""
**ARIMA**  
Think of ARIMA as a “trend detector.” It’s good for data that’s stable over time, without big seasonal swings.  
- *Example:* Predicting sales for a product that doesn’t have big holiday spikes.
- [ARIMA explained (Analytics Vidhya)](https://www.analyticsvidhya.com/blog/2018/08/auto-arima-time-series-forecasting-python/)
""")
        st.warning("ARIMA may perform poorly on real-world data with trend, seasonality, or outliers. It is best suited for personal configuration and advanced users who can tune and preprocess their data.")
    elif model_choice == "SARIMA":
        st.markdown("""
**SARIMA**  
SARIMA is like ARIMA’s cousin who loves patterns—great for data with regular cycles (like seasons or weekends).  
- *Example:* Forecasting ice cream sales, which spike every summer.
- [SARIMA explained (Towards Data Science)](https://towardsdatascience.com/sarima-model-for-time-series-forecasting-in-python-3b0d3c4e5f5c)
""")
        st.warning("SARIMA may perform poorly on complex or non-stationary data. It is best suited for personal configuration and advanced users who can tune and preprocess their data.")
    elif model_choice == "Naive":
        st.markdown("""
**Naive Model**  
The “copy-paste” model: it just predicts the next value will be the same as the last one.  
- *Example:* If you sold 100 units yesterday, it predicts 100 for today.
- *Use:* Good as a simple baseline to compare other models.
- [Naive forecasting (Wikipedia)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Naive_forecasting)
""")
    elif model_choice == "RandomForest":
        st.markdown("""
**Random Forest**  
Imagine asking a group of friends to guess tomorrow’s weather, then averaging their answers. Random Forest combines many “mini-models” to make a smart prediction.  
- *Use:* Great for data with lots of features (like date, weather, holidays).
- [Random Forest for time series (KDnuggets)](https://www.kdnuggets.com/2020/01/random-forest-time-series-forecasting-python.html)
""")
    elif model_choice == "XGBoost":
        st.markdown("""
**XGBoost**  
XGBoost is a “supercharged” tree model—fast, powerful, and often a top performer in competitions.  
- *Use:* Excellent for complex data with many features and patterns.
- *Tip:* Needs more data and careful tuning, but can deliver great results.
- [XGBoost for time series (Medium)](https://medium.com/swlh/using-xgboost-for-time-series-forecasting-9b78c6cd88e0)
""")
    elif model_choice == "LightGBM":
        st.markdown("""
**LightGBM**  
LightGBM is another “supercharged” tree model—fast, efficient, and great for structured/tabular time series data.  
- *Use:* Excellent for complex data with many features and patterns.
- *Tip:* Needs more data and careful tuning, but can deliver great results.
- [LightGBM for time series (Kaggle)](https://www.kaggle.com/code/robikscube/time-series-forecasting-with-lightgbm)
""")
    elif model_choice == "CatBoost":
        st.markdown("""
**CatBoost**  
CatBoost is a “supercharged” tree model that’s especially good with categorical features (like days of the week, product types).  
- *Use:* Great for time series with lots of categories and engineered features.
- [CatBoost for time series (Medium)](https://medium.com/@aravanshad/forecasting-time-series-data-with-catboost-7b49b36e3c56)
""")
    st.markdown("---")
    import base64
    if st.button(f"Train {model_choice} Model"):
        with st.spinner(f"Training {model_choice} model and tracking with MLflow..."):
            train_response = requests.post(
                "http://localhost:8000/train/", json={"model": model_choice}
            )
            if train_response.status_code == 200:
                result = train_response.json()
                st.success(f"{model_choice} model trained!")
                st.markdown(f"### Forecast Results for {model_choice}")
                st.image(base64.b64decode(result['forecast_plot']), use_container_width=True)
                st.markdown(f"**MAE:** {result['mae']:.3f} | **RMSE:** {result['rmse']:.3f}")
                st.markdown(":bar_chart: The plot shows actual vs. predicted values. Use the metrics to compare models.")
                if result.get('warning'):
                    st.warning(result['warning'])
                if result.get('run_id'):
                    st.markdown(f"**MLflow Run ID:** `{result['run_id']}`")
                # Fetch updated model history after training
                history_resp = requests.get("http://localhost:8000/model_history/")
                if history_resp.status_code == 200:
                    st.session_state['model_history'] = history_resp.json()
            else:
                try:
                    error_json = train_response.json()
                    st.error(f"Training failed: {error_json.get('error', train_response.text)}")
                except Exception:
                    st.error(f"Training failed: {train_response.text}")

# --- Model History Section ---
import numpy as np
def fetch_model_history():
    history_resp = requests.get("http://localhost:8000/model_history/")
    if history_resp.status_code == 200:
        st.session_state['model_history'] = history_resp.json()

# Always fetch model history on app start if not present
if st.session_state['data_analyzed'] and not st.session_state['model_history']:
    fetch_model_history()

if st.session_state['model_history']:
    st.markdown("---")
    st.markdown("### Model Comparison Table")
    df_history = pd.DataFrame(st.session_state['model_history'])
    if not df_history.empty:
        df_history['start_time'] = pd.to_datetime(df_history['start_time'], unit='ms')
        df_history = df_history.sort_values("rmse")
        cols = st.columns([2, 2, 2, 3, 1, 1])
        headers = ["Model", "MAE", "RMSE", "Trained", "Delete", "Download"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        for idx, row in df_history.iterrows():
            cols = st.columns([2, 2, 2, 3, 1, 1])
            cols[0].write(row['model'])
            cols[1].write(np.round(row['mae'], 3) if not pd.isnull(row['mae']) else "-")
            cols[2].write(np.round(row['rmse'], 3) if not pd.isnull(row['rmse']) else "-")
            cols[3].write(row['start_time'])
            if cols[4].button("Delete", key=f"delete_{row['run_id']}"):
                del_resp = requests.delete(f"http://localhost:8000/delete_model/{row['run_id']}")
                if del_resp.status_code == 200:
                    st.success(f"Deleted model {row['model']}.")
                    fetch_model_history()
                else:
                    st.error(f"Failed to delete model: {del_resp.text}")
            dl_resp = requests.get(f"http://localhost:8000/download_model/{row['run_id']}")
            if dl_resp.status_code == 200:
                cols[5].download_button(
                    label="Download",
                    data=BytesIO(dl_resp.content),
                    file_name=f"{row['model']}_model.joblib",
                    mime='application/octet-stream',
                    key=f"dlbtn_{row['run_id']}"
                )
            else:
                cols[5].write("-")
        st.markdown("Best model is highlighted at the top (lowest RMSE).")
        st.markdown("---")
        st.header("Forecasting")
        show_forecast_section(st.session_state['model_history'])

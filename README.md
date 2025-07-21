# EnergyZello: No-Code Time Series Forecasting Platform

## Objective
EnergyZello is a user-friendly, no-code platform for time series forecasting, designed for both beginners and experts. It helps users analyze, understand, and forecast time series data (like energy consumption) without writing code, while also supporting advanced experiment tracking and model management for power users.

---

## Key Features
- **No-Code UI:** Upload data, analyze, train, and forecast directly in your browser with Streamlit.
- **Educational Guidance:** Step-by-step explanations, visualizations, and tooltips for every concept and plot.
- **Automated EDA:** Visual and statistical analysis of your time series, including trends, seasonality, and outliers.
- **Model Recommendation:** Smart suggestions based on your data’s properties (stationarity, seasonality, etc.).
- **Multiple Model Support:** ARIMA, SARIMA, Naive, RandomForest, XGBoost, LightGBM, CatBoost.
- **Experiment Tracking:** MLflow integration for model history, metrics, and artifact management.
- **Model Registry:** View, compare, delete, and download trained models.
- **Forecasting:** Generate and compare forecasts from multiple models.

---

## Architecture

```
[User] <-> [Streamlit UI] <-> [FastAPI Backend] <-> [ML/Stats Libraries, MLflow]
```

- **Frontend:** Streamlit (Python)
- **Backend:** FastAPI (Python)
- **ML/Stats:** scikit-learn, statsmodels, xgboost, lightgbm, catboost, prophet
- **Tracking:** MLflow
- **Visualization:** matplotlib, seaborn, plotly

### Main Components
- `src/ui/app.py` — Main Streamlit app (UI logic, state, workflow)
- `src/api/main.py` — FastAPI backend (data processing, model training, forecasting)
- `src/features/` — EDA, feature engineering, preprocessing
- `src/models/` — Model training utilities
- `src/ui/` — Modular UI sections (EDA, features, missing, decomposition, history, forecast)

---

## Concepts Explained

### 1. **Time Series**
A sequence of data points measured over time (e.g., hourly energy use, daily sales).

### 2. **EDA (Exploratory Data Analysis)**
Visual and statistical summaries to help you understand your data before modeling. Includes:
- Line plots (trends)
- Histograms/boxplots (distribution, outliers)
- Calendar heatmaps (seasonality)
- Autocorrelation plots (repeating patterns)

### 3. **Stationarity**
A stationary series has constant mean/variance over time. Many models require this. The app explains and tests for it (ADF test).

### 4. **Seasonality**
Repeating patterns (e.g., higher use on weekends). Detected via autocorrelation and visualizations.

### 5. **Feature Engineering**
Creating new variables (lags, date parts) to help models learn from history and patterns.

### 6. **Model Recommendation**
The app suggests models based on your data’s properties:
- ARIMA/SARIMA for stationary/seasonal data
- Tree-based models for complex, non-linear patterns

### 7. **Model Training & History**
Train models, view metrics (MAE, RMSE), and compare runs. All tracked in MLflow.

### 8. **Forecasting**
Generate future predictions using trained models. Visualize and compare results.

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd zello
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the backend (FastAPI):**
   ```bash
   uvicorn src.api.main:app --reload
   ```
4. **Start the frontend (Streamlit):**
   ```bash
   streamlit run src/ui/app.py
   ```
5. **Open your browser:**
   Go to http://localhost:8501

---

## Limitations & Known Issues
- **No automated data validation:** Garbage in, garbage out—check your data carefully.
- **No hyperparameter tuning UI:** Model settings are mostly default.
- **No advanced MLOps (CI/CD, serving, monitoring):** MLflow is used for tracking, but not for deployment.
- **Prophet support may require extra dependencies (see their docs).**
- **Large datasets may be slow in the browser.**

---

## Who is this for?
- **Beginners:** Learn time series forecasting, concepts, and best practices with hands-on, no-code tools.
- **Experts:** Rapidly experiment, track, and compare models locally with MLflow integration.

---

## Contributing
Pull requests and suggestions are welcome! Please open an issue for bugs or feature requests.

---

## License
MIT License

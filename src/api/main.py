from fastapi import FastAPI, UploadFile, File, Request, Query
import pandas as pd
from src.features.preprocessing import load_and_prepare_auto, add_time_features
from fastapi.responses import JSONResponse, FileResponse
from fastapi import status
import tempfile
import matplotlib
import matplotlib.pyplot as plt
from src.features.eda import basic_eda, visual_eda, date_feature_distributions, outliers_and_features_eda, missing_data_analysis, decomposition_eda
from statsmodels.tsa.stattools import adfuller
import numpy as np
import mlflow
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.tracking import MlflowClient
import hashlib
import os

matplotlib.use('Agg')

app = FastAPI()

@app.post("/upload/")
async def upload_data(file: UploadFile = File(...), datetime_col: str = Query(None), target_col: str = Query(None)):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
        if datetime_col and target_col:
            # Use user-specified columns
            from src.features.preprocessing import load_and_prepare
            df = load_and_prepare(tmp_path, datetime_col, target_col)
        else:
            df = load_and_prepare_auto(df)
        # Save with datetime as a column for reloading
        df_reset = df.reset_index()
        tmp_csv_path = "tmp_uploaded.csv"
        df_reset.to_csv(tmp_csv_path, index=False)
        # Compute dataset hash for experiment scoping
        dataset_hash = hashlib.md5(content).hexdigest()
        with open("current_dataset_hash.txt", "w") as f:
            f.write(dataset_hash)
        return {"rows": len(df), "columns": list(df.columns), "index_type": str(type(df.index)), "head": df.head(5).reset_index().to_dict()}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/eda/")
async def eda():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        eda_results = basic_eda(df)
        return eda_results
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/visual_eda/")
async def visual_eda_endpoint():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        vis_results = visual_eda(df)
        return vis_results
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/date_features/")
async def date_features_endpoint():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        plots = date_feature_distributions(df)
        return plots
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/outliers_features/")
async def outliers_features_endpoint():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        results = outliers_and_features_eda(df)
        return results
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/missing_data/")
async def missing_data_endpoint():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        results = missing_data_analysis(df)
        return results
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/decomposition/")
async def decomposition_endpoint():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        results = decomposition_eda(df)
        return results
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/stat_tests/")
async def stat_tests():
    try:
        tmp_path = "tmp_uploaded.csv"
        df = pd.read_csv(tmp_path)
        df = load_and_prepare_auto(df)
        # ADF test for stationarity
        adf_result = adfuller(df['consumption'])
        adf = {
            'stat': float(adf_result[0]),
            'pvalue': float(adf_result[1])
        }
        # Seasonality: max autocorrelation (lags 1-24)
        acfs = [df['consumption'].autocorr(lag) for lag in range(1, 25)]
        max_acf = float(np.nanmax(np.abs(acfs)))
        seasonality = {'max_acf': max_acf}
        # Model recommendation logic (no Prophet, no combined SARIMA/Prophet)
        if adf['pvalue'] > 0.05 and max_acf > 0.5:
            recommendation = "SARIMA"
            explanation = "Your data is non-stationary and shows strong seasonality. SARIMA is well-suited for such series."
        elif adf['pvalue'] > 0.05:
            recommendation = "ARIMA"
            explanation = "Your data is non-stationary but does not show strong seasonality. ARIMA is a good starting point."
        elif max_acf > 0.5:
            recommendation = "SARIMA"
            explanation = "Your data is stationary but shows strong seasonality. SARIMA is recommended."
        else:
            recommendation = "ARIMA"
            explanation = "Your data is stationary and does not show strong seasonality. ARIMA is a good choice."
        return {
            'adf': adf,
            'seasonality': seasonality,
            'recommendation': recommendation,
            'explanation': explanation
        }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(e)}
        )

@app.post("/train/")
async def train(request: Request):
    data = await request.json()
    model_name = data.get("model")
    tmp_path = "tmp_uploaded.csv"
    df = pd.read_csv(tmp_path)
    df = load_and_prepare_auto(df)
    # Simple train/test split (last 20% for test)
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    y_train, y_test = train_df['consumption'], test_df['consumption']
    forecast = None
    mae = rmse = None
    import os
    if os.path.exists("current_dataset_hash.txt"):
        with open("current_dataset_hash.txt") as f:
            dataset_hash = f.read().strip()
    else:
        dataset_hash = "time_series_forecasting"
    mlflow.set_experiment(dataset_hash)
    with mlflow.start_run() as run:
        try:
            if model_name == "ARIMA":
                import pmdarima as pm
                # Use a sample for parameter selection if data is large
                if len(y_train) > 5000:
                    y_train_sample = y_train.iloc[-5000:]
                else:
                    y_train_sample = y_train
                arima_model = pm.auto_arima(
                    y_train_sample,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=3, max_q=3, max_d=2,
                    n_jobs=-1
                )
                order = arima_model.order
                model = ARIMA(y_train, order=order).fit()
                pred = model.forecast(len(y_test))
                warning = (
                    "ARIMA may perform poorly if your data is non-stationary, has strong seasonality, or contains outliers. "
                    "For best results, ARIMA often needs careful preprocessing and tuning."
                )
            elif model_name == "SARIMA":
                import pmdarima as pm
                # Use a sample for parameter selection if data is large
                if len(y_train) > 5000:
                    y_train_sample = y_train.iloc[-5000:]
                else:
                    y_train_sample = y_train
                sarima_model = pm.auto_arima(
                    y_train_sample,
                    seasonal=True,
                    m=12,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    max_p=2, max_q=2, max_d=1,
                    max_P=1, max_Q=1, max_D=1,
                    n_jobs=-1
                )
                order = sarima_model.order
                seasonal_order = sarima_model.seasonal_order
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order).fit(disp=False)
                pred = model.forecast(len(y_test))
                warning = (
                    "SARIMA may perform poorly if your data is non-stationary, has complex seasonality, or contains outliers. "
                    "For best results, SARIMA often needs careful preprocessing and tuning."
                )
            elif model_name == "Prophet":
                try:
                    prophet_df = train_df.reset_index()[['datetime', 'consumption']].rename(columns={'datetime': 'ds', 'consumption': 'y'})
                    m = Prophet()
                    m.fit(prophet_df)
                    future = test_df.reset_index()[['datetime']].rename(columns={'datetime': 'ds'})
                    pred = m.predict(future)['yhat'].values
                    warning = None
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"Prophet failed: {str(e)}"})
            elif model_name == "Naive":
                try:
                    pred = [y_train.iloc[-1]] * len(y_test)
                    warning = None
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"Naive model failed: {str(e)}"})
            elif model_name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
                try:
                    train_feat = train_df.copy()
                    test_feat = test_df.copy()
                    for df_ in [train_feat, test_feat]:
                        df_['lag1'] = df_['consumption'].shift(1)
                        df_['month'] = df_.index.month
                        df_['weekday'] = df_.index.weekday
                    train_feat = train_feat.dropna()
                    X_train = train_feat[['lag1', 'month', 'weekday']]
                    y_train_ml = train_feat['consumption']
                    test_feat = test_feat.dropna()
                    X_test = test_feat[['lag1', 'month', 'weekday']]
                    y_test_ml = test_feat['consumption']
                    if model_name == "RandomForest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    elif model_name == "XGBoost":
                        model = XGBRegressor(n_estimators=100, random_state=42)
                    elif model_name == "LightGBM":
                        from lightgbm import LGBMRegressor
                        model = LGBMRegressor(n_estimators=100, random_state=42)
                    elif model_name == "CatBoost":
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
                    model.fit(X_train, y_train_ml)
                    pred = model.predict(X_test)
                    # Ensure lengths match for metrics
                    if len(y_test_ml) == len(pred):
                        mae = float(mean_absolute_error(y_test_ml, pred))
                        rmse = float(np.sqrt(mean_squared_error(y_test_ml, pred)))
                    else:
                        print(f"Length mismatch: y_test_ml={len(y_test_ml)}, pred={len(pred)}")
                        mae = rmse = float('nan')
                    y_test = y_test_ml
                except Exception as e:
                    return JSONResponse(status_code=400, content={"error": f"ML model ({model_name}) failed: {str(e)}"})
            else:
                return JSONResponse(status_code=400, content={"error": "Unknown model"})
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Training failed: {str(e)}"})
        # Plot actual vs predicted
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(y_test.index, y_test, label='Actual')
        ax.plot(y_test.index, pred, label='Forecast')
        ax.set_title(f"Actual vs Forecast - {model_name}")
        ax.legend()
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        forecast_plot = base64.b64encode(buf.read()).decode('utf-8')
        import math
        # Validate metrics before logging
        if mae is None or rmse is None or math.isnan(mae) or math.isnan(rmse):
            return JSONResponse(status_code=400, content={"error": f"{model_name} produced invalid metrics (mae={mae}, rmse={rmse}). Check your data and model configuration."})
        mlflow.log_param("model", model_name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_artifact(tmp_path, artifact_path="data")
        mlflow.log_figure(fig, "forecast_plot.png")
        # --- Save and log model artifact ---
        import joblib, tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, f"{model_name}_model.joblib")
            joblib.dump(model, model_file)
            mlflow.log_artifact(model_file, artifact_path="model")
        # --- End model artifact logging ---
        run_id = run.info.run_id
    return {
        "mae": mae,
        "rmse": rmse,
        "forecast_plot": forecast_plot,
        "warning": warning if 'warning' in locals() else None,
        "run_id": run_id
    }

@app.get("/model_history/")
async def model_history():
    import os
    if os.path.exists("current_dataset_hash.txt"):
        with open("current_dataset_hash.txt") as f:
            dataset_hash = f.read().strip()
    else:
        dataset_hash = "time_series_forecasting"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(dataset_hash)
    if experiment is None:
        return []
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    history = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        history.append({
            "model": params.get("model"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "start_time": run.info.start_time,
            "run_id": run.info.run_id
        })
    return history

@app.post("/forecast/")
async def forecast(request: Request):
    data = await request.json()
    models = data.get("models", [])
    horizon = int(data.get("horizon", 24))
    period_type = data.get("period_type", "days").lower()
    freq = data.get("freq", None)
    import os
    import logging
    if os.path.exists("current_dataset_hash.txt"):
        with open("current_dataset_hash.txt") as f:
            dataset_hash = f.read().strip()
    else:
        dataset_hash = "time_series_forecasting"
    client = MlflowClient()
    experiment = client.get_experiment_by_name(dataset_hash)
    if experiment is None:
        return {"error": "No MLflow experiment found for this dataset. No models have been trained."}
    runs = client.search_runs([experiment.experiment_id], order_by=["attributes.start_time DESC"])
    model_to_run = {}
    for run in runs:
        m = run.data.params.get("model")
        if m and m not in model_to_run:
            model_to_run[m] = run.info.run_id
    forecasts = {}
    errors = {}
    tmp_path = "tmp_uploaded.csv"
    if not os.path.exists(tmp_path):
        return {"error": "No uploaded dataset found. Please upload a dataset and train models first."}
    df = pd.read_csv(tmp_path)
    df = load_and_prepare_auto(df)
    last_datetime = df.index[-1]
    # Use provided freq or infer
    if freq:
        freq_used = freq
    else:
        freq_used = pd.infer_freq(df.index) or "D"
    future_dates = pd.date_range(start=last_datetime, periods=horizon+1, freq=freq_used)[1:]
    for model in models:
        run_id = model_to_run.get(model)
        if not run_id:
            errors[model] = f"No trained model found for '{model}'. Please train this model first."
            continue
        try:
            model_path = client.download_artifacts(run_id, "model", ".")
            import glob, joblib
            model_files = glob.glob(f"{model_path}/*.pkl") + glob.glob(f"{model_path}/*.joblib")
            if not model_files:
                errors[model] = f"No model artifact found for '{model}'."
                continue
            model_obj = joblib.load(model_files[0])
            if model in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
                future_df = pd.DataFrame(index=future_dates)
                last_val = df['consumption'].iloc[-1]
                future_df['lag1'] = last_val
                future_df['month'] = future_df.index.month
                future_df['weekday'] = future_df.index.weekday
                X_future = future_df[['lag1', 'month', 'weekday']]
                forecast_vals = model_obj.predict(X_future)
            elif model == "Naive":
                forecast_vals = [df['consumption'].iloc[-1]] * horizon
            elif model == "ARIMA":
                forecast_vals = model_obj.forecast(horizon)
            elif model == "SARIMA":
                forecast_vals = model_obj.forecast(horizon)
            else:
                errors[model] = f"Unknown model type '{model}'."
                continue
            forecasts[model] = {"datetime": future_dates.strftime("%Y-%m-%d %H:%M:%S").tolist(), "forecast": list(forecast_vals)}
        except Exception as e:
            errors[model] = f"Forecast failed for '{model}': {str(e)}"
            logging.exception(f"Forecast failed for model {model}")
    debug_info = {
        "models_requested": models,
        "models_found": list(model_to_run.keys()),
        "dataset_shape": df.shape,
        "dataset_head": df.head(2).to_dict(),
        "errors": errors
    }
    if not forecasts:
        return {"error": "No forecasts could be generated.", "details": debug_info}
    forecasts["debug_info"] = debug_info
    return forecasts

@app.delete("/delete_model/{run_id}")
async def delete_model(run_id: str):
    client = MlflowClient()
    try:
        client.delete_run(run_id)
        return {"status": "success", "message": f"Model run {run_id} deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/download_model/{run_id}")
async def download_model(run_id: str):
    import glob
    client = MlflowClient()
    model_path = client.download_artifacts(run_id, "model", ".")
    model_files = glob.glob(f"{model_path}/*.joblib") + glob.glob(f"{model_path}/*.pkl")
    if not model_files:
        return {"status": "error", "message": "No model artifact found."}
    return FileResponse(model_files[0], filename=model_files[0].split(os.sep)[-1])

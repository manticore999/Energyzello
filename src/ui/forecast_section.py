import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go

def show_forecast_section(model_history):
    st.write("Select one or more trained models and a forecast period to generate and compare forecasts.")
    if not model_history:
        st.info("No trained models available for forecasting. Train at least one model to enable forecasting.")
        return
    model_names = [m['model'] for m in model_history]
    selected_models = st.multiselect(
        "Models to forecast with:",
        model_names,
        help="Choose one or more models you have trained.",
        key=f"forecast_multiselect_{str(model_names)}"
    )
    period_type = st.selectbox("Forecast period type:", ["Days", "Months", "Years"], help="Select the unit for the forecast horizon.")
    period_value = st.number_input(f"Forecast horizon (number of {period_type.lower()}):", min_value=1, max_value=1000, value=1, help="How many periods into the future to forecast.")
    forecast_btn = st.button("Generate Forecasts")
    if forecast_btn:
        if not selected_models:
            st.warning("Please select at least one model to forecast with.")
            return
        with st.spinner("Generating forecasts..."):
            response = requests.post(
                "http://localhost:8000/forecast/",
                json={"models": selected_models, "horizon": int(period_value), "period_type": period_type.lower()}
            )
        if response.status_code == 200:
            forecasts = response.json()
            if not forecasts:
                st.info("No forecasts could be generated for the selected models.")
                return
            # Show backend debug info if present
            if 'error' in forecasts:
                st.error(forecasts['error'])
                if 'details' in forecasts:
                    st.expander('Debug Info').write(forecasts['details'])
                return
            fig = go.Figure()
            for model, data in forecasts.items():
                if model == 'debug_info':
                    with st.expander('Debug Info'):
                        st.write(data)
                    continue
                if isinstance(data, dict) and "datetime" in data and "forecast" in data and len(data["datetime"]) == len(data["forecast"]) and len(data["datetime"]) > 0:
                    df_pred = pd.DataFrame(data)
                    fig.add_trace(go.Scatter(x=df_pred['datetime'], y=df_pred['forecast'], mode='lines', name=model))
                else:
                    st.warning(f"Forecast for {model} is invalid or empty.")
            fig.update_layout(title="Forecast Comparison", xaxis_title="Date", yaxis_title="Forecasted Value")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Failed to generate forecasts: " + response.text)

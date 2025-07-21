import streamlit as st
import base64

def show_advanced_viz_section(vis):
    st.markdown("### Calendar Heatmap of Daily Average Consumption")
    st.image(base64.b64decode(vis['calendar_heatmap']), use_container_width=True)
    st.markdown(":calendar: **This heatmap shows the average consumption for each day of each month.**\n\n- Darker colors indicate higher consumption.\n- Use this to spot seasonal patterns, holidays, or unusual days.")
    st.markdown("---")
    st.markdown("### Autocorrelation & Partial Autocorrelation (ACF/PACF)")
    st.image(base64.b64decode(vis['acf_pacf_plot']), use_container_width=True)
    st.markdown(":repeat: **ACF shows how current values relate to past values (lags).**\n\n- Peaks at regular lags suggest seasonality.\n- PACF helps identify the order of autoregressive models.\n- Use these plots to understand memory and seasonality in your data.")

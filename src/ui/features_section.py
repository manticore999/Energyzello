import streamlit as st
import base64

def show_features_section(date_feats, out_feats):
    st.markdown("## Date Feature Distributions")
    st.markdown("### Distribution by Year")
    st.image(base64.b64decode(date_feats['year']), use_container_width=True)
    st.markdown(":calendar: **This plot shows how many records fall in each year.**\n\n- Use this to check for missing years, data gaps, or trends over time.")
    st.markdown("### Distribution by Month")
    st.image(base64.b64decode(date_feats['month']), use_container_width=True)
    st.markdown(":calendar: **This plot shows the distribution of records by month.**\n\n- Peaks may indicate seasonality or data collection patterns.")
    st.markdown("### Distribution by Day of Month")
    st.image(base64.b64decode(date_feats['day']), use_container_width=True)
    st.markdown(":calendar: **This plot shows how data is distributed across days of the month.**\n\n- Use this to spot missing days or irregular sampling.")
    st.markdown("### Distribution by Weekday")
    st.image(base64.b64decode(date_feats['weekday']), use_container_width=True)
    st.markdown(":calendar: **This plot shows the distribution by weekday (0=Monday, 6=Sunday).**\n\n- Use this to spot weekly seasonality or business/holiday effects.")
    st.markdown("---")
    st.markdown("## Outlier Detection and Advanced Features")
    st.markdown("### Outliers in Time Series")
    st.image(base64.b64decode(out_feats['outlier_plot']), use_container_width=True)
    st.markdown(":rotating_light: **Red points are outliers detected using the IQR method.**\n\n- Outliers are values that are unusually high or low compared to the rest of the data.\n- They can indicate errors, rare events, or important changes.\n- Outliers can strongly affect forecasting models, so review them carefully.")
    st.markdown("### 7-day Rolling Mean")
    st.image(base64.b64decode(out_feats['rolling_mean_plot']), use_container_width=True)
    st.markdown(":repeat: **The orange line is the 7-day rolling mean.**\n\n- Rolling means smooth out short-term fluctuations and highlight longer-term trends.\n- Use this to see overall patterns and seasonality.")
    st.markdown("### Lag Feature (Lag-1)")
    st.image(base64.b64decode(out_feats['lag_feature_plot']), use_container_width=True)
    st.markdown(":leftwards_arrow_with_hook: **This scatter plot shows the relationship between the previous value (lag-1) and the current value.**\n\n- Strong correlation means the past value is a good predictor of the next value.\n- Lag features are important for time series forecasting models.")

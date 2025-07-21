import streamlit as st
import base64

def show_decomposition_section(decomp_response):
    if decomp_response.status_code == 200:
        decomp = decomp_response.json()
        if decomp.get('success'):
            st.markdown("## Time Series Decomposition")
            st.markdown("### Trend Component")
            st.image(base64.b64decode(decomp['plots']['trend']), use_container_width=True)
            st.markdown(""":chart_with_upwards_trend: **The trend component shows the long-term progression of the series.**
- The trend graph helps you see if your data is generally increasing, decreasing, or stable over time.
- If the line goes up, your data is rising; if it goes down, it’s falling. Focus on the overall direction, not short-term bumps.
- Understanding the trend is crucial for long-term forecasting and business planning.
""")
            st.markdown("### Seasonal Component")
            st.image(base64.b64decode(decomp['plots']['seasonal']), use_container_width=True)
            st.markdown(""":calendar: **The seasonal component captures repeating patterns (e.g., daily, weekly, yearly).**
- Seasonality means regular cycles in your data, like higher sales on weekends or energy use in summer.
- Peaks and valleys that repeat at regular intervals mean your data has seasonality. This is important for choosing models that can handle such patterns.
""")
            st.markdown("### Residual Component")
            st.image(base64.b64decode(decomp['plots']['resid']), use_container_width=True)
            st.markdown(""":mag: **The residual (or remainder) is what's left after removing trend and seasonality.**
- It shows irregular, random, or unexplained variation in your data.
- Large residuals may indicate outliers, anomalies, or model limitations.
""")
        else:
            st.error(f"Decomposition failed: {decomp.get('error')}")
    else:
        st.error("Decomposition analysis failed: " + decomp_response.text)

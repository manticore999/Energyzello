import streamlit as st
import base64

def show_missing_section(missing):
    if missing.get('has_missing'):
        st.markdown("## Missing Data Analysis")
        st.markdown(f":warning: **There are {missing['n_missing']} missing values in your data.**\n\n- Missing data can occur due to sensor errors, data loss, or other issues.\n- It can affect model performance and bias results if not handled.")
        st.image(base64.b64decode(missing['missing_plot']), use_container_width=True)
        st.markdown(":mag: **The plot above shows when data is missing.**\n\n- If you see clusters of missing values, consider imputation or removal.\n- If missingness is random and rare, simple methods may suffice.")
    else:
        st.markdown(":white_check_mark: **No missing data detected in your time series!**")

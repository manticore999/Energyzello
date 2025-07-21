import streamlit as st
import pandas as pd
import requests
from io import BytesIO

def show_model_history_section():
    st.markdown("## Model History")
    response = requests.get("http://localhost:8000/model_history/")
    if response.status_code == 200:
        history = response.json()
        if history:
            df_history = pd.DataFrame(history)
            df_history['start_time'] = pd.to_datetime(df_history['start_time'], unit='ms')
            # Only show user-relevant columns
            st.dataframe(df_history[['model', 'mae', 'rmse', 'start_time']].sort_values("rmse"))
            st.markdown("Best model is highlighted at the top (lowest RMSE).")
            for idx, row in df_history.iterrows():
                st.write(f"**Model:** {row['model']} | **MAE:** {row['mae']} | **RMSE:** {row['rmse']} | **Trained:** {row['start_time']}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Delete {row['model']} ({row['start_time']})", key=f"delete_{idx}"):
                        del_resp = requests.delete(f"http://localhost:8000/delete_model/{row['run_id']}")
                        if del_resp.status_code == 200:
                            st.success(f"Deleted model {row['model']}.")
                        else:
                            st.error(f"Failed to delete model: {del_resp.text}")
                with col2:
                    if st.button(f"Download {row['model']} ({row['start_time']})", key=f"download_{idx}"):
                        dl_resp = requests.get(f"http://localhost:8000/download_model/{row['run_id']}")
                        if dl_resp.status_code == 200:
                            st.download_button(
                                label=f"Download {row['model']} Model File",
                                data=BytesIO(dl_resp.content),
                                file_name=f"{row['model']}_model.joblib",
                                mime='application/octet-stream'
                            )
                        else:
                            st.error(f"Failed to download model: {dl_resp.text}")
        else:
            st.info("No model history found for this dataset.")
    else:
        st.error("Failed to fetch model history.")

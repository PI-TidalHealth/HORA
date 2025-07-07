import streamlit as st
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px 
from pathlib import Path
from modules.layout import set_narrow,set_fullwidth
from modules.function import process_schedule_excel
def render_upload_page():
    set_narrow(800)
    ROOT = Path(__file__).resolve().parents[1]
    LOGO = ROOT / "assets" / "Logo4.PNG"
    col_left, col_middle, col_right = st.columns([1, 12, 1])
    with col_middle:
        st.image(str(LOGO), use_container_width=True)
        uploaded = st.file_uploader(
            "Upload a file", 
            type=["csv", "xlsx"], 
            key="crna_uploader"
        )

    if not uploaded:
        return

    # Don't show buttons if the dialog is supposed to be open
    if not st.session_state.get("show_capacity_date_dialog"):
        btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([2,2,1,2,2])
        with btn_col2:
            demand_clicked = st.button("Demand ➡️", use_container_width=True, key="demand_btn")
        with btn_col4:
            capacity_clicked = st.button("Capacity ➡️", use_container_width=True, key="capacity_btn")

        if demand_clicked:
            st.session_state.uploaded_file = uploaded
            st.session_state.page = "step1"
            st.rerun()

        if capacity_clicked:
            st.session_state.show_capacity_date_dialog = True
            st.session_state.uploaded_file_for_capacity = uploaded
            st.rerun()

    if st.session_state.get("show_capacity_date_dialog"):
        @st.dialog("Select Date Range")
        def get_dates():
            default_start = datetime(2025, 1, 1)
            default_end = datetime(2025, 3, 31)
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=default_end)
            if st.button("Next ➡️"):
                st.session_state.run_processing = {
                    "start_date": start_date.strftime('%Y/%m/%d'),
                    "end_date": end_date.strftime('%Y/%m/%d')
                }
                st.rerun()

        get_dates()

    if st.session_state.get("run_processing"):
        processing_info = st.session_state.run_processing
        uploaded = st.session_state.get("uploaded_file_for_capacity")

        if uploaded:
            import tempfile
            import os

            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded.getbuffer())
                tmp_path = tmp_file.name

            # process_schedule_excel returns a pandas DataFrame
            result_df_pd = process_schedule_excel(
                tmp_path,
                start_date_str=processing_info["start_date"],
                end_date_str=processing_info["end_date"]
            )
            
            # --- Auto-select columns and prepare for analysis ---
            # Keep only the necessary columns and rename them for consistency
            final_df_pd = result_df_pd[['Date', 'In Time', 'Out Time']].copy()
            final_df_pd.rename(columns={'In Time': 'In Room', 'Out Time': 'Out Room'}, inplace=True)

            # Add a 'Count' column, defaulting to 1, similar to the demand flow
            final_df_pd['Count'] = 1
            # --- End of auto-selection ---
            
            # --- Calculate total months for normalization ---
            if not final_df_pd.empty:
                # Ensure 'Date' column is in datetime format
                final_df_pd['Date'] = pd.to_datetime(final_df_pd['Date'])
                # Calculate the number of unique months
                total_months = final_df_pd['Date'].dt.to_period('M').nunique()
            else:
                total_months = 1 # Avoid division by zero if dataframe is empty
            
            st.session_state['total_months'] = total_months
            # --- End of total months calculation ---

            # Convert to Polars DataFrame for consistency with other pages
            crna_data_pl = pl.from_pandas(final_df_pd)

            # Set the data and analysis parameters directly for the next page
            st.session_state["crna_data"] = crna_data_pl
            st.session_state.analysis_type = "presence"
            st.session_state.analysis_view = "month"
            
            st.session_state.start_date_str = processing_info["start_date"]
            st.session_state.end_date_str = processing_info["end_date"]

            st.session_state.page = "sce2_capacity"

            del st.session_state.run_processing
            if "show_capacity_date_dialog" in st.session_state:
                del st.session_state.show_capacity_date_dialog
            if "uploaded_file_for_capacity" in st.session_state:
                del st.session_state.uploaded_file_for_capacity
            st.rerun()
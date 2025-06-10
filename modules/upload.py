try:
    import polars as pl
except ImportError as e:
    import streamlit as st
    st.error(f"Failed to import polars. Error: {str(e)}")
    st.info("Try restarting the Streamlit server after installing polars.")
    import sys
    st.write("Python path:", sys.path)
    st.write("Python version:", sys.version)
    raise

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px 
from pathlib import Path
from modules.layout import set_narrow,set_fullwidth

def render_upload_page():

    set_narrow(800)
    ROOT = Path(__file__).resolve().parents[1]

    LOGO = ROOT / "assets" / "Logo4.PNG"
    col_left, col_middle, col_right = st.columns([1, 12, 1])
    with col_middle:
        # This is your logo
        st.image(str(LOGO), use_container_width=True)

        # This is the single file uploader (do NOT duplicate it below)
        uploaded = st.file_uploader(
            "Upload a file", 
            type=["csv", "xlsx"], 
            key="crna_uploader"
        )

    if not uploaded:
      return


    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([2,2,1,2,2])
    with btn_col3:  
        if st.button("Next ➡️", use_container_width=True):  
            st.session_state.uploaded_file = uploaded
            st.session_state.page = "step1"
            st.rerun()
    return  
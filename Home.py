import pandas as pd
from modules.upload import render_upload_page
from modules.sce3 import week_analysis
from modules.sce2 import month_analysis 
from modules.sce4 import duration_month_analysis
from modules.sce5 import duration_week_analysis
from modules.step1 import uploadstep1_page
from modules.sce2_capacity import month_capacity_analysis 
import streamlit as st 
#Streamlit 
st.set_page_config(
    page_title="HORA",
    page_icon="assets/HORA-Webpage-Icon.png",
    layout="wide" 
)


if "page" not in st.session_state:
    st.session_state.page = "Upload"
if st.session_state.page == "Upload":
    render_upload_page()
elif st.session_state.page== "step1":
    uploadstep1_page()
# if session_state.page == "Upload", we will run the function "render_upload_page()"
elif st.session_state.page == "sce2":
    month_analysis()
elif st.session_state.page == "sce3":
    week_analysis()
elif st.session_state.page == "sce4":
    duration_month_analysis()
elif st.session_state.page == "sce5":
    duration_week_analysis()
elif st.session_state.page == "sce2_capacity":
    month_capacity_analysis()

import streamlit as st
from pathlib import Path
from modules.layout import set_narrow

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
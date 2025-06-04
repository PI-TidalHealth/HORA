import pandas as pd
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
    #找到root目录
    LOGO = ROOT / "assets" / "Logo1.PNG"
    #这样做的目的是找到项目的根目录

    st.image(
        str(LOGO),
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload a file", type=["csv","xlsx"], key="crna_uploader")
    #KEY是用来确保每次上传文件时，文件名都是唯一的,用于在 session_state 中存储上传状态
    if not uploaded:
      return

    # 使用5列布局，中间列放按钮
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([2,2,1,2,2])
    with btn_col3:  # 使用中间列
        if st.button("Next ➡️", use_container_width=True):  # 添加 use_container_width=True 使按钮填充列宽
            st.session_state.uploaded_file = uploaded
            st.session_state.page = "step1"
            st.rerun()
    return  
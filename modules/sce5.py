import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import zipfile
import polars as pl
from modules.function import _parse_time_series
from modules.function import _compute_duration_matrix  
from modules.function import _compute_week_hm_data

# —— Global constants and precomputed values —— #

_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']



@st.cache_data(show_spinner=False)
def _assign_month_week(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a 'week_of_month' column (Week 1 through Week 5) to a DataFrame that already has a 'Date' column.
    """
    # Ensure Date is in the correct format
    temp = df
    if temp.schema["Date"] != pl.Date:
        try:
            temp = temp.with_columns([
                pl.col("Date").str.strptime(pl.Date, format="%Y-%m-%d").alias("Date")
            ])
        except:
            temp = temp.with_columns([
                pl.col("Date").cast(pl.Date).alias("Date")
            ])
    
    # Add day and week_of_month columns
    temp = temp.with_columns([
        pl.col("Date").dt.day().alias("day")
    ]).with_columns([
        pl.when(pl.col("day") <= 7)
          .then(pl.lit("Week 1"))
          .when(pl.col("day") <= 14)
          .then(pl.lit("Week 2"))
          .when(pl.col("day") <= 21)
          .then(pl.lit("Week 3"))
          .when(pl.col("day") <= 28)
          .then(pl.lit("Week 4"))
          .otherwise(pl.lit("Week 5"))
          .alias("week_of_month")
    ])
    
    return temp


def duration_week_analysis():
    st.markdown(
        """
        <style>
        /* Limit expander maximum width */
        div[data-testid="stExpander"] {
            max-width: 100px;
        }
        /* Reduce padding inside the expander header */
        div[role="button"][aria-expanded] {
            padding: 0.25rem 0.5rem;
        }
        /* Reduce padding inside the expander content */
        div[data-testid="stExpander"] > div {
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    raw_df = st.session_state['crna_data']
    if isinstance(raw_df, pl.DataFrame):
        df_pl = raw_df.clone()
    else:
        import pandas as _pd  # fallback if user uploaded pandas DataFrame
        df_pl = pl.from_pandas(raw_df)

    # —— 1. Ensure 'Date' is date —— #
    # Convert Date column more robustly
    try:
        df_pl = df_pl.with_columns([
            # First standardize the format by replacing all separators with '/'
            pl.col('Date').str.replace_all(r'[-.]', '/').alias('Date')
        ]).with_columns([
            # Then try both date formats with strict=False
            pl.when(pl.col('Date').str.strptime(pl.Date, format='%Y/%m/%d', strict=False).is_not_null())
                .then(pl.col('Date').str.strptime(pl.Date, format='%Y/%m/%d', strict=False))
                .otherwise(
                    pl.col('Date').str.strptime(pl.Date, format='%m/%d/%Y', strict=False)
                )
                .alias('Date')
        ])
    except:
        # Fallback: if above fails, try without format conversion
        df_pl = df_pl.with_columns([
            pl.col("Date").cast(pl.Date).alias("Date")
        ])

    # —— 2. Compute Duration matrix (with cache + spinner) —— #
    with st.spinner("Computing duration data (this may take a few seconds)…"):
        output = _compute_duration_matrix(df_pl)

    # —— 3. Add 'week_of_month' column to output (with cache) —— #
    weekfile_detail = _assign_month_week(output)

    # —— 4. Dropdown for user to select which week to display —— #
    selected_wk = st.selectbox("📊 Select Week to Display", _WEEKS_LIST)

    # —— 5. Compute the heatmap data for the selected week (with cache + spinner) —— #
    with st.spinner(f"Computing heatmap data for {selected_wk}…"):
        hm_data_pl = _compute_week_hm_data(weekfile_detail, selected_wk)
        hm_data = hm_data_pl.to_pandas().set_index('weekday')

    # —— 6. Let user customize the chart title —— #
    default_title = f"Duration for {selected_wk}"
    key = f"title_{selected_wk}"
    title_input = st.text_input(
        label=f"{selected_wk} Chart Title",
        value=default_title,
        key=key
    )

    # —— 7. Plot the heatmap for the selected week —— #
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(hm_data, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title_input, fontdict={'fontsize': 18, 'fontweight': 'bold'}, loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # —— 8. Left column: download the current week's PNG/CSV —— #
    col_l, _, col_r = st.columns([1, 8, 1])
    with col_l:
        with st.expander(f"💾 Save {selected_wk}", expanded=False):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                label="🏞️ PNG",
                data=buf.getvalue(),
                file_name=f"{title_input}.png",
                mime="image/png"
            )
            csv_bytes = hm_data.to_csv().encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv_bytes,
                file_name=f"{title_input}.csv",
                mime="text/csv"
            )

    # —— 9. Right column: download all weeks' PNGs and CSVs zipped —— #
    with col_r:
        with st.expander("💾 Save All Weeks", expanded=False):
            # Create an in-memory ZIP file for all PNGs
            png_zip = io.BytesIO()
            with zipfile.ZipFile(png_zip, mode="w") as zf:
                for wk in _WEEKS_LIST:
                    df_hm_pl = _compute_week_hm_data(weekfile_detail, wk)
                    df_hm = df_hm_pl.to_pandas().set_index('weekday')
                    title_key = f"title_{wk}"
                    user_title = st.session_state.get(title_key, f"Duration for {wk}")
                    fig_w, ax_w = plt.subplots(figsize=(10, 3))
                    sns.heatmap(df_hm, annot=True, linewidths=0.5, cmap="RdYlGn_r", ax=ax_w)
                    ax_w.set_title(user_title, loc="center")
                    plt.tight_layout()
                    buf_w = io.BytesIO()
                    fig_w.savefig(buf_w, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_w)
                    zf.writestr(f"{wk}_duration_heatmap.png", buf_w.getvalue())
            png_zip.seek(0)
            st.download_button(
                label="🏞️ PNGs",
                data=png_zip.getvalue(),
                file_name="all_weeks_duration_heatmaps.zip",
                mime="application/zip"
            )
            # Create a separate in-memory ZIP file for all CSVs
            csv_zip = io.BytesIO()
            with zipfile.ZipFile(csv_zip, mode="w") as zf2:
                for wk in _WEEKS_LIST:
                    df_hm_pl = _compute_week_hm_data(weekfile_detail, wk)
                    df_hm = df_hm_pl.to_pandas().set_index('weekday')
                    csv_bytes = df_hm.to_csv().encode("utf-8")
                    zf2.writestr(f"{wk}_duration_heatmap_data.csv", csv_bytes)
            csv_zip.seek(0)
            st.download_button(
                label="📥 CSVs",
                data=csv_zip.getvalue(),
                file_name="all_weeks_duration_heatmap_data.zip",
                mime="application/zip"
            )

    # —— 10. 'Back' and 'Go to Month Analysis' buttons —— #
    back_col, _, month_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            keys_to_remove = [
                "crna_data",
                "analysis_type",
                "analysis_view",
                "col_error",
                "duplicate_error",
                "format_error",
                "format_error_msg",
                "uploaded_file",
                "crna_uploader"
            ]
            for key in keys_to_remove:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.page = "Upload"
            st.rerun()

    with month_col:
        if st.button("🔍 Go to Month Analysis"):
            if st.session_state.analysis_type == "presence":
                st.session_state.page = "sce2"
            else:
                st.session_state.page = "sce4"
            st.rerun()

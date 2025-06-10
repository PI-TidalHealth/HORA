import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io, zipfile
import pandas as pd

# —— Global constants and precomputed values —— #
_REFERENCE_DATE = "1900-01-01 "
_TIME_BIN_START = [datetime.strptime(f"{_REFERENCE_DATE}{h:02d}:00", "%Y-%m-%d %H:%M") for h in range(24)]
_TIME_BIN_END   = [datetime.strptime(f"{_REFERENCE_DATE}{(h+1)%24:02d}:00", "%Y-%m-%d %H:%M") for h in range(24)]
_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']


@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorize the original df (which must include 'In Room', 'Out Room', 'Count')
    into a DataFrame containing 24 columns (0–23 hours).
    """
    # Convert to pandas temporarily for easier time processing
    temp_pd = df.to_pandas()
    
    # Drop rows with null values
    temp_pd = temp_pd.dropna(subset=['In Room', 'Out Room']).copy()
    
    if len(temp_pd) == 0:
        # Return empty result with all required columns
        empty_cols = {str(h): 0 for h in range(24)}
        return pl.from_pandas(temp_pd.assign(**empty_cols))
    
    # Parse times using pandas (more forgiving)
    try:
        temp_pd['In_time'] = pd.to_datetime('1900-01-01 ' + temp_pd['In Room'].astype(str))
        temp_pd['Out_time'] = pd.to_datetime('1900-01-01 ' + temp_pd['Out Room'].astype(str))
    except:
        # Fallback: try parsing as datetime directly
        temp_pd['In_time'] = pd.to_datetime(temp_pd['In Room'])
        temp_pd['Out_time'] = pd.to_datetime(temp_pd['Out Room'])
    
    # Handle cross-day cases
    mask = temp_pd['Out_time'] < temp_pd['In_time']
    temp_pd.loc[mask, 'Out_time'] += pd.Timedelta(days=1)
    
    # Ensure Count is integer
    temp_pd['Count'] = temp_pd['Count'].astype(int)
    
    # Create 24 hour columns
    for h in range(24):
        hour_start = pd.to_datetime(f'1900-01-01 {h:02d}:00:00')
        hour_end = pd.to_datetime(f'1900-01-01 {(h+1)%24:02d}:00:00')
        if h == 23:  # Handle 23:00-00:00 (next day)
            hour_end = pd.to_datetime('1900-01-02 00:00:00')
        
        # Check overlap: in_time < hour_end AND out_time > hour_start
        overlap = (temp_pd['In_time'] < hour_end) & (temp_pd['Out_time'] > hour_start)
        temp_pd[str(h)] = (overlap * temp_pd['Count']).astype(int)
    
    # Convert back to polars
    return pl.from_pandas(temp_pd)


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


@st.cache_data(show_spinner=False)
def _compute_week_hm_data(df_with_time: pl.DataFrame, week_label: str) -> pl.DataFrame:
    """
    Filter df_with_time by a given week_label ('Week 1'…'Week 5') and generate a 7×24 heatmap DataFrame:
    1. Group by 'weekday' × hours 0–23 to get raw_counts;
    2. If week_label is 'Week 5', fill NaN with 0;
    3. Divide by 3, round, and convert to int.
    Return a sorted DataFrame with index _WEEKDAY_ORDER and columns 0–23.
    """
    # First ensure all hour columns exist with default 0
    hour_cols = [str(h) for h in range(24)]
    for col in hour_cols:
        if col not in df_with_time.columns:
            df_with_time = df_with_time.with_columns(pl.lit(0).alias(col))

    # Filter by week and compute weekday
    df_wk = (
        df_with_time
        .filter(pl.col("week_of_month") == week_label)
        .with_columns([
            pl.col("Date").dt.strftime("%A").alias("weekday")
        ])
    )

    # Sum by weekday for each hour
    agg = (
        df_wk
        .group_by("weekday")
        .agg([
            pl.col(col).sum().alias(col) for col in hour_cols
        ])
        .filter(pl.col("weekday").is_in(_WEEKDAY_ORDER))
    )
    
    # Create a mapping for weekday order and sort manually
    weekday_order_map = {day: i for i, day in enumerate(_WEEKDAY_ORDER)}
    agg = agg.with_columns([
        pl.col("weekday").map_elements(lambda x: weekday_order_map.get(x, 999), return_dtype=pl.Int32).alias("weekday_order")
    ]).sort("weekday_order").drop("weekday_order")

    if week_label == "Week 5":
        agg = agg.fill_null(0)

    # Normalize by dividing by 3
    hm_data = agg.with_columns([
        (pl.col(col) / 3).round().cast(pl.Int64).alias(col) for col in hour_cols
    ])

    return hm_data


def week_analysis():
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
            pl.col("Date")
              .cast(pl.String)
              .str.replace_all("/", "-")
              .str.strptime(pl.Date, format="%Y-%m-%d")
              .alias("Date")
        ])
    except:
        # Fallback: if above fails, try without format conversion
        df_pl = df_pl.with_columns([
            pl.col("Date").cast(pl.Date).alias("Date")
        ])

    # —— 2. Compute Presence matrix (with cache + spinner) —— #
    with st.spinner("Computing presence data (this may take a few seconds)…"):
        output = _compute_presence_matrix(df_pl)

    # —— 3. Add 'week_of_month' column to output (with cache) —— #
    weekfile_detail = _assign_month_week(output)

    # —— 4. Dropdown for user to select which week to display —— #
    selected_wk = st.selectbox("📊 Select Week to Display", _WEEKS_LIST)

    # —— 5. Compute the heatmap data for the selected week (with cache + spinner) —— #
    with st.spinner(f"Computing heatmap data for {selected_wk}…"):
        hm_data_pl = _compute_week_hm_data(weekfile_detail, selected_wk)
        hm_data = hm_data_pl.to_pandas().set_index('weekday')

    # —— 6. Let user customize the chart title —— #
    default_title = f"Presense for {selected_wk}"
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
                file_name=f"{selected_wk}_heatmap.png",
                mime="image/png"
            )
            csv_bytes = hm_data.to_csv().encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv_bytes,
                file_name=f"{selected_wk}_heatmap_data.csv",
                mime="text/csv"
            )

    # —— 9. Right column: download all weeks' PNGs and CSVs zipped —— #
    with col_r:
        with st.expander("💾 Save All Weeks", expanded=False):
            # Create an in-memory ZIP file for all PNGs
            png_zip = io.BytesIO()
            with zipfile.ZipFile(png_zip, mode="w") as zf:
                for wk in _WEEKS_LIST:
                    # Recompute this week's heatmap data:
                    df_hm_pl = _compute_week_hm_data(weekfile_detail, wk)
                    df_hm = df_hm_pl.to_pandas().set_index('weekday')

                    # Get the user's custom title for this week from session_state:
                    # If the user never changed it, fall back to the default
                    title_key = f"title_{wk}"
                    user_title = st.session_state.get(title_key, f"Demand for {wk}")

                    # Build a small figure for this week's heatmap:
                    fig_w, ax_w = plt.subplots(figsize=(20, 5))
                    sns.heatmap(df_hm, annot=True, linewidths=0.5, cmap="RdYlGn_r", ax=ax_w)

                    # Use the user's custom title
                    ax_w.set_title(user_title, loc="center")
                    plt.tight_layout()

                    # Save that figure into a bytes buffer
                    buf_w = io.BytesIO()
                    fig_w.savefig(buf_w, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_w)

                    # Write the buffer to the ZIP under a chosen filename
                    zf.writestr(f"{wk}_heatmap.png", buf_w.getvalue())

            png_zip.seek(0)
            st.download_button(
                label="🏞️ PNGs",
                data=png_zip.getvalue(),
                file_name="all_weeks_heatmaps.zip",
                mime="application/zip"
            )

            # Create a separate in-memory ZIP file for all CSVs
            csv_zip = io.BytesIO()
            with zipfile.ZipFile(csv_zip, mode="w") as zf2:
                for wk in _WEEKS_LIST:
                    df_hm_pl = _compute_week_hm_data(weekfile_detail, wk)
                    df_hm = df_hm_pl.to_pandas().set_index('weekday')
                    csv_bytes = df_hm.to_csv().encode("utf-8")
                    zf2.writestr(f"{wk}_heatmap_data.csv", csv_bytes)
            csv_zip.seek(0)
            st.download_button(
                label="📥 CSVs",
                data=csv_zip.getvalue(),
                file_name="all_weeks_heatmap_data.zip",
                mime="application/zip"
            )

    # —— 10. 'Back' and 'Go to Month Analysis' buttons —— #
    back_col, _, month_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            # Clear all relevant session_state keys
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

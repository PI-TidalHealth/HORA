import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import zipfile

# —— Global constants & precomputations —— #
_REFERENCE_DATE = "1900-01-01 "
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])
_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """Convert 'HH:MM' strings into Timestamp('1900-01-01 HH:MM')."""
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_duration_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize each row's In/Out into 24 columns of duration (in hours).
    Returns original columns + new columns 0..23 (floats).
    """
    temp = (
        df
        .dropna(subset=['In Room', 'Out Room'])
        .reset_index(drop=True)
        .copy()
    )
    temp['In Room']  = temp['In Room'].astype(str).str.slice(0, 5)
    temp['Out Room'] = temp['Out Room'].astype(str).str.slice(0, 5)

    in_times  = _parse_time_series(temp['In Room'])
    out_times = _parse_time_series(temp['Out Room'])
    wrap_mask = out_times < in_times
    out_times = out_times.where(~wrap_mask, out_times + pd.Timedelta(days=1))

    # Build (N×1) arrays for row start/end, (1×24) for each hour bin
    start_matrix = in_times.values.reshape(-1, 1).astype('datetime64[ns]')
    end_matrix   = out_times.values.reshape(-1, 1).astype('datetime64[ns]')
    bin_starts   = _TIME_BIN_START.values.reshape(1, -1).astype('datetime64[ns]')
    bin_ends     = _TIME_BIN_END.values.reshape(1, -1).astype('datetime64[ns]')

    # Compute overlap start/end: shape (N,24)
    overlap_start = np.maximum(start_matrix, bin_starts)
    overlap_end   = np.minimum(end_matrix, bin_ends)

    # Overlap in seconds, clip negatives to zero
    seconds = (overlap_end.astype('datetime64[s]') - overlap_start.astype('datetime64[s]')).astype(int)
    seconds = np.where(seconds < 0, 0, seconds)

    # Convert to hours (float)
    hours_matrix = seconds / 3600.0  # shape (N,24)

    # Multiply by Count if present
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    duration_matrix = hours_matrix * counts  # shape (N,24)

    # Build DataFrame for new columns 0..23
    dur_df = pd.DataFrame(
        data=duration_matrix,
        index=temp.index,
        columns=list(range(24))
    )

    # Concatenate original columns with the new 24 columns
    result = pd.concat([temp.reset_index(drop=True), dur_df.reset_index(drop=True)], axis=1)
    return result


@st.cache_data(show_spinner=False)
def _assign_month_and_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'day' and 'week_of_month' columns based on df['Date'] (assumed datetime64).
    """
    temp = df.copy()
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp['day']  = temp['Date'].dt.day
    temp['week_of_month'] = pd.cut(
        temp['day'],
        bins=[0, 7, 14, 21, 28, 31],
        labels=_WEEKS_LIST,
        right=True
    )
    return temp


@st.cache_data(show_spinner=False)
def _compute_weekday_duration(df_wk: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame already containing 0..23 columns and 'weekday',
    group by 'weekday' and sum columns 0..23, then return Series of total hours per weekday.
    """
    summed = df_wk.groupby('weekday')[list(range(24))].sum()
    # total across hours
    summed['TotalDuration'] = summed.sum(axis=1)
    return summed['TotalDuration'].reindex(_WEEKDAY_ORDER).fillna(0)


@st.cache_data(show_spinner=False)
def _compute_week1_pie_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pie chart data (Count/3) for Week 1 only.
    """
    temp = _assign_month_and_week(df)
    week1 = temp[temp['week_of_month'] == 'Week 1'][['Date', 'In Room', 'Out Room', 'Count']].copy()
    week1['Date'] = pd.to_datetime(week1['Date'])
    week1['Weekday'] = week1['Date'].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    summary = week1.groupby('Weekday')['Count'].sum().reindex(order).fillna(0).reset_index()
    summary['Count'] = (summary['Count'] / 3).round().astype(int)
    return summary


def duration_week_analysis():
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] { max-width: 400px; }
        div[role="button"][aria-expanded] { padding: 0.25rem 0.5rem; }
        div[data-testid="stExpander"] > div { padding: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    df_orig = st.session_state['crna_data'].copy()

    # 1) Compute pie data for Week 1
    with st.spinner("Computing pie chart for Week 1..."):
        pie_df = _compute_week1_pie_data(df_orig)

    st.markdown("# Week Data Charts")
    fig1 = px.pie(
        pie_df,
        values='Count',
        names='Weekday',
        title='Demand Distribution by Weekday (Week 1)',
        hole=0.3
    )
    fig1.update_layout(height=350, width=400)
    st.plotly_chart(fig1, use_container_width=False)

    # 2) Compute full duration matrix once
    with st.spinner("Computing duration matrix for each row..."):
        df_full = df_orig.copy()
        df_full['Date'] = pd.to_datetime(df_full['Date'])
        df_full['weekday'] = df_full['Date'].dt.day_name()
        df_full = _compute_duration_matrix(df_full)
        df_full = _assign_month_and_week(df_full)

    # 3) Let user select which week to display
    selected_wk = st.selectbox("📊 Select Week to Display", _WEEKS_LIST)

    # 4) Build heatmap data for the selected week
    wk_df = df_full[df_full['week_of_month'] == selected_wk]
    if selected_wk == 'Week 5':
        # Ensure missing weekdays appear as zeros
        total_series = _compute_weekday_duration(wk_df).fillna(0)
    else:
        total_series = _compute_weekday_duration(wk_df)

    # Divide by 3 (average) and convert to int
    hm_data = (total_series.div(3).round().astype(int)).to_frame().T  # shape (1,7)
    # But we need a 7×24 matrix: 
    # Actually, we want the per-hour heatmap: group by weekday over columns 0..23 and divide by 3
    raw = wk_df.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER).fillna(0)
    if selected_wk == 'Week 5':
        raw = raw.fillna(0)
    hm_matrix = (raw.div(3).round().astype(int))

    # 5) Title input for heatmap
    default_title = f"Average Demand Heatmap - {selected_wk}"
    title_input = st.text_input(label=f"{selected_wk} Chart Title", value=default_title, key=f"title_{selected_wk}")
    if len(title_input) > 40:
        title_input = title_input[:37] + "..."

    # 6) Plot heatmap
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(hm_matrix, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title_input, fontdict={'fontsize': 18, 'fontweight': 'bold'}, loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    col_l, _, col_r = st.columns([1, 8, 1])

    # —— Left: Download only selected week's PNG/CSV —— #
    with col_l:
        with st.expander("💾 Save Selected Week", expanded=False):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                label="🏞️ PNG",
                data=buf.getvalue(),
                file_name=f"{selected_wk}_heatmap.png",
                mime="image/png"
            )
            csv_bytes = hm_matrix.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv_bytes,
                file_name=f"{selected_wk}_heatmap_data.csv",
                mime="text/csv"
            )

    # —— Right: Download all weeks' PNGs and CSVs zipped —— #
    with col_r:
        with st.expander("💾 Save All Weeks", expanded=False):
            # 7) ZIP all PNGs
            png_zip = io.BytesIO()
            with zipfile.ZipFile(png_zip, mode="w") as zf:
                for wk in _WEEKS_LIST:
                    sub_df = df_full[df_full['week_of_month'] == wk]
                    raw_sub = sub_df.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER).fillna(0)
                    if wk == 'Week 5':
                        raw_sub = raw_sub.fillna(0)
                    hm_sub = (raw_sub.div(3).round().astype(int))

                    fig_w, ax_w = plt.subplots(figsize=(10, 3))
                    sns.heatmap(hm_sub, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax_w)
                    ax_w.set_title(f"Average Demand Heatmap - {wk}", loc='center', fontsize=14, fontweight='bold')
                    ax_w.set_ylabel("DOW", fontsize=12)
                    plt.tight_layout()

                    buf_w = io.BytesIO()
                    fig_w.savefig(buf_w, format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_w)
                    zf.writestr(f"{wk}_heatmap.png", buf_w.getvalue())

            png_zip.seek(0)
            st.download_button(
                label="🏞️ All Weeks PNGs",
                data=png_zip.getvalue(),
                file_name="all_weeks_heatmaps.zip",
                mime="application/zip"
            )

            # 8) ZIP all CSVs
            csv_zip = io.BytesIO()
            with zipfile.ZipFile(csv_zip, mode="w") as zf2:
                for wk in _WEEKS_LIST:
                    sub_df = df_full[df_full['week_of_month'] == wk]
                    raw_sub = sub_df.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER).fillna(0)
                    if wk == 'Week 5':
                        raw_sub = raw_sub.fillna(0)
                    hm_sub = (raw_sub.div(3).round().astype(int))

                    csv_bytes = hm_sub.to_csv(index=False).encode("utf-8")
                    zf2.writestr(f"{wk}_heatmap_data.csv", csv_bytes)

            csv_zip.seek(0)
            st.download_button(
                label="📥 All Weeks CSVs",
                data=csv_zip.getvalue(),
                file_name="all_weeks_heatmap_data.zip",
                mime="application/zip"
            )

    # —— Navigation Buttons —— #
    back_col, _, month_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            # 清除所有相关的session state
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

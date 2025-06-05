import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io, zipfile

# —— Global constants and precomputed values —— #
_REFERENCE_DATE = "1900-01-01 "
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])
_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """Convert strings in "HH:MM" format into Timestamp (1900-01-01 HH:MM) at once."""
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize the original df (which must include 'In Room', 'Out Room', 'Count')
    into a DataFrame that contains 24 columns (0–23 hours).
    Return value: original rows (without NaN) + 24 columns of "people in room" counts.
    """
    temp = (
        df
        .dropna(subset=['In Room', 'Out Room'])
        .reset_index(drop=True)
        .copy()
    )
    temp['In Room']  = temp['In Room'].astype(str)
    temp['Out Room'] = temp['Out Room'].astype(str)

    in_times  = _parse_time_series(temp['In Room'])
    out_times = _parse_time_series(temp['Out Room'])

    # If out_time < in_time, then add one day
    wrap_mask = out_times < in_times
    out_times = out_times.where(~wrap_mask, out_times + pd.Timedelta(days=1))

    # Use broadcasting to create an (N×24) boolean matrix indicating whether each row overlaps each hour
    overlap = (
        (in_times.values.reshape(-1, 1) < _TIME_BIN_END.values.reshape(1, -1)) &
        (out_times.values.reshape(-1, 1) > _TIME_BIN_START.values.reshape(1, -1))
    )
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    presence_matrix = overlap * counts  # shape (N×24), elements are either 0 or Count

    presence_df = pd.DataFrame(
        presence_matrix,
        index=temp.index,
        columns=list(range(24))
    )
    out_df = pd.concat([temp.reset_index(drop=True), presence_df.reset_index(drop=True)], axis=1)
    return out_df


@st.cache_data(show_spinner=False)
def _assign_month_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'week_of_month' column (Week 1 through Week 5) to a DataFrame that already has a 'Date' column.
    Requirement: df['Date'] is already in datetime format.
    """
    temp = df.copy()
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp['day'] = temp['Date'].dt.day
    temp['week_of_month'] = pd.cut(
        temp['day'],
        bins=[0, 7, 14, 21, 28, 31],
        labels=_WEEKS_LIST,
        right=True
    )
    return temp


@st.cache_data(show_spinner=False)
def _compute_week_hm_data(df_with_time: pd.DataFrame, week_label: str) -> pd.DataFrame:
    """
    Filter df_with_time by a given week_label ('Week 1'…'Week 5') and generate a 7×24 heatmap DataFrame:
    1. Group by 'weekday' × hours 0–23 to get raw_counts;
    2. If week_label is 'Week 5', fill NaN with 0;
    3. Divide by 3, round, and convert to int.
    Return a sorted DataFrame with index _WEEKDAY_ORDER and columns 0–23.
    """
    # Keep only rows for the specified week
    df_wk = df_with_time[df_with_time['week_of_month'] == week_label]
    df_wk['weekday'] = df_wk['Date'].dt.day_name()

    raw = df_wk.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER)
    if week_label == 'Week 5':
        raw = raw.fillna(0)

    hm_data = raw.div(3).round().astype(int)
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

    df = st.session_state['crna_data'].copy()
    st.markdown("# Week Data Charts")

    # —— 1. Convert 'Date' to datetime —— #
    df['Date'] = pd.to_datetime(df['Date'])

    # —— 2. Compute Presence matrix (with cache + spinner) —— #
    with st.spinner("Computing presence data (this may take a few seconds)…"):
        output = _compute_presence_matrix(df)

    # —— 3. Add 'week_of_month' column to output (with cache) —— #
    weekfile_detail = _assign_month_week(output)

    # —— 4. Dropdown for user to select which week to display —— #
    selected_wk = st.selectbox("📊 Select Week to Display", _WEEKS_LIST)

    # —— 5. Compute the heatmap data for the selected week (with cache + spinner) —— #
    with st.spinner(f"Computing heatmap data for {selected_wk}…"):
        hm_data = _compute_week_hm_data(weekfile_detail, selected_wk)

    # —— 6. Let user customize the chart title —— #
    default_title = f"Demand for {selected_wk}"
    key = f"title_{selected_wk}"
    if key not in st.session_state:
        # First time seeing this week: preload with the default
        st.session_state[key] = default_title
    title_input = st.text_input(
        label=f"{selected_wk} Chart Title",
        value=st.session_state[key],      # only used on first render; thereafter the stored value is used
        key=key
    )

    # —— 7. Plot the heatmap for the selected week —— #
    fig, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(hm_data, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title_input, fontdict={'fontsize': 18, 'fontweight': 'bold'}, loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # —— 8. Left column: download the current week’s PNG/CSV —— #
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
            csv_bytes = hm_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv_bytes,
                file_name=f"{selected_wk}_heatmap_data.csv",
                mime="text/csv"
            )

    # —— 9. Right column: download all weeks’ PNGs and CSVs zipped —— #
    with col_r:
        with st.expander("💾 Save All Weeks", expanded=False):
            # Create an in-memory ZIP file for all PNGs
            png_zip = io.BytesIO()
            with zipfile.ZipFile(png_zip, mode="w") as zf:
                for wk in _WEEKS_LIST:
                    # Recompute this week’s heatmap data:
                    df_hm = _compute_week_hm_data(weekfile_detail, wk)

                    # Get the user’s custom title for this week from session_state:
                    # If the user never changed it, fall back to the default
                    title_key = f"title_{wk}"
                    user_title = st.session_state.get(title_key, f"Demand for {wk}")

                    # Build a small figure for this week’s heatmap:
                    fig_w, ax_w = plt.subplots(figsize=(10, 3))
                    sns.heatmap(df_hm, annot=True, linewidths=0.5, cmap="RdYlGn_r", ax=ax_w)

                    # Use the user’s custom title
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
                    df_hm = _compute_week_hm_data(weekfile_detail, wk)
                    csv_bytes = df_hm.to_csv(index=False).encode("utf-8")
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

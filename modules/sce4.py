import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io

# —— Global constants and precomputations —— #
_REFERENCE_DATE = "1900-01-01 "
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])
_WEEKDAY_ORDER  = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """Convert 'HH:MM' strings to Timestamp on 1900-01-01."""
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_duration_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize each row's In/Out into a 24-column matrix of durations (hours).
    Returns original rows + 24 columns (0–23) of duration values.
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

    # Compute duration overlap matrix (N x 24)
    start_matrix = in_times.values.reshape(-1, 1)
    end_matrix   = out_times.values.reshape(-1, 1)
    bin_starts   = _TIME_BIN_START.values.reshape(1, -1)
    bin_ends     = _TIME_BIN_END.values.reshape(1, -1)

    # Overlap duration in seconds per bin
    overlap_start = np.maximum(
        start_matrix.astype("datetime64[ns]"),
        bin_starts.astype("datetime64[ns]")
    )

    # Compute overlap-end = minimum(row_end, bin_end) for each cell
    overlap_end = np.minimum(
        end_matrix.astype("datetime64[ns]"),
        bin_ends.astype("datetime64[ns]")
    )
    seconds = (overlap_end.astype('datetime64[s]') - overlap_start.astype('datetime64[s]'))
    delta_seconds = seconds.astype(int)
    delta_seconds = np.where(delta_seconds < 0, 0, delta_seconds)
    hours_matrix = delta_seconds / 3600.0

    # Multiply by Count if present
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    duration_matrix = hours_matrix * counts

    dur_df = pd.DataFrame(
        duration_matrix,
        index=temp.index,
        columns=list(range(24))
    )
    result = pd.concat([temp.reset_index(drop=True), dur_df.reset_index(drop=True)], axis=1)
    return result


@st.cache_data(show_spinner=False)
def _compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize 'Count' by 'Month' (Period) and generate 'MonthLabel', without modifying the original df.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_summary = df.groupby('Month')['Count'].sum().reset_index()
    
    # Ensure months are sorted chronologically
    monthly_summary = monthly_summary.sort_values('Month')
    
    monthly_summary['MonthLabel'] = (
        monthly_summary['Month']
        .dt.to_timestamp()
        .dt.strftime('%b %y')     # Format as 'Jan 25'
    )

    return monthly_summary


@st.cache_data(show_spinner=False)
def _weekday_duration_summary(df_with_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'weekday' on the DataFrame that already has columns 0–23,
    returning a 7×1 DataFrame 'TotalDuration'.
    """
    summed = df_with_matrix.groupby('weekday')[list(range(24))].sum()
    summed['TotalDuration'] = summed.sum(axis=1)
    return summed.reindex(_WEEKDAY_ORDER)[['TotalDuration']]


@st.cache_data(show_spinner=False)
def _compute_normalized_duration_heatmap(df_with_matrix: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    1. Group by 'weekday' × 24 columns => raw durations (7×24).
    2. Count days in each weekday between start_date and end_date => days_per_weekday (7,).
    3. Divide raw by days, round to int.
    """
    raw = df_with_matrix.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER)
    drange = pd.date_range(start=start_date, end=end_date)
    day_counts = drange.day_name().value_counts().reindex(_WEEKDAY_ORDER).fillna(0).astype(int)
    normalized = (raw.div(day_counts, axis=0)).round().astype(int)
    return normalized.fillna(0)


def duration_month_analysis():
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] { max-width: 100px; }
        div[role="button"][aria-expanded] { padding: 0.25rem 0.5rem; }
        div[data-testid="stExpander"] > div { padding: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    # Copy DataFrame to avoid mutating session_state
    df = st.session_state['crna_data'].copy()

    # 1) Convert 'Date' to datetime and add 'Month' and 'weekday'
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df['weekday'] = df['Date'].dt.day_name()

    # —— 2) First, build a 24-column duration matrix for each row —— #
    with st.spinner("Computing per-row duration matrix…"):
        output = _compute_duration_matrix(df)
        # 'output' now includes columns "Month", 0,1,2,…,23

    # —— 3) Next, compute monthly summary (cache + spinner) —— #
    with st.spinner("Computing monthly duration summary…"):
        monthly_summary = _compute_monthly_summary(output)
        # This ensures the monthly aggregation correctly includes columns 0…23

    # —— 4) Draw pie chart —— #
    fig1 = px.pie(
        monthly_summary,
        values='Count',
        names='MonthLabel',
        hole=0.3
    )
    fig1.update_layout(
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1, yanchor="top"),
        template="plotly_white"
    )
    title1 = st.text_input("Pie Chart Title", "Monthly Total Duration", key="title1")

    fig1.update_layout(title={"text": title1, "x": 0.5, "xanchor": "center"})

    # —— 5) Draw weekday bar chart —— #
    with st.spinner("Computing total duration by weekday…"):
        df2 = _weekday_duration_summary(output)
    df2 = df2.reset_index().round()
    title2 = st.text_input("Bar Chart Title", "Total Duration by Weekday", key="title2")

    fig2 = px.bar(
        df2,
        x='weekday',
        y='TotalDuration',
        labels={'weekday': 'Day of Week', 'TotalDuration': 'Total Duration (hrs)'},
        text='TotalDuration',
        template='plotly_white'
    )
    single_color = px.colors.qualitative.Plotly[0]
    fig2.update_traces(marker_color=single_color)
    fig2.update_layout(title={'text': title2, 'x': 0.5, 'xanchor': 'center'})

    # —— 6) Draw normalized heatmap —— #
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date   = df['Date'].max().strftime('%Y-%m-%d')
    with st.spinner("Computing normalized heatmap data…"):
        agg_df = _compute_normalized_duration_heatmap(output, start_date, end_date)

    title3 = st.text_input("Heatmap Title", "Normalized Duration Heatmap", key="title3")

    fig3, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(agg_df, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(title3, fontdict={'fontsize': 18, 'fontweight': 'bold'}, loc='center', pad=20)
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()

    # —— 7) Render pie chart + download buttons —— #
    st.subheader("Monthly Total Duration")
    st.plotly_chart(fig1, use_container_width=True)
    col_l, col_c, col_r = st.columns([3,1,3])
    with col_c:
        with st.expander("💾 Save Pie Chart", expanded=False):
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", scale=2)
            st.download_button("🏞️ PNG", data=buf1.getvalue(), file_name="monthly_duration.png", mime="image/png")
            csv1 = monthly_summary.to_csv(index=False).encode("utf-8")
            st.download_button("📥 CSV", data=csv1, file_name="monthly_duration.csv", mime="text/csv")

    # —— 8) Render bar chart + download buttons —— #
    st.subheader("Total Duration by Weekday")
    st.plotly_chart(fig2, use_container_width=True)
    col_l, col_c, col_r = st.columns([3,1,3])
    with col_c:
        with st.expander("💾 Save Bar Chart", expanded=False):
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", scale=2)
            st.download_button("🏞️ PNG", data=buf2.getvalue(), file_name="weekday_duration.png", mime="image/png")
            csv2 = df2.to_csv(index=False).encode("utf-8")
            st.download_button("📥 CSV", data=csv2, file_name="weekday_duration.csv", mime="text/csv")

    # —— 9) Render heatmap + download buttons —— #
    st.subheader("Normalized Duration Heatmap")
    st.pyplot(fig3)
    col_l, col_c, col_r = st.columns([3,1,3])
    with col_c:
        with st.expander("💾 Save Heatmap", expanded=False):
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
            st.download_button("🏞️ PNG", data=buf3.getvalue(), file_name=f"{title3}.png", mime="image/png")
            csv3 = agg_df.to_csv(index=True).encode("utf-8")
            st.download_button("📥 CSV", data=csv3, file_name="normalized_duration.csv", mime="text/csv")

    # —— 10) Navigation buttons —— #
    back_col, _, week_col = st.columns([1,8,1])
    with back_col:
        if st.button("⬅️ Back"):
            # Clear all related session_state values
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

    with week_col:
        if st.button("🔍 Go to Week Analysis"):
            if st.session_state.analysis_type == "presence":
                st.session_state.page = "sce3"
            else:
                st.session_state.page = "sce5"
            st.rerun()

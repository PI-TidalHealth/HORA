import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io

_REFERENCE_DATE = "1900-01-01 "

# Pre-generate the start and end times of the 24-hour interval
_TIME_BIN_START = pd.to_datetime([f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)])
_TIME_BIN_END   = pd.to_datetime([f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)])

_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def _parse_time_series(series: pd.Series) -> pd.Series:
    """
    Convert a string in "HH:MM" format to a Timestamp (1900-01-01 HH:MM) for each entry.
    """
    return pd.to_datetime(_REFERENCE_DATE + series.astype(str))


@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize the original df (which must contain three columns: 'In Room', 'Out Room', 'Count')
    into a DataFrame containing 24 columns (0–23 hours). 
    Return value: original df (without NaN) + 24 columns indicating the number of people in the room.
    """
    temp = (
        df
        .dropna(subset=['In Room', 'Out Room'])
        .reset_index(drop=True)
        .copy()
    )
    temp['In Room']  = temp['In Room'].astype(str)
    temp['Out Room'] = temp['Out Room'].astype(str)

    # Vectorized parsing of "HH:MM" to Timestamp
    in_times  = _parse_time_series(temp['In Room'])
    out_times = _parse_time_series(temp['Out Room'])

    # If out < in then add one day
    wrap_mask = out_times < in_times
    out_times = out_times.where(~wrap_mask, out_times + pd.Timedelta(days=1))

    # Use broadcasting to create an (N×24) boolean matrix indicating whether each row overlaps each hour interval
    overlap = (
        (in_times.values.reshape(-1, 1) < _TIME_BIN_END.values.reshape(1, -1)) &
        (out_times.values.reshape(-1, 1) > _TIME_BIN_START.values.reshape(1, -1))
    )

    # "Count" is used as a weight
    counts = temp['Count'].astype(int).values.reshape(-1, 1)
    presence_matrix = overlap * counts  # shapes (N×24), elements are either 0 or the count

    # Convert to DataFrame with columns 0–23
    presence_df = pd.DataFrame(
        presence_matrix,
        index=temp.index,
        columns=list(range(24))
    )

    # Final DataFrame = original row info + 24 columns of presence data
    out_df = pd.concat([temp.reset_index(drop=True), presence_df.reset_index(drop=True)], axis=1)
    return out_df


@st.cache_data(show_spinner=False)
def _compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize 'Count' by 'Month' (Period) and generate 'MonthLabel', without modifying the original df.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    monthly_summary = df.groupby('Month')['Count'].sum().reset_index()
    
    # Ensure months are sorted in chronological order
    monthly_summary = monthly_summary.sort_values('Month')
    
    monthly_summary['MonthLabel'] = (
        monthly_summary['Month']
          .dt.to_timestamp()
          .dt.strftime('%b %y')     # Format as 'Jan 25'
    )

    return monthly_summary


@st.cache_data(show_spinner=False)
def _weekday_total_summary(df_with_time: pd.DataFrame) -> pd.DataFrame:
    """
    Group by 'weekday' on a DataFrame that already contains 24 columns (0–23) 
    and a 'weekday' column, returning a 7×1 DataFrame named 'Total'.
    """
    summed = df_with_time.groupby('weekday')[list(range(24))].sum()
    summed['Total'] = summed.sum(axis=1)
    return summed.reindex(_WEEKDAY_ORDER)[['Total']]


@st.cache_data(show_spinner=False)
def _compute_normalized_heatmap(df_with_time: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    1. Sum by 'weekday' × 24 hours to get raw_counts (7×24).
    2. Count how many times each weekday occurs between start_date and end_date.
    3. Divide raw_counts by day_counts, round, and return an integer-format heatmap matrix.
    """
    raw = df_with_time.groupby('weekday')[list(range(24))].sum().reindex(_WEEKDAY_ORDER)
    drange = pd.date_range(start=start_date, end=end_date)
    day_counts = drange.day_name().value_counts().reindex(_WEEKDAY_ORDER).fillna(0).astype(int)
    normalized = (raw.div(day_counts, axis=0)).round().astype(int)
    return normalized.fillna(0)


def month_analysis():
    st.markdown(
        """
        <style>
        /* Limit the maximum width of expanders */
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

    # —— 1. Check if data has been uploaded and processed —— #
    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    # Copy df to avoid modifying the original session_state
    df = st.session_state['crna_data'].copy()
    # Standardize Date and weekday columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    df['weekday'] = df['Date'].dt.day_name()

    # —— 2. Monthly summary (with cache + spinner) —— #
    with st.spinner("Calculating monthly summary…"):
        monthly_summary = _compute_monthly_summary(df)

    # Handle title for pie chart (only initialize once)
    title1 = st.text_input("Pie Chart Title", "Monthly Demand", key="title1")

    # Create pie chart
    fig1 = px.pie(
        monthly_summary,
        values='Count',
        names='MonthLabel',
        hole=0.3,
        category_orders={"MonthLabel": monthly_summary['MonthLabel'].tolist()},
        labels={'MonthLabel': 'Month', 'Count': 'Demand'},
        custom_data=['Count']
    )
    
    # Update all traces and layout at once
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="Month: %{label}<br>Demand: %{value}<extra></extra>"
    )
    
    fig1.update_layout(
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.1, yanchor="top"
        ),
        template="plotly_white",
        title={"text": title1, "x": 0.5, "xanchor": "center"},
        margin=dict(t=50, l=20, r=20, b=50),
        height=450
    )

    # —— 3. Compute Presence matrix (with cache + spinner) —— #
    with st.spinner("Computing presence matrix, may take a few seconds…"):
        output = _compute_presence_matrix(df)

    # —— 4. Aggregate 'Total' by weekday (with cache + spinner) —— #
    with st.spinner("Aggregating total demand by weekday…"):
        df2 = _weekday_total_summary(output)
    df2 = df2.reset_index()  # Reset index so that 'weekday' becomes a column

    title2 = st.text_input("Bar Chart Title", "Total Month Demand by Weekday", key="title2")

    fig2 = px.bar(
        df2,
        x='weekday',
        y='Total',
        labels={'weekday': 'Day of Week', 'Total': 'Total Demand'},
        text='Total',
        template='plotly_white'
    )
    single_color = px.colors.qualitative.Plotly[0]
    fig2.update_traces(marker_color=single_color)
    fig2.update_layout(title={'text': title2, 'x': 0.5, 'xanchor': 'center'})

    # —— 5. Normalized heatmap (with cache + spinner) —— #
    # Use the actual data range for start_date/end_date
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date   = df['Date'].max().strftime('%Y-%m-%d')
    with st.spinner("Calculating normalized heatmap data…"):
        agg_df = _compute_normalized_heatmap(output, start_date, end_date)

    title3 = st.text_input("Heatmap Title", "Normalized Demand Heatmap", key="title3")

    fig3, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(agg_df, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(
        title3,
        fontdict={'fontsize': 18, 'fontweight': 'bold'},
        loc='center',
        pad=20
    )
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()

    # —— 6. Display pie chart + download buttons —— #
    st.subheader("Monthly Demand")
    st.plotly_chart(fig1, use_container_width=True)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf1.getvalue(),
                file_name="monthly_distribution.png",
                mime="image/png"
            )
            csv1 = monthly_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv1,
                file_name="monthly_distribution.csv",
                mime="text/csv"
            )

    # —— 7. Display bar chart + download buttons —— #
    st.subheader("Total Month Demand by Weekday")
    st.plotly_chart(fig2, use_container_width=True)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf2.getvalue(),
                file_name="weekday_summary.png",
                mime="image/png"
            )
            csv2 = df2.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv2,
                file_name="weekday_summary.csv",
                mime="text/csv"
            )

    # —— 8. Display heatmap + download buttons —— #
    st.subheader("Normalized Demand Heatmap")
    st.pyplot(fig3)
    col_l, col_c, col_r = st.columns([3, 1, 3])
    with col_c:
        with st.expander("💾 Save ", expanded=False):
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
            st.download_button(
                label="🏞️ PNG",
                data=buf3.getvalue(),
                file_name=f"{title3}.png",
                mime="image/png"
            )
            csv3 = agg_df.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv3,
                file_name="normalized_heatmap.csv",
                mime="text/csv"
            )

    # —— 9. 'Back' and 'Go to Week Analysis' buttons —— #
    back_col, _, week_col = st.columns([1, 8, 1])
    with back_col:
        if st.button("⬅️ Back"):
            # Clear all related session_state keys
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

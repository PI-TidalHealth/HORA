import polars as pl
import pandas as _pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import numpy as np
import pandas as pd

# Constants
_REFERENCE_DATE = "1900-01-01 "
_TIME_FORMAT = "%H:%M"
_DATETIME_FORMAT = "%Y-%m-%d %H:%M"
_TIME_BIN_START = [f"{_REFERENCE_DATE}{h:02d}:00" for h in range(24)]
_TIME_BIN_END = [f"{_REFERENCE_DATE}{(h+1)%24:02d}:00" for h in range(24)]

_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

def _parse_time_series(df: pl.DataFrame) -> pl.DataFrame:
    """Parse time series data and handle edge cases."""
    # Drop rows with null values in In Room or Out Room
    temp = df.filter(
        ~pl.col('In Room').is_null() & 
        ~pl.col('Out Room').is_null()
    )
    

    temp = temp.with_columns([
        (pl.col('Date').cast(str) + ' ' + pl.col('In Room')).alias('In_str'),
        (pl.col('Date').cast(str) + ' ' + pl.col('Out Room')).alias('Out_str')
    ])
    
    # Then convert to datetime by adding reference date
    temp = temp.with_columns([
        pl.col('In_str').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M').alias('In_dt'),
        pl.col('Out_str').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M').alias('Out_dt')
    ])
    
    # Handle cross-day cases
    temp = temp.with_columns([
        pl.when(pl.col('Out_dt') < pl.col('In_dt'))
            .then(pl.col('Out_dt') + timedelta(days=1))
            .otherwise(pl.col('Out_dt'))
            .alias('Out_dt')
    ])
    print(temp)
    return temp

@st.cache_data(show_spinner=False)
def _compute_duration_matrix(df: pd.DataFrame) -> pl.DataFrame:
    # 1. 转为 pandas DataFrame
    df = df.to_pandas()

    # 2. 解析时间
    df['In_dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['In Room'])
    df['Out_dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Out Room'])
    # 跨天处理
    df.loc[df['Out_dt'] <= df['In_dt'], 'Out_dt'] += pd.Timedelta(days=1)

    # 3. 生成每小时的时间点列表
    def hour_range(row):
        start = row['In_dt'].replace(minute=0, second=0, microsecond=0)
        end = row['Out_dt']
        hours = []
        cur = start
        while cur < end:
            hours.append(cur)
            cur += timedelta(hours=1)
        return hours

    df['hour_ts'] = df.apply(hour_range, axis=1)
    df_exploded = df.explode('hour_ts')

    # 4. 提取日期、小时
    df_exploded['Date'] = df_exploded['hour_ts'].dt.date
    df_exploded['hour'] = df_exploded['hour_ts'].dt.hour

    # 5. 计算每小时的 duration（小时数）
    def calc_duration(row):
        hour_start = row['hour_ts']
        hour_end = hour_start + timedelta(hours=1)
        overlap_start = max(row['In_dt'], hour_start)
        overlap_end = min(row['Out_dt'], hour_end)
        duration = (overlap_end - overlap_start).total_seconds() / 3600.0
        return max(duration, 0) * row['Count']

    df_exploded['duration'] = df_exploded.apply(calc_duration, axis=1)

    # 6. 按日期和小时统计 duration
    result = (
        df_exploded
        .groupby(['Date', 'hour'], as_index=False)['duration']
        .sum()
        .pivot(index='Date', columns='hour', values='duration')
        .fillna(0)
    )

    # 7. 补齐所有小时列
    for h in range(24):
        if h not in result.columns:
            result[h] = 0.0
    result = result[[col for col in ['Date'] + list(range(24)) if col in result.columns]]
    result = result.sort_index(axis=1)

    # 8. 加 weekday 列
    result['weekday'] = pd.to_datetime(result.index).strftime('%A')

    return pl.from_pandas(result.reset_index())

@st.cache_data(show_spinner=False)
def _compute_monthly_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize 'Count' by 'Month' (Period) and generate 'MonthLabel'."""
    monthly_summary = (
        df.with_columns([
            pl.col('Date').dt.strftime('%Y-%m').alias('Month'),
            pl.col('Date').dt.strftime('%b %y').alias('MonthLabel')
        ])
        .group_by(['Month', 'MonthLabel'])
        .agg([
            pl.col('Count').sum().alias('Count')
        ])
        .sort('Month')
    )
    
    return monthly_summary

@st.cache_data(show_spinner=False)
def _weekday_total_summary(df_with_time: pl.DataFrame) -> pl.DataFrame:
    """
    Group by 'weekday' on a DataFrame that already contains 24 columns (0–23) 
    and a 'weekday' column, returning a 7×1 DataFrame named 'Total'.
    """
    # Get all hour columns
    hour_cols = [str(h) for h in range(24)]
    # 先生成 Total 列
    df_with_time = df_with_time.with_columns(
        pl.sum_horizontal(hour_cols).alias('Total')
    )
    # 再 group_by
    summed = (
        df_with_time
        .group_by('weekday')
        .agg([
            pl.col('Total').sum().alias('Total')
        ])
        .filter(pl.col('weekday').is_in(_WEEKDAY_ORDER))
        .sort('weekday')
    )
    return summed

@st.cache_data(show_spinner=False)
def _compute_normalized_heatmap(df_with_time: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    """
    1. Sum by 'weekday' × 24 hours to get raw_counts (7×24).
    2. Count how many times each weekday occurs between start_date and end_date.
    3. Divide raw_counts by day_counts, round, and return an integer-format heatmap matrix.
    """
    # Get all hour columns
    hour_cols = [str(h) for h in range(24)]
    
    # Sum by weekday for each hour
    raw = (
        df_with_time
        .group_by('weekday')
        .agg([
            pl.sum(col).alias(col) for col in hour_cols
        ])
        .filter(pl.col('weekday').is_in(_WEEKDAY_ORDER))
        .sort('weekday')
    )
    
    # Convert start/end to python datetime for robustness
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt   = datetime.strptime(end_date, '%Y-%m-%d')

    date_list = []
    cur = start_dt
    while cur <= end_dt:
        date_list.append(cur)
        cur += timedelta(days=1)

    day_counts = (
        pl.DataFrame({'date': date_list})
        .with_columns([
            pl.col('date').dt.strftime('%A').alias('weekday')
        ])
        .group_by('weekday')
        .count()
        .filter(pl.col('weekday').is_in(_WEEKDAY_ORDER))
        .sort('weekday')
    )

    
    # Normalize the counts
    normalized = raw.with_columns([
        (pl.col(col) / day_counts.get_column('count')).round().cast(pl.Int64).alias(col)
        for col in hour_cols
    ])
    
    return normalized.fill_null(0)

def duration_month_analysis():
    """Main function for monthly analysis visualization."""
    # Apply custom CSS
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] {
            max-width: 100px;
        }
        div[role="button"][aria-expanded] {
            padding: 0.25rem 0.5rem;
        }
        div[data-testid="stExpander"] > div {
            padding: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Check if data exists
    if 'crna_data' not in st.session_state:
        st.info("Please upload a file to begin.")
        return

    # Initialize DataFrame
    df = st.session_state['crna_data']
    
    # Convert pandas DataFrame to polars if needed
    if isinstance(df, _pd.DataFrame):
        df = pl.from_pandas(df)
    df = df.clone()
    
    # Standardize Date column
    df = df.with_columns([
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
    
    # Add weekday column
    df = df.with_columns([
        pl.col('Date').dt.strftime('%A').alias('weekday')
    ])

    # —— 2. Monthly summary (with cache + spinner) —— #
    with st.spinner("Calculating monthly summary…"):
        monthly_summary = _compute_monthly_summary(df)

    # Handle title for pie chart (only initialize once)
    title1 = st.text_input("Pie Chart Title", "Monthly Duration", key="title1")

    # Create pie chart
    fig1 = px.pie(
        monthly_summary.to_pandas(),
        values='Count',
        names='MonthLabel',
        hole=0.3,
        category_orders={"MonthLabel": monthly_summary.get_column('MonthLabel').to_list()},
        labels={'MonthLabel': 'Month', 'Count': 'Duration (hours)'},
        custom_data=['Count']
    )
    
    # Update all traces and layout at once
    fig1.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="Month: %{label}<br>Duration: %{value} hours<extra></extra>"
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

    # —— 3. Compute Duration matrix (with cache + spinner) —— #
    with st.spinner("Computing duration matrix, may take a few seconds…"):
        output = _compute_duration_matrix(df)

    # —— 4. Aggregate 'Total' by weekday (with cache + spinner) —— #
    with st.spinner("Aggregating total duration by weekday…"):
        df2 = _weekday_total_summary(output)

    df2_plot = df2.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER).reset_index()
    title2 = st.text_input("Bar Chart Title", "Total Month Duration by Weekday", key="title2")

    fig2 = px.bar(
        df2_plot,
        x='weekday',
        y='Total',
        labels={'weekday': 'Day of Week', 'Total': 'Total Duration (hours)'},
        text='Total',
        template='plotly_white'
    )
    single_color = px.colors.qualitative.Plotly[0]
    fig2.update_traces(marker_color=single_color)
    fig2.update_layout(title={'text': title2, 'x': 0.5, 'xanchor': 'center'})
    fig2.update_traces(texttemplate='%{text:.0f}')

    # —— 5. Normalized heatmap (with cache + spinner) —— #
    # Use the actual data range for start_date/end_date
    start_date = df.get_column('Date').cast(pl.String).str.strptime(pl.Date, None).min().strftime('%Y-%m-%d')
    end_date = df.get_column('Date').cast(pl.String).str.strptime(pl.Date, None).max().strftime('%Y-%m-%d')
    
    with st.spinner("Calculating normalized heatmap data…"):
        agg_df = _compute_normalized_heatmap(output, start_date, end_date)

    df_plot = agg_df.to_pandas().set_index('weekday').reindex(_WEEKDAY_ORDER)
    title3 = st.text_input("Heatmap Title", "Normalized Duration Heatmap", key="title3")

    fig3, ax = plt.subplots(figsize=(20, 5))
    sns.heatmap(df_plot, annot=True, linewidths=0.5, cmap='RdYlGn_r', ax=ax)
    ax.set_title(
        title3,
        fontdict={'fontsize': 18, 'fontweight': 'bold'},
        loc='center',
        pad=20
    )
    ax.set_ylabel("DOW", fontsize=14)
    plt.tight_layout()

    # —— 6. Display pie chart + bar chart in one row —— #
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("💾 Save ", expanded=False):
            buf1 = io.BytesIO()
            fig1.write_image(buf1, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf1.getvalue(),
                file_name=f"{title1}.png",
                mime="image/png"
            )
            csv1 = monthly_summary.write_csv().encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv1,
                file_name=f"{title1}.csv",
                mime="text/csv"
            )
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("💾 Save ", expanded=False):
            buf2 = io.BytesIO()
            fig2.write_image(buf2, format="png", scale=2)
            st.download_button(
                label="🏞️ PNG",
                data=buf2.getvalue(),
                file_name=f"{title2}.png",
                mime="image/png"
            )
            csv2 = df2.write_csv().encode("utf-8")
            st.download_button(
                label="📥 CSV",
                data=csv2,
                file_name=f"{title2}.csv",
                mime="text/csv"
            )

    # —— 7. Display heatmap in a new row —— #
    st.pyplot(fig3)
    with st.expander("💾 Save ", expanded=False):
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
        st.download_button(
            label="🏞️ PNG",
            data=buf3.getvalue(),
            file_name=f"{title3}.png",
            mime="image/png"
        )
        csv3 = df_plot.to_csv().encode("utf-8")
        st.download_button(
            label="📥 CSV",
            data=csv3,
            file_name=f"{title3}.csv",
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

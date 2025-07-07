import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import io
import re
import matplotlib
import pandas as pd
_WEEKS_LIST     = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
_WEEKDAY_ORDER = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

@st.cache_data(show_spinner=False)
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
    return temp

@st.cache_data(show_spinner=False)
def _compute_presence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 解析时间
    df = df.to_pandas()
    df = df.copy()
    df['In_dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['In Room'])
    df['Out_dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Out Room'])
    # 跨天处理
    df.loc[df['Out_dt'] <= df['In_dt'], 'Out_dt'] += pd.Timedelta(days=1)

    # 2. 生成每小时的时间点列表
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

    # 3. 提取日期、小时
    df_exploded['Date'] = df_exploded['hour_ts'].dt.date
    df_exploded['hour'] = df_exploded['hour_ts'].dt.hour

    # 4. 按日期和小时统计 presence
    result = (
        df_exploded
        .groupby(['Date', 'hour'], as_index=False)['Count']
        .sum()
        .pivot(index='Date', columns='hour', values='Count')
        .fillna(0)
        .astype(int)
    )

    # 5. 补齐所有小时列
    for h in range(24):
        if h not in result.columns:
            result[h] = 0
    result = result[[col for col in ['Date'] + list(range(24)) if col in result.columns]]
    result = result.sort_index(axis=1)

    # 6. 加 weekday 列
    result['weekday'] = pd.to_datetime(result.index).strftime('%A')
    print(result)

    return pl.from_pandas(result.reset_index())

@st.cache_data(show_spinner=False)
def _compute_monthly_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Summarize number of records by 'Month' (Period) and generate 'MonthLabel'."""
    monthly_summary = (
        df.with_columns([
            pl.col('Date').dt.strftime('%Y-%m').alias('Month'),
            pl.col('Date').dt.strftime('%b %y').alias('MonthLabel')
        ])
        .group_by(['Month', 'MonthLabel'])
        .agg([
            pl.count().alias('Count')
        ])
        .sort('Month')
    )
    
    return monthly_summary

@st.cache_data(show_spinner=False)
def _weekday_total_summary(df_with_time: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
    """
    Group by 'weekday' on a DataFrame that已经包含24小时列（0~23）和'weekday'列，
    返回一个7×2的DataFrame，每行是weekday，Total为该weekday所有小时的总和，
    再除以24和该weekday在[start_date, end_date]区间内的天数。
    """
    hour_cols = [str(h) for h in range(24)]
    # 按 weekday 分组，对每小时列求和
    result = (
        df_with_time
        .group_by('weekday')
        .agg([
            pl.col(col).sum().alias(col) for col in hour_cols
        ])
        .filter(pl.col('weekday').is_in(_WEEKDAY_ORDER))
        .sort('weekday')
    )
    # 统计每个 weekday 在区间内的天数
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
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
    # 新增 Total 列
    result = result.with_columns(
        (pl.sum_horizontal(hour_cols) / 24 / day_counts.get_column('count')).alias('Total')
    )
    return result.select(['weekday', 'Total'])

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
    print(raw)
    # Convert start/end to python datetime for robustness
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt   = datetime.strptime(end_date, '%Y-%m-%d')

    date_list = []
    cur = start_dt
    while cur <= end_dt:
        date_list.append(cur)
        cur += timedelta(days=1)

    # 统计 presence matrix 里所有实际出现过的日期
    all_dates = df_with_time.get_column('Date').unique().to_list()
    date_df = pl.DataFrame({'date': all_dates})
    date_df = date_df.with_columns([
        pl.col('date').dt.strftime('%A').alias('weekday')
    ])
    day_counts = (
        date_df
        .group_by('weekday')
        .count()
        .filter(pl.col('weekday').is_in(_WEEKDAY_ORDER))
        .sort('weekday')
    )
    print(day_counts)
    # 生成完整的 weekday DataFrame
    all_weekdays = pl.DataFrame({'weekday': _WEEKDAY_ORDER})

    # 补齐 raw 和 day_counts
    raw = all_weekdays.join(raw, on='weekday', how='left').fill_null(0)
    day_counts = all_weekdays.join(day_counts, on='weekday', how='left').fill_null(0)
    
    # Normalize the counts
    normalized = raw.with_columns([
        (pl.col(col) / day_counts.get_column('count'))
            .fill_nan(0)
            .round()
            .cast(pl.Int64)
            .alias(col)
        for col in hour_cols
    ])
    
    return normalized.fill_null(0)

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

    # 统计数据集有多少不同的月份
    if 'Date' in df_wk.columns:
        month_count = df_wk.get_column('Date').cast(pl.Date).dt.strftime('%Y-%m').n_unique()
    else:
        month_count = 1
    if month_count == 0:
        month_count = 1

    # Normalize by dividing by month_count
    hm_data = agg.with_columns([
        (pl.col(col) / month_count).round().cast(pl.Int64).alias(col) for col in hour_cols
    ])

    return hm_data

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


#Please read the README.md before you run the code
def process_schedule_excel(
    excel_path,
    start_date_str='2025/01/01',
    end_date_str='2025/03/31',
):
    try:
            # FIX: Added header=None to correctly read CSV files
            df = pd.read_csv(excel_path, header=None)
    except Exception:
            df = pd.read_excel(excel_path, header=None)
    print(df)

    df_raw = df.iloc[:, 0:49]
    rows_to_keep = [0, 1, -1]
    df_raw = df_raw.iloc[rows_to_keep]
    print(df_raw)
    weekday_cols = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result = []
    n_cols = df_raw.shape[1]
    for start in range(0, n_cols, 7):
        sheet_val = df_raw.iloc[0, start]
        if pd.isna(sheet_val):
            continue
        # Replace "GOR" with "3:30p-7a"
        if isinstance(sheet_val, str) and sheet_val.strip().upper() == "GOR":
            sheet_val = "3p-7a"
        # Extract time range like 7a-7a, 8a-4p, 3:30p-7a, etc.
        elif isinstance(sheet_val, str):
            match = re.search(r'(\d{1,2}(?::\d{2})?[ap]-\d{1,2}(?::\d{2})?[ap])', sheet_val, re.IGNORECASE)
            if match:
                sheet_val = match.group(1)
        counts = df_raw.iloc[2, start:start+7].tolist()
        row = {'Sheet': sheet_val}
        for i, day in enumerate(weekday_cols):
            row[day] = counts[i] if i < len(counts) else 0
        result.append(row)

    df_result = pd.DataFrame(result)
    df_result.replace(['', ' '], 0, inplace=True)
    df_result.fillna(0, inplace=True)


    def parse_time_interval(s):
        s = s.replace(' ', '')
        # 1. 处理 7a-3:30p 这种格式
        match = re.match(r'(\d{1,2})(?::(\d{2}))?([ap])-(\d{1,2})(?::(\d{2}))?([ap])', s)
        if match:
            h1, m1, ap1, h2, m2, ap2 = match.groups()
            m1 = m1 or '00'
            m2 = m2 or '00'
            t1 = f"{int(h1):02d}:{m1} {ap1}m"
            t2 = f"{int(h2):02d}:{m2} {ap2}m"
            t1 = pd.to_datetime(t1, format='%I:%M %p').strftime('%H:%M')
            t2 = pd.to_datetime(t2, format='%I:%M %p').strftime('%H:%M')
            return t1, t2

        match = re.match(r'(\d{2}):?(\d{2})-(\d{2}):?(\d{2})', s)
        if match:
            h1, m1, h2, m2 = match.groups()
            t1 = f"{h1}:{m1}"
            t2 = f"{h2}:{m2}"
            return t1, t2
        return '', ''

    df_result[['In Time', 'Out Time']] = df_result['Sheet'].apply(lambda x: pd.Series(parse_time_interval(str(x))))

    df_result = df_result.drop(columns=['Sheet'])


    cols = ['In Time', 'Out Time'] + [col for col in df_result.columns if col not in ['In Time', 'Out Time']]
    df = df_result[cols]

    start_date = datetime.strptime(start_date_str, '%Y/%m/%d')
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d')
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    virtual_dates = {day: [] for day in weekday_cols}
    for d in all_dates:
        weekday = d.strftime('%A')
        if weekday in virtual_dates:
            virtual_dates[weekday].append(d.strftime('%Y/%m/%d'))


    result = []
    for _, row in df.iterrows():
        for day in weekday_cols:
            count = int(row[day])
            dates_for_day = virtual_dates[day]
            n_dates = len(dates_for_day)
            for i in range(count):
                # 用取余的方式循环取日期
                date = dates_for_day[i % n_dates]
                result.append({
                    'Date': date,
                    'Weekday': day,
                    'In Time': row['In Time'],
                    'Out Time': row['Out Time']
                })

    result_df = pd.DataFrame(result)
    result_df.to_csv("output.csv", index=False)
    print(result_df)
    return result_df
